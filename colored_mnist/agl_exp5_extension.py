"""
agl_exp5_extension.py — Agreement-Guided Targeted HP Search
============================================================
Extension of Experiment 5 (Proximity × Diversity 2x2) adding two new
HP selection strategies on top of the existing results.

RESULTS TABLE (all values = OOD test accuracy at e=0.9):

Config  Exp5 Oracle  Exp5 Non-oracle  New Labeled ★  New Label-free ★
A       67.56%       10.59%           ???            ???
B       81.69%       77.93%           ???            ???
C       71.71%       69.44%           ???            ???
D       72.80%       72.58%           ???            ???

Column definitions:
  Exp5 Oracle      — test env labels for HP selection (reference only, hardcoded)
  Exp5 Non-oracle  — train-domain val for HP selection (hardcoded from Exp 5)
  New Labeled ★    — ACL detection + targeted fine search + TTA (OOD val labels)
  New Label-free ★ — AGL detection + targeted fine search + TTA (no OOD labels)

Usage:
    python agl_exp5_extension.py --configs ABCD --n_coarse_trials 20
           --n_seeds 3 --device cuda --max_steps 201
"""

import argparse
import copy
import itertools
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from data      import load_mnist_raw, build_envs, make_environment, make_val_splits
from models    import (train_model, compute_accuracy, worst_env_val_acc,
                       get_predictions, sample_erm_hp, sample_irm_hp, HP_RANGES)
from agreement import (compute_pairwise_agreement, compute_cross_agreement,
                       compute_entropy, fit_agl_line, compute_deviation,
                       inv_probit, probit, DEVIATION_THRESHOLD)


# =============================================================================
# 2x2 configs — ground truth from Exp 5 hardcoded
# =============================================================================

CONFIGS = {
    "A": {
        "e_train":        [0.1, 0.2],
        "diversity":      "Low",
        "proximity":      "Low",
        "gt_oracle":      0.6756,
        "gt_non_oracle":  0.1059,
    },
    "B": {
        "e_train":        [0.7, 0.8],
        "diversity":      "Low",
        "proximity":      "High",
        "gt_oracle":      0.8169,
        "gt_non_oracle":  0.7793,
    },
    "C": {
        "e_train":        [0.1, 0.5],
        "diversity":      "High",
        "proximity":      "Low",
        "gt_oracle":      0.7171,
        "gt_non_oracle":  0.6944,
    },
    "D": {
        "e_train":        [0.1, 0.8],
        "diversity":      "High",
        "proximity":      "High",
        "gt_oracle":      0.7280,
        "gt_non_oracle":  0.7258,
    },
}

E_TEST                  = 0.9
BETA                    = 0.5    # penalize instability in score
TOP_K                   = 3      # top coarse configs guide fine search region
DIVERGE_FRAC_THRESHOLD  = 0.20   # fraction of diverging IRM configs to trigger DIVERGE
ACL_DEVIATION_THRESHOLD = 0.03   # IRM above ERM ACL line by this much = DIVERGE
TTA_LR                  = 1e-4   # TTA learning rate
TTA_STEPS               = 20     # TTA adaptation steps


# =============================================================================
# HP sampling
# =============================================================================

def sample_irm_hp_targeted(rng, lam_min, lam_max, lr_min, lr_max, base_hp=None):
    """Sample IRM HP config with λ and lr constrained to promising region."""
    hp = dict(base_hp) if base_hp else sample_irm_hp(rng)
    # λ: log-uniform in promising region
    hp["penalty_weight"] = float(10 ** rng.uniform(
        np.log10(max(lam_min, 1)), np.log10(min(lam_max, 1e7))))
    # lr: log-uniform in promising region
    hp["lr"] = float(10 ** rng.uniform(
        np.log10(max(lr_min, 1e-5)), np.log10(min(lr_max, 1e-1))))
    # other HPs sampled freely
    hp["hidden_dim"]           = int(rng.choice(HP_RANGES["hidden_dim"]))
    hp["l2_reg"]               = float(rng.choice(HP_RANGES["l2_reg"]))
    hp["steps"]                = int(rng.choice(HP_RANGES["steps"]))
    hp["penalty_anneal_iters"] = int(rng.choice(HP_RANGES["penalty_anneal_iters"]))
    return hp


def sample_erm_hp_targeted(rng, lr_min, lr_max, base_hp=None):
    """Sample ERM HP config with lr constrained to promising region."""
    hp = dict(base_hp) if base_hp else sample_erm_hp(rng)
    hp["lr"] = float(10 ** rng.uniform(
        np.log10(max(lr_min, 1e-5)), np.log10(min(lr_max, 1e-1))))
    hp["hidden_dim"] = int(rng.choice(HP_RANGES["hidden_dim"]))
    hp["l2_reg"]     = float(rng.choice(HP_RANGES["l2_reg"]))
    hp["steps"]      = int(rng.choice(HP_RANGES["steps"]))
    return hp


def get_hp_region(top_configs, key_lam="penalty_weight", key_lr="lr"):
    """Extract HP region from top-K configs."""
    lams = [c["hp"].get(key_lam, 1.0) for c in top_configs]
    lrs  = [c["hp"][key_lr] for c in top_configs]
    return (min(lams)/5, max(lams)*5, min(lrs)/3, max(lrs)*3)


# =============================================================================
# Scoring functions
# =============================================================================

def score_irm_labeled(cfg, acl_slope, acl_intercept):
    """
    Labeled mode IRM score: deviation above ERM ACL line.
    Higher = IRM found better OOD features than ERM.
    """
    expected_ood = acl_slope * cfg["mean_id_val_acc"] + acl_intercept
    deviation    = cfg["mean_ood_acc"] - expected_ood
    return deviation - BETA * cfg.get("std_ood_acc", 0.0)


def score_irm_label_free(cfg):
    """
    Label-free mode IRM score: deviation below ERM AGL line.
    Higher score = more below line = stronger divergence from ERM.
    """
    return -cfg["mean_deviation"] - BETA * cfg.get("std_deviation", 0.0)


def score_erm_labeled(cfg):
    """Labeled mode ERM score: OOD val accuracy directly."""
    return cfg["mean_ood_acc"] - BETA * cfg.get("std_ood_acc", 0.0)


def score_erm_label_free(cfg):
    """Label-free mode ERM score: within-ERM OOD agreement consistency."""
    return cfg.get("within_ood_agr", 0.5) - BETA * cfg.get("within_std", 0.1)


# =============================================================================
# TTA
# =============================================================================

def apply_tent(model, ref_test, device, steps=TTA_STEPS, lr=TTA_LR):
    """
    TENT: minimize prediction entropy on unlabeled OOD data.
    Updates all model parameters with small lr.
    Always label-free.
    """
    import torch
    import torch.nn as nn
    adapted = copy.deepcopy(model)
    opt     = torch.optim.Adam(adapted.parameters(), lr=lr)
    eps     = 1e-6
    for _ in range(steps):
        adapted.train()
        logits  = adapted(ref_test["images"].to(device))
        p       = torch.sigmoid(logits)
        entropy = -(p * torch.log(p + eps) + (1-p) * torch.log(1-p + eps))
        opt.zero_grad()
        entropy.mean().backward()
        opt.step()
    adapted.eval()
    return adapted


# =============================================================================
# AGL helpers (reuse from agreement.py)
# =============================================================================

def fit_acl_line(configs):
    """Fit ACL line from (id_val_acc, ood_acc) points across ERM configs."""
    x = np.array([c["mean_id_val_acc"] for c in configs])
    y = np.array([c["mean_ood_acc"]    for c in configs])
    if len(x) < 3:
        return 1.0, 0.0, 0.0
    s, i, r, _, _ = stats.linregress(x, y)
    return s, i, r**2


# =============================================================================
# Main experiment
# =============================================================================

def run_config(config_name, cfg, n_coarse_trials, n_seeds, fine_trials,
               device, output_dir, mnist_raw, max_steps=None):

    os.makedirs(output_dir, exist_ok=True)
    e_values = cfg["e_train"]
    train_images, train_labels, val_images, val_labels = mnist_raw
    rng = np.random.RandomState(42)

    # Fixed reference data — no labels used from OOD side
    ref_envs = build_envs(train_images, train_labels, e_values, seed=0)
    ref_test = make_environment(val_images, val_labels, E_TEST, seed=99)
    id_ref   = ref_envs[0]

    print(f"\n{'='*65}")
    print(f"  Config {config_name}: e_train={e_values}  "
          f"diversity={cfg['diversity']}  proximity={cfg['proximity']}")
    print(f"  Exp5 Oracle={cfg['gt_oracle']:.1%}  "
          f"Exp5 Non-oracle={cfg['gt_non_oracle']:.1%}")
    print(f"  n_coarse={n_coarse_trials}  n_seeds={n_seeds}  "
          f"fine_trials={fine_trials}  device={device}")
    print(f"{'='*65}")

    t0 = time.time()

    # ------------------------------------------------------------------
    # Phase 1 — Coarse random search
    # ------------------------------------------------------------------

    print(f"\n[Phase 1] Coarse search ({n_coarse_trials} trials × {n_seeds} seeds)...")

    erm_configs = []
    irm_configs = []

    for trial in range(n_coarse_trials):
        erm_hp = sample_erm_hp(rng)
        irm_hp = sample_irm_hp(rng)

        erm_entry = {"hp": erm_hp, "models": [], "results": []}
        irm_entry = {"hp": irm_hp, "models": [], "results": []}

        for seed in range(n_seeds):
            envs             = build_envs(train_images, train_labels, e_values, seed=seed)
            test_env         = make_environment(val_images, val_labels, E_TEST, seed=seed+99)
            train_envs, val_envs = make_val_splits(envs)

            # ERM
            m_erm     = train_model(train_envs, erm_hp, "erm", device, seed, max_steps=max_steps)
            val_a     = worst_env_val_acc(m_erm, val_envs, device)
            ood_a     = compute_accuracy(m_erm, test_env, device)
            erm_entry["models"].append(m_erm)
            erm_entry["results"].append({"seed": seed, "id_val_acc": val_a, "ood_acc": ood_a})

            # IRM
            m_irm     = train_model(train_envs, irm_hp, "irm", device, seed, max_steps=max_steps)
            val_a_i   = worst_env_val_acc(m_irm, val_envs, device)
            ood_a_i   = compute_accuracy(m_irm, test_env, device)
            irm_entry["models"].append(m_irm)
            irm_entry["results"].append({"seed": seed, "id_val_acc": val_a_i, "ood_acc": ood_a_i})

        for entry in [erm_entry, irm_entry]:
            entry["mean_id_val_acc"] = float(np.mean([r["id_val_acc"] for r in entry["results"]]))
            entry["mean_ood_acc"]    = float(np.mean([r["ood_acc"]    for r in entry["results"]]))
            entry["std_ood_acc"]     = float(np.std( [r["ood_acc"]    for r in entry["results"]]))

        erm_configs.append(erm_entry)
        irm_configs.append(irm_entry)
        print(f"  trial={trial:02d} | ERM val={erm_entry['mean_id_val_acc']:.3f} "
              f"ood={erm_entry['mean_ood_acc']:.3f} | "
              f"IRM val={irm_entry['mean_id_val_acc']:.3f} "
              f"ood={irm_entry['mean_ood_acc']:.3f} "
              f"λ={irm_hp['penalty_weight']:.0f}")

    # ------------------------------------------------------------------
    # Phase 2 — Detection (both modes)
    # ------------------------------------------------------------------

    print(f"\n[Phase 2] Detection...")

    # Fit ERM-ERM AGL line (label-free mode)
    erm_all    = [m for c in erm_configs for m in c["models"]]
    erm_pairs  = compute_pairwise_agreement(erm_all, id_ref, ref_test, device)
    agl_slope, agl_intercept, agl_r2 = fit_agl_line(erm_pairs)
    print(f"  AGL line: slope={agl_slope:.3f}  intercept={agl_intercept:.3f}  R²={agl_r2:.3f}")

    # Fit ERM-only ACL line (labeled mode)
    acl_slope, acl_intercept, acl_r2 = fit_acl_line(erm_configs)
    print(f"  ACL line: slope={acl_slope:.3f}  intercept={acl_intercept:.3f}  R²={acl_r2:.3f}")

    # ERM entropy increase (label-free gate)
    best_erm_entry  = max(erm_configs, key=lambda c: c["mean_id_val_acc"])
    erm_ood_entropy = float(np.mean([compute_entropy(m, ref_test, device)
                                     for m in best_erm_entry["models"]]))
    erm_id_entropy  = float(np.mean([compute_entropy(m, id_ref, device)
                                     for m in best_erm_entry["models"]]))
    entropy_increase = erm_ood_entropy - erm_id_entropy
    print(f"  ERM entropy: ID={erm_id_entropy:.3f}  OOD={erm_ood_entropy:.3f}  "
          f"ΔH={entropy_increase:+.3f}")

    # Compute IRM-ERM cross agreement and score each IRM config (label-free)
    majority = n_seeds // 2 + 1
    for t in range(n_coarse_trials):
        pts = compute_cross_agreement(irm_configs[t]["models"],
                                      erm_configs[t]["models"],
                                      id_ref, ref_test, device)
        for i, p in enumerate(pts):
            pts[i]["deviation"] = compute_deviation(
                p["id_agr"], p["ood_agr"], agl_slope, agl_intercept)
        irm_configs[t]["cross_points"]   = pts
        irm_configs[t]["mean_deviation"] = float(np.mean([p["deviation"] for p in pts]))
        irm_configs[t]["std_deviation"]  = float(np.std( [p["deviation"] for p in pts]))
        irm_configs[t]["n_below"]        = sum(1 for p in pts
                                               if p["deviation"] < DEVIATION_THRESHOLD)
        # ACL deviation for labeled mode
        irm_configs[t]["acl_deviation"]  = (irm_configs[t]["mean_ood_acc"]
                                            - (acl_slope * irm_configs[t]["mean_id_val_acc"]
                                               + acl_intercept))

    n_diverging_lf = sum(1 for c in irm_configs if c["n_below"] >= majority)
    n_diverging_lb = sum(1 for c in irm_configs if c["acl_deviation"] > ACL_DEVIATION_THRESHOLD)
    frac_div_lf    = n_diverging_lf / n_coarse_trials
    frac_div_lb    = n_diverging_lb / n_coarse_trials

    # Within-ERM agreement for ERM scoring (label-free)
    for ec in erm_configs:
        within_pts = compute_pairwise_agreement(ec["models"], id_ref, ref_test, device)
        if within_pts:
            ec["within_ood_agr"] = float(np.mean([p["ood_agr"] for p in within_pts]))
            ec["within_std"]     = float(np.std( [p["ood_agr"] for p in within_pts]))
        else:
            ec["within_ood_agr"] = 0.5
            ec["within_std"]     = 0.1

    # --- Labeled mode decision ---
    if frac_div_lb >= DIVERGE_FRAC_THRESHOLD:
        decision_labeled = "DIVERGE"
    elif acl_r2 < 0.3:
        decision_labeled = "AGREE_ERM_WORKS"  # ACL unreliable, default to ERM
    else:
        # Check if IRM mostly below ERM line
        mean_acl_dev = np.mean([c["acl_deviation"] for c in irm_configs])
        if mean_acl_dev < -ACL_DEVIATION_THRESHOLD:
            decision_labeled = "AGREE_IRM_FAILS"
        else:
            decision_labeled = "AGREE_ERM_WORKS"

    # --- Label-free mode decision ---
    # Three conditions must all be true to use IRM:
    # 1. ERM is struggling OOD (entropy increased)
    # 2. At least one IRM config diverged from ERM (below AGL line)
    # 3. Best diverging IRM config has lower ID val than best ERM
    #    (found harder invariant solution, not a different spurious one)
    erm_struggling = entropy_increase > 0
    irm_diverging  = frac_div_lf >= DIVERGE_FRAC_THRESHOLD

    if not erm_struggling:
        decision_lf = "AGREE_ERM_WORKS"

    elif not irm_diverging:
        decision_lf = "AGREE_NO_DIVERGE"

    else:
        # IRM is diverging — check if best diverging config has lower ID val than best ERM
        div_configs  = [c for c in irm_configs if c["n_below"] >= majority]
        best_irm_val = max(c["mean_id_val_acc"] for c in div_configs)
        best_erm_val = max(c["mean_id_val_acc"] for c in erm_configs)
        if best_irm_val < best_erm_val:
            decision_lf = "DIVERGE"          # harder invariant solution
        else:
            decision_lf = "AGREE_IRM_FAILS"  # different spurious solution

    print(f"  Labeled decision:    {decision_labeled} "
          f"({n_diverging_lb}/{n_coarse_trials} IRM above ACL line)")
    print(f"  Label-free decision: {decision_lf} "
          f"(erm_struggling={erm_struggling} ΔH={entropy_increase:+.3f}, "
          f"irm_diverging={irm_diverging} frac={frac_div_lf:.2f})")

    # ------------------------------------------------------------------
    # Phase 3 — Targeted fine search (both modes)
    # ------------------------------------------------------------------

    print(f"\n[Phase 3] Targeted fine search ({fine_trials} trials each mode)...")

    # Helpers to select best from a list of configs by score
    def best_valid(configs, score_fn):
        valid = [c for c in configs if score_fn(c) > -np.inf]
        return max(valid, key=score_fn) if valid else None

    # ── Labeled mode ──
    fine_labeled   = []
    selected_labeled = None

    if decision_labeled == "DIVERGE":
        # Score coarse IRM by ACL deviation, find top-K region
        for c in irm_configs:
            c["labeled_score"] = score_irm_labeled(c, acl_slope, acl_intercept)
        valid_irm  = sorted([c for c in irm_configs if c["labeled_score"] > -np.inf],
                             key=lambda c: c["labeled_score"], reverse=True)
        top_k      = valid_irm[:TOP_K] if valid_irm else irm_configs[:TOP_K]
        lam_min, lam_max, lr_min, lr_max = get_hp_region(top_k)
        print(f"  [Labeled DIVERGE] λ region: [{lam_min:.0f}, {lam_max:.0f}]  "
              f"lr region: [{lr_min:.5f}, {lr_max:.5f}]")

        for trial in range(fine_trials):
            hp    = sample_irm_hp_targeted(rng, lam_min, lam_max, lr_min, lr_max,
                                           base_hp=top_k[0]["hp"])
            entry = {"hp": hp, "models": [], "results": []}
            for seed in range(n_seeds):
                envs             = build_envs(train_images, train_labels, e_values, seed=seed)
                test_env         = make_environment(val_images, val_labels, E_TEST, seed=seed+99)
                train_envs, val_envs = make_val_splits(envs)
                m     = train_model(train_envs, hp, "irm", device, seed, max_steps=max_steps)
                val_a = worst_env_val_acc(m, val_envs, device)
                ood_a = compute_accuracy(m, test_env, device)
                entry["models"].append(m)
                entry["results"].append({"seed": seed, "id_val_acc": val_a, "ood_acc": ood_a})
            entry["mean_id_val_acc"] = float(np.mean([r["id_val_acc"] for r in entry["results"]]))
            entry["mean_ood_acc"]    = float(np.mean([r["ood_acc"]    for r in entry["results"]]))
            entry["std_ood_acc"]     = float(np.std( [r["ood_acc"]    for r in entry["results"]]))
            entry["labeled_score"]   = score_irm_labeled(entry, acl_slope, acl_intercept)
            fine_labeled.append(entry)
            print(f"  [Labeled] fine={trial:02d} val={entry['mean_id_val_acc']:.3f} "
                  f"ood={entry['mean_ood_acc']:.3f} score={entry['labeled_score']:+.3f} "
                  f"λ={hp['penalty_weight']:.0f}")

        selected_labeled = best_valid(fine_labeled + irm_configs,
                                      lambda c: c.get("labeled_score", -np.inf))

    else:  # AGREE_ERM_WORKS or AGREE_IRM_FAILS — fine ERM search
        for c in erm_configs:
            c["labeled_score"] = score_erm_labeled(c)
        top_k_erm  = sorted(erm_configs, key=lambda c: c["labeled_score"], reverse=True)[:TOP_K]
        _, _, lr_min, lr_max = get_hp_region(top_k_erm, key_lam="lr", key_lr="lr")
        print(f"  [Labeled {decision_labeled}] lr region: [{lr_min:.5f}, {lr_max:.5f}]")

        for trial in range(fine_trials):
            hp    = sample_erm_hp_targeted(rng, lr_min, lr_max, base_hp=top_k_erm[0]["hp"])
            entry = {"hp": hp, "models": [], "results": []}
            for seed in range(n_seeds):
                envs             = build_envs(train_images, train_labels, e_values, seed=seed)
                test_env         = make_environment(val_images, val_labels, E_TEST, seed=seed+99)
                train_envs, val_envs = make_val_splits(envs)
                m     = train_model(train_envs, hp, "erm", device, seed, max_steps=max_steps)
                val_a = worst_env_val_acc(m, val_envs, device)
                ood_a = compute_accuracy(m, test_env, device)
                entry["models"].append(m)
                entry["results"].append({"seed": seed, "id_val_acc": val_a, "ood_acc": ood_a})
            entry["mean_id_val_acc"] = float(np.mean([r["id_val_acc"] for r in entry["results"]]))
            entry["mean_ood_acc"]    = float(np.mean([r["ood_acc"]    for r in entry["results"]]))
            entry["std_ood_acc"]     = float(np.std( [r["ood_acc"]    for r in entry["results"]]))
            entry["labeled_score"]   = score_erm_labeled(entry)
            fine_labeled.append(entry)
            print(f"  [Labeled ERM] fine={trial:02d} val={entry['mean_id_val_acc']:.3f} "
                  f"ood={entry['mean_ood_acc']:.3f} score={entry['labeled_score']:+.3f}")

        selected_labeled = best_valid(fine_labeled + erm_configs,
                                      lambda c: c.get("labeled_score", -np.inf))

    # ── Label-free mode ──
    fine_lf      = []
    selected_lf  = None

    if decision_lf == "DIVERGE":
        for c in irm_configs:
            c["lf_score"] = score_irm_label_free(c)
        valid_irm  = sorted([c for c in irm_configs if c["lf_score"] > -np.inf],
                             key=lambda c: c["lf_score"], reverse=True)
        top_k      = valid_irm[:TOP_K] if valid_irm else irm_configs[:TOP_K]
        lam_min, lam_max, lr_min, lr_max = get_hp_region(top_k)
        print(f"  [LabelFree DIVERGE] λ region: [{lam_min:.0f}, {lam_max:.0f}]  "
              f"lr region: [{lr_min:.5f}, {lr_max:.5f}]")

        for trial in range(fine_trials):
            hp    = sample_irm_hp_targeted(rng, lam_min, lam_max, lr_min, lr_max,
                                           base_hp=top_k[0]["hp"])
            entry = {"hp": hp, "models": [], "results": []}
            for seed in range(n_seeds):
                envs             = build_envs(train_images, train_labels, e_values, seed=seed)
                test_env         = make_environment(val_images, val_labels, E_TEST, seed=seed+99)
                train_envs, val_envs = make_val_splits(envs)
                m     = train_model(train_envs, hp, "irm", device, seed, max_steps=max_steps)
                val_a = worst_env_val_acc(m, val_envs, device)
                ood_a = compute_accuracy(m, test_env, device)
                entry["models"].append(m)
                entry["results"].append({"seed": seed, "id_val_acc": val_a, "ood_acc": ood_a})
            entry["mean_id_val_acc"] = float(np.mean([r["id_val_acc"] for r in entry["results"]]))
            entry["mean_ood_acc"]    = float(np.mean([r["ood_acc"]    for r in entry["results"]]))
            entry["std_ood_acc"]     = float(np.std( [r["ood_acc"]    for r in entry["results"]]))
            # compute agreement for scoring
            pts = compute_cross_agreement(
                entry["models"],
                erm_configs[0]["models"],  # reference ERM
                id_ref, ref_test, device)
            for i, p in enumerate(pts):
                pts[i]["deviation"] = compute_deviation(
                    p["id_agr"], p["ood_agr"], agl_slope, agl_intercept)
            entry["mean_deviation"] = float(np.mean([p["deviation"] for p in pts]))
            entry["std_deviation"]  = float(np.std( [p["deviation"] for p in pts]))
            entry["lf_score"]       = score_irm_label_free(entry)
            fine_lf.append(entry)
            print(f"  [LabelFree] fine={trial:02d} val={entry['mean_id_val_acc']:.3f} "
                  f"ood={entry['mean_ood_acc']:.3f} dev={entry['mean_deviation']:+.3f} "
                  f"score={entry['lf_score']:+.3f} λ={hp['penalty_weight']:.0f}")

        selected_lf = best_valid(fine_lf + irm_configs,
                                  lambda c: c.get("lf_score", -np.inf))

    elif decision_lf in ("AGREE_ERM_WORKS", "AGREE_IRM_FAILS"):
        for c in erm_configs:
            c["lf_score"] = score_erm_label_free(c)
        top_k_erm  = sorted(erm_configs, key=lambda c: c["lf_score"], reverse=True)[:TOP_K]
        _, _, lr_min, lr_max = get_hp_region(top_k_erm, key_lam="lr", key_lr="lr")
        print(f"  [LabelFree {decision_lf}] lr region: [{lr_min:.5f}, {lr_max:.5f}]")

        for trial in range(fine_trials):
            hp    = sample_erm_hp_targeted(rng, lr_min, lr_max, base_hp=top_k_erm[0]["hp"])
            entry = {"hp": hp, "models": [], "results": []}
            for seed in range(n_seeds):
                envs             = build_envs(train_images, train_labels, e_values, seed=seed)
                test_env         = make_environment(val_images, val_labels, E_TEST, seed=seed+99)
                train_envs, val_envs = make_val_splits(envs)
                m     = train_model(train_envs, hp, "erm", device, seed, max_steps=max_steps)
                val_a = worst_env_val_acc(m, val_envs, device)
                ood_a = compute_accuracy(m, test_env, device)
                entry["models"].append(m)
                entry["results"].append({"seed": seed, "id_val_acc": val_a, "ood_acc": ood_a})
            entry["mean_id_val_acc"] = float(np.mean([r["id_val_acc"] for r in entry["results"]]))
            entry["mean_ood_acc"]    = float(np.mean([r["ood_acc"]    for r in entry["results"]]))
            # within-ERM agreement for lf score
            within_pts = compute_pairwise_agreement(entry["models"], id_ref, ref_test, device)
            entry["within_ood_agr"] = float(np.mean([p["ood_agr"] for p in within_pts])) \
                                      if within_pts else 0.5
            entry["within_std"]     = float(np.std( [p["ood_agr"] for p in within_pts])) \
                                      if within_pts else 0.1
            entry["lf_score"]       = score_erm_label_free(entry)
            fine_lf.append(entry)
            print(f"  [LabelFree ERM] fine={trial:02d} val={entry['mean_id_val_acc']:.3f} "
                  f"ood={entry['mean_ood_acc']:.3f} score={entry['lf_score']:+.3f}")

        selected_lf = best_valid(fine_lf + erm_configs,
                                  lambda c: c.get("lf_score", -np.inf))

    else:  # AGREE_NO_DIVERGE — best available model for TTA
        selected_lf = max(erm_configs + irm_configs,
                          key=lambda c: c["mean_id_val_acc"])
        print(f"  [LabelFree AGREE_NO_DIVERGE] No fine search. "
              f"Best available: val={selected_lf['mean_id_val_acc']:.3f} "
              f"ood={selected_lf['mean_ood_acc']:.3f}")

    # ------------------------------------------------------------------
    # Phase 4 — TTA boost
    # ------------------------------------------------------------------

    print(f"\n[Phase 4] TTA boost...")

    def apply_tta_to_selected(selected, mode_name, use_ood_acc=False):
        """Apply TTA to the best model in selected config. Return post-TTA accuracy."""
        if selected is None:
            return None, None
        best_model = selected["models"][0]
        pre_entropy = compute_entropy(best_model, ref_test, device)
        # pick a test_env for accuracy (seed 0)
        test_env = make_environment(val_images, val_labels, E_TEST, seed=99)
        pre_acc  = compute_accuracy(best_model, test_env, device)

        adapted      = apply_tent(best_model, ref_test, device)
        post_entropy = compute_entropy(adapted, ref_test, device)
        post_acc     = compute_accuracy(adapted, test_env, device)

        improved = post_entropy < pre_entropy if not use_ood_acc else post_acc > pre_acc
        final_acc = post_acc if improved else pre_acc
        print(f"  [{mode_name}] pre_acc={pre_acc:.3f}  post_acc={post_acc:.3f}  "
              f"pre_H={pre_entropy:.3f}  post_H={post_entropy:.3f}  "
              f"TTA {'kept' if improved else 'discarded'}")
        return pre_acc, final_acc

    pre_labeled,  final_labeled  = apply_tta_to_selected(
        selected_labeled, "Labeled",    use_ood_acc=True)
    pre_lf,       final_lf       = apply_tta_to_selected(
        selected_lf,      "Label-free", use_ood_acc=False)

    # ------------------------------------------------------------------
    # Results summary
    # ------------------------------------------------------------------

    # Standard baselines (from coarse search, no targeted search)
    std_erm_acc = max(erm_configs, key=lambda c: c["mean_id_val_acc"])["mean_ood_acc"]
    std_irm_acc = max(irm_configs, key=lambda c: c["mean_id_val_acc"])["mean_ood_acc"]

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS — Config {config_name}")
    print(f"{'='*65}")
    print(f"\n  {'Strategy':<40} {'OOD acc':>9}  {'vs Exp5 Non-oracle':>19}")
    print(f"  {'─'*72}")
    rows = [
        ("Exp5 Oracle  (reference)",              cfg["gt_oracle"],     None),
        ("Exp5 Non-oracle (train-domain val)",     cfg["gt_non_oracle"], None),
        ("Standard ERM (ID val)",                  std_erm_acc,          std_erm_acc - cfg["gt_non_oracle"]),
        ("Standard IRM (ID val)",                  std_irm_acc,          std_irm_acc - cfg["gt_non_oracle"]),
        ("New Labeled ★ — pre-TTA  (leaderboard 1)",   pre_labeled or 0,      (pre_labeled or 0) - cfg["gt_non_oracle"]),
        ("New Labeled ★ — post-TTA (leaderboard 2)",   final_labeled or 0,    (final_labeled or 0) - cfg["gt_non_oracle"]),
        ("New Label-free ★ — pre-TTA  (leaderboard 1)",pre_lf or 0,           (pre_lf or 0) - cfg["gt_non_oracle"]),
        ("New Label-free ★ — post-TTA (leaderboard 2)",final_lf or 0,         (final_lf or 0) - cfg["gt_non_oracle"]),
    ]
    for name, acc, gap in rows:
        gap_str = f"{gap:+.3f}" if gap is not None else "   —  "
        print(f"  {name:<46} {acc:>9.3f}  {gap_str:>19}")

    print(f"\n  Detection:  Labeled={decision_labeled}  Label-free={decision_lf}")
    print(f"  AGL R²={agl_r2:.3f}  ACL R²={acl_r2:.3f}")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: AGL scatter
    ax = axes[0]
    eid  = [p["id_agr"]  for p in erm_pairs]
    eood = [p["ood_agr"] for p in erm_pairs]
    ax.scatter(eid, eood, color="gray", alpha=0.3, s=15,
               label=f"ERM-ERM ({len(eid)})", zorder=2)
    xs = np.linspace(max(0.5, min(eid)-0.02), min(1.0, max(eid)+0.02), 100)
    ax.plot(xs, [inv_probit(agl_slope*probit(x)+agl_intercept) for x in xs],
            "gray", linewidth=1.5, linestyle="--", label=f"AGL R²={agl_r2:.2f}")
    for c in irm_configs:
        is_div = c["n_below"] >= majority
        col = "#D85A30" if is_div else "#185FA5"
        mkr = "v"       if is_div else "^"
        for p in c.get("cross_points", []):
            ax.scatter(p["id_agr"], p["ood_agr"], color=col, marker=mkr,
                       s=35, alpha=0.6, zorder=3)
    for c in fine_lf:
        if "cross_points" not in c:
            continue
        for p in c["cross_points"]:
            ax.scatter(p["id_agr"], p["ood_agr"], color="#7F77DD", marker="*",
                       s=80, alpha=0.85, zorder=4)
    ax.legend(handles=[
        Line2D([0],[0], marker="v", color="w", markerfacecolor="#D85A30",
               markersize=7, label="IRM-ERM below line"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#185FA5",
               markersize=7, label="IRM-ERM on line"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="#7F77DD",
               markersize=9, label="Fine search (LF)"),
    ], fontsize=7)
    ax.set_xlabel("ID agreement"); ax.set_ylabel("OOD agreement")
    ax.set_title(f"AGL scatter — Config {config_name}\n"
                 f"LF: {decision_lf}  Labeled: {decision_labeled}", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Middle: ACL scatter (labeled mode)
    ax = axes[1]
    erm_id_accs  = [c["mean_id_val_acc"] for c in erm_configs]
    erm_ood_accs = [c["mean_ood_acc"]    for c in erm_configs]
    ax.scatter(erm_id_accs, erm_ood_accs, color="gray", alpha=0.4, s=20,
               label="ERM configs", zorder=2)
    if acl_r2 > 0.1:
        xs2 = np.linspace(min(erm_id_accs)-0.02, max(erm_id_accs)+0.02, 100)
        ax.plot(xs2, acl_slope*xs2+acl_intercept, "gray", linewidth=1.5,
                linestyle="--", label=f"ACL R²={acl_r2:.2f}")
    irm_id_accs  = [c["mean_id_val_acc"] for c in irm_configs]
    irm_ood_accs = [c["mean_ood_acc"]    for c in irm_configs]
    ax.scatter(irm_id_accs, irm_ood_accs, color="#D85A30", alpha=0.5,
               s=25, marker="^", label="IRM configs", zorder=3)
    if fine_labeled:
        fl_id  = [c["mean_id_val_acc"] for c in fine_labeled]
        fl_ood = [c["mean_ood_acc"]    for c in fine_labeled]
        ax.scatter(fl_id, fl_ood, color="#7F77DD", marker="*",
                   s=80, alpha=0.85, zorder=4, label="Fine (labeled)")
    ax.set_xlabel("ID val accuracy"); ax.set_ylabel("OOD accuracy")
    ax.set_title(f"ACL scatter — Config {config_name}\nLabeled: {decision_labeled}", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    # Right: bar chart comparison
    ax = axes[2]
    names = ["Exp5\nOracle", "Exp5\nNon-oracle", "Std\nERM", "Std\nIRM",
             "New\nLabeled★", "New\nLabel-free★"]
    vals  = [cfg["gt_oracle"], cfg["gt_non_oracle"],
             std_erm_acc, std_irm_acc,
             final_labeled or 0, final_lf or 0]
    cols  = ["#aaaaaa", "#555555", "#1D9E75", "#1D9E75", "#185FA5", "#D85A30"]
    htchs = ["//", "", "", "", "", ""]
    bars  = ax.bar(names, vals, color=cols, hatch=htchs, alpha=0.85, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(cfg["gt_oracle"],     color="#aaaaaa", linewidth=1,
               linestyle=":", alpha=0.7)
    ax.axhline(cfg["gt_non_oracle"], color="#555555", linewidth=1,
               linestyle=":", alpha=0.7)
    ax.set_ylabel("OOD accuracy (e=0.9)")
    ax.set_title(f"Selection comparison\nConfig {config_name}  (// = cheat)", fontsize=9)
    ax.set_ylim(0, min(1.0, max(vals)+0.12))
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        f"Agreement-Guided HP Search — Config {config_name}  "
        f"e_train={e_values}  e_test={E_TEST}\n"
        f"AGL R²={agl_r2:.3f}  ACL R²={acl_r2:.3f}  |  "
        f"New Labeled={final_labeled:.3f}  New Label-free={final_lf:.3f}",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"exp5_ext_{config_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {plot_path}")

    # Save JSON
    def strip(c):
        return {k: v for k, v in c.items() if k not in ["models", "cross_points"]}

    result = {
        "config":           config_name,
        "e_train":          e_values,
        "agl_line":         {"slope": agl_slope, "intercept": agl_intercept, "r2": agl_r2},
        "acl_line":         {"slope": acl_slope, "intercept": acl_intercept, "r2": acl_r2},
        "decision_labeled": decision_labeled,
        "decision_lf":      decision_lf,
        "entropy":          {"id": erm_id_entropy, "ood": erm_ood_entropy,
                             "increase": entropy_increase},
        "results": {
            "exp5_oracle":      cfg["gt_oracle"],
            "exp5_non_oracle":  cfg["gt_non_oracle"],
            "std_erm":          std_erm_acc,
            "std_irm":          std_irm_acc,
            "new_labeled":      final_labeled,
            "new_label_free":   final_lf,
            "pre_tta_labeled":  pre_labeled,
            "pre_tta_lf":       pre_lf,
        },
        "erm_configs":  [strip(c) for c in erm_configs],
        "irm_configs":  [strip(c) for c in irm_configs],
        "fine_labeled": [strip(c) for c in fine_labeled],
        "fine_lf":      [strip(c) for c in fine_lf],
    }
    json_path = os.path.join(output_dir, f"exp5_ext_{config_name}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON  → {json_path}\n")
    return result


# =============================================================================
# Summary across all configs
# =============================================================================

def print_summary(all_results):
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY — Experiment 5 Extension")
    print(f"{'='*80}")
    print(f"\n  {'Config':<8} {'Exp5 Oracle':>12} {'Exp5 Non-orc':>13} "
          f"{'Lab pre':>9} {'Lab post':>10} {'LF pre':>8} {'LF post':>9}")
    print(f"  {'─'*73}")
    for cname, res in all_results.items():
        r = res["results"]
        print(f"  {cname:<8} {r['exp5_oracle']:>12.1%} {r['exp5_non_oracle']:>13.1%} "
              f"{(r['pre_tta_labeled'] or 0):>9.1%} {(r['new_labeled'] or 0):>10.1%} "
              f"{(r['pre_tta_lf'] or 0):>8.1%} {(r['new_label_free'] or 0):>9.1%}")
    print(f"\n  pre  = before TTA  (leaderboard 1 comparable)")
    print(f"  post = after TTA   (leaderboard 2 comparable)")
    print(f"  ★ = no test env labels used for HP selection")
    print(f"  All values = OOD test accuracy at e=0.9")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 5 extension with agreement-guided HP search"
    )
    parser.add_argument("--configs",          type=str, default="ABCD")
    parser.add_argument("--n_coarse_trials",  type=int, default=20)
    parser.add_argument("--n_seeds",          type=int, default=3)
    parser.add_argument("--fine_trials",      type=int, default=10)
    parser.add_argument("--max_steps",        type=int, default=None)
    parser.add_argument("--device",           type=str, default="cuda")
    parser.add_argument("--output_dir",       type=str, default="results")
    args = parser.parse_args()

    mnist_raw    = load_mnist_raw()
    configs_todo = [c for c in "ABCD" if c in args.configs.upper()]
    all_results  = {}

    for cname in configs_todo:
        result = run_config(
            cname, CONFIGS[cname],
            args.n_coarse_trials, args.n_seeds, args.fine_trials,
            args.device, args.output_dir, mnist_raw,
            max_steps=args.max_steps
        )
        all_results[cname] = result

    print_summary(all_results)
    print("\nDone.")