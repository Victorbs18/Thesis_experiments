"""
agl_targeted_search.py — Agreement-Guided Targeted HP Search
=============================================================
Extends the 2x2 diagnostic with a principled HP search for IRM
that uses disagreement with ERM as the optimization objective.

THE FULL PIPELINE
-----------------

Phase 1 — Coarse random search
  Train ERM and IRM with random HP configs (same as agl_2x2.py).
  Fit AGL line from ERM-ERM pairs.
  Compute IRM-ERM cross-method agreement.

Phase 2 — Detection
  If IRM-ERM pairs are on the line → mild shift → use ERM, stop.
  If IRM-ERM pairs below the line → strong shift → proceed to Phase 3.

Phase 3 — Targeted HP search (the new contribution)
  Score each IRM config by:
    score(λ) = -mean_deviation        (more below line = better)
             - β × std_deviation      (penalize instability)
    subject to: id_val_acc > τ        (reject collapsed configs)

  Use top-K configs from coarse search to identify promising λ region.
  Run a fine search around that region.
  Select final config by the same score.

Phase 4 — Comparison
  Compare four selection strategies:
    1. Standard non-oracle (ID val acc)   — current practice, often fails
    2. Agreement-guided coarse            — Phase 2 selection, no OOD labels
    3. Agreement-guided fine (targeted)   — Phase 3 selection, no OOD labels
    4. Oracle (OOD acc)                   — upper bound, uses test labels

THE KEY HYPOTHESIS
------------------
The targeted search (Phase 3) closes the gap between standard non-oracle
selection and oracle selection, without ever using OOD labels. The
disagreement with ERM is the signal that guides the search toward the
invariant solution that standard ID val acc selection misses.

Usage:
    python agl_targeted_search.py --env_config diverse --device cuda
    python agl_targeted_search.py --all_configs --device cuda

Env configs (matching Experiment 5):
    original   {0.1, 0.2}  Low diversity,  Low proximity
    diverse    {0.1, 0.5}  High diversity, Low proximity
    proximate  {0.7, 0.8}  Low diversity,  High proximity
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data      import load_mnist_raw, build_envs, make_environment, make_val_splits
from models    import (train_model, compute_accuracy, worst_env_val_acc,
                       get_predictions, sample_erm_hp, sample_irm_hp, HP_RANGES)
from agreement import (compute_pairwise_agreement, compute_cross_agreement,
                       compute_entropy, fit_agl_line, compute_deviation,
                       inv_probit, probit,
                       DEVIATION_THRESHOLD, ENTROPY_THRESHOLD)


# =============================================================================
# Configuration
# =============================================================================

ENV_CONFIGS = {
    "A": {"e_train": [0.1, 0.2], "diversity": "Low",  "proximity": "Low",
          "expected": "BOTH_FAIL",
          "gt_erm": 0.175, "gt_irm_oracle": 0.670, "gt_irm_train_val": 0.106},
    "B": {"e_train": [0.7, 0.8], "diversity": "Low",  "proximity": "High",
          "expected": "AGREE",
          "gt_erm": 0.781, "gt_irm_oracle": 0.817, "gt_irm_train_val": 0.779},
    "C": {"e_train": [0.1, 0.5], "diversity": "High", "proximity": "Low",
          "expected": "DIVERGE",
          "gt_erm": 0.319, "gt_irm_oracle": 0.717, "gt_irm_train_val": 0.694},
    "D": {"e_train": [0.1, 0.8], "diversity": "High", "proximity": "High",
          "expected": "AGREE",
          "gt_erm": 0.724, "gt_irm_oracle": 0.728, "gt_irm_train_val": 0.726},
}

E_TEST           = 0.9
ID_ACC_THRESHOLD = 0.55   # below this → IRM collapsed, reject config
BETA             = 0.5    # penalty weight for instability in score
TOP_K            = 3      # number of top coarse configs to guide fine search
ERM_ENTROPY_GATE = 0.40   # ERM confident OOD = ERM working = no search needed
FINE_TRIALS      = 10     # number of fine search trials around top-K region



# =============================================================================
# Scoring function — the core of the targeted search
# =============================================================================

def score_irm_config(cfg, id_acc_threshold=ID_ACC_THRESHOLD, beta=BETA):
    """
    Score an IRM config by how much it disagrees with ERM under shift.

    score = -mean_deviation - β × std_deviation
            if id_val_acc > threshold else -inf

    More negative deviation = further below AGL line = stronger divergence
    from ERM = more likely IRM found invariant features.

    Penalizing std_deviation rejects configs that diverge inconsistently
    across seeds (lucky divergence, not principled invariant learning).

    Rejecting configs with low id_val_acc filters out degenerate solutions
    that diverge from ERM by collapsing rather than by learning invariance.
    """
    if cfg["mean_id_val_acc"] < id_acc_threshold:
        return -np.inf   # collapsed config, reject

    # Primary signal: deviation below line (negative = below = good)
    # We negate because we want to MAXIMIZE score but deviation is negative when good
    score = -cfg["mean_deviation"] - beta * cfg["std_deviation"]
    return score


def get_lambda_region(top_configs):
    """
    Extract the λ region suggested by top-K coarse configs.
    Returns (lambda_min, lambda_max) for the fine search.
    """
    lambdas = [c["hp"]["penalty_weight"] for c in top_configs]
    lam_min = min(lambdas) / 5    # expand slightly below
    lam_max = max(lambdas) * 5    # expand slightly above
    return lam_min, lam_max


def sample_irm_hp_targeted(rng, lam_min, lam_max, base_hp=None):
    """
    Sample IRM HP config with λ constrained to the promising region.
    Other HPs either sampled randomly or inherited from base_hp.
    """
    hp = sample_irm_hp(rng) if base_hp is None else dict(base_hp)

    # Sample λ from the promising region (log-uniform)
    log_lam = rng.uniform(np.log10(max(lam_min, 1)),
                          np.log10(min(lam_max, 1e7)))
    hp["penalty_weight"] = float(10 ** log_lam)

    # Also vary anneal iters and steps within ranges
    hp["penalty_anneal_iters"] = int(rng.choice(HP_RANGES["penalty_anneal_iters"]))
    hp["steps"]                = int(rng.choice(HP_RANGES["steps"]))
    return hp


# =============================================================================
# Main experiment
# =============================================================================

def run(env_config, n_coarse_trials, n_seeds, device, output_dir,
        mnist_raw, max_steps=None):

    os.makedirs(output_dir, exist_ok=True)
    cfg      = ENV_CONFIGS[env_config]
    e_values = cfg["e_train"]
    train_images, train_labels, val_images, val_labels = mnist_raw
    rng      = np.random.RandomState(42)

    # Fixed reference data for agreement (no OOD labels used)
    ref_envs = build_envs(train_images, train_labels, e_values, seed=0)
    ref_test = make_environment(val_images, val_labels, E_TEST, seed=99)
    id_ref   = ref_envs[0]

    print(f"\n{'='*65}")
    print(f"  Agreement-Guided Targeted HP Search")
    print(f"  env={env_config}  e_train={e_values}  e_test={E_TEST}")
    print(f"  n_coarse={n_coarse_trials}  n_seeds={n_seeds}  "
          f"fine_trials={FINE_TRIALS}  device={device}")
    print(f"  Ground truth: ERM={cfg['gt_erm']:.1%}  "
          f"IRM oracle={cfg['gt_irm_oracle']:.1%}  "
          f"IRM train_val={cfg['gt_irm_train_val']:.1%}")
    print(f"{'='*65}")

    t_total = time.time()

    # ------------------------------------------------------------------
    # Phase 1 — Coarse random search
    # ------------------------------------------------------------------

    print(f"\n[Phase 1] Coarse random search "
          f"({n_coarse_trials} trials × {n_seeds} seeds)...")

    erm_configs = []
    irm_configs = []

    for trial in range(n_coarse_trials):
        erm_hp = sample_erm_hp(rng)
        irm_hp = sample_irm_hp(rng)

        erm_entry = {"hp": erm_hp, "models": [], "results": []}
        irm_entry = {"hp": irm_hp, "models": [], "results": []}

        for seed in range(n_seeds):
            envs               = build_envs(train_images, train_labels, e_values, seed=seed)
            test_env           = make_environment(val_images, val_labels, E_TEST, seed=seed+99)
            train_envs, val_envs = make_val_splits(envs)

            steps_cap = max_steps

            # ERM
            m_erm     = train_model(train_envs, erm_hp, "erm", device, seed, max_steps=steps_cap)
            erm_val_a = worst_env_val_acc(m_erm, val_envs, device)
            erm_ood_a = compute_accuracy(m_erm, test_env, device)
            erm_entry["models"].append(m_erm)
            erm_entry["results"].append({"seed": seed, "id_val_acc": erm_val_a,
                                          "ood_acc": erm_ood_a})

            # IRM
            m_irm     = train_model(train_envs, irm_hp, "irm", device, seed, max_steps=steps_cap)
            irm_val_a = worst_env_val_acc(m_irm, val_envs, device)
            irm_ood_a = compute_accuracy(m_irm, test_env, device)
            irm_entry["models"].append(m_irm)
            irm_entry["results"].append({"seed": seed, "id_val_acc": irm_val_a,
                                          "ood_acc": irm_ood_a})

        for entry in [erm_entry, irm_entry]:
            entry["mean_id_val_acc"] = float(np.mean([r["id_val_acc"] for r in entry["results"]]))
            entry["mean_ood_acc"]    = float(np.mean([r["ood_acc"]    for r in entry["results"]]))

        erm_configs.append(erm_entry)
        irm_configs.append(irm_entry)

        print(f"  coarse trial={trial:02d} | "
              f"ERM val={erm_entry['mean_id_val_acc']:.3f} ood={erm_entry['mean_ood_acc']:.3f} | "
              f"IRM val={irm_entry['mean_id_val_acc']:.3f} ood={irm_entry['mean_ood_acc']:.3f} "
              f"λ={irm_hp['penalty_weight']:.0f}")

    # ------------------------------------------------------------------
    # Phase 2 — Fit AGL line + compute cross-method agreement
    # ------------------------------------------------------------------

    print(f"\n[Phase 2] Fitting AGL line and computing cross-method agreement...")

    erm_all_models = [m for c in erm_configs for m in c["models"]]
    erm_erm_points = compute_pairwise_agreement(erm_all_models, id_ref, ref_test, device)
    slope, intercept, r2 = fit_agl_line(erm_erm_points)
    print(f"  AGL line: slope={slope:.3f}  intercept={intercept:.3f}  R²={r2:.3f}")

    # Compute cross-method agreement and score each IRM config
    for trial in range(n_coarse_trials):
        irm_ms = irm_configs[trial]["models"]
        erm_ms = erm_configs[trial]["models"]
        points = compute_cross_agreement(irm_ms, erm_ms, id_ref, ref_test, device)

        for i, p in enumerate(points):
            dev = compute_deviation(p["id_agr"], p["ood_agr"], slope, intercept)
            points[i]["deviation"] = dev

        irm_configs[trial]["cross_points"]   = points
        irm_configs[trial]["mean_deviation"] = float(np.mean([p["deviation"] for p in points]))
        irm_configs[trial]["std_deviation"]  = float(np.std( [p["deviation"] for p in points]))
        irm_configs[trial]["n_below"]        = sum(1 for p in points
                                                   if p["deviation"] < DEVIATION_THRESHOLD)
        irm_configs[trial]["score"]          = score_irm_config(irm_configs[trial])

    # Detection decision
    majority    = n_seeds // 2 + 1
    n_diverging = sum(1 for c in irm_configs if c["n_below"] >= majority)
    frac_div    = n_diverging / n_coarse_trials

    best_erm_entry  = max(erm_configs, key=lambda c: c["mean_id_val_acc"])
    erm_ood_entropy = float(np.mean([
        compute_entropy(m, ref_test, device)
        for m in best_erm_entry["models"]
    ]))

    print(f"\n  Detection: {n_diverging}/{n_coarse_trials} IRM configs diverge "
          f"from ERM (frac={frac_div:.2f})")
    print(f"  ERM OOD entropy: {erm_ood_entropy:.3f} "
          f"({'uncertain' if erm_ood_entropy > ERM_ENTROPY_GATE else 'confident'})")

    if erm_ood_entropy <= ERM_ENTROPY_GATE:
        print(f"  → AGREE_ERM_WORKS: ERM confident OOD. No targeted search needed.")
        decision = "AGREE_ERM_WORKS"
    elif frac_div < 0.2:
        print(f"  → AGREE_NO_DIVERGE: ERM uncertain but IRM not diverging.")
        decision = "AGREE_NO_DIVERGE"
    else:
        print(f"  → DIVERGE: ERM failing AND IRM diverging. Proceed to targeted search.")
        decision = "DIVERGE"

    # Standard non-oracle selection (ID val acc)
    standard_irm = max(irm_configs, key=lambda c: c["mean_id_val_acc"])
    oracle_irm   = max(irm_configs, key=lambda c: c["mean_ood_acc"])
    val_erm      = max(erm_configs, key=lambda c: c["mean_id_val_acc"])

    # Agreement-guided coarse selection
    diverging_coarse = [c for c in irm_configs if c["n_below"] >= majority
                        and c["score"] > -np.inf]
    if diverging_coarse:
        agr_coarse = max(diverging_coarse, key=lambda c: c["score"])
    else:
        agr_coarse = standard_irm

    print(f"\n  Coarse results:")
    print(f"  {'Config':<35} {'OOD acc':>9}  {'λ':>10}")
    print(f"  {'─'*58}")
    print(f"  {'ERM (ID val)':<35} {val_erm['mean_ood_acc']:>9.3f}  {'—':>10}")
    print(f"  {'IRM standard (ID val acc)':<35} {standard_irm['mean_ood_acc']:>9.3f}  "
          f"{standard_irm['hp']['penalty_weight']:>10.0f}")
    print(f"  {'IRM agr-guided coarse':<35} {agr_coarse['mean_ood_acc']:>9.3f}  "
          f"{agr_coarse['hp']['penalty_weight']:>10.0f}")
    print(f"  {'IRM oracle //':<35} {oracle_irm['mean_ood_acc']:>9.3f}  "
          f"{oracle_irm['hp']['penalty_weight']:>10.0f}")

    # ------------------------------------------------------------------
    # Phase 3 — Targeted fine search (only if DIVERGE)
    # ------------------------------------------------------------------

    fine_configs = []
    agr_fine     = agr_coarse  # fallback if no fine search

    if decision == "DIVERGE":
        print(f"\n[Phase 3] Targeted fine search ({FINE_TRIALS} trials)...")

        # Identify λ region from top-K scoring coarse configs
        valid_coarse = [c for c in irm_configs if c["score"] > -np.inf]
        valid_coarse.sort(key=lambda c: c["score"], reverse=True)
        top_k        = valid_coarse[:TOP_K]
        lam_min, lam_max = get_lambda_region(top_k)

        print(f"  Top-{TOP_K} coarse configs suggest λ region: "
              f"[{lam_min:.0f}, {lam_max:.0f}]")
        print(f"  Top-{TOP_K} λ values: "
              f"{[c['hp']['penalty_weight'] for c in top_k]}")

        for trial in range(FINE_TRIALS):
            # Sample HP with λ from the promising region
            # Inherit shared HPs from the best coarse config
            irm_hp = sample_irm_hp_targeted(rng, lam_min, lam_max,
                                             base_hp=top_k[0]["hp"])

            fine_entry = {"hp": irm_hp, "models": [], "results": []}

            for seed in range(n_seeds):
                envs               = build_envs(train_images, train_labels, e_values, seed=seed)
                test_env           = make_environment(val_images, val_labels, E_TEST, seed=seed+99)
                train_envs, val_envs = make_val_splits(envs)

                m_irm     = train_model(train_envs, irm_hp, "irm", device, seed,
                                        max_steps=max_steps)
                irm_val_a = worst_env_val_acc(m_irm, val_envs, device)
                irm_ood_a = compute_accuracy(m_irm, test_env, device)
                fine_entry["models"].append(m_irm)
                fine_entry["results"].append({"seed": seed,
                                               "id_val_acc": irm_val_a,
                                               "ood_acc": irm_ood_a})

            fine_entry["mean_id_val_acc"] = float(np.mean(
                [r["id_val_acc"] for r in fine_entry["results"]]))
            fine_entry["mean_ood_acc"]    = float(np.mean(
                [r["ood_acc"] for r in fine_entry["results"]]))

            # Compute agreement and score
            irm_ms = fine_entry["models"]
            erm_ms = erm_configs[0]["models"]  # use first ERM as reference
            points = compute_cross_agreement(irm_ms, erm_ms, id_ref, ref_test, device)
            for i, p in enumerate(points):
                points[i]["deviation"] = compute_deviation(
                    p["id_agr"], p["ood_agr"], slope, intercept)

            fine_entry["cross_points"]   = points
            fine_entry["mean_deviation"] = float(np.mean([p["deviation"] for p in points]))
            fine_entry["std_deviation"]  = float(np.std( [p["deviation"] for p in points]))
            fine_entry["n_below"]        = sum(1 for p in points
                                               if p["deviation"] < DEVIATION_THRESHOLD)
            fine_entry["score"]          = score_irm_config(fine_entry)
            fine_configs.append(fine_entry)

            print(f"  fine trial={trial:02d} | "
                  f"val={fine_entry['mean_id_val_acc']:.3f} "
                  f"ood={fine_entry['mean_ood_acc']:.3f} "
                  f"dev={fine_entry['mean_deviation']:+.3f} "
                  f"score={fine_entry['score']:+.3f} "
                  f"λ={irm_hp['penalty_weight']:.0f}")

        # Select best fine config by agreement score
        valid_fine = [c for c in fine_configs if c["score"] > -np.inf]
        if valid_fine:
            agr_fine = max(valid_fine, key=lambda c: c["score"])
        else:
            agr_fine = agr_coarse
            print("  No valid fine configs found, falling back to coarse selection")

    # ------------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------------

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS — {env_config}")
    print(f"{'='*65}")
    print(f"\n  {'Strategy':<40} {'OOD acc':>9}  {'vs GT train_val':>16}  {'λ':>10}")
    print(f"  {'─'*80}")

    gt_tv = cfg["gt_irm_train_val"]
    rows  = [
        ("ERM (ID val — honest baseline)",   val_erm["mean_ood_acc"],      None,                                "—"),
        ("IRM standard (ID val acc)",         standard_irm["mean_ood_acc"], standard_irm["mean_ood_acc"]-gt_tv, f"{standard_irm['hp']['penalty_weight']:.0f}"),
        ("IRM agr-guided coarse",             agr_coarse["mean_ood_acc"],   agr_coarse["mean_ood_acc"]-gt_tv,   f"{agr_coarse['hp']['penalty_weight']:.0f}"),
        ("IRM agr-guided fine (targeted) ★", agr_fine["mean_ood_acc"],     agr_fine["mean_ood_acc"]-gt_tv,     f"{agr_fine['hp']['penalty_weight']:.0f}"),
        ("IRM oracle // (OOD labels)",        oracle_irm["mean_ood_acc"],   oracle_irm["mean_ood_acc"]-gt_tv,   f"{oracle_irm['hp']['penalty_weight']:.0f}"),
    ]
    for name, ood, gap, lam in rows:
        gap_str = f"{gap:+.3f}" if gap is not None else "   —  "
        print(f"  {name:<40} {ood:>9.3f}  {gap_str:>16}  {lam:>10}")

    print(f"\n  Ground truth references (Exp 5):")
    print(f"    ERM:             {cfg['gt_erm']:.3f}")
    print(f"    IRM train_val:   {cfg['gt_irm_train_val']:.3f}")
    print(f"    IRM oracle:      {cfg['gt_irm_oracle']:.3f}")

    gap_fine_vs_standard = agr_fine["mean_ood_acc"] - standard_irm["mean_ood_acc"]
    gap_fine_vs_oracle   = agr_fine["mean_ood_acc"] - oracle_irm["mean_ood_acc"]
    print(f"\n  Key gaps:")
    print(f"    Fine targeted vs standard non-oracle: {gap_fine_vs_standard:+.3f}")
    print(f"    Fine targeted vs oracle (ceiling):    {gap_fine_vs_oracle:+.3f}")
    print(f"    Total time: {(time.time()-t_total)/60:.1f} min")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: AGL scatter with coarse + fine points
    ax = axes[0]
    eid  = [p["id_agr"]  for p in erm_erm_points]
    eood = [p["ood_agr"] for p in erm_erm_points]
    ax.scatter(eid, eood, color="gray", alpha=0.3, s=15,
               label=f"ERM-ERM ({len(eid)} pairs)", zorder=2)

    xs = np.linspace(max(0.5, min(eid)-0.02), min(1.0, max(eid)+0.02), 100)
    ax.plot(xs, [inv_probit(slope*probit(x)+intercept) for x in xs],
            "gray", linewidth=1.5, linestyle="--",
            label=f"AGL line R²={r2:.2f}", zorder=2)

    # Coarse IRM-ERM points
    for c in irm_configs:
        is_div = c["n_below"] >= majority
        col    = "#D85A30" if is_div else "#185FA5"
        mkr    = "v"       if is_div else "^"
        for p in c["cross_points"]:
            ax.scatter(p["id_agr"], p["ood_agr"],
                       color=col, marker=mkr, s=35, alpha=0.6, zorder=3)

    # Fine search points
    for c in fine_configs:
        for p in c["cross_points"]:
            ax.scatter(p["id_agr"], p["ood_agr"],
                       color="#7F77DD", marker="*", s=80, alpha=0.85, zorder=4)

    ax.legend(handles=[
        Line2D([0],[0], marker="v", color="w", markerfacecolor="#D85A30",
               markersize=7, label="Coarse: below line"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#185FA5",
               markersize=7, label="Coarse: on line"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="#7F77DD",
               markersize=9, label="Fine search"),
    ], fontsize=8)
    ax.set_xlabel("ID agreement"); ax.set_ylabel("OOD agreement")
    ax.set_title(f"AGL scatter — {env_config}\n"
                 f"▼ coarse diverging   * fine targeted", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Middle: Score vs OOD accuracy (validation of score as proxy)
    ax = axes[1]
    all_configs = irm_configs + fine_configs
    scores = [c["score"] for c in all_configs if c["score"] > -np.inf]
    oods   = [c["mean_ood_acc"] for c in all_configs if c["score"] > -np.inf]
    colors = (["#D85A30" if c["n_below"] >= majority else "#185FA5"
               for c in irm_configs if c["score"] > -np.inf] +
              ["#7F77DD"] * sum(1 for c in fine_configs if c["score"] > -np.inf))

    ax.scatter(scores, oods, c=colors, s=60, alpha=0.8, zorder=3)

    # Highlight selected configs
    ax.scatter([agr_fine["score"]], [agr_fine["mean_ood_acc"]],
               color="#7F77DD", marker="*", s=200, zorder=5,
               label=f"Fine selected ({agr_fine['mean_ood_acc']:.3f})")
    ax.scatter([oracle_irm["score"]], [oracle_irm["mean_ood_acc"]],
               color="#185FA5", marker="D", s=100, zorder=5,
               label=f"Oracle ({oracle_irm['mean_ood_acc']:.3f})")
    ax.scatter([standard_irm["score"]], [standard_irm["mean_ood_acc"]],
               color="#1D9E75", marker="s", s=100, zorder=5,
               label=f"Standard ({standard_irm['mean_ood_acc']:.3f})")

    if len(scores) > 2:
        from scipy import stats
        corr, pval = stats.pearsonr(scores, oods)
        m, b = np.polyfit(scores, oods, 1)
        xf   = np.linspace(min(scores)-0.05, max(scores)+0.05, 100)
        ax.plot(xf, m*xf+b, color="gray", linewidth=1.2, alpha=0.5,
                label=f"r={corr:.2f} p={pval:.3f}")

    ax.set_xlabel("Agreement score\n(higher = more below line = stronger invariance)")
    ax.set_ylabel("OOD accuracy")
    ax.set_title("Does score predict OOD accuracy?\n(validates score as label-free proxy)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    # Right: Final comparison bar chart
    ax = axes[2]
    names = ["ERM\n(ID val)", "IRM\nstandard", "IRM\ncoarse\nagr", "IRM\nfine\n★", "IRM\noracle //"]
    vals  = [val_erm["mean_ood_acc"], standard_irm["mean_ood_acc"],
             agr_coarse["mean_ood_acc"], agr_fine["mean_ood_acc"],
             oracle_irm["mean_ood_acc"]]
    cols  = ["#555555", "#1D9E75", "#D85A30", "#7F77DD", "#185FA5"]
    htchs = ["", "", "", "", "//"]
    bars  = ax.bar(names, vals, color=cols, hatch=htchs, alpha=0.85, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(cfg["gt_irm_oracle"], color="#185FA5", linewidth=1,
               linestyle=":", alpha=0.6, label=f"GT oracle={cfg['gt_irm_oracle']:.2f}")
    ax.axhline(cfg["gt_irm_train_val"], color="#1D9E75", linewidth=1,
               linestyle=":", alpha=0.6, label=f"GT train_val={cfg['gt_irm_train_val']:.2f}")
    ax.set_ylabel("OOD accuracy")
    ax.set_title("Selection strategy comparison\n(// = requires OOD labels  ★ = targeted)", fontsize=9)
    ax.set_ylim(0, min(1.0, max(vals) + 0.12))
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        f"Agreement-Guided Targeted HP Search — {env_config}  "
        f"e_train={ENV_CONFIGS[env_config]['e_train']}  e_test={E_TEST}\n"
        f"AGL R²={r2:.3f}  |  Decision: {decision}  |  "
        f"Fine targeted vs standard: {gap_fine_vs_standard:+.3f}  |  "
        f"vs oracle: {gap_fine_vs_oracle:+.3f}",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"targeted_search_{env_config}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {plot_path}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------

    def strip(c):
        out = {k: v for k, v in c.items() if k not in ["models"]}
        return out

    result = {
        "env_config": env_config,
        "e_train":    e_values,
        "e_test":     E_TEST,
        "decision":   decision,
        "agl_line":   {"slope": slope, "intercept": intercept, "r2": r2},
        "selection": {
            "erm_val":       val_erm["mean_ood_acc"],
            "irm_standard":  standard_irm["mean_ood_acc"],
            "irm_agr_coarse":agr_coarse["mean_ood_acc"],
            "irm_agr_fine":  agr_fine["mean_ood_acc"],
            "irm_oracle":    oracle_irm["mean_ood_acc"],
        },
        "gaps": {
            "fine_vs_standard": gap_fine_vs_standard,
            "fine_vs_oracle":   gap_fine_vs_oracle,
        },
        "ground_truth": cfg,
        "coarse_configs": [strip(c) for c in irm_configs],
        "fine_configs":   [strip(c) for c in fine_configs],
        "erm_erm_points": erm_erm_points,
    }

    json_path = os.path.join(output_dir, f"targeted_search_{env_config}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON  → {json_path}\n")
    return result


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agreement-guided targeted HP search for IRM"
    )
    parser.add_argument("--env_config",     type=str, default="C",
                        choices=["A", "B", "C", "D"],
                        help="Which 2x2 config to run")
    parser.add_argument("--all_configs",    action="store_true",
                        help="Run all four ABCD configs sequentially")
    parser.add_argument("--n_coarse_trials",type=int, default=15,
                        help="Random HP configs in coarse search")
    parser.add_argument("--n_seeds",        type=int, default=3,
                        help="Seeds per HP config")
    parser.add_argument("--max_steps",      type=int, default=None,
                        help="Cap training steps. None=use sampled. 201=fast check.")
    parser.add_argument("--device",         type=str, default="cpu")
    parser.add_argument("--output_dir",     type=str, default="results")
    args = parser.parse_args()

    mnist_raw = load_mnist_raw()
    configs   = (["A", "B", "C", "D"]
                 if args.all_configs else [args.env_config])

    for env_cfg in configs:
        run(env_cfg, args.n_coarse_trials, args.n_seeds,
            args.device, args.output_dir, mnist_raw,
            max_steps=args.max_steps)

    print("Done.")