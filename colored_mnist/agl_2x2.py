"""
agl_2x2.py — AGL Diagnostic on the 2x2 Proximity × Diversity Matrix
=====================================================================
Replicates Experiment 5 (proximity vs diversity) from the practical work
and adds the IRM-ERM agreement diagnostic on top.

The 2x2 matrix gives four qualitatively different scenarios, each with
a known ground truth from your existing experiments. This validates
whether the agreement signal correctly identifies when IRM helps.

Config  e_train     Diversity  Proximity  Expected    Ground truth (from Exp 5)
A       {0.1, 0.2}  Low        Low        AGREE*      IRM oracle=67%, non-oracle=10%
B       {0.7, 0.8}  Low        High       AGREE       IRM=82%, ERM=78% (both good)
C       {0.1, 0.5}  High       Low        DIVERGE     IRM oracle=71%, non-oracle=69%
D       {0.1, 0.8}  High       High       WEAK DIV    IRM=73%, ERM=72% (marginal)

* Config A: oracle IRM diverges but non-oracle collapses.
  The framework should detect DIVERGE for good HP configs only.

Usage:
    python agl_2x2.py --n_trials 20 --n_seeds 3 --device cpu
    python agl_2x2.py --n_trials 30 --n_seeds 5 --device cpu  (thesis quality)
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
from models    import train_model, compute_accuracy, worst_env_val_acc, \
                      sample_erm_hp, sample_irm_hp, get_predictions
from agreement import (compute_pairwise_agreement, compute_cross_agreement,
                       compute_entropy, fit_agl_line, compute_deviation,
                       predict_ood_agreement, make_decision,
                       probit, inv_probit,
                       DEVIATION_THRESHOLD, ENTROPY_THRESHOLD)


# =============================================================================
# 2x2 configuration
# =============================================================================

CONFIGS = {
    "A": {
        "e_train":   [0.1, 0.2],
        "diversity": "Low",
        "proximity": "Low",
        "expected":  "BOTH_FAIL",
        "gt_erm":            0.175,
        "gt_irm_oracle":     0.670,
        "gt_irm_train_val":  0.106,  # train domain val from Exp 5
    },
    "B": {
        "e_train":   [0.7, 0.8],
        "diversity": "Low",
        "proximity": "High",
        "expected":  "AGREE",
        "gt_erm":            0.781,
        "gt_irm_oracle":     0.817,
        "gt_irm_train_val":  0.779,
    },
    "C": {
        "e_train":   [0.1, 0.5],
        "diversity": "High",
        "proximity": "Low",
        "expected":  "DIVERGE",
        "gt_erm":            0.319,
        "gt_irm_oracle":     0.717,
        "gt_irm_train_val":  0.694,
    },
    "D": {
        "e_train":   [0.1, 0.8],
        "diversity": "High",
        "proximity": "High",
        "expected":  "AGREE",
        "gt_erm":            0.724,
        "gt_irm_oracle":     0.728,
        "gt_irm_train_val":  0.726,
    },
}
E_TEST = 0.9


# =============================================================================
# Single config experiment
# =============================================================================

def run_config(config_name, cfg, n_trials, n_seeds, device,
               output_dir, mnist_raw,max_steps=None):

    e_values = cfg["e_train"]
    train_images, train_labels, val_images, val_labels = mnist_raw
    rng = np.random.RandomState(42)

    # Fixed reference environments for agreement computation (no labels used OOD)
    ref_envs = build_envs(train_images, train_labels, e_values, seed=0)
    ref_test = make_environment(val_images, val_labels, E_TEST, seed=99)
    id_ref   = ref_envs[0]

    print(f"\n  Config {config_name}: e_train={e_values}  "
          f"diversity={cfg['diversity']}  proximity={cfg['proximity']}")
    print(f"  Expected: {cfg['expected']}")
    print(f"  Ground truth (from Exp 5): "
          f"ERM={cfg['gt_erm']:.1%}  "
          f"IRM oracle={cfg['gt_irm_oracle']:.1%}  "
          f"IRM train_val={cfg['gt_irm_train_val']:.1%}")
    print()

    t0 = time.time()

    # ------------------------------------------------------------------
    # Train n_trials HP configs × n_seeds each
    # ------------------------------------------------------------------

    erm_configs = []
    irm_configs = []

    for trial in range(n_trials):
        erm_hp = sample_erm_hp(rng)
        irm_hp = sample_irm_hp(rng)

        erm_entry = {"hp": erm_hp, "models": [], "preds_id": [], "preds_ood": []}
        irm_entry = {"hp": irm_hp, "models": [], "preds_id": [], "preds_ood": []}

        for seed in range(n_seeds):
            envs             = build_envs(train_images, train_labels, e_values, seed=seed)
            test_env         = make_environment(val_images, val_labels, E_TEST, seed=seed+99)
            train_envs, val_envs = make_val_splits(envs)

            # ERM
            m_erm     = train_model(train_envs, erm_hp, "erm", device, seed,max_steps=max_steps)
            erm_val_a = worst_env_val_acc(m_erm, val_envs, device)
            erm_ood_a = compute_accuracy(m_erm, test_env, device)
            erm_H     = compute_entropy(m_erm, ref_test, device)
            erm_entry["models"].append(m_erm)
            erm_entry.setdefault("results", []).append({
                "seed": seed, "id_val_acc": erm_val_a,
                "ood_acc": erm_ood_a, "entropy": erm_H,
            })

            # IRM
            m_irm     = train_model(train_envs, irm_hp, "irm", device, seed,max_steps=max_steps)
            irm_val_a = worst_env_val_acc(m_irm, val_envs, device)
            irm_ood_a = compute_accuracy(m_irm, test_env, device)
            irm_H     = compute_entropy(m_irm, ref_test, device)
            irm_entry["models"].append(m_irm)
            irm_entry.setdefault("results", []).append({
                "seed": seed, "id_val_acc": irm_val_a,
                "ood_acc": irm_ood_a, "entropy": irm_H,
            })

        # Aggregate per config
        for entry in [erm_entry, irm_entry]:
            entry["mean_id_val_acc"] = float(np.mean([r["id_val_acc"] for r in entry["results"]]))
            entry["mean_ood_acc"]    = float(np.mean([r["ood_acc"]    for r in entry["results"]]))
            entry["mean_entropy"]    = float(np.mean([r["entropy"]    for r in entry["results"]]))

        erm_configs.append(erm_entry)
        irm_configs.append(irm_entry)

        print(f"    trial={trial:02d} | "
              f"ERM  val={erm_entry['mean_id_val_acc']:.3f} "
              f"ood={erm_entry['mean_ood_acc']:.3f} "
              f"H={erm_entry['mean_entropy']:.3f} | "
              f"IRM  val={irm_entry['mean_id_val_acc']:.3f} "
              f"ood={irm_entry['mean_ood_acc']:.3f} "
              f"H={irm_entry['mean_entropy']:.3f} "
              f"λ={irm_hp['penalty_weight']:.0f}")

    print(f"\n    Training done in {(time.time()-t0)/60:.1f} min")

    # ------------------------------------------------------------------
    # Fit AGL line from ERM-ERM pairs
    # ------------------------------------------------------------------

    erm_all_models = [m for cfg_e in erm_configs for m in cfg_e["models"]]
    erm_erm_points = compute_pairwise_agreement(erm_all_models, id_ref, ref_test, device)
    slope, intercept, r2 = fit_agl_line(erm_erm_points)

    print(f"\n    AGL line: slope={slope:.3f}  intercept={intercept:.3f}  R²={r2:.3f}")

    # ------------------------------------------------------------------
    # Compute IRM-ERM cross agreement per trial
    # ------------------------------------------------------------------

    for trial in range(n_trials):
        irm_ms  = irm_configs[trial]["models"]
        erm_ms  = erm_configs[trial]["models"]
        points  = compute_cross_agreement(irm_ms, erm_ms, id_ref, ref_test, device)

        for i, p in enumerate(points):
            dev = compute_deviation(p["id_agr"], p["ood_agr"], slope, intercept)
            irm_configs[trial]["results"][i].update({
                "id_agr": p["id_agr"], "ood_agr": p["ood_agr"], "deviation": dev
            })
            points[i]["deviation"] = dev

        irm_configs[trial]["cross_points"]    = points
        irm_configs[trial]["mean_deviation"]  = float(np.mean([p["deviation"] for p in points]))
        irm_configs[trial]["std_deviation"]   = float(np.std( [p["deviation"] for p in points]))
        irm_configs[trial]["n_below"]         = sum(1 for p in points
                                                    if p["deviation"] < DEVIATION_THRESHOLD)

    # ------------------------------------------------------------------
    # Decision per trial + overall
    # ------------------------------------------------------------------

    all_cross_points = [p for cfg_i in irm_configs for p in cfg_i["cross_points"]]
    all_erm_H        = [cfg_e["mean_entropy"] for cfg_e in erm_configs]
    all_irm_H        = [cfg_i["mean_entropy"] for cfg_i in irm_configs]

    decision_result  = make_decision(all_cross_points, all_erm_H, all_irm_H)

    # ------------------------------------------------------------------
    # Selection comparison
    # ------------------------------------------------------------------

    majority     = n_seeds // 2 + 1
    oracle_irm   = max(irm_configs, key=lambda c: c["mean_ood_acc"])
    oracle_erm   = max(erm_configs, key=lambda c: c["mean_ood_acc"])
    val_irm      = max(irm_configs, key=lambda c: c["mean_id_val_acc"])
    val_erm      = max(erm_configs, key=lambda c: c["mean_id_val_acc"])
    gt_train_val = cfg["gt_irm_train_val"]
    gap_agr_vs_train_val = agr_ood - gt_train_val
    gap_agr_vs_oracle    = agr_ood - cfg["gt_irm_oracle"]

    # Agreement-based: among diverging configs, pick best by ID val acc
    diverging = [c for c in irm_configs if c["n_below"] >= majority]
    agr_irm   = (max(diverging, key=lambda c: c["mean_id_val_acc"])
                 if diverging else val_erm)
    agr_ood   = agr_irm["mean_ood_acc"] if diverging else val_erm["mean_ood_acc"]

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print(f"\n    *** DECISION: {decision_result['decision']} ***")
    print(f"    {decision_result['explanation']}")
    print(f"    Agreement: {decision_result['agreement']['signal']}  "
          f"frac_below={decision_result['agreement']['frac_below']:.2f}  "
          f"mean_dev={decision_result['agreement']['mean_dev']:+.3f}")
    print(f"    Entropy:   ERM={decision_result['entropy']['mean_erm_H']:.3f} "
          f"({decision_result['entropy']['erm_state']})  "
          f"IRM={decision_result['entropy']['mean_irm_H']:.3f} "
          f"({decision_result['entropy']['irm_state']})")

    

    print(f"\n    Selection comparison:")
    print(f"    (// = uses OOD labels — not available in practice)")
    print(f"    {'Strategy':<35} {'OOD acc':>9}  {'vs Exp5 ref':>12}")
    print(f"    {'─'*60}")
    print(f"    {'ERM (ID val) — honest baseline':<35} "
          f"{val_erm['mean_ood_acc']:>9.3f}  "
          f"GT={cfg['gt_erm']:.3f}")
    print(f"    {'IRM (ID val) — standard non-oracle':<35} "
          f"{val_irm['mean_ood_acc']:>9.3f}  "
          f"GT={gt_train_val:.3f}")
    print(f"    {'IRM (agreement) — our contribution':<35} "
          f"{agr_ood:>9.3f}  "
          f"vs train_val={gap_agr_vs_train_val:+.3f}")
    print(f"    {'─'*60}")
    print(f"    {'IRM oracle // (OOD labels — cheat)':<35} "
          f"{oracle_irm['mean_ood_acc']:>9.3f}  "
          f"GT={cfg['gt_irm_oracle']:.3f}")
    print(f"\n    Key gap: agreement vs standard non-oracle: "
          f"{gap_agr_vs_train_val:+.3f}")
    print(f"    Key gap: agreement vs oracle (ceiling):    "
          f"{gap_agr_vs_oracle:+.3f}")

    correct = decision_result["decision"] == cfg["expected"].split("(")[0].strip()
    print(f"\n    Framework correct? "
          f"{'✓ YES' if correct else '✗ NO — expected: ' + cfg['expected']}")

    return {
        "config":          config_name,
        "e_train":         e_values,
        "agl_line":        {"slope": slope, "intercept": intercept, "r2": r2},
        "decision":        decision_result,
        "selection": {
            "erm_oracle":   oracle_erm["mean_ood_acc"],
            "erm_val":      val_erm["mean_ood_acc"],
            "irm_oracle":   oracle_irm["mean_ood_acc"],
            "irm_val":      val_irm["mean_ood_acc"],
            "irm_agr":      agr_ood,
        },
        "ground_truth": {
            "erm":          cfg["gt_erm"],
            "irm_oracle":   cfg["gt_irm_oracle"],
            "irm_nonoracle":cfg["gt_irm_nonoracle"],
        },
        "erm_erm_points":  erm_erm_points,
        "irm_configs_summary": [
            {
                "trial":         t,
                "lambda":        irm_configs[t]["hp"]["penalty_weight"],
                "mean_dev":      irm_configs[t]["mean_deviation"],
                "std_dev":       irm_configs[t]["std_deviation"],
                "n_below":       irm_configs[t]["n_below"],
                "mean_id_val":   irm_configs[t]["mean_id_val_acc"],
                "mean_ood":      irm_configs[t]["mean_ood_acc"],
                "mean_entropy":  irm_configs[t]["mean_entropy"],
                "erm_mean_ood":  erm_configs[t]["mean_ood_acc"],
            }
            for t in range(n_trials)
        ],
        "erm_configs_summary": [
            {
                "trial":        t,
                "mean_id_val":  erm_configs[t]["mean_id_val_acc"],
                "mean_ood":     erm_configs[t]["mean_ood_acc"],
                "mean_entropy": erm_configs[t]["mean_entropy"],
            }
            for t in range(n_trials)
        ],
        "_raw": {
            "erm_configs": erm_configs,
            "irm_configs": irm_configs,
            "slope": slope, "intercept": intercept,
        }
    }


# =============================================================================
# Plotting — summary across all 4 configs
# =============================================================================

def plot_2x2(all_results, output_dir):
    config_names  = list(all_results.keys())
    n             = len(config_names)
    fig, axes     = plt.subplots(2, n, figsize=(5*n, 9))

    decision_colors = {
        "DIVERGE":        "#1D9E75",
        "AGREE":          "#185FA5",
        "BOTH_FAIL":      "#D85A30",
        "IRM_DEGENERATE": "#854F0B",
        "WEAK":           "#7F77DD",
    }

    for col, cname in enumerate(config_names):
        res  = all_results[cname]
        cfg  = CONFIGS[cname]
        raw  = res["_raw"]
        erm_configs = raw["erm_configs"]
        irm_configs = raw["irm_configs"]
        slope, intercept = raw["slope"], raw["intercept"]
        r2   = res["agl_line"]["r2"]
        dec  = res["decision"]["decision"]
        col_c = decision_colors.get(dec, "#888780")
        majority = 2   # n_seeds//2 + 1 for n_seeds=3

        # ── Row 1: AGL scatter ──
        ax = axes[0, col]
        eid  = [p["id_agr"]  for p in res["erm_erm_points"]]
        eood = [p["ood_agr"] for p in res["erm_erm_points"]]
        ax.scatter(eid, eood, color="gray", alpha=0.3, s=18,
                   label=f"ERM-ERM ({len(eid)})", zorder=2)

        xs = np.linspace(max(0.5, min(eid)-0.02),
                         min(1.0, max(eid)+0.02), 100)
        ax.plot(xs, [inv_probit(slope*probit(x)+intercept) for x in xs],
                color="gray", linewidth=1.5, linestyle="--",
                label=f"AGL R²={r2:.2f}", zorder=2)

        for t, ic in enumerate(irm_configs):
            is_div = ic["n_below"] >= majority
            c      = "#D85A30" if is_div else "#185FA5"
            m      = "v"       if is_div else "^"
            for p in ic["cross_points"]:
                ax.scatter(p["id_agr"], p["ood_agr"],
                           color=c, marker=m, s=40, alpha=0.7, zorder=3)

        ax.set_xlabel("ID agreement", fontsize=9)
        ax.set_ylabel("OOD agreement", fontsize=9)
        ax.set_title(
            f"Config {cname}: {cfg['e_train']}\n"
            f"div={cfg['diversity']}  prox={cfg['proximity']}\n"
            f"Decision: {dec}",
            fontsize=9, color=col_c, fontweight="bold"
        )
        ax.legend(handles=[
            Line2D([0],[0], marker="v", color="w", markerfacecolor="#D85A30",
                   markersize=7, label="IRM-ERM below line"),
            Line2D([0],[0], marker="^", color="w", markerfacecolor="#185FA5",
                   markersize=7, label="IRM-ERM on line"),
        ], fontsize=7)
        ax.grid(True, alpha=0.25)

        # ── Row 2: Selection strategy bar ──
        ax2    = axes[1, col]
        sel    = res["selection"]
        gt     = res["ground_truth"]
        labels = ["ERM\noracle", "ERM\nval", "IRM\noracle",
                  "IRM\nval", "IRM\nagr"]
        vals   = [sel["erm_oracle"], sel["erm_val"], sel["irm_oracle"],
                  sel["irm_val"],    sel["irm_agr"]]
        bcols  = ["#aaaaaa", "#555555", "#185FA5", "#1D9E75", "#D85A30"]
        htchs  = ["//", "", "//", "", ""]
        bars   = ax2.bar(labels, vals, color=bcols,
                         hatch=htchs, alpha=0.85, width=0.55)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        # Ground truth reference lines
        ax2.axhline(gt["erm"],          color="#aaaaaa", linewidth=1,
                    linestyle=":", alpha=0.7, label=f"GT ERM={gt['erm']:.2f}")
        ax2.axhline(gt["irm_oracle"],   color="#185FA5", linewidth=1,
                    linestyle=":", alpha=0.7, label=f"GT IRM ora={gt['irm_oracle']:.2f}")
        ax2.axhline(gt["irm_nonoracle"],color="#1D9E75", linewidth=1,
                    linestyle=":", alpha=0.7, label=f"GT IRM non={gt['irm_nonoracle']:.2f}")
        ax2.set_ylabel("OOD accuracy", fontsize=9)
        ax2.set_title(f"Selection comparison\n(// = needs OOD labels)", fontsize=9)
        ax2.set_ylim(0, min(1.0, max(vals + [gt["irm_oracle"]]) + 0.12))
        ax2.legend(fontsize=7, loc="lower right")
        ax2.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        "AGL Diagnostic — 2×2 Proximity × Diversity Matrix\n"
        "Colored MNIST  |  e_test=0.9  |  "
        "▽ = IRM-ERM below line   △ = IRM-ERM on line   // = requires OOD labels",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "agl_2x2_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {path}")
    return path


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AGL diagnostic on the 2x2 proximity/diversity matrix"
    )
    parser.add_argument("--n_trials",   type=int,  default=20,
                        help="Random HP configs per config (like n_trials in search.py)")
    parser.add_argument("--n_seeds",    type=int,  default=3,
                        help="Seeds per HP config")
    parser.add_argument("--device",     type=str,  default="cpu")
    parser.add_argument("--output_dir", type=str,  default="results")
    parser.add_argument("--configs",    type=str,  default="ABCD",
                        help="Which configs to run, e.g. 'AC' or 'ABCD'")
    parser.add_argument("--max_steps", type=int, default=None,
                    help="Cap training steps for quick sanity checks. "
                         "None = use sampled value. 101 = fast local run.")
    args = parser.parse_args()
    

    os.makedirs(args.output_dir, exist_ok=True)
    mnist_raw    = load_mnist_raw()
    configs_todo = [c for c in "ABCD" if c in args.configs.upper()]

    print(f"\n{'='*65}")
    print(f"  AGL 2x2 Experiment")
    print(f"  Configs: {configs_todo}  n_trials={args.n_trials}  "
          f"n_seeds={args.n_seeds}  device={args.device}")
    print(f"  Total models per config: "
          f"{args.n_trials * args.n_seeds * 2} "
          f"({args.n_trials} trials × {args.n_seeds} seeds × 2 methods)")
    print(f"{'='*65}")

    all_results = {}
    for cname in configs_todo:
        print(f"\n{'─'*65}")
        print(f"  Running Config {cname}")
        print(f"{'─'*65}")
        result = run_config(
            cname, CONFIGS[cname],
            args.n_trials, args.n_seeds,
            args.device, args.output_dir, mnist_raw, max_steps=args.max_steps
        )
        # Remove raw data before storing for JSON serialization
        result_clean = {k: v for k, v in result.items() if k != "_raw"}
        all_results[cname] = result
        all_results[cname + "_clean"] = result_clean

    # Summary table
    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Config':<8} {'Expected':<35} {'Decision':<20} {'Correct':>8}")
    print(f"  {'─'*75}")
    for cname in configs_todo:
        res     = all_results[cname]
        dec     = res["decision"]["decision"]
        exp     = CONFIGS[cname]["expected"]
        correct = dec.startswith(exp.split()[0])
        print(f"  {cname:<8} {exp:<35} {dec:<20} {'✓' if correct else '✗':>8}")

    # Plot
    plot_results = {k: all_results[k] for k in configs_todo}
    plot_2x2(plot_results, args.output_dir)

    # Save JSON
    json_out = {}
    for cname in configs_todo:
        json_out[cname] = all_results[cname + "_clean"]
    json_path = os.path.join(args.output_dir, "agl_2x2_results.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  JSON  saved → {json_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
