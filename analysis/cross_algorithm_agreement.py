"""
Cross-algorithm agreement diagnostic + label-free model selection.

Two lines per DG algorithm (detection):
    Cross-R line     : Pearson R in probit space, all ERM x DG pairs (population-level).
    Seed-level line  : Pearson R in probit space, one point per DG seed
                       (mean ID agr and mean OOD agr with full ERM pool).
    Both lines use the same probit-space fit. If they agree in R and slope,
    the seed-level aggregation preserves the population signal, validating
    CrA as a cheaper approximation of Cross-R that requires no ID predictions.

Usage:
    python analysis/cross_algorithm_agreement.py \
        --preds_dir    results/coloredmnist/test_env2/cnn/random/models \
        --records_path results/coloredmnist/test_env2/cnn/random/records.json \
        --test_env_idx 2 --n_envs 3 --algo_a ERM \
        --algo_b IRM VREx GroupDRO CORAL DANN \
        --dataset_name ColoredMNIST --test_env_name "-90% (env2)" \
        --multi_plot_path cmnist_multi.png \
        --select --evaluate_selection
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri as probit
from scipy.special import ndtr as normal_cdf
from scipy.stats import pearsonr
from itertools import combinations

from utils import (
    load_probs, load_predictions, get_all_trials, get_record_info,
    compute_agreement, compute_entropy, get_id_agr, get_ood_agr,
    get_valid_seeds, fit_line,
)
from distance_metrics import compute_mmd, compute_wasserstein, compute_pad


# ---------------------------------------------------------------------------
# Input-space shift (dataset-level, model- and seed-independent)
# ---------------------------------------------------------------------------

def _flatten_env_split(env, max_n=2000, seed=42):
    """
    Extract and flatten one environment's out-split inputs to (N, D) float32.
    Subsamples BEFORE materializing so image datasets don't load/transform
    examples that will just be discarded. Handles both tensor envs
    ({'images': ...}, e.g. ColoredMNIST/ACSIncome) and Dataset/Subset envs
    (e.g. RotatedMNIST/PACS).
    """
    if isinstance(env, dict) and 'images' in env:
        x = env['images']
        n = len(x)
        if n > max_n:
            idx = np.random.default_rng(seed).choice(n, max_n, replace=False)
            x = x[idx]
        return x.reshape(len(x), -1).numpy().astype(np.float32)
    else:
        n = len(env)
        idx = (np.arange(n) if n <= max_n
               else np.random.default_rng(seed).choice(n, max_n, replace=False))
        rows = [np.asarray(env[int(i)][0]).reshape(-1) for i in idx]
        return np.stack(rows).astype(np.float32)


def compute_input_space_shift(dataset_key, data_dir, test_env_idx, n_envs,
                               backbone='resnet50', run_mmd=False,
                               run_wass=False, run_pad=False):
    """
    MMD / Wasserstein / PAD between pooled ID envs' raw inputs and the OOD
    (test) env's raw inputs -- computed directly on the data itself, never
    touching model predictions. This is a property of the dataset/split,
    not of any algorithm or seed, so it's computed once per dataset rather
    than once per (algo, seed) the way the old probability-space "shift"
    metric was.
    """
    if not (run_mmd or run_wass or run_pad):
        return None
    if dataset_key is None or data_dir is None:
        raise ValueError(
            "--distance_metrics requires --input_dataset and --input_data_dir "
            "to load the raw inputs for the input-space shift computation."
        )

    import sys
    _repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    sys.path.insert(0, _repo_root)
    sys.path.insert(0, os.path.join(_repo_root, 'DomainBed'))
    from src.datasets import get_dataset

    envs_splits = get_dataset(dataset_key, data_dir=data_dir,
                              test_env_idx=test_env_idx, backbone=backbone)

    id_parts = []
    for i in range(n_envs):
        if i == test_env_idx:
            continue
        _, out_env = envs_splits[i]
        id_parts.append(_flatten_env_split(out_env))
    id_pool  = np.vstack(id_parts)
    _, test_out_env = envs_splits[test_env_idx]
    ood_pool = _flatten_env_split(test_out_env)

    return {
        'mmd':         compute_mmd(id_pool, ood_pool)         if run_mmd  else None,
        'wasserstein': compute_wasserstein(id_pool, ood_pool) if run_wass else None,
        'pad':         compute_pad(id_pool, ood_pool)         if run_pad  else None,
        'id_n':  len(id_pool),
        'ood_n': len(ood_pool),
        'dim':   id_pool.shape[1],
    }


def get_entropy_mean_std(preds_dir, algo, seed, n_trials, test_env_idx):
    probs_list = get_all_trials(preds_dir, algo, seed, n_trials,
                                test_env_idx, load_probs)
    if not probs_list:
        return None, None
    per_trial_h = [compute_entropy(p) for p in probs_list]
    return float(np.mean(per_trial_h)), float(np.std(per_trial_h))


# ---------------------------------------------------------------------------
# Agreement lines (probit-space linear fit)
# ---------------------------------------------------------------------------

def compute_erm_line(preds_dir, valid_seeds, n_trials, test_env_idx, n_envs):
    id_agrs, ood_agrs = [], []
    for seed_i, seed_j in combinations(valid_seeds, 2):
        id_agr  = get_id_agr(preds_dir, 'ERM', 'ERM', seed_i, seed_j,
                             n_trials, test_env_idx, n_envs)
        ood_agr = get_ood_agr(preds_dir, 'ERM', 'ERM', seed_i, seed_j,
                              n_trials, test_env_idx)
        if id_agr is not None and ood_agr is not None:
            id_agrs.append(id_agr)
            ood_agrs.append(ood_agr)
    line = fit_line(id_agrs, ood_agrs)
    line['n_seeds'] = len(valid_seeds)
    return line


def compute_cross_line(preds_dir, valid_a, valid_b, algo_a, algo_b,
                       n_trials, test_env_idx, n_envs):
    """
    Cross-R: Pearson R of line through all ERM x DG pairs in probit space.
    One point per (ERM seed i, DG seed j) -- population-level.
    len(valid_a) * len(valid_b) points total.
    """
    id_agrs, ood_agrs = [], []
    for seed_i in valid_a:
        for seed_j in valid_b:
            id_agr  = get_id_agr(preds_dir, algo_a, algo_b, seed_i, seed_j,
                                 n_trials, test_env_idx, n_envs)
            ood_agr = get_ood_agr(preds_dir, algo_a, algo_b, seed_i, seed_j,
                                  n_trials, test_env_idx)
            if id_agr is not None and ood_agr is not None:
                id_agrs.append(id_agr)
                ood_agrs.append(ood_agr)
    if len(id_agrs) < 2:
        return None
    return fit_line(id_agrs, ood_agrs)


def compute_seed_level_line(preds_dir, valid_a, valid_b, algo_a, algo_b,
                             n_trials, test_env_idx, n_envs):
    """
    Seed-level aggregated line: one point per algo_b seed.

    For each DG seed j:
        x_j = mean ID  agreement of seed j with ALL ERM seeds in valid_a
        y_j = mean OOD agreement of seed j with ALL ERM seeds in valid_a

    This gives len(valid_b) points instead of len(valid_a)*len(valid_b) pairs.
    Fit in probit space -- same as fit_line() -- so R is directly comparable
    to Cross-R.

    Interpretation:
        If seed-level R ~ Cross-R  -> aggregating to seed level preserves the
                                       population signal. CrA (Y axis only) is
                                       a valid cheap approximation of Cross-R.
        If they diverge            -> pair-level structure carries information
                                       the seed mean loses.

    Note: y_j (mean OOD agr with ERM pool) is exactly the CrA score of seed j.
          So the Y axis of this line IS CrA. The X axis adds mean ID agreement,
          making this the seed-level analog of the Cross-R scatter.

    Returned dict has same keys as compute_cross_line() plus:
        seed_ids   : algo_b seed index for each point
        x_per_seed : mean ID  agr per seed (raw, before probit)
        y_per_seed : mean OOD agr per seed (raw, before probit) == CrA scores
    """
    x_per_seed = []
    y_per_seed = []
    seed_ids   = []

    for seed_b in valid_b:
        id_agrs_b  = []
        ood_agrs_b = []
        for seed_a in valid_a:
            id_agr  = get_id_agr(preds_dir, algo_a, algo_b, seed_a, seed_b,
                                  n_trials, test_env_idx, n_envs)
            ood_agr = get_ood_agr(preds_dir, algo_a, algo_b, seed_a, seed_b,
                                   n_trials, test_env_idx)
            if id_agr is not None and ood_agr is not None:
                id_agrs_b.append(id_agr)
                ood_agrs_b.append(ood_agr)
        if not id_agrs_b:
            continue
        x_per_seed.append(float(np.mean(id_agrs_b)))
        y_per_seed.append(float(np.mean(ood_agrs_b)))
        seed_ids.append(seed_b)

    if len(x_per_seed) < 2:
        return None

    line = fit_line(x_per_seed, y_per_seed)
    line['seed_ids']   = seed_ids
    line['x_per_seed'] = x_per_seed
    line['y_per_seed'] = y_per_seed
    return line


# ---------------------------------------------------------------------------
# algo_b-algo_b agreement among candidate / on-line seeds
# ---------------------------------------------------------------------------

def compute_candidate_agreement(preds_dir, candidate_seeds, algo_b,
                                n_trials, test_env_idx):
    if len(candidate_seeds) < 2:
        return None
    agr_vals = []
    for seed_i, seed_j in combinations(candidate_seeds, 2):
        ood_agr = get_ood_agr(preds_dir, algo_b, algo_b, seed_i, seed_j,
                              n_trials, test_env_idx)
        if ood_agr is not None:
            agr_vals.append(ood_agr)
    return float(np.mean(agr_vals)) if agr_vals else None


# ---------------------------------------------------------------------------
# Main diagnostic computation
# ---------------------------------------------------------------------------

def compute_cross_algorithm_agreement(
    preds_dir,
    records_path,
    test_env_idx,
    n_envs,
    n_hparams,
    n_trials=3,
    algo_a='ERM',
    algo_b='IRM',
    entropy_threshold=0.9,
    cross_line_r_threshold=0.3,
    disagreement_rate_threshold=0.5,
    distance_metrics=None,
    irm_anneal_threshold=None,
    input_dataset=None,
    input_data_dir=None,
    input_backbone='resnet50',
    precomputed_input_shift=None,
):
    with open(records_path) as f:
        records = json.load(f)

    n_classes = 2
    for seed in range(n_hparams):
        for trial in range(n_trials):
            probs = load_probs(preds_dir, algo_a, seed, trial, test_env_idx)
            if probs is not None:
                n_classes = probs.shape[1]
                break
        else:
            continue
        break
    max_entropy = float(np.log(n_classes))

    print(f"\n  n_classes={n_classes}  max_entropy={max_entropy:.4f}")
    print(f"  entropy_threshold={entropy_threshold}")
    distance_metrics = [m.lower() for m in (distance_metrics or [])]
    run_mmd  = 'mmd'         in distance_metrics
    run_wass = 'wasserstein' in distance_metrics
    run_pad  = 'pad'         in distance_metrics
    run_any  = run_mmd or run_wass or run_pad

    if run_any:
        print(f"  distance metrics: {', '.join(distance_metrics)}")
    else:
        print(f"  distance metrics: OFF")

    # Input-space shift: dataset-level, computed once (not per seed/algo) --
    # see compute_input_space_shift's docstring for why this replaced the
    # old per-seed probability-space "shift" metric.
    input_shift = None
    if run_any:
        # Dataset-level and identical for every algo_b, so callers that loop
        # over multiple algo_b values in one run should compute this once
        # and pass it in via precomputed_input_shift instead of reloading
        # the raw dataset on every call.
        input_shift = precomputed_input_shift or compute_input_space_shift(
            input_dataset, input_data_dir, test_env_idx, n_envs,
            backbone=input_backbone,
            run_mmd=run_mmd, run_wass=run_wass, run_pad=run_pad)
        print(f"  input-space shift (ID pool vs OOD, n_id={input_shift['id_n']} "
              f"n_ood={input_shift['ood_n']} dim={input_shift['dim']}):")
        if run_mmd:
            print(f"    mmd_shift         = {input_shift['mmd']:.5f}")
        if run_wass:
            print(f"    wasserstein_shift = {input_shift['wasserstein']:.5f}")
        if run_pad:
            print(f"    pad_shift         = {input_shift['pad']:.5f}")

    # ---- Step 0: symmetric entropy filter ----
    valid_a, excluded_a = get_valid_seeds(preds_dir, algo_a, n_hparams, n_trials,
                                          test_env_idx, max_entropy, entropy_threshold)
    valid_b, excluded_b = get_valid_seeds(preds_dir, algo_b, n_hparams, n_trials,
                                          test_env_idx, max_entropy, entropy_threshold)
    valid_seeds = sorted(set(valid_a) & set(valid_b))

    print(f"\n  Valid {algo_a} seeds: {valid_a}")
    if excluded_a:
        print(f"  Excluded {algo_a} (rel_H>={entropy_threshold}): " +
              ', '.join(f"{s}({h:.3f})" for s, h in excluded_a))
    print(f"  Valid {algo_b} seeds: {valid_b}")
    if excluded_b:
        print(f"  Excluded {algo_b} (rel_H>={entropy_threshold}): " +
              ', '.join(f"{s}({h:.3f})" for s, h in excluded_b))
    print(f"  Valid seeds (both, {len(valid_seeds)}/{n_hparams}): {valid_seeds}")

    # ---- Step 1a: ERM-ERM reference line ----
    print(f"\n  Computing {algo_a}-{algo_a} reference line...")
    erm_line = compute_erm_line(preds_dir, valid_a, n_trials, test_env_idx, n_envs)
    erm_ood_median = erm_line['ood_median']
    print(f"  {algo_a}-{algo_a}: R={erm_line['R']:+.3f}  "
          f"slope={erm_line['slope']:.3f}  intercept={erm_line['intercept']:.3f}  "
          f"ood_median={erm_ood_median:.3f}  n_pairs={erm_line['n_pairs']}")

    # ---- Step 1b: Cross-R pair-level line ----
    print(f"\n  Computing {algo_a}-{algo_b} Cross-R (pair-level, {len(valid_a)}x{len(valid_b)} pairs)...")
    valid_seeds_cross = valid_seeds
    if irm_anneal_threshold is not None:
        valid_seeds_cross = [
            s for s in valid_seeds
            if (get_record_info(records, algo_b, s, test_env_idx) or {}).get('anneal') is None
            or (get_record_info(records, algo_b, s, test_env_idx) or {}).get('anneal') < irm_anneal_threshold
        ]
        n_filtered = len(valid_seeds) - len(valid_seeds_cross)
        if n_filtered:
            print(f"  Excluded {n_filtered} high-anneal seeds: "
                  f"{[s for s in valid_seeds if s not in valid_seeds_cross]}")

    cross_line = compute_cross_line(preds_dir, valid_seeds_cross, valid_seeds_cross,
                                    algo_a, algo_b, n_trials, test_env_idx, n_envs)
    if cross_line is None:
        print(f"  {algo_a}-{algo_b}: insufficient valid seeds (n<2)")
        cross_line_misspecified = None
    else:
        print(f"  {algo_a}-{algo_b} Cross-R:    R={cross_line['R']:+.3f}  "
              f"slope={cross_line['slope']:.3f}  "
              f"n_pairs={cross_line['n_pairs']}")
        cross_line_misspecified = cross_line['R'] >= cross_line_r_threshold
        cross_symbol = ('misspecified (shared trend, ERM sufficient)'
                        if cross_line_misspecified
                        else 'well-specified (separate trend, DG may help)')
        print(f"  Cross-R >= {cross_line_r_threshold}? {cross_line_misspecified} -> {cross_symbol}")

    # ---- Step 1c: Seed-level aggregated line ----
    print(f"\n  Computing {algo_a}-{algo_b} seed-level line ({len(valid_seeds_cross)} points)...")
    seed_level_line = compute_seed_level_line(
        preds_dir, valid_seeds_cross, valid_seeds_cross,
        algo_a, algo_b, n_trials, test_env_idx, n_envs)

    if seed_level_line is None:
        print(f"  {algo_a}-{algo_b} seed-level: insufficient seeds (n<2)")
    else:
        print(f"  {algo_a}-{algo_b} seed-level: R={seed_level_line['R']:+.3f}  "
              f"slope={seed_level_line['slope']:.3f}  "
              f"n_seeds={len(seed_level_line['seed_ids'])}")
        if cross_line is not None:
            same_sign = (cross_line['R'] >= 0) == (seed_level_line['R'] >= 0)
            print(f"  Sign agreement Cross-R vs seed-level: {same_sign}  "
                  f"(Cross-R={cross_line['R']:+.3f}, seed-level={seed_level_line['R']:+.3f})")

    # ---- Step 2: same-seed pairs ----
    print(f"\n  Computing {algo_a}-{algo_b} same-seed pairs (Step 2)...")

    def _ms(v, s, width=14, p=3):
        if v is None:
            return f"{'x':>{width}}"
        s_str = f"{s:.{p}f}" if s is not None else "?"
        return f"{v:.{p}f}+-{s_str}"[:width].rjust(width)

    base_header = (f"  {'seed':>4} | "
                   f"{'ID_agr':>14} | {'OOD_agr':>14} | {'pred_OOD':>8} | {'|dev|':>6} | "
                   f"{'rel_h_b':>7} | {'dH/max':>7} | "
                   f"{'acc_'+algo_a:>14} | {'acc_'+algo_b:>14} | "
                   f"{'H('+algo_a+')':>14} | {'H('+algo_b+')':>14}")
    dist_header = ""
    if run_mmd:
        dist_header += f" | {'mmd_x':>7}"
    if run_wass:
        dist_header += f" | {'wss_x':>7}"
    if run_pad:
        dist_header += f" | {'pad_x':>7}"
    print(base_header + dist_header + f" | {'lambda':>10} | {'anneal':>6} | status")
    print(f"  {'-'*180}")

    id_agrs, ood_agrs, deviations = [], [], []
    id_agr_stds, ood_agr_stds = [], []
    id_agr_all_trials, ood_agr_all_trials = [], []
    rel_dhs, rel_h_bs = [], []
    entropies_a, entropies_b = [], []
    entropies_a_std, entropies_b_std = [], []
    mmd_crosses, wass_crosses, pad_crosses = [], [], []
    seeds_used, step2_statuses = [], []
    acc_bs_all = []
    eps = 1e-6

    for seed in valid_seeds:
        id_agr_trials  = get_id_agr(preds_dir, algo_a, algo_b, seed, seed,
                                    n_trials, test_env_idx, n_envs, return_all=True)
        ood_agr_trials = get_ood_agr(preds_dir, algo_a, algo_b, seed, seed,
                                     n_trials, test_env_idx, return_all=True)
        if id_agr_trials is None or ood_agr_trials is None:
            continue

        id_agr      = float(np.mean(id_agr_trials))
        id_agr_std  = float(np.std(id_agr_trials))
        ood_agr     = float(np.mean(ood_agr_trials))
        ood_agr_std = float(np.std(ood_agr_trials))

        id_probit_val   = probit(np.clip(id_agr, eps, 1 - eps))
        pred_ood_probit = erm_line['slope'] * id_probit_val + erm_line['intercept']
        pred_ood_agr    = float(normal_cdf(pred_ood_probit))
        deviation       = abs(ood_agr - pred_ood_agr)

        entropy_a, entropy_a_std = get_entropy_mean_std(
            preds_dir, algo_a, seed, n_trials, test_env_idx)
        entropy_b, entropy_b_std = get_entropy_mean_std(
            preds_dir, algo_b, seed, n_trials, test_env_idx)
        rel_h_b = entropy_b / max_entropy if entropy_b is not None else None
        rel_dH  = ((entropy_b - entropy_a) / max_entropy
                   if entropy_b is not None and entropy_a is not None else None)

        mmd_c = wass_c = pad_c = None
        if run_any:
            probs_a_dist = get_all_trials(preds_dir, algo_a, seed, n_trials,
                                          test_env_idx, load_probs)
            probs_b_dist = get_all_trials(preds_dir, algo_b, seed, n_trials,
                                          test_env_idx, load_probs)
            if probs_a_dist and probs_b_dist:
                cross_pairs = [(pa, pb) for pa in probs_a_dist for pb in probs_b_dist]
                if run_mmd:
                    mmd_c = float(np.mean([compute_mmd(pa, pb) for pa, pb in cross_pairs]))
                if run_wass:
                    wass_c = float(np.mean([compute_wasserstein(pa, pb) for pa, pb in cross_pairs]))
                if run_pad:
                    pad_c = float(np.mean([compute_pad(pa, pb) for pa, pb in cross_pairs]))

        step2 = 'candidate' if ood_agr < erm_ood_median else 'on line'

        info_a = get_record_info(records, algo_a, seed, test_env_idx)
        info_b = get_record_info(records, algo_b, seed, test_env_idx)
        acc_a     = info_a['ood_acc']     if info_a else None
        acc_a_std = info_a['ood_acc_std'] if info_a else None
        acc_b     = info_b['ood_acc']     if info_b else None
        acc_b_std = info_b['ood_acc_std'] if info_b else None
        lambda_b  = info_b['lambda']      if info_b else None
        anneal_b  = info_b['anneal']      if info_b else None

        id_agrs.append(id_agr);          ood_agrs.append(ood_agr)
        id_agr_stds.append(id_agr_std);  ood_agr_stds.append(ood_agr_std)
        id_agr_all_trials.append(id_agr_trials)
        ood_agr_all_trials.append(ood_agr_trials)
        deviations.append(deviation)
        rel_dhs.append(rel_dH);   rel_h_bs.append(rel_h_b)
        entropies_a.append(entropy_a);      entropies_b.append(entropy_b)
        entropies_a_std.append(entropy_a_std); entropies_b_std.append(entropy_b_std)
        mmd_crosses.append(mmd_c)
        wass_crosses.append(wass_c)
        pad_crosses.append(pad_c)
        seeds_used.append(seed);    step2_statuses.append(step2)
        acc_bs_all.append(acc_b)

        def _f(v, w=7, p=5):
            return f"{v:{w}.{p}f}" if v is not None else f"{'x':>{w}}"

        lambda_str = f"{lambda_b:10.1f}" if lambda_b is not None else f"{'x':>10}"
        anneal_str = f"{anneal_b:6d}"    if anneal_b is not None else f"{'x':>6}"
        base = (f"  {seed:>4} | "
                f"{_ms(id_agr, id_agr_std)} | "
                f"{_ms(ood_agr, ood_agr_std)} | "
                f"{pred_ood_agr:>8.3f} | {deviation:>6.3f} | "
                f"{_f(rel_h_b)} | {_f(rel_dH, p=3):>7} | "
                f"{_ms(acc_a, acc_a_std)} | {_ms(acc_b, acc_b_std)} | "
                f"{_ms(entropy_a, entropy_a_std, p=4)} | {_ms(entropy_b, entropy_b_std, p=4)}")
        dist_cols = ""
        if run_mmd:   dist_cols += f" | {_f(mmd_c)}"
        if run_wass:  dist_cols += f" | {_f(wass_c)}"
        if run_pad:   dist_cols += f" | {_f(pad_c)}"
        print(base + dist_cols + f" | {lambda_str} | {anneal_str} | {step2}")

    # ---- Step 3 ----
    candidate_seeds = [seeds_used[i] for i, s in enumerate(step2_statuses) if s == 'candidate']
    on_line_seeds   = [seeds_used[i] for i, s in enumerate(step2_statuses) if s == 'on line']

    print(f"\n  Candidates: {candidate_seeds}")
    print(f"  On line:    {on_line_seeds}")

    candidate_agreement = compute_candidate_agreement(
        preds_dir, candidate_seeds, algo_b, n_trials, test_env_idx)
    online_agreement = compute_candidate_agreement(
        preds_dir, on_line_seeds, algo_b, n_trials, test_env_idx)

    print(f"\n  {algo_b}-{algo_b} OOD agr (candidates): " +
          (f"{candidate_agreement:.3f}" if candidate_agreement is not None else "N/A"))
    print(f"  {algo_b}-{algo_b} OOD agr (on line):    " +
          (f"{online_agreement:.3f}"    if online_agreement   is not None else "N/A"))
    print(f"  {algo_a}-{algo_a} OOD median (ref):      {erm_ood_median:.3f}")

    n_candidates = len(candidate_seeds)
    n_on_line    = len(on_line_seeds)

    if n_candidates < 2:
        dg_verdict        = 'inconclusive'
        genuine_escape    = None
        disagreement_rate = None
    else:
        disagreement_rate = candidate_agreement / erm_ood_median
        genuine_escape    = disagreement_rate > disagreement_rate_threshold
        dg_verdict        = ('reproducible divergence' if genuine_escape
                             else 'divergence looks like noise')

    if run_any:
        print(f"\n  Distance metrics summary:")
        names_and_lists = []
        if run_mmd:   names_and_lists.append(('mmd',         mmd_crosses))
        if run_wass:  names_and_lists.append(('wasserstein', wass_crosses))
        if run_pad:   names_and_lists.append(('pad',         pad_crosses))
        for name, cv_raw in names_and_lists:
            cp = [(v, a) for v, a in zip(cv_raw, acc_bs_all) if v is not None and a is not None]
            if cp:
                cv = [v for v, _ in cp]; ca = [a for _, a in cp]
                r_c = pearsonr(cv, ca)[0] if len(cv) >= 3 else float('nan')
                print(f"  {name}_cross: mean={np.mean(cv):.5f}  r(acc)={r_c:+.3f}")
        # input-space shift is a single dataset-level number (see above),
        # not a per-seed series, so no r(acc) correlation applies to it

    return {
        'erm_line':                erm_line,
        'cross_line':              cross_line,
        'seed_level_line':         seed_level_line,
        'cross_line_misspecified': cross_line_misspecified,
        'id_agrs':                 id_agrs,
        'id_agr_stds':             id_agr_stds,
        'id_agr_all_trials':       id_agr_all_trials,
        'ood_agrs':                ood_agrs,
        'ood_agr_stds':            ood_agr_stds,
        'ood_agr_all_trials':      ood_agr_all_trials,
        'deviations':              deviations,
        'rel_dhs':                 rel_dhs,
        'rel_h_bs':                rel_h_bs,
        'entropies_a':             entropies_a,
        'entropies_a_std':         entropies_a_std,
        'entropies_b':             entropies_b,
        'entropies_b_std':         entropies_b_std,
        'mmd_crosses':             mmd_crosses,
        'wass_crosses':            wass_crosses,
        'pad_crosses':             pad_crosses,
        'input_shift':             input_shift,
        'seeds_used':              seeds_used,
        'step2_statuses':          step2_statuses,
        'candidate_seeds':         candidate_seeds,
        'on_line_seeds':           on_line_seeds,
        'candidate_agreement':     candidate_agreement,
        'online_agreement':        online_agreement,
        'genuine_escape':          genuine_escape,
        'dg_verdict':              dg_verdict,
        'n_candidates':            n_candidates,
        'n_on_line':               n_on_line,
        'disagreement_rate':       disagreement_rate,
        'n_pairs':                 len(id_agrs),
        'n_valid_seeds':           len(valid_seeds),
        'valid_a':                 valid_a,
        'valid_b':                 valid_b,
        'algo_a':                  algo_a,
        'algo_b':                  algo_b,
        'max_entropy':             max_entropy,
        'n_classes':               n_classes,
        'test_env_idx':            test_env_idx,
        'n_envs':                  n_envs,
        'cross_line_r_threshold':       cross_line_r_threshold,
        'disagreement_rate_threshold':  disagreement_rate_threshold,
    }


# ---------------------------------------------------------------------------
# Per-algorithm model selection
# ---------------------------------------------------------------------------

def evaluate_per_algorithm(results_list, records_path, test_env_idx, algo_a, n_hparams):
    with open(records_path) as f:
        records = json.load(f)
    rows = []

    erm_oracle_cands, erm_iid_cands = [], []
    for s in range(n_hparams):
        info = get_record_info(records, algo_a, s, test_env_idx)
        if not info:
            continue
        erm_oracle_cands.append((info['ood_acc'], s))
        matching = [r for r in records
                    if r['algorithm'] == algo_a and r['args']['hparams_seed'] == s]
        if not matching:
            continue
        train_accs = [matching[0][k] for k in matching[0]
                      if k.endswith('_out_acc') and k != f'env{test_env_idx}_out_acc']
        if train_accs:
            erm_iid_cands.append((float(np.mean(train_accs)), info['ood_acc'], s))

    erm_oracle = max(erm_oracle_cands, key=lambda t: t[0]) if erm_oracle_cands else None
    erm_iid    = max(erm_iid_cands,    key=lambda t: t[0]) if erm_iid_cands    else None
    rows.append({
        'algorithm': algo_a, 'benchmark': 'x',
        'our_seed': None, 'our_acc': None, 'our_pool': 'N/A',
        'iid_seed':    erm_iid[2]    if erm_iid    else None,
        'iid_acc':     erm_iid[1]    if erm_iid    else None,
        'oracle_seed': erm_oracle[1] if erm_oracle else None,
        'oracle_acc':  erm_oracle[0] if erm_oracle else None,
    })

    for res in results_list:
        algo_b             = res['algo_b']
        seeds_used         = res['seeds_used']
        rel_h_bs           = res['rel_h_bs']
        step2_statuses     = res['step2_statuses']
        cross_line_misspec = res['cross_line_misspecified']

        if cross_line_misspec:
            pool_label = 'on-line'
            pool = [(rel_h_bs[i], seeds_used[i]) for i, s in enumerate(step2_statuses)
                    if s == 'on line' and rel_h_bs[i] is not None]
        else:
            pool_label = 'candidates'
            pool = [(rel_h_bs[i], seeds_used[i]) for i, s in enumerate(step2_statuses)
                    if s == 'candidate' and rel_h_bs[i] is not None]

        our_pick = min(pool, key=lambda t: t[0]) if pool else None
        our_seed = our_pick[1] if our_pick else None
        our_info = get_record_info(records, algo_b, our_seed, test_env_idx) if our_seed is not None else None
        our_acc  = our_info['ood_acc'] if our_info else None

        oracle_cands, iid_cands = [], []
        for s in res['valid_b']:
            info = get_record_info(records, algo_b, s, test_env_idx)
            if not info:
                continue
            oracle_cands.append((info['ood_acc'], s))
            matching = [r for r in records
                        if r['algorithm'] == algo_b and r['args']['hparams_seed'] == s]
            if not matching:
                continue
            train_accs = [matching[0][k] for k in matching[0]
                          if k.endswith('_out_acc') and k != f'env{test_env_idx}_out_acc']
            if train_accs:
                iid_cands.append((float(np.mean(train_accs)), info['ood_acc'], s))

        oracle = max(oracle_cands, key=lambda t: t[0]) if oracle_cands else None
        iid    = max(iid_cands,    key=lambda t: t[0]) if iid_cands    else None
        rows.append({
            'algorithm': algo_b,
            'benchmark': 'mis' if cross_line_misspec else 'well',
            'our_seed': our_seed, 'our_acc': our_acc, 'our_pool': pool_label,
            'iid_seed':    iid[2]    if iid    else None,
            'iid_acc':     iid[1]    if iid    else None,
            'oracle_seed': oracle[1] if oracle else None,
            'oracle_acc':  oracle[0] if oracle else None,
        })
    return rows


def print_per_algorithm_selection(rows, dataset_name, test_env_name):
    print(f"\n{'='*100}")
    print(f"  Per-algorithm model selection -- {dataset_name} (test: {test_env_name})")
    print(f"{'='*100}")
    print(f"  {'Algo':<10} {'type':>6} {'pool':>12} | "
          f"{'our seed':>8} {'our acc':>8} | "
          f"{'iid seed':>8} {'iid acc':>8} | "
          f"{'ora seed':>8} {'ora acc':>8} | "
          f"{'d(our-ora)':>10} {'d(iid-ora)':>10} | {'our>iid?':>8}")
    print(f"  {'x'*98}")
    for row in rows:
        our_str = (f"{row['our_seed']:>8d} {row['our_acc']:>8.3f}"
                   if row['our_acc'] is not None else f"{'N/A':>8} {'N/A':>8}")
        iid_str = (f"{row['iid_seed']:>8d} {row['iid_acc']:>8.3f}"
                   if row['iid_acc'] is not None else f"{'N/A':>8} {'N/A':>8}")
        ora_str = (f"{row['oracle_seed']:>8d} {row['oracle_acc']:>8.3f}"
                   if row['oracle_acc'] is not None else f"{'N/A':>8} {'N/A':>8}")
        gap_our = (f"{row['our_acc'] - row['oracle_acc']:>+10.3f}"
                   if row['our_acc'] is not None and row['oracle_acc'] is not None
                   else f"{'N/A':>10}")
        gap_iid = (f"{row['iid_acc'] - row['oracle_acc']:>+10.3f}"
                   if row['iid_acc'] is not None and row['oracle_acc'] is not None
                   else f"{'N/A':>10}")
        beats = ('Y' if row['our_acc'] is not None and row['iid_acc'] is not None
                  and row['our_acc'] > row['iid_acc'] else 'N') \
                if row['our_acc'] is not None else 'x'
        print(f"  {row['algorithm']:<10} {row['benchmark']:>6} {row['our_pool']:>12} | "
              f"{our_str} | {iid_str} | {ora_str} | {gap_our} {gap_iid} | {beats:>8}")


# ---------------------------------------------------------------------------
# Selection strategy ablation
# ---------------------------------------------------------------------------

def compare_selection_strategies(results_list, records_path, test_env_idx, algo_a, n_hparams):
    with open(records_path) as f:
        records = json.load(f)
    rows = []
    for res in results_list:
        algo_b             = res['algo_b']
        seeds_used         = res['seeds_used']
        rel_h_bs           = res['rel_h_bs']
        rel_dhs            = res['rel_dhs']
        deviations         = res['deviations']
        mmd_crosses        = res['mmd_crosses']
        step2_statuses     = res['step2_statuses']
        cross_line_misspec = res['cross_line_misspecified']

        oracle_cands = []
        for s in res['valid_b']:
            info = get_record_info(records, algo_b, s, test_env_idx)
            if info:
                oracle_cands.append((info['ood_acc'], s))
        oracle = max(oracle_cands, key=lambda t: t[0]) if oracle_cands else None

        if cross_line_misspec:
            pool_idx   = [i for i, s in enumerate(step2_statuses) if s == 'on line']
            pool_label = 'on-line'
        else:
            pool_idx   = [i for i, s in enumerate(step2_statuses) if s == 'candidate']
            pool_label = 'candidates'
        if not pool_idx:
            continue

        def _pick(signal_vals, higher_is_better):
            pairs = [(signal_vals[i], seeds_used[i]) for i in pool_idx
                     if signal_vals[i] is not None]
            if not pairs:
                return None, None
            chosen = max(pairs, key=lambda t: t[0]) if higher_is_better \
                     else min(pairs, key=lambda t: t[0])
            info = get_record_info(records, algo_b, chosen[1], test_env_idx)
            return chosen[1], info['ood_acc'] if info else None

        s1_seed, s1_acc = _pick(rel_h_bs,   higher_is_better=False)
        s2_seed, s2_acc = _pick(deviations,  higher_is_better=not cross_line_misspec)
        s3_seed, s3_acc = _pick(rel_dhs,     higher_is_better=not cross_line_misspec)
        has_mmd_cross   = any(v is not None for v in mmd_crosses)
        s5_seed, s5_acc = (_pick(mmd_crosses, higher_is_better=not cross_line_misspec)
                           if has_mmd_cross else (None, None))

        rows.append({
            'algorithm': algo_b, 'benchmark': 'mis' if cross_line_misspec else 'well',
            'pool': pool_label,
            'oracle_seed': oracle[1] if oracle else None,
            'oracle_acc':  oracle[0] if oracle else None,
            'min_entropy_seed': s1_seed, 'min_entropy_acc': s1_acc,
            'adapt_dev_seed':   s2_seed, 'adapt_dev_acc':   s2_acc,
            'adapt_dH_seed':    s3_seed, 'adapt_dH_acc':    s3_acc,
            'adapt_mmd_x_seed':   s5_seed, 'adapt_mmd_x_acc':   s5_acc,
        })
    return rows


def print_strategy_comparison(rows, dataset_name, test_env_name):
    strategies = [('min_entropy','min H'), ('adapt_dev','adap |dev|'),
                  ('adapt_dH','adap dH'), ('adapt_mmd_x','adap mmd_x')]
    print(f"\n{'='*110}")
    print(f"  Selection strategy comparison -- {dataset_name} (test: {test_env_name})")
    print(f"{'='*110}")
    hdr = f"  {'Algo':<10} {'type':>5} {'pool':>12} | {'oracle':>14}"
    for _, label in strategies:
        hdr += f" | {label:>12}"
    print(hdr)
    print(f"  {'x'*108}")
    for row in rows:
        ora = (f"s={row['oracle_seed']} {row['oracle_acc']:.3f}"
               if row['oracle_acc'] is not None else 'N/A')
        line = f"  {row['algorithm']:<10} {row['benchmark']:>5} {row['pool']:>12} | {ora:>14}"
        for key, _ in strategies:
            seed = row[f'{key}_seed']; acc = row[f'{key}_acc']
            if acc is None:
                cell = f"{'x':>12}"
            else:
                gap  = acc - row['oracle_acc'] if row['oracle_acc'] is not None else 0
                cell = f"s={seed} {acc:.3f}({gap:+.3f})"
            line += f" | {cell:>12}"
        print(line)


def select_best_model(preds_dir, records, algo, valid_seeds, n_trials,
                       test_env_idx, max_entropy):
    candidates = []
    for seed in valid_seeds:
        probs = get_all_trials(preds_dir, algo, seed, n_trials, test_env_idx, load_probs)
        if not probs:
            continue
        rel_h = float(np.mean([compute_entropy(p) for p in probs])) / max_entropy
        candidates.append({'seed': seed, 'rel_h': rel_h})
    if not candidates:
        return None
    candidates.sort(key=lambda c: c['rel_h'])
    best = candidates[0]; best['algorithm'] = algo
    return best


def compute_candidate_seeds_only(preds_dir, algo_a, algo_b, erm_line,
                                  valid_seeds, n_trials, test_env_idx):
    erm_ood_median = erm_line['ood_median']
    return [
        seed for seed in valid_seeds
        if (get_ood_agr(preds_dir, algo_a, algo_b, seed, seed,
                        n_trials, test_env_idx) or 1.0) < erm_ood_median
        and get_ood_agr(preds_dir, algo_a, algo_b, seed, seed,
                        n_trials, test_env_idx) is not None
    ]


def mean_relative_entropy(preds_dir, algo, seeds, n_trials, test_env_idx, max_entropy):
    rel_hs = []
    for seed in seeds:
        probs = get_all_trials(preds_dir, algo, seed, n_trials, test_env_idx, load_probs)
        if not probs:
            continue
        rel_hs.append(float(np.mean([compute_entropy(p) for p in probs])) / max_entropy)
    return float(np.mean(rel_hs)) if rel_hs else None


def recommend_algorithm_and_model(
    preds_dir, records_path, test_env_idx, n_envs, n_hparams,
    n_trials=3, algo_a='ERM', algo_bs=None,
    entropy_threshold=0.9, cross_line_r_threshold=0.3,
):
    """
    Regime detection uses a MAJORITY VOTE across all algo_bs' own Cross-R
    values, rather than each algorithm's individual Cross-R sign. A single
    algorithm's optimization quirks (e.g. IRM's bimodal seed population
    under bad hyperparameters) can flip its own Cross-R without changing
    what every other algorithm sees on the same benchmark/environment —
    the regime is a property of the benchmark, not of any one algorithm,
    so every algo_b is treated as a DG candidate (or not) based on the
    same dataset-level vote rather than its own individual R.
    """
    with open(records_path) as f:
        records = json.load(f)
    algo_bs = algo_bs or []

    n_classes = 2
    for seed in range(n_hparams):
        for trial in range(n_trials):
            probs = load_probs(preds_dir, algo_a, seed, trial, test_env_idx)
            if probs is not None:
                n_classes = probs.shape[1]; break
        else:
            continue
        break
    max_entropy = float(np.log(n_classes))

    valid_a, _ = get_valid_seeds(preds_dir, algo_a, n_hparams, n_trials,
                                  test_env_idx, max_entropy, entropy_threshold)
    erm_line = compute_erm_line(preds_dir, valid_a, n_trials, test_env_idx, n_envs)

    # Pass 1: each algo_b casts one Cross-R vote
    all_cross_lines, valid_seeds_by_algo, valid_b_by_algo = {}, {}, {}
    for algo_b in algo_bs:
        valid_b, _ = get_valid_seeds(preds_dir, algo_b, n_hparams, n_trials,
                                      test_env_idx, max_entropy, entropy_threshold)
        valid_seeds = sorted(set(valid_a) & set(valid_b))
        if len(valid_seeds) < 2:
            continue
        cross_line = compute_cross_line(preds_dir, valid_seeds, valid_seeds,
                                         algo_a, algo_b, n_trials, test_env_idx, n_envs)
        if cross_line is None:
            continue
        all_cross_lines[algo_b]     = cross_line['R']
        valid_seeds_by_algo[algo_b] = valid_seeds
        valid_b_by_algo[algo_b]     = valid_b

    # Majority vote on the dataset-level regime
    n_mis  = sum(1 for r in all_cross_lines.values() if r >= cross_line_r_threshold)
    n_well = sum(1 for r in all_cross_lines.values() if r <  cross_line_r_threshold)
    well_specified = n_well > n_mis if all_cross_lines else False

    # Pass 2: if the majority says well-specified, every algo_b is a DG
    # candidate regardless of its own individual Cross-R sign
    separate_trend_algos, all_mean_candidate_entropies = [], {}
    if well_specified:
        for algo_b, R in all_cross_lines.items():
            candidate_seeds = compute_candidate_seeds_only(
                preds_dir, algo_a, algo_b, erm_line,
                valid_seeds_by_algo[algo_b], n_trials, test_env_idx)
            mean_h = mean_relative_entropy(preds_dir, algo_b, candidate_seeds,
                                           n_trials, test_env_idx, max_entropy)
            all_mean_candidate_entropies[algo_b] = mean_h
            if mean_h is not None:
                separate_trend_algos.append(
                    (algo_b, R, mean_h, valid_b_by_algo[algo_b]))

    if separate_trend_algos:
        separate_trend_algos.sort(key=lambda t: t[2])
        chosen_algo, chosen_R, chosen_mean_h, chosen_valid = separate_trend_algos[0]
        best = select_best_model(preds_dir, records, chosen_algo,
                                  chosen_valid, n_trials, test_env_idx, max_entropy)
        return {
            'recommendation': 'use_dg', 'algorithm': chosen_algo,
            'seed': best['seed'], 'rel_h': best['rel_h'],
            'cross_line_R': chosen_R, 'mean_candidate_entropy': chosen_mean_h,
            'all_cross_lines': all_cross_lines,
            'all_mean_candidate_entropies': all_mean_candidate_entropies,
            'valid_a': valid_a,
            'n_mis': n_mis, 'n_well': n_well,
        }
    else:
        best = select_best_model(preds_dir, records, algo_a, valid_a,
                                  n_trials, test_env_idx, max_entropy)
        return {
            'recommendation': 'use_erm', 'algorithm': algo_a,
            'seed': best['seed'] if best else None,
            'rel_h': best['rel_h'] if best else None,
            'cross_line_R': None, 'mean_candidate_entropy': None,
            'all_cross_lines': all_cross_lines,
            'all_mean_candidate_entropies': all_mean_candidate_entropies,
            'valid_a': valid_a,
            'n_mis': n_mis, 'n_well': n_well,
        }


def evaluate_selection_quality(selection, records_path, test_env_idx,
                               algo_a, algo_bs, n_hparams):
    with open(records_path) as f:
        records = json.load(f)
    all_algos = [algo_a] + list(algo_bs)
    our_info  = get_record_info(records, selection['algorithm'],
                                selection['seed'], test_env_idx)
    our_acc   = our_info['ood_acc'] if our_info else None

    oracle_candidates, iid_candidates = [], []
    for algo in all_algos:
        for seed in range(n_hparams):
            info = get_record_info(records, algo, seed, test_env_idx)
            if info is None:
                continue
            oracle_candidates.append((info['ood_acc'], algo, seed))
            matching = [r for r in records
                        if r['algorithm'] == algo and r['args']['hparams_seed'] == seed]
            if not matching:
                continue
            train_accs = [matching[0][k] for k in matching[0]
                          if k.endswith('_out_acc') and k != f'env{test_env_idx}_out_acc']
            if train_accs:
                iid_candidates.append((float(np.mean(train_accs)), info['ood_acc'], algo, seed))

    if not oracle_candidates:
        return None
    oracle_acc, oracle_algo, oracle_seed = max(oracle_candidates, key=lambda t: t[0])
    iid_result = None
    if iid_candidates:
        best_iid = max(iid_candidates, key=lambda t: t[0])
        iid_result = {'algorithm': best_iid[2], 'seed': best_iid[3], 'ood_acc': best_iid[1]}
    return {
        'our_algorithm':  selection['algorithm'],
        'our_seed':       selection['seed'],
        'our_acc':        our_acc,
        'oracle_acc':     float(oracle_acc),
        'oracle_algorithm': oracle_algo,
        'oracle_seed':    oracle_seed,
        'iid_acc':        iid_result['ood_acc']   if iid_result else None,
        'iid_algorithm':  iid_result['algorithm'] if iid_result else None,
        'iid_seed':       iid_result['seed']       if iid_result else None,
        'gap_to_oracle':     float(oracle_acc) - our_acc if our_acc is not None else None,
        'gap_to_oracle_iid': float(oracle_acc) - iid_result['ood_acc']
                             if iid_result is not None else None,
        'our_beats_iid': (our_acc is not None and iid_result is not None
                          and our_acc > iid_result['ood_acc']),
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_table(results, dataset_name, test_env_name):
    print(f"\n{'='*80}")
    print(f"  Diagnostic summary -- {dataset_name} (test: {test_env_name})")
    print(f"{'='*80}")
    if results is None:
        print("  No results"); return

    erm    = results['erm_line']
    cross  = results['cross_line']
    sl     = results['seed_level_line']
    algo_a = results['algo_a']
    algo_b = results['algo_b']
    r_thresh = results['cross_line_r_threshold']

    print(f"  {algo_a}-{algo_a} R (reference):         {erm['R']:+.3f}  "
          f"(ood_median={erm['ood_median']:.3f}, n_pairs={erm['n_pairs']})")
    if cross is not None:
        print(f"  {algo_a}-{algo_b} Cross-R (pairs):      {cross['R']:+.3f}  "
              f"(n_pairs={cross['n_pairs']})")
    else:
        print(f"  {algo_a}-{algo_b} Cross-R:              N/A")

    if sl is not None:
        print(f"  {algo_a}-{algo_b} Seed-level R:         {sl['R']:+.3f}  "
              f"(n_seeds={len(sl['seed_ids'])})")
        if cross is not None:
            same_sign = (cross['R'] >= 0) == (sl['R'] >= 0)
            print(f"  Sign agreement (Cross-R vs seed-level): {same_sign}")
    else:
        print(f"  {algo_a}-{algo_b} Seed-level R:         N/A")

    if results['cross_line_misspecified'] is None:
        print(f"\n  Cross-R >= {r_thresh} -> N/A")
    else:
        sym = ('misspecified (shared trend, ERM sufficient)'
               if results['cross_line_misspecified']
               else 'well-specified (separate trend, DG may help)')
        print(f"\n  Cross-R >= {r_thresh} -> {sym}")

    cand = results['candidate_agreement']
    onl  = results['online_agreement']
    print(f"\n  {algo_b}-{algo_b} OOD agr (candidates): " +
          (f"{cand:.3f}" if cand is not None else "N/A"))
    print(f"  {algo_b}-{algo_b} OOD agr (on line):    " +
          (f"{onl:.3f}"  if onl  is not None else "N/A"))
    print(f"  {algo_a}-{algo_a} OOD median (ref):      {erm['ood_median']:.3f}")
    print(f"\n  Candidates: {results['n_candidates']}  On line: {results['n_on_line']}")

    if results['genuine_escape'] is None:
        print(f"\n  Verdict: INCONCLUSIVE ({results['n_candidates']} candidate(s))")
    else:
        dr  = results['disagreement_rate']
        sym = 'Y' if results['genuine_escape'] else 'N'
        print(f"\n  Disagreement rate: {dr:.3f}  Verdict: {results['dg_verdict']} [{sym}]")


def print_selection_report(selection, evaluation, dataset_name, test_env_name):
    print(f"\n{'='*80}")
    print(f"  Label-free model selection -- {dataset_name} (test: {test_env_name})")
    print(f"{'='*80}")
    print(f"  Stage 1 -- per algorithm:")
    print(f"    {'algorithm':<12} {'cross_line_R':>12} {'mean_cand_rel_h':>16}")
    for algo, R in selection['all_cross_lines'].items():
        mean_h = selection['all_mean_candidate_entropies'].get(algo)
        print(f"    {algo:<12} {R:>+12.3f} {mean_h:.3f if mean_h is not None else 'N/A':>16}")
    print(f"\n  Recommendation: {selection['recommendation']}")
    if selection['seed'] is not None:
        print(f"  Selected: {selection['algorithm']} seed={selection['seed']} "
              f"(rel_h={selection['rel_h']:.3f})")
    if evaluation is not None:
        print(f"\n  POST-HOC EVALUATION")
        if evaluation['our_acc'] is not None:
            print(f"  Our:    {evaluation['our_algorithm']:<10} "
                  f"seed={evaluation['our_seed']} acc={evaluation['our_acc']:.3f}")
        if evaluation['iid_acc'] is not None:
            print(f"  IID:    {evaluation['iid_algorithm']:<10} "
                  f"seed={evaluation['iid_seed']} acc={evaluation['iid_acc']:.3f}")
        print(f"  Oracle: {evaluation['oracle_algorithm']:<10} "
              f"seed={evaluation['oracle_seed']} acc={evaluation['oracle_acc']:.3f}")
        if evaluation['gap_to_oracle'] is not None:
            print(f"  Gap to oracle (ours): {evaluation['gap_to_oracle']:+.3f}")
        if evaluation['gap_to_oracle_iid'] is not None:
            print(f"  Gap to oracle (IID):  {evaluation['gap_to_oracle_iid']:+.3f}")
        sym = 'Y' if evaluation['our_beats_iid'] else 'N'
        print(f"  Our beats IID: {sym}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _agreement_pct_ticks(all_agrs, n_ticks=8):
    lo, hi = all_agrs.min() * 100, all_agrs.max() * 100
    pad = (hi - lo) * 0.1
    lo, hi = max(lo - pad, 1), min(hi + pad, 99)
    tick_pcts   = np.linspace(lo, hi, n_ticks)
    tick_probit = probit(tick_pcts / 100.0)
    return tick_pcts, tick_probit


def plot_cross_algorithm(results, dataset_name, test_env_name, save_path=None):
    """
    Three lines on one plot:
      Gray  dashed  : ERM-ERM reference
      Orange solid  : Cross-R (all ERM x DG pairs)
      Red    dashed : Seed-level (one diamond per DG seed, mean agr with ERM pool)
    Blue circles = on-line seeds, green triangles = candidate seeds (same-seed pairs).
    """
    erm    = results['erm_line']
    cross  = results['cross_line']
    sl     = results['seed_level_line']
    algo_a = results['algo_a']
    algo_b = results['algo_b']
    eps    = 1e-6

    fig, ax = plt.subplots(figsize=(8, 7))

    # ERM-ERM reference
    erm_id_p  = probit(np.clip(np.array(erm['id_agrs']),  eps, 1 - eps))
    erm_ood_p = probit(np.clip(np.array(erm['ood_agrs']), eps, 1 - eps))
    ax.scatter(erm_id_p, erm_ood_p, s=15, color='lightgray', alpha=0.6, zorder=1,
               label=f'{algo_a}-{algo_a} pairs (n={len(erm_id_p)})')
    x1 = np.linspace(erm_id_p.min(), erm_id_p.max(), 100)
    ax.plot(x1, erm['slope'] * x1 + erm['intercept'],
            color='gray', linestyle='--', linewidth=2, zorder=2,
            label=f'{algo_a}-{algo_a} R={erm["R"]:+.3f}')

    all_agr_parts = [np.array(erm['id_agrs']), np.array(erm['ood_agrs'])]

    # Cross-R pair-level (solid orange)
    if cross is not None:
        cross_id_p  = probit(np.clip(np.array(cross['id_agrs']),  eps, 1 - eps))
        cross_ood_p = probit(np.clip(np.array(cross['ood_agrs']), eps, 1 - eps))
        ax.scatter(cross_id_p, cross_ood_p, s=8, color='lightsalmon',
                   alpha=0.3, zorder=1,
                   label=f'{algo_a}-{algo_b} pairs (n={len(cross_id_p)})')
        x2 = np.linspace(cross_id_p.min(), cross_id_p.max(), 100)
        ax.plot(x2, cross['slope'] * x2 + cross['intercept'],
                color='tab:orange', linestyle='-', linewidth=2, zorder=3,
                label=f'Cross-R (pairs) R={cross["R"]:+.3f}')
        all_agr_parts += [np.array(cross['id_agrs']), np.array(cross['ood_agrs'])]

    # Seed-level (dashed red, diamond markers)
    if sl is not None:
        sl_id_p  = probit(np.clip(np.array(sl['x_per_seed']), eps, 1 - eps))
        sl_ood_p = probit(np.clip(np.array(sl['y_per_seed']), eps, 1 - eps))
        ax.scatter(sl_id_p, sl_ood_p, s=80, color='tab:red',
                   marker='D', edgecolor='k', zorder=5, alpha=0.9,
                   label=f'{algo_b} seeds (n={len(sl_id_p)})')
        x3 = np.linspace(sl_id_p.min(), sl_id_p.max(), 100)
        ax.plot(x3, sl['slope'] * x3 + sl['intercept'],
                color='tab:red', linestyle='--', linewidth=2, zorder=4,
                label=f'Seed-level R={sl["R"]:+.3f}')
        for i, sid in enumerate(sl['seed_ids']):
            ax.annotate(str(sid), (sl_id_p[i], sl_ood_p[i]),
                        textcoords='offset points', xytext=(4, 4),
                        fontsize=7, color='tab:red')

    # Same-seed candidate / on-line scatter
    if results['id_agrs']:
        id_p     = probit(np.clip(np.array(results['id_agrs']),  eps, 1 - eps))
        ood_p    = probit(np.clip(np.array(results['ood_agrs']), eps, 1 - eps))
        statuses = np.array(results['step2_statuses'])
        seeds    = results['seeds_used']
        cand_mask = statuses == 'candidate'
        line_mask = statuses == 'on line'
        ax.scatter(id_p[line_mask], ood_p[line_mask], s=70, color='tab:blue',
                   marker='o', edgecolor='k', zorder=4,
                   label=f'on line (n={line_mask.sum()})')
        ax.scatter(id_p[cand_mask], ood_p[cand_mask], s=70, color='tab:green',
                   marker='^', edgecolor='k', zorder=4,
                   label=f'candidates (n={cand_mask.sum()})')
        for i, seed in enumerate(seeds):
            ax.annotate(str(seed), (id_p[i], ood_p[i]),
                        textcoords='offset points', xytext=(4, 4), fontsize=7)

    all_agrs = np.concatenate(all_agr_parts)
    tick_pcts, tick_probit = _agreement_pct_ticks(all_agrs)
    ax.set_xticks(tick_probit); ax.set_xticklabels([f'{p:.0f}' for p in tick_pcts])
    ax.set_yticks(tick_probit); ax.set_yticklabels([f'{p:.0f}' for p in tick_pcts])
    ax.set_xlabel('ID agreement (%)')
    ax.set_ylabel('OOD agreement (%)')

    title = f'{dataset_name} ({test_env_name})'
    if cross is not None and sl is not None:
        title += f'\nCross-R={cross["R"]:+.3f}  |  Seed-level R={sl["R"]:+.3f}'
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'\n  Plot saved to: {save_path}')
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_multi_algorithm(results_list, dataset_name, test_env_name, save_path=None):
    """
    All DG algorithms on one plot.
    Same color per algorithm:
      Solid line   = Cross-R (all ERM x DG pairs)
      Dotted line  = Seed-level (mean agr per DG seed with ERM pool)
      Diamond markers = individual DG seeds
    Gray dashed = ERM-ERM reference.
    """
    eps    = 1e-6
    fig, ax = plt.subplots(figsize=(9, 8))
    algo_a  = results_list[0]['algo_a']

    erm       = results_list[0]['erm_line']
    erm_id_p  = probit(np.clip(np.array(erm['id_agrs']),  eps, 1 - eps))
    erm_ood_p = probit(np.clip(np.array(erm['ood_agrs']), eps, 1 - eps))
    ax.scatter(erm_id_p, erm_ood_p, s=12, color='lightgray', alpha=0.4, zorder=1)
    x1 = np.linspace(erm_id_p.min(), erm_id_p.max(), 100)
    ax.plot(x1, erm['slope'] * x1 + erm['intercept'],
            color='gray', linestyle='--', linewidth=2, zorder=2,
            label=f'{algo_a}-{algo_a} R={erm["R"]:+.3f}')

    colors        = plt.cm.tab10(np.linspace(0, 1, max(len(results_list), 1)))
    all_agr_parts = [np.array(erm['id_agrs']), np.array(erm['ood_agrs'])]

    for res, color in zip(results_list, colors):
        cross  = res['cross_line']
        sl     = res['seed_level_line']
        algo_b = res['algo_b']

        if cross is not None:
            cross_id_p  = probit(np.clip(np.array(cross['id_agrs']),  eps, 1 - eps))
            cross_ood_p = probit(np.clip(np.array(cross['ood_agrs']), eps, 1 - eps))
            x2 = np.linspace(cross_id_p.min(), cross_id_p.max(), 100)
            dr_str = f"{res['disagreement_rate']:.2f}" if res['disagreement_rate'] is not None else 'N/A'
            ax.plot(x2, cross['slope'] * x2 + cross['intercept'],
                    color=color, linestyle='-', linewidth=2, zorder=3,
                    label=f'{algo_b} Cross-R={cross["R"]:+.3f} DR={dr_str}')
            all_agr_parts += [np.array(cross['id_agrs']), np.array(cross['ood_agrs'])]

        if sl is not None:
            sl_id_p  = probit(np.clip(np.array(sl['x_per_seed']), eps, 1 - eps))
            sl_ood_p = probit(np.clip(np.array(sl['y_per_seed']), eps, 1 - eps))
            ax.scatter(sl_id_p, sl_ood_p, s=60, color=color,
                       marker='D', edgecolor='k', zorder=5, alpha=0.85)
            x3 = np.linspace(sl_id_p.min(), sl_id_p.max(), 100)
            ax.plot(x3, sl['slope'] * x3 + sl['intercept'],
                    color=color, linestyle=':', linewidth=2, zorder=4,
                    label=f'{algo_b} seed-level R={sl["R"]:+.3f}')
            for i, sid in enumerate(sl['seed_ids']):
                ax.annotate(str(sid), (sl_id_p[i], sl_ood_p[i]),
                            textcoords='offset points', xytext=(3, 3),
                            fontsize=6, color=color)

    all_agrs = np.concatenate(all_agr_parts)
    tick_pcts, tick_probit = _agreement_pct_ticks(all_agrs)
    ax.set_xticks(tick_probit); ax.set_xticklabels([f'{p:.0f}' for p in tick_pcts])
    ax.set_yticks(tick_probit); ax.set_yticklabels([f'{p:.0f}' for p in tick_pcts])
    ax.set_xlabel('ID agreement (%)')
    ax.set_ylabel('OOD agreement (%)')
    ax.set_title(f'{dataset_name} ({test_env_name})\n'
                 f'Solid = Cross-R (pairs)  |  Dotted = Seed-level  |  Diamonds = DG seeds')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3); fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'\n  Multi-algorithm plot saved to: {save_path}')
        plt.close(fig)
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_dir',      type=str, required=True)
    parser.add_argument('--records_path',   type=str, required=True)
    parser.add_argument('--test_env_idx',   type=int, required=True)
    parser.add_argument('--n_envs',         type=int, required=True)
    parser.add_argument('--n_hparams',      type=int, default=20)
    parser.add_argument('--n_trials',       type=int, default=3)
    parser.add_argument('--algo_a',         type=str, default='ERM')
    parser.add_argument('--algo_b',         type=str, default=['IRM'], nargs='+')
    parser.add_argument('--dataset_name',   type=str, default='Dataset')
    parser.add_argument('--test_env_name',  type=str, default='test')
    parser.add_argument('--entropy_threshold',           type=float, default=0.9)
    parser.add_argument('--cross_line_r_threshold',      type=float, default=0.3)
    parser.add_argument('--disagreement_rate_threshold', type=float, default=0.5)
    parser.add_argument('--irm_anneal_threshold', type=int, default=None)
    parser.add_argument('--plot_path',       type=str, default=None)
    parser.add_argument('--multi_plot_path', type=str, default=None)
    parser.add_argument('--select',             action='store_true')
    parser.add_argument('--evaluate_selection', action='store_true')
    parser.add_argument('--per_algo_selection', action='store_true')
    parser.add_argument('--compare_strategies', action='store_true')
    parser.add_argument('--distance_metrics', type=str, nargs='*', default=[],
                        choices=['mmd', 'wasserstein', 'pad'])
    parser.add_argument('--input_dataset', type=str, default=None,
                        help='src.datasets.DATASET_CONFIGS key (e.g. ColoredMNIST, '
                             'RotatedMNIST, PACS, ACSIncome) for input-space distance_metrics')
    parser.add_argument('--input_data_dir', type=str, default=None,
                        help='Raw data dir for --input_dataset (required if --distance_metrics set)')
    parser.add_argument('--input_backbone', type=str, default='resnet50',
                        help='Transform backbone for --input_dataset (PACS only; ignored otherwise)')
    args = parser.parse_args()

    # Input-space shift is a dataset-level property (identical for every
    # algo_b), so compute it once here rather than once per algo_b below.
    precomputed_input_shift = None
    if args.distance_metrics:
        precomputed_input_shift = compute_input_space_shift(
            args.input_dataset, args.input_data_dir, args.test_env_idx, args.n_envs,
            backbone=args.input_backbone,
            run_mmd='mmd' in args.distance_metrics,
            run_wass='wasserstein' in args.distance_metrics,
            run_pad='pad' in args.distance_metrics,
        )

    results_list = []
    for algo_b in args.algo_b:
        print(f"\nComputing cross-algorithm agreement ({args.algo_a} vs {algo_b})...")
        results = compute_cross_algorithm_agreement(
            preds_dir                   = args.preds_dir,
            records_path                = args.records_path,
            test_env_idx                = args.test_env_idx,
            n_envs                      = args.n_envs,
            n_hparams                   = args.n_hparams,
            n_trials                    = args.n_trials,
            algo_a                      = args.algo_a,
            algo_b                      = algo_b,
            entropy_threshold           = args.entropy_threshold,
            cross_line_r_threshold      = args.cross_line_r_threshold,
            disagreement_rate_threshold = args.disagreement_rate_threshold,
            distance_metrics            = args.distance_metrics,
            irm_anneal_threshold        = args.irm_anneal_threshold,
            input_dataset               = args.input_dataset,
            input_data_dir              = args.input_data_dir,
            input_backbone              = args.input_backbone,
            precomputed_input_shift     = precomputed_input_shift,
        )
        print_table(results, args.dataset_name, args.test_env_name)
        results_list.append(results)
        if len(args.algo_b) == 1 and args.plot_path:
            plot_cross_algorithm(results, args.dataset_name, args.test_env_name,
                                 save_path=args.plot_path)

    if len(results_list) > 1 or args.multi_plot_path:
        plot_multi_algorithm(results_list, args.dataset_name, args.test_env_name,
                             save_path=args.multi_plot_path)
    elif len(args.algo_b) == 1 and args.plot_path is None:
        plot_cross_algorithm(results_list[0], args.dataset_name,
                             args.test_env_name, save_path=None)

    if args.per_algo_selection:
        rows = evaluate_per_algorithm(results_list, args.records_path,
                                      args.test_env_idx, args.algo_a, args.n_hparams)
        print_per_algorithm_selection(rows, args.dataset_name, args.test_env_name)

    if args.compare_strategies:
        rows = compare_selection_strategies(results_list, args.records_path,
                                            args.test_env_idx, args.algo_a, args.n_hparams)
        print_strategy_comparison(rows, args.dataset_name, args.test_env_name)

    if args.select:
        selection = recommend_algorithm_and_model(
            preds_dir=args.preds_dir, records_path=args.records_path,
            test_env_idx=args.test_env_idx, n_envs=args.n_envs,
            n_hparams=args.n_hparams, n_trials=args.n_trials,
            algo_a=args.algo_a, algo_bs=args.algo_b,
            entropy_threshold=args.entropy_threshold,
            cross_line_r_threshold=args.cross_line_r_threshold,
        )
        evaluation = None
        if args.evaluate_selection:
            evaluation = evaluate_selection_quality(
                selection=selection, records_path=args.records_path,
                test_env_idx=args.test_env_idx, algo_a=args.algo_a,
                algo_bs=args.algo_b, n_hparams=args.n_hparams,
            )
        print_selection_report(selection, evaluation,
                               args.dataset_name, args.test_env_name)