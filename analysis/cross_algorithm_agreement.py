# analysis/cross_algorithm_agreement.py
"""
Cross-algorithm agreement diagnostic.

Pipeline:
    Step 0: Entropy filter — restrict to seeds where both algo_a and algo_b
            produce non-degenerate predictions (normalized entropy < threshold).

    Step 1: Compute ERM-ERM reference line AND algo_a-algo_b cross-pair line,
            both in probit space.
            High R on the cross-pair line (analogous to Salaudeen ACL R >= 0.3)
            suggests ID agreement predicts OOD agreement consistently across
            algorithms -> better-ID tends to be better-OOD -> MISSPECIFIED.
            Low R -> no such transfer -> well-specified (DG may be needed).

    Step 2: Identify candidate algo_b escapes (same-seed pairs):
            OOD_agr < erm_ood_median  ->  'candidate'
            else                      ->  'on line'

    Step 3: Validate candidates via algo_b-algo_b agreement:
            disagreement_rate = candidate_agreement / erm_ood_median
            High -> candidates converge to a shared alternative -> DG useful.
            Low  -> candidates scatter (degenerate) -> DG not useful here.

Usage:
    python analysis/cross_algorithm_agreement.py \
        --preds_dir    results/coloredmnist/test_env2/cnn/random/models \
        --records_path results/coloredmnist/test_env2/cnn/random/records.json \
        --test_env_idx 2 \
        --n_envs       3 \
        --n_hparams    20 \
        --n_trials     3 \
        --algo_a       ERM \
        --algo_b       IRM \
        --dataset_name ColoredMNIST \
        --test_env_name "-90% (env2)" \
        --plot_path    cmnist_plot.png
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri as probit
from scipy.special import ndtr as normal_cdf
from scipy.stats import pearsonr, linregress
from itertools import combinations


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

def load_predictions(preds_dir, algorithm, hparams_seed, trial_seed, env_idx):
    fname = (f"{algorithm}_hpseed{hparams_seed}"
             f"_trial{trial_seed}_env{env_idx}_preds.npy")
    path = os.path.join(preds_dir, fname)
    return np.load(path) if os.path.exists(path) else None


def load_probs(preds_dir, algorithm, hparams_seed, trial_seed, env_idx):
    fname = (f"{algorithm}_hpseed{hparams_seed}"
             f"_trial{trial_seed}_env{env_idx}_probs.npy")
    path = os.path.join(preds_dir, fname)
    return np.load(path) if os.path.exists(path) else None


def get_all_trials(preds_dir, algorithm, hparams_seed, n_trials,
                   env_idx, loader_fn):
    results = []
    for trial in range(n_trials):
        r = loader_fn(preds_dir, algorithm, hparams_seed, trial, env_idx)
        if r is not None:
            results.append(r)
    return results


def get_record_info(records, algo, seed, test_env_idx):
    matching = [r for r in records
                if r['algorithm'] == algo
                and r['args']['hparams_seed'] == seed]
    if not matching:
        return None
    ood_acc = np.mean([r[f'env{test_env_idx}_out_acc'] for r in matching])
    hp = matching[0]['hparams']
    return {
        'ood_acc': float(ood_acc),
        'lr':      hp.get('lr', 0),
        'lambda':  hp.get('irm_lambda', hp.get('mmd_gamma', None)),
        'anneal':  hp.get('irm_penalty_anneal_iters', None),
        'bs':      hp.get('batch_size', None),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_agreement(preds_i, preds_j):
    return float(np.mean(preds_i == preds_j))


def compute_entropy(probs):
    return float(-np.sum(probs * np.log(probs + 1e-8), axis=1).mean())


def get_id_agr(preds_dir, algo_a, algo_b, seed_a, seed_b,
               n_trials, test_env_idx, n_envs):
    id_agr_vals = []
    for env_idx in range(n_envs):
        if env_idx == test_env_idx:
            continue
        preds_a = get_all_trials(preds_dir, algo_a, seed_a,
                                 n_trials, env_idx, load_predictions)
        preds_b = get_all_trials(preds_dir, algo_b, seed_b,
                                 n_trials, env_idx, load_predictions)
        if not preds_a or not preds_b:
            continue
        for pa in preds_a:
            for pb in preds_b:
                id_agr_vals.append(compute_agreement(pa, pb))
    return float(np.mean(id_agr_vals)) if id_agr_vals else None


def get_ood_agr(preds_dir, algo_a, algo_b, seed_a, seed_b,
                n_trials, test_env_idx):
    preds_a = get_all_trials(preds_dir, algo_a, seed_a,
                             n_trials, test_env_idx, load_predictions)
    preds_b = get_all_trials(preds_dir, algo_b, seed_b,
                             n_trials, test_env_idx, load_predictions)
    if not preds_a or not preds_b:
        return None
    return float(np.mean([
        compute_agreement(pa, pb)
        for pa in preds_a for pb in preds_b
    ]))


# ---------------------------------------------------------------------------
# Entropy-based seed filtering
# ---------------------------------------------------------------------------

def get_valid_seeds(preds_dir, algorithm, n_hparams, n_trials, test_env_idx,
                    max_entropy, entropy_threshold=0.9):
    valid, excluded = [], []
    for seed in range(n_hparams):
        probs = get_all_trials(preds_dir, algorithm, seed, n_trials,
                               test_env_idx, load_probs)
        if not probs:
            continue
        rel_h = float(np.mean([compute_entropy(p) for p in probs])) / max_entropy
        if rel_h < entropy_threshold:
            valid.append(seed)
        else:
            excluded.append((seed, rel_h))
    return valid, excluded


# ---------------------------------------------------------------------------
# Agreement lines (probit-space linear fit)
# ---------------------------------------------------------------------------

def _fit_line(id_agrs, ood_agrs):
    id_agrs  = np.array(id_agrs)
    ood_agrs = np.array(ood_agrs)
    eps = 1e-6
    id_probit  = probit(np.clip(id_agrs,  eps, 1 - eps))
    ood_probit = probit(np.clip(ood_agrs, eps, 1 - eps))
    R, p_value = pearsonr(id_probit, ood_probit)
    reg = linregress(id_probit, ood_probit)
    return {
        'R':          float(R),
        'slope':      float(reg.slope),
        'intercept':  float(reg.intercept),
        'p_value':    float(p_value),
        'id_agrs':    id_agrs.tolist(),
        'ood_agrs':   ood_agrs.tolist(),
        'ood_median': float(np.median(ood_agrs)),
        'ood_mean':   float(np.mean(ood_agrs)),
        'n_pairs':    len(id_agrs),
    }


def compute_erm_line(preds_dir, valid_seeds, n_trials, test_env_idx, n_envs):
    """ERM-ERM reference line: all unordered pairs among valid_seeds."""
    id_agrs, ood_agrs = [], []
    for seed_i, seed_j in combinations(valid_seeds, 2):
        id_agr  = get_id_agr(preds_dir, 'ERM', 'ERM', seed_i, seed_j,
                             n_trials, test_env_idx, n_envs)
        ood_agr = get_ood_agr(preds_dir, 'ERM', 'ERM', seed_i, seed_j,
                              n_trials, test_env_idx)
        if id_agr is not None and ood_agr is not None:
            id_agrs.append(id_agr)
            ood_agrs.append(ood_agr)
    line = _fit_line(id_agrs, ood_agrs)
    line['n_seeds'] = len(valid_seeds)
    return line


def compute_cross_line(preds_dir, valid_a, valid_b, algo_a, algo_b,
                       n_trials, test_env_idx, n_envs):
    """algo_a-algo_b line over ALL cross pairs (seed_i in valid_a x seed_j in valid_b)."""
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
    return _fit_line(id_agrs, ood_agrs)


# ---------------------------------------------------------------------------
# algo_b-algo_b agreement among candidate / on-line seeds
# ---------------------------------------------------------------------------

def compute_candidate_agreement(preds_dir, candidate_seeds, algo_b,
                                n_trials, test_env_idx):
    """
    Mean OOD agreement between all pairs of candidate algo_b seeds.
    High -> candidates converged to same solution -> genuine escape.
    Low  -> candidates scattered -> degenerate.
    """
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
# Main computation
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
):
    with open(records_path) as f:
        records = json.load(f)

    # Get n_classes from first probs file
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

    # ---- Step 0: entropy filter ----
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
    erm_line = compute_erm_line(preds_dir, valid_seeds, n_trials, test_env_idx, n_envs)
    erm_ood_median = erm_line['ood_median']

    print(f"  {algo_a}-{algo_a}: R={erm_line['R']:+.3f}  "
          f"slope={erm_line['slope']:.3f}  "
          f"intercept={erm_line['intercept']:.3f}  "
          f"ood_median={erm_ood_median:.3f}  "
          f"n_pairs={erm_line['n_pairs']}")

    # ---- Step 1b: algo_a-algo_b cross-pair line (misspecification probe) ----
    print(f"\n  Computing {algo_a}-{algo_b} cross-pair line...")
    cross_line = compute_cross_line(preds_dir, valid_seeds, valid_seeds,
                                    algo_a, algo_b, n_trials, test_env_idx, n_envs)

    print(f"  {algo_a}-{algo_b}: R={cross_line['R']:+.3f}  "
          f"slope={cross_line['slope']:.3f}  "
          f"intercept={cross_line['intercept']:.3f}  "
          f"ood_median={cross_line['ood_median']:.3f}  "
          f"n_pairs={cross_line['n_pairs']}")

    cross_line_misspecified = cross_line['R'] >= cross_line_r_threshold
    cross_symbol = 'misspecified' if cross_line_misspecified else 'well-specified'
    print(f"  {algo_a}-{algo_b} R >= {cross_line_r_threshold}? "
          f"{cross_line_misspecified}  -> {cross_symbol}")

    # ---- Step 2: same-seed algo_a-algo_b pairs -> candidate / on-line ----
    print(f"\n  Computing {algo_a}-{algo_b} same-seed pairs...")
    print(f"\n  {'seed':>4} | "
          f"{'ID_agr':>6} | {'OOD_agr':>7} | "
          f"{'pred_OOD':>8} | {'|dev|':>6} | "
          f"{'rel_h_b':>7} | {'dH/max':>7} | "
          f"{'acc_'+algo_a:>8} | {'acc_'+algo_b:>8} | "
          f"{'H('+algo_a+')':>7} | {'H('+algo_b+')':>7} | "
          f"{'lambda/gamma':>12} | {'anneal':>6} | step2_status")
    print(f"  {'-'*137}")

    id_agrs, ood_agrs, deviations = [], [], []
    rel_dhs, rel_h_bs = [], []
    entropies_a, entropies_b = [], []
    seeds_used, step2_statuses = [], []

    eps = 1e-6

    for seed in valid_seeds:
        id_agr  = get_id_agr(preds_dir, algo_a, algo_b, seed, seed,
                             n_trials, test_env_idx, n_envs)
        ood_agr = get_ood_agr(preds_dir, algo_a, algo_b, seed, seed,
                              n_trials, test_env_idx)
        if id_agr is None or ood_agr is None:
            continue

        id_probit_val   = probit(np.clip(id_agr, eps, 1 - eps))
        pred_ood_probit = erm_line['slope'] * id_probit_val + erm_line['intercept']
        pred_ood_agr    = float(normal_cdf(pred_ood_probit))
        deviation       = abs(ood_agr - pred_ood_agr)

        probs_a   = get_all_trials(preds_dir, algo_a, seed, n_trials, test_env_idx, load_probs)
        probs_b   = get_all_trials(preds_dir, algo_b, seed, n_trials, test_env_idx, load_probs)
        entropy_a = float(np.mean([compute_entropy(p) for p in probs_a])) if probs_a else None
        entropy_b = float(np.mean([compute_entropy(p) for p in probs_b])) if probs_b else None
        rel_h_b   = entropy_b / max_entropy if entropy_b is not None else None
        rel_dH    = (entropy_b - entropy_a) / max_entropy \
                    if entropy_b is not None and entropy_a is not None else None

        step2 = 'candidate' if ood_agr < erm_ood_median else 'on line'

        info_a   = get_record_info(records, algo_a, seed, test_env_idx)
        info_b   = get_record_info(records, algo_b, seed, test_env_idx)
        acc_a    = info_a['ood_acc'] if info_a else None
        acc_b    = info_b['ood_acc'] if info_b else None
        lambda_b = info_b['lambda']  if info_b else None
        anneal_b = info_b['anneal']  if info_b else None

        id_agrs.append(id_agr)
        ood_agrs.append(ood_agr)
        deviations.append(deviation)
        rel_dhs.append(rel_dH)
        rel_h_bs.append(rel_h_b)
        entropies_a.append(entropy_a)
        entropies_b.append(entropy_b)
        seeds_used.append(seed)
        step2_statuses.append(step2)

        lambda_str = f"{lambda_b:12.1f}" if lambda_b is not None else f"{'—':>12}"
        anneal_str = f"{anneal_b:6d}"    if anneal_b is not None else f"{'—':>6}"
        acc_a_str  = f"{acc_a:8.3f}"     if acc_a is not None else f"{'—':>8}"
        acc_b_str  = f"{acc_b:8.3f}"     if acc_b is not None else f"{'—':>8}"
        ha_str     = f"{entropy_a:7.4f}" if entropy_a is not None else f"{'—':>7}"
        hb_str     = f"{entropy_b:7.4f}" if entropy_b is not None else f"{'—':>7}"
        rhb_str    = f"{rel_h_b:7.3f}"   if rel_h_b is not None else f"{'—':>7}"
        rdh_str    = f"{rel_dH:+7.3f}"   if rel_dH is not None else f"{'—':>7}"

        print(f"  {seed:>4} | "
              f"{id_agr:>6.3f} | {ood_agr:>7.3f} | "
              f"{pred_ood_agr:>8.3f} | {deviation:>6.3f} | "
              f"{rhb_str} | {rdh_str} | "
              f"{acc_a_str} | {acc_b_str} | "
              f"{ha_str} | {hb_str} | "
              f"{lambda_str} | {anneal_str} | {step2}")

    # ---- Step 3: algo_b-algo_b agreement among candidates / on-line ----
    candidate_seeds = [seeds_used[i] for i, s in enumerate(step2_statuses)
                       if s == 'candidate']
    on_line_seeds   = [seeds_used[i] for i, s in enumerate(step2_statuses)
                       if s == 'on line']

    print(f"\n  Candidates: {candidate_seeds}")
    print(f"  On line:    {on_line_seeds}")

    candidate_agreement = compute_candidate_agreement(
        preds_dir, candidate_seeds, algo_b, n_trials, test_env_idx)
    online_agreement = compute_candidate_agreement(
        preds_dir, on_line_seeds, algo_b, n_trials, test_env_idx)

    print(f"\n  Step 3 — {algo_b}-{algo_b} agreement among candidates:")
    if candidate_agreement is not None:
        print(f"  {algo_b}-{algo_b} OOD agr (candidates): {candidate_agreement:.3f}")
    else:
        print(f"  {algo_b}-{algo_b} OOD agr (candidates): N/A")
    if online_agreement is not None:
        print(f"  {algo_b}-{algo_b} OOD agr (on line):    {online_agreement:.3f}")
    else:
        print(f"  {algo_b}-{algo_b} OOD agr (on line):    N/A")
    print(f"  {algo_a}-{algo_a} OOD median (reference):  {erm_ood_median:.3f}")

    # ---- Verdict ----
    n_candidates = len(candidate_seeds)
    n_on_line    = len(on_line_seeds)

    if n_candidates < 2:
        dg_verdict        = 'inconclusive'
        genuine_escape    = None
        disagreement_rate = None
    else:
        disagreement_rate = candidate_agreement / erm_ood_median
        genuine_escape    = disagreement_rate > disagreement_rate_threshold
        dg_verdict        = 'DG useful' if genuine_escape else 'DG not useful'

    results = {
        'erm_line':                     erm_line,
        'cross_line':                   cross_line,
        'cross_line_misspecified':      cross_line_misspecified,
        'id_agrs':                      id_agrs,
        'ood_agrs':                     ood_agrs,
        'deviations':                   deviations,
        'rel_dhs':                      rel_dhs,
        'rel_h_bs':                     rel_h_bs,
        'entropies_a':                  entropies_a,
        'entropies_b':                  entropies_b,
        'seeds_used':                   seeds_used,
        'step2_statuses':               step2_statuses,
        'candidate_seeds':              candidate_seeds,
        'on_line_seeds':                on_line_seeds,
        'candidate_agreement':          candidate_agreement,
        'online_agreement':             online_agreement,
        'genuine_escape':               genuine_escape,
        'dg_verdict':                   dg_verdict,
        'n_candidates':                 n_candidates,
        'n_on_line':                    n_on_line,
        'disagreement_rate':            disagreement_rate,
        'n_pairs':                      len(id_agrs),
        'n_valid_seeds':                len(valid_seeds),
        'algo_a':                       algo_a,
        'algo_b':                       algo_b,
        'max_entropy':                  max_entropy,
        'n_classes':                    n_classes,
        'cross_line_r_threshold':           cross_line_r_threshold,
        'disagreement_rate_threshold':      disagreement_rate_threshold,
    }

    return results


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(results, dataset_name, test_env_name):
    print(f"\n{'='*80}")
    print(f"  Diagnostic summary — {dataset_name} (test: {test_env_name})")
    print(f"{'='*80}")

    if results is None:
        print("  No results")
        return

    erm    = results['erm_line']
    cross  = results['cross_line']
    algo_a = results['algo_a']
    algo_b = results['algo_b']
    cand   = results['candidate_agreement']
    onl    = results['online_agreement']

    print(f"  {algo_a}-{algo_a} R:                    {erm['R']:+.3f}  "
          f"(ood_median={erm['ood_median']:.3f}, n_pairs={erm['n_pairs']})")
    print(f"  {algo_a}-{algo_b} R (cross-pair line):  {cross['R']:+.3f}  "
          f"(ood_median={cross['ood_median']:.3f}, n_pairs={cross['n_pairs']})")

    r_thresh = results['cross_line_r_threshold']
    cross_symbol = '✗ misspecified' if results['cross_line_misspecified'] else '✓ well-specified'
    print(f"  {algo_a}-{algo_b} R >= {r_thresh} -> {cross_symbol}")

    print(f"\n  {algo_b}-{algo_b} OOD agr (candidates): " +
          (f"{cand:.3f}" if cand is not None else "N/A"))
    print(f"  {algo_b}-{algo_b} OOD agr (on line):    " +
          (f"{onl:.3f}" if onl is not None else "N/A"))
    print(f"  {algo_a}-{algo_a} OOD median (ref):      {erm['ood_median']:.3f}")

    print(f"\n  Candidates (step 2): {results['n_candidates']}")
    print(f"  On line   (step 2):  {results['n_on_line']}")

    if results['genuine_escape'] is None:
        print(f"\n  DG verdict: INCONCLUSIVE "
              f"(only {results['n_candidates']} candidate(s) after entropy filtering)")
    else:
        dr      = results['disagreement_rate']
        dr_thr  = results['disagreement_rate_threshold']
        symbol  = '✓' if results['genuine_escape'] else '✗'
        print(f"\n  Disagreement rate: {dr:.3f} (threshold={dr_thr})")
        print(f"  DG verdict: {results['dg_verdict']} {symbol}")

    print(f"\n  {'─'*50}")
    print(f"  SUMMARY")
    print(f"  {'─'*50}")
    print(f"  Misspecified ({algo_a}-{algo_b} R >= {r_thresh}): "
          f"{results['cross_line_misspecified']}")
    if results['genuine_escape'] is not None:
        print(f"  {algo_b} useful (disagreement_rate > "
              f"{results['disagreement_rate_threshold']}): {results['genuine_escape']}")
    else:
        print(f"  {algo_b} useful: inconclusive")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_cross_algorithm(results, dataset_name, test_env_name, save_path=None):
    """
    Scatter plot: ID agreement (probit) vs OOD agreement (probit).
      gray cloud + dashed line   : ERM-ERM reference pairs
      orange cloud + dotted line : algo_a-algo_b ALL cross pairs
      blue circles               : algo_a-algo_b same-seed 'on line'
      red triangles              : algo_a-algo_b same-seed 'candidate'
    """
    erm   = results['erm_line']
    cross = results['cross_line']
    algo_a, algo_b = results['algo_a'], results['algo_b']
    eps = 1e-6

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # ERM-ERM reference cloud + line
    erm_id_p  = probit(np.clip(np.array(erm['id_agrs']),  eps, 1 - eps))
    erm_ood_p = probit(np.clip(np.array(erm['ood_agrs']), eps, 1 - eps))
    ax.scatter(erm_id_p, erm_ood_p, s=15, color='lightgray', alpha=0.6, zorder=1,
               label=f'{algo_a}-{algo_a} pairs (n={len(erm_id_p)})')
    x1 = np.linspace(erm_id_p.min(), erm_id_p.max(), 100)
    ax.plot(x1, erm['slope'] * x1 + erm['intercept'],
            color='gray', linestyle='--', zorder=2,
            label=f"{algo_a}-{algo_a} line (R={erm['R']:+.3f})")

    # algo_a-algo_b cross-pair cloud + line
    cross_id_p  = probit(np.clip(np.array(cross['id_agrs']),  eps, 1 - eps))
    cross_ood_p = probit(np.clip(np.array(cross['ood_agrs']), eps, 1 - eps))
    ax.scatter(cross_id_p, cross_ood_p, s=8, color='lightsalmon', alpha=0.3, zorder=1,
               label=f'{algo_a}-{algo_b} cross pairs (n={len(cross_id_p)})')
    x2 = np.linspace(cross_id_p.min(), cross_id_p.max(), 100)
    ax.plot(x2, cross['slope'] * x2 + cross['intercept'],
            color='tab:orange', linestyle=':', zorder=2,
            label=f"{algo_a}-{algo_b} line (R={cross['R']:+.3f})")

    # algo_a-algo_b same-seed points, colored by step2 status
    id_p  = probit(np.clip(np.array(results['id_agrs']),  eps, 1 - eps))
    ood_p = probit(np.clip(np.array(results['ood_agrs']), eps, 1 - eps))
    statuses  = np.array(results['step2_statuses'])
    seeds     = results['seeds_used']
    cand_mask = statuses == 'candidate'
    line_mask = statuses == 'on line'

    ax.scatter(id_p[line_mask], ood_p[line_mask], s=70, color='tab:blue',
               marker='o', edgecolor='k', zorder=3,
               label=f'{algo_a}-{algo_b} on line (n={line_mask.sum()})')
    ax.scatter(id_p[cand_mask], ood_p[cand_mask], s=70, color='tab:red',
               marker='^', edgecolor='k', zorder=3,
               label=f'{algo_a}-{algo_b} candidates (n={cand_mask.sum()})')

    for i, seed in enumerate(seeds):
        ax.annotate(str(seed), (id_p[i], ood_p[i]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_xlabel('ID agreement (probit)')
    ax.set_ylabel('OOD agreement (probit)')

    dr_str = (f", disagreement_rate={results['disagreement_rate']:.2f}"
              if results['disagreement_rate'] is not None else "")
    ax.set_title(f'{dataset_name} ({test_env_name})\n'
                 f'{algo_a}-{algo_b} cross R={cross["R"]:+.3f}{dr_str}')

    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"\n  Plot saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_multi_algorithm(results_list, dataset_name, test_env_name, save_path=None):
    """
    Overlay multiple algo_a-algo_b cross-pair lines on one plot, one per
    algo_b, against a shared algo_a-algo_a (ERM-ERM) reference.

    results_list: list of result dicts from compute_cross_algorithm_agreement,
                   one per algo_b (all sharing the same algo_a).
    """
    eps = 1e-6
    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    algo_a = results_list[0]['algo_a']

    # ERM-ERM reference (from the first result; recomputed per-run but should
    # be very similar across algo_b choices since it only depends on algo_a)
    erm = results_list[0]['erm_line']
    erm_id_p  = probit(np.clip(np.array(erm['id_agrs']),  eps, 1 - eps))
    erm_ood_p = probit(np.clip(np.array(erm['ood_agrs']), eps, 1 - eps))
    ax.scatter(erm_id_p, erm_ood_p, s=12, color='lightgray', alpha=0.4, zorder=1,
               label=f'{algo_a}-{algo_a} pairs (n={len(erm_id_p)})')
    x1 = np.linspace(erm_id_p.min(), erm_id_p.max(), 100)
    ax.plot(x1, erm['slope'] * x1 + erm['intercept'],
            color='gray', linestyle='--', zorder=2, linewidth=2,
            label=f"{algo_a}-{algo_a} (R={erm['R']:+.3f})")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results_list), 1)))

    for res, color in zip(results_list, colors):
        cross  = res['cross_line']
        algo_b = res['algo_b']
        cross_id_p  = probit(np.clip(np.array(cross['id_agrs']),  eps, 1 - eps))
        cross_ood_p = probit(np.clip(np.array(cross['ood_agrs']), eps, 1 - eps))
        x2 = np.linspace(cross_id_p.min(), cross_id_p.max(), 100)

        dr = res['disagreement_rate']
        dr_str = f"{dr:.2f}" if dr is not None else "N/A"

        ax.plot(x2, cross['slope'] * x2 + cross['intercept'],
                color=color, linestyle='-', linewidth=2, zorder=3,
                label=f"{algo_a}-{algo_b} (R={cross['R']:+.3f}, DR={dr_str})")

    ax.set_xlabel('ID agreement (probit)')
    ax.set_ylabel('OOD agreement (probit)')
    ax.set_title(f'{dataset_name} ({test_env_name})\n{algo_a} vs DG algorithms')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"\n  Multi-algorithm plot saved to: {save_path}")
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
    parser.add_argument('--algo_b',         type=str, default=['IRM'], nargs='+',
                         help='One or more algo_b names, e.g. --algo_b IRM VREx GroupDRO')
    parser.add_argument('--dataset_name',   type=str, default='Dataset')
    parser.add_argument('--test_env_name',  type=str, default='test')
    parser.add_argument('--entropy_threshold',           type=float, default=0.9)
    parser.add_argument('--cross_line_r_threshold',      type=float, default=0.3)
    parser.add_argument('--disagreement_rate_threshold', type=float, default=0.5)
    parser.add_argument('--plot_path',      type=str,  default=None,
                         help='If set with a single algo_b, save its individual plot here.')
    parser.add_argument('--multi_plot_path', type=str, default=None,
                         help='If set with multiple algo_b values, save the overlay plot here.')
    args = parser.parse_args()

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
        )
        print_table(results, args.dataset_name, args.test_env_name)
        results_list.append(results)

        # Individual plot: only if a single algo_b was requested and --plot_path given
        if len(args.algo_b) == 1 and args.plot_path:
            plot_cross_algorithm(results, args.dataset_name, args.test_env_name,
                                  save_path=args.plot_path)

    # Multi-algorithm overlay: if multiple algo_b values, or --multi_plot_path explicitly given
    if len(results_list) > 1 or args.multi_plot_path:
        plot_multi_algorithm(results_list, args.dataset_name, args.test_env_name,
                              save_path=args.multi_plot_path)
    elif len(args.algo_b) == 1 and args.plot_path is None:
        # default: show single-algorithm plot interactively if nothing else requested
        plot_cross_algorithm(results_list[0], args.dataset_name, args.test_env_name,
                              save_path=None)