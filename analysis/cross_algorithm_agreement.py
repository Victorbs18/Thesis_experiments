# analysis/cross_algorithm_agreement.py
"""
Cross-algorithm agreement diagnostic.

Pipeline:
    Step 1: Compute ERM-ERM agreement line — reference line (label-free)
            slope, intercept, R, ood_median computed in probit space.

    Step 2: Identify candidate IRM escapes:
            rel_dH > escape_dh_min AND OOD_agr < erm_ood_median

    Step 3: Validate candidates via IRM-IRM agreement:
            Genuine escape:  IRM candidates agree WITH EACH OTHER OOD
                             (IRM-IRM OOD agr > erm_ood_median)
            Degenerate:      IRM candidates disagree WITH EACH OTHER OOD
                             (IRM-IRM OOD agr < erm_ood_median)

    Step 4: escape_rate = genuine / n_hparams
            escape_rate > 0 → well-specified ✓
            escape_rate = 0 → misspecified ✗

Usage:
    python analysis/cross_algorithm_agreement.py \
        --preds_dir    results/coloredmnist/test_env2/cnn/random/models \
        --records_path results/coloredmnist/test_env2/cnn/random/records.json \
        --test_env_idx 2 \
        --n_envs       3 \
        --n_hparams    20 \
        --n_trials     3 \
        --dataset_name ColoredMNIST \
        --test_env_name "-90% (env2)"
"""

import os
import json
import argparse
import numpy as np
from scipy.special import ndtri as probit
from scipy.special import ndtr as normal_cdf
from scipy.stats import pearsonr, linregress
from itertools import combinations


# Loading utilities

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
        'lambda':  hp.get('irm_lambda', None),
        'anneal':  hp.get('irm_penalty_anneal_iters', None),
        'bs':      hp.get('batch_size', None),
    }


# Metrics

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


# ERM-ERM reference line

def compute_erm_line(preds_dir, n_hparams, n_trials, test_env_idx, n_envs):
    """
    Fit ERM-ERM agreement line in probit space.
    OOD median used as dataset-specific agreement threshold.
    """
    id_agrs  = []
    ood_agrs = []

    for seed_i, seed_j in combinations(range(n_hparams), 2):
        id_agr  = get_id_agr(preds_dir, 'ERM', 'ERM', seed_i, seed_j,
                              n_trials, test_env_idx, n_envs)
        ood_agr = get_ood_agr(preds_dir, 'ERM', 'ERM', seed_i, seed_j,
                               n_trials, test_env_idx)
        if id_agr is not None and ood_agr is not None:
            id_agrs.append(id_agr)
            ood_agrs.append(ood_agr)

    id_agrs  = np.array(id_agrs)
    ood_agrs = np.array(ood_agrs)
    eps      = 1e-6

    id_probit  = probit(np.clip(id_agrs,  eps, 1 - eps))
    ood_probit = probit(np.clip(ood_agrs, eps, 1 - eps))

    R, p_value = pearsonr(id_probit, ood_probit)
    reg        = linregress(id_probit, ood_probit)

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


# IRM-IRM agreement among candidate seeds

def compute_irm_irm_agreement(preds_dir, candidate_seeds,
                               n_trials, test_env_idx):
    """
    Compute mean OOD agreement between all pairs of candidate IRM seeds.
    High agreement → candidates converged to same solution → genuine escape
    Low agreement  → candidates collapsed to different solutions → degenerate
    """
    if len(candidate_seeds) < 2:
        return None

    agr_vals = []
    for seed_i, seed_j in combinations(candidate_seeds, 2):
        ood_agr = get_ood_agr(preds_dir, 'IRM', 'IRM', seed_i, seed_j,
                               n_trials, test_env_idx)
        if ood_agr is not None:
            agr_vals.append(ood_agr)

    return float(np.mean(agr_vals)) if agr_vals else None


# Main computation

def compute_cross_algorithm_agreement(
    preds_dir,
    records_path,
    test_env_idx,
    n_envs,
    n_hparams,
    n_trials=3,
    algo_a='ERM',
    algo_b='IRM',
    escape_dh_min=0.1,   # rel_dH > this → IRM more uncertain than ERM
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
    print(f"  escape_dh_min={escape_dh_min}")

    # Step 1: ERM-ERM reference line
    print(f"\n  Computing ERM-ERM reference line...")
    erm_line = compute_erm_line(preds_dir, n_hparams, n_trials,
                                test_env_idx, n_envs)
    erm_ood_median = erm_line['ood_median']

    print(f"  ERM-ERM: R={erm_line['R']:+.3f}  "
          f"slope={erm_line['slope']:.3f}  "
          f"intercept={erm_line['intercept']:.3f}  "
          f"ood_median={erm_ood_median:.3f}  "
          f"n_pairs={erm_line['n_pairs']}")

    # Step 2: ERM-IRM pairs — identify candidates
    print(f"\n  Computing ERM-IRM pairs...")
    print(f"\n  {'seed':>4} | "
          f"{'ID_agr':>6} | {'OOD_agr':>7} | "
          f"{'pred_OOD':>8} | {'|dev|':>6} | "
          f"{'rel_h_b':>7} | {'dH/max':>7} | "
          f"{'acc_ERM':>8} | {'acc_IRM':>8} | "
          f"{'H(ERM)':>7} | {'H(IRM)':>7} | "
          f"{'lambda':>10} | {'anneal':>6} | step2_status")
    print(f"  {'-'*135}")

    id_agrs     = []
    ood_agrs    = []
    deviations  = []
    rel_dhs     = []
    rel_h_bs    = []
    entropies_a = []
    entropies_b = []
    seeds_used  = []
    step2_statuses = []

    eps = 1e-6

    for seed in range(n_hparams):

        id_agr  = get_id_agr(preds_dir, algo_a, algo_b, seed, seed,
                              n_trials, test_env_idx, n_envs)
        ood_agr = get_ood_agr(preds_dir, algo_a, algo_b, seed, seed,
                               n_trials, test_env_idx)
        if id_agr is None or ood_agr is None:
            continue

        # Predicted OOD agreement from ERM line
        id_probit_val   = probit(np.clip(id_agr, eps, 1 - eps))
        pred_ood_probit = (erm_line['slope'] * id_probit_val
                           + erm_line['intercept'])
        pred_ood_agr    = float(normal_cdf(pred_ood_probit))
        deviation       = abs(ood_agr - pred_ood_agr)

        # Entropy
        probs_a   = get_all_trials(preds_dir, algo_a, seed,
                                   n_trials, test_env_idx, load_probs)
        probs_b   = get_all_trials(preds_dir, algo_b, seed,
                                   n_trials, test_env_idx, load_probs)
        entropy_a = float(np.mean([compute_entropy(p) for p in probs_a])) \
                    if probs_a else None
        entropy_b = float(np.mean([compute_entropy(p) for p in probs_b])) \
                    if probs_b else None

        rel_h_b = entropy_b / max_entropy \
                  if entropy_b is not None else None
        rel_dH  = (entropy_b - entropy_a) / max_entropy \
                  if entropy_b is not None and entropy_a is not None else None

        # Step 2 classification — candidate or not
        if rel_dH is None:
            step2 = 'unknown'
        elif rel_dH > escape_dh_min and ood_agr < erm_ood_median:
            step2 = 'candidate'
        elif ood_agr >= erm_ood_median:
            step2 = 'on line'
        else:
            step2 = 'ambiguous'

        # Record info
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

        lambda_str = f"{lambda_b:10.1f}" if lambda_b else f"{'—':>10}"
        anneal_str = f"{anneal_b:6d}"    if anneal_b else f"{'—':>6}"
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

    # Step 3: IRM-IRM agreement among candidates
    candidate_seeds = [seeds_used[i] for i, s in enumerate(step2_statuses)
                       if s == 'candidate']
    on_line_seeds   = [seeds_used[i] for i, s in enumerate(step2_statuses)
                       if s == 'on line']

    print(f"\n  Candidates: {candidate_seeds}")
    print(f"  On line:    {on_line_seeds}")

    irm_irm_candidates = compute_irm_irm_agreement(
        preds_dir, candidate_seeds, n_trials, test_env_idx)
    irm_irm_online = compute_irm_irm_agreement(
        preds_dir, on_line_seeds, n_trials, test_env_idx)

    print(f"\n  Step 3 — IRM-IRM agreement among candidates:")
    print(f"  IRM-IRM OOD agr (candidates): "
          f"{irm_irm_candidates:.3f}" if irm_irm_candidates else "  N/A")
    print(f"  IRM-IRM OOD agr (on line):    "
          f"{irm_irm_online:.3f}" if irm_irm_online else "  N/A")
    print(f"  ERM-ERM OOD median (reference): {erm_ood_median:.3f}")

    # Verdict
    if irm_irm_candidates is not None:
        genuine_escape = irm_irm_candidates > erm_ood_median * 0.5
    else:
        genuine_escape = False

    n_candidates = len(candidate_seeds)
    n_on_line    = len(on_line_seeds)
    escape_rate  = n_candidates / n_hparams if genuine_escape else 0.0

    results = {
        'erm_line':             erm_line,
        'id_agrs':              id_agrs,
        'ood_agrs':             ood_agrs,
        'deviations':           deviations,
        'rel_dhs':              rel_dhs,
        'rel_h_bs':             rel_h_bs,
        'entropies_a':          entropies_a,
        'entropies_b':          entropies_b,
        'seeds_used':           seeds_used,
        'step2_statuses':       step2_statuses,
        'candidate_seeds':      candidate_seeds,
        'on_line_seeds':        on_line_seeds,
        'irm_irm_candidates':   irm_irm_candidates,
        'irm_irm_online':       irm_irm_online,
        'genuine_escape':       genuine_escape,
        'n_candidates':         n_candidates,
        'n_on_line':            n_on_line,
        'escape_rate':          escape_rate,
        'n_pairs':              len(id_agrs),
        'algo_a':               algo_a,
        'algo_b':               algo_b,
        'max_entropy':          max_entropy,
        'n_classes':            n_classes,
    }

    return results


# Print table

def print_table(results, dataset_name, test_env_name):
    print(f"\n{'='*80}")
    print(f"  Diagnostic summary — {dataset_name} (test: {test_env_name})")
    print(f"{'='*80}")

    if results is None:
        print("  No results")
        return

    erm    = results['erm_line']
    irm_c  = results['irm_irm_candidates']
    irm_ol = results['irm_irm_online']

    print(f"  ERM-ERM OOD median:           {erm['ood_median']:.3f}")
    print(f"  IRM-IRM OOD agr (candidates): "
          f"{irm_c:.3f}" if irm_c is not None else "  N/A")
    print(f"  IRM-IRM OOD agr (on line):    "
          f"{irm_ol:.3f}" if irm_ol is not None else "  N/A")

    print(f"\n  Candidates (step 2): {results['n_candidates']}")
    print(f"  On line (step 2):    {results['n_on_line']}")

    verdict = 'well-specified ✓' if results['genuine_escape'] \
              else 'misspecified ✗'
    print(f"\n  Genuine escape: {results['genuine_escape']}  → {verdict}")
    print(f"  Escape rate: {results['escape_rate']:.2f}")


# Entry point

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_dir',      type=str, required=True)
    parser.add_argument('--records_path',   type=str, required=True)
    parser.add_argument('--test_env_idx',   type=int, required=True)
    parser.add_argument('--n_envs',         type=int, required=True)
    parser.add_argument('--n_hparams',      type=int, default=20)
    parser.add_argument('--n_trials',       type=int, default=3)
    parser.add_argument('--algo_a',         type=str, default='ERM')
    parser.add_argument('--algo_b',         type=str, default='IRM')
    parser.add_argument('--dataset_name',   type=str, default='Dataset')
    parser.add_argument('--test_env_name',  type=str, default='test')
    parser.add_argument('--escape_dh_min',  type=float, default=0.1)
    args = parser.parse_args()

    print(f"Computing cross-algorithm agreement "
          f"({args.algo_a} vs {args.algo_b})...")
    results = compute_cross_algorithm_agreement(
        preds_dir     = args.preds_dir,
        records_path  = args.records_path,
        test_env_idx  = args.test_env_idx,
        n_envs        = args.n_envs,
        n_hparams     = args.n_hparams,
        n_trials      = args.n_trials,
        algo_a        = args.algo_a,
        algo_b        = args.algo_b,
        escape_dh_min = args.escape_dh_min,
    )
    print_table(results, args.dataset_name, args.test_env_name)