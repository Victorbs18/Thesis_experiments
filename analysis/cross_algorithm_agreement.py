# analysis/cross_algorithm_agreement.py
"""
Cross-algorithm agreement diagnostic.

For each matched pair (ERM_i, IRM_i) with the same hparams_seed:
    ID agreement  = fraction of examples where ERM_i and IRM_i
                    predict the same class on training env val splits
    OOD agreement = fraction of examples where ERM_i and IRM_i
                    predict the same class on test env

Hypothesis:
    Well-specified dataset: ERM learns spurious features, IRM learns
    invariant features: LOW OOD

    Misspecified dataset: both algorithms learn same features:  OOD
    agreement tracks ID agreement: positive R

R < 0.3 → well-specified ✓
R > 0.3 → misspecified ✗

Usage:
    python analysis/cross_algorithm_agreement.py \
        --preds_dir    results/coloredmnist/test_env2/cnn/random/models \
        --test_env_idx 2 \
        --n_envs       3 \
        --n_hparams    20 \
        --n_trials     3 \
        --dataset_name ColoredMNIST \
        --test_env_name "-90% (env2)"
"""

import os
import argparse
import numpy as np
from scipy.special import ndtri as probit
from scipy.stats import pearsonr, linregress


# Prediction loading

def load_predictions(preds_dir, algorithm, hparams_seed, trial_seed, env_idx):
    """Load saved prediction vector for one model and environment."""
    fname = (
        f"{algorithm}"
        f"_hpseed{hparams_seed}"
        f"_trial{trial_seed}"
        f"_env{env_idx}_preds.npy"
    )
    path = os.path.join(preds_dir, fname)
    if not os.path.exists(path):
        return None
    return np.load(path)


def compute_agreement(preds_i, preds_j):
    """Fraction of examples where two models predict the same class."""
    return float(np.mean(preds_i == preds_j))


def get_mean_predictions(preds_dir, algorithm, hparams_seed, n_trials, env_idx):
    """Load predictions for all trials of one HP config on one env."""
    preds = []
    for trial in range(n_trials):
        p = load_predictions(preds_dir, algorithm, hparams_seed, trial, env_idx)
        if p is not None:
            preds.append(p)
    return preds


# Cross-algorithm agreement

def compute_cross_algorithm_agreement(
    preds_dir,
    test_env_idx,
    n_envs,
    n_hparams,
    n_trials=3,
    algo_a='ERM',
    algo_b='IRM',
):
    """
    Compute cross-algorithm agreement for matched HP pairs.

    For each hparams_seed i:
        - Load all trials of algo_a with seed i
        - Load all trials of algo_b with seed i
        - Compute ID agreement (mean over training envs)
        - Compute OOD agreement (test env)

    Returns dict with per-seed results and overall R value.
    """
    id_agrs  = []
    ood_agrs = []
    seeds_used = []

    for seed in range(n_hparams):

        # OOD agreement: test env
        preds_a_ood = get_mean_predictions(
            preds_dir, algo_a, seed, n_trials, test_env_idx)
        preds_b_ood = get_mean_predictions(
            preds_dir, algo_b, seed, n_trials, test_env_idx)

        if not preds_a_ood or not preds_b_ood:
            continue

        ood_agr_vals = [
            compute_agreement(pa, pb)
            for pa in preds_a_ood
            for pb in preds_b_ood
        ]
        ood_agr = float(np.mean(ood_agr_vals))

        # ID agreement: mean over training envs
        id_agr_vals = []
        for env_idx in range(n_envs):
            if env_idx == test_env_idx:
                continue
            preds_a_id = get_mean_predictions(
                preds_dir, algo_a, seed, n_trials, env_idx)
            preds_b_id = get_mean_predictions(
                preds_dir, algo_b, seed, n_trials, env_idx)
            if not preds_a_id or not preds_b_id:
                continue
            for pa in preds_a_id:
                for pb in preds_b_id:
                    id_agr_vals.append(compute_agreement(pa, pb))

        if not id_agr_vals:
            continue

        id_agr = float(np.mean(id_agr_vals))

        id_agrs.append(id_agr)
        ood_agrs.append(ood_agr)
        seeds_used.append(seed)

        print(f"  seed={seed:2d} | "
              f"ID_agr={id_agr:.3f} | "
              f"OOD_agr={ood_agr:.3f} | "
              f"OOD_dis={1-ood_agr:.3f}")

    if len(id_agrs) < 3:
        print(f"  Not enough pairs ({len(id_agrs)})")
        return None

    id_agrs  = np.array(id_agrs)
    ood_agrs = np.array(ood_agrs)

    # Probit transform
    eps        = 1e-6
    id_probit  = probit(np.clip(id_agrs,  eps, 1 - eps))
    ood_probit = probit(np.clip(ood_agrs, eps, 1 - eps))

    R, p_value = pearsonr(id_probit, ood_probit)
    reg        = linregress(id_probit, ood_probit)

    # Escape rate: fraction of seeds where OOD agreement is low
    # (IRM found a different solution from ERM)
    escape_threshold = 0.7  # models agree less than 70% OOD → escaped
    escape_rate = float(np.mean(ood_agrs < escape_threshold))

    results = {
        'R':            float(R),
        'slope':        float(reg.slope),
        'intercept':    float(reg.intercept),
        'p_value':      float(p_value),
        'std_error':    float(reg.stderr),
        'id_agrs':      id_agrs.tolist(),
        'ood_agrs':     ood_agrs.tolist(),
        'seeds_used':   seeds_used,
        'escape_rate':  escape_rate,
        'n_pairs':      len(id_agrs),
        'algo_a':       algo_a,
        'algo_b':       algo_b,
    }

    label = '✓ well-specified' if R < 0.3 else '✗ misspecified'
    print(f"\n  {algo_a} vs {algo_b}: R={R:+.3f}  slope={reg.slope:.3f}  "
          f"p={p_value:.2e}  escape_rate={escape_rate:.2f}  {label}")

    return results


def print_table(results, dataset_name, test_env_name):
    print(f"\n{'='*75}")
    print(f"  Cross-algorithm agreement — {dataset_name} (test: {test_env_name})")
    print(f"{'='*75}")
    print(f"  {'Pair':<15} {'R':>8} {'<0.3?':>6} {'slope':>8} "
          f"{'p-value':>10} {'escape_rate':>12} {'n_pairs':>8}")
    print(f"  {'─'*70}")

    if results is None:
        print("  No results")
        return

    label = '✓' if results['R'] < 0.3 else '✗'
    pair  = f"{results['algo_a']} vs {results['algo_b']}"
    print(f"  {pair:<15} {results['R']:>+8.3f} {label:>6} "
          f"{results['slope']:>8.3f} {results['p_value']:>10.2e} "
          f"{results['escape_rate']:>12.2f} {results['n_pairs']:>8}")


# Entry point

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_dir',     type=str, required=True)
    parser.add_argument('--test_env_idx',  type=int, required=True)
    parser.add_argument('--n_envs',        type=int, required=True)
    parser.add_argument('--n_hparams',     type=int, default=20)
    parser.add_argument('--n_trials',      type=int, default=3)
    parser.add_argument('--algo_a',        type=str, default='ERM')
    parser.add_argument('--algo_b',        type=str, default='IRM')
    parser.add_argument('--dataset_name',  type=str, default='Dataset')
    parser.add_argument('--test_env_name', type=str, default='test')
    args = parser.parse_args()

    print(f"Computing cross-algorithm agreement "
          f"({args.algo_a} vs {args.algo_b})...")
    results = compute_cross_algorithm_agreement(
        preds_dir     = args.preds_dir,
        test_env_idx  = args.test_env_idx,
        n_envs        = args.n_envs,
        n_hparams     = args.n_hparams,
        n_trials      = args.n_trials,
        algo_a        = args.algo_a,
        algo_b        = args.algo_b,
    )
    print_table(results, args.dataset_name, args.test_env_name)