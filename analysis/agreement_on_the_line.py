# analysis/agreement_on_the_line.py
"""
Computes agreement-on-the-line metrics from saved prediction vectors.

For each pair of models (i, j) trained with the same algorithm but
different HP configs:
    ID agreement  = fraction of examples where model_i and model_j
                    predict the same class on training env val splits
    OOD agreement = fraction of examples where model_i and model_j
                    predict the same class on test env

Plots (ID agreement, OOD agreement) across all HP pairs and computes
Pearson R — should match sign of R from accuracy_on_the_line.py.

Usage:
    python analysis/agreement_on_the_line.py \
        --records_path results/coloredmnist/test_env2/cnn/random/records.json \
        --preds_dir    results/coloredmnist/test_env2/cnn/random/models \
        --test_env_idx 2 \
        --n_envs       3
"""

import json
import os
import sys
import argparse
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_predictions, compute_agreement, get_all_trials, fit_line


def get_hp_mean_predictions(preds_dir, algorithm, hparams_seed,
                             n_trials, test_env_idx):
    """
    Load predictions for all trials of a HP config.
    Returns list of prediction arrays, one per trial.
    """
    return get_all_trials(preds_dir, algorithm, hparams_seed,
                          n_trials, test_env_idx, load_predictions)


# ID agreement: average agreement on training env val splits
#
# NOTE: this deliberately does NOT use utils.get_id_agr — that function
# pairs trials by matching index (trial_i == trial_j) only, whereas this
# averages over the full trial_i x trial_j cross product below. The two
# are different aggregations, not interchangeable.

def compute_id_agreement(preds_dir, algorithm, seed_i, seed_j,
                          n_trials, test_env_idx, n_envs):
    """
    True ID agreement — mean prediction agreement across training env
    val splits (out splits of all non-test environments).
    """
    id_agr_vals = []
    for env_idx in range(n_envs):
        if env_idx == test_env_idx:
            continue
        for trial_i in range(n_trials):
            for trial_j in range(n_trials):
                pi = load_predictions(preds_dir, algorithm,
                                      seed_i, trial_i, env_idx)
                pj = load_predictions(preds_dir, algorithm,
                                      seed_j, trial_j, env_idx)
                if pi is not None and pj is not None:
                    id_agr_vals.append(compute_agreement(pi, pj))

    if not id_agr_vals:
        return None
    return float(np.mean(id_agr_vals))



def compute_agreement_on_line(
    records_path,
    preds_dir,
    test_env_idx,
    n_envs,
    n_trials=3,
    algorithms=None,
):
    """
    Compute agreement-on-the-line for each algorithm.

    For each pair of HP configs (i, j) from the same algorithm:
        - OOD agreement: direct from saved prediction vectors
        - ID agreement: proxy from training env accuracy similarity

    Returns dict of results per algorithm.
    """
    with open(records_path) as f:
        records = json.load(f)

    # Get unique algorithms and HP seeds
    if algorithms is None:
        algorithms = list(set(r['algorithm'] for r in records))

    hp_seeds_per_algo = {}
    for algo in algorithms:
        seeds = list(set(
            r['args']['hparams_seed']
            for r in records
            if r['algorithm'] == algo
        ))
        hp_seeds_per_algo[algo] = sorted(seeds)

    results = {}

    for algo in algorithms:
        seeds    = hp_seeds_per_algo[algo]
        id_agrs  = []
        ood_agrs = []

        # All pairs (i, j) with i < j
        for seed_i, seed_j in combinations(seeds, 2):

            # OOD agreement: from saved prediction vectors
            # Average over trials for each HP config
            preds_i_list = get_hp_mean_predictions(
                preds_dir, algo, seed_i, n_trials, test_env_idx)
            preds_j_list = get_hp_mean_predictions(
                preds_dir, algo, seed_j, n_trials, test_env_idx)

            if not preds_i_list or not preds_j_list:
                continue

            # Average agreement across all trial combinations
            ood_agr_vals = []
            for pi in preds_i_list:
                for pj in preds_j_list:
                    ood_agr_vals.append(compute_agreement(pi, pj))
            ood_agr = float(np.mean(ood_agr_vals))

            # ID agreement proxy from records
            id_agr = compute_id_agreement(preds_dir, algo, seed_i, seed_j,n_trials, test_env_idx, n_envs)

            if id_agr is None:
                continue

            id_agrs.append(id_agr)
            ood_agrs.append(ood_agr)

        if len(id_agrs) < 3:
            print(f"  {algo}: not enough pairs ({len(id_agrs)})")
            continue

        line = fit_line(id_agrs, ood_agrs)
        results[algo] = {
            'R':          line['R'],
            'slope':      line['slope'],
            'intercept':  line['intercept'],
            'p_value':    line['p_value'],
            'std_error':  line['std_error'],
            'id_agrs':    line['id_agrs'],
            'ood_agrs':   line['ood_agrs'],
            'n_pairs':    line['n_pairs'],
        }

        R = line['R']
        label = '✓ well-specified' if R < 0.3 else '✗ misspecified'
        print(f"  {algo:<12} R={R:+.3f}  slope={line['slope']:.3f}  "
              f"p={line['p_value']:.2e}  n_pairs={line['n_pairs']}  {label}")

    return results


def print_table(results, dataset_name, test_env_name):
    print(f"\n{'='*75}")
    print(f"  Agreement-on-the-line — {dataset_name} (test: {test_env_name})")
    print(f"{'='*75}")
    print(f"  {'Algorithm':<12} {'R':>8} {'<0.3?':>6} {'slope':>8} "
          f"{'p-value':>10} {'n_pairs':>8}")
    print(f"  {'─'*60}")

    for algo, r in results.items():
        label = '✓' if r['R'] < 0.3 else '✗'
        print(f"  {algo:<12} {r['R']:>+8.3f} {label:>6} "
              f"{r['slope']:>8.3f} {r['p_value']:>10.2e} "
              f"{r['n_pairs']:>8}")


# Main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records_path',  type=str, required=True)
    parser.add_argument('--preds_dir',     type=str, required=True)
    parser.add_argument('--test_env_idx',  type=int, required=True)
    parser.add_argument('--n_envs',        type=int, required=True)
    parser.add_argument('--n_trials',      type=int, default=3)
    parser.add_argument('--dataset_name',  type=str, default='Dataset')
    parser.add_argument('--test_env_name', type=str, default='test')
    args = parser.parse_args()

    print("Computing agreement-on-the-line metrics...")
    results = compute_agreement_on_line(
        records_path = args.records_path,
        preds_dir    = args.preds_dir,
        test_env_idx = args.test_env_idx,
        n_envs       = args.n_envs,
        n_trials     = args.n_trials,
    )
    print_table(results, args.dataset_name, args.test_env_name)