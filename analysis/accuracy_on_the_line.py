# analysis/accuracy_on_the_line.py
"""
Computes accuracy-on-the-line metrics from sweep records.

Reproduces Salaudeen et al. (2025) Table 1:
    R value (Pearson correlation of probit-transformed ID/OOD accuracies)
    slope and intercept of OOD regressed on ID
    p-value and standard error

Input: records.json from a sweep
Output: R, slope, intercept, p-value, std_error per algorithm

Usage:
    from analysis.accuracy_on_the_line import compute_rvalue
    results = compute_rvalue('results/coloredmnist/test_env2/cnn/random/records.json',
                              test_env_idx=2, n_envs=3)
"""

import json
import numpy as np
import argparse
from scipy.special import ndtri as probit
from scipy.stats import pearsonr, linregress


def compute_id_acc(record, test_env_idx, n_envs):
    """
    ID accuracy = mean out_acc across training environments.
    This is the IID selection signal.
    """
    train_accs = [
        record[f'env{i}_out_acc']
        for i in range(n_envs)
        if i != test_env_idx
    ]
    return np.mean(train_accs)


def compute_ood_acc(record, test_env_idx):
    """
    OOD accuracy = out_acc on test environment.
    """
    return record[f'env{test_env_idx}_out_acc']


def compute_rvalue(
    records_path,
    test_env_idx,
    n_envs,
    algorithms=None,
):
    """
    Compute accuracy-on-the-line metrics per algorithm.

    Parameters
    ----------
    records_path  : str: path to records.json
    test_env_idx  : int: index of test environment
    n_envs        : int: total number of environments
    algorithms    : list: of str or None — algorithms to compute (None = all)

    Returns
    -------
    results : dict — {algorithm: {R, slope, intercept, p_value, std_error,
                                   id_accs, ood_accs}}
    """
    with open(records_path) as f:
        records = json.load(f)

    # Group by algorithm and hparams_seed
    # Average over trials for each HP config
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))

    for r in records:
        algo     = r['algorithm']
        hp_seed  = r['args']['hparams_seed']
        id_acc   = compute_id_acc(r, test_env_idx, n_envs)
        ood_acc  = compute_ood_acc(r, test_env_idx)
        grouped[algo][hp_seed].append((id_acc, ood_acc))

    if algorithms is None:
        algorithms = list(grouped.keys())

    results = {}

    for algo in algorithms:
        if algo not in grouped:
            continue

        # Average over trials per HP config
        id_accs  = []
        ood_accs = []
        for hp_seed, trial_pairs in grouped[algo].items():
            id_mean  = np.mean([p[0] for p in trial_pairs])
            ood_mean = np.mean([p[1] for p in trial_pairs])
            id_accs.append(id_mean)
            ood_accs.append(ood_mean)

        id_accs  = np.array(id_accs)
        ood_accs = np.array(ood_accs)

        # Clip to avoid probit(0) or probit(1) = ±inf
        eps = 1e-6
        id_accs_clipped  = np.clip(id_accs,  eps, 1 - eps)
        ood_accs_clipped = np.clip(ood_accs, eps, 1 - eps)

        # Probit transform: matches Salaudeen et al.
        id_probit  = probit(id_accs_clipped)
        ood_probit = probit(ood_accs_clipped)

        # Pearson R
        R, p_value = pearsonr(id_probit, ood_probit)

        # Linear regression: OOD ~ slope * ID + intercept
        reg = linregress(id_probit, ood_probit)

        results[algo] = {
            'R':          float(R),
            'slope':      float(reg.slope),
            'intercept':  float(reg.intercept),
            'p_value':    float(p_value),
            'std_error':  float(reg.stderr),
            'id_accs':    id_accs.tolist(),
            'ood_accs':   ood_accs.tolist(),
            'id_probit':  id_probit.tolist(),
            'ood_probit': ood_probit.tolist(),
            'n_points':   len(id_accs),
        }

        # Well-specified if R < 0.3 (Salaudeen threshold)
        label = '✓ well-specified' if R < 0.3 else '✗ misspecified'
        print(f"  {algo:<12} R={R:+.3f}  slope={reg.slope:.3f}  "
              f"intercept={reg.intercept:.3f}  "
              f"p={p_value:.2e}  se={reg.stderr:.3f}  {label}")

    return results


def print_table(results, dataset_name, test_env_name):
    """Print results in Salaudeen Table 1 format."""
    print(f"\n{'='*75}")
    print(f"  Accuracy-on-the-line — {dataset_name} (test: {test_env_name})")
    print(f"{'='*75}")
    print(f"  {'Algorithm':<12} {'R':>8} {'<0.3?':>6} {'slope':>8} "
          f"{'intercept':>10} {'p-value':>10} {'std_err':>8}")
    print(f"  {'─'*70}")

    for algo, r in results.items():
        label = '✓' if r['R'] < 0.3 else '✗'
        print(f"  {algo:<12} {r['R']:>+8.3f} {label:>6} "
              f"{r['slope']:>8.3f} {r['intercept']:>10.3f} "
              f"{r['p_value']:>10.2e} {r['std_error']:>8.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records_path', type=str, required=True)
    parser.add_argument('--test_env_idx', type=int, required=True)
    parser.add_argument('--n_envs',       type=int, required=True)
    parser.add_argument('--dataset_name', type=str, default='Dataset')
    parser.add_argument('--test_env_name', type=str, default='test')
    args = parser.parse_args()

    print("Computing accuracy-on-the-line metrics...")
    results = compute_rvalue(
        records_path = args.records_path,
        test_env_idx = args.test_env_idx,
        n_envs       = args.n_envs,
    )
    print_table(results, args.dataset_name, args.test_env_name)