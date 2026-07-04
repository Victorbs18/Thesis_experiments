# analysis/irm_env_variance_check.py
"""
Population-level check (no new training runs): for ALL valid IRM seeds on a
dataset (not just candidates -- detection must work before any candidate/
on-line split is trusted), compute the standard deviation of accuracy ACROSS
TRAINING environments only (never touches the test/OOD environment or its
labels).

Purpose: DETECTION, not just candidate ranking. We want a population-level
statistic -- analogous to cross-line R -- that differs between well-specified
and misspecified benchmarks, computed independently of any prior candidate/
on-line split (which is exactly the thing under question for IRM).

Reports, across ALL valid seeds:
  - mean train_acc_std (population average)
  - correlation between train_acc_std and lambda (does higher penalty
    -> lower cross-env variance, as IRM's objective intends?)
  - correlation between train_acc_std and ood_acc (does lower variance
    predict better transfer?)
  - fraction of seeds with "low variance at non-trivial accuracy" (i.e.
    std below a threshold AND train_mean above chance level -- this
    excludes trivially-collapsed seeds where near-zero std is meaningless)

This uses only records.json -- no OOD/test labels, no new predictions,
no retraining.

Usage:
    python analysis/irm_env_variance_check.py \
        --records_path results/coloredmnist/test_env2/cnn/random/records.json \
        --test_env_idx 2 --n_envs 3 --algo IRM \
        --dataset_name ColoredMNIST --n_hparams 20 --chance_level 0.5

    python analysis/irm_env_variance_check.py \
        --records_path results/rotatedmnist/test_env5/cnn/random/records.json \
        --test_env_idx 5 --n_envs 6 --algo IRM \
        --dataset_name RotatedMNIST --n_hparams 20 --chance_level 0.1
"""

import json
import argparse
import numpy as np


def get_record_info(records, algo, seed, test_env_idx, n_envs):
    matching = [r for r in records
                if r['algorithm'] == algo
                and r['args']['hparams_seed'] == seed]
    if not matching:
        return None

    train_accs = []
    for env_idx in range(n_envs):
        if env_idx == test_env_idx:
            continue
        key = f'env{env_idx}_out_acc'
        vals = [r[key] for r in matching if key in r]
        if vals:
            train_accs.append(float(np.mean(vals)))

    ood_accs = [r[f'env{test_env_idx}_out_acc'] for r in matching
                if f'env{test_env_idx}_out_acc' in r]
    ood_acc = float(np.mean(ood_accs)) if ood_accs else None

    hp = matching[0]['hparams']
    lam = hp.get('irm_lambda', hp.get('mmd_gamma', None))

    return {
        'train_accs': train_accs,
        'train_acc_mean': float(np.mean(train_accs)) if train_accs else None,
        'train_acc_std': float(np.std(train_accs)) if len(train_accs) > 1 else None,
        'ood_acc': ood_acc,
        'lambda': lam,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records_path', type=str, required=True)
    parser.add_argument('--test_env_idx', type=int, required=True)
    parser.add_argument('--n_envs', type=int, required=True)
    parser.add_argument('--algo', type=str, default='IRM')
    parser.add_argument('--dataset_name', type=str, default='Dataset')
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--chance_level', type=float, required=True,
                        help='Chance-level accuracy for this task '
                             '(0.5 for binary CMNIST, 0.1 for 10-class RotMNIST)')
    parser.add_argument('--std_threshold', type=float, default=0.015,
                        help='Threshold below which cross-env std counts as "low"')
    args = parser.parse_args()

    with open(args.records_path) as f:
        records = json.load(f)

    print(f"\n{'='*105}")
    print(f"  Training-env accuracy variance — POPULATION LEVEL — {args.dataset_name} — {args.algo}")
    print(f"  (all valid seeds, not just candidates -- for detection, not just ranking)")
    print(f"{'='*105}")
    print(f"  {'seed':>4} | {'lambda':>10} | {'train_mean':>10} | {'train_std':>9} | "
          f"{'ood_acc':>7} | {'above_chance':>12} | {'low_std':>7}")
    print(f"  {'-'*90}")

    rows = []
    for seed in range(args.n_hparams):
        info = get_record_info(records, args.algo, seed, args.test_env_idx, args.n_envs)
        if info is None or info['train_acc_std'] is None:
            continue
        above_chance = info['train_acc_mean'] > args.chance_level * 1.5
        low_std = info['train_acc_std'] < args.std_threshold
        lam_str = f"{info['lambda']:.1f}" if info['lambda'] is not None else "—"
        ood_str = f"{info['ood_acc']:.3f}" if info['ood_acc'] is not None else "—"
        print(f"  {seed:>4} | {lam_str:>10} | {info['train_acc_mean']:>10.3f} | "
              f"{info['train_acc_std']:>9.4f} | {ood_str:>7} | "
              f"{'yes' if above_chance else 'no':>12} | {'yes' if low_std else 'no':>7}")
        rows.append({'seed': seed, 'above_chance': above_chance,
                     'low_std': low_std, **info})

    print(f"\n  {'─'*60}")
    print(f"  POPULATION-LEVEL SUMMARY (n={len(rows)} valid seeds)")
    print(f"  {'─'*60}")

    all_stds = [r['train_acc_std'] for r in rows]
    print(f"  Mean train_acc_std (all seeds):         {np.mean(all_stds):.4f}")
    print(f"  Median train_acc_std (all seeds):       {np.median(all_stds):.4f}")

    non_collapsed = [r for r in rows if r['above_chance']]
    print(f"  Seeds above chance*1.5 ({args.chance_level*1.5:.2f}):        "
          f"{len(non_collapsed)}/{len(rows)}")
    if non_collapsed:
        nc_stds = [r['train_acc_std'] for r in non_collapsed]
        print(f"  Mean train_acc_std (non-collapsed only): {np.mean(nc_stds):.4f}")

    good_seeds = [r for r in rows if r['above_chance'] and r['low_std']]
    print(f"  Seeds with LOW std AND above-chance acc: {len(good_seeds)}/{len(rows)}")
    if good_seeds:
        print(f"    -> seeds: {[r['seed'] for r in good_seeds]}")
        print(f"    -> their ood_acc: "
              f"{[round(r['ood_acc'], 3) for r in good_seeds if r['ood_acc'] is not None]}")

    # Correlations
    valid = [r for r in rows if r['ood_acc'] is not None]
    if len(valid) >= 3:
        from scipy.stats import pearsonr
        stds = [r['train_acc_std'] for r in valid]
        oods = [r['ood_acc'] for r in valid]
        lams = [r['lambda'] for r in valid if r['lambda'] is not None]
        r1, p1 = pearsonr(stds, oods)
        print(f"\n  Correlation train_acc_std vs ood_acc (ALL seeds): "
              f"R={r1:+.3f} (p={p1:.3f}, n={len(valid)})")
        if len(lams) == len(valid):
            r2, p2 = pearsonr(np.log10([max(l, 1e-3) for l in lams]), stds)
            print(f"  Correlation log(lambda) vs train_acc_std (ALL seeds): "
                  f"R={r2:+.3f} (p={p2:.3f}, n={len(valid)})")

    if non_collapsed and len(non_collapsed) >= 3:
        nc_valid = [r for r in non_collapsed if r['ood_acc'] is not None]
        if len(nc_valid) >= 3:
            stds = [r['train_acc_std'] for r in nc_valid]
            oods = [r['ood_acc'] for r in nc_valid]
            r3, p3 = pearsonr(stds, oods)
            print(f"  Correlation train_acc_std vs ood_acc (non-collapsed only): "
                  f"R={r3:+.3f} (p={p3:.3f}, n={len(nc_valid)})")


if __name__ == '__main__':
    main()