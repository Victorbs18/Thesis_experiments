# analysis/accuracy_on_the_line.py
"""
Computes accuracy-on-the-line metrics from sweep records, and optionally
plots the (ID_acc, OOD_acc) scatter with the fitted regression line.

Reproduces Salaudeen et al. (2025) Table 1:
    R value (Pearson correlation of probit-transformed ID/OOD accuracies)
    slope and intercept of OOD regressed on ID
    p-value and standard error

Input: records.json from a sweep
Output: R, slope, intercept, p-value, std_error per algorithm (+ optional PNG)

Usage:
    from analysis.accuracy_on_the_line import compute_rvalue
    results = compute_rvalue('results/coloredmnist/test_env2/cnn/random/records.json',
                              test_env_idx=2, n_envs=3)

    python analysis/accuracy_on_the_line.py \
        --records_path results/pacs/test_env0/clip/random/records.json \
        --test_env_idx 0 --n_envs 4 \
        --dataset_name "PACS (CLIP)" --test_env_name "Art (env0)" \
        --plot_path acl_pacs_clip.png
"""

import json
import numpy as np
import argparse
from scipy.special import ndtri as probit
from scipy.stats import pearsonr, linregress
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    verbose=True,
):
    """
    Compute accuracy-on-the-line metrics per algorithm.

    Parameters
    ----------
    records_path  : str: path to records.json
    test_env_idx  : int: index of test environment
    n_envs        : int: total number of environments
    algorithms    : list: of str or None — algorithms to compute (None = all)
    verbose       : bool: print per-algorithm line as it's computed

    Returns
    -------
    results : dict — {algorithm: {R, slope, intercept, p_value, std_error,
                                   id_accs, ood_accs, hp_seeds}}
    """
    with open(records_path) as f:
        records = json.load(f)

    # Group by algorithm and hparams_seed
    # Average over trials for each HP config
    
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
        hp_seeds = []
        id_accs  = []
        ood_accs = []
        for hp_seed, trial_pairs in sorted(grouped[algo].items()):
            id_mean  = np.mean([p[0] for p in trial_pairs])
            ood_mean = np.mean([p[1] for p in trial_pairs])
            hp_seeds.append(hp_seed)
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
            'hp_seeds':   hp_seeds,
            'id_accs':    id_accs.tolist(),
            'ood_accs':   ood_accs.tolist(),
            'id_probit':  id_probit.tolist(),
            'ood_probit': ood_probit.tolist(),
            'n_points':   len(id_accs),
        }

        if verbose:
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


def plot_acl(results, dataset_name, test_env_name, plot_path,
             space='accuracy'):
    """
    Single-axes scatter plot of ID vs OOD accuracy (or probit-space),
    overlaying all algorithms with their fitted regression lines, so
    slopes/intercepts can be compared directly.

    Parameters
    ----------
    results       : dict returned by compute_rvalue
    dataset_name  : str, used in the title
    test_env_name : str, used in the title
    plot_path     : str, output PNG path
    space         : 'accuracy' (raw acc, default) or 'probit' (probit-
                    transformed space, where the regression line is linear)
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    color_cycle = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))
    colors = {'ERM': '#1f77b4', 'IRM': '#d62728'}

    # Collect combined accuracy range across all algorithms for adaptive ticks
    all_accs = np.concatenate([
        np.concatenate([np.array(r['id_accs']), np.array(r['ood_accs'])])
        for r in results.values()
    ])

    for (algo, r), c in zip(results.items(), color_cycle):
        color = colors.get(algo, c)

        if space == 'probit':
            x = np.array(r['id_probit'])
            y = np.array(r['ood_probit'])
        else:
            x = np.array(r['id_accs'])
            y = np.array(r['ood_accs'])

        ax.scatter(x, y, c=[color], s=50, alpha=0.8,
                   edgecolors='black', linewidths=0.5, zorder=3,
                   label=f'{algo} (R={r["R"]:+.3f})')

        # Fitted line — always computed in probit space; convert if needed
        x_probit = np.array(r['id_probit'])
        x_line_probit = np.linspace(x_probit.min(), x_probit.max(), 100)
        y_line_probit = r['slope'] * x_line_probit + r['intercept']

        if space == 'probit':
            x_line, y_line = x_line_probit, y_line_probit
        else:
            from scipy.special import ndtr as inv_probit
            x_line = inv_probit(x_line_probit)
            y_line = inv_probit(y_line_probit)

        ax.plot(x_line, y_line, '--', color=color, linewidth=2, zorder=2)

    if space == 'probit':
        xlabel = 'ID accuracy (%)'
        ylabel = 'OOD accuracy (%)'

        lo, hi = all_accs.min() * 100, all_accs.max() * 100
        pad = (hi - lo) * 0.1
        lo, hi = max(lo - pad, 1), min(hi + pad, 99)
        tick_pcts = np.linspace(lo, hi, 8)
        tick_probit = probit(tick_pcts / 100.0)

        ax.set_xticks(tick_probit)
        ax.set_xticklabels([f"{p:.0f}" for p in tick_pcts])
        ax.set_yticks(tick_probit)
        ax.set_yticklabels([f"{p:.0f}" for p in tick_pcts])
    else:
        xlabel = 'ID accuracy (mean train-env out_acc)'
        ylabel = f'OOD accuracy ({test_env_name} out_acc)'

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Accuracy-on-the-line — {dataset_name} (test: {test_env_name})')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to {plot_path}")
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records_path', type=str, required=True)
    parser.add_argument('--test_env_idx', type=int, required=True)
    parser.add_argument('--n_envs',       type=int, required=True)
    parser.add_argument('--dataset_name', type=str, default='Dataset')
    parser.add_argument('--test_env_name', type=str, default='test')
    parser.add_argument('--plot_path', type=str, default=None,
                        help='If set, save a scatter plot (ID vs OOD acc) here')
    parser.add_argument('--plot_space', type=str, default='accuracy',
                        choices=['accuracy', 'probit'],
                        help='Plot in raw accuracy space (default) or probit space')
    args = parser.parse_args()

    print("Computing accuracy-on-the-line metrics...")
    results = compute_rvalue(
        records_path = args.records_path,
        test_env_idx = args.test_env_idx,
        n_envs       = args.n_envs,
        verbose      = False,
    )
    print_table(results, args.dataset_name, args.test_env_name)

    if args.plot_path:
        plot_acl(results, args.dataset_name, args.test_env_name,
                args.plot_path, space=args.plot_space)