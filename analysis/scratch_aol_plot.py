# Scratch script: reproduce an Agreement-on-the-Line style figure (Baek et al.)
# overlaying accuracy-on-the-line, general agreement-on-the-line, and our
# ERM-anchored Cross-R, all in probit space, for a given CONFIG.
#
# Not part of the analysis/ package -- exploratory/diagnostic only.

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import ndtri as probit
from utils import (CONFIGS, N_HPARAMS, N_TRIALS, load_probs, load_preds,
                    compute_entropy, fit_line)

EPS = 1e-3


def get_valid_seeds(preds_dir, algo, test_env, threshold=0.9):
    entropies = {}
    for seed in range(N_HPARAMS):
        hs = [compute_entropy(load_probs(preds_dir, algo, seed, trial, test_env))
              for trial in range(N_TRIALS)
              if load_probs(preds_dir, algo, seed, trial, test_env) is not None]
        if hs:
            entropies[seed] = sum(hs) / len(hs)
    if not entropies:
        return []
    max_h = max(max(entropies.values()), 1e-6)
    return [s for s, h in entropies.items() if (h / max_h) < threshold]


def id_ood_acc(records, algo, seed, test_env, n_envs):
    recs = [r for r in records if r['algorithm'] == algo and r['args']['hparams_seed'] == seed]
    if not recs:
        return None, None
    id_accs, ood_accs = [], []
    for r in recs:
        oa = r.get(f'env{test_env}_out_acc')
        if oa is not None:
            ood_accs.append(oa)
        tr = [r.get(f'env{e}_out_acc') for e in range(n_envs) if e != test_env]
        tr = [v for v in tr if v is not None]
        if tr:
            id_accs.append(float(np.mean(tr)))
    if not id_accs or not ood_accs:
        return None, None
    return float(np.mean(id_accs)), float(np.mean(ood_accs))


def id_ood_agr(preds_dir, test_env, n_envs, algo_a, seed_a, algo_b, seed_b):
    id_vals = []
    for trial in range(N_TRIALS):
        env_agrs = []
        for e in range(n_envs):
            if e == test_env:
                continue
            pa = load_preds(preds_dir, algo_a, seed_a, trial, e)
            pb = load_preds(preds_dir, algo_b, seed_b, trial, e)
            if pa is not None and pb is not None:
                env_agrs.append(float((pa == pb).mean()))
        if env_agrs:
            id_vals.append(float(np.mean(env_agrs)))
    if not id_vals:
        return None, None
    ood_vals = []
    for trial in range(N_TRIALS):
        pa = load_preds(preds_dir, algo_a, seed_a, trial, test_env)
        pb = load_preds(preds_dir, algo_b, seed_b, trial, test_env)
        if pa is not None and pb is not None:
            ood_vals.append(float((pa == pb).mean()))
    if not ood_vals:
        return None, None
    return float(np.mean(id_vals)), float(np.mean(ood_vals))


def to_probit(vals):
    return probit(np.clip(np.asarray(vals, dtype=float), EPS, 1 - EPS))


def plot_cloud(ax, xs, ys, color, label):
    if len(xs) < 2:
        return None
    xp, yp = to_probit(xs), to_probit(ys)
    ax.scatter(xp, yp, s=16, alpha=0.45, color=color, label=f'{label} (n={len(xs)})')
    z = np.polyfit(xp, yp, 1)
    xr = np.linspace(xp.min(), xp.max(), 10)
    ax.plot(xr, np.polyval(z, xr), color=color, linewidth=2)
    return fit_line(xs, ys)['R']


def build_figure(cfg, out_path, n_agr_pairs=150, seed=0):
    preds_dir, test_env, n_envs = cfg['preds_dir'], cfg['test_env_idx'], cfg['n_envs']
    records = json.load(open(cfg['records_path']))
    available = sorted(set(r['algorithm'] for r in records))
    valid = {a: get_valid_seeds(preds_dir, a, test_env) for a in available}

    # 1. Accuracy-on-the-line: one point per (algo, seed), pooled across ALL algorithms
    acc_x, acc_y = [], []
    for a in available:
        for s in valid[a]:
            x, y = id_ood_acc(records, a, s, test_env, n_envs)
            if x is not None:
                acc_x.append(x); acc_y.append(y)

    # 2. General agreement-on-the-line: random pairs of ANY two models (any algo x any algo)
    all_models = [(a, s) for a in available for s in valid[a]]
    rng = random.Random(seed)
    pairs, seen = [], set()
    max_pairs = len(all_models) * (len(all_models) - 1) // 2
    n_pairs = min(n_agr_pairs, max_pairs)
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 30 and len(all_models) >= 2:
        attempts += 1
        i, j = rng.sample(range(len(all_models)), 2)
        key = tuple(sorted([i, j]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((all_models[i], all_models[j]))
    agr_x, agr_y = [], []
    for (a1, s1), (a2, s2) in pairs:
        x, y = id_ood_agr(preds_dir, test_env, n_envs, a1, s1, a2, s2)
        if x is not None:
            agr_x.append(x); agr_y.append(y)

    # 3. Our Cross-R: ERM-anchored pairs only, pooled across all DG algorithms
    cr_x, cr_y = [], []
    if 'ERM' in valid:
        for a in available:
            if a == 'ERM':
                continue
            for sa in valid['ERM']:
                for sb in valid[a]:
                    x, y = id_ood_agr(preds_dir, test_env, n_envs, 'ERM', sa, a, sb)
                    if x is not None:
                        cr_x.append(x); cr_y.append(y)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    r_acc = plot_cloud(ax, acc_x, acc_y, 'tab:blue', 'Accuracy-on-the-line')
    r_agr = plot_cloud(ax, agr_x, agr_y, 'violet', 'Agreement-on-the-line (general)')
    r_cr = plot_cloud(ax, cr_x, cr_y, 'tab:green', 'Cross-R (ERM-anchored)')

    ticks_pct = [10, 30, 50, 70, 90]
    ticks_probit = [probit(p / 100) for p in ticks_pct]
    ax.set_xticks(ticks_probit); ax.set_xticklabels(ticks_pct)
    ax.set_yticks(ticks_probit); ax.set_yticklabels(ticks_pct)
    ax.set_xlabel('ID (%)'); ax.set_ylabel('OOD (%)')

    subtitle = []
    if r_acc is not None: subtitle.append(f'AccR={r_acc:+.3f}')
    if r_agr is not None: subtitle.append(f'AgrR={r_agr:+.3f}')
    if r_cr is not None: subtitle.append(f'CrossR={r_cr:+.3f}')
    ax.set_title(f"{cfg['name']}\n" + '  '.join(subtitle), fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print('saved', out_path, '|', ' '.join(subtitle))


if __name__ == '__main__':
    names = sys.argv[1:] or ['ColoredMNIST (env2)', 'ColoredMNIST per-trial (env2)']
    out_dir = sys.argv[0]  # unused placeholder
    for name in names:
        cfg = next((c for c in CONFIGS if c['name'] == name), None)
        if cfg is None or not os.path.exists(cfg['records_path']):
            print('skip (no data):', name)
            continue
        safe = name.replace(' ', '_').replace('(', '').replace(')', '')
        out_path = (r'C:\Users\Usuario\AppData\Local\Temp\claude\c--Users-Usuario-Thesis-experiments'
                    r'\43f2de79-0305-4cec-b4bd-7bb0d85c64b0\scratchpad\aol_' + safe + '.png')
        build_figure(cfg, out_path)
