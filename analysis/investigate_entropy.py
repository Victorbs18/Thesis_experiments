# analysis/investigate_entropy.py
"""
Investigates relationships between agreement, entropy and accuracy.
Used to find label-free signals that correlate with IRM performance.
"""

import json
import os
import numpy as np
from scipy.stats import pearsonr

# Paths
records_path = 'results/coloredmnist/test_env2/cnn/random/records.json'
preds_dir    = 'results/coloredmnist/test_env2/cnn/random/models'
test_env_idx = 2

with open(records_path) as f:
    records = json.load(f)

data = []
for seed in range(20):
    erm = [r for r in records if r['algorithm'] == 'ERM'
           and r['args']['hparams_seed'] == seed]
    irm = [r for r in records if r['algorithm'] == 'IRM'
           and r['args']['hparams_seed'] == seed]
    if not erm or not irm:
        continue

    acc_erm = np.mean([r[f'env{test_env_idx}_out_acc'] for r in erm])
    acc_irm = np.mean([r[f'env{test_env_idx}_out_acc'] for r in irm])

    probs_erm_list = []
    probs_irm_list = []
    preds_erm_list = []
    preds_irm_list = []

    for t in range(3):
        pe = np.load(os.path.join(preds_dir,
             f'ERM_hpseed{seed}_trial{t}_env{test_env_idx}_probs.npy'))
        pi = np.load(os.path.join(preds_dir,
             f'IRM_hpseed{seed}_trial{t}_env{test_env_idx}_probs.npy'))
        de = np.load(os.path.join(preds_dir,
             f'ERM_hpseed{seed}_trial{t}_env{test_env_idx}_preds.npy'))
        di = np.load(os.path.join(preds_dir,
             f'IRM_hpseed{seed}_trial{t}_env{test_env_idx}_preds.npy'))
        probs_erm_list.append(pe)
        probs_irm_list.append(pi)
        preds_erm_list.append(de)
        preds_irm_list.append(di)

    h_erm = float(np.mean([
        -np.sum(p * np.log(p + 1e-8), axis=1).mean()
        for p in probs_erm_list
    ]))
    h_irm = float(np.mean([
        -np.sum(p * np.log(p + 1e-8), axis=1).mean()
        for p in probs_irm_list
    ]))

    ood_agr = float(np.mean([
        np.mean(de == di)
        for de in preds_erm_list
        for di in preds_irm_list
    ]))

    lam    = irm[0]['hparams'].get('irm_lambda', 0)
    anneal = irm[0]['hparams'].get('irm_penalty_anneal_iters', 0)

    data.append({
        'seed':    seed,
        'acc_erm': float(acc_erm),
        'acc_irm': float(acc_irm),
        'h_erm':   h_erm,
        'h_irm':   h_irm,
        'dh':      h_irm - h_erm,
        'ood_agr': ood_agr,
        'lambda':  lam,
        'anneal':  anneal,
    })

# Print full table
print(f"\n{'='*100}")
print(f"  Per-seed analysis — ColoredMNIST env2")
print(f"{'='*100}")
print(f"  {'seed':>4} {'acc_ERM':>8} {'acc_IRM':>8} {'H_ERM':>7} "
      f"{'H_IRM':>7} {'dH':>7} {'OOD_agr':>8} {'lambda':>10} {'anneal':>7}")
print(f"  {'-'*90}")

for d in data:
    print(f"  {d['seed']:>4} {d['acc_erm']:>8.3f} {d['acc_irm']:>8.3f} "
          f"{d['h_erm']:>7.4f} {d['h_irm']:>7.4f} {d['dh']:>7.4f} "
          f"{d['ood_agr']:>8.3f} {d['lambda']:>10.1f} {d['anneal']:>7d}")

# Correlations with acc_irm
print(f"\n{'='*60}")
print(f"  Pearson correlations with acc_IRM")
print(f"{'='*60}")

keys   = ['acc_erm', 'h_erm', 'h_irm', 'dh', 'ood_agr', 'lambda']
vals   = {k: np.array([d[k] for d in data]) for k in keys}
target = np.array([d['acc_irm'] for d in data])

for k in keys:
    r, p = pearsonr(vals[k], target)
    print(f"  {k:12} vs acc_IRM:  R={r:+.3f}  p={p:.3f}")

# Correlations with ood_agr (label-free signal)
print(f"\n  Pearson correlations with OOD_agr (label-free)")
print(f"  {'-'*50}")
for k in ['acc_erm', 'acc_irm', 'h_erm', 'h_irm', 'dh', 'lambda']:
    v = vals[k] if k in vals else target  # target = acc_irm
    r, p = pearsonr(v, vals['ood_agr'])
    print(f"  {k:12} vs OOD_agr:  R={r:+.3f}  p={p:.3f}")