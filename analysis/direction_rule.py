"""
Direction rule: predict the SIGN of accuracy-on-the-line (ACL) without test labels,
then select one model per algorithm.

What it uses, per trained model (= one algorithm + one hyperparameter seed):
  - its predictions on the held-out part of each TRAINING environment   (*_preds.npy)
  - its held-out accuracy in each TRAINING environment                  (records.json)
The test environment is only used at the very end, to grade the result.

The idea in two sentences:
  If the models that behave differently from ERM are also the ones whose accuracy is
  steadier across training environments, the dataset is one where DG can beat ERM
  (predicted ACL < 0). Otherwise ERM-like models are the safe choice (predicted ACL > 0).

Run from the repo root:  python analysis/direction_rule.py
"""
import json
import os
import sys
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import CONFIGS, load_preds, fit_line

DATASETS = ['ColoredMNIST per-trial (env2)', 'RotatedMNIST (env5)', 'ACSIncome (env9 MI)',
            'PACS ResNet50 (env0)', 'PACS ResNet50 (env1)', 'PACS CLIP (env0)', 'PACS CLIP (env1)']
CHANCE = {'coloredmnist': 1 / 2, 'acsincome': 1 / 2, 'rotatedmnist': 1 / 10, 'pacs': 1 / 7}
ALGORITHMS = ['ERM', 'IRM', 'VREx', 'GroupDRO', 'CORAL', 'DANN']
DIVERGENCE_THRESHOLD = 2.0   # "clearly different from ERM" = twice the ERM-vs-ERM disagreement


# ---------------------------------------------------------------- loading

def load_models(cfg):
    """One dict per trained model with its accuracies (averaged over trials)."""
    records = json.load(open(cfg['records_path']))
    test_env = cfg['test_env_idx']
    train_envs = [e for e in range(cfg['n_envs']) if e != test_env]
    trials = sorted({r['args']['trial_seed'] for r in records})

    models = []
    for algo in ALGORITHMS:
        for seed in sorted({r['args']['hparams_seed'] for r in records if r['algorithm'] == algo}):
            runs = [r for r in records if r['algorithm'] == algo and r['args']['hparams_seed'] == seed]
            train_accs = np.array([np.mean([r[f'env{e}_out_acc'] for r in runs]) for e in train_envs])
            models.append({
                'algo': algo, 'seed': seed,
                'train_accs': train_accs,                  # one accuracy per training env
                'id_acc': train_accs.mean(),               # what standard selection looks at
                'ood_acc': np.mean([r[f'env{test_env}_out_acc'] for r in runs]),   # GRADING ONLY
            })
    return models, train_envs, trials


# ---------------------------------------------------------------- the two signals

def disagreement(cfg, model_a, model_b, train_envs, trials):
    """Fraction of held-out training-env images where the two models predict different classes.
    Compared within the same trial (same images), then averaged over envs and trials."""
    values = []
    for t in trials:
        for e in train_envs:
            pa = load_preds(cfg['preds_dir'], model_a['algo'], model_a['seed'], t, e)
            pb = load_preds(cfg['preds_dir'], model_b['algo'], model_b['seed'], t, e)
            if pa is not None and pb is not None:
                values.append((pa != pb).mean())
    return np.mean(values)


def add_divergence(cfg, models, train_envs, trials):
    """divergence = disagreement with ERM models / disagreement among ERM models themselves.
    1 means 'as different from ERM as another ERM seed'."""
    erm = [m for m in models if m['algo'] == 'ERM']
    erm_vs_erm = np.mean([disagreement(cfg, a, b, train_envs, trials) for a, b in combinations(erm, 2)])
    for m in models:
        others = [r for r in erm if r is not m]
        m['divergence'] = np.mean([disagreement(cfg, m, r, train_envs, trials) for r in others]) / erm_vs_erm


def add_spread(models, chance):
    """relative spread = std of accuracy across training envs / (mean accuracy - chance),
    divided by the ERM average. 1 means 'as steady as ERM', < 1 steadier.
    Dividing by (accuracy - chance) stops near-random models from looking steady:
    a model at chance gets an infinite spread."""
    for m in models:
        above_chance = m['id_acc'] - chance
        m['raw_spread'] = m['train_accs'].std() / above_chance if above_chance > 1e-6 else np.inf
    erm_level = np.mean([m['raw_spread'] for m in models if m['algo'] == 'ERM' and np.isfinite(m['raw_spread'])])
    for m in models:
        m['spread'] = m['raw_spread'] / erm_level


# ---------------------------------------------------------------- decision and selection

def predict_acl_sign(models):
    """Do the models that differ from ERM tend to be steadier?
    Spearman < 0: yes -> predict NEGATIVE ACL (DG can win, look for unlike-ERM models)
    Spearman > 0: no  -> predict POSITIVE ACL (trust ID accuracy)"""
    rho, p_value = spearmanr([m['divergence'] for m in models], [m['spread'] for m in models])
    return ('negative' if rho < 0 else 'positive'), rho, p_value


def select(candidates, predicted_sign):
    """Pick one model from `candidates` without test labels."""
    best_id = max(candidates, key=lambda m: m['id_acc'])
    if predicted_sign == 'positive':
        return best_id, ''                         # ID accuracy is trustworthy: standard selection
    different = [m for m in candidates
                 if m['divergence'] > DIVERGENCE_THRESHOLD and np.isfinite(m['spread'])]
    if not different:
        return best_id, 'never left ERM range'     # nothing different to choose from
    return min(different, key=lambda m: m['spread']), ''   # steadiest model that differs from ERM


# ---------------------------------------------------------------- run + grade

def main():
    for name in DATASETS:
        cfg = next(c for c in CONFIGS if c['name'] == name)
        models, train_envs, trials = load_models(cfg)
        add_divergence(cfg, models, train_envs, trials)
        add_spread(models, CHANCE[cfg['dtype']])
        sign, rho, p_value = predict_acl_sign(models)

        # grading: uses test labels
        true_acl = fit_line([m['id_acc'] for m in models], [m['ood_acc'] for m in models])['R']
        correct = (sign == 'negative') == (true_acl < 0)
        print(f'\n=== {name} ===')
        print(f'Spearman(divergence, spread) = {rho:+.2f} (p={p_value:.1g}) -> predicted ACL {sign} | '
              f'true ACL R = {true_acl:+.2f} -> {"correct" if correct else "WRONG"}')
        print(f'  {"algorithm":10s} {"our pick":>16s} {"highest ID acc":>16s} {"oracle":>16s}')
        for algo in ALGORITHMS + ['ALL']:
            pool = models if algo == 'ALL' else [m for m in models if m['algo'] == algo]
            if not pool:
                continue
            ours, note = select(pool, sign)
            standard = max(pool, key=lambda m: m['id_acc'])
            oracle = max(pool, key=lambda m: m['ood_acc'])
            show = lambda m: f'{m["algo"] if algo == "ALL" else ""} s{m["seed"]} {100 * m["ood_acc"]:.1f}%'
            print(f'  {algo:10s} {show(ours):>16s} {show(standard):>16s} {show(oracle):>16s}  {note}')


if __name__ == '__main__':
    main()
