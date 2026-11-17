"""
ACL slope vs test color agreement on the REAL ColoredMNIST models (no retraining).

Model weights are not saved, but every model's predictions on the test env are, together with
each test image's label and color. Split the test images into two groups:
  agree    = color matches the label
  conflict = color contradicts the label
A test env where a fraction p of images agree has accuracy
  acc(p) = p * acc_agree + (1 - p) * acc_conflict
This is exact because color is assigned independently of the digit given the label: the
images inside each group look the same whatever p is, only the mix changes.
The real test env (env2) has p ~ 0.10; the training envs have p ~ 0.9 and 0.8.

For each p we compute:
  true ACL slope   - probit line of acc(p) vs ID accuracy over all models (uses test labels)
  method 2 slope   - same line, using method 2's label-free OOD estimates
  direction rule   - one sign from the training envs, the same for every p

Run from the repo root:  python analysis/acl_vs_test_color.py
"""
import os
import sys

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import CONFIGS, load_preds, fit_line
from direction_rule import load_models, add_divergence, add_spread, predict_acl_sign, CHANCE

DATASET = 'ColoredMNIST per-trial (env2)'
TEST_P = np.round(np.arange(0.0, 1.0001, 0.05), 2)
TRAIN_P = [0.9, 0.8]
EPS = 1e-3
probit = lambda p: norm.ppf(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))


def load_test_groups(cfg, trial, test_env):
    """Boolean mask of the 'agree' images in this trial's test split (color == label)."""
    labels = np.load(os.path.join(cfg['preds_dir'], f'env{test_env}_trial{trial}_labels.npy'))
    colors = np.load(os.path.join(cfg['preds_dir'], f'env{test_env}_trial{trial}_colors.npy'))
    return labels, colors == labels


def add_group_accuracies(cfg, models, trials, test_env):
    """acc_agree / acc_conflict per model, averaged over trials. Needs test labels (grading only)."""
    for m in models:
        agree_accs, conflict_accs = [], []
        for t in trials:
            labels, agree = load_test_groups(cfg, t, test_env)
            correct = load_preds(cfg['preds_dir'], m['algo'], m['seed'], t, test_env) == labels
            agree_accs.append(correct[agree].mean())
            conflict_accs.append(correct[~agree].mean())
        m['acc_agree'], m['acc_conflict'] = np.mean(agree_accs), np.mean(conflict_accs)


def majority_disagreement(cfg, models, trial, env):
    """Per image: fraction of models that disagree with the majority vote of all models. No labels."""
    stack = np.stack([load_preds(cfg['preds_dir'], m['algo'], m['seed'], trial, env) for m in models])
    majority = (stack.mean(0) > 0.5).astype(stack.dtype)   # 2 classes
    return (stack != majority).mean(0)


def environment_coordinates(cfg, models, trials, train_envs, test_env):
    """Method 2's position of each env = mean disagreement with the majority (no labels).
    For the test env, it is split by group so it can be reweighted to any p like accuracy.
    Splitting needs the labels, but only to BUILD the hypothetical test env of agreement p:
    on a real test env, c is simply the mean over all its images, no labels needed."""
    c_train = np.array([np.mean([majority_disagreement(cfg, models, t, e).mean() for t in trials])
                        for e in train_envs])
    c_agree, c_conflict = [], []
    for t in trials:
        d = majority_disagreement(cfg, models, t, test_env)
        _, agree = load_test_groups(cfg, t, test_env)
        c_agree.append(d[agree].mean())
        c_conflict.append(d[~agree].mean())
    return c_train, np.mean(c_agree), np.mean(c_conflict)


def main():
    cfg = next(c for c in CONFIGS if c['name'] == DATASET)
    test_env = cfg['test_env_idx']
    models, train_envs, trials = load_models(cfg)
    add_group_accuracies(cfg, models, trials, test_env)

    # sanity check: at the real p, the reweighted accuracy must equal the recorded test accuracy
    p_real = np.mean([load_test_groups(cfg, t, test_env)[1].mean() for t in trials])
    gap = max(abs(p_real * m['acc_agree'] + (1 - p_real) * m['acc_conflict'] - m['ood_acc']) for m in models)
    print(f'{len(models)} models | real test p = {p_real:.3f} | max |reweighted - recorded| = {gap:.4f}')

    # direction rule: training envs only -> one sign for every p
    add_divergence(cfg, models, train_envs, trials)
    add_spread(models, CHANCE[cfg['dtype']])
    sign, rho, _ = predict_acl_sign(models)
    print(f'direction rule: Spearman(divergence, spread) = {rho:+.2f} -> predicts {sign} for EVERY p')

    # method 2: per model, line probit(acc) = a + b*c through the 2 training envs
    c_train, c_agree, c_conflict = environment_coordinates(cfg, models, trials, train_envs, test_env)
    lines = [np.polyfit(c_train, probit(m['train_accs']), 1) for m in models]
    print(f'coordinates: train envs {c_train.round(3)} | test agree group {c_agree:.3f}, '
          f'conflict group {c_conflict:.3f}')

    id_acc = np.array([m['id_acc'] for m in models])
    true_slopes, est_slopes = [], []
    for p in TEST_P:
        true_ood = np.array([p * m['acc_agree'] + (1 - p) * m['acc_conflict'] for m in models])
        c_test = p * c_agree + (1 - p) * c_conflict
        est_ood = np.array([norm.cdf(a + b * c_test) for b, a in lines])
        true_line, est_line = fit_line(id_acc, true_ood), fit_line(id_acc, est_ood)
        true_slopes.append(true_line['slope'])
        est_slopes.append(est_line['slope'])
        print(f'p = {p:4.2f}: true slope {true_line["slope"]:+7.2f} (R {true_line["R"]:+.2f}) | '
              f'method 2 slope {est_line["slope"]:+7.2f} | rule says {sign} | c_test = {c_test:.3f}')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(TEST_P, true_slopes, 'o-', color='black', label='true ACL slope (real models, reweighted)')
    ax.plot(TEST_P, est_slopes, 's--', color='tab:blue', label='method 2 estimate (extrapolation)')
    ax.axhline(0, color='grey', lw=0.7)
    lo = min(min(true_slopes), min(est_slopes)) - 0.5
    hi = max(max(true_slopes), max(est_slopes)) + 0.5
    ax.fill_between([0, 1], 0, lo if sign == 'negative' else hi, color='tab:orange', alpha=0.08,
                    label=f'direction rule: always predicts {sign}')
    for p in TRAIN_P:
        ax.axvline(p, color='tab:green', ls=':', lw=1.2)
    ax.axvline(p_real, color='tab:red', ls=':', lw=1.2)
    ax.text(0.85, 0, ' train envs', color='tab:green', fontsize=9, va='bottom')
    ax.text(p_real, 0, ' ColoredMNIST test', color='tab:red', fontsize=9, va='bottom')
    ax.set_ylim(lo, hi)
    ax.set_xlabel('test environment: how often color agrees with the label')
    ax.set_ylabel('ACL slope (probit space)')
    ax.set_title(f'Real ColoredMNIST models ({len(models)} = 6 algorithms x 20 hparams, mean of 3 trials): '
                 'ACL slope vs test color agreement', fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/real_acl_vs_test_color.png', dpi=100)
    print('saved figures/real_acl_vs_test_color.png')


if __name__ == '__main__':
    main()
