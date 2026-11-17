"""
Test-aware direction rule, graded on the real ColoredMNIST models for every test color agreement p.

Step 1 (no labels, training envs): the direction rule picks a reference model S = steadiest model
        that clearly differs from ERM. If there is none, trust ID accuracy (positive ACL).
Step 2 (assumption): S keeps its accuracy at test:  a_S(test) = a_S(ID).
Step 3 (no labels, test env): every other model's test accuracy from its agreement with S
        (2 classes, independent errors):
            agree = a_S * a_m + (1 - a_S) * (1 - a_m)   ->   a_m = (agree - (1 - a_S)) / (2 a_S - 1)
Step 4: the estimates give the ACL sign/slope, a ranking and a selection.

Check of the independence assumption: the same formula on the TRAINING envs, where labels exist.

Run from the repo root:  python analysis/test_aware_rule.py
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import CONFIGS, load_preds, fit_line
from direction_rule import load_models, add_divergence, add_spread, predict_acl_sign, select, CHANCE
from acl_vs_test_color import DATASET, TEST_P, load_test_groups, add_group_accuracies


def agreement_with(cfg, m, ref, trials, env, mask=None):
    """Fraction of images (optionally only those in `mask`) where m and ref predict the same class."""
    vals = []
    for t in trials:
        same = (load_preds(cfg['preds_dir'], m['algo'], m['seed'], t, env) ==
                load_preds(cfg['preds_dir'], ref['algo'], ref['seed'], t, env))
        vals.append(same[mask(t)].mean() if mask else same.mean())
    return np.mean(vals)


def accuracy_from_agreement(agree, a_ref):
    """2 classes, independent errors: agree = a_ref*a + (1-a_ref)*(1-a)  ->  a."""
    return np.clip((agree - (1 - a_ref)) / (2 * a_ref - 1), 0, 1)


def main():
    cfg = next(c for c in CONFIGS if c['name'] == DATASET)
    test_env = cfg['test_env_idx']
    models, train_envs, trials = load_models(cfg)
    add_group_accuracies(cfg, models, trials, test_env)          # grading only
    add_divergence(cfg, models, train_envs, trials)
    add_spread(models, CHANCE[cfg['dtype']])

    # step 1: reference model
    sign, rho, _ = predict_acl_sign(models)
    ref, note = select(models, sign)
    if sign == 'positive' or note:
        print('no steady model that differs from ERM -> trust ID accuracy'); return
    others = [m for m in models if m is not ref]
    print(f'reference S = {ref["algo"]} s{ref["seed"]}: ID acc {ref["id_acc"]:.3f}, '
          f'train accs {ref["train_accs"].round(3)}, spread {ref["spread"]:.2f}, divergence {ref["divergence"]:.1f}')

    # check of the independence assumption on the training envs (labels known there)
    for i, e in enumerate(train_envs):
        est = [accuracy_from_agreement(agreement_with(cfg, m, ref, trials, e), ref['train_accs'][i]) for m in others]
        true = [m['train_accs'][i] for m in others]
        print(f'  check on train env{e}: mean |estimated - true| accuracy = {100 * np.mean(np.abs(np.array(est) - true)):.1f} points, '
              f'Spearman {spearmanr(est, true)[0]:+.2f}')

    # step 3 on the test env, per group so it can be reweighted to any p
    agree_group = lambda t: load_test_groups(cfg, t, test_env)[1]
    for m in others:
        m['agr_agree'] = agreement_with(cfg, m, ref, trials, test_env, agree_group)
        m['agr_conflict'] = agreement_with(cfg, m, ref, trials, test_env, lambda t: ~agree_group(t))

    id_acc = np.array([m['id_acc'] for m in models])
    rows = []
    for p in TEST_P:
        true = np.array([p * m['acc_agree'] + (1 - p) * m['acc_conflict'] for m in models])
        est = np.array([ref['id_acc'] if m is ref else
                        accuracy_from_agreement(p * m['agr_agree'] + (1 - p) * m['agr_conflict'], ref['id_acc'])
                        for m in models])
        t_line, e_line = fit_line(id_acc, true), fit_line(id_acc, est)
        rows.append(dict(p=p, true_slope=t_line['slope'], est_slope=e_line['slope'],
                         rank=spearmanr(est, true)[0], pick=true[est.argmax()],
                         id_pick=true[id_acc.argmax()], rule_pick=true[models.index(ref)], best=true.max(),
                         mae=np.mean(np.abs(est - true))))
        r = rows[-1]
        print(f'p={p:4.2f} | slope true {r["true_slope"]:+6.2f} est {r["est_slope"]:+6.2f} '
              f'{"ok " if np.sign(r["true_slope"]) == np.sign(r["est_slope"]) else "BAD"} | '
              f'rank corr {r["rank"]:+.2f} | MAE {100 * r["mae"]:4.1f} | true acc of pick: new {100 * r["pick"]:5.1f} '
              f'old rule {100 * r["rule_pick"]:5.1f} ID {100 * r["id_pick"]:5.1f} best {100 * r["best"]:5.1f}')

    P = [r['p'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(P, [r['true_slope'] for r in rows], 'o-', color='black', label='true ACL slope')
    ax.plot(P, [r['est_slope'] for r in rows], 's--', color='tab:blue', label='test-aware estimate')
    ax.axhline(0, color='grey', lw=0.7)
    ax.set_xlabel('test color agreement p'); ax.set_ylabel('ACL slope (probit)'); ax.legend()
    ax.set_title('ACL slope')
    ax = axes[1]
    ax.plot(P, [100 * r['best'] for r in rows], '-', color='black', label='best model (oracle)')
    ax.plot(P, [100 * r['pick'] for r in rows], 's--', color='tab:blue', label='test-aware pick')
    ax.plot(P, [100 * r['rule_pick'] for r in rows], '^:', color='tab:orange', label='old direction rule pick')
    ax.plot(P, [100 * r['id_pick'] for r in rows], 'v:', color='tab:grey', label='highest ID accuracy')
    ax.set_xlabel('test color agreement p'); ax.set_ylabel('true test accuracy of the pick (%)'); ax.legend()
    ax.set_title('Model selection')
    for ax in axes:
        for x in (0.8, 0.9):
            ax.axvline(x, color='tab:green', ls=':', lw=1)
        ax.axvline(0.1, color='tab:red', ls=':', lw=1)
        ax.grid(alpha=0.3)
    fig.suptitle(f'Real ColoredMNIST models, reference S = {ref["algo"]} s{ref["seed"]} '
                 '(green = train envs, red = real test)')
    plt.tight_layout()
    plt.savefig('figures/test_aware_rule.png', dpi=100)
    print('saved figures/test_aware_rule.png')


if __name__ == '__main__':
    main()
