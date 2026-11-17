"""
ERM-referenced ACL estimate + CrA selection, graded on the real ColoredMNIST models.

CrA(m)   = agreement of model m with the ERM seeds on the test env (mean over ERM seeds and trials).
           Selection: minimum CrA if the ACL is negative (DG can win), maximum CrA if positive.

ERM-referenced ACL estimate (gives CrA its direction, no test labels):
  1. c(e) = mean over all models of their disagreement with the ERM consensus (majority vote of the
     ERM seeds) on env e. Computed on every env, including the unlabeled test env.
  2. per model, a straight line in RAW accuracy through its two training envs:
         acc_m(c) = acc_m(c_0) + (acc_m(c_1) - acc_m(c_0)) / (c_1 - c_0) * (c - c_0)
  3. estimated test accuracy = acc_m(c_test); estimated ACL = probit line of these vs ID accuracy.
     Its sign sets the CrA direction.

Grading uses the test-color reweighting (see acl_vs_test_color.py): every quantity above is a mean
over test images, so it is computed separately on the 'agree' and 'conflict' images and mixed with
weight p. p ~ 0.10 is the real ColoredMNIST test env.

Run from the repo root:  python analysis/erm_referenced.py
"""
import os
import sys
from functools import lru_cache

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import CONFIGS, load_preds, fit_line
from direction_rule import load_models, ALGORITHMS
from acl_vs_test_color import DATASET, TEST_P, load_test_groups, add_group_accuracies


def main():
    cfg = next(c for c in CONFIGS if c['name'] == DATASET)
    test_env = cfg['test_env_idx']
    models, train_envs, trials = load_models(cfg)
    add_group_accuracies(cfg, models, trials, test_env)            # grading only
    erm = [m for m in models if m['algo'] == 'ERM']
    dg = [m for m in models if m['algo'] != 'ERM']                 # selection candidates

    @lru_cache(maxsize=None)
    def preds(algo, seed, t, e):
        return load_preds(cfg['preds_dir'], algo, seed, t, e)

    P = lambda m, t, e: preds(m['algo'], m['seed'], t, e)
    agree_mask = {t: load_test_groups(cfg, t, test_env)[1] for t in trials}

    def split_by_group(values_per_trial):
        """per-image values per trial -> (mean on agree images, mean on conflict images)"""
        return (np.mean([v[agree_mask[t]].mean() for t, v in values_per_trial.items()]),
                np.mean([v[~agree_mask[t]].mean() for t, v in values_per_trial.items()]))

    # ---- step 1: c(e) = disagreement of the pool with the ERM consensus
    def consensus(t, e):
        return (np.mean([P(r, t, e) for r in erm], axis=0) > 0.5).astype(int)

    c_train = np.array([np.mean([np.mean([(P(m, t, e) != consensus(t, e)).mean() for m in models])
                                 for t in trials]) for e in train_envs])
    per_image = {t: np.mean([P(m, t, test_env) != consensus(t, test_env) for m in models], axis=0)
                 for t in trials}
    c_test_agree, c_test_conflict = split_by_group(per_image)
    print(f'disagreement with ERM consensus: train envs {c_train.round(3)} | '
          f'test agree images {c_test_agree:.3f}, conflict images {c_test_conflict:.3f}')

    # ---- CrA per DG model, split by group
    for m in dg:
        per_image = {t: np.mean([P(m, t, test_env) == P(r, t, test_env) for r in erm], axis=0) for t in trials}
        m['cra_agree'], m['cra_conflict'] = split_by_group(per_image)

    id_acc = np.array([m['id_acc'] for m in models])
    rows = []
    for p in TEST_P:
        true = {id(m): p * m['acc_agree'] + (1 - p) * m['acc_conflict'] for m in models}
        cra = {id(m): p * m['cra_agree'] + (1 - p) * m['cra_conflict'] for m in dg}
        c_test = p * c_test_agree + (1 - p) * c_test_conflict

        # ---- steps 2-3: raw-accuracy extrapolation per model
        est = np.array([np.clip(m['train_accs'][0] + (m['train_accs'][1] - m['train_accs'][0])
                                / (c_train[1] - c_train[0]) * (c_test - c_train[0]), 0, 1) for m in models])
        true_slope = fit_line(id_acc, [true[id(m)] for m in models])['slope']
        est_slope = fit_line(id_acc, est)['slope']

        def cra_pick(pool, negative):
            return (min if negative else max)(pool, key=lambda m: cra[id(m)])

        row = dict(p=p, c_test=c_test, true_slope=true_slope, est_slope=est_slope,
                   est_dir=true[id(cra_pick(dg, est_slope < 0))],
                   true_dir=true[id(cra_pick(dg, true_slope < 0))],
                   id_pick=true[id(max(dg, key=lambda m: m['id_acc']))],
                   best=max(true[id(m)] for m in dg))
        rows.append(row)
        print(f'p={p:4.2f} c_test={c_test:.3f} | slope true {true_slope:+6.2f} est {est_slope:+6.2f} '
              f'{"ok " if np.sign(true_slope) == np.sign(est_slope) else "BAD"} | true acc of pick (all DG models): '
              f'CrA est. direction {100 * row["est_dir"]:5.1f}  CrA true direction {100 * row["true_dir"]:5.1f}  '
              f'highest ID {100 * row["id_pick"]:5.1f}  best {100 * row["best"]:5.1f}')

    # ---- check 1: per algorithm, at the real test env (p = 0.10), with the TRUE direction (min CrA)
    p = 0.10
    print(f'\nper algorithm at p = {p} (real test env), true OOD accuracy of the pick:')
    print(f'  {"algorithm":9s} {"min CrA":>16s} {"max CrA":>16s} {"highest ID":>16s} {"oracle":>16s}')
    for algo in [a for a in ALGORITHMS if a != 'ERM']:
        pool = [m for m in dg if m['algo'] == algo]
        acc = lambda m: p * m['acc_agree'] + (1 - p) * m['acc_conflict']
        score = lambda m: p * m['cra_agree'] + (1 - p) * m['cra_conflict']
        show = lambda m: f's{m["seed"]} {100 * acc(m):5.1f}% (CrA {score(m):.2f})'
        print(f'  {algo:9s} {show(min(pool, key=score)):>16s} {show(max(pool, key=score)):>16s} '
              f'{show(max(pool, key=lambda m: m["id_acc"])):>16s} {show(max(pool, key=acc)):>16s}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = [r['p'] for r in rows]
    ax = axes[0]
    ax.plot(x, [r['true_slope'] for r in rows], 'o-', color='black', label='true ACL slope')
    ax.plot(x, [r['est_slope'] for r in rows], 's--', color='tab:blue', label='ERM-referenced estimate')
    ax.axhline(0, color='grey', lw=0.7)
    ax.set_ylabel('ACL slope (probit)'); ax.set_title('ACL slope')
    ax = axes[1]
    ax.plot(x, [100 * r['best'] for r in rows], '-', color='black', label='best DG model (oracle)')
    ax.plot(x, [100 * r['est_dir'] for r in rows], 's--', color='tab:blue', label='CrA, estimated direction')
    ax.plot(x, [100 * r['true_dir'] for r in rows], '^:', color='tab:orange', label='CrA, true direction')
    ax.plot(x, [100 * r['id_pick'] for r in rows], 'v:', color='tab:grey', label='highest ID accuracy')
    ax.set_ylabel('true test accuracy of the pick (%)'); ax.set_title('Selection among all DG models')
    for ax in axes:
        ax.set_xlabel('test color agreement p')
        for v in (0.8, 0.9):
            ax.axvline(v, color='tab:green', ls=':', lw=1)
        ax.axvline(0.1, color='tab:red', ls=':', lw=1)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle('Real ColoredMNIST models: ERM-referenced ACL estimate and CrA (green = train envs, red = real test)')
    plt.tight_layout()
    plt.savefig('figures/erm_referenced.png', dpi=100)
    print('saved figures/erm_referenced.png')


if __name__ == '__main__':
    main()
