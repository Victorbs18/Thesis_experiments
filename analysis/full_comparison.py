"""
Full comparison table: CrA with its direction from the ERM-referenced ACL estimate, against the
ODP-Bench surrogate scores, per dataset. Same scores as odp_bench_comparison.py; the regime
(= CrA direction) comes from the sign of the ERM-referenced ACL estimate (no test labels):
     1. c(e) = mean over all models of their disagreement with the ERM consensus
        (majority vote of the ERM seeds) on env e, for every env incl. the unlabeled test env
     2. per model, least-squares line acc_m(c) in RAW accuracy through its training envs
     3. estimated test accuracy = acc_m(c_test); estimated ACL = probit line vs ID accuracy
   estimated slope < 0 -> well-specified -> CrA picks the seed LEAST like ERM (min CrA)
   estimated slope > 0 -> misspecified   -> CrA picks the seed MOST like ERM (max CrA)

CrA-TV: training-domain validation if the estimated ACL slope is positive, minimum CrA if negative.
ERM seeds get a leave-one-out CrA: agreement with the OTHER ERM seeds of the same trial.

Test accuracy follows DomainBed: label-free scores use the saved predictions on the test env's 20%
OUT split; every method reports the 80% IN accuracy of the configuration it selects; the oracle
selects by OUT accuracy and reports IN accuracy. Every ranking uses the mean over a
configuration's 3 trials.

Each dataset block prints the checks, then:
  ACL    : true ACL (slope, R; IN accuracy) vs estimated ACL, and the resulting regime
  Table 1: Spearman r of each score with the true IN test accuracy over the 20 hparam seeds
  Table 2: true IN test accuracy (%) of the seed each method selects (mean +- std over 3 trials)
  Pooled : selections over all DG seeds, and over all models incl. ERM
  OUT vs IN: before/after comparison of the reported accuracies

Run from the repo root:  python analysis/full_comparison.py
Writes tables/full_comparison_{acl,spearman,selection,summary}.csv
(render them with analysis/tables_to_html.py)
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import CONFIGS, ALGOS, load_preds, fit_line, get_records, get_test_labels
from direction_rule import DATASETS, load_models
from odp_bench_comparison import (extract_seed_metrics, build_erm_pool, add_distance_scores,
                                  evaluate, evaluate_sel, evaluate_regime_sel)

PACS_DATA_DIR = 'C:/Users/Usuario/Downloads/pacs_data/pacs_data'   # same default as odp_bench_comparison


# ---------------------------------------------------------------- ERM-referenced ACL estimate

def test_in_accuracy(records, algo, seed, test_env):
    """Mean and std over trials of the test env's IN-split accuracy (DomainBed's reported number)."""
    accs = [r[f'env{test_env}_in_acc'] for r in get_records(records, algo, seed)]
    return float(np.mean(accs)), float(np.std(accs))


def erm_referenced_acl(cfg, records):
    """True ACL (IN labels) and estimated ACL (no test labels) over all models, plus the env axis c."""
    models, train_envs, trials = load_models(cfg)
    test_env, n_classes = cfg['test_env_idx'], len(cfg['class_names'])
    pred = lambda m, t, e: load_preds(cfg['preds_dir'], m['algo'], m['seed'], t, e)
    complete = [m for m in models
                if all(pred(m, t, e) is not None for t in trials for e in range(cfg['n_envs']))]
    erm = [m for m in complete if m['algo'] == 'ERM']

    def c_of(e):
        values = []
        for t in trials:
            stack = np.stack([pred(r, t, e) for r in erm])
            consensus = np.stack([(stack == k).sum(0) for k in range(n_classes)]).argmax(0)
            values.append(np.mean([(pred(m, t, e) != consensus).mean() for m in complete]))
        return np.mean(values)

    c_train = np.array([c_of(e) for e in train_envs])
    c_test = c_of(test_env)
    est = np.array([np.clip(np.polyval(np.polyfit(c_train, m['train_accs'], 1), c_test), 0, 1)
                    for m in models])
    id_acc = [m['id_acc'] for m in models]
    test_in = np.array([test_in_accuracy(records, m['algo'], m['seed'], test_env)[0] for m in models])
    true = fit_line(id_acc, test_in)
    true_out = fit_line(id_acc, [m['ood_acc'] for m in models])
    estimated = fit_line(id_acc, est)
    return dict(true_slope=true['slope'], true_R=true['R'], true_slope_out=true_out['slope'],
                est_slope=estimated['slope'], est_R=estimated['R'], c_train=c_train, c_test=c_test,
                n_models=len(models), mae=np.mean(np.abs(est - test_in)))


def regime_of(slope):
    return 'well-specified' if slope < 0 else 'misspecified'


# ---------------------------------------------------------------- additions to the seed metrics

def add_test_in(seed_data, records, algo, test_env):
    for seed, d in seed_data.items():
        d['test_in'], d['test_in_std'] = test_in_accuracy(records, algo, seed, test_env)


def add_loo_cra(cfg, erm_seed_data, trials):
    """ERM seeds: CrA = mean test-image agreement with the OTHER ERM seeds of the same trial."""
    test_env = cfg['test_env_idx']
    pred = lambda s, t: load_preds(cfg['preds_dir'], 'ERM', s, t, test_env)
    for s, d in erm_seed_data.items():
        values = [(pred(s, t) == pred(r, t)).mean()
                  for t in trials for r in erm_seed_data
                  if r != s and pred(s, t) is not None and pred(r, t) is not None]
        d['crossagr'] = float(np.mean(values)) if values else None


# ---------------------------------------------------------------- selection rules

SCORES = [  # (column, seed_data key, how the old script selects with it)
    ('ATC', 'atc', 'max'), ('DOC', 'doc', 'min'), ('Nuc', 'nuc_norm', 'max'), ('MDE', 'mde', 'min'),
    ('Dsp', 'dispersion', 'max'), ('Agr', 'agreement', 'max'), ('TV', 'trainval', 'max'),
]
DISTANCES = [('MMD', 'mmd_cross'), ('Wass', 'wass_cross'), ('PAD', 'pad_cross')]
SEL_COLS = ['ATC', 'DOC', 'Nuc', 'MDE', 'Dsp', 'Agr', 'TV', 'MMD', 'Wass', 'PAD', 'CrA', 'CrA-TV', 'Oracle']
HIDDEN_COLS = ['CrA*', 'CrA-TV*']          # true-sign variants: computed, shown only if the signs differ
R_COLS = ['ATC', 'DOC', 'Nuc', 'MDE', 'Dsp', 'Agr', 'TV', 'CrA', 'MMD', 'Wass', 'PAD']


def cra_tv(seed_data, regime, report_key):
    """CrA-TV: training-domain validation if misspecified, minimum CrA if well-specified."""
    if regime == 'misspecified':
        return evaluate_sel(seed_data, 'trainval', report_key=report_key, return_key=True)
    return evaluate_regime_sel(seed_data, 'crossagr', regime, higher_is_similar=True,
                               report_key=report_key, return_key=True)


def select_all(seed_data, is_erm, regime, true_regime, report_key):
    """{column: (accuracy, std, selected seed)} for every method, reporting `report_key`."""
    sel = {}
    for col, key, how in SCORES:
        sel[col] = evaluate_sel(seed_data, key, minimize=(how == 'min'), report_key=report_key, return_key=True)
    for col, key in DISTANCES:
        sel[col] = ((None, None, None) if is_erm else
                    evaluate_regime_sel(seed_data, key, regime, higher_is_similar=False,
                                        report_key=report_key, return_key=True))
    for col, reg in [('CrA', regime), ('CrA*', true_regime)]:
        sel[col] = evaluate_regime_sel(seed_data, 'crossagr', reg, higher_is_similar=True,
                                       report_key=report_key, return_key=True)
    sel['CrA-TV'] = cra_tv(seed_data, regime, report_key)
    sel['CrA-TV*'] = cra_tv(seed_data, true_regime, report_key)
    # oracle: select by the test env's OUT accuracy, report `report_key` (DomainBed)
    sel['Oracle'] = evaluate_sel(seed_data, 'oracle', report_key=report_key, return_key=True)
    return sel


# ---------------------------------------------------------------- checks

def check_test_labels(cfg, records):
    """The saved test labels exist once per results folder, not per algorithm. If every run of every
    algorithm reproduces its RECORDED test accuracy from its saved predictions and these labels,
    all algorithms were scored on the same labeled test images."""
    test_env, preds_dir = cfg['test_env_idx'], cfg['preds_dir']
    trials = sorted({r['args']['trial_seed'] for r in records})
    labels, source = {}, ''
    for t in trials:
        per_trial = os.path.join(preds_dir, f'env{test_env}_trial{t}_labels.npy')
        shared = os.path.join(preds_dir, f'env{test_env}_labels.npy')
        if os.path.exists(per_trial):
            labels[t], source = np.load(per_trial), 'saved per-trial label files'
        elif os.path.exists(shared):
            labels[t], source = np.load(shared), 'saved label file'
        else:
            try:
                rebuilt = labels.get('rebuilt')
                if rebuilt is None:
                    cfg_l = dict(cfg, data_dir=PACS_DATA_DIR if cfg['dtype'] == 'pacs' else cfg['data_dir'])
                    rebuilt = labels['rebuilt'] = get_test_labels(cfg_l)
                labels[t], source = rebuilt, 'labels rebuilt from the dataset (none were saved)'
            except Exception as e:
                return False, f'labels unavailable ({type(e).__name__}: {e})'
    bad, missing = {}, 0
    for r in records:
        t, algo = r['args']['trial_seed'], r['algorithm']
        pred = load_preds(preds_dir, algo, r['args']['hparams_seed'], t, test_env)
        if pred is None:
            missing += 1
            continue
        if not (len(pred) == len(labels[t])
                and abs((pred == labels[t]).mean() - r[f'env{test_env}_out_acc']) < 1e-6):
            bad[algo] = bad.get(algo, 0) + 1
    n = len(records) - missing
    note = f'{n}/{n} runs with saved predictions' + (f', {missing} runs without' if missing else '')
    if not bad:
        return True, f'{note} reproduce their recorded test accuracy ({source})'
    detail = ', '.join(f'{a}: {k} runs' for a, k in bad.items())
    return False, f'{note}; recorded accuracy NOT reproduced for {detail} ({source})'


# ---------------------------------------------------------------- printing / export helpers

def fmt_r(v):
    return f'{v:+.2f}' if v is not None and v == v else '  -  '


def fmt_acc(v, sd=None):
    if v is None:
        return '-'
    return f'{100 * v:.1f}' + (f'±{100 * sd:.1f}' if sd is not None else '')


def pct(v):
    return round(100 * v, 2) if v is not None else None


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'saved {path}')


def main():
    acl_table, spearman_table, selection_table, summary_table = [], [], [], []
    for name in DATASETS:
        cfg = next(c for c in CONFIGS if c['name'] == name)
        records = json.load(open(cfg['records_path']))
        available = [a for a in ALGOS if any(r['algorithm'] == a for r in records)]
        trials = sorted({r['args']['trial_seed'] for r in records})
        test_env = cfg['test_env_idx']

        acl = erm_referenced_acl(cfg, records)
        regime, true_regime = regime_of(acl['est_slope']), regime_of(acl['true_slope'])
        sign_match = regime == true_regime
        print(f'\n{"=" * 120}\n{name}   ({acl["n_models"]} models, algorithms: {", ".join(available)})')

        # ---- compute every algorithm's seed metrics (existing functions) + IN accuracy + LOO CrA
        per_algo, erm_pool = {}, None
        for algo in available:
            seed_data = extract_seed_metrics(cfg, records, algo)
            add_test_in(seed_data, records, algo, test_env)
            if algo == 'ERM':
                erm_pool = build_erm_pool(seed_data)
                add_loo_cra(cfg, seed_data, trials)
            else:
                add_distance_scores(seed_data, erm_pool)
            per_algo[algo] = seed_data

        sel_in = {a: select_all(sd, a == 'ERM', regime, true_regime, 'test_in') for a, sd in per_algo.items()}
        sel_out = {a: select_all(sd, a == 'ERM', regime, true_regime, 'oracle') for a, sd in per_algo.items()}

        # ---- checks
        print('  CHECKS')
        ok, msg = check_test_labels(cfg, records)
        print(f'    test labels identical across algorithms: {"yes" if ok else "NO"} - {msg}')
        tv_ok = all(abs(d['trainval'] - np.mean([np.mean([r[f'env{e}_out_acc'] for e in range(cfg['n_envs'])
                                                          if e != test_env])
                                                 for r in get_records(records, a, s)])) < 1e-9
                    for a, sd in per_algo.items() for s, d in sd.items())
        print(f'    TV ranks each configuration by the mean over its {len(trials)} trials: {"yes" if tv_ok else "NO"}')
        if regime == 'well-specified':
            same = all(sel_in[a]['CrA'][2] == sel_in[a]['CrA-TV'][2] for a in per_algo)
            print(f'    CrA and CrA-TV give identical selections (estimated ACL < 0): {"yes" if same else "NO"}')
        if 'ERM' in per_algo:
            erm = per_algo['ERM']
            max_cra = max(erm, key=lambda s: erm[s]['crossagr'])
            max_agr = max(erm, key=lambda s: erm[s]['agreement'])
            score_gap = max(abs(d['crossagr'] - d['agreement']) for d in erm.values())
            print(f'    ERM: leave-one-out max CrA picks s{max_cra}, Agr picks s{max_agr}: '
                  f'{"same" if max_cra == max_agr else "DIFFERENT"} '
                  f'(largest |LOO CrA - Agr| over seeds = {score_gap:.1e}; both are the mean agreement '
                  f'with the other ERM seeds of the same trial)')
        print(f'    estimated sign = true sign: {"yes" if sign_match else "no"}')

        # ---- ACL
        print(f'  ACL true (IN)  : slope {acl["true_slope"]:+.2f}  R {acl["true_R"]:+.2f}  -> {true_regime}'
              f'   (OUT: slope {acl["true_slope_out"]:+.2f})')
        print(f'  ACL estimated  : slope {acl["est_slope"]:+.2f}  R {acl["est_R"]:+.2f}  -> {regime}  '
              f'(axis c: train {acl["c_train"].round(3)}, test {acl["c_test"]:.3f}; '
              f'per-model OOD estimate MAE {100 * acl["mae"]:.1f} pts vs IN accuracy)')

        # ---- Table 1: Spearman with IN accuracy
        print('\n  Table 1: Spearman r with true IN test accuracy (over hparam seeds)')
        print('  ' + f'{"algo":9s}{"N":>3s} |' + ''.join(f'{c:>7s}' for c in R_COLS))
        keys = dict([(c, k) for c, k, _ in SCORES] + DISTANCES + [('CrA', 'crossagr')])
        for algo, sd in per_algo.items():
            r = {c: evaluate(sd, keys[c], oracle_key='test_in') for c in R_COLS}
            print('  ' + f'{algo:9s}{len(sd):3d} |' + ''.join(f'{fmt_r(r[c]):>7s}' for c in R_COLS))
            spearman_table.append({'dataset': name, 'algorithm': algo, 'n_seeds': len(sd),
                                   **{c: (round(v, 3) if v is not None and v == v else None) for c, v in r.items()}})

        # ---- Table 2: selection, IN accuracy
        print(f'\n  Table 2: true IN test accuracy (%) of the selected seed   [direction: {regime}]')
        print('  ' + f'{"algo":9s} |' + ''.join(f'{c:>11s}' for c in SEL_COLS))
        for algo, sel in sel_in.items():
            print('  ' + f'{algo:9s} |' + ''.join(f'{fmt_acc(*sel[c][:2]):>11s}' for c in SEL_COLS))
            entry = {'dataset': name, 'algorithm': algo, 'direction': regime}
            entry.update({c: pct(sel[c][0]) for c in SEL_COLS + HIDDEN_COLS})
            entry.update({c + '_std': pct(sel[c][1]) for c in SEL_COLS + HIDDEN_COLS})
            selection_table.append(entry)
        if not sign_match:
            print('  estimated sign differs from the true sign -> true-sign variants:')
            for algo, sel in sel_in.items():
                print(f'    {algo:9s} CrA* {fmt_acc(*sel["CrA*"][:2])}   CrA-TV* {fmt_acc(*sel["CrA-TV*"][:2])}')

        dg = [a for a in per_algo if a != 'ERM']
        mean_of = lambda sel, c: (np.mean([sel[a][c][0] for a in dg if sel[a][c][0] is not None])
                                  if any(sel[a][c][0] is not None for a in dg) else None)
        summary_table.append({'dataset': name, 'n_dg_algorithms': len(dg), 'direction': regime,
                              **{c: pct(mean_of(sel_in, c)) for c in SEL_COLS + HIDDEN_COLS}})

        # ---- pooled selections
        pooled_dg = {f'{a} s{s}': d for a in dg for s, d in per_algo[a].items()}
        pooled_all = {f'{a} s{s}': d for a in per_algo for s, d in per_algo[a].items()}
        pooled = {
            'pooled_cra_dg': evaluate_regime_sel(pooled_dg, 'crossagr', regime, True, 'test_in', True),
            'pooled_oracle_dg': evaluate_sel(pooled_dg, 'oracle', report_key='test_in', return_key=True),
            'pooled_cra_all': evaluate_regime_sel(pooled_all, 'crossagr', regime, True, 'test_in', True),
            'pooled_cratv_all': cra_tv(pooled_all, regime, 'test_in'),
            'pooled_oracle_all': evaluate_sel(pooled_all, 'oracle', report_key='test_in', return_key=True),
        }
        labels = {'pooled_cra_dg': 'Pooled CrA (DG only)', 'pooled_oracle_dg': 'Pooled oracle (DG only)',
                  'pooled_cra_all': 'Pooled CrA (all)', 'pooled_cratv_all': 'Pooled CrA-TV (all)',
                  'pooled_oracle_all': 'Pooled oracle (all)'}
        print()
        for k, (acc, sd, key) in pooled.items():
            print(f'  {labels[k]:24s}: {key} -> {fmt_acc(acc, sd)}%')

        # ---- OUT vs IN
        print('\n  OUT vs IN test accuracy (mean of DG algorithms, %):')
        print('  ' + ' | '.join(f'{c} {fmt_acc(mean_of(sel_out, c))}->{fmt_acc(mean_of(sel_in, c))}'
                                for c in SEL_COLS))

        acl_table.append({
            'dataset': name, 'n_models': acl['n_models'], 'algorithms': ' '.join(available),
            'true_slope': round(acl['true_slope'], 3), 'true_R': round(acl['true_R'], 3),
            'est_slope': round(acl['est_slope'], 3), 'est_R': round(acl['est_R'], 3),
            'true_regime': true_regime, 'est_regime': regime, 'sign_match': sign_match,
            'ood_estimate_mae_pts': pct(acl['mae']),
            'c_train': ' '.join(f'{c:.3f}' for c in acl['c_train']), 'c_test': round(acl['c_test'], 3),
            **{f'{k}_pick': v[2] for k, v in pooled.items()},
            **{f'{k}_acc': pct(v[0]) for k, v in pooled.items()},
            'labels_check': ('ok: ' if ok else 'FAILED: ') + msg,
        })

    write_csv('tables/full_comparison_acl.csv', acl_table)
    write_csv('tables/full_comparison_spearman.csv', spearman_table)
    write_csv('tables/full_comparison_selection.csv', selection_table)
    write_csv('tables/full_comparison_summary.csv', summary_table)


if __name__ == '__main__':
    main()
