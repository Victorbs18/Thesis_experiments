# analysis/cross_algorithm_agreement.py
"""
Cross-algorithm agreement diagnostic.

For each matched pair (ERM_i, IRM_i) with the same hparams_seed:
    ID agreement  = fraction of examples where ERM_i and IRM_i
                    predict the same class on training env val splits
    OOD agreement = fraction of examples where ERM_i and IRM_i
                    predict the same class on test env
    Entropy       = mean prediction entropy of each algorithm on test env
    Accuracy      = OOD accuracy from records.json
    Lambda        = IRM penalty weight

Usage:
    python analysis/cross_algorithm_agreement.py \
        --preds_dir    results/coloredmnist/test_env2/cnn/random/models \
        --records_path results/coloredmnist/test_env2/cnn/random/records.json \
        --test_env_idx 2 \
        --n_envs       3 \
        --n_hparams    20 \
        --n_trials     3 \
        --dataset_name ColoredMNIST \
        --test_env_name "-90% (env2)"
"""

import os
import json
import argparse
import numpy as np
from scipy.special import ndtri as probit
from scipy.stats import pearsonr, linregress


# Loading utilities

def load_predictions(preds_dir, algorithm, hparams_seed, trial_seed, env_idx):
    fname = (f"{algorithm}_hpseed{hparams_seed}"
             f"_trial{trial_seed}_env{env_idx}_preds.npy")
    path = os.path.join(preds_dir, fname)
    return np.load(path) if os.path.exists(path) else None


def load_probs(preds_dir, algorithm, hparams_seed, trial_seed, env_idx):
    fname = (f"{algorithm}_hpseed{hparams_seed}"
             f"_trial{trial_seed}_env{env_idx}_probs.npy")
    path = os.path.join(preds_dir, fname)
    return np.load(path) if os.path.exists(path) else None


def get_all_trials(preds_dir, algorithm, hparams_seed, n_trials,
                   env_idx, loader_fn):
    results = []
    for trial in range(n_trials):
        r = loader_fn(preds_dir, algorithm, hparams_seed, trial, env_idx)
        if r is not None:
            results.append(r)
    return results


def get_record_info(records, algo, seed, test_env_idx):
    matching = [r for r in records
                if r['algorithm'] == algo
                and r['args']['hparams_seed'] == seed]
    if not matching:
        return None
    ood_acc = np.mean([r[f'env{test_env_idx}_out_acc'] for r in matching])
    hp = matching[0]['hparams']
    return {
        'ood_acc': float(ood_acc),
        'lr':      hp.get('lr', 0),
        'lambda':  hp.get('irm_lambda', None),
        'anneal':  hp.get('irm_penalty_anneal_iters', None),
        'bs':      hp.get('batch_size', None),
    }


# Metrics

def compute_agreement(preds_i, preds_j):
    return float(np.mean(preds_i == preds_j))


def compute_entropy(probs):
    return float(-np.sum(probs * np.log(probs + 1e-8), axis=1).mean())


# Main computation

def compute_cross_algorithm_agreement(
    preds_dir,
    records_path,
    test_env_idx,
    n_envs,
    n_hparams,
    n_trials=3,
    algo_a='ERM',
    algo_b='IRM',
):
    with open(records_path) as f:
        records = json.load(f)

    id_agrs     = []
    ood_agrs    = []
    entropies_a = []
    entropies_b = []
    seeds_used  = []

    max_entropy = float(np.log(2))

    print(f"\n  {'seed':>4} | "
          f"{'ID_agr':>6} | {'OOD_agr':>7} | {'OOD_dis':>7} | "
          f"{'acc_'+algo_a:>8} | {'acc_'+algo_b:>8} | "
          f"{'H('+algo_a+')':>8} | {'H('+algo_b+')':>8} | "
          f"{'lambda':>10} | {'anneal':>6}")
    print(f"  {'-'*105}")

    for seed in range(n_hparams):

        preds_a_ood = get_all_trials(preds_dir, algo_a, seed,
                                     n_trials, test_env_idx, load_predictions)
        preds_b_ood = get_all_trials(preds_dir, algo_b, seed,
                                     n_trials, test_env_idx, load_predictions)
        if not preds_a_ood or not preds_b_ood:
            continue

        ood_agr = float(np.mean([
            compute_agreement(pa, pb)
            for pa in preds_a_ood
            for pb in preds_b_ood
        ]))

        id_agr_vals = []
        for env_idx in range(n_envs):
            if env_idx == test_env_idx:
                continue
            preds_a_id = get_all_trials(preds_dir, algo_a, seed,
                                        n_trials, env_idx, load_predictions)
            preds_b_id = get_all_trials(preds_dir, algo_b, seed,
                                        n_trials, env_idx, load_predictions)
            if not preds_a_id or not preds_b_id:
                continue
            for pa in preds_a_id:
                for pb in preds_b_id:
                    id_agr_vals.append(compute_agreement(pa, pb))
        if not id_agr_vals:
            continue
        id_agr = float(np.mean(id_agr_vals))

        probs_a   = get_all_trials(preds_dir, algo_a, seed,
                                   n_trials, test_env_idx, load_probs)
        probs_b   = get_all_trials(preds_dir, algo_b, seed,
                                   n_trials, test_env_idx, load_probs)
        entropy_a = float(np.mean([compute_entropy(p) for p in probs_a])) \
                    if probs_a else None
        entropy_b = float(np.mean([compute_entropy(p) for p in probs_b])) \
                    if probs_b else None

        info_a = get_record_info(records, algo_a, seed, test_env_idx)
        info_b = get_record_info(records, algo_b, seed, test_env_idx)

        acc_a    = info_a['ood_acc'] if info_a else None
        acc_b    = info_b['ood_acc'] if info_b else None
        lambda_b = info_b['lambda']  if info_b else None
        anneal_b = info_b['anneal']  if info_b else None

        id_agrs.append(id_agr)
        ood_agrs.append(ood_agr)
        entropies_a.append(entropy_a)
        entropies_b.append(entropy_b)
        seeds_used.append(seed)

        lambda_str = f"{lambda_b:10.1f}" if lambda_b else f"{'—':>10}"
        anneal_str = f"{anneal_b:6d}"    if anneal_b else f"{'—':>6}"
        acc_a_str  = f"{acc_a:8.3f}"     if acc_a is not None else f"{'—':>8}"
        acc_b_str  = f"{acc_b:8.3f}"     if acc_b is not None else f"{'—':>8}"

        flag = ''
        if entropy_b is not None:
            if entropy_b > 0.65:
                flag = ' ← random'
            elif ood_agr < 0.7 and entropy_b < 0.6:
                flag = ' ← escaped?'

        print(f"  {seed:>4} | "
              f"{id_agr:>6.3f} | {ood_agr:>7.3f} | {1-ood_agr:>7.3f} | "
              f"{acc_a_str} | {acc_b_str} | "
              f"{entropy_a:>8.4f} | {entropy_b:>8.4f} | "
              f"{lambda_str} | {anneal_str}{flag}")

    if len(id_agrs) < 3:
        print(f"  Not enough pairs ({len(id_agrs)})")
        return None

    id_agrs  = np.array(id_agrs)
    ood_agrs = np.array(ood_agrs)

    eps        = 1e-6
    id_probit  = probit(np.clip(id_agrs,  eps, 1 - eps))
    ood_probit = probit(np.clip(ood_agrs, eps, 1 - eps))

    R, p_value = pearsonr(id_probit, ood_probit)
    reg        = linregress(id_probit, ood_probit)

    valid_mask      = np.array([e < 0.6 if e else False for e in entropies_b])
    escape_mask     = (ood_agrs < 0.7) & valid_mask
    escape_rate     = float(np.mean(escape_mask))
    escape_rate_raw = float(np.mean(ood_agrs < 0.7))

    valid_entropy_b = [e for e in entropies_b if e is not None]
    valid_entropy_a = [e for e in entropies_a if e is not None]

    results = {
        'R':               float(R),
        'slope':           float(reg.slope),
        'intercept':       float(reg.intercept),
        'p_value':         float(p_value),
        'std_error':       float(reg.stderr),
        'id_agrs':         id_agrs.tolist(),
        'ood_agrs':        ood_agrs.tolist(),
        'entropies_a':     entropies_a,
        'entropies_b':     entropies_b,
        'seeds_used':      seeds_used,
        'escape_rate':     escape_rate,
        'escape_rate_raw': escape_rate_raw,
        'n_pairs':         len(id_agrs),
        'algo_a':          algo_a,
        'algo_b':          algo_b,
        'mean_entropy_a':  float(np.mean(valid_entropy_a)),
        'mean_entropy_b':  float(np.mean(valid_entropy_b)),
        'max_entropy':     max_entropy,
    }

    print(f"\n  {'-'*105}")
    print(f"  R={R:+.3f}  slope={reg.slope:.3f}  p={p_value:.2e}  "
          f"escape_rate={escape_rate:.2f} (raw={escape_rate_raw:.2f})")
    print(f"  Mean H({algo_a})={np.mean(valid_entropy_a):.4f}  "
          f"Mean H({algo_b})={np.mean(valid_entropy_b):.4f}  "
          f"max_entropy={max_entropy:.4f}")

    return results


# Print table

def print_table(results, dataset_name, test_env_name):
    print(f"\n{'='*80}")
    print(f"  Cross-algorithm agreement — {dataset_name} (test: {test_env_name})")
    print(f"{'='*80}")
    print(f"  {'Pair':<15} {'R':>8} {'<0.3?':>6} {'slope':>8} "
          f"{'p-value':>10} {'escape':>8} {'escape_raw':>10} "
          f"{'H(b)':>8} {'n_pairs':>8}")
    print(f"  {'-'*75}")

    if results is None:
        print("  No results")
        return

    label = '✓' if results['R'] < 0.3 else '✗'
    pair  = f"{results['algo_a']} vs {results['algo_b']}"
    print(f"  {pair:<15} {results['R']:>+8.3f} {label:>6} "
          f"{results['slope']:>8.3f} {results['p_value']:>10.2e} "
          f"{results['escape_rate']:>8.2f} {results['escape_rate_raw']:>10.2f} "
          f"{results['mean_entropy_b']:>8.4f} "
          f"{results['n_pairs']:>8}")


# Entry point

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_dir',     type=str, required=True)
    parser.add_argument('--records_path',  type=str, required=True)
    parser.add_argument('--test_env_idx',  type=int, required=True)
    parser.add_argument('--n_envs',        type=int, required=True)
    parser.add_argument('--n_hparams',     type=int, default=20)
    parser.add_argument('--n_trials',      type=int, default=3)
    parser.add_argument('--algo_a',        type=str, default='ERM')
    parser.add_argument('--algo_b',        type=str, default='IRM')
    parser.add_argument('--dataset_name',  type=str, default='Dataset')
    parser.add_argument('--test_env_name', type=str, default='test')
    args = parser.parse_args()

    print(f"Computing cross-algorithm agreement "
          f"({args.algo_a} vs {args.algo_b})...")
    results = compute_cross_algorithm_agreement(
        preds_dir    = args.preds_dir,
        records_path = args.records_path,
        test_env_idx = args.test_env_idx,
        n_envs       = args.n_envs,
        n_hparams    = args.n_hparams,
        n_trials     = args.n_trials,
        algo_a       = args.algo_a,
        algo_b       = args.algo_b,
    )
    print_table(results, args.dataset_name, args.test_env_name)