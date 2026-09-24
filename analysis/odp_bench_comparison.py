# analysis/odp_bench_comparison.py
"""
ODP-Bench evaluation: surrogate scores as model rankers.

Scores:
  ATC      - Average Thresholded Confidence (Garg et al. 2022)
  DOC      - Difference of Confidence: val_conf - test_conf (lower = better)
  NucNorm  - Nuclear norm of test softmax matrix / N
  MDE      - Mean Dispersion Energy: -mean(log(sum(exp(p/T))))
  Disp     - Pseudo-label dispersion: mean distance of per-class centroids
  Agr      - Agreement: mean pairwise argmax match with same-algo seeds
  CrA      - CrossAgr: mean argmax match with ERM seeds, label-free
  TV       - TrainVal: mean val accuracy on training envs (IID baseline)
  MMD      - Maximum Mean Discrepancy of OOD softmax vs pooled ERM, label-free
  Wass     - Sinkhorn Wasserstein of OOD softmax vs pooled ERM, label-free
  PAD      - Proxy A-Distance of OOD softmax vs pooled ERM, label-free

CrA (our contribution) — fully label-free model selection:
  For each hyperparameter seed of a DG algorithm, crossagr = mean argmax
  agreement between that seed's predictions and ERM's predictions on the
  OOD test env. The question is which direction to select in: does the
  seed that resembles ERM the MOST make the best pick, or the one that
  resembles ERM the LEAST? That direction (the "sign") is decided without
  ever touching oracle test labels, via a majority vote:

  For each available DG algorithm, compute Cross-R: over entropy-filtered
  (ERM seed, DG seed) pairs, fit ID agreement (training envs) vs OOD
  agreement (test env) in probit space and take the Pearson R.

    Cross-R >= crossr_threshold -> that algorithm votes misspecified
    Cross-R <  crossr_threshold -> that algorithm votes well-specified

  The regime for the whole benchmark/environment is the MAJORITY vote
  across all DG algorithms, not any single algorithm's Cross-R. A single
  algorithm's optimization quirks (e.g. IRM's bimodal seed population)
  can flip its own Cross-R sign without changing what every other
  algorithm sees — the regime is a property of the benchmark, not of
  any one algorithm, so the vote is computed once per dataset and
  applied to every algorithm's CrA selection:

    misspecified   -> pick the seed that resembles ERM the MOST (max crossagr)
    well-specified -> pick the seed that resembles ERM the LEAST (min crossagr)

  This is fully label-free (agreement never needs OOD ground truth) but
  requires in-distribution predictions, unlike the OOD-only crossagr score
  itself. The "CrA r" column (Spearman correlation of crossagr with oracle
  accuracy) and the Oracle column are shown only to validate the method —
  they use labels for reporting, never for the selection itself.

MMD / Wass / PAD (also label-free) — same regime sign, applied to the full
  predictive distribution instead of just the argmax:
    Each low-entropy DG seed's OOD softmax matrix is compared against the
    pooled OOD softmax of ERM's own low-entropy seeds (same entropy filter
    as the Cross-R vote). This gives a continuous distributional distance
    to ERM, rather than crossagr's discrete argmax-agreement rate.
      misspecified   -> pick the seed CLOSEST to ERM (min distance)
      well-specified -> pick the seed FARTHEST from ERM (max distance)
    (the mirror image of CrA's max/min, since these are distances, not
    similarities). High-entropy seeds are excluded rather than scored.

Usage:
  python analysis/odp_bench_comparison.py [--pacs_data_dir PATH] [--per_class]
                                           [--crossr_threshold R]
"""

import os, sys, json, argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DomainBed'))

_cross_agr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _cross_agr_dir)
from utils import (
    CONFIGS, ALGOS, N_HPARAMS, N_TRIALS,
    get_test_labels, load_probs, load_preds, get_records,
    compute_entropy, fit_line, agreement_scores,
)
from scores import (
    atc_threshold, atc_score, doc_score,
    nuclear_norm_score, mde_score, dispersion_score,
)
from distance_metrics import compute_mmd, compute_wasserstein, compute_pad

# ---------------------------------------------------------------------------
# Per-seed metric extraction
# ---------------------------------------------------------------------------

def extract_seed_metrics(cfg, records, algo, true_labels=None, erm_probs=None):
    test_env  = cfg['test_env_idx']
    n_envs    = cfg['n_envs']
    preds_dir = cfg['preds_dir']
    val_envs  = [e for e in range(n_envs) if e != test_env]
    n_classes = len(cfg['class_names'])
    seed_data = {}

    for hpseed in range(N_HPARAMS):
        recs_hp = get_records(records, algo, hpseed)
        if not recs_hp:
            continue

        test_probs_trials = []
        val_probs_trials  = []
        val_acc_list      = []
        atc_trials        = []

        for trial in range(N_TRIALS):
            tp = load_probs(preds_dir, algo, hpseed, trial, test_env)
            if tp is None:
                continue
            test_probs_trials.append(tp)

            rec_t = get_records(records, algo, hpseed, trial)
            rec_t = rec_t[0] if rec_t else None
            vp_parts, va_parts = [], []
            for ve in val_envs:
                vp = load_probs(preds_dir, algo, hpseed, trial, ve)
                if vp is not None and rec_t is not None:
                    va = rec_t.get(f'env{ve}_out_acc')
                    if va is not None:
                        vp_parts.append(vp); va_parts.append(va)
            if vp_parts:
                vp_pool = np.vstack(vp_parts)
                va_mean = float(np.mean(va_parts))
                val_probs_trials.append(vp_pool)
                val_acc_list.append(va_mean)
                t = atc_threshold(vp_pool, va_mean)
                atc_trials.append(atc_score(tp, t))

        if not test_probs_trials:
            continue

        # pooled_test: all trials' rows stacked together (not averaged).
        # Used for every score that treats predictions as a SET/distribution
        # (entropy, nuc_norm, mde, dispersion, doc's test side, and the
        # MMD/Wasserstein/PAD point clouds) — valid whether or not trials
        # share the same underlying examples, since it never assumes row i
        # of one trial corresponds to row i of another. This is also more
        # correct than an elementwise mean even in single-split mode: the
        # entropy of an averaged distribution is a different (and less
        # meaningful) quantity than the average entropy of individual
        # predictions.
        #
        # mean_test: the old elementwise average, kept ONLY for the
        # per-class breakdown below, which pairs predictions index-by-index
        # against a single true_labels array — that alignment requires all
        # trials to share the same physical examples, which is only true in
        # single-split mode (true_labels is None under per-trial splits, so
        # this path is simply unused there).
        pooled_test = np.vstack(test_probs_trials)
        mean_test   = np.mean(test_probs_trials, axis=0)
        mean_val    = np.vstack(val_probs_trials) if val_probs_trials else None

        oracle_accs = [r[f'env{test_env}_out_acc'] for r in recs_hp
                       if f'env{test_env}_out_acc' in r]
        oracle     = float(np.mean(oracle_accs)) if oracle_accs else None
        oracle_std = (float(np.std(oracle_accs)) if len(oracle_accs) > 1
                      else (0.0 if oracle_accs else None))
        tv_accs  = [r.get(f'env{e}_out_acc') for r in recs_hp
                    for e in val_envs if r.get(f'env{e}_out_acc') is not None]
        trainval = float(np.mean(tv_accs)) if tv_accs else None

        # Per-class breakdown
        per_class = {}
        if true_labels is not None and mean_val is not None:
            t_global  = atc_threshold(mean_val, float(np.mean(val_acc_list)))
            preds_all = mean_test.argmax(axis=1)
            for c in range(n_classes):
                mask = true_labels == c
                if mask.sum() == 0:
                    continue
                cra_c = None
                if erm_probs and hpseed in erm_probs:
                    cra_c = float((preds_all[mask] ==
                                   erm_probs[hpseed][mask].argmax(axis=1)).mean())
                per_class[c] = {
                    'atc_c':    float((mean_test[mask].max(axis=1) > t_global).mean()),
                    'oracle_c': float((preds_all[mask] == c).mean()),
                    'cra_c':    cra_c,
                    'agr_c':    None,
                }

        seed_data[hpseed] = {
            'atc':        float(np.mean(atc_trials)) if atc_trials else None,
            'doc':        doc_score(mean_val, pooled_test) if mean_val is not None else None,
            'entropy':    compute_entropy(pooled_test),
            'nuc_norm':   nuclear_norm_score(pooled_test),
            'mde':        mde_score(pooled_test),
            'dispersion': dispersion_score(pooled_test),
            'agreement':  None,
            'crossagr':   None,
            'trainval':   trainval,
            'oracle':     oracle,
            'oracle_std': oracle_std,
            'test_probs': pooled_test,
            'test_probs_trials': test_probs_trials,
            'test_probs_mean_for_per_class': mean_test,
            'per_class':  per_class,
        }

    # Agreement (same-algo, across seeds) — computed per trial (same trial
    # index = same physical held-out split, shared across every seed) then
    # averaged, so this stays valid under per-trial splits: it never
    # compares rows from different trials against each other.
    valid_trials = {s: d['test_probs_trials'] for s, d in seed_data.items()
                    if d.get('test_probs_trials')}
    if len(valid_trials) >= 2:
        per_trial_scores = {s: [] for s in valid_trials}
        for trial in range(N_TRIALS):
            trial_probs = {s: tp[trial] for s, tp in valid_trials.items()
                           if trial < len(tp)}
            if len(trial_probs) < 2:
                continue
            for s, sc in agreement_scores(trial_probs).items():
                per_trial_scores[s].append(sc)
        for s, scs in per_trial_scores.items():
            seed_data[s]['agreement'] = float(np.mean(scs)) if scs else None

        if true_labels is not None:
            # Single-split only (true_labels is None under per-trial splits,
            # matching per_class's guard above) — must use the same
            # elementwise-averaged, true_labels-aligned array as the
            # per_class block above, not the pooled (3x-length) test_probs.
            valid_probs = {s: d['test_probs_mean_for_per_class']
                           for s, d in seed_data.items()
                           if d.get('test_probs_mean_for_per_class') is not None}
            for c in range(n_classes):
                mask = true_labels == c
                if mask.sum() < 2:
                    continue
                agr_c = agreement_scores({s: p[mask] for s, p in valid_probs.items()})
                for s, sc in agr_c.items():
                    if c in seed_data[s]['per_class']:
                        seed_data[s]['per_class'][c]['agr_c'] = sc

    # CrossAgr (vs ERM, preds-based)
    erm_preds_flat = {}
    for hpseed in range(N_HPARAMS):
        for trial in range(N_TRIALS):
            p = load_preds(preds_dir, 'ERM', hpseed, trial, test_env)
            if p is not None:
                erm_preds_flat.setdefault(hpseed, {})[trial] = p

    for hpseed, d in seed_data.items():
        agr_vals = []
        for trial in range(N_TRIALS):
            p_algo = load_preds(preds_dir, algo, hpseed, trial, test_env)
            if p_algo is None:
                continue
            for erm_seed, erm_t in erm_preds_flat.items():
                if trial in erm_t:
                    agr_vals.append(float((p_algo == erm_t[trial]).mean()))
        d['crossagr'] = float(np.mean(agr_vals)) if agr_vals else None

    return seed_data

# ---------------------------------------------------------------------------
# Cross-distance to ERM (MMD / Wasserstein / PAD) — same idea as crossagr,
# but on the full OOD softmax distribution instead of just the argmax.
# ---------------------------------------------------------------------------

def build_erm_pool(erm_seed_data, entropy_threshold=0.9):
    """
    Pool the OOD softmax rows of ERM's low-entropy seeds into one reference
    distribution for MMD/Wasserstein/PAD comparisons. Same entropy filter
    as compute_regime_sign's _valid_seeds (relative to this pool's own max).
    """
    entropies = {s: d['entropy'] for s, d in erm_seed_data.items()
                 if d.get('entropy') is not None}
    if not entropies:
        return None
    max_h = max(max(entropies.values()), 1e-6)
    pools = [d['test_probs'] for s, d in erm_seed_data.items()
             if d.get('test_probs') is not None
             and (entropies.get(s, max_h) / max_h) < entropy_threshold]
    return np.vstack(pools) if pools else None


def add_distance_scores(seed_data, erm_pool, entropy_threshold=0.9):
    """
    For each of this algorithm's low-entropy seeds, compute MMD/Wasserstein/
    PAD between its OOD softmax distribution and the pooled ERM reference.
    High-entropy seeds (relative to this algorithm's own seed pool) are left
    unscored, same filter as compute_regime_sign's _valid_seeds.
    """
    if erm_pool is None:
        return
    entropies = {s: d['entropy'] for s, d in seed_data.items()
                 if d.get('entropy') is not None}
    if not entropies:
        return
    max_h = max(max(entropies.values()), 1e-6)
    for s, d in seed_data.items():
        if d.get('test_probs') is None or entropies.get(s) is None:
            continue
        if (entropies[s] / max_h) >= entropy_threshold:
            continue
        d['mmd_cross']  = compute_mmd(d['test_probs'], erm_pool)
        d['wass_cross'] = compute_wasserstein(d['test_probs'], erm_pool)
        d['pad_cross']  = compute_pad(d['test_probs'], erm_pool)

# ---------------------------------------------------------------------------
# Evaluation — oracle-based (research benchmarking only)
# ---------------------------------------------------------------------------

def evaluate(seed_data, score_key, oracle_key='oracle'):
    valid = {s: d for s, d in seed_data.items()
             if d.get(score_key) is not None and d.get(oracle_key) is not None}
    if len(valid) < 3:
        return None
    scores  = [d[score_key]  for d in valid.values()]
    oracles = [d[oracle_key] for d in valid.values()]
    return float(spearmanr(scores, oracles).statistic)


def evaluate_sel(seed_data, score_key, minimize=False):
    """
    Selected model OOD acc (± its trial std). minimize=True picks argmin
    (e.g. DOC, MDE). The std is the selected seed's own trial-to-trial std
    (oracle_std), not variance across the 20 hparam seeds.
    """
    valid = {s: d for s, d in seed_data.items()
             if d.get(score_key) is not None and d.get('oracle') is not None}
    if not valid:
        return None, None
    pick = min if minimize else max
    best = pick(valid, key=lambda s: valid[s][score_key])
    return float(valid[best]['oracle']), valid[best].get('oracle_std')

# ---------------------------------------------------------------------------
# Evaluation — CrA: majority-vote Cross-R regime sign
# ---------------------------------------------------------------------------

def compute_regime_sign(cfg, available_algos, entropy_threshold=0.9,
                         crossr_threshold=0.3):
    """
    Compute Cross-R for all available DG algorithms and return the
    majority-vote regime sign.

    Each algorithm casts one vote:
      Cross-R >= crossr_threshold -> misspecified vote
      Cross-R <  crossr_threshold -> well-specified vote

    The mode across all algorithms is the regime signal used for CrA
    selection direction. This is more robust than per-algorithm Cross-R
    because a single algorithm's optimization quirks (e.g. IRM's bimodal
    seed population under bad hyperparameters) can produce a spurious
    Cross-R sign even when every other algorithm agrees on the regime —
    the regime is a property of the benchmark and environment, not of
    any individual algorithm.

    Returns:
      regime        : 'misspecified', 'well-specified', or None
      votes         : dict {algo: cross_r} for algorithms that voted
      n_mis, n_well : vote counts
      erm_self_r    : Cross-R for ERM vs ERM (different-seed pairs) — the
                      baseline reference every other algorithm's Cross-R is
                      implicitly compared against. Shown for context only;
                      never cast as a vote (see module docstring for why).
    """
    preds_dir = cfg['preds_dir']
    test_env  = cfg['test_env_idx']
    n_envs    = cfg['n_envs']

    def _valid_seeds(algo):
        entropies = {}
        for seed in range(N_HPARAMS):
            hs = [compute_entropy(load_probs(preds_dir, algo, seed, trial, test_env))
                  for trial in range(N_TRIALS)
                  if load_probs(preds_dir, algo, seed, trial, test_env) is not None]
            if hs:
                entropies[seed] = float(np.mean(hs))
        if not entropies:
            return []
        max_h = max(max(entropies.values()), 1e-6)
        return [s for s, h in entropies.items() if (h / max_h) < entropy_threshold]

    def _id_ood_agr(algo_a, seed_a, algo_b, seed_b):
        id_agr_vals = []
        for trial in range(N_TRIALS):
            env_agrs = []
            for env_idx in range(n_envs):
                if env_idx == test_env:
                    continue
                pa = load_preds(preds_dir, algo_a, seed_a, trial, env_idx)
                pb = load_preds(preds_dir, algo_b, seed_b, trial, env_idx)
                if pa is not None and pb is not None:
                    env_agrs.append(float((pa == pb).mean()))
            if env_agrs:
                id_agr_vals.append(float(np.mean(env_agrs)))
        if not id_agr_vals:
            return None, None
        ood_agr_vals = []
        for trial in range(N_TRIALS):
            pa = load_preds(preds_dir, algo_a, seed_a, trial, test_env)
            pb = load_preds(preds_dir, algo_b, seed_b, trial, test_env)
            if pa is not None and pb is not None:
                ood_agr_vals.append(float((pa == pb).mean()))
        if not ood_agr_vals:
            return None, None
        return float(np.mean(id_agr_vals)), float(np.mean(ood_agr_vals))

    valid_erm = _valid_seeds('ERM')
    if len(valid_erm) < 2:
        return None, {}, 0, 0, None

    # ERM-vs-ERM reference (different-seed pairs only) — context, not a vote
    erm_id_agrs, erm_ood_agrs = [], []
    for seed_a, seed_b in combinations(valid_erm, 2):
        id_agr, ood_agr = _id_ood_agr('ERM', seed_a, 'ERM', seed_b)
        if id_agr is not None and ood_agr is not None:
            erm_id_agrs.append(id_agr)
            erm_ood_agrs.append(ood_agr)
    erm_self_r = (fit_line(erm_id_agrs, erm_ood_agrs)['R']
                  if len(erm_id_agrs) >= 2 else None)

    votes, n_mis, n_well = {}, 0, 0
    dg_algos = [a for a in ALGOS if a != 'ERM' and a in available_algos]

    for algo in dg_algos:
        valid_dg = _valid_seeds(algo)
        if len(valid_dg) < 2:
            continue

        id_agrs, ood_agrs = [], []
        for seed_a in valid_erm:
            for seed_b in valid_dg:
                id_agr, ood_agr = _id_ood_agr('ERM', seed_a, algo, seed_b)
                if id_agr is not None and ood_agr is not None:
                    id_agrs.append(id_agr)
                    ood_agrs.append(ood_agr)

        if len(id_agrs) < 2:
            continue

        cross_r = fit_line(id_agrs, ood_agrs)['R']
        votes[algo] = cross_r
        if cross_r >= crossr_threshold:
            n_mis += 1
        else:
            n_well += 1

    if n_mis == 0 and n_well == 0:
        return None, votes, 0, 0, erm_self_r

    regime = 'misspecified' if n_mis >= n_well else 'well-specified'
    return regime, votes, n_mis, n_well, erm_self_r


def evaluate_regime_sel(seed_data, score_key, regime, higher_is_similar):
    """
    Shared selection rule behind CrA and the MMD/Wasserstein/PAD cross-
    distance scores: pick the seed matching the majority-vote regime's
    implied direction relative to ERM.

      misspecified   -> pick the seed that resembles ERM the MOST
      well-specified -> pick the seed that resembles ERM the LEAST
      regime is None -> no vote available, return None

    `higher_is_similar` says whether a HIGHER score_key value means MORE
    similar to ERM (True for crossagr, an agreement score) or LESS similar
    (False for mmd_cross/wass_cross/pad_cross, distance scores).

    The sign comes from compute_regime_sign's majority vote across all DG
    algorithms on this benchmark (computed once per dataset, not per
    algorithm) — never from this algorithm's own oracle accuracy.

    Returns:
      sel_acc : OOD accuracy of the selected model
      sel_std : that seed's own trial-to-trial std (not variance across seeds)
    """
    if regime is None:
        return None, None

    valid = {s: d for s, d in seed_data.items()
             if d.get(score_key) is not None
             and d.get('oracle')  is not None}
    if len(valid) < 3:
        return None, None

    want_most_similar = (regime == 'misspecified')
    want_max = want_most_similar == higher_is_similar
    pick = max if want_max else min
    best = pick(valid, key=lambda s: valid[s][score_key])
    return float(valid[best]['oracle']), valid[best].get('oracle_std')


def evaluate_cra(seed_data, regime):
    """CrA: label-free selection via crossagr (argmax agreement with ERM)."""
    return evaluate_regime_sel(seed_data, 'crossagr', regime, higher_is_similar=True)

# ---------------------------------------------------------------------------
# Per-class evaluation
# ---------------------------------------------------------------------------

def evaluate_per_class(seed_data, score_key, c):
    rows = [(d['per_class'][c][score_key], d['per_class'][c]['oracle_c'])
            for d in seed_data.values()
            if c in d.get('per_class', {})
            and d['per_class'][c].get(score_key) is not None
            and d['per_class'][c].get('oracle_c') is not None]
    if len(rows) < 3:
        return None
    s, o = zip(*rows)
    return float(spearmanr(s, o).statistic)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pacs_data_dir', type=str,
                        default='C:/Users/Usuario/Downloads/pacs_data/pacs_data')
    parser.add_argument('--per_class', action='store_true',
                        help='Show per-class breakdown')
    parser.add_argument('--crossr_threshold', type=float, default=0.3,
                        help='Cross-R threshold for regime detection (default 0.3)')
    args = parser.parse_args()

    for cfg in CONFIGS:
        if cfg['dtype'] == 'pacs' and cfg['data_dir'] is None:
            cfg['data_dir'] = args.pacs_data_dir

    def _r(v): return f'{v:+.3f}' if v is not None else '  -  '
    def _a(v): return f'{v*100:5.1f}' if v is not None else '   - '
    def _a2(v, sd):
        """mean±std, both as percentages. sd may be missing even if v isn't."""
        if v is None:
            return f"{'-':>10}"
        if sd is None:
            return f"{v*100:5.1f}".ljust(10)
        return f"{v*100:5.1f}±{sd*100:4.1f}"

    # Table 1 — how well each score's ranking correlates with the oracle
    # (Spearman r), plus each algorithm's own Cross-R vote for context.
    hdr1 = (
        f"{'Algo':<8}  N  "
        f"|{'ATC r':>6}{'DOC r':>6}{'Nuc r':>6}{'MDE r':>6}{'Dsp r':>6}{'Agr r':>6}{'CrA r':>6}{'TV r':>6}"
        f"{'MMDr':>6}{'Wasr':>6}{'PADr':>6}"
        f"|{'CrossR':>7}"
    )
    # Table 2 — the actual OOD accuracy of the model each score selects,
    # against the oracle ceiling. Every value is the selected seed's own
    # mean OOD accuracy ± its trial-to-trial std (not variance across seeds).
    hdr2 = (
        f"{'Algo':<8}  N  "
        f"|{'ATC%':>10}{'DOC%':>10}{'Nuc%':>10}{'MDE%':>10}{'Dsp%':>10}{'Agr%':>10}{'CrA%':>10}{'TV%':>10}"
        f"{'MMD%':>10}{'Wass%':>10}{'PAD%':>10}"
        f"|{'Oracle':>10}"
    )
    W = max(len(hdr1), len(hdr2)) + 2

    print()
    print('=' * W)
    print('  ODP-Bench surrogate model-selection rankers')
    print('  Table 1: r      = Spearman correlation with oracle OOD accuracy')
    print('           CrossR = this algo\'s own Cross-R vote (Pearson R of ID/OOD')
    print('                    agr pairs in probit space) — shown for reference;')
    print('                    the regime actually used is the MAJORITY vote')
    print('                    across all DG algorithms, printed once above')
    print(f'                    CrossR >= {args.crossr_threshold} -> votes misspecified -> MAX CrA')
    print(f'                    CrossR <  {args.crossr_threshold} -> votes well-specified -> MIN CrA')
    print('           CrA r < 0 -> well-specified   CrA r > 0 -> misspecified')
    print('  Table 2: sel%   = OOD accuracy (%) of the model selected by each ranker,')
    print('                    shown as mean±std across that selected seed\'s 3')
    print('                    trials (not variance across the 20 hp seeds)')
    print('           CrA%   = our contribution: fully label-free selection —')
    print('                    sign comes from the majority-vote Cross-R regime')
    print('                    (see Table 1 / above), never from oracle labels')
    print('           Oracle = best achievable OOD accuracy (max over hp seeds)')
    print('=' * W)

    for cfg in CONFIGS:
        if not os.path.exists(cfg['records_path']):
            continue

        with open(cfg['records_path']) as f:
            records = json.load(f)

        available = set(r['algorithm'] for r in records)
        n_classes  = len(cfg['class_names'])

        true_labels = None
        if args.per_class:
            try:
                true_labels = get_test_labels(cfg)
            except Exception as e:
                print(f"  [warn] {cfg['name']}: labels unavailable ({e})")

        print()
        print(f"  {'-' * (W - 2)}")
        print(f"  {cfg['name']}")
        print(f"  {'-' * (W - 2)}")

        regime, regime_votes, n_mis, n_well, erm_self_r = compute_regime_sign(
            cfg, available_algos=available, crossr_threshold=args.crossr_threshold)
        regime_str = regime if regime is not None else 'unknown'
        erm_self_str = f"{erm_self_r:+.3f}" if erm_self_r is not None else 'n/a'
        print(f"  Regime (majority vote): {regime_str}  "
              f"(misspecified: {n_mis}, well-specified: {n_well})")
        print(f"    {'ERM (self)':<10} CrossR={erm_self_str}  "
              f"[reference baseline — not counted in the vote]")
        for algo_v, r_v in regime_votes.items():
            print(f"    {algo_v:<10} CrossR={r_v:+.3f}")

        erm_probs = None
        if true_labels is not None and 'ERM' in available:
            erm_probs = {}
            for hpseed in range(N_HPARAMS):
                trials = [load_probs(cfg['preds_dir'], 'ERM', hpseed, trial,
                                     cfg['test_env_idx'])
                          for trial in range(N_TRIALS)]
                trials = [t for t in trials if t is not None]
                if trials:
                    erm_probs[hpseed] = np.mean(trials, axis=0)

        # ---- Pass 1: compute everything once per algorithm ----
        rows = []
        erm_pool = None  # pooled OOD softmax of ERM's low-entropy seeds
        pooled_seed_data = {}  # {"algo#hpseed": seed_data entry} across all DG algos
        for algo in ALGOS:
            if algo not in available:
                continue

            seed_data = extract_seed_metrics(
                cfg, records, algo,
                true_labels=true_labels,
                erm_probs=erm_probs if algo != 'ERM' else None,
            )
            if not seed_data:
                continue

            if algo == 'ERM':
                erm_pool = build_erm_pool(seed_data)
            else:
                add_distance_scores(seed_data, erm_pool)
                for hpseed, d in seed_data.items():
                    pooled_seed_data[f"{algo}#{hpseed}"] = d

            n_seeds = len(seed_data)
            oracle_entries = [(d['oracle'], d.get('oracle_std'))
                              for d in seed_data.values()
                              if d['oracle'] is not None]
            best_oracle, best_oracle_std = (
                max(oracle_entries, key=lambda t: t[0])
                if oracle_entries else (None, None))

            # CrA rho: correlation of the raw crossagr score with oracle
            # accuracy — shown for validation only, not used for selection
            cra_rho = evaluate(seed_data, 'crossagr')

            # CrA selection: sign comes from the dataset-level majority
            # vote, not this algorithm's own Cross-R or oracle labels
            cra_sel = (evaluate_cra(seed_data, regime) if algo != 'ERM'
                       else (None, None))
            cross_r = regime_votes.get(algo)  # this algo's own vote, for display

            # MMD/Wasserstein/PAD: same regime sign, full-distribution
            # distance to pooled ERM instead of crossagr's argmax agreement
            if algo != 'ERM':
                mmd_sel  = evaluate_regime_sel(seed_data, 'mmd_cross',  regime, higher_is_similar=False)
                wass_sel = evaluate_regime_sel(seed_data, 'wass_cross', regime, higher_is_similar=False)
                pad_sel  = evaluate_regime_sel(seed_data, 'pad_cross',  regime, higher_is_similar=False)
            else:
                mmd_sel = wass_sel = pad_sel = (None, None)

            per_class_rows = []
            if true_labels is not None:
                for c in range(n_classes):
                    n_c = int((true_labels == c).sum())
                    if n_c == 0:
                        continue
                    oc_vals = [d['per_class'][c]['oracle_c']
                               for d in seed_data.values()
                               if c in d.get('per_class', {})]
                    per_class_rows.append({
                        'name':      cfg['class_names'][c],
                        'n_c':       n_c,
                        'rho_atc_c': evaluate_per_class(seed_data, 'atc_c', c),
                        'rho_agr_c': evaluate_per_class(seed_data, 'agr_c', c),
                        'rho_cra_c': evaluate_per_class(seed_data, 'cra_c', c),
                        'oracle_c':  max(oc_vals) if oc_vals else None,
                    })

            rows.append({
                'algo': algo, 'n_seeds': n_seeds,
                'rho_atc': evaluate(seed_data, 'atc'),
                'rho_doc': evaluate(seed_data, 'doc'),
                'rho_nuc': evaluate(seed_data, 'nuc_norm'),
                'rho_mde': evaluate(seed_data, 'mde'),
                'rho_dsp': evaluate(seed_data, 'dispersion'),
                'rho_agr': evaluate(seed_data, 'agreement'),
                'rho_tv':  evaluate(seed_data, 'trainval'),
                'cra_rho': cra_rho, 'cross_r': cross_r,
                'rho_mmd':  evaluate(seed_data, 'mmd_cross'),
                'rho_wass': evaluate(seed_data, 'wass_cross'),
                'rho_pad':  evaluate(seed_data, 'pad_cross'),
                'sel_atc': evaluate_sel(seed_data, 'atc'),
                'sel_doc': evaluate_sel(seed_data, 'doc',      minimize=True),
                'sel_nuc': evaluate_sel(seed_data, 'nuc_norm'),
                'sel_mde': evaluate_sel(seed_data, 'mde',      minimize=True),
                'sel_dsp': evaluate_sel(seed_data, 'dispersion'),
                'sel_agr': evaluate_sel(seed_data, 'agreement'),
                'sel_tv':  evaluate_sel(seed_data, 'trainval'),
                'cra_sel': cra_sel,
                'mmd_sel': mmd_sel, 'wass_sel': wass_sel, 'pad_sel': pad_sel,
                'best_oracle': (best_oracle, best_oracle_std),
                'per_class_rows': per_class_rows,
            })

        # ---- Overall CrA pick: pool every DG algorithm's seeds together and
        # apply the SAME regime-based extremum rule CrA uses within one
        # algorithm, just over the wider pool. This is exploratory: crossagr
        # is comparable across algorithms (same quantity vs. the same ERM
        # reference), but cross-algorithm variation may partly reflect each
        # algorithm's structural distance from ERM rather than tuning
        # quality — unlike within-algorithm CrA, this has not been
        # separately validated. Caveat, not (yet) a second contribution.
        overall_sel_acc, overall_sel_std = evaluate_regime_sel(
            pooled_seed_data, 'crossagr', regime, higher_is_similar=True)
        overall_winner = None
        if regime is not None:
            valid_pool = {k: v for k, v in pooled_seed_data.items()
                          if v.get('crossagr') is not None
                          and v.get('oracle')  is not None}
            if len(valid_pool) >= 3:
                pick = max if regime == 'misspecified' else min
                overall_winner = pick(valid_pool, key=lambda k: valid_pool[k]['crossagr'])

        # ---- Pass 2a: Table 1 — correlation with oracle ----
        print(f"  {'-' * (W - 2)}")
        print(f"  {hdr1}")
        print(f"  {'-' * (W - 2)}")
        for r in rows:
            cr_str = f"{r['cross_r']:+.3f}" if r['cross_r'] is not None else '  -  '
            print(
                f"  {r['algo']:<8} {r['n_seeds']:>2}  "
                f"|{_r(r['rho_atc']):>6}{_r(r['rho_doc']):>6}{_r(r['rho_nuc']):>6}"
                f"{_r(r['rho_mde']):>6}{_r(r['rho_dsp']):>6}{_r(r['rho_agr']):>6}"
                f"{_r(r['cra_rho']):>6}{_r(r['rho_tv']):>6}"
                f"{_r(r['rho_mmd']):>6}{_r(r['rho_wass']):>6}{_r(r['rho_pad']):>6}"
                f"|{cr_str:>7}"
            )
            _b = '      '
            for pc in r['per_class_rows']:
                print(
                    f"    -> {pc['name']:<14} n={pc['n_c']:>4}  "
                    f"|{_r(pc['rho_atc_c']):>6}{_b}{_b}{_b}{_b}"
                    f"{_r(pc['rho_agr_c']):>6}{_r(pc['rho_cra_c']):>6}{_b}"
                    f"{_b}{_b}{_b}"
                    f"|{'':>7}"
                )
        print()

        # ---- Pass 2b: Table 2 — selected OOD accuracy ----
        print(f"  {'-' * (W - 2)}")
        print(f"  {hdr2}")
        print(f"  {'-' * (W - 2)}")
        for r in rows:
            print(
                f"  {r['algo']:<8} {r['n_seeds']:>2}  "
                f"|{_a2(*r['sel_atc']):>10}{_a2(*r['sel_doc']):>10}{_a2(*r['sel_nuc']):>10}"
                f"{_a2(*r['sel_mde']):>10}{_a2(*r['sel_dsp']):>10}{_a2(*r['sel_agr']):>10}"
                f"{_a2(*r['cra_sel']):>10}{_a2(*r['sel_tv']):>10}"
                f"{_a2(*r['mmd_sel']):>10}{_a2(*r['wass_sel']):>10}{_a2(*r['pad_sel']):>10}"
                f"|{_a2(*r['best_oracle']):>10}"
            )
            for pc in r['per_class_rows']:
                oracle_c = _a(pc['oracle_c'])
                print(
                    f"    -> {pc['name']:<14} n={pc['n_c']:>4}  "
                    f"|{'':>10}{'':>10}{'':>10}{'':>10}{'':>10}{'':>10}{'':>10}{'':>10}"
                    f"{'':>10}{'':>10}{'':>10}"
                    f"|{oracle_c:>10}"
                )
        print()

        # ---- Overall CrA pick (pooled across all algorithms) ----
        if overall_winner is not None:
            algo_w, seed_w = overall_winner.split('#')
            print(f"  Overall CrA pick (pooled across all DG algorithms): "
                  f"{algo_w} hp={seed_w}  ->  {_a2(overall_sel_acc, overall_sel_std).strip()}%  "
                  f"[exploratory — see Notes]")
        else:
            print("  Overall CrA pick (pooled across all DG algorithms): unavailable (no regime)")
        print()

    print('Notes:')
    print('  ATC   = Average Thresholded Confidence (Garg et al. 2022);         sel: argmax')
    print('  DOC   = val_conf_mean - test_conf_mean (confidence gap);            sel: argmin')
    print('  Nuc   = nuclear_norm(test_probs) / N (prediction matrix structure); sel: argmax')
    print('  MDE   = -mean(log(sum(exp(p/T)))) (energy; more negative = peaked); sel: argmin')
    print('  Dsp   = mean distance of pseudo-class centroids from global centroid;sel: argmax')
    print('  Agr   = mean pairwise argmax match with same-algo seeds on OOD test; sel: argmax')
    print('  CrA   = mean argmax match with ERM seeds (crossagr), selected fully')
    print('          label-free — our contribution. Sign comes from a majority')
    print('          vote, never from oracle labels:')
    print('            each DG algorithm casts one vote from its own Cross-R —')
    print('            the Pearson R of (ID agr, OOD agr) pairs in probit space')
    print('            over entropy-filtered (ERM seed, DG seed) pairs')
    print(f'            Cross-R >= {args.crossr_threshold} -> vote misspecified')
    print(f'            Cross-R <  {args.crossr_threshold} -> vote well-specified')
    print('            the MAJORITY vote across all DG algos on the dataset sets')
    print('            the regime, applied to every algo:')
    print('              misspecified   -> pick the seed most like ERM (max crossagr)')
    print('              well-specified -> pick the seed least like ERM (min crossagr)')
    print('            (regime printed once per dataset, above the tables)')
    print('  Overall CrA pick = same regime rule as CrA, but the max/min is taken')
    print('          over ALL DG algorithms\' seeds pooled together, not within one')
    print('          algorithm. Exploratory: crossagr is directly comparable across')
    print('          algorithms (same quantity vs. the same ERM reference), but')
    print('          cross-algorithm variation may partly reflect each algorithm\'s')
    print('          structural distance from ERM rather than tuning quality — this')
    print('          has not been separately validated the way within-algorithm')
    print('          CrA has. Treat as a secondary, exploratory result.')
    print('  TV    = TrainVal: mean val acc on training envs (IID baseline);      sel: argmax')
    print('  MMD/Wass/PAD = same regime sign as CrA, but on the full OOD softmax')
    print('          distribution vs pooled ERM (entropy-filtered), label-free:')
    print('            misspecified   -> pick the seed CLOSEST to ERM (min dist)')
    print('            well-specified -> pick the seed FARTHEST from ERM (max dist)')
    print('          (mirror image of CrA\'s max/min: these are distances, not')
    print('          similarities). High-entropy seeds are excluded, not scored.')
    print('  CrossR (row) = this algo\'s own vote, shown for reference only —')
    print('           the selection sign comes from the dataset-level majority')
    print('  CrA r = Spearman correlation of crossagr with oracle accuracy —')
    print('          shown to validate the method; not used for selection')
    print('  ±sd   = std across the 3 trials of the best (oracle) seed\'s OOD acc —')
    print('          how noisy that seed\'s reported accuracy is trial-to-trial,')
    print('          not variance across the 20 hparam seeds')
    print('  --per_class  to show per-class r breakdown (loads labels from dataset)')


if __name__ == '__main__':
    main()