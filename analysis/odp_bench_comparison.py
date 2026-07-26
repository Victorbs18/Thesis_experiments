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
  CrA      - CrossAgr: mean argmax match with ERM seeds (regime signal via sign of ρ)
  TV       - TrainVal: mean val accuracy on training envs (IID baseline)

Usage:
  python analysis/odp_bench_comparison.py [--pacs_data_dir PATH] [--per_class]
"""

import os, sys, json, argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DomainBed'))

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        'name':         'ColoredMNIST (env2)',
        'records_path': 'results/coloredmnist/test_env2/cnn/random/records.json',
        'preds_dir':    'results/coloredmnist/test_env2/cnn/random/models',
        'test_env_idx': 2,
        'n_envs':       3,
        'dtype':        'coloredmnist',
        'data_dir':     './data',
        'class_names':  ['y=0 (label<5)', 'y=1 (label≥5)'],
    },
    {
        'name':         'RotatedMNIST (env5)',
        'records_path': 'results/rotatedmnist/test_env5/cnn/random/records.json',
        'preds_dir':    'results/rotatedmnist/test_env5/cnn/random/models',
        'test_env_idx': 5,
        'n_envs':       6,
        'dtype':        'rotatedmnist',
        'data_dir':     './data',
        'class_names':  [str(d) for d in range(10)],
    },
    {
        'name':         'PACS ResNet50 (env0)',
        'records_path': 'results/pacs/test_env0/resnet50/random/records.json',
        'preds_dir':    'results/pacs/test_env0/resnet50/random/models',
        'test_env_idx': 0,
        'n_envs':       4,
        'dtype':        'pacs',
        'data_dir':     None,
        'class_names':  ['dog','elephant','giraffe','guitar','horse','house','person'],
    },
    {
        'name':         'PACS ResNet50 (env1)',
        'records_path': 'results/pacs/test_env1/resnet50/random/records.json',
        'preds_dir':    'results/pacs/test_env1/resnet50/random/models',
        'test_env_idx': 1,
        'n_envs':       4,
        'dtype':        'pacs',
        'data_dir':     None,
        'class_names':  ['dog','elephant','giraffe','guitar','horse','house','person'],
    },
    {
        'name':         'PACS CLIP (env1)',
        'records_path': 'results/pacs/test_env1/clip/random/records.json',
        'preds_dir':    'results/pacs/test_env1/clip/random/models',
        'test_env_idx': 1,
        'n_envs':       4,
        'dtype':        'pacs',
        'data_dir':     None,
        'class_names':  ['dog','elephant','giraffe','guitar','horse','house','person'],
    },
    {
            'name':         'PACS CLIP (env1)',
            'records_path': 'results/pacs/test_env1/clip/random/pacs/test_env1/clip/random/records.json',
            'preds_dir':    'results/pacs/test_env1/clip/random/pacs/test_env1/clip/random/models',
            'test_env_idx': 1,
            'n_envs':       4,
            'dtype':        'pacs',
            'data_dir':     None,
            'class_names':  ['dog','elephant','giraffe','guitar','horse','house','person'],
        },
    {
            'name':         'PACS CLIP (env0)',
            'records_path': 'results/pacs/test_env0/clip/random/records.json',
            'preds_dir':    'results/pacs/test_env0/clip/random/models',
            'test_env_idx': 0,
            'n_envs':       4,
            'dtype':        'pacs',
            'data_dir':     None,
            'class_names':  ['dog','elephant','giraffe','guitar','horse','house','person'],
        },

]

ALGOS     = ['ERM', 'IRM', 'VREx', 'GroupDRO', 'CORAL', 'DANN']
N_HPARAMS = 20
N_TRIALS  = 3

# ---------------------------------------------------------------------------
# Ground-truth label loading (labels only — no images)
# ---------------------------------------------------------------------------

def get_test_labels(cfg):
    dtype    = cfg['dtype']
    test_env = cfg['test_env_idx']
    data_dir = cfg['data_dir']

    if dtype == 'coloredmnist':
        import torch
        from torchvision.datasets import MNIST
        mnist_train = MNIST(data_dir, train=True,  download=True)
        mnist_test  = MNIST(data_dir, train=False, download=True)
        labels_raw  = torch.cat([mnist_train.targets, mnist_test.targets])
        rng  = torch.Generator(); rng.manual_seed(0)
        perm = torch.randperm(len(labels_raw), generator=rng)
        labels_raw = labels_raw[perm]
        env_labels = labels_raw[test_env::3]
        torch.manual_seed(0)
        bin_labels = (env_labels < 5).long()
        noise      = (torch.rand(len(bin_labels)) < 0.25).long()
        bin_labels = (bin_labels ^ noise)
        n    = len(bin_labels)
        perm2 = np.random.RandomState(0).permutation(n)
        return bin_labels[perm2[:int(n * 0.2)]].numpy()

    elif dtype == 'rotatedmnist':
        from domainbed.datasets import RotatedMNIST as DB_RotatedMNIST
        db  = DB_RotatedMNIST(data_dir, test_envs=[test_env], hparams={})
        env = db.datasets[test_env]
        all_labels = np.array([int(env[i][1]) for i in range(len(env))])
        n    = len(all_labels)
        perm = np.random.RandomState(0).permutation(n)
        return all_labels[perm[:int(n * 0.2)]]

    elif dtype == 'pacs':
        from torchvision.datasets import ImageFolder
        from torchvision import transforms
        env_dirs = sorted(f.name for f in os.scandir(data_dir) if f.is_dir())
        dataset  = ImageFolder(os.path.join(data_dir, env_dirs[test_env]),
                               transform=transforms.ToTensor())
        all_labels = np.array(dataset.targets)
        n    = len(all_labels)
        perm = np.random.RandomState(0).permutation(n)
        return all_labels[perm[:int(n * 0.2)]]

    raise ValueError(f"Unknown dtype: {dtype}")

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_probs(preds_dir, algo, hpseed, trial, env_idx):
    fname = f"{algo}_hpseed{hpseed}_trial{trial}_env{env_idx}_probs.npy"
    path  = os.path.join(preds_dir, fname)
    return np.load(path).astype(np.float32) if os.path.exists(path) else None


def load_preds(preds_dir, algo, hpseed, trial, env_idx):
    fname = f"{algo}_hpseed{hpseed}_trial{trial}_env{env_idx}_preds.npy"
    path  = os.path.join(preds_dir, fname)
    return np.load(path) if os.path.exists(path) else None


def get_records(records, algo, hpseed, trial=None):
    out = [r for r in records
           if r['algorithm'] == algo and r['args']['hparams_seed'] == hpseed]
    if trial is not None:
        out = [r for r in out if r['args']['trial_seed'] == trial]
    return out

# ---------------------------------------------------------------------------
# Surrogate score functions
# ---------------------------------------------------------------------------

def atc_threshold(val_probs, val_acc):
    n = len(val_probs)
    k = max(1, min(n, int(round(n * float(val_acc)))))
    return float(np.sort(val_probs.max(axis=1))[::-1][k - 1])


def atc_score(test_probs, threshold):
    return float((test_probs.max(axis=1) > threshold).mean())


def doc_score(val_probs, test_probs):
    """Confidence gap: val_conf - test_conf. Lower = smaller gap = better."""
    return float(val_probs.max(axis=1).mean() - test_probs.max(axis=1).mean())


def nuclear_norm_score(test_probs):
    """Nuclear norm / N. Higher = more structured prediction matrix."""
    return float(np.linalg.norm(test_probs, 'nuc') / len(test_probs))


def mde_score(test_probs, T=1.0):
    """Mean Dispersion Energy. More negative = more peaked distributions."""
    return float(-np.mean(T * np.log(np.sum(np.exp(test_probs / T), axis=1))))


def dispersion_score(test_probs):
    """Mean distance of pseudo-label class centroids from global centroid."""
    pseudo   = test_probs.argmax(axis=1)
    centroid = test_probs.mean(axis=0)
    scores   = []
    for c in range(test_probs.shape[1]):
        mask = pseudo == c
        if mask.sum() == 0:
            continue
        scores.append(np.linalg.norm(test_probs[mask].mean(axis=0) - centroid))
    return float(np.mean(scores)) if scores else 0.0


def _agreement_scores(probs_dict):
    seeds = list(probs_dict.keys())
    preds = {s: probs_dict[s].argmax(axis=1) for s in seeds}
    scores = {}
    for s in seeds:
        agrs = [(preds[s] == preds[t]).mean() for t in seeds if t != s]
        scores[s] = float(np.mean(agrs)) if agrs else 0.0
    return scores

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
        val_probs_trials  = []   # pooled val probs per trial
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

        mean_test = np.mean(test_probs_trials, axis=0)
        mean_val  = np.vstack(val_probs_trials) if val_probs_trials else None

        oracle_accs = [r[f'env{test_env}_out_acc'] for r in recs_hp
                       if f'env{test_env}_out_acc' in r]
        oracle   = float(np.mean(oracle_accs)) if oracle_accs else None
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
            'doc':        doc_score(mean_val, mean_test) if mean_val is not None else None,
            'nuc_norm':   nuclear_norm_score(mean_test),
            'mde':        mde_score(mean_test),
            'dispersion': dispersion_score(mean_test),
            'agreement':  None,
            'crossagr':   None,
            'trainval':   trainval,
            'oracle':     oracle,
            'test_probs': mean_test,
            'per_class':  per_class,
        }

    # Agreement (same-algo, across seeds)
    valid_probs = {s: d['test_probs'] for s, d in seed_data.items()
                   if d['test_probs'] is not None}
    if len(valid_probs) >= 2:
        for s, sc in _agreement_scores(valid_probs).items():
            seed_data[s]['agreement'] = sc
        if true_labels is not None:
            for c in range(n_classes):
                mask = true_labels == c
                if mask.sum() < 2:
                    continue
                agr_c = _agreement_scores({s: p[mask] for s, p in valid_probs.items()})
                for s, sc in agr_c.items():
                    if c in seed_data[s]['per_class']:
                        seed_data[s]['per_class'][c]['agr_c'] = sc

    # CrossAgr (vs ERM, preds-based for whole-dataset; probs-based for per-class)
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(seed_data, score_key, oracle_key='oracle'):
    valid = {s: d for s, d in seed_data.items()
             if d.get(score_key) is not None and d.get(oracle_key) is not None}
    if len(valid) < 3:
        return None
    scores  = [d[score_key]  for d in valid.values()]
    oracles = [d[oracle_key] for d in valid.values()]
    return float(spearmanr(scores, oracles).statistic)


def evaluate_cra_aware(seed_data):
    valid = {s: d for s, d in seed_data.items()
             if d.get('crossagr') is not None and d.get('oracle') is not None}
    if len(valid) < 3:
        return None, None
    scores  = [d['crossagr'] for d in valid.values()]
    oracles = [d['oracle']   for d in valid.values()]
    rho     = float(spearmanr(scores, oracles).statistic)
    pick    = max if rho >= 0 else min
    best    = pick(valid, key=lambda s: valid[s]['crossagr'])
    return rho, float(valid[best]['oracle'])


def evaluate_sel(seed_data, score_key, minimize=False):
    """Selected model OOD acc. minimize=True picks argmin (e.g. DOC, MDE)."""
    valid = {s: d for s, d in seed_data.items()
             if d.get(score_key) is not None and d.get('oracle') is not None}
    if not valid:
        return None
    pick = min if minimize else max
    best = pick(valid, key=lambda s: valid[s][score_key])
    return float(valid[best]['oracle'])


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


def evaluate_cra_per_class(seed_data, c):
    rows = [(d['per_class'][c]['cra_c'], d['per_class'][c]['oracle_c'])
            for d in seed_data.values()
            if c in d.get('per_class', {})
            and d['per_class'][c].get('cra_c') is not None
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
    args = parser.parse_args()

    for cfg in CONFIGS:
        if cfg['dtype'] == 'pacs' and cfg['data_dir'] is None:
            cfg['data_dir'] = args.pacs_data_dir

    def _r(v): return f'{v:+.3f}' if v is not None else '  ─  '
    def _a(v): return f'{v*100:5.1f}' if v is not None else '   ─ '

    # Left label col: 13 chars (algo≤8 + 2 spaces + N≤2 + space)
    # ρ block: │ + 8×6 = 49    sel block: │ + 8×5 = 41    oracle: │+6 = 7
    # Total ≈ 13 + 49 + 41 + 7 = 110 chars
    hdr = (
        f"{'Algo':<8}  N  "
        f"│{'ATC ρ':>6}{'DOC ρ':>6}{'Nuc ρ':>6}{'MDE ρ':>6}{'Dsp ρ':>6}{'Agr ρ':>6}{'CrA ρ':>6}{'TV ρ':>6}"
        f"│{'ATC%':>5}{'DOC%':>5}{'Nuc%':>5}{'MDE%':>5}{'Dsp%':>5}{'Agr%':>5}{'CrA%':>5}{'TV%':>5}"
        f"│{'Oracle':>6}"
    )
    W = len(hdr) + 2

    print()
    print('=' * W)
    print('  ODP-Bench surrogate model-selection rankers')
    print('  ρ = Spearman correlation with oracle OOD accuracy')
    print('  sel% = OOD accuracy (%) of the model selected by each ranker')
    print('  CrA ρ < 0 → well-specified (IRM regime)   CrA ρ > 0 → misspecified (ERM regime)')
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
        print(f"  {'─' * (W - 2)}")
        print(f"  {cfg['name']}")
        print(f"  {'─' * (W - 2)}")
        print(f"  {hdr}")
        print(f"  {'─' * (W - 2)}")

        # ERM mean probs for per-class CrossAgr
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

            n_seeds = len(seed_data)
            oracles = [d['oracle'] for d in seed_data.values() if d['oracle'] is not None]
            cra_rho, cra_sel = evaluate_cra_aware(seed_data)

            rho_atc = evaluate(seed_data, 'atc')
            rho_doc = evaluate(seed_data, 'doc')
            rho_nuc = evaluate(seed_data, 'nuc_norm')
            rho_mde = evaluate(seed_data, 'mde')
            rho_dsp = evaluate(seed_data, 'dispersion')
            rho_agr = evaluate(seed_data, 'agreement')
            rho_tv  = evaluate(seed_data, 'trainval')

            sel_atc = evaluate_sel(seed_data, 'atc')
            sel_doc = evaluate_sel(seed_data, 'doc',      minimize=True)
            sel_nuc = evaluate_sel(seed_data, 'nuc_norm')
            sel_mde = evaluate_sel(seed_data, 'mde',      minimize=True)
            sel_dsp = evaluate_sel(seed_data, 'dispersion')
            sel_agr = evaluate_sel(seed_data, 'agreement')
            sel_tv  = evaluate_sel(seed_data, 'trainval')

            print(
                f"  {algo:<8} {n_seeds:>2}  "
                f"│{_r(rho_atc):>6}{_r(rho_doc):>6}{_r(rho_nuc):>6}"
                f"{_r(rho_mde):>6}{_r(rho_dsp):>6}{_r(rho_agr):>6}{_r(cra_rho):>6}{_r(rho_tv):>6}"
                f"│{_a(sel_atc):>5}{_a(sel_doc):>5}{_a(sel_nuc):>5}"
                f"{_a(sel_mde):>5}{_a(sel_dsp):>5}{_a(sel_agr):>5}{_a(cra_sel):>5}{_a(sel_tv):>5}"
                f"│{_a(max(oracles) if oracles else None):>6}"
            )

            if true_labels is not None:
                for c in range(n_classes):
                    n_c = int((true_labels == c).sum())
                    if n_c == 0:
                        continue
                    oc_vals = [d['per_class'][c]['oracle_c']
                               for d in seed_data.values()
                               if c in d.get('per_class', {})]
                    rho_atc_c = evaluate_per_class(seed_data, 'atc_c', c)
                    rho_agr_c = evaluate_per_class(seed_data, 'agr_c', c)
                    rho_cra_c = evaluate_cra_per_class(seed_data, c)
                    oracle_c  = _a(max(oc_vals) if oc_vals else None)
                    _b = '      '  # blank, 6 chars
                    print(
                        f"    ↳ {cfg['class_names'][c]:<14} n={n_c:>4}  "
                        f"│{_r(rho_atc_c):>6}{_b}{_b}{_b}{_b}"
                        f"{_r(rho_agr_c):>6}{_r(rho_cra_c):>6}{_b}"
                        f"│{'':>5}{'':>5}{'':>5}{'':>5}{'':>5}{'':>5}{'':>5}{'':>5}"
                        f"│{oracle_c:>6}"
                    )
        print()

    print('Notes:')
    print('  ATC  = Average Thresholded Confidence (Garg et al. 2022);        sel: argmax')
    print('  DOC  = val_conf_mean - test_conf_mean (confidence gap);           sel: argmin')
    print('  Nuc  = nuclear_norm(test_probs) / N (prediction matrix structure);sel: argmax')
    print('  MDE  = -mean(log(sum(exp(p/T)))) (energy; more negative = peaked);sel: argmin')
    print('  Dsp  = mean distance of pseudo-class centroids from global centroid;sel: argmax')
    print('  Agr  = mean pairwise argmax match with same-algo seeds on OOD test;sel: argmax')
    print('  CrA* = mean argmax match with ERM seeds; sel uses sign(ρ) for direction')
    print('  TV   = TrainVal: mean val acc on training envs (IID baseline);     sel: argmax')
    print('  sel% = OOD accuracy of the model selected by that ranker (pick direction per score)')
    print('  --per_class to show per-class ρ breakdown (loads labels from dataset)')


if __name__ == '__main__':
    main()
