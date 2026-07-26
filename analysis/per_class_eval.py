# analysis/per_class_eval.py
"""
Per-class breakdown of ATC, Agreement, and Cross-Agreement (ERM vs algo_b).

For each dataset, prints one row per class showing:
  - N_test     : number of test samples of that class
  - Acc        : mean OOD accuracy on that class (oracle, averaged over seeds/trials)
  - ATC        : mean max-confidence on test samples of that class
  - Agr        : mean pairwise argmax agreement within algorithm, per class
  - CrossAgr   : mean ERM vs algo_b argmax agreement per class (same hparams_seed)

Ground-truth labels are reconstructed from the dataset loader with the same
fixed seed=0 split used during training, so the per-class grouping is exact.

Usage:
  python analysis/per_class_eval.py
  python analysis/per_class_eval.py --algo_b IRM --pacs_data_dir C:/path/to/pacs
"""

import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DomainBed'))

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        'name':          'ColoredMNIST (env2)',
        'records_path':  'results/coloredmnist/test_env2/cnn/random/records.json',
        'preds_dir':     'results/coloredmnist/test_env2/cnn/random/models',
        'test_env_idx':  2,
        'n_envs':        3,
        'dtype':         'coloredmnist',
        'data_dir':      './data',
        'class_names':   ['label_0 (y<5)', 'label_1 (y≥5)'],
    },
    {
        'name':          'RotatedMNIST (env5)',
        'records_path':  'results/rotatedmnist/test_env5/cnn/random/records.json',
        'preds_dir':     'results/rotatedmnist/test_env5/cnn/random/models',
        'test_env_idx':  5,
        'n_envs':        6,
        'dtype':         'rotatedmnist',
        'data_dir':      './data',
        'class_names':   [str(d) for d in range(10)],
    },
    {
        'name':          'PACS ResNet50 (env0)',
        'records_path':  'results/pacs/test_env0/resnet50/random/records.json',
        'preds_dir':     'results/pacs/test_env0/resnet50/random/models',
        'test_env_idx':  0,
        'n_envs':        4,
        'dtype':         'pacs',
        'data_dir':      None,   # set via --pacs_data_dir
        'class_names':   ['dog','elephant','giraffe','guitar','horse','house','person'],
    },
    {
        'name':          'PACS ResNet50 (env1)',
        'records_path':  'results/pacs/test_env1/resnet50/random/records.json',
        'preds_dir':     'results/pacs/test_env1/resnet50/random/models',
        'test_env_idx':  1,
        'n_envs':        4,
        'dtype':         'pacs',
        'data_dir':      None,
        'class_names':   ['dog','elephant','giraffe','guitar','horse','house','person'],
    },
]

N_HPARAMS = 20
N_TRIALS  = 3

# ---------------------------------------------------------------------------
# Ground-truth label reconstruction
# ---------------------------------------------------------------------------

def get_test_labels(cfg):
    """
    Reconstruct ground-truth labels for the test env's out_split (20% holdout,
    seed=0), exactly as done during training. Returns numpy array of shape (N,).
    """
    dtype   = cfg['dtype']
    test_env = cfg['test_env_idx']
    data_dir = cfg['data_dir']

    if dtype == 'coloredmnist':
        import torch
        from torchvision.datasets import MNIST

        def _bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def _xor(a, b):
            return (a - b).abs()

        mnist_train = MNIST(data_dir, train=True,  download=True)
        mnist_test  = MNIST(data_dir, train=False, download=True)
        images_raw  = torch.cat([mnist_train.data, mnist_test.data]).float()
        labels_raw  = torch.cat([mnist_train.targets, mnist_test.targets])

        rng  = torch.Generator(); rng.manual_seed(0)
        perm = torch.randperm(len(images_raw), generator=rng)
        labels_raw = labels_raw[perm]

        # env2 = indices 2, 5, 8, ... of permuted data
        environments = [0.1, 0.2, 0.9]
        env_labels_raw = labels_raw[test_env::len(environments)]

        # Replicate _color_dataset label transformation (binary)
        torch.manual_seed(0)   # fix seed so bernoulli matches training
        bin_labels = (env_labels_raw < 5).float()
        bin_labels = _xor(bin_labels, _bernoulli(0.25, len(bin_labels)))
        bin_labels = bin_labels.long()

        # Replicate split_env (seed=0)
        n   = len(bin_labels)
        rng2 = np.random.RandomState(0)
        perm2 = rng2.permutation(n)
        n_val = int(n * 0.2)
        val_idx = perm2[:n_val]
        return bin_labels[val_idx].numpy()

    elif dtype == 'rotatedmnist':
        from domainbed.datasets import RotatedMNIST as DB_RotatedMNIST

        db  = DB_RotatedMNIST(data_dir, test_envs=[test_env], hparams={})
        env = db.datasets[test_env]  # torch Subset

        # Collect all labels in this env
        all_labels = np.array([int(env[i][1]) for i in range(len(env))])

        # Replicate split_env_subset (seed=0)
        n   = len(all_labels)
        rng = np.random.RandomState(0)
        perm = rng.permutation(n)
        n_val = int(n * 0.2)
        val_idx = perm[:n_val]
        return all_labels[val_idx]

    elif dtype == 'pacs':
        import os
        from torchvision.datasets import ImageFolder
        from torchvision import transforms

        env_dirs = sorted([f.name for f in os.scandir(data_dir) if f.is_dir()])
        env_path = os.path.join(data_dir, env_dirs[test_env])
        dataset  = ImageFolder(env_path, transform=transforms.ToTensor())
        all_labels = np.array(dataset.targets)

        n   = len(all_labels)
        rng = np.random.RandomState(0)
        perm = rng.permutation(n)
        n_val = int(n * 0.2)
        val_idx = perm[:n_val]
        return all_labels[val_idx]

    else:
        raise ValueError(f"Unknown dtype: {dtype}")

# ---------------------------------------------------------------------------
# Probs loading
# ---------------------------------------------------------------------------

def load_probs(preds_dir, algo, hpseed, trial, env_idx):
    fname = f"{algo}_hpseed{hpseed}_trial{trial}_env{env_idx}_probs.npy"
    path  = os.path.join(preds_dir, fname)
    return np.load(path).astype(np.float32) if os.path.exists(path) else None

# ---------------------------------------------------------------------------
# Per-class metric computation
# ---------------------------------------------------------------------------

def per_class_acc(probs_list, true_labels, n_classes):
    """Per-class accuracy: mean over seeds×trials, then per class."""
    accs = np.zeros(n_classes)
    counts = np.zeros(n_classes)
    for probs in probs_list:
        preds = probs.argmax(axis=1)
        for c in range(n_classes):
            mask = true_labels == c
            if mask.sum() > 0:
                accs[c]   += (preds[mask] == c).mean()
                counts[c] += 1
    return np.where(counts > 0, accs / counts, np.nan)


def per_class_atc(probs_list, val_probs_list, val_accs, true_labels, n_classes):
    """
    Mean max-confidence on test samples of each class.
    Threshold is learned on val (pooled), same as in odp_bench_comparison.
    """
    # Learn threshold from val
    val_pool = np.vstack(val_probs_list)
    val_acc  = float(np.mean(val_accs))
    n_val    = len(val_pool)
    k        = max(1, min(n_val, int(round(n_val * val_acc))))
    threshold = np.sort(val_pool.max(axis=1))[::-1][k - 1]

    atcs = np.zeros(n_classes)
    for probs in probs_list:
        conf = probs.max(axis=1)
        for c in range(n_classes):
            mask = true_labels == c
            if mask.sum() > 0:
                atcs[c] += (conf[mask] > threshold).mean()
    return atcs / max(len(probs_list), 1)


def per_class_agreement(probs_per_seed, true_labels, n_classes):
    """
    Mean pairwise argmax agreement per class, averaged over all seed pairs.
    probs_per_seed: list of (N, C) arrays (one per valid hpseed, mean over trials).
    """
    if len(probs_per_seed) < 2:
        return np.full(n_classes, np.nan)
    preds = [p.argmax(axis=1) for p in probs_per_seed]
    agrs  = np.zeros(n_classes)
    count = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            for c in range(n_classes):
                mask = true_labels == c
                if mask.sum() > 0:
                    agrs[c] += (preds[i][mask] == preds[j][mask]).mean()
            count += 1
    return agrs / max(count, 1)


def per_class_cross_agreement(erm_probs_per_seed, algob_probs_per_seed,
                               true_labels, n_classes):
    """
    ERM vs algo_b mean argmax agreement per class, paired by hpseed.
    erm_probs_per_seed, algob_probs_per_seed: dicts {hpseed: (N, C)}.
    """
    common = sorted(set(erm_probs_per_seed) & set(algob_probs_per_seed))
    if not common:
        return np.full(n_classes, np.nan)
    agrs  = np.zeros(n_classes)
    count = 0
    for s in common:
        p_erm  = erm_probs_per_seed[s].argmax(axis=1)
        p_algob = algob_probs_per_seed[s].argmax(axis=1)
        for c in range(n_classes):
            mask = true_labels == c
            if mask.sum() > 0:
                agrs[c] += (p_erm[mask] == p_algob[mask]).mean()
        count += 1
    return agrs / max(count, 1)

# ---------------------------------------------------------------------------
# Build per-seed mean probs dict for one algorithm
# ---------------------------------------------------------------------------

def collect_mean_probs(cfg, records, algo):
    """Returns {hpseed: mean_probs_over_trials (N, C)} for valid seeds."""
    result = {}
    for hpseed in range(N_HPARAMS):
        recs = [r for r in records
                if r['algorithm'] == algo and r['args']['hparams_seed'] == hpseed]
        if not recs:
            continue
        trials = []
        for trial in range(N_TRIALS):
            tp = load_probs(cfg['preds_dir'], algo, hpseed, trial, cfg['test_env_idx'])
            if tp is not None:
                trials.append(tp)
        if trials:
            result[hpseed] = np.mean(trials, axis=0)
    return result


def collect_val_probs_and_accs(cfg, records, algo, hpseed):
    """Collect val probs and val accs across trials × val envs."""
    val_envs = [e for e in range(cfg['n_envs']) if e != cfg['test_env_idx']]
    vp_list, va_list = [], []
    for trial in range(N_TRIALS):
        recs_t = [r for r in records
                  if r['algorithm'] == algo
                  and r['args']['hparams_seed'] == hpseed
                  and r['args']['trial_seed'] == trial]
        if not recs_t:
            continue
        rec = recs_t[0]
        for ve in val_envs:
            vp = load_probs(cfg['preds_dir'], algo, hpseed, trial, ve)
            if vp is not None:
                va = rec.get(f'env{ve}_out_acc')
                if va is not None:
                    vp_list.append(vp)
                    va_list.append(va)
    return vp_list, va_list


def collect_test_probs_list(cfg, records, algo):
    """All (hpseed, trial) test probs as a flat list."""
    out = []
    for hpseed in range(N_HPARAMS):
        recs = [r for r in records
                if r['algorithm'] == algo and r['args']['hparams_seed'] == hpseed]
        if not recs:
            continue
        for trial in range(N_TRIALS):
            tp = load_probs(cfg['preds_dir'], algo, hpseed, trial, cfg['test_env_idx'])
            if tp is not None:
                out.append(tp)
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo_b',       type=str, default='IRM',
                        help='Algorithm to compare against ERM for cross-agreement')
    parser.add_argument('--pacs_data_dir', type=str,
                        default='C:/Users/Usuario/Downloads/pacs_data/pacs_data')
    args = parser.parse_args()

    # Inject PACS data dir
    for cfg in CONFIGS:
        if cfg['dtype'] == 'pacs' and cfg['data_dir'] is None:
            cfg['data_dir'] = args.pacs_data_dir

    for cfg in CONFIGS:
        if not os.path.exists(cfg['records_path']):
            print(f"\n[skip] {cfg['name']}: records not found")
            continue

        print(f"\n{'='*80}")
        print(f"  {cfg['name']}")
        print(f"{'='*80}")

        with open(cfg['records_path']) as f:
            records = json.load(f)

        available_algos = set(r['algorithm'] for r in records)

        # Reconstruct ground-truth labels for test env out_split
        try:
            true_labels = get_test_labels(cfg)
        except Exception as e:
            print(f"  [ERROR] Could not reconstruct labels: {e}")
            continue

        n_classes   = len(cfg['class_names'])
        class_names = cfg['class_names']
        n_test      = len(true_labels)
        print(f"  Test samples: {n_test}  |  Classes: {n_classes}  |  "
              f"Cross-Agr: ERM vs {args.algo_b}")

        # Collect probs per algo
        algo_probs   = {}   # {algo: {hpseed: mean_probs}}
        algo_all     = {}   # {algo: [all test probs flat]}
        algo_val     = {}   # {algo: (vp_list, va_list)} pooled across seeds

        for algo in ['ERM', args.algo_b]:
            if algo not in available_algos:
                continue
            algo_probs[algo] = collect_mean_probs(cfg, records, algo)
            algo_all[algo]   = collect_test_probs_list(cfg, records, algo)

            # Pool val probs across all seeds for ATC threshold
            all_vp, all_va = [], []
            for hpseed in algo_probs[algo]:
                vp, va = collect_val_probs_and_accs(cfg, records, algo, hpseed)
                all_vp.extend(vp); all_va.extend(va)
            algo_val[algo] = (all_vp, all_va)

        if 'ERM' not in algo_probs:
            print("  [skip] ERM results not found")
            continue

        # --- Per-class metrics ---
        erm_test_list = algo_all['ERM']
        erm_val_vp, erm_val_va = algo_val['ERM']

        pc_acc = per_class_acc(erm_test_list, true_labels, n_classes)
        pc_atc = (per_class_atc(erm_test_list, erm_val_vp, erm_val_va,
                                true_labels, n_classes)
                  if erm_val_vp else np.full(n_classes, np.nan))
        pc_agr = per_class_agreement(
            list(algo_probs['ERM'].values()), true_labels, n_classes)

        pc_cross = np.full(n_classes, np.nan)
        if args.algo_b in algo_probs:
            pc_cross = per_class_cross_agreement(
                algo_probs['ERM'], algo_probs[args.algo_b],
                true_labels, n_classes)

        # --- Print table ---
        col_w = max(len(c) for c in class_names) + 1
        hdr = (f"  {'Class':<{col_w}} | {'N':>5} | {'Acc':>6} | "
               f"{'ATC':>6} | {'Agr(ERM)':>8} | {'X-Agr(ERM-'+args.algo_b+')':>14}")
        print(hdr)
        print('  ' + '─' * (len(hdr) - 2))

        order = np.argsort(pc_acc)   # sort hardest first
        for c in order:
            n_c = int((true_labels == c).sum())
            def _f(v): return f'{v*100:5.1f}%' if not np.isnan(v) else '   —  '
            print(f"  {class_names[c]:<{col_w}} | {n_c:>5} | "
                  f"{_f(pc_acc[c]):>6} | {_f(pc_atc[c]):>6} | "
                  f"{_f(pc_agr[c]):>8} | {_f(pc_cross[c]):>14}")

        print()
        print(f"  Overall Acc(ERM mean): "
              f"{np.nanmean(pc_acc)*100:.1f}%   "
              f"hardest class: {class_names[order[0]]} ({pc_acc[order[0]]*100:.1f}%)")


if __name__ == '__main__':
    main()
