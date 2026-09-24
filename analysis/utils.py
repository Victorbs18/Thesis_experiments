# analysis/utils.py
"""
Shared utilities for the analysis/ scripts.

Consolidates functions that were independently copy-pasted (with minor
naming drift) across cross_algorithm_agreement.py, odp_bench_comparison.py,
accuracy_on_the_line.py, agreement_on_the_line.py, per_class_eval.py,
shift_detection.py, pooled_acl.py and irm_env_variance_check.py.

Only functions that were byte-identical (or provably equivalent) across
their original copies were merged here. A few near-duplicates were
deliberately left in their original files because they encode different
statistical choices under similar-looking code (see NOTE comments in the
scripts that still have local copies) — merging those would silently
change computed results.
"""

import os
import numpy as np
from scipy.special import ndtri as probit
from scipy.stats import pearsonr, linregress

# ---------------------------------------------------------------------------
# Dataset registry (single source of truth — was duplicated, and drifting,
# between odp_bench_comparison.py and per_class_eval.py)
# ---------------------------------------------------------------------------

ALGOS     = ['ERM', 'IRM', 'VREx', 'GroupDRO', 'CORAL', 'DANN']
N_HPARAMS = 20
N_TRIALS  = 3

CONFIGS = [
    {
        'name':         'ColoredMNIST (env2)',
        'records_path': 'results/coloredmnist/test_env2/cnn/random/records.json',
        'preds_dir':    'results/coloredmnist/test_env2/cnn/random/models',
        'test_env_idx': 2,
        'n_envs':       3,
        'dtype':        'coloredmnist',
        'data_dir':     './data',
        'class_names':  ['y=0 (label<5)', 'y=1 (label>=5)'],
    },
    {
        'name':         'ColoredMNIST per-trial (env2)',
        'records_path': 'results/coloredmnist/test_env2/cnn/random_pertrial/records.json',
        'preds_dir':    'results/coloredmnist/test_env2/cnn/random_pertrial/models',
        'test_env_idx': 2,
        'n_envs':       3,
        'dtype':        'coloredmnist',
        'data_dir':     './data',
        'class_names':  ['y=0 (label<5)', 'y=1 (label>=5)'],
        'split_mode':   'per_trial',
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
        'name':         'PACS CLIP (env1) broken',
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
    {
        'name':         'ACSIncome (env9 MI)',
        'records_path': 'results/acsincome/test_env9/cnn/random/records.json',
        'preds_dir':    'results/acsincome/test_env9/cnn/random/models',
        'test_env_idx': 9,
        'n_envs':       10,
        'dtype':        'acsincome',
        'data_dir':     None,
        'class_names':  ['income<50k', 'income>=50k'],
    },
]

# ---------------------------------------------------------------------------
# Ground-truth label loading (labels only — no images)
# ---------------------------------------------------------------------------

def get_test_labels(cfg):
    """
    Reconstruct ground-truth labels for the test env's 20% holdout split,
    exactly matching the split used during training (seed=0 throughout).

    Returns None for split_mode='per_trial' configs: the holdout set (and
    therefore the true labels) differs per trial there, so a single labels
    array can't be matched against pooled/per-trial predictions the way
    this function assumes. Per-class analysis (which needs this) is a
    known gap for per-trial configs — see the module docstring in
    odp_bench_comparison.py.
    """
    if cfg.get('split_mode') == 'per_trial':
        return None

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

    elif dtype == 'acsincome':
        return None

    raise ValueError(f"Unknown dtype: {dtype}")

# ---------------------------------------------------------------------------
# I/O helpers — loading saved predictions / probabilities
# ---------------------------------------------------------------------------

def load_probs(preds_dir, algo, hpseed, trial, env_idx):
    fname = f"{algo}_hpseed{hpseed}_trial{trial}_env{env_idx}_probs.npy"
    path  = os.path.join(preds_dir, fname)
    return np.load(path).astype(np.float32) if os.path.exists(path) else None


def load_preds(preds_dir, algo, hpseed, trial, env_idx):
    fname = f"{algo}_hpseed{hpseed}_trial{trial}_env{env_idx}_preds.npy"
    path  = os.path.join(preds_dir, fname)
    return np.load(path) if os.path.exists(path) else None


# alias — cross_algorithm_agreement.py originally called this load_predictions
load_predictions = load_preds


def get_all_trials(preds_dir, algorithm, hparams_seed, n_trials,
                    env_idx, loader_fn):
    """Load loader_fn(preds_dir, algorithm, hparams_seed, trial, env_idx) for
    every trial, skipping missing files. loader_fn is load_probs or load_preds."""
    results = []
    for trial in range(n_trials):
        r = loader_fn(preds_dir, algorithm, hparams_seed, trial, env_idx)
        if r is not None:
            results.append(r)
    return results


def get_records(records, algo, hpseed, trial=None):
    """Raw records.json entries matching (algo, hpseed[, trial])."""
    out = [r for r in records
           if r['algorithm'] == algo and r['args']['hparams_seed'] == hpseed]
    if trial is not None:
        out = [r for r in out if r['args']['trial_seed'] == trial]
    return out


def get_record_info(records, algo, seed, test_env_idx, n_envs=None):
    """
    Aggregate one (algo, seed)'s records.json entries into oracle accuracy,
    hyperparameters, and (only if n_envs is given) training-env accuracy
    mean/std across trials.
    """
    matching = [r for r in records
                if r['algorithm'] == algo
                and r['args']['hparams_seed'] == seed]
    if not matching:
        return None

    key  = f'env{test_env_idx}_out_acc'
    accs = [r[key] for r in matching if key in r]
    hp   = matching[0]['hparams']

    info = {
        'ood_acc':        float(np.mean(accs)) if accs else None,
        'ood_acc_std':    float(np.std(accs))  if accs else None,
        'lr':             hp.get('lr', 0),
        'lambda':         hp.get('irm_lambda', hp.get('mmd_gamma', None)),
        'anneal':         hp.get('irm_penalty_anneal_iters', None),
        'bs':             hp.get('batch_size', None),
        'train_accs':     None,
        'train_acc_mean': None,
        'train_acc_std':  None,
    }

    if n_envs is not None:
        train_accs = []
        for env_idx in range(n_envs):
            if env_idx == test_env_idx:
                continue
            tkey = f'env{env_idx}_out_acc'
            vals = [r[tkey] for r in matching if tkey in r]
            if vals:
                train_accs.append(float(np.mean(vals)))
        info['train_accs']     = train_accs
        info['train_acc_mean'] = float(np.mean(train_accs)) if train_accs else None
        info['train_acc_std']  = float(np.std(train_accs)) if len(train_accs) > 1 else None

    return info

# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def compute_entropy(probs):
    """Mean Shannon entropy over N samples."""
    return float(-np.sum(probs * np.log(probs + 1e-8), axis=1).mean())

# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def compute_agreement(preds_i, preds_j):
    """Fraction of examples where two models predict the same class."""
    return float(np.mean(preds_i == preds_j))


def agreement_scores(probs_dict):
    """
    {seed: mean pairwise argmax agreement with every other seed in the dict}.
    """
    seeds = list(probs_dict.keys())
    preds = {s: probs_dict[s].argmax(axis=1) for s in seeds}
    scores = {}
    for s in seeds:
        agrs = [(preds[s] == preds[t]).mean() for t in seeds if t != s]
        scores[s] = float(np.mean(agrs)) if agrs else 0.0
    return scores


def get_id_agr(preds_dir, algo_a, algo_b, seed_a, seed_b,
               n_trials, test_env_idx, n_envs, return_all=False):
    """
    Mean prediction agreement between (algo_a, seed_a) and (algo_b, seed_b)
    on training-env out-splits, matched by trial index (trial_i == trial_j).
    """
    per_trial_agrs = []
    for trial in range(n_trials):
        env_agrs = []
        for env_idx in range(n_envs):
            if env_idx == test_env_idx:
                continue
            pa = load_preds(preds_dir, algo_a, seed_a, trial, env_idx)
            pb = load_preds(preds_dir, algo_b, seed_b, trial, env_idx)
            if pa is not None and pb is not None:
                env_agrs.append(compute_agreement(pa, pb))
        if env_agrs:
            per_trial_agrs.append(float(np.mean(env_agrs)))
    if not per_trial_agrs:
        return None
    return per_trial_agrs if return_all else float(np.mean(per_trial_agrs))


def get_ood_agr(preds_dir, algo_a, algo_b, seed_a, seed_b,
                n_trials, test_env_idx, return_all=False):
    """
    Mean prediction agreement between (algo_a, seed_a) and (algo_b, seed_b)
    on the test env, matched by trial index (trial_i == trial_j).
    """
    per_trial_agrs = []
    for trial in range(n_trials):
        pa = load_preds(preds_dir, algo_a, seed_a, trial, test_env_idx)
        pb = load_preds(preds_dir, algo_b, seed_b, trial, test_env_idx)
        if pa is not None and pb is not None:
            per_trial_agrs.append(compute_agreement(pa, pb))
    if not per_trial_agrs:
        return None
    return per_trial_agrs if return_all else float(np.mean(per_trial_agrs))

# ---------------------------------------------------------------------------
# Entropy-based seed filtering
# ---------------------------------------------------------------------------

def get_valid_seeds(preds_dir, algorithm, n_hparams, n_trials, test_env_idx,
                     max_entropy, entropy_threshold=0.9):
    """
    Seeds whose mean test-env entropy, relative to max_entropy = log(n_classes),
    stays below entropy_threshold (excludes near-uniform/degenerate seeds).
    """
    valid, excluded = [], []
    for seed in range(n_hparams):
        probs = get_all_trials(preds_dir, algorithm, seed, n_trials,
                                test_env_idx, load_probs)
        if not probs:
            continue
        rel_h = float(np.mean([compute_entropy(p) for p in probs])) / max_entropy
        if rel_h < entropy_threshold:
            valid.append(seed)
        else:
            excluded.append((seed, rel_h))
    return valid, excluded

# ---------------------------------------------------------------------------
# Accuracy-on-the-line helpers
# ---------------------------------------------------------------------------

def compute_id_acc(record, test_env_idx, n_envs):
    """ID accuracy = mean out_acc across training environments."""
    train_accs = [
        record[f'env{i}_out_acc']
        for i in range(n_envs)
        if i != test_env_idx
    ]
    return np.mean(train_accs)


def compute_ood_acc(record, test_env_idx):
    """OOD accuracy = out_acc on the test environment."""
    return record[f'env{test_env_idx}_out_acc']

# ---------------------------------------------------------------------------
# Probit-space line fit — shared "accuracy/agreement-on-the-line" fitting
# ---------------------------------------------------------------------------

def fit_line(x_vals, y_vals):
    """
    Fit a line in probit space between two [0, 1]-valued series (accuracies
    or agreement fractions) and return R/slope/intercept/etc.
    """
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    eps = 1e-6
    x_probit = probit(np.clip(x_vals, eps, 1 - eps))
    y_probit = probit(np.clip(y_vals, eps, 1 - eps))
    R, p_value = pearsonr(x_probit, y_probit)
    reg = linregress(x_probit, y_probit)
    return {
        'R':          float(R),
        'slope':      float(reg.slope),
        'intercept':  float(reg.intercept),
        'p_value':    float(p_value),
        'std_error':  float(reg.stderr),
        'id_agrs':    x_vals.tolist(),
        'ood_agrs':   y_vals.tolist(),
        'id_probit':  x_probit.tolist(),
        'ood_probit': y_probit.tolist(),
        'ood_median': float(np.median(y_vals)),
        'ood_mean':   float(np.mean(y_vals)),
        'n_pairs':    len(x_vals),
    }
