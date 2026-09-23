# analysis/shift_detection.py
"""
Distribution shift detection for domain generalisation benchmarks.

Three label-free shift signals, each normalised to [0, 1]:

  Covariate shift  - P(X) changes
    Measured directly in pixel space: normalised L2 distance between
    mean pixel vectors of ID and OOD environments.
    Does NOT go through the model — avoids conflating input shift
    with model confidence (a confidently wrong model would corrupt
    any model-based covariate shift signal).

  Label shift      - P(Y) changes, P(X|Y) stable
    Jensen-Shannon divergence between predicted class distributions
    on ID and OOD, using ERM argmax predictions.
    Fully label-free on the OOD side — detects that the model
    predicts different class frequencies on OOD vs ID.

  Concept shift    - P(Y|X) changes
    Mutual information term from the predictive entropy decomposition:
      MI = H(mean softmax over seeds) - mean(H(per-seed softmax))
    Measured on OOD only, normalised by the ID baseline.
    Large MI means seeds are individually confident but disagree —
    they each learned a different P(Y|X) mapping, consistent with
    P(Y|X) being unstable across domains.

All three signals are computed without OOD labels.
Each is normalised relative to its ID baseline (computed across
training environment pairs) so values are on a comparable scale:
  0 = no shift beyond what is seen across ID environments
  1 = maximum observed shift

Usage:
  python analysis/shift_detection.py \\
      --records_path results/coloredmnist/test_env2/cnn/random/records.json \\
      --preds_dir    results/coloredmnist/test_env2/cnn/random/models \\
      --test_env_idx 2 --n_envs 3 --dtype coloredmnist \\
      --data_dir     ./data --dataset_name ColoredMNIST

  python analysis/shift_detection.py \\
      --records_path results/rotatedmnist/test_env5/cnn/random/records.json \\
      --preds_dir    results/rotatedmnist/test_env5/cnn/random/models \\
      --test_env_idx 5 --n_envs 6 --dtype rotatedmnist \\
      --data_dir     ./data --dataset_name RotatedMNIST

  python analysis/shift_detection.py \\
      --records_path results/pacs/test_env0/resnet50/random/records.json \\
      --preds_dir    results/pacs/test_env0/resnet50/random/models \\
      --test_env_idx 0 --n_envs 4 --dtype pacs \\
      --data_dir     C:/Users/Usuario/Downloads/pacs_data/pacs_data \\
      --dataset_name PACS
"""

import os
import sys
import json
import argparse
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DomainBed'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import N_HPARAMS, N_TRIALS, load_probs, load_preds

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def pool_probs(preds_dir, algo, env_idx, n_hparams=N_HPARAMS, n_trials=N_TRIALS):
    """Pool softmax probability arrays across all seeds and trials for one env."""
    parts = []
    for hpseed in range(n_hparams):
        for trial in range(n_trials):
            p = load_probs(preds_dir, algo, hpseed, trial, env_idx)
            if p is not None:
                parts.append(p)
    return np.vstack(parts) if parts else None


def pool_preds(preds_dir, algo, env_idx, n_hparams=N_HPARAMS, n_trials=N_TRIALS):
    """Pool argmax predictions across all seeds and trials for one env."""
    parts = []
    for hpseed in range(n_hparams):
        for trial in range(n_trials):
            p = load_preds(preds_dir, algo, hpseed, trial, env_idx)
            if p is not None:
                parts.append(p)
    return np.concatenate(parts) if parts else None

# ---------------------------------------------------------------------------
# Dataset image loading — pixel statistics only, no labels needed
# ---------------------------------------------------------------------------

def get_pixel_mean(dtype, env_idx, data_dir, n_samples=1000):
    """
    Compute mean pixel vector (per channel) for one environment.
    Subsamples up to n_samples images for speed.
    Returns flat array of shape (C,) — mean pixel intensity per channel.
    Does not use or load labels.
    """
    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    rng = np.random.default_rng(42)

    if dtype == 'coloredmnist':
        # ColoredMNIST is procedurally generated at runtime — pixel stats
        # are not accessible from raw MNIST because the colouring is applied
        # by the DomainBed dataloader. Return None to signal unavailability.
        return None

    elif dtype == 'rotatedmnist':
        from domainbed.datasets import RotatedMNIST as DB_RotatedMNIST
        db  = DB_RotatedMNIST(data_dir, test_envs=[env_idx], hparams={})
        env = db.datasets[env_idx]
        n   = min(len(env), n_samples)
        idx = rng.choice(len(env), n, replace=False)
        pixels = []
        for i in idx:
            img, _ = env[i]
            if not hasattr(img, 'numpy'):
                img = to_tensor(img)
            pixels.append(img.numpy().reshape(img.shape[0], -1).mean(axis=1))
        return np.array(pixels).mean(axis=0)  # shape (C,)

    elif dtype == 'pacs':
        from torchvision.datasets import ImageFolder
        env_dirs = sorted(f.name for f in os.scandir(data_dir) if f.is_dir())
        dataset  = ImageFolder(
            os.path.join(data_dir, env_dirs[env_idx]),
            transform=to_tensor)
        n   = min(len(dataset), n_samples)
        idx = rng.choice(len(dataset), n, replace=False)
        pixels = []
        for i in idx:
            img, _ = dataset[i]
            pixels.append(img.numpy().reshape(img.shape[0], -1).mean(axis=1))
        return np.array(pixels).mean(axis=0)  # shape (C,)

    raise ValueError(f"Unknown dtype: {dtype}")

# ---------------------------------------------------------------------------
# Shift signal functions — raw (unnormalised)
# ---------------------------------------------------------------------------

def covariate_shift_raw(dtype, env_a, env_b, data_dir):
    """
    Normalised L2 distance between mean pixel vectors of two environments.

    Measures P(X) shift directly in input space without going through
    the model. A confidently wrong model would corrupt any model-based
    covariate shift signal, so we use raw pixels instead.

    Returns scalar >= 0, or None if pixel stats unavailable (ColoredMNIST).
    """
    mu_a = get_pixel_mean(dtype, env_a, data_dir)
    mu_b = get_pixel_mean(dtype, env_b, data_dir)
    if mu_a is None or mu_b is None:
        return None
    norm_a = np.linalg.norm(mu_a)
    denom  = max(norm_a, 1e-6)
    return float(np.linalg.norm(mu_a - mu_b) / denom)


def label_shift_raw(preds_dir, env_a, env_b, n_classes,
                    algo='ERM'):
    """
    Jensen-Shannon divergence between predicted class distributions of
    two environments, using ERM argmax predictions.

    Detects P(Y) shift — if the model predicts very different class
    frequencies on OOD vs ID, the class prior has shifted.

    Fully label-free on both sides: uses only argmax predictions,
    not ground-truth labels.

    JS divergence is symmetric and in [0, log(2)].
    """
    preds_a = pool_preds(preds_dir, algo, env_a)
    preds_b = pool_preds(preds_dir, algo, env_b)
    if preds_a is None or preds_b is None:
        return None

    eps   = 1e-10
    p_a   = np.array([(preds_a == c).mean() for c in range(n_classes)])
    p_b   = np.array([(preds_b == c).mean() for c in range(n_classes)])
    m     = 0.5 * (p_a + p_b)
    js    = 0.5 * np.sum(p_a * np.log((p_a + eps) / (m + eps)))
    js   += 0.5 * np.sum(p_b * np.log((p_b + eps) / (m + eps)))
    return float(max(js, 0.0))


def concept_shift_raw(preds_dir, env_idx, n_classes,
                      n_hparams=N_HPARAMS, n_trials=N_TRIALS):
    """
    Mutual information term from the predictive entropy decomposition,
    measured on one environment using ERM seeds.

    For each test sample x, pool softmax outputs across all seeds:
      p_mean(x) = (1/S) * sum_s p^(s)(x)

    Then:
      H_total(x) = H(p_mean(x))          — total uncertainty
      H_data(x)  = (1/S) * sum_s H(p^(s)(x)) — average aleatoric uncertainty
      MI(x)      = H_total(x) - H_data(x)    — epistemic / disagreement

    MI > 0 means seeds are individually confident but disagree about
    which class is correct — consistent with P(Y|X) being unstable,
    i.e. concept shift.

    Returns mean MI across all test samples, or None if insufficient data.
    """
    # collect softmax arrays per seed (mean over trials)
    seed_probs = []
    for hpseed in range(n_hparams):
        trials = []
        for trial in range(n_trials):
            p = load_probs(preds_dir, 'ERM', hpseed, trial, env_idx)
            if p is not None:
                trials.append(p)
        if trials:
            seed_probs.append(np.mean(trials, axis=0))  # (N, C)

    if len(seed_probs) < 2:
        return None

    # stack: shape (S, N, C)
    stack  = np.stack(seed_probs, axis=0)
    p_mean = stack.mean(axis=0)                # (N, C)

    def entropy(p):
        return float(-np.sum(p * np.log(p + 1e-8), axis=-1).mean())

    h_total = entropy(p_mean)
    h_data  = float(np.mean(
        [-np.sum(p * np.log(p + 1e-8), axis=-1).mean()
         for p in seed_probs]))

    mi = h_total - h_data
    return float(max(mi, 0.0))

# ---------------------------------------------------------------------------
# Normalised shift signals
# ---------------------------------------------------------------------------

def compute_id_baselines(dtype, preds_dir, val_envs, n_classes, data_dir):
    """
    Compute each shift signal across all pairs of ID environments.
    The resulting distributions give the baseline — what each signal
    looks like when there is no OOD shift (just natural variation
    across training environments).

    Returns dict with keys 'covariate', 'label', 'concept', each
    containing a list of raw values across ID environment pairs.
    """
    baselines = {'covariate': [], 'label': [], 'concept': []}

    for env_a, env_b in combinations(val_envs, 2):
        # covariate
        v = covariate_shift_raw(dtype, env_a, env_b, data_dir)
        if v is not None:
            baselines['covariate'].append(v)

        # label
        v = label_shift_raw(preds_dir, env_a, env_b, n_classes)
        if v is not None:
            baselines['label'].append(v)

    # concept: per-environment MI, baseline is mean across ID envs
    for env_idx in val_envs:
        v = concept_shift_raw(preds_dir, env_idx, n_classes)
        if v is not None:
            baselines['concept'].append(v)

    return baselines


def normalise(value, baseline_vals):
    """
    Normalise a shift value relative to its ID baseline distribution.

    normalised = (value - baseline_mean) / (baseline_std + epsilon)

    Positive values mean the shift exceeds typical ID variation.
    Clipped to [0, inf) — negative values mean less shift than ID baseline.

    If no baseline is available, return the raw value.
    """
    if not baseline_vals or value is None:
        return value
    mu  = float(np.mean(baseline_vals))
    std = float(np.std(baseline_vals))
    if std < 1e-8:
        # all baseline values are identical — use simple ratio
        return float(value / max(mu, 1e-8))
    return float(max((value - mu) / std, 0.0))


def compute_all_shifts(dtype, preds_dir, test_env_idx, val_envs,
                       n_classes, data_dir):
    """
    Compute all three normalised shift signals for the test environment.

    Steps:
      1. Compute ID baselines across all pairs of training environments
      2. Compute each raw signal for (ID pool, OOD test env)
      3. Normalise by the ID baseline

    Returns dict:
      raw      : {'covariate', 'label', 'concept'} — unnormalised values
      norm     : {'covariate', 'label', 'concept'} — normalised values
      baseline : {'covariate', 'label', 'concept'} — ID baseline lists
      dominant : which shift type has the largest normalised value
    """
    print("  Computing ID baselines...")
    baselines = compute_id_baselines(
        dtype, preds_dir, val_envs, n_classes, data_dir)

    print("  Computing OOD shift signals...")

    # covariate: average over all ID env vs OOD pairs
    cov_vals = []
    for env_a in val_envs:
        v = covariate_shift_raw(dtype, env_a, test_env_idx, data_dir)
        if v is not None:
            cov_vals.append(v)
    raw_cov = float(np.mean(cov_vals)) if cov_vals else None

    # label: average over all ID env vs OOD pairs
    lab_vals = []
    for env_a in val_envs:
        v = label_shift_raw(preds_dir, env_a, test_env_idx, n_classes)
        if v is not None:
            lab_vals.append(v)
    raw_lab = float(np.mean(lab_vals)) if lab_vals else None

    # concept: MI on OOD test environment
    raw_con = concept_shift_raw(preds_dir, test_env_idx, n_classes)

    raw  = {'covariate': raw_cov, 'label': raw_lab, 'concept': raw_con}
    norm = {
        'covariate': normalise(raw_cov, baselines['covariate']),
        'label':     normalise(raw_lab, baselines['label']),
        'concept':   normalise(raw_con, baselines['concept']),
    }

    # dominant shift type: largest normalised value
    valid_norm = {k: v for k, v in norm.items() if v is not None}
    dominant   = max(valid_norm, key=valid_norm.get) if valid_norm else None

    return {'raw': raw, 'norm': norm, 'baseline': baselines,
            'dominant': dominant}

# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_shift_report(result, dataset_name, test_env_idx):
    raw  = result['raw']
    norm = result['norm']
    base = result['baseline']
    dom  = result['dominant']

    def _f(v):  return f'{v:.4f}' if v is not None else '  N/A '
    def _b(lst): return (f'mean={np.mean(lst):.4f} std={np.std(lst):.4f}'
                         if lst else 'N/A')

    print(f"\n{'='*72}")
    print(f"  Shift detection — {dataset_name}  (test env: {test_env_idx})")
    print(f"{'='*72}")
    print(f"  {'Signal':<16} {'raw':>8} {'normalised':>12}  baseline")
    print(f"  {'-'*68}")
    print(f"  {'Covariate':<16} {_f(raw['covariate']):>8} "
          f"{_f(norm['covariate']):>12}  {_b(base['covariate'])}")
    print(f"  {'Label':<16} {_f(raw['label']):>8} "
          f"{_f(norm['label']):>12}  {_b(base['label'])}")
    print(f"  {'Concept':<16} {_f(raw['concept']):>8} "
          f"{_f(norm['concept']):>12}  {_b(base['concept'])}")
    print(f"  {'-'*68}")
    print(f"  Dominant shift type: {dom if dom else 'undetermined'}")
    print()
    print(f"  Interpretation:")
    print(f"    Covariate : normalised L2 distance between mean pixel vectors")
    print(f"                ID vs OOD (direct input space, model-free)")
    print(f"    Label     : Jensen-Shannon divergence of predicted class")
    print(f"                frequencies ID vs OOD (ERM argmax predictions)")
    print(f"    Concept   : predictive MI = H(mean softmax) - mean H(softmax)")
    print(f"                on OOD test env — seeds individually confident")
    print(f"                but disagreeing signals P(Y|X) instability")
    print(f"    Normalised: (raw - ID_mean) / ID_std, clipped to [0, inf)")
    print(f"                0 = no shift beyond normal ID variation")
    print(f"                >1 = shift exceeds one ID std deviation")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records_path',  type=str, required=True)
    parser.add_argument('--preds_dir',     type=str, required=True)
    parser.add_argument('--test_env_idx',  type=int, required=True)
    parser.add_argument('--n_envs',        type=int, required=True)
    parser.add_argument('--dtype',         type=str, required=True,
                        choices=['coloredmnist', 'rotatedmnist', 'pacs'])
    parser.add_argument('--data_dir',      type=str, required=True)
    parser.add_argument('--dataset_name',  type=str, default='Dataset')
    parser.add_argument('--n_samples_pixel', type=int, default=1000,
                        help='Images to subsample for pixel statistics')
    args = parser.parse_args()

    val_envs = [e for e in range(args.n_envs) if e != args.test_env_idx]

    # infer n_classes from saved softmax files
    n_classes = 2
    for hpseed in range(N_HPARAMS):
        for trial in range(N_TRIALS):
            p = load_probs(args.preds_dir, 'ERM', hpseed, trial,
                           args.test_env_idx)
            if p is not None:
                n_classes = p.shape[1]
                break
        else:
            continue
        break
    print(f"\n  n_classes = {n_classes}")
    print(f"  val envs  = {val_envs}")
    print(f"  test env  = {args.test_env_idx}")

    result = compute_all_shifts(
        dtype        = args.dtype,
        preds_dir    = args.preds_dir,
        test_env_idx = args.test_env_idx,
        val_envs     = val_envs,
        n_classes    = n_classes,
        data_dir     = args.data_dir,
    )

    print_shift_report(result, args.dataset_name, args.test_env_idx)


if __name__ == '__main__':
    main()