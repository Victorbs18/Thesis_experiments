# src/datasets.py
"""
Dataset loading for all domain generalization experiments.

Each dataset returns a list of (in_env, out_env) tuples:
    envs_splits[i] = (in_env, out_env)
    in_env:  80% train portion: used for training
    out_env: 20% val portion: used for selection

For tensor datasets (ColoredMNIST):
    env = {'images': Tensor(N, C, H, W), 'labels': Tensor(N,)}

For image datasets (PACS):
    env = torch.utils.data.Subset of an ImageFolder dataset

Usage:
    from src.datasets import get_dataset
    envs_splits = get_dataset('ColoredMNIST', data_dir='./data')
    envs_splits = get_dataset('PACS', data_dir='./data', test_env_idx=0)
    envs_splits = get_dataset('PACS', data_dir='./data', test_env_idx=0, backbone='clip')
"""

import os
import sys
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import Subset
from domainbed.datasets import (
    WILDSCamelyon as DB_WILDSCamelyon,
    RotatedMNIST  as DB_RotatedMNIST,
)


# Shared utilities

def is_tensor_env(env):
    """Check if env is a tensor dict (ColoredMNIST) or a Dataset (PACS)."""
    return isinstance(env, dict) and 'images' in env


def split_env(env, holdout_frac=0.2, seed=0):
    """
    Split tensor environment into train (in) and val (out) subsets.
    DomainBed protocol: 20% holdout, seed=0.
    Returns (in_env, out_env): same dict structure as input.
    """
    n     = len(env['images'])
    rng   = np.random.RandomState(seed)
    perm  = rng.permutation(n)
    n_val = int(n * holdout_frac)

    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]

    in_env = {
        'images': env['images'][train_idx],
        'labels': env['labels'][train_idx],
    }
    out_env = {
        'images': env['images'][val_idx],
        'labels': env['labels'][val_idx],
    }
    return in_env, out_env


def split_env_subset(dataset, holdout_frac=0.2, seed=0):
    """
    Split an ImageFolder-style dataset into train/val subsets.
    Returns (in_subset, out_subset) — both are torch Subset objects.
    """
    n     = len(dataset)
    rng   = np.random.RandomState(seed)
    perm  = rng.permutation(n)
    n_val = int(n * holdout_frac)

    val_idx   = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# Transforms

def get_image_transforms(backbone='resnet50'):
    """
    Return (aug_transform, eval_transform) for PACS.
    backbone='resnet50' → ImageNet normalization
    backbone='clip'     → CLIP normalization + BICUBIC interpolation
    """
    if backbone == 'clip':
        normalize     = transforms.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),
            std= (0.26862954, 0.26130258, 0.27577711),
        )
        interpolation = transforms.InterpolationMode.BICUBIC
        eval_resize   = transforms.Compose([
            transforms.Resize(224, interpolation=interpolation),
            transforms.CenterCrop(224),
        ])
    else:
        normalize     = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225],
        )
        interpolation = transforms.InterpolationMode.BILINEAR
        eval_resize   = transforms.Resize((224, 224))

    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0),
                                     interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        eval_resize,
        transforms.ToTensor(),
        normalize,
    ])

    return aug_transform, eval_transform


# RotatedMNIST

def _build_rotated_mnist_envs(data_dir, test_env_idx):
    """
    Generate RotatedMNIST's 6 per-rotation environments ONCE. Call this once
    and reuse the result across multiple splits — DomainBed's own
    MultipleEnvironmentMNIST base class shuffles the source images with a
    completely unseeded torch.randperm(...) (worse than ColoredMNIST's own
    RNG issue, which at least seeds its permutation), so a fresh
    DB_RotatedMNIST(...) call would reassign which images go to which of the
    6 rotation environments every time — not just produce a different
    train/val partition of the same data.
    """
    db_dataset = DB_RotatedMNIST(data_dir, test_envs=[test_env_idx], hparams={})
    env_names  = DB_RotatedMNIST.ENVIRONMENTS  # ['0', '15', '30', '45', '60', '75']

    print(f"RotatedMNIST loaded (test env: {env_names[test_env_idx]}°):")
    for i, env in enumerate(db_dataset.datasets):
        marker = ' : test' if i == test_env_idx else ''
        print(f"  env{i} ({env_names[i]}°): {len(env)} samples{marker}")

    return db_dataset.datasets


def get_rotated_mnist(data_dir='./data', test_env_idx=5,
                      holdout_frac=0.2, seed=0, backbone='cnn'):
    """
    Load RotatedMNIST via DomainBed's implementation.
    6 environments: rotations 0°, 15°, 30°, 45°, 60°, 75°.
    backbone argument accepted but ignored (always uses CNN).
    Returns list of (in_env, out_env) tuples.
    """
    envs = _build_rotated_mnist_envs(data_dir, test_env_idx)
    return [split_env_subset(env, holdout_frac, seed) for env in envs]


def get_rotated_mnist_per_trial(data_dir='./data', n_trials=3, test_env_idx=5,
                                holdout_frac=0.2, backbone='cnn'):
    """
    Same RotatedMNIST base data as get_rotated_mnist, generated ONCE, split
    n_trials different ways (seed=0..n_trials-1) — one train/val partition
    per trial, shared across every algorithm/hparams_seed that uses that
    trial index (required for Cross-R/CrA's trial-matched agreement).
    Returns {trial_seed: envs_splits}.
    """
    envs = _build_rotated_mnist_envs(data_dir, test_env_idx)
    return {
        t: [split_env_subset(env, holdout_frac, seed=t) for env in envs]
        for t in range(n_trials)
    }


# ColoredMNIST

def _bernoulli(p, size):
    return (torch.rand(size) < p).float()


def _xor(a, b):
    return (a - b).abs()


def _color_dataset(images, labels, environment):
    """Exact DomainBed color_dataset function."""
    labels = (labels < 5).float()
    labels = _xor(labels, _bernoulli(0.25, len(labels)))
    colors = _xor(labels, _bernoulli(environment, len(labels)))
    images = torch.stack([images, images], dim=1)
    images[torch.arange(len(images)), (1 - colors).long(), :, :] *= 0
    x = images.float().div_(255.0)
    y = labels.view(-1).long()
    return {'images': x, 'labels': y}


def _build_colored_mnist_envs(data_dir):
    """
    Generate the 3 ColoredMNIST environments (shared base data, before
    train/val splitting). Call this ONCE and reuse the result across
    multiple splits — _color_dataset's label-noise/color assignment uses
    PyTorch's global RNG (a known, pre-existing non-determinism), so calling
    this repeatedly would produce a different synthetic realization each
    time, not just a different train/val partition of the same data.
    """
    mnist_train = MNIST(data_dir, train=True,  download=True)
    mnist_test  = MNIST(data_dir, train=False, download=True)

    images = torch.cat([mnist_train.data, mnist_test.data]).float()
    labels = torch.cat([mnist_train.targets, mnist_test.targets])

    rng = torch.Generator()
    rng.manual_seed(0)
    perm   = torch.randperm(len(images), generator=rng)
    images = images[perm]
    labels = labels[perm]

    environments = [0.1, 0.2, 0.9]
    envs = [
        _color_dataset(images[i::len(environments)],
                       labels[i::len(environments)], e)
        for i, e in enumerate(environments)
    ]
    return envs, environments


def get_colored_mnist(data_dir='./data', holdout_frac=0.2, seed=0,
                      backbone='cnn'):
    """
    Build ColoredMNIST exactly as DomainBed does.
    3 environments: e=0.1 (+90%), e=0.2 (+80%), e=0.9 (-90%)
    Returns list of (in_env, out_env) tuples.
    backbone argument accepted but ignored (always uses CNN).
    """
    envs, environments = _build_colored_mnist_envs(data_dir)

    print(f"ColoredMNIST loaded:")
    for i, (e, env) in enumerate(zip(environments, envs)):
        marker = ': test' if i == 2 else ''
        print(f"  env{i} (e={e}): {len(env['images'])} samples{marker}")

    envs_splits = [split_env(env, holdout_frac, seed) for env in envs]
    return envs_splits


def get_colored_mnist_per_trial(data_dir='./data', n_trials=3,
                                holdout_frac=0.2, backbone='cnn'):
    """
    Same ColoredMNIST base data as get_colored_mnist, generated ONCE, split
    n_trials different ways (seed=0..n_trials-1) — one train/val partition
    per trial, shared across every algorithm/hparams_seed that uses that
    trial index (so predictions stay comparable trial-to-trial across
    algorithms — required for Cross-R/CrA's trial-matched agreement).
    Returns {trial_seed: envs_splits}.
    """
    envs, environments = _build_colored_mnist_envs(data_dir)

    print(f"ColoredMNIST loaded (per-trial split, {n_trials} trials):")
    for i, (e, env) in enumerate(zip(environments, envs)):
        marker = ': test' if i == 2 else ''
        print(f"  env{i} (e={e}): {len(env['images'])} samples{marker}")

    return {
        t: [split_env(env, holdout_frac, seed=t) for env in envs]
        for t in range(n_trials)
    }


# PACS

def get_pacs(data_dir='./data', test_env_idx=0,
             holdout_frac=0.2, seed=0, backbone='resnet50'):
    """
    Load PACS dataset.
    4 environments: art_painting, cartoon, photo, sketch
    backbone controls preprocessing transforms:
        'resnet50' → ImageNet normalization (DomainBed standard)
        'clip'     → CLIP normalization + BICUBIC interpolation
    Returns list of (in_env, out_env) tuples.
    """
    aug_transform, eval_transform = get_image_transforms(backbone)

    env_names = sorted([f.name for f in os.scandir(data_dir) if f.is_dir()])
    print(f"PACS loaded (backbone={backbone}, "
          f"test env: {env_names[test_env_idx]}):")

    envs_splits = []
    for i, name in enumerate(env_names):
        marker    = ' : test' if i == test_env_idx else ''
        env_path  = os.path.join(data_dir, name)
        # test env uses eval transform, training envs use augmentation
        transform = eval_transform if i == test_env_idx else aug_transform
        dataset   = ImageFolder(env_path, transform=transform)
        print(f"  env{i} ({name}): {len(dataset)} images{marker}")
        in_env, out_env = split_env_subset(dataset, holdout_frac, seed)
        envs_splits.append((in_env, out_env))

    return envs_splits


# WILDSCamelyon

def get_wildscamelyon(data_dir='./data', test_env_idx=2,
                       holdout_frac=0.2, seed=0, backbone='resnet50'):
    """
    Load WILDSCamelyon via DomainBed's wrapper (requires `wilds` package and
    Camelyon17 data downloaded under data_dir).
    5 environments: hospital_0..hospital_4 (test_env_idx=2 by default).
    backbone controls transforms via get_image_transforms (same as PACS).
    Returns list of (in_env, out_env) tuples.
    """
    hparams = {'data_augmentation': True}
    db_dataset = DB_WILDSCamelyon(data_dir, test_envs=[test_env_idx], hparams=hparams)

    aug_transform, eval_transform = get_image_transforms(backbone)

    print(f"WILDSCamelyon loaded (backbone={backbone}, "
          f"test env: {db_dataset.ENVIRONMENTS[test_env_idx]}):")
    envs_splits = []
    for i, env in enumerate(db_dataset.datasets):
        # Override DomainBed's fixed ImageNet transforms with backbone-aware ones
        env.transform = eval_transform if i == test_env_idx else aug_transform
        marker = ' : test' if i == test_env_idx else ''
        print(f"  env{i} ({db_dataset.ENVIRONMENTS[i]}): {len(env)} samples{marker}")
        in_env, out_env = split_env_subset(env, holdout_frac, seed)
        envs_splits.append((in_env, out_env))

    return envs_splits


# ACSIncome

ACS_STATES = ['CA', 'TX', 'NY', 'FL', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']


def get_acs_income(data_dir='./data', test_env_idx=9,
                    holdout_frac=0.2, seed=0, backbone=None):
    """
    Load ACS Income via folktables. Each US state is one environment;
    task is binary classification of income >= 50k. Geographic covariate
    shift across states.

    backbone argument accepted but ignored — tabular data gets DomainBed's
    built-in MLP featurizer, auto-selected for 1D input_shape.

    Reuses the tensor-env pipeline (same as ColoredMNIST): the 'images' key
    just holds plain feature vectors here, not images.
    Returns list of (in_env, out_env) tuples.
    """
    from folktables import ACSDataSource, ACSIncome
    from sklearn.preprocessing import StandardScaler

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year',
                                survey='person', root_dir=data_dir)

    print(f"ACSIncome loaded (test env: {ACS_STATES[test_env_idx]}):")
    X_raw, y_raw = [], []
    for state in ACS_STATES:
        data    = data_source.get_data(states=[state], download=True)
        X, y, _ = ACSIncome.df_to_numpy(data)
        X_raw.append(X.astype(np.float32))
        y_raw.append(y.astype(np.int64))

    # fit scaler on training environments only — avoids test-env leakage
    train_idx = [i for i in range(len(ACS_STATES)) if i != test_env_idx]
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_raw[i] for i in train_idx]))

    envs_splits = []
    for i, state in enumerate(ACS_STATES):
        marker   = ' : test' if i == test_env_idx else ''
        X_scaled = scaler.transform(X_raw[i])
        print(f"  env{i} ({state}): {len(X_scaled)} samples{marker}")
        env = {
            'images': torch.tensor(X_scaled, dtype=torch.float32),
            'labels': torch.tensor(y_raw[i], dtype=torch.long),
        }
        in_env, out_env = split_env(env, holdout_frac, seed)
        envs_splits.append((in_env, out_env))

    return envs_splits


# Dataset registry

DATASET_CONFIGS = {
    'RotatedMNIST': {
        'loader':            get_rotated_mnist,
        'n_envs':            6,
        'test_env_idx':      5,
        'n_steps':           5001,
        'env_names':         ['0°', '15°', '30°', '45°', '60°', '75°'],
        'input_shape':       (1, 28, 28),
        'n_classes':         10,
        'selection_methods': ['IIDAccuracySelectionMethod',
                              'LeaveOneOutSelectionMethod',
                              'OracleSelectionMethod'],
    },
    'ColoredMNIST': {
        'loader':            get_colored_mnist,
        'n_envs':            3,
        'test_env_idx':      2,
        'n_steps':           5001,
        'env_names':         ['+90%', '+80%', '-90%'],
        'input_shape':       (2, 28, 28),
        'n_classes':         2,
        'selection_methods': ['IIDAccuracySelectionMethod',
                              'OracleSelectionMethod'],
    },
    'PACS': {
        'loader':            get_pacs,
        'n_envs':            4,
        'test_env_idx':      0,
        'n_steps':           5001,
        'env_names':         ['A', 'C', 'P', 'S'],
        'input_shape':       (3, 224, 224),
        'n_classes':         7,
        'selection_methods': ['IIDAccuracySelectionMethod',
                              'LeaveOneOutSelectionMethod',
                              'OracleSelectionMethod'],
    },
    'WILDSCamelyon': {
        'loader':            get_wildscamelyon,
        'n_envs':            5,
        'test_env_idx':      2,
        'n_steps':           5001,
        'env_names':         ['H0', 'H1', 'H2', 'H3', 'H4'],
        'input_shape':       (3, 224, 224),
        'n_classes':         2,
        'selection_methods': ['IIDAccuracySelectionMethod',
                              'OracleSelectionMethod'],
    },
    'ACSIncome': {
        'loader':            get_acs_income,
        'n_envs':            10,
        'test_env_idx':      9,
        'n_steps':           5001,
        'env_names':         ACS_STATES,
        'input_shape':       (10,),
        'n_classes':         2,
        'selection_methods': ['IIDAccuracySelectionMethod',
                              'OracleSelectionMethod'],
    },
}


def get_dataset(dataset_name, data_dir, test_env_idx=None,
                holdout_frac=0.2, seed=0, backbone='resnet50',
                split_mode='single', n_trials=None):
    """
    Single entry point for all datasets.

    split_mode='single' (default): one fixed 80/20 split (seed), reused
      across all trials — identical to every prior invocation.
    split_mode='per_trial': one DIFFERENT split per trial (seed=0..n_trials-1),
      shared across every algorithm/hparams_seed for that trial. Returns
      {trial_seed: envs_splits} instead of a single envs_splits list.
      Currently implemented for ColoredMNIST and RotatedMNIST.

    Usage:
        envs_splits = get_dataset('ColoredMNIST', data_dir='./data')
        envs_splits = get_dataset('PACS', data_dir='./data', test_env_idx=0)
        envs_splits = get_dataset('PACS', data_dir='./data',
                                  test_env_idx=0, backbone='clip')
        per_trial   = get_dataset('ColoredMNIST', data_dir='./data',
                                  split_mode='per_trial', n_trials=3)
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    cfg = DATASET_CONFIGS[dataset_name]

    if test_env_idx is None:
        test_env_idx = cfg['test_env_idx']

    if split_mode == 'per_trial':
        if n_trials is None:
            raise ValueError("n_trials is required when split_mode='per_trial'")
        if dataset_name == 'ColoredMNIST':
            return get_colored_mnist_per_trial(data_dir, n_trials, holdout_frac,
                                               backbone)
        elif dataset_name == 'RotatedMNIST':
            return get_rotated_mnist_per_trial(data_dir, n_trials, test_env_idx,
                                               holdout_frac, backbone)
        else:
            raise NotImplementedError(
                f"split_mode='per_trial' is currently only implemented for "
                f"ColoredMNIST and RotatedMNIST (got '{dataset_name}')."
            )

    if dataset_name == 'ColoredMNIST':
        return cfg['loader'](data_dir, holdout_frac, seed, backbone)
    else:
        return cfg['loader'](data_dir, test_env_idx, holdout_frac, seed,
                             backbone)