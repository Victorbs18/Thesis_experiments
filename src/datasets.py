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
"""

import os
import sys
import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Subset


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


def get_colored_mnist(data_dir='./data', holdout_frac=0.2, seed=0):
    """
    Build ColoredMNIST exactly as DomainBed does.
    3 environments: e=0.1 (+90%), e=0.2 (+80%), e=0.9 (-90%)
    Returns list of (in_env, out_env) tuples.
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

    print(f"ColoredMNIST loaded:")
    for i, (e, env) in enumerate(zip(environments, envs)):
        marker = ': test' if i == 2 else ''
        print(f"  env{i} (e={e}): {len(env['images'])} samples{marker}")

    envs_splits = [split_env(env, holdout_frac, seed) for env in envs]
    return envs_splits


# PACS

def get_pacs(data_dir='./data', test_env_idx=0,
             holdout_frac=0.2, seed=0):
    sys.path.insert(0, 'DomainBed')
    from domainbed.datasets import MultipleEnvironmentImageFolder

    hparams = {'data_augmentation': True}

    # Use MultipleEnvironmentImageFolder directly with the actual data path
    dataset = MultipleEnvironmentImageFolder(
        data_dir, [test_env_idx], hparams['data_augmentation'], hparams)

    env_names = sorted([f.name for f in os.scandir(data_dir) if f.is_dir()])
    print(f"PACS loaded (test env: {env_names[test_env_idx]}):")

    envs_splits = []
    for i, env_dataset in enumerate(dataset.datasets):
        marker = ' : test' if i == test_env_idx else ''
        print(f"  env{i} ({env_names[i]}): {len(env_dataset)} images{marker}")
        in_env, out_env = split_env_subset(env_dataset, holdout_frac, seed)
        envs_splits.append((in_env, out_env))

    return envs_splits


# Dataset registry

DATASET_CONFIGS = {
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
}


def get_dataset(dataset_name, data_dir, test_env_idx=None,
                holdout_frac=0.2, seed=0):
    """
    Single entry point for all datasets.

    Usage:
        envs_splits = get_dataset('ColoredMNIST', data_dir='./data')
        envs_splits = get_dataset('PACS', data_dir='./data', test_env_idx=0)
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    cfg = DATASET_CONFIGS[dataset_name]

    if test_env_idx is None:
        test_env_idx = cfg['test_env_idx']

    if dataset_name == 'ColoredMNIST':
        return cfg['loader'](data_dir, holdout_frac, seed)
    else:
        return cfg['loader'](data_dir, test_env_idx, holdout_frac, seed)