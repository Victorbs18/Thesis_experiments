# src/datasets.py
"""
Dataset loading for all domain generalization experiments.

Each dataset returns a list of (in_env, out_env) tuples:
    envs_splits[i] = (in_env, out_env)
    in_env:  80% train portion: used for training
    out_env: 20% val portion: used for selection

Each env is a dict:
    {'images': Tensor(N, C, H, W), 'labels': Tensor(N,)}

Usage:
    from src.datasets import get_dataset
    envs_splits = get_dataset('ColoredMNIST', data_dir='./data')
"""

import os
import numpy as np
import torch
from torchvision.datasets import MNIST

# Shared utilities

def split_env(env, holdout_frac=0.2, seed=0):
    """
    Split environment into train (in) and val (out) subsets.
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

    # Fixed shuffle seed for reproducibility
    rng  = torch.Generator()
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
        marker = '← test' if i == 2 else ''
        print(f"  env{i} (e={e}): {len(env['images'])} samples {marker}")

    envs_splits = [split_env(env, holdout_frac, seed) for env in envs]
    return envs_splits


# Dataset registry

DATASET_CONFIGS = {
    'ColoredMNIST': {
        'loader':          get_colored_mnist,
        'n_envs':          3,
        'test_env_idx':    2,
        'n_steps':         5001,
        'env_names':       ['+90%', '+80%', '-90%'],
        'input_shape':     (2, 28, 28),
        'n_classes':       2,
        'selection_methods': ['IIDAccuracySelectionMethod',
                              'OracleSelectionMethod'],
        'domainbed_ref': {
            'ERM': {'iid':    {'env0': 71.7, 'env1': 72.9, 'env2': 10.0},
                    'oracle': {'env0': 71.8, 'env1': 72.9, 'env2': 28.7}},
            'IRM': {'iid':    {'env0': 72.5, 'env1': 73.3, 'env2': 10.2},
                    'oracle': {'env0': 72.0, 'env1': 72.5, 'env2': 58.5}},
        },
    },
    # PACS, Camelyon, OfficeHome 
}


def get_dataset(dataset_name, data_dir, holdout_frac=0.2, seed=0):
    """
    Single entry point for all datasets.

    Usage:
        envs_splits = get_dataset('ColoredMNIST', data_dir='./data')
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    cfg    = DATASET_CONFIGS[dataset_name]
    loader = cfg['loader']
    return loader(data_dir, holdout_frac, seed)