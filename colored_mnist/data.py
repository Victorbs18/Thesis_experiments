"""
data.py — Colored MNIST data loading and environment construction
=================================================================
Self-contained. No dependency on practical work codebase.
"""

import numpy as np
import torch


def load_mnist_raw():
    """Load raw MNIST tensors once. Returns train and val splits."""
    from torchvision import datasets
    print("Loading MNIST (once)...")
    mnist        = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    train_images = mnist.data[:50000]
    train_labels = mnist.targets[:50000]
    val_images   = mnist.data[50000:]
    val_labels   = mnist.targets[50000:]
    print(f"  train: {train_images.shape}   val: {val_images.shape}")
    return train_images, train_labels, val_images, val_labels


def make_environment(images, labels, e, seed=0):
    """
    Build one colored environment from raw tensors.

    e = correlation strength between color and label.
    e=0.1 means color is almost anticorrelated with label (hard OOD).
    e=0.9 means color is strongly correlated with label (easy, close to train).

    Causal feature: digit shape (invariant across all e values)
    Spurious feature: color (correlation = e, changes across environments)
    """
    rng    = np.random.RandomState(seed)
    labels = (labels < 5).float()
    # Flip label with 25% noise
    labels = torch.logical_xor(labels, torch.from_numpy(
        rng.binomial(1, 0.25, size=labels.shape).astype(bool)
    )).float()
    # Color correlated with label at strength e
    colors = torch.logical_xor(labels.bool(), torch.from_numpy(
        rng.binomial(1, 1 - e, size=labels.shape).astype(bool)
    )).float()
    imgs = images.float() / 255.0
    imgs = imgs.unsqueeze(1).repeat(1, 3, 1, 1)
    imgs[:, 0, :, :] *= (1 - colors).unsqueeze(1).unsqueeze(2)  # red channel
    imgs[:, 1, :, :] *= colors.unsqueeze(1).unsqueeze(2)         # green channel
    imgs[:, 2, :, :] *= 0.0                                       # blue = 0
    return {
        "images": imgs.view(imgs.shape[0], -1),  # flatten: (N, 3*28*28)
        "labels": labels.unsqueeze(1),            # (N, 1)
    }


def build_envs(train_images, train_labels, e_values, seed):
    """
    Shuffle training data and split into len(e_values) environments.
    Each environment gets a different color-label correlation strength.
    """
    rng  = np.random.RandomState(seed)
    idx  = rng.permutation(len(train_images))
    imgs = train_images[idx]
    lbls = train_labels[idx]
    n    = len(imgs) // len(e_values)
    return [
        make_environment(imgs[i*n:(i+1)*n], lbls[i*n:(i+1)*n], e, seed=seed+i)
        for i, e in enumerate(e_values)
    ]


def make_val_splits(envs, val_frac=0.2):
    """
    Hold out val_frac of each environment for train-domain validation.
    Returns (train_envs, val_envs) — same structure as envs.
    This is the non-oracle selection signal: no OOD labels used.
    """
    train_envs, val_envs = [], []
    for env in envs:
        n     = len(env["images"])
        split = int((1 - val_frac) * n)
        train_envs.append({
            "images": env["images"][:split],
            "labels": env["labels"][:split],
        })
        val_envs.append({
            "images": env["images"][split:],
            "labels": env["labels"][split:],
        })
    return train_envs, val_envs
