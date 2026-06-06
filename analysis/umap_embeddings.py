# analysis/umap_embeddings.py
"""
UMAP visualization of feature embeddings — Salaudeen et al. procedure.

Fits UMAP on ID examples (env0 + env1 out splits) and transforms
OOD examples (env2 out split) into the same 2D space.

Points colored by 4 combinations of label x spurious feature:
    < 5 / Red    (label 0, color 0)
    >= 5 / Red   (label 1, color 0)
    < 5 / Green  (label 0, color 1)
    >= 5 / Green (label 1, color 1)

Shape distinguishes ID vs OOD:
    Circle → ID examples
    X      → OOD examples

Usage:
    python analysis/umap_embeddings.py \
        --features_dir results/coloredmnist/test_env2/cnn/random/models \
        --data_dir     ./data \
        --test_env_idx 2 \
        --output_dir   results/coloredmnist/test_env2/cnn/random/plots
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DomainBed'))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap

from src.datasets import get_colored_mnist


# Data utilities

def get_labels_and_colors(envs_splits, env_idx):
    """
    Extract labels and colors for out_split of given environment.
    labels: 0 or 1 (digit < 5)
    colors: 0=red, 1=green
    """
    _, out_env = envs_splits[env_idx]
    labels = out_env['labels'].numpy()
    images = out_env['images']
    colors = (images[:, 1, :, :].sum(dim=(1, 2)) > 0).numpy().astype(int)
    return labels, colors


def load_features(features_dir, algo, hparams_seed, trial, env_idx):
    fname = (f"{algo}_hpseed{hparams_seed}"
             f"_trial{trial}_env{env_idx}_features.npy")
    path = os.path.join(features_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features not found: {path}")
    return np.load(path)


# UMAP plot

def plot_umap_salaudeen(
    features_id, labels_id, colors_id,
    features_ood, labels_ood, colors_ood,
    title, output_path
):
    """
    Fit UMAP on ID features, transform OOD features.
    Plot both with circle=ID, X=OOD, colored by label x color.
    """
    print(f"  Fitting UMAP on ID features {features_id.shape}...")
    reducer      = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_id  = reducer.fit_transform(features_id)
    print(f"  Transforming OOD features {features_ood.shape}...")
    embedding_ood = reducer.transform(features_ood)

    # 4 groups: label x color
    groups = {
        (0, 0): ('<5 / Red',    '#FF6B6B'),
        (1, 0): ('>=5 / Red',   '#8B0000'),
        (0, 1): ('<5 / Green',  '#51CF66'),
        (1, 1): ('>=5 / Green', '#1B4D1F'),
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot ID points — circles
    for (lbl, col), (name, color) in groups.items():
        mask = (labels_id == lbl) & (colors_id == col)
        ax.scatter(
            embedding_id[mask, 0],
            embedding_id[mask, 1],
            c=color, marker='o',
            s=3, alpha=0.4,
            label=f"{name} ID"
        )

    # Plot OOD points — X markers, larger
    for (lbl, col), (name, color) in groups.items():
        mask = (labels_ood == lbl) & (colors_ood == col)
        ax.scatter(
            embedding_ood[mask, 0],
            embedding_ood[mask, 1],
            c=color, marker='X',
            s=15, alpha=0.8,
            label=f"{name} OOD"
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(markerscale=3, loc='best', fontsize=7,
              ncol=2, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


# Side by side comparison

def plot_comparison(
    features_dir, envs_splits, test_env_idx,
    models, output_path
):
    """
    Plot ERM and IRM side by side.
    models: list of (algo, hparams_seed, trial, title)
    """
    # Load labels/colors once
    id_env_idxs = [i for i in range(len(envs_splits)) if i != test_env_idx]

    # ID: combine env0 + env1 out splits
    id_features_list = []
    id_labels_list   = []
    id_colors_list   = []

    # OOD: env2 out split
    labels_ood, colors_ood = get_labels_and_colors(envs_splits, test_env_idx)

    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 7))
    if len(models) == 1:
        axes = [axes]

    groups = {
        (0, 0): ('<5 / Red',    '#FF6B6B'),
        (1, 0): ('>=5 / Red',   '#8B0000'),
        (0, 1): ('<5 / Green',  '#51CF66'),
        (1, 1): ('>=5 / Green', '#1B4D1F'),
    }

    for ax, (algo, hpseed, trial, title) in zip(axes, models):

        # Load ID features
        id_features_list = []
        id_labels_list   = []
        id_colors_list   = []
        for env_idx in id_env_idxs:
            feats          = load_features(features_dir, algo, hpseed, trial, env_idx)
            lbls, cols     = get_labels_and_colors(envs_splits, env_idx)
            id_features_list.append(feats)
            id_labels_list.append(lbls)
            id_colors_list.append(cols)

        features_id = np.concatenate(id_features_list)
        labels_id   = np.concatenate(id_labels_list)
        colors_id   = np.concatenate(id_colors_list)

        # Load OOD features
        features_ood = load_features(features_dir, algo, hpseed, trial, test_env_idx)

        # Fit UMAP on ID, transform OOD
        print(f"  [{title}] Fitting UMAP on ID {features_id.shape}...")
        reducer      = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_id  = reducer.fit_transform(features_id)
        print(f"  [{title}] Transforming OOD {features_ood.shape}...")
        embedding_ood = reducer.transform(features_ood)

        # Plot ID — circles
        for (lbl, col), (name, color) in groups.items():
            mask = (labels_id == lbl) & (colors_id == col)
            ax.scatter(embedding_id[mask, 0], embedding_id[mask, 1],
                       c=color, marker='o', s=3, alpha=0.3,
                       label=f"{name} ID")

        # Plot OOD — X
        for (lbl, col), (name, color) in groups.items():
            mask = (labels_ood == lbl) & (colors_ood == col)
            ax.scatter(embedding_ood[mask, 0], embedding_ood[mask, 1],
                       c=color, marker='X', s=20, alpha=0.9,
                       label=f"{name} OOD")

        ax.set_title(title, fontsize=13)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend(markerscale=3, fontsize=7, ncol=2, framealpha=0.7)

    plt.suptitle('Feature embeddings: ID (circles) vs OOD (X)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison saved to {output_path}")


# Entry point

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir',  type=str, required=True)
    parser.add_argument('--data_dir',      type=str, default='./data')
    parser.add_argument('--test_env_idx',  type=int, default=2)
    parser.add_argument('--output_dir',    type=str, default='./plots')
    parser.add_argument('--ermseed',       type=int, default=19)
    parser.add_argument('--irmseed',       type=int, default=3)
    parser.add_argument('--trial',         type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset once
    print("Loading ColoredMNIST...")
    envs_splits = get_colored_mnist(data_dir=args.data_dir)

    models = [
        ('ERM', args.ermseed, args.trial,
         f'ERM seed={args.ermseed} (spurious)'),
        ('IRM', args.irmseed, args.trial,
         f'IRM seed={args.irmseed} (invariant)'),
    ]

    output_path = os.path.join(args.output_dir, 'umap_comparison.png')

    plot_comparison(
        features_dir  = args.features_dir,
        envs_splits   = envs_splits,
        test_env_idx  = args.test_env_idx,
        models        = models,
        output_path   = output_path,
    )