# analysis/umap_embeddings.py
"""
UMAP visualization of feature embeddings: Salaudeen et al. procedure.

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
        --test_env_idx 2 \
        --output_dir   results/coloredmnist/test_env2/cnn/random/plots \
        --ermseed      8 \
        --irmseed      14
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap


# Data utilities

def load_env_data(features_dir, algo, hparams_seed, trial, env_idx):
    """Load features, labels and colors for one env."""
    features = np.load(os.path.join(features_dir,
        f"{algo}_hpseed{hparams_seed}_trial{trial}_env{env_idx}_features.npy"))
    labels   = np.load(os.path.join(features_dir,
        f"env{env_idx}_labels.npy"))
    colors   = np.load(os.path.join(features_dir,
        f"env{env_idx}_colors.npy"))
    return features, labels, colors


# UMAP comparison plot

def plot_comparison(features_dir, test_env_idx, models, output_path):
    """
    Plot ERM and IRM side by side.
    models: list of (algo, hparams_seed, trial, title)
    """
    id_env_idxs = [i for i in range(3) if i != test_env_idx]

    groups = {
        (0, 0): ('<5 / Red',    '#FF6B6B'),
        (1, 0): ('>=5 / Red',   '#8B0000'),
        (0, 1): ('<5 / Green',  '#51CF66'),
        (1, 1): ('>=5 / Green', '#1B4D1F'),
    }

    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 7))
    if len(models) == 1:
        axes = [axes]

    for ax, (algo, hpseed, trial, title) in zip(axes, models):

        # Load ID features + labels + colors
        id_features_list = []
        id_labels_list   = []
        id_colors_list   = []
        for env_idx in id_env_idxs:
            feats, lbls, cols = load_env_data(
                features_dir, algo, hpseed, trial, env_idx)
            id_features_list.append(feats)
            id_labels_list.append(lbls)
            id_colors_list.append(cols)

        features_id = np.concatenate(id_features_list)
        labels_id   = np.concatenate(id_labels_list)
        colors_id   = np.concatenate(id_colors_list)

        # Load OOD features + labels + colors
        features_ood, labels_ood, colors_ood = load_env_data(
            features_dir, algo, hpseed, trial, test_env_idx)

        # Fit UMAP on ID, transform OOD
        print(f"  [{title}] Fitting UMAP on ID {features_id.shape}...")
        reducer      = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_id  = reducer.fit_transform(features_id)
        print(f"  [{title}] Transforming OOD {features_ood.shape}...")
        embedding_ood = reducer.transform(features_ood)

        # Plot ID — small circles
        for (lbl, col), (name, color) in groups.items():
            mask = (labels_id == lbl) & (colors_id == col)
            ax.scatter(embedding_id[mask, 0], embedding_id[mask, 1],
                       c=color, marker='o', s=4, alpha=0.3,
                       label=f"{name} ID")

        # Plot OOD — X markers
        for (lbl, col), (name, color) in groups.items():
            mask = (labels_ood == lbl) & (colors_ood == col)
            ax.scatter(embedding_ood[mask, 0], embedding_ood[mask, 1],
                       c=color, marker='X', s=25, alpha=0.9,
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
    print(f"\nSaved to {output_path}")


# Entry point

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir',  type=str, required=True)
    parser.add_argument('--test_env_idx',  type=int, default=2)
    parser.add_argument('--output_dir',    type=str, default='./plots')
    parser.add_argument('--ermseed',       type=int, default=8)
    parser.add_argument('--irmseed',       type=int, default=14)
    parser.add_argument('--trial',         type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models = [
        ('ERM', args.ermseed, args.trial,
         f'ERM seed={args.ermseed} (spurious)'),
        ('IRM', args.irmseed, args.trial,
         f'IRM seed={args.irmseed} (invariant)'),
    ]

    output_path = os.path.join(args.output_dir, 'umap_comparison.png')

    plot_comparison(
        features_dir = args.features_dir,
        test_env_idx = args.test_env_idx,
        models       = models,
        output_path  = output_path,
    )