# analysis/umap_embeddings.py
"""
UMAP visualization of feature embeddings: Salaudeen et al. procedure,
generalized to support any (dataset, backbone, algorithm) combination
and an arbitrary number of models arranged in a grid.

Fits UMAP on ID examples (all training-env out splits) and transforms
OOD examples (test-env out split) into the same 2D space.

For ColoredMNIST (n_classes=2, has color attribute):
    Points colored by 4 combinations of label x spurious color:
        < 5 / Red, >= 5 / Red, < 5 / Green, >= 5 / Green

For other datasets (e.g. PACS, no color attribute):
    Points colored by class label only (n_classes colors).

Shape distinguishes ID vs OOD:
    Circle → ID examples
    X      → OOD examples

Usage (single model):
    python analysis/umap_embeddings.py \
        --models coloredmnist:cnn:ERM:8:0 \
        --test_env_idx 2 --n_envs 3 \
        --output_path plots/umap_erm.png

Usage (grid comparison, multiple models):
    python analysis/umap_embeddings.py \
        --models coloredmnist:cnn:ERM:8:0:"ERM seed=8 (spurious)" \
                 coloredmnist:cnn:IRM:14:0:"IRM seed=14 (invariant)" \
        --test_env_idx 2 --n_envs 3 \
        --output_path plots/umap_comparison.png \
        --base_dir results

Each --models entry format: dataset:backbone:algorithm:hpseed:trial[:label]
    - dataset/backbone determine the features_dir path:
        {base_dir}/{dataset}/test_env{test_env_idx}/{backbone}/{search_method}/models
    - label is optional; defaults to "{algorithm} seed={hpseed} ({backbone})"
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap


# Class label names per dataset (for legend / class-only coloring)

DATASET_CLASS_NAMES = {
    'coloredmnist': ['0-4', '5-9'],
    'pacs':         ['dog', 'elephant', 'giraffe', 'guitar',
                     'horse', 'house', 'person'],
}

# Fixed color palette for class-only coloring (up to 10 classes)
CLASS_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


# Data utilities

def load_env_data(features_dir, algo, hparams_seed, trial, env_idx,
                  has_colors):
    """Load features, labels and (optionally) colors for one env."""
    features = np.load(os.path.join(features_dir,
        f"{algo}_hpseed{hparams_seed}_trial{trial}_env{env_idx}_features.npy"))
    labels   = np.load(os.path.join(features_dir,
        f"env{env_idx}_labels.npy"))
    colors = None
    if has_colors:
        colors_path = os.path.join(features_dir, f"env{env_idx}_colors.npy")
        if os.path.exists(colors_path):
            colors = np.load(colors_path)
    return features, labels, colors


def build_groups(dataset, labels, has_colors):
    """
    Return {group_key: (name, color_hex)} for the legend/coloring scheme.
    group_key is used to build boolean masks against (labels[, colors]).
    """
    if has_colors:
        # ColoredMNIST-style: label x color combinations
        return {
            (0, 0): ('label0 / color0 (red)',   '#FF6B6B'),
            (1, 0): ('label1 / color0 (red)',   '#8B0000'),
            (0, 1): ('label0 / color1 (green)', '#51CF66'),
            (1, 1): ('label1 / color1 (green)', '#1B4D1F'),
        }
    else:
        # Generic: class-only coloring
        n_classes  = len(set(labels.tolist()))
        class_names = DATASET_CLASS_NAMES.get(dataset.lower())
        groups = {}
        for c in range(n_classes):
            name = class_names[c] if class_names and c < len(class_names) \
                   else f'class {c}'
            groups[c] = (name, CLASS_PALETTE[c % len(CLASS_PALETTE)])
        return groups


def group_mask(group_key, labels, colors, has_colors):
    if has_colors:
        lbl, col = group_key
        return (labels == lbl) & (colors == col)
    else:
        return labels == group_key


# UMAP comparison plot

def plot_single_env(base_dir, env_idx, search_method, models,
                    output_path, n_cols=None):
    """
    Fit and plot UMAP using ONLY one environment's features (no ID/OOD
    split). Useful when label and spurious-color attributes are
    decorrelated in that env (e.g. ColoredMNIST env2), making the
    'clusters by color vs by label' question meaningful.

    For each model, produces a side-by-side pair of panels:
    colored by label, and colored by color attribute (if available).
    """
    n_models = len(models)
    has_colors_list = []

    # First pass: detect color availability per model
    for m in models:
        features_dir = os.path.join(
            base_dir, m['dataset'].lower(), f'test_env{env_idx}',
            m['backbone'], search_method, 'models')
        has_colors_list.append(os.path.exists(
            os.path.join(features_dir, f'env{env_idx}_colors.npy')))

    # Each model gets 1 or 2 panels (label-colored, and color-colored if available)
    panels_per_model = [2 if hc else 1 for hc in has_colors_list]
    total_panels = sum(panels_per_model)

    if n_cols is None:
        n_cols = total_panels
    n_rows = (total_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(7 * n_cols, 6 * n_rows),
                              squeeze=False)
    axes_flat = axes.flatten()

    panel_idx = 0
    for m, has_colors in zip(models, has_colors_list):
        dataset, backbone, algo = m['dataset'], m['backbone'], m['algorithm']
        hpseed, trial, title = m['hpseed'], m['trial'], m['title']

        features_dir = os.path.join(
            base_dir, dataset.lower(), f'test_env{env_idx}',
            backbone, search_method, 'models')

        features, labels, colors = load_env_data(
            features_dir, algo, hpseed, trial, env_idx, has_colors)

        print(f"  [{title}] Fitting UMAP on env{env_idx} {features.shape}...")
        reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
        embedding = reducer.fit_transform(features)

        n_classes  = len(set(labels.tolist()))
        class_names = DATASET_CLASS_NAMES.get(dataset.lower())

        # Panel 1: colored by label
        ax = axes_flat[panel_idx]
        for c in range(n_classes):
            name = class_names[c] if class_names and c < len(class_names) \
                   else f'label {c}'
            mask = labels == c
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=CLASS_PALETTE[c % len(CLASS_PALETTE)],
                       marker='o', s=6, alpha=0.5, label=name)
        ax.set_title(f"{title}\n(colored by label)", fontsize=12)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend(markerscale=2, fontsize=8)
        panel_idx += 1

        # Panel 2: colored by spurious color attribute (if available)
        if has_colors:
            ax = axes_flat[panel_idx]
            color_groups = {0: ('color 0 (red)', '#FF6B6B'),
                            1: ('color 1 (green)', '#51CF66')}
            for c, (name, hexcol) in color_groups.items():
                mask = colors == c
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=hexcol, marker='o', s=6, alpha=0.5, label=name)
            ax.set_title(f"{title}\n(colored by spurious color)", fontsize=12)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.legend(markerscale=2, fontsize=8)
            panel_idx += 1

    for ax in axes_flat[panel_idx:]:
        ax.axis('off')

    plt.suptitle(f'Feature embeddings on env{env_idx} only: '
                  f'label vs spurious-color clustering',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {output_path}")


def plot_comparison(base_dir, test_env_idx, n_envs, search_method, models,
                    output_path, n_cols=None):
    """
    Plot a grid of UMAP embeddings, one panel per model spec.

    models: list of dicts with keys
        dataset, backbone, algorithm, hpseed, trial, title
    """
    id_env_idxs = [i for i in range(n_envs) if i != test_env_idx]

    n_models = len(models)
    if n_cols is None:
        n_cols = n_models
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(8 * n_cols, 7 * n_rows),
                              squeeze=False)
    axes_flat = axes.flatten()

    for ax, m in zip(axes_flat, models):
        dataset   = m['dataset']
        backbone  = m['backbone']
        algo      = m['algorithm']
        hpseed    = m['hpseed']
        trial     = m['trial']
        title     = m['title']

        features_dir = os.path.join(
            base_dir, dataset.lower(), f'test_env{test_env_idx}',
            backbone, search_method, 'models')

        # Detect whether color attribute exists for this dataset
        has_colors = os.path.exists(
            os.path.join(features_dir, f'env{id_env_idxs[0]}_colors.npy'))

        # Load ID features + labels + colors
        id_features_list = []
        id_labels_list   = []
        id_colors_list   = []
        for env_idx in id_env_idxs:
            feats, lbls, cols = load_env_data(
                features_dir, algo, hpseed, trial, env_idx, has_colors)
            id_features_list.append(feats)
            id_labels_list.append(lbls)
            if has_colors:
                id_colors_list.append(cols)

        features_id = np.concatenate(id_features_list)
        labels_id   = np.concatenate(id_labels_list)
        colors_id   = np.concatenate(id_colors_list) if has_colors else None

        # Load OOD features + labels + colors
        features_ood, labels_ood, colors_ood = load_env_data(
            features_dir, algo, hpseed, trial, test_env_idx, has_colors)

        # Fit UMAP on ID, transform OOD
        print(f"  [{title}] Fitting UMAP on ID {features_id.shape}...")
        reducer       = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
        embedding_id  = reducer.fit_transform(features_id)
        print(f"  [{title}] Transforming OOD {features_ood.shape}...")
        embedding_ood = reducer.transform(features_ood)

        groups = build_groups(dataset, labels_id, has_colors)

        # Plot ID — small circles
        for group_key, (name, color) in groups.items():
            mask = group_mask(group_key, labels_id, colors_id, has_colors)
            ax.scatter(embedding_id[mask, 0], embedding_id[mask, 1],
                       c=color, marker='o', s=4, alpha=0.3,
                       label=f"{name} (ID)")

        # Plot OOD — X markers
        for group_key, (name, color) in groups.items():
            mask = group_mask(group_key, labels_ood, colors_ood, has_colors)
            ax.scatter(embedding_ood[mask, 0], embedding_ood[mask, 1],
                       c=color, marker='X', s=25, alpha=0.9,
                       label=f"{name} (OOD)")

        ax.set_title(title, fontsize=13)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend(markerscale=3, fontsize=7, ncol=2, framealpha=0.7)

    # Hide unused axes if grid is larger than n_models
    for ax in axes_flat[n_models:]:
        ax.axis('off')

    plt.suptitle('Feature embeddings: ID (circles) vs OOD (X)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {output_path}")


# Entry point

def parse_model_spec(spec):
    """
    Parse 'dataset:backbone:algorithm:hpseed:trial[:title]'
    """
    parts = spec.split(':')
    if len(parts) < 5:
        raise ValueError(
            f"Invalid model spec '{spec}'. "
            f"Expected dataset:backbone:algorithm:hpseed:trial[:title]")
    dataset, backbone, algorithm, hpseed, trial = parts[:5]
    title = parts[5] if len(parts) > 5 else \
        f"{algorithm} seed={hpseed} ({backbone})"
    return {
        'dataset':   dataset,
        'backbone':  backbone,
        'algorithm': algorithm,
        'hpseed':    int(hpseed),
        'trial':     int(trial),
        'title':     title,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help="One or more 'dataset:backbone:algorithm:"
                             "hpseed:trial[:title]' specs")
    parser.add_argument('--test_env_idx',  type=int, required=True)
    parser.add_argument('--n_envs',        type=int, required=True)
    parser.add_argument('--base_dir',      type=str, default='./results')
    parser.add_argument('--search_method', type=str, default='random')
    parser.add_argument('--output_path',   type=str, required=True)
    parser.add_argument('--n_cols',        type=int, default=None,
                        help='Columns in grid (default: all in one row)')
    parser.add_argument('--single_env',    action='store_true',
                        help='Fit+plot UMAP using only test_env_idx '
                             '(no ID/OOD split), with separate '
                             'label-colored and color-colored panels')
    args = parser.parse_args()

    models = [parse_model_spec(s) for s in args.models]

    if args.single_env:
        plot_single_env(
            base_dir      = args.base_dir,
            env_idx       = args.test_env_idx,
            search_method = args.search_method,
            models        = models,
            output_path   = args.output_path,
            n_cols        = args.n_cols,
        )
    else:
        plot_comparison(
            base_dir      = args.base_dir,
            test_env_idx  = args.test_env_idx,
            n_envs        = args.n_envs,
            search_method = args.search_method,
            models        = models,
            output_path   = args.output_path,
            n_cols        = args.n_cols,
        )