# analysis/extract_features.py
"""
Retrains selected models and saves feature vectors for all environments.
Used for UMAP visualization after sweep + selection.

Usage:
    python analysis/extract_features.py \
        --records_path results/coloredmnist/test_env2/cnn/random/records.json \
        --output_dir   results/coloredmnist/test_env2/cnn/random/models \
        --data_dir     ./data \
        --models ERM:19:0 IRM:3:0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DomainBed'))

import json
import argparse
import numpy as np
import torch

from domainbed.algorithms import ERM, IRM
from src.datasets import get_colored_mnist
from src.train import make_infinite_loader

ALGORITHMS = {'ERM': ERM, 'IRM': IRM}


def retrain_and_extract(
    algo_name, hparams_seed, trial_seed,
    records_path, output_dir, data_dir,
    test_env_idx=2, n_steps=5001, device='cuda'
):
    """Retrain one model with fixed seeds and save features for all envs."""

    # Load hparams from records
    with open(records_path) as f:
        records = json.load(f)

    matching = [r for r in records
                if r['algorithm'] == algo_name
                and r['args']['hparams_seed'] == hparams_seed
                and r['args']['trial_seed'] == trial_seed]

    if not matching:
        print(f"No record found for {algo_name} seed={hparams_seed} trial={trial_seed}")
        return

    hp = matching[0]['hparams']
    print(f"\nRetraining {algo_name} hpseed={hparams_seed} trial={trial_seed}")
    print(f"  hp: lr={hp['lr']:.5f} bs={hp['batch_size']}")

    # Load data
    envs_splits = get_colored_mnist(data_dir=data_dir)

    # Training envs
    train_envs = [envs_splits[i][0]
                  for i in range(len(envs_splits))
                  if i != test_env_idx]

    # Set seeds
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)

    # Build algorithm
    input_shape = tuple(train_envs[0]['images'].shape[1:])
    n_classes   = 2
    n_domains   = len(train_envs)

    algo_class = ALGORITHMS[algo_name]
    algorithm  = algo_class(input_shape, n_classes, n_domains, hp).to(device)

    # Train
    loaders = [make_infinite_loader(env, hp['batch_size'], device)
               for env in train_envs]

    print(f"  Training for {n_steps} steps...")
    for step in range(n_steps):
        algorithm.train()
        minibatches = [next(loader) for loader in loaders]
        algorithm.update(minibatches)

    algorithm.eval()

    # Save features for ALL env out splits
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i, (in_env, out_env) in enumerate(envs_splits):
            x        = out_env['images'].to(device)
            features = algorithm.featurizer(x).cpu().numpy()
            fname    = (f"{algo_name}_hpseed{hparams_seed}"
                        f"_trial{trial_seed}_env{i}_features.npy")
            path = os.path.join(output_dir, fname)
            np.save(path, features)
            print(f"  Saved env{i} features: {features.shape} → {path}")

    print(f"  Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records_path', type=str, required=True)
    parser.add_argument('--output_dir',   type=str, required=True)
    parser.add_argument('--data_dir',     type=str, default='./data')
    parser.add_argument('--test_env_idx', type=int, default=2)
    parser.add_argument('--n_steps',      type=int, default=5001)
    parser.add_argument('--device',       type=str, default='cuda')
    parser.add_argument('--models',       type=str, nargs='+',
                        default=['ERM:19:0', 'IRM:3:0'],
                        help='Models to retrain as ALGO:hpseed:trial')
    args = parser.parse_args()

    for model_str in args.models:
        algo, hpseed, trial = model_str.split(':')
        retrain_and_extract(
            algo_name    = algo,
            hparams_seed = int(hpseed),
            trial_seed   = int(trial),
            records_path = args.records_path,
            output_dir   = args.output_dir,
            data_dir     = args.data_dir,
            test_env_idx = args.test_env_idx,
            n_steps      = args.n_steps,
            device       = args.device,
        )