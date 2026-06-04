# main.py
"""
Single entry point for all domain generalization experiments.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DomainBed'))

import argparse
import json
import os
import time
import numpy as np
import torch
from domainbed.lib.query import Q
from domainbed.algorithms import ERM, IRM, GroupDRO, CORAL, DANN, VREx
from domainbed.model_selection import (
    IIDAccuracySelectionMethod,
    LeaveOneOutSelectionMethod,
    OracleSelectionMethod,
)

from src.datasets import get_dataset, DATASET_CONFIGS
from src.train import run_sweep

# Algorithm registry

ALGORITHMS = {
    'ERM':      ERM,
    'IRM':      IRM,
    'GroupDRO': GroupDRO,
    'CORAL':    CORAL,
    'DANN':     DANN,
    'VREx':     VREx,
}

SELECTION_METHODS = {
    'IIDAccuracySelectionMethod':  IIDAccuracySelectionMethod,
    'LeaveOneOutSelectionMethod':  LeaveOneOutSelectionMethod,
    'OracleSelectionMethod':       OracleSelectionMethod,
}

# Argument parsing

def parse_args():
    parser = argparse.ArgumentParser(
        description='Domain generalization sweep'
    )
    parser.add_argument('--dataset',       type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to run')
    parser.add_argument('--algorithms',    type=str, default=None,
                        help='Comma-separated algorithms. Default: all')
    parser.add_argument('--n_hparams',     type=int, default=20)
    parser.add_argument('--n_trials',      type=int, default=3)
    parser.add_argument('--n_steps',       type=int, default=None,
                        help='Training steps. Default: dataset default')
    parser.add_argument('--test_env',      type=int, default=None,
                        help='Test env index. Default: dataset default')
    parser.add_argument('--device',        type=str, default='cuda')
    parser.add_argument('--data_dir',      type=str, default='./data')
    parser.add_argument('--output_dir',    type=str, default='./results')
    parser.add_argument('--search_method', type=str, default='random',
                        choices=['random', 'grid', 'bayesian'])
    parser.add_argument('--backbone', type=str, default='cnn',
                    choices=['cnn', 'resnet50', 'clip'])
    return parser.parse_args()


# Results printing

def print_results(all_results, dataset_cfg, algo_names, dataset_name):
    env_names  = dataset_cfg['env_names']
    n_envs     = dataset_cfg['n_envs']
    ref        = dataset_cfg.get('domainbed_ref', {})
    sel_names  = dataset_cfg['selection_methods']

    print(f"\n{'='*75}")
    print(f"  RESULTS — {dataset_name}")
    print(f"{'='*75}")

    for sel_name in sel_names:
        method = SELECTION_METHODS[sel_name]
        print(f"\n  Selection: {method.name}")
        print(f"  {'Algorithm':<12} "
              + " ".join(f"{e:>16}" for e in env_names)
              + f"  {'Ref (IID)':>10}")
        print(f"  {'─'*75}")

        for algo_name in algo_names:
            if algo_name not in all_results:
                continue
            if sel_name not in all_results[algo_name]:
                continue

            accs     = all_results[algo_name][sel_name]
            ref_acc  = ref.get(algo_name, {}).get('iid', {})
            test_idx = dataset_cfg['test_env_idx']
            ref_str  = (f"{ref_acc.get(f'env{test_idx}', 0):.1f}%"
                        if ref_acc else "—")

            row = " ".join(
                f"{accs[f'env{i}_mean']*100:>6.1f}±{accs[f'env{i}_std']*100:>4.1f}%"
                for i in range(n_envs)
            )
            print(f"  {algo_name:<12} {row}  {ref_str:>10}")


# Main

def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Dataset config
    dataset_cfg  = DATASET_CONFIGS[args.dataset]
    test_env_idx = args.test_env  if args.test_env  is not None \
                                  else dataset_cfg['test_env_idx']
    n_steps      = args.n_steps   if args.n_steps   is not None \
                                  else dataset_cfg['n_steps']

    # Output dir
    output_dir = os.path.join(args.output_dir,args.dataset.lower(),f'test_env{test_env_idx}',args.backbone,args.search_method)

    # Algorithms
    if args.algorithms is not None:
        algo_names = args.algorithms.split(',')
    else:
        algo_names = list(ALGORITHMS.keys())

    algo_classes = []
    for name in algo_names:
        if name not in ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{name}'. "
                f"Available: {list(ALGORITHMS.keys())}"
            )
        algo_classes.append(ALGORITHMS[name])

    # Selection methods for this dataset
    sel_methods = [
        SELECTION_METHODS[s]
        for s in dataset_cfg['selection_methods']
    ]

    print(f"{'='*60}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Algorithms:   {algo_names}")
    print(f"  Test env:     {test_env_idx} "
          f"({dataset_cfg['env_names'][test_env_idx]})")
    print(f"  n_hparams:    {args.n_hparams}")
    print(f"  n_trials:     {args.n_trials}")
    print(f"  n_steps:      {n_steps}")
    print(f"  Device:       {device}")
    print(f"  Search:       {args.search_method}")
    print(f"{'='*60}")

    # Load data
    envs_splits = get_dataset(
        args.dataset,
        data_dir=args.data_dir,
    )

    # Run sweep
    save_dir = os.path.join(output_dir, 'models')
    t0       = time.time()

    records = run_sweep(
        algorithm_classes = algo_classes,
        dataset_name      = args.dataset,
        envs_splits       = envs_splits,
        test_env_idx      = test_env_idx,
        n_hparams         = args.n_hparams,
        n_trials          = args.n_trials,
        device            = device,
        n_steps           = n_steps,
        save_dir          = save_dir,
        search_method     = args.search_method,
    )

    print(f"\nSweep completed in {(time.time()-t0)/60:.1f} min")

    # Save raw records
    records_path = os.path.join(output_dir, 'records.json')
    with open(records_path, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Records saved to {records_path}")

    # Apply selection methods using DomainBed's Q object
    q          = Q(records)
    all_results = {}

    for algo_name in algo_names:
        algo_records = q.filter(lambda r: r['algorithm'] == algo_name)
        all_results[algo_name] = {}

        for method in sel_methods:
            sel_name = method.__name__

            # Get best HP config
            hparams_accs = method.hparams_accs(algo_records)
            if not len(hparams_accs):
                print(f"  {algo_name} [{method.name}]: no results")
                continue

            # Get all records for the selected HP config
            best_records = hparams_accs[0][1]

            # Filter to last step only
            last_step_records = best_records.sorted(lambda r: r['step'])
            n_envs = dataset_cfg['n_envs']

            # Compute mean ± std across trials for each env
            per_env_accs = {}
            for i in range(n_envs):
                trial_accs = [r[f'env{i}_out_acc'] for r in last_step_records]
                per_env_accs[f'env{i}_mean'] = float(np.mean(trial_accs))
                per_env_accs[f'env{i}_std']  = float(np.std(trial_accs))

            all_results[algo_name][sel_name] = per_env_accs

            test_mean = per_env_accs[f'env{test_env_idx}_mean']
            test_std  = per_env_accs[f'env{test_env_idx}_std']
            print(f"  {algo_name} [{method.name}]: "
                f"test_acc={test_mean:.3f} ± {test_std:.3f}")

    # Print summary table
    print_results(all_results, dataset_cfg, algo_names, args.dataset)

    # Save final results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal results saved to {results_path}")


if __name__ == '__main__':
    main()
