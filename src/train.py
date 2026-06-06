# src/train.py
"""
Training loop

- All data kept in memory 
- Single process
- Evaluates only at final step
- Saves model weights for later analysis (UMAP, probing, agreement)
- Returns results in DomainBed's flat record format for direct
  compatibility with their Q object and selection methods

Usage:
    from src.train import run_sweep
    from src.hparams import RandomSearch

    records = run_sweep(
        algorithm_classes = [ERM, IRM],
        dataset_name      = 'ColoredMNIST',
        envs_splits       = envs_splits,   # list of (in_env, out_env) tuples
        test_env_idx      = 2,
        n_hparams         = 20,
        n_trials          = 3,
        device            = 'cuda',
        n_steps           = 5001,
        save_dir          = './results/colored_mnist',
        search_method     = 'random',
    )
"""

import os
import time
import numpy as np
import torch

from src.hparams import HP_SEARCH_METHODS

SKIP_HPARAMS = {
    'data_augmentation', 'resnet18', 'resnet50_augmix', 'dinov2',
    'vit', 'vit_attn_tune', 'freeze_bn', 'lars', 'linear_steps',
    'resnet_dropout', 'vit_dropout', 'class_balanced', 'nonlinear_classifier'
}


# Infinite data loader


def make_infinite_loader(env, batch_size,device):
    """
    Infinite loader over an in-memory environment dict.
    
    env: {'images': Tensor(N, C, H, W), 'labels': Tensor(N,)}
    
    Keeps all data in memory.
    Shuffles every epoch automatically.
    """
    x = env['images'].to(device)  
    y = env['labels'].to(device)
    n    = len(x)
    while True:
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:   # following DomainBed: drop last incomplete batch
                continue
            yield x[idx], y[idx]


# Evaluation

@torch.no_grad()
def evaluate(algorithm, env, device, batch_size=512):
    """
    Evaluate classification accuracy on an in-memory environment.
    
    Uses batched inference to handle large environments.
    Returns float in [0, 1].
    """
    algorithm.eval()
    x = env['images'].to(device)
    y = env['labels'].to(device)
    n = len(x)
    correct = 0
    for i in range(0, n, batch_size):
        xb      = x[i:i + batch_size]
        yb      = y[i:i + batch_size]
        pred    = algorithm.predict(xb).argmax(1)
        correct += (pred == yb).sum().item()
    algorithm.train()
    return correct / n


# Single training run

def run_single(
    algorithm_class, # DomainBed algorithm class (ERM, IRM, ...)
    dataset_name, # str: used for HP sampling ranges
    train_envs, #  list of in_env dicts for training
    all_envs, # list of (in_env, out_env) tuples for evaluation
    test_env_idx, # int: index of held-out test environment
    hparams_seed, # identifies which HP config this is
    trial_seed, # int: controls weight init and batch order
    hp, # dict: hyperparameters for this run
    device, # str: 'cuda' or 'cpu'
    n_steps=5001, # int: number of training steps
    save_dir=None, # str or None: directory to save model weights
    search_method='random', # str: logged in record for reference
):
    """
    Train one (algorithm, hparams, seed) configuration.

    Returns
    -------
    record : dict in DomainBed's flat format, compatible with Q object:
        {
            'args':         {'test_envs': [2], 'hparams_seed': 0, ...},
            'hparams':      {'lr': 0.001, 'batch_size': 64, ...},
            'step':         5001,
            'algorithm':    'ERM',
            'env0_in_acc':  0.89,
            'env0_out_acc': 0.85,
            'env1_in_acc':  0.87,
            'env1_out_acc': 0.83,
            'env2_in_acc':  0.10,
            'env2_out_acc': 0.09,
            'train_time':   45.2,
            'model_path':   './results/ERM_hp0_trial0_testenv2.pt',
            'search_method': 'random',
        }
    """
    # Reproducibility
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)

    # Infer problem dimensions from data
    input_shape = tuple(train_envs[0]['images'].shape[1:])
    n_classes   = int(max(
        env['labels'].max().item()
        for in_env, out_env in all_envs
        for env in [in_env, out_env]
    ) + 1)
    n_domains = len(train_envs)

    # Instantiate DomainBed algorithm
    algorithm = algorithm_class(
        input_shape, n_classes, n_domains, hp
    ).to(device)

    # Infinite loaders: data stays in memory
    loaders = [
        make_infinite_loader(env, hp['batch_size'],device)
        for env in train_envs
    ]

    # Training loop
    t0 = time.time()
    for step in range(n_steps):
        algorithm.train()
        minibatches = [next(loader) for loader in loaders]
        algorithm.update(minibatches)
    train_time = time.time() - t0

    # Build DomainBed-format record
    record = {
        'args': {
            'test_envs':     [test_env_idx],
            'hparams_seed':  hparams_seed,
            'trial_seed':    trial_seed,
            'dataset':       dataset_name,
            'algorithm':     algorithm_class.__name__,
            'search_method': search_method,
        },
        'hparams':       dict(hp),
        'step':          n_steps,
        'algorithm':     algorithm_class.__name__,
        'train_time':    train_time,
        'model_path':    None,
        'search_method': search_method,
    }

    # Evaluate on all environments × both splits
    for i, (in_env, out_env) in enumerate(all_envs):
        record[f'env{i}_in_acc']  = evaluate(algorithm, in_env,  device)
        record[f'env{i}_out_acc'] = evaluate(algorithm, out_env, device)

    # Save model weights for later analysis
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        algorithm.eval()
        with torch.no_grad():

            # Save logits/probs on ALL env out splits
            for i, (in_env, out_env) in enumerate(all_envs):
                x      = out_env['images'].to(device)
                logits = algorithm.predict(x)
                preds  = logits.argmax(1).cpu().numpy()
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                
                np.save(os.path.join(save_dir,
                    f"{algorithm_class.__name__}_hpseed{hparams_seed}_trial{trial_seed}_env{i}_preds.npy"), preds)
                np.save(os.path.join(save_dir,
                    f"{algorithm_class.__name__}_hpseed{hparams_seed}_trial{trial_seed}_env{i}_probs.npy"), probs)
                
                record[f'env{i}_pred_path'] = os.path.join(save_dir, 
                    f"{algorithm_class.__name__}_hpseed{hparams_seed}_trial{trial_seed}_env{i}_preds.npy")
                record[f'env{i}_prob_path'] = os.path.join(save_dir,
                    f"{algorithm_class.__name__}_hpseed{hparams_seed}_trial{trial_seed}_env{i}_probs.npy")

            # Save feature vectors on test env out split only
            _, test_out_env = all_envs[test_env_idx]
            x        = test_out_env['images'].to(device)
            features = algorithm.featurizer(x).cpu().numpy()
            fname    = (f"{algorithm_class.__name__}"
                        f"_hpseed{hparams_seed}"
                        f"_trial{trial_seed}"
                        f"_testenv{test_env_idx}_features.npy")
            np.save(os.path.join(save_dir, fname), features)
            record['feat_path'] = os.path.join(save_dir, fname)
    return record

# Full sweep

def run_sweep(
    algorithm_classes, # list of DomainBed algorithm classes
    dataset_name, # str:  'ColoredMNIST', 'PACS'
    envs_splits, # list of (in_env, out_env) tuples, one per environment: in_env:  80% training portion, out_env: 20% validation portion
    test_env_idx, # int: index of held-out test environment
    n_hparams, # int: number of HP configurations to try
    n_trials, # int: number of seeds per HP configuration
    device, # str: 'cuda' or 'cpu'
    n_steps=5001,
    save_dir=None,# str or None: directory to save model weights
    search_method='random', 
):
    """
    Run a full hyperparameter sweep.

    Returns
    -------
    records : list of flat record dicts, ready for DomainBed's Q object
    """
    if search_method not in HP_SEARCH_METHODS:
        raise ValueError(
            f"Unknown search method '{search_method}'. "
            f"Available: {list(HP_SEARCH_METHODS.keys())}"
        )

    searcher = HP_SEARCH_METHODS[search_method]

    # Training envs = in_split of all non-test environments
    train_envs = [
        envs_splits[i][0]
        for i in range(len(envs_splits))
        if i != test_env_idx
    ]

    # Save dataset metadata once: labels and colors for all env out splits
    # Same dataset object as training: guaranteed alignment with saved features
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i, (in_env, out_env) in enumerate(envs_splits):
            labels = out_env['labels'].numpy()
            images = out_env['images']
            colors = (images[:, 1, :, :].sum(dim=(1, 2)) > 0).numpy().astype(np.int32)
            np.save(os.path.join(save_dir, f'env{i}_labels.npy'), labels)
            np.save(os.path.join(save_dir, f'env{i}_colors.npy'), colors)
        print(f"  Dataset metadata saved to {save_dir}")

    records = []
    total   = len(algorithm_classes) * n_hparams * n_trials
    done    = 0

    for algorithm_class in algorithm_classes:

        # Get HP configs from chosen search method
        hp_configs = searcher.get_hparams(
            algorithm_class.__name__,
            dataset_name,
            n_hparams,
        )

        for hp_config in hp_configs:
            hparams_seed = hp_config['hparams_seed']
            hp           = hp_config['hparams']

            for trial_seed in range(n_trials):
                done += 1
                hp_str = ' '.join(
                    f"{k}={v:.3g}" for k, v in hp.items()
                    if k not in SKIP_HPARAMS
                )
                print(
                    f"[{done}/{total}] {algorithm_class.__name__} "
                    f"hp={hparams_seed} trial={trial_seed} | {hp_str}",
                    flush=True,
                )
                record = run_single(
                    algorithm_class = algorithm_class,
                    dataset_name    = dataset_name,
                    train_envs      = train_envs,
                    all_envs        = envs_splits,
                    test_env_idx    = test_env_idx,
                    hparams_seed    = hparams_seed,
                    trial_seed      = trial_seed,
                    hp              = hp,
                    device          = device,
                    n_steps         = n_steps,
                    save_dir        = save_dir,
                    search_method   = search_method,
                )

                # Print progress
                env_accs = " | ".join(
                    f"env{i} in={record[f'env{i}_in_acc']:.3f} "
                    f"out={record[f'env{i}_out_acc']:.3f}"
                    for i in range(len(envs_splits))
                )
                print(
                    f"  {env_accs} | "
                    f"time={record['train_time']:.1f}s",
                    flush=True,
                )

                records.append(record)

    return records