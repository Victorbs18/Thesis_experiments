# src/train.py
"""
Training loop

- Handles both tensor envs (ColoredMNIST) and Dataset envs (PACS)
- All data kept in memory where possible
- Single process
- Evaluates only at final step
- Saves predictions, probabilities and features for later analysis
- Returns results in DomainBed's flat record format
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.hparams import HP_SEARCH_METHODS
from src.datasets import is_tensor_env, DATASET_CONFIGS

SKIP_HPARAMS = {
    'data_augmentation', 'resnet18', 'resnet50_augmix', 'dinov2',
    'vit', 'vit_attn_tune', 'freeze_bn', 'lars', 'linear_steps',
    'resnet_dropout', 'vit_dropout', 'class_balanced', 'nonlinear_classifier'
}


# Infinite data loader

def make_infinite_loader(env, batch_size, device):
    """
    Infinite loader — handles both tensor envs and Dataset envs.

    Tensor env (ColoredMNIST): data already in memory, shuffle manually.
    Dataset env (PACS): use DataLoader with num_workers.
    """
    if is_tensor_env(env):
        x = env['images'].to(device)
        y = env['labels'].to(device)
        n = len(x)
        while True:
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                if len(idx) < 2:
                    continue
                yield x[idx], y[idx]
    else:
        loader = DataLoader(
            env,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True,
        )
        while True:
            for x, y in loader:
                yield x.to(device), y.to(device)


# Evaluation

@torch.no_grad()
def evaluate(algorithm, env, device, batch_size=512):
    """
    Evaluate classification accuracy — handles both env types.
    """
    algorithm.eval()

    if is_tensor_env(env):
        x = env['images'].to(device)
        y = env['labels'].to(device)
        n = len(x)
        correct = 0
        for i in range(0, n, batch_size):
            xb   = x[i:i + batch_size]
            yb   = y[i:i + batch_size]
            pred = algorithm.predict(xb).argmax(1)
            correct += (pred == yb).sum().item()
        algorithm.train()
        return correct / n
    else:
        loader  = DataLoader(env, batch_size=batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=True)
        correct = total = 0
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            pred  = algorithm.predict(x).argmax(1)
            correct += (pred == y).sum().item()
            total   += len(y)
        algorithm.train()
        return correct / total if total > 0 else 0.0


# Save predictions, probs and features

@torch.no_grad()
def save_model_outputs(algorithm, all_envs, test_env_idx,
                       algorithm_name, hparams_seed, trial_seed,
                       save_dir, device, record):
    """
    Save predictions, probabilities and test env features for all envs.
    Handles both tensor and Dataset envs.
    """
    algorithm.eval()
    batch_size = 256

    for i, (in_env, out_env) in enumerate(all_envs):

        if is_tensor_env(out_env):
            x      = out_env['images'].to(device)
            logits = algorithm.predict(x)
            preds  = logits.argmax(1).cpu().numpy()
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            loader = DataLoader(out_env, batch_size=batch_size,
                                shuffle=False, num_workers=4,
                                pin_memory=True)
            preds_list = []
            probs_list = []
            for x, _ in loader:
                x      = x.to(device)
                logits = algorithm.predict(x)
                preds_list.append(logits.argmax(1).cpu().numpy())
                probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds = np.concatenate(preds_list)
            probs = np.concatenate(probs_list)

        np.save(os.path.join(save_dir,
            f"{algorithm_name}_hpseed{hparams_seed}_trial{trial_seed}_env{i}_preds.npy"),
            preds)
        np.save(os.path.join(save_dir,
            f"{algorithm_name}_hpseed{hparams_seed}_trial{trial_seed}_env{i}_probs.npy"),
            probs)
        record[f'env{i}_pred_path'] = os.path.join(save_dir,
            f"{algorithm_name}_hpseed{hparams_seed}_trial{trial_seed}_env{i}_preds.npy")
        record[f'env{i}_prob_path'] = os.path.join(save_dir,
            f"{algorithm_name}_hpseed{hparams_seed}_trial{trial_seed}_env{i}_probs.npy")

    # Save features on test env out split only
    _, test_out_env = all_envs[test_env_idx]

    if is_tensor_env(test_out_env):
        x        = test_out_env['images'].to(device)
        features = algorithm.featurizer(x).cpu().numpy()
    else:
        loader = DataLoader(test_out_env, batch_size=batch_size,
                            shuffle=False, num_workers=4,
                            pin_memory=True)
        features_list = []
        for x, _ in loader:
            features_list.append(algorithm.featurizer(x.to(device)).cpu().numpy())
        features = np.concatenate(features_list)

    fname = (f"{algorithm_name}_hpseed{hparams_seed}"
             f"_trial{trial_seed}_testenv{test_env_idx}_features.npy")
    np.save(os.path.join(save_dir, fname), features)
    record['feat_path'] = os.path.join(save_dir, fname)


# Single training run

def run_single(
    algorithm_class,
    dataset_name,
    train_envs,
    all_envs,
    test_env_idx,
    hparams_seed,
    trial_seed,
    hp,
    device,
    n_classes,
    n_steps=5001,
    save_dir=None,
    search_method='random',
):
    # Reproducibility
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)

    # Infer input shape
    if is_tensor_env(train_envs[0]):
        input_shape = tuple(train_envs[0]['images'].shape[1:])
    else:
        sample_x, _ = next(iter(DataLoader(train_envs[0], batch_size=2)))
        input_shape  = tuple(sample_x.shape[1:])

    n_domains = len(train_envs)

    algorithm = algorithm_class(
        input_shape, n_classes, n_domains, hp
    ).to(device)

    loaders = [
        make_infinite_loader(env, hp['batch_size'], device)
        for env in train_envs
    ]

    # Training loop
    t0 = time.time()
    for step in range(n_steps):
        algorithm.train()
        minibatches = [next(loader) for loader in loaders]
        algorithm.update(minibatches)
    train_time = time.time() - t0

    # Build record
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

    # Evaluate on all environments
    for i, (in_env, out_env) in enumerate(all_envs):
        record[f'env{i}_in_acc']  = evaluate(algorithm, in_env,  device)
        record[f'env{i}_out_acc'] = evaluate(algorithm, out_env, device)

    # Save outputs
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        algorithm.eval()
        with torch.no_grad():
            save_model_outputs(
                algorithm      = algorithm,
                all_envs       = all_envs,
                test_env_idx   = test_env_idx,
                algorithm_name = algorithm_class.__name__,
                hparams_seed   = hparams_seed,
                trial_seed     = trial_seed,
                save_dir       = save_dir,
                device         = device,
                record         = record,
            )

    return record


# Full sweep

def run_sweep(
    algorithm_classes,
    dataset_name,
    envs_splits,
    test_env_idx,
    n_hparams,
    n_trials,
    device,
    n_steps=5001,
    save_dir=None,
    search_method='random',
    backbone='resnet50',
):
    if search_method not in HP_SEARCH_METHODS:
        raise ValueError(
            f"Unknown search method '{search_method}'. "
            f"Available: {list(HP_SEARCH_METHODS.keys())}"
        )

    searcher  = HP_SEARCH_METHODS[search_method]
    n_classes = DATASET_CONFIGS[dataset_name]['n_classes']

    train_envs = [
        envs_splits[i][0]
        for i in range(len(envs_splits))
        if i != test_env_idx
    ]

    # Save dataset metadata once — labels and colors for tensor envs only
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _, sample_out = envs_splits[0]
        if is_tensor_env(sample_out):
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

        hp_configs = searcher.get_hparams(
            algorithm_class.__name__,
            dataset_name,
            n_hparams,
            backbone = backbone,
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
                    n_classes       = n_classes,
                    n_steps         = n_steps,
                    save_dir        = save_dir,
                    search_method   = search_method,
                )

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