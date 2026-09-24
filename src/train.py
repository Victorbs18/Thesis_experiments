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
import random
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
def evaluate(algorithm, env, device, batch_size=512, timers=None):
    """
    Evaluate classification accuracy — handles both env types.

    timers: optional dict accumulator. When given, adds elapsed time to
    timers['eval_loader'] (DataLoader construction) and
    timers['eval_forward'] (forward pass + accuracy bookkeeping).
    """
    algorithm.eval()

    if is_tensor_env(env):
        x = env['images'].to(device)
        y = env['labels'].to(device)
        n = len(x)
        correct = 0
        t0 = time.time()
        for i in range(0, n, batch_size):
            xb   = x[i:i + batch_size]
            yb   = y[i:i + batch_size]
            pred = algorithm.predict(xb).argmax(1)
            correct += (pred == yb).sum().item()
        if timers is not None:
            timers['eval_forward'] = timers.get('eval_forward', 0.0) + (time.time() - t0)
        algorithm.train()
        return correct / n
    else:
        t0 = time.time()
        loader  = DataLoader(env, batch_size=batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=True)
        if timers is not None:
            timers['eval_loader'] = timers.get('eval_loader', 0.0) + (time.time() - t0)
        correct = total = 0
        t0 = time.time()
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            pred  = algorithm.predict(x).argmax(1)
            correct += (pred == y).sum().item()
            total   += len(y)
        if timers is not None:
            timers['eval_forward'] = timers.get('eval_forward', 0.0) + (time.time() - t0)
        algorithm.train()
        return correct / total if total > 0 else 0.0


@torch.no_grad()
def predict_env(algorithm, env, device, batch_size=256, timers=None):
    """
    One forward pass over `env`, returning (preds, probs, labels) as numpy
    arrays in the same order. Lets a caller derive both accuracy and saved
    predictions from a single pass instead of running inference twice
    (once via evaluate(), once via save_model_outputs()) over the same data.

    timers: optional dict accumulator, same convention as evaluate() — adds
    to timers['infer_loader'] / timers['infer_forward'].
    """
    algorithm.eval()

    def _t(key, t0):
        if timers is not None:
            timers[key] = timers.get(key, 0.0) + (time.time() - t0)

    if is_tensor_env(env):
        t0     = time.time()
        x      = env['images'].to(device)
        logits = algorithm.predict(x)
        preds  = logits.argmax(1).cpu().numpy()
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        labels = env['labels'].numpy()
        _t('infer_forward', t0)
    else:
        t0 = time.time()
        loader = DataLoader(env, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
        _t('infer_loader', t0)
        t0 = time.time()
        preds_list, probs_list, labels_list = [], [], []
        for x, y in loader:
            x      = x.to(device)
            logits = algorithm.predict(x)
            preds_list.append(logits.argmax(1).cpu().numpy())
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels_list.append(y.numpy())
        preds  = np.concatenate(preds_list)
        probs  = np.concatenate(probs_list)
        labels = np.concatenate(labels_list)
        _t('infer_forward', t0)

    algorithm.train()
    return preds, probs, labels


# Save predictions, probs and features

@torch.no_grad()
def save_model_outputs(algorithm, all_envs, test_env_idx,
                       algorithm_name, hparams_seed, trial_seed,
                       save_dir, device, record, timers=None,
                       precomputed_out=None):
    """
    Save predictions, probabilities and test env features for all envs.
    Handles both tensor and Dataset envs.

    timers: optional dict accumulator. When given, adds elapsed time to
    timers['save_loader'] (DataLoader construction), timers['save_forward']
    (forward pass), and timers['save_io'] (np.save calls).

    precomputed_out: optional {env_idx: (preds, probs)}. When an entry is
    given for env i, reuses it instead of re-running inference — avoids a
    redundant forward pass when the caller (run_single, via predict_env)
    already computed preds/probs for that env's out split.
    """
    algorithm.eval()
    batch_size = 256
    precomputed_out = precomputed_out or {}

    def _t(key, t0):
        if timers is not None:
            timers[key] = timers.get(key, 0.0) + (time.time() - t0)

    for i, (in_env, out_env) in enumerate(all_envs):

        if i in precomputed_out:
            preds, probs = precomputed_out[i]
        elif is_tensor_env(out_env):
            t0     = time.time()
            x      = out_env['images'].to(device)
            logits = algorithm.predict(x)
            preds  = logits.argmax(1).cpu().numpy()
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            _t('save_forward', t0)
        else:
            t0 = time.time()
            loader = DataLoader(out_env, batch_size=batch_size,
                                shuffle=False, num_workers=4,
                                pin_memory=True)
            _t('save_loader', t0)
            t0 = time.time()
            preds_list = []
            probs_list = []
            for x, _ in loader:
                x      = x.to(device)
                logits = algorithm.predict(x)
                preds_list.append(logits.argmax(1).cpu().numpy())
                probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds = np.concatenate(preds_list)
            probs = np.concatenate(probs_list)
            _t('save_forward', t0)

        t0 = time.time()
        np.save(os.path.join(save_dir,
            f"{algorithm_name}_hpseed{hparams_seed}_trial{trial_seed}_env{i}_preds.npy"),
            preds)
        np.save(os.path.join(save_dir,
            f"{algorithm_name}_hpseed{hparams_seed}_trial{trial_seed}_env{i}_probs.npy"),
            probs)
        _t('save_io', t0)
        record[f'env{i}_pred_path'] = os.path.join(save_dir,
            f"{algorithm_name}_hpseed{hparams_seed}_trial{trial_seed}_env{i}_preds.npy")
        record[f'env{i}_prob_path'] = os.path.join(save_dir,
            f"{algorithm_name}_hpseed{hparams_seed}_trial{trial_seed}_env{i}_probs.npy")

    # Save features on test env out split only
    _, test_out_env = all_envs[test_env_idx]

    if is_tensor_env(test_out_env):
        t0       = time.time()
        x        = test_out_env['images'].to(device)
        features = algorithm.featurizer(x).cpu().numpy()
        _t('save_forward', t0)
    else:
        t0 = time.time()
        loader = DataLoader(test_out_env, batch_size=batch_size,
                            shuffle=False, num_workers=4,
                            pin_memory=True)
        _t('save_loader', t0)
        t0 = time.time()
        features_list = []
        for x, _ in loader:
            features_list.append(algorithm.featurizer(x.to(device)).cpu().numpy())
        features = np.concatenate(features_list)
        _t('save_forward', t0)

    t0 = time.time()
    fname = (f"{algorithm_name}_hpseed{hparams_seed}"
             f"_trial{trial_seed}_testenv{test_env_idx}_features.npy")
    np.save(os.path.join(save_dir, fname), features)
    record['feat_path'] = os.path.join(save_dir, fname)
    _t('save_io', t0)


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
    debug=False,
):
    timers = {} if debug else None

    # Reproducibility — torch.manual_seed also seeds all CUDA devices
    # (calls torch.cuda.manual_seed_all internally); random.seed covers
    # torchvision transforms that use Python's RNG rather than torch's.
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)
    random.seed(trial_seed)

    # Infer input shape
    t0 = time.time()
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
    if debug:
        timers['setup'] = time.time() - t0

    # Training loop (includes lazy DataLoader/worker construction on the
    # first next(loader) call per domain, since make_infinite_loader is a
    # generator — that cost is folded into train_time, not 'setup')
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

    # Evaluate on all environments. When we're about to save outputs anyway,
    # get out_env's preds/probs from ONE forward pass (predict_env) and
    # derive accuracy from them directly, instead of running inference
    # twice over the same data (evaluate() then save_model_outputs()).
    precomputed_out = {}
    for i, (in_env, out_env) in enumerate(all_envs):
        record[f'env{i}_in_acc'] = evaluate(algorithm, in_env, device, timers=timers)

        if save_dir is not None:
            preds, probs, labels = predict_env(algorithm, out_env, device, timers=timers)
            record[f'env{i}_out_acc'] = float((preds == labels).mean())
            precomputed_out[i] = (preds, probs)
        else:
            record[f'env{i}_out_acc'] = evaluate(algorithm, out_env, device, timers=timers)

    # Save outputs
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        algorithm.eval()
        with torch.no_grad():
            save_model_outputs(
                algorithm       = algorithm,
                all_envs        = all_envs,
                test_env_idx    = test_env_idx,
                precomputed_out = precomputed_out,
                algorithm_name = algorithm_class.__name__,
                hparams_seed   = hparams_seed,
                trial_seed     = trial_seed,
                save_dir       = save_dir,
                device         = device,
                record         = record,
                timers         = timers,
            )

    if debug:
        record['timers'] = dict(timers)
        print(
            f"    [debug] setup={timers.get('setup', 0.0):.2f}s "
            f"train={train_time:.2f}s "
            f"eval-in(loader={timers.get('eval_loader', 0.0):.2f}s "
            f"fwd={timers.get('eval_forward', 0.0):.2f}s) "
            f"infer-out(loader={timers.get('infer_loader', 0.0):.2f}s "
            f"fwd={timers.get('infer_forward', 0.0):.2f}s) "
            f"save(loader={timers.get('save_loader', 0.0):.2f}s "
            f"fwd={timers.get('save_forward', 0.0):.2f}s "
            f"io={timers.get('save_io', 0.0):.2f}s)",
            flush=True,
        )

    return record


# Full sweep

def run_sweep(
    algorithm_classes,
    dataset_name,
    test_env_idx,
    n_hparams,
    n_trials,
    device,
    envs_splits=None,
    envs_splits_per_trial=None,
    n_steps=5001,
    save_dir=None,
    search_method='random',
    backbone='resnet50',
    debug=False,
):
    """
    envs_splits: single fixed split, shared across all trials (default mode).
    envs_splits_per_trial: {trial_seed: envs_splits} — a DIFFERENT split per
      trial, shared across every algorithm/hparams_seed for that trial index
      (required for Cross-R/CrA's trial-matched agreement to stay valid).
    Pass exactly one of the two.
    """
    if (envs_splits is None) == (envs_splits_per_trial is None):
        raise ValueError(
            "run_sweep requires exactly one of envs_splits / "
            "envs_splits_per_trial (got both or neither)."
        )
    if search_method not in HP_SEARCH_METHODS:
        raise ValueError(
            f"Unknown search method '{search_method}'. "
            f"Available: {list(HP_SEARCH_METHODS.keys())}"
        )

    searcher  = HP_SEARCH_METHODS[search_method]
    n_classes = DATASET_CONFIGS[dataset_name]['n_classes']

    def _train_envs(splits):
        return [splits[i][0] for i in range(len(splits)) if i != test_env_idx]

    if envs_splits_per_trial is None:
        train_envs = _train_envs(envs_splits)

    # Save dataset metadata — labels for all tensor envs, plus the
    # ColoredMNIST-specific color subgroup when the tensor is image-shaped
    # (colors encode the spurious color channel; meaningless for tabular
    # feature vectors like ACSIncome, so skip it there). Per-trial mode
    # saves one metadata set PER TRIAL (env{i}_trial{t}_*), since the test
    # env's out split differs by trial; single mode keeps the original
    # unsuffixed filenames unchanged.
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        def _save_metadata(splits, suffix=''):
            _, sample_out = splits[0]
            if not is_tensor_env(sample_out):
                return
            is_image = sample_out['images'].dim() == 4
            for i, (in_env, out_env) in enumerate(splits):
                labels = out_env['labels'].numpy()
                np.save(os.path.join(save_dir, f'env{i}{suffix}_labels.npy'), labels)
                if is_image:
                    images = out_env['images']
                    colors = (images[:, 1, :, :].sum(dim=(1, 2)) > 0).numpy().astype(np.int32)
                    np.save(os.path.join(save_dir, f'env{i}{suffix}_colors.npy'), colors)

        if envs_splits_per_trial is not None:
            for t, splits in envs_splits_per_trial.items():
                _save_metadata(splits, suffix=f'_trial{t}')
        else:
            _save_metadata(envs_splits)
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
                if envs_splits_per_trial is not None:
                    trial_splits     = envs_splits_per_trial[trial_seed]
                    trial_train_envs = _train_envs(trial_splits)
                else:
                    trial_splits     = envs_splits
                    trial_train_envs = train_envs

                record = run_single(
                    algorithm_class = algorithm_class,
                    dataset_name    = dataset_name,
                    train_envs      = trial_train_envs,
                    all_envs        = trial_splits,
                    test_env_idx    = test_env_idx,
                    hparams_seed    = hparams_seed,
                    trial_seed      = trial_seed,
                    hp              = hp,
                    device          = device,
                    n_classes       = n_classes,
                    n_steps         = n_steps,
                    save_dir        = save_dir,
                    search_method   = search_method,
                    debug           = debug,
                )

                env_accs = " | ".join(
                    f"env{i} in={record[f'env{i}_in_acc']:.3f} "
                    f"out={record[f'env{i}_out_acc']:.3f}"
                    for i in range(len(trial_splits))
                )
                print(
                    f"  {env_accs} | "
                    f"time={record['train_time']:.1f}s",
                    flush=True,
                )

                records.append(record)

    return records