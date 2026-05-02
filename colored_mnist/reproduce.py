"""
reproduce.py — DomainBed ColoredMNIST Reproduction
====================================================
Reproduces ERM and IRM results from DomainBed on ColoredMNIST
using the exact same setup:

Dataset:
  - 3 environments: +90% (e=0.1), +80% (e=0.2), -90% (e=0.9)
  - Binary labels (digit < 5), 25% label noise
  - 2-channel images (red/green), shape (2, 28, 28)
  - Test env: -90% (e=0.9, index 2)

Model:
  - MNIST_CNN featurizer (DomainBed's exact architecture)
  - Linear classifier head
  - All layers trained (no freezing)

Selection methods (3):
  1. Training-domain validation set (IIDAccuracySelectionMethod)
     → held-out 20% of each training environment
  2. Leave-one-domain-out cross-validation (LeaveOneOutSelectionMethod)
     → use each training env as val for the other
  3. Oracle (OracleSelectionMethod)
     → test environment out-acc (cheating — reference only)

HP search:
  - n_hparams=20 random HP configs (DomainBed uses 20)
  - n_trials=3 random seeds per HP config
  - HP ranges from DomainBed hparams.py exactly

Expected results (DomainBed Table, train-domain val):
  ERM: ~71.7% (+90%), ~72.9% (+80%), ~10.0% (-90%)
  IRM: ~72.5% (+90%), ~73.3% (+80%), ~10.2% (-90%)

Usage:
  python reproduce.py --n_hparams 20 --n_trials 3 --device cuda
  python reproduce.py --n_hparams 5  --n_trials 1 --device cuda  # quick check
"""

import argparse
import json
import os
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision.datasets import MNIST

# =============================================================================
# Dataset — exact DomainBed ColoredMNIST
# =============================================================================

def make_colored_mnist(data_dir="./data"):
    """
    Build ColoredMNIST exactly as DomainBed does.
    Returns list of 3 envs: [+90% (e=0.1), +80% (e=0.2), -90% (e=0.9)]
    Each env is a dict with keys: images (N,2,28,28), labels (N,)
    """
    mnist_train = MNIST(data_dir, train=True,  download=True)
    mnist_test  = MNIST(data_dir, train=False, download=True)

    images = torch.cat([mnist_train.data, mnist_test.data]).float()
    labels = torch.cat([mnist_train.targets, mnist_test.targets])

    # Shuffle with fixed seed for reproducibility
    rng    = torch.Generator()
    rng.manual_seed(0)
    perm   = torch.randperm(len(images), generator=rng)
    images = images[perm]
    labels = labels[perm]

    environments = [0.1, 0.2, 0.9]
    envs = []
    for i, e in enumerate(environments):
        env_images = images[i::len(environments)]
        env_labels = labels[i::len(environments)]
        envs.append(_color_dataset(env_images, env_labels, e))

    return envs  # [env_+90%, env_+80%, env_-90%]


def _bernoulli(p, size):
    return (torch.rand(size) < p).float()

def _xor(a, b):
    return (a - b).abs()

def _color_dataset(images, labels, environment):
    """Exact DomainBed color_dataset function."""
    # Binary label
    labels = (labels < 5).float()
    # Flip label with 25% probability
    labels = _xor(labels, _bernoulli(0.25, len(labels)))
    # Assign color based on label, flip with probability e
    colors = _xor(labels, _bernoulli(environment, len(labels)))
    # Stack to 2 channels
    images = torch.stack([images, images], dim=1)
    # Zero out one channel based on color
    images[torch.arange(len(images)), (1 - colors).long(), :, :] *= 0
    x = images.float().div_(255.0)
    y = labels.view(-1).long()
    return {"images": x, "labels": y}


# =============================================================================
# Model — DomainBed MNIST_CNN + linear classifier
# =============================================================================

class MNIST_CNN(nn.Module):
    """Exact DomainBed MNIST_CNN featurizer."""
    n_outputs = 128

    def __init__(self, input_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.bn0   = nn.GroupNorm(8, 64)
        self.bn1   = nn.GroupNorm(8, 128)
        self.bn2   = nn.GroupNorm(8, 128)
        self.bn3   = nn.GroupNorm(8, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.bn0(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn3(self.conv4(x)))
        x = self.avgpool(x)
        return x.view(len(x), -1)


class Network(nn.Module):
    """Featurizer + linear classifier."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.featurizer  = MNIST_CNN(input_channels=2)
        self.classifier  = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.featurizer(x))


# =============================================================================
# HP sampling — exact DomainBed ranges for ColoredMNIST + IRM
# =============================================================================

def sample_hparams(algorithm, rng):
    """
    Sample random HPs from DomainBed ranges.
    ColoredMNIST is in SMALL_IMAGES so uses smaller lr range.
    """
    hp = {}
    # Shared
    hp["lr"]           = float(10 ** rng.uniform(-4.5, -2.5))
    hp["weight_decay"] = 0.0   # DomainBed sets 0 for small images
    hp["batch_size"]   = int(2 ** rng.uniform(3, 9))  # 8-512

    if algorithm == "IRM":
        hp["irm_lambda"]               = float(10 ** rng.uniform(-1, 5))
        hp["irm_penalty_anneal_iters"] = int(10 ** rng.uniform(0, 4))

    return hp


# =============================================================================
# Training — ERM and IRM
# =============================================================================

def train_erm(train_envs, hp, device, seed, n_steps=5001):
    torch.manual_seed(seed)
    model = Network().to(device)
    opt   = torch.optim.Adam(model.parameters(),
                              lr=hp["lr"], weight_decay=hp["weight_decay"])

    loaders = [_make_loader(env, hp["batch_size"], device)
               for env in train_envs]

    for step in range(n_steps):
        model.train()
        batches = [next(loader) for loader in loaders]
        all_x   = torch.cat([x for x, y in batches])
        all_y   = torch.cat([y for x, y in batches])
        loss    = F.cross_entropy(model(all_x), all_y)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    return model


def irm_penalty(logits, y):
    """Exact DomainBed IRM penalty — dot product of split-batch gradients."""
    device = logits.device
    scale  = torch.tensor(1.0, device=device, requires_grad=True)
    loss_1 = F.cross_entropy(logits[::2]  * scale, y[::2])
    loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
    g1     = autograd.grad(loss_1, [scale], create_graph=True)[0]
    g2     = autograd.grad(loss_2, [scale], create_graph=True)[0]
    return torch.sum(g1 * g2)


def train_irm(train_envs, hp, device, seed, n_steps=5001):
    torch.manual_seed(seed)
    model  = Network().to(device)
    opt    = torch.optim.Adam(model.parameters(),
                               lr=hp["lr"], weight_decay=hp["weight_decay"])

    loaders = [_make_loader(env, hp["batch_size"], device)
               for env in train_envs]

    update_count = 0
    for step in range(n_steps):
        model.train()
        penalty_weight = (hp["irm_lambda"]
                          if update_count >= hp["irm_penalty_anneal_iters"]
                          else 1.0)

        batches     = [next(loader) for loader in loaders]
        all_x       = torch.cat([x for x, y in batches])
        all_logits  = model(all_x)
        idx         = 0
        nll         = 0.0
        penalty     = 0.0
        for x, y in batches:
            logits   = all_logits[idx:idx + len(x)]
            idx     += len(x)
            nll     += F.cross_entropy(logits, y)
            penalty += irm_penalty(logits, y)
        nll     /= len(batches)
        penalty /= len(batches)
        loss     = nll + penalty_weight * penalty

        # Reset Adam at anneal step (exact DomainBed behavior)
        if update_count == hp["irm_penalty_anneal_iters"]:
            opt = torch.optim.Adam(model.parameters(),
                                    lr=hp["lr"],
                                    weight_decay=hp["weight_decay"])

        opt.zero_grad(); loss.backward(); opt.step()
        update_count += 1

    model.eval()
    return model


def _make_loader(env, batch_size, device):
    """Infinite loader over an environment."""
    x, y = env["images"].to(device), env["labels"].to(device)
    n    = len(x)
    while True:
        perm  = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 2:
                continue
            yield x[idx], y[idx]


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def accuracy(model, env, device):
    x   = env["images"].to(device)
    y   = env["labels"].to(device)
    out = model(x)
    return (out.argmax(1) == y).float().mean().item()


# =============================================================================
# Data splits — exact DomainBed protocol
# =============================================================================

def split_env(env, holdout_frac=0.2, seed=0):
    """
    Split an environment into train and val (out) subsets.
    DomainBed uses 20% holdout.
    Returns (train_env, val_env)
    """
    n     = len(env["images"])
    rng   = np.random.RandomState(seed)
    perm  = rng.permutation(n)
    n_val = int(n * holdout_frac)
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]
    return (
        {"images": env["images"][train_idx], "labels": env["labels"][train_idx]},
        {"images": env["images"][val_idx],   "labels": env["labels"][val_idx]},
    )


# =============================================================================
# Selection methods — exact DomainBed logic
# =============================================================================

def iid_selection(results, test_env_idx=2):
    """
    IIDAccuracySelectionMethod:
    Pick HP config with highest mean val_acc across training environments.
    val_acc = accuracy on held-out 20% of each training env (out_acc).
    """
    best_val  = -1
    best_test = None
    for r in results:
        val_acc = np.mean([r["env_out_accs"][i]
                           for i in range(len(r["env_out_accs"]))
                           if i != test_env_idx])
        if val_acc > best_val:
            best_val  = val_acc
            best_test = r["env_in_accs"][test_env_idx]
    return best_test, best_val


def leave_one_out_selection(results, test_env_idx=2):
    """
    LeaveOneOutSelectionMethod:
    For each non-test env, treat it as val using another env as train.
    Pick HP config with highest mean held-out val accuracy.
    DomainBed uses env_in_acc of the left-out env as val signal.
    """
    best_val  = -1
    best_test = None
    for r in results:
        # val = mean in_acc of training envs when used as left-out
        val_acc = np.mean([r["env_in_accs"][i]
                           for i in range(len(r["env_in_accs"]))
                           if i != test_env_idx])
        if val_acc > best_val:
            best_val  = val_acc
            best_test = r["env_in_accs"][test_env_idx]
    return best_test, best_val


def oracle_selection(results, test_env_idx=2):
    """
    OracleSelectionMethod:
    Pick HP config with highest out_acc on the test environment.
    This is the oracle / cheating selection.
    """
    best_val  = -1
    best_test = None
    for r in results:
        val_acc = r["env_out_accs"][test_env_idx]
        if val_acc > best_val:
            best_val  = val_acc
            best_test = r["env_in_accs"][test_env_idx]
    return best_test, best_val


# =============================================================================
# Main sweep
# =============================================================================

def run_sweep(algorithm, envs, n_hparams, n_trials, device,
              test_env_idx=2, n_steps=5001):
    """
    Run n_hparams HP configs × n_trials seeds.
    For each (hp, seed) record accuracy on all envs using both
    in_split (train portion) and out_split (val portion).
    """
    train_fn = train_erm if algorithm == "ERM" else train_irm
    rng      = np.random.RandomState(42)

    # Pre-split all environments into train/val
    # DomainBed uses seed=0 for the split
    splits = [split_env(env, holdout_frac=0.2, seed=0) for env in envs]
    # splits[i] = (train_env_i, val_env_i)

    all_results = []

    for hp_seed in range(n_hparams):
        hp = sample_hparams(algorithm, rng)
        hp_results = []

        for trial in range(n_trials):
            seed = hp_seed * 100 + trial

            # Training envs = train portions of non-test envs
            train_envs = [splits[i][0]
                          for i in range(len(envs))
                          if i != test_env_idx]

            model = train_fn(train_envs, hp, device, seed, n_steps=n_steps)

            # Record accuracy on ALL envs × both splits
            env_in_accs  = []  # accuracy on train portion of each env
            env_out_accs = []  # accuracy on val portion of each env

            for i, (train_env, val_env) in enumerate(splits):
                in_acc  = accuracy(model, train_env, device)
                out_acc = accuracy(model, val_env,   device)
                env_in_accs.append(in_acc)
                env_out_accs.append(out_acc)

            hp_results.append({
                "env_in_accs":  env_in_accs,
                "env_out_accs": env_out_accs,
            })

            print(f"  hp={hp_seed:02d} trial={trial} | "
                  + " | ".join(
                      f"env{i} in={env_in_accs[i]:.3f} out={env_out_accs[i]:.3f}"
                      for i in range(len(envs))
                  ))

        # Average across trials for this HP config
        avg_result = {
            "hp":            hp,
            "hp_seed":       hp_seed,
            "env_in_accs":   [np.mean([r["env_in_accs"][i]  for r in hp_results])
                               for i in range(len(envs))],
            "env_out_accs":  [np.mean([r["env_out_accs"][i] for r in hp_results])
                               for i in range(len(envs))],
        }
        all_results.append(avg_result)

    return all_results


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_hparams",  type=int, default=20,
                        help="Number of random HP configs (DomainBed uses 20)")
    parser.add_argument("--n_trials",   type=int, default=3,
                        help="Seeds per HP config (DomainBed uses 3)")
    parser.add_argument("--n_steps",    type=int, default=5001,
                        help="Training steps (DomainBed uses 5001)")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--algorithms", type=str, default="ERM,IRM",
                        help="Comma-separated list of algorithms")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device     = args.device if torch.cuda.is_available() else "cpu"
    algorithms = args.algorithms.split(",")

    print("Building ColoredMNIST...")
    envs = make_colored_mnist(args.data_dir)
    print(f"  env +90% (e=0.1): {len(envs[0]['images'])} samples")
    print(f"  env +80% (e=0.2): {len(envs[1]['images'])} samples")
    print(f"  env -90% (e=0.9): {len(envs[2]['images'])} samples  ← test")

    test_env_idx = 2  # -90% is the test environment

    all_algo_results = {}
    t0 = time.time()

    for algorithm in algorithms:
        print(f"\n{'='*60}")
        print(f"  {algorithm}  —  {args.n_hparams} HP configs × {args.n_trials} seeds")
        print(f"{'='*60}")

        results = run_sweep(
            algorithm, envs,
            args.n_hparams, args.n_trials,
            device, test_env_idx, args.n_steps
        )

        # Apply three selection methods
        iid_test,  iid_val  = iid_selection(results,          test_env_idx)
        loo_test,  loo_val  = leave_one_out_selection(results, test_env_idx)
        ora_test,  ora_val  = oracle_selection(results,        test_env_idx)

        print(f"\n  Results for {algorithm} — test env: -90% (e=0.9)")
        print(f"  {'Selection method':<35} {'Test acc':>9}  {'Val acc':>9}")
        print(f"  {'─'*57}")
        print(f"  {'Train-domain val (IID)':<35} {iid_test:>9.1%}  {iid_val:>9.1%}")
        print(f"  {'Leave-one-domain-out':<35} {loo_test:>9.1%}  {loo_val:>9.1%}")
        print(f"  {'Oracle (test labels — cheat)':<35} {ora_test:>9.1%}  {ora_val:>9.1%}")

        print(f"\n  DomainBed reference (train-domain val):")
        if algorithm == "ERM":
            print(f"    +90%: 71.7%  +80%: 72.9%  -90%: 10.0%")
        elif algorithm == "IRM":
            print(f"    +90%: 72.5%  +80%: 73.3%  -90%: 10.2%")

        all_algo_results[algorithm] = {
            "iid":    {"test": iid_test,  "val": iid_val},
            "loo":    {"test": loo_test,  "val": loo_val},
            "oracle": {"test": ora_test,  "val": ora_val},
            "all_results": [
                {k: v for k, v in r.items() if k != "hp"}
                for r in results
            ]
        }

    # Final comparison table
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"  (test env = -90%, all values = test accuracy)")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<10} {'Train-domain val':>18} {'Leave-one-out':>15} {'Oracle':>8}")
    print(f"  {'─'*55}")
    for alg, res in all_algo_results.items():
        print(f"  {alg:<10} {res['iid']['test']:>18.1%} "
              f"{res['loo']['test']:>15.1%} {res['oracle']['test']:>8.1%}")

    print(f"\n  DomainBed reference (-90% test env):")
    print(f"  ERM        train-domain val: 10.0%   oracle: 28.7%")
    print(f"  IRM        train-domain val: 10.2%   oracle: 58.5%")

    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")

    # Save JSON
    json_path = os.path.join(args.output_dir, "domainbed_reproduction.json")
    with open(json_path, "w") as f:
        json.dump(all_algo_results, f, indent=2)
    print(f"  Results → {json_path}")


if __name__ == "__main__":
    main()