"""
reproduce_coloredmnist_clip.py — Backbone comparison on ColoredMNIST
=====================================================================
Runs ERM and IRM with ResNet50 and CLIP ViT-B/32 backbones on
DomainBed ColoredMNIST (test env = -90%, e=0.9).

Key preprocessing:
  - ColoredMNIST is 28×28 with 2-channel coloring
  - We upsample to 224×224 and replicate channel 0→RGB for ResNet50 / CLIP
  - Color is applied as a 2-channel mask; we reconstruct RGB by:
      R = digit_mask * red_color + background
      G = digit_mask * green_color + background
      B = digit_mask * blue_color + background
  - Standard DomainBed ColoredMNIST construction is reproduced exactly

Selection: train-domain val (IID) only — no oracle.
Reference: MNIST_CNN numbers from our reproduce.py run.

Usage:
  python reproduce_coloredmnist_clip.py --device cuda
  python reproduce_coloredmnist_clip.py --device cuda --n_hparams 5 --n_steps 2001
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("WARNING: CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")

# =============================================================================
# Reference numbers from our reproduce.py (MNIST_CNN, train-domain val)
# =============================================================================
REFERENCE = {
    "MNIST_CNN": {
        "ERM": {"iid_test": 9.9,  "domainbed_ref": 10.0},
        "IRM": {"iid_test": 9.9,  "oracle_test": 58.9, "domainbed_ref": 10.2},
    }
}

# =============================================================================
# ColoredMNIST construction — exact DomainBed protocol
# =============================================================================

def make_environment(images, labels, e, rng):
    """
    Construct one ColoredMNIST environment.
    e = probability that color = label (spurious correlation strength).
    Returns dict with 'images' (N,2,28,28) and 'labels' (N,).
    """
    # Binary label: digit >= 5
    labels = (labels >= 5).float()
    # Flip label with prob 0.25 (add noise to make invariant features imperfect)
    labels = torch.abs(labels - (torch.rand(len(labels)) < 0.25).float())
    # Assign color: match label with prob (1-e), flip with prob e
    colors = torch.abs(labels - (torch.rand(len(labels)) < e).float())
    # Build 2-channel image: channel 0 = digit (red), channel 1 = digit (green)
    images = images.float() / 255.0
    images = images.unsqueeze(1).expand(-1, 2, -1, -1).clone()
    images[:, 0, :, :] *= (1 - colors).unsqueeze(1).unsqueeze(1)
    images[:, 1, :, :] *= colors.unsqueeze(1).unsqueeze(1)
    return {"images": images, "labels": labels.long()}


def build_colored_mnist(data_dir="/tmp"):
    """Build all 3 ColoredMNIST environments."""
    mnist_train = MNIST(data_dir, train=True,  download=True)
    mnist_val   = MNIST(data_dir, train=False, download=True)

    rng = np.random.RandomState(42)

    # Split train into two environments
    n = len(mnist_train)
    perm = torch.randperm(n)
    envs = [
        make_environment(mnist_train.data[perm[:n//2]],
                         mnist_train.targets[perm[:n//2]], 0.1, rng),   # +90% (e=0.1)
        make_environment(mnist_train.data[perm[n//2:]],
                         mnist_train.targets[perm[n//2:]], 0.2, rng),   # +80% (e=0.2)
        make_environment(mnist_val.data,
                         mnist_val.targets,               0.9, rng),    # -90% (e=0.9) test
    ]
    print(f"  env +90% (e=0.1): {len(envs[0]['images'])} samples")
    print(f"  env +80% (e=0.2): {len(envs[1]['images'])} samples")
    print(f"  env -90% (e=0.9): {len(envs[2]['images'])} samples  ← test")
    return envs


# =============================================================================
# Preprocessing: 2-channel 28×28 → 3-channel 224×224
# =============================================================================

def cmnist_to_rgb_224(images):
    """
    Convert 2-channel 28×28 ColoredMNIST images to 3-channel 224×224.
    Channel 0 = red component, channel 1 = green component.
    We set blue channel = 0 to preserve the color meaning.

    Shape: (N,2,28,28) → (N,3,224,224)
    """
    N = images.shape[0]
    # Reconstruct RGB
    r = images[:, 0:1, :, :]   # red channel
    g = images[:, 1:2, :, :]   # green channel
    b = torch.zeros_like(r)     # blue = 0
    rgb = torch.cat([r, g, b], dim=1)  # (N,3,28,28)
    # Upsample to 224×224
    rgb_224 = F.interpolate(rgb, size=(224, 224), mode='bilinear', align_corners=False)
    return rgb_224


def preprocess_envs(envs):
    """Preprocess all environments to 3-channel 224×224 tensors."""
    print("Preprocessing: 2ch 28×28 → 3ch 224×224...")
    processed = []
    for i, env in enumerate(envs):
        imgs_224 = cmnist_to_rgb_224(env["images"])
        processed.append({"images": imgs_224, "labels": env["labels"]})
        print(f"  env {i}: {imgs_224.shape}")
    return processed


# =============================================================================
# Models
# =============================================================================

class ResNet50Model(nn.Module):
    """ResNet50 with full finetuning + classifier head."""
    def __init__(self, num_classes=2, dropout=0.0):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        n_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


class CLIPModel(nn.Module):
    """CLIP ViT-B/32 with last block + head finetuned."""
    def __init__(self, num_classes=2, dropout=0.0):
        super().__init__()
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.visual = clip_model.visual
        self.n_outputs = 512

        # Freeze all
        for param in self.visual.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block + ln_post + projection
        for param in self.visual.transformer.resblocks[-1].parameters():
            param.requires_grad = True
        for param in self.visual.ln_post.parameters():
            param.requires_grad = True
        if self.visual.proj is not None:
            self.visual.proj.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.n_outputs, num_classes)
        )

    def forward(self, x):
        # CLIP normalization (input already [0,1])
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                             device=x.device).view(1,3,1,1)
        x = (x - mean) / std
        features = self.visual(x.float())
        return self.classifier(features)


# =============================================================================
# IRM penalty — exact DomainBed
# =============================================================================

def irm_penalty(logits, y):
    device = logits.device
    scale  = torch.tensor(1.0, device=device, requires_grad=True)
    loss_1 = F.cross_entropy(logits[::2]  * scale, y[::2])
    loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
    g1     = autograd.grad(loss_1, [scale], create_graph=True)[0]
    g2     = autograd.grad(loss_2, [scale], create_graph=True)[0]
    return torch.sum(g1 * g2)


# =============================================================================
# HP sampling
# =============================================================================

def sample_hparams_resnet(algorithm, rng):
    hp = {}
    hp["lr"]           = float(10 ** rng.uniform(-4.5, -2.5))
    hp["weight_decay"] = float(10 ** rng.uniform(-6, -2))
    hp["batch_size"]   = int(rng.choice([32, 64]))
    hp["dropout"]      = float(rng.choice([0.0, 0.1, 0.5]))
    if algorithm == "IRM":
        hp["irm_lambda"]              = float(10 ** rng.uniform(1, 5))
        hp["irm_penalty_anneal_iters"] = int(10 ** rng.uniform(1, 3))
    return hp


def sample_hparams_clip(algorithm, rng):
    hp = {}
    hp["lr"]           = float(10 ** rng.uniform(-6, -4))   # smaller for CLIP
    hp["weight_decay"] = float(10 ** rng.uniform(-6, -2))
    hp["batch_size"]   = int(rng.choice([16, 32]))
    hp["dropout"]      = float(rng.choice([0.0, 0.1, 0.5]))
    if algorithm == "IRM":
        hp["irm_lambda"]              = float(10 ** rng.uniform(1, 5))
        hp["irm_penalty_anneal_iters"] = int(10 ** rng.uniform(1, 3))
    return hp


# =============================================================================
# Training
# =============================================================================

def make_loader(images, labels, batch_size, shuffle=True):
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=True, num_workers=0)


def train_model(backbone, algorithm, train_envs, hp, device, seed, n_steps):
    """Train one model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if backbone == "resnet50":
        model = ResNet50Model(num_classes=2,
                              dropout=hp.get("dropout", 0.0)).to(device)
    elif backbone == "clip":
        model = CLIPModel(num_classes=2,
                          dropout=hp.get("dropout", 0.0)).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"    {backbone}: {n_trainable:,} / {n_total:,} params trainable "
          f"({100*n_trainable/n_total:.1f}%)")

    opt = torch.optim.Adam(trainable,
                           lr=hp["lr"],
                           weight_decay=hp["weight_decay"])

    loaders = [make_loader(env["images"], env["labels"], hp["batch_size"])
               for env in train_envs]
    iters   = [iter(l) for l in loaders]

    def get_batch(loader_idx):
        nonlocal iters
        try:
            return next(iters[loader_idx])
        except StopIteration:
            iters[loader_idx] = iter(loaders[loader_idx])
            return next(iters[loader_idx])

    update_count = 0
    for step in range(n_steps):
        model.train()
        batches = [get_batch(i) for i in range(len(train_envs))]
        batches = [(x.to(device), y.to(device)) for x, y in batches]

        if algorithm == "ERM":
            all_x = torch.cat([x for x, y in batches])
            all_y = torch.cat([y for x, y in batches])
            loss  = F.cross_entropy(model(all_x), all_y)

        elif algorithm == "IRM":
            penalty_weight = (hp["irm_lambda"]
                              if update_count >= hp["irm_penalty_anneal_iters"]
                              else 1.0)
            all_x      = torch.cat([x for x, y in batches])
            all_logits = model(all_x)
            idx = nll = penalty = 0
            for x, y in batches:
                logits   = all_logits[idx:idx + len(x)]
                idx     += len(x)
                nll     += F.cross_entropy(logits, y)
                penalty += irm_penalty(logits, y)
            nll     /= len(batches)
            penalty /= len(batches)
            loss     = nll + penalty_weight * penalty

            if update_count == hp["irm_penalty_anneal_iters"]:
                opt = torch.optim.Adam(trainable,
                                       lr=hp["lr"],
                                       weight_decay=hp["weight_decay"])

        opt.zero_grad()
        loss.backward()
        opt.step()
        update_count += 1

    model.eval()
    return model


@torch.no_grad()
def evaluate(model, images, labels, device, batch_size=256):
    ds     = TensorDataset(images, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = total = 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        pred    = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += len(y)
    return correct / total if total > 0 else 0.0


# =============================================================================
# Sweep for one backbone+algorithm
# =============================================================================

def run_sweep(backbone, algorithm, train_envs, val_env, test_env,
              n_hparams, n_trials, n_steps, device, rng):
    """Run n_hparams HP configs × n_trials seeds."""

    sample_fn = sample_hparams_clip if backbone == "clip" else sample_hparams_resnet
    results   = []

    for hp_idx in range(n_hparams):
        hp = sample_fn(algorithm, rng)
        trial_id_vals  = []
        trial_ood_accs = []

        for trial in range(n_trials):
            seed  = hp_idx * 100 + trial
            model = train_model(backbone, algorithm, train_envs,
                                hp, device, seed, n_steps)

            # Evaluate on ID val (env0 val split) and OOD test (env2)
            id_val  = evaluate(model, val_env["images"],  val_env["labels"],  device)
            ood_acc = evaluate(model, test_env["images"], test_env["labels"], device)
            trial_id_vals.append(id_val)
            trial_ood_accs.append(ood_acc)

        mean_id_val  = float(np.mean(trial_id_vals))
        mean_ood_acc = float(np.mean(trial_ood_accs))
        print(f"  hp={hp_idx:02d} | id_val={mean_id_val:.3f} ood={mean_ood_acc:.3f} "
              f"  lr={hp['lr']:.1e}"
              + (f"  λ={hp.get('irm_lambda',0):.0f}" if algorithm == "IRM" else ""))

        results.append({
            "hp":             hp,
            "mean_id_val":    mean_id_val,
            "mean_ood_acc":   mean_ood_acc,
            "trial_id_vals":  trial_id_vals,
            "trial_ood_accs": trial_ood_accs,
        })

    # IID selection: highest ID val
    iid_best = max(results, key=lambda r: r["mean_id_val"])
    return results, iid_best


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, default="/tmp")
    parser.add_argument("--n_hparams", type=int, default=5)
    parser.add_argument("--n_trials",  type=int, default=1)
    parser.add_argument("--n_steps",   type=int, default=2001)
    parser.add_argument("--device",    type=str, default="cuda")
    parser.add_argument("--backbones", type=str, default="resnet50,clip",
                        help="comma-separated: resnet50,clip")
    parser.add_argument("--output_dir", type=str, default="./results_cmnist_clip")
    args = parser.parse_args()

    device   = args.device if torch.cuda.is_available() else "cpu"
    backbones = [b.strip() for b in args.backbones.split(",")]
    print(f"Device: {device}  Backbones: {backbones}")

    # Build ColoredMNIST
    print("\nBuilding ColoredMNIST...")
    envs = build_colored_mnist(args.data_dir)

    # Preprocess to 224×224 RGB
    envs_224 = preprocess_envs(envs)

    # Split env0 into train + id_val (80/20)
    n0   = len(envs_224[0]["images"])
    perm = torch.randperm(n0, generator=torch.Generator().manual_seed(0))
    n_val = int(n0 * 0.2)
    val_env = {
        "images": envs_224[0]["images"][perm[:n_val]],
        "labels": envs_224[0]["labels"][perm[:n_val]],
    }
    train_env0 = {
        "images": envs_224[0]["images"][perm[n_val:]],
        "labels": envs_224[0]["labels"][perm[n_val:]],
    }
    train_envs = [train_env0, envs_224[1]]  # env0 train + env1
    test_env   = envs_224[2]                # env2: -90%

    all_results = {}
    rng = np.random.RandomState(42)
    t0  = time.time()

    for backbone in backbones:
        if backbone == "clip" and not CLIP_AVAILABLE:
            print(f"\nSkipping CLIP — not installed.")
            continue

        all_results[backbone] = {}

        for algorithm in ["ERM", "IRM"]:
            print(f"\n{'='*60}")
            print(f"  {backbone.upper()} — {algorithm}")
            print(f"  {args.n_hparams} HP configs × {args.n_trials} seeds × {args.n_steps} steps")
            print(f"{'='*60}")

            results, iid_best = run_sweep(
                backbone, algorithm, train_envs, val_env, test_env,
                args.n_hparams, args.n_trials, args.n_steps, device, rng
            )

            iid_test  = iid_best["mean_ood_acc"]
            iid_val   = iid_best["mean_id_val"]

            print(f"\n  Results — {backbone.upper()} {algorithm}:")
            print(f"  {'Method':<30} {'Test acc':>10}  {'Val acc':>10}")
            print(f"  {'─'*55}")
            print(f"  {'IID val selection':<30} {iid_test:>10.1%}  {iid_val:>10.1%}")

            all_results[backbone][algorithm] = {
                "results": results,
                "iid": {"test": iid_test, "val": iid_val},
            }

    # Print comparison table
    total_time = (time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE — ColoredMNIST test env -90%  (IID val selection)")
    print(f"{'='*70}")
    print(f"  {'Backbone':<15} {'ERM':>10}  {'IRM':>10}  {'IRM oracle ref':>16}")
    print(f"  {'─'*55}")

    # Reference
    print(f"  {'MNIST_CNN (ref)':<15} {'10.0%':>10}  {'10.2%':>10}  {'58.5%':>16}")
    for backbone in backbones:
        if backbone not in all_results:
            continue
        erm_acc = all_results[backbone].get("ERM", {}).get("iid", {}).get("test", 0)
        irm_acc = all_results[backbone].get("IRM", {}).get("iid", {}).get("test", 0)
        print(f"  {backbone:<15} {erm_acc:>10.1%}  {irm_acc:>10.1%}  {'(run oracle separately)':>16}")

    print(f"\n  Total time: {total_time:.1f} min")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "experiment": "ColoredMNIST backbone comparison: ResNet50 + CLIP",
        "settings": {
            "n_hparams": args.n_hparams,
            "n_trials":  args.n_trials,
            "n_steps":   args.n_steps,
            "selection": "train-domain val (IID)",
            "test_env":  "-90% (e=0.9)",
            "preprocessing": "2ch 28x28 → RGB 224x224 (bilinear upsample, blue=0)",
        },
        "results": {
            backbone: {
                alg: {
                    "iid": v["iid"],
                    "configs": [
                        {k: r[k] for k in ["hp", "mean_id_val", "mean_ood_acc",
                                           "trial_id_vals", "trial_ood_accs"]}
                        for r in v["results"]
                    ]
                }
                for alg, v in alg_results.items()
            }
            for backbone, alg_results in all_results.items()
        },
        "reference_mnist_cnn": REFERENCE,
        "total_time_min": total_time,
    }
    out_path = os.path.join(args.output_dir, "coloredmnist_backbone_comparison.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  JSON → {out_path}")


if __name__ == "__main__":
    main()