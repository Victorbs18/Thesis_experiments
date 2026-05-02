"""
reproduce_pacs_clip.py — PACS with CLIP ViT-B/32 backbone
==========================================================
Same protocol as reproduce_pacs.py (DomainBed standard) but
using CLIP ViT-B/32 as the backbone instead of ResNet50.

Key differences from ResNet50 version:
  - Backbone: CLIP ViT-B/32 pretrained on 400M image-text pairs
  - Finetuning: last transformer block + classification head only
    (frozen lower layers preserve pretrained representation)
  - Learning rate: smaller range [1e-6, 1e-4] to avoid destroying
    pretrained features
  - Input: CLIP preprocessing (normalize with CLIP mean/std)

Dataset, selection method, HP search protocol: identical to ResNet50.

Research question:
  Does CLIP ViT-B/32 + ERM already outperform ResNet50 + IRM?
  Does IRM on top of CLIP add value beyond CLIP + ERM?

Usage:
  python reproduce_pacs_clip.py --data_dir /path/to/pacs --device cuda
  python reproduce_pacs_clip.py --data_dir /path/to/pacs --n_hparams 5 --n_trials 1
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
import torchvision.models as models
import torchvision.transforms as transforms
try:
    import clip
except ImportError:
    clip = None
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image

# =============================================================================
# Dataset — PACS
# =============================================================================

PACS_DOMAINS     = ["art_painting", "cartoon", "photo", "sketch"]
PACS_DOMAIN_NAMES = ["A", "C", "P", "S"]
PACS_CLASSES     = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
N_CLASSES        = 7

# DomainBed ResNet50 reference (for comparison)
DOMAINBED_REF = {
    "ERM": {"A": 84.7, "C": 80.8, "P": 97.2, "S": 79.3, "Avg": 85.5},
    "IRM": {"A": 84.8, "C": 76.4, "P": 96.7, "S": 76.1, "Avg": 83.5},
}

# DomainBed ResNet50 oracle reference (for comparison)
DOMAINBED_ORACLE = {
    "ERM": {"A": 86.5, "C": 81.3, "P": 96.2, "S": 82.7, "Avg": 86.7},
    "IRM": {"A": 84.2, "C": 79.7, "P": 95.9, "S": 78.3, "Avg": 84.5},
}
# Note: CLIP numbers are novel — no published DomainBed baseline exists


class PACSDataset(Dataset):
    """
    PACS domain dataset — preloads all images into memory as PIL Images.
    This avoids repeated disk I/O and makes training much faster.
    PACS is small enough (~10k images) to fit in RAM easily.
    """

    def __init__(self, domain_dir, transform=None):
        self.transform = transform
        self.images    = []   # preloaded PIL images
        self.labels    = []

        print(f"    Preloading {os.path.basename(domain_dir)}...", end=" ", flush=True)
        for class_idx, class_name in enumerate(PACS_CLASSES):
            class_dir = os.path.join(domain_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img = Image.open(
                        os.path.join(class_dir, fname)
                    ).convert("RGB")
                    self.images.append(img)
                    self.labels.append(class_idx)
        print(f"{len(self.images)} images loaded")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_transforms(augment=True):
    """
    CLIP preprocessing — uses CLIP mean/std normalization.
    Augmentation strategy same as DomainBed for fair comparison.
    """
    # CLIP normalization values
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711]
    )
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


def load_pacs(data_dir):
    """
    Load all PACS domains.
    Returns list of 4 dicts with keys: dataset, name
    """
    domains = []
    for domain_name, domain_short in zip(PACS_DOMAINS, PACS_DOMAIN_NAMES):
        domain_dir = os.path.join(data_dir, domain_name)
        if not os.path.isdir(domain_dir):
            # Try alternative naming
            for alt in [domain_name.replace("_", ""), domain_short.lower(),
                        domain_name.split("_")[0]]:
                alt_dir = os.path.join(data_dir, alt)
                if os.path.isdir(alt_dir):
                    domain_dir = alt_dir
                    break
        if not os.path.isdir(domain_dir):
            raise FileNotFoundError(
                f"Domain directory not found: {domain_dir}\n"
                f"Expected structure: {data_dir}/art_painting/dog/image.jpg"
            )
        dataset = PACSDataset(domain_dir, transform=None)
        domains.append({"dataset": dataset, "name": domain_short, "dir": domain_dir})
        print(f"  Domain {domain_short} ({domain_name}): {len(dataset)} images")
    return domains


# =============================================================================
# Model — ResNet50 featurizer + linear classifier
# =============================================================================

class CLIPModel(nn.Module):
    """
    CLIP ViT-B/32 with last transformer block + head finetuned.
    Lower layers frozen to preserve pretrained representation.

    Finetuning strategy:
      - Frozen: patch embedding, positional embedding, blocks 0-10
      - Trainable: last transformer block (block 11) + ln_post + projection
      - Trainable: classification head (linear)

    This is standard practice for finetuning CLIP on downstream tasks.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        if clip is None:
            raise ImportError("Please install clip: pip install git+https://github.com/openai/CLIP.git")

        # Load CLIP ViT-B/32
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.visual    = clip_model.visual
        self.n_outputs = 512  # CLIP ViT-B/32 output dim

        # Freeze all layers first
        for param in self.visual.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block (block 11 of 12)
        for param in self.visual.transformer.resblocks[-1].parameters():
            param.requires_grad = True

        # Unfreeze final layer norm and projection
        for param in self.visual.ln_post.parameters():
            param.requires_grad = True
        if self.visual.proj is not None:
            self.visual.proj.requires_grad = True

        # Classification head — always trainable
        self.classifier = nn.Linear(self.n_outputs, num_classes)
        self.dropout    = nn.Dropout(0.0)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"    CLIP ViT-B/32: {n_trainable:,} / {n_total:,} params trainable "
              f"({100*n_trainable/n_total:.1f}%)")

    def forward(self, x):
        # CLIP visual encoder expects float32
        features = self.visual(x.float())
        features = self.dropout(features)
        return self.classifier(features)

    def set_dropout(self, p):
        self.dropout = nn.Dropout(p)


# =============================================================================
# HP sampling — exact DomainBed ranges for PACS (non-small images)
# =============================================================================

def sample_hparams(algorithm, rng):
    """
    Sample random HPs for CLIP finetuning.
    Key difference from ResNet50: smaller lr range to avoid
    destroying pretrained CLIP representations.
    """
    hp = {}
    hp["lr"]           = float(10 ** rng.uniform(-6, -4))   # 1e-6 to 1e-4 (smaller than ResNet)
    hp["weight_decay"] = float(10 ** rng.uniform(-6, -2))
    hp["batch_size"]   = int(2 ** rng.uniform(3, 5))        # 8-32
    hp["batch_size"]   = min(hp["batch_size"], 32)
    hp["resnet_dropout"] = float(rng.choice([0., 0.1, 0.5]))

    if algorithm == "IRM":
        hp["irm_lambda"]               = float(10 ** rng.uniform(-1, 5))
        hp["irm_penalty_anneal_iters"] = int(10 ** rng.uniform(0, 4))

    return hp


# =============================================================================
# IRM penalty — exact DomainBed implementation
# =============================================================================

def irm_penalty(logits, y):
    """Exact DomainBed IRM penalty — dot product of split-batch gradients."""
    device = logits.device
    scale  = torch.tensor(1.0, device=device, requires_grad=True)
    loss_1 = F.cross_entropy(logits[::2]  * scale, y[::2])
    loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
    g1     = autograd.grad(loss_1, [scale], create_graph=True)[0]
    g2     = autograd.grad(loss_2, [scale], create_graph=True)[0]
    return torch.sum(g1 * g2)


# =============================================================================
# Training
# =============================================================================

def _infinite_loader(dataset, batch_size, augment, device):
    """Infinite loader with augmentation. Images already in RAM so num_workers=2 is safe."""
    transform = get_transforms(augment=augment)
    underlying = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    underlying.transform = transform
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    while True:
        for x, y in loader:
            yield x.to(device), y.to(device)


def train_model(algorithm, train_envs, hp, device, seed, n_steps=5001):
    """Train ERM or IRM on PACS training environments."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CLIPModel(num_classes=N_CLASSES).to(device)
    model.set_dropout(hp.get("resnet_dropout", 0.0))

    opt = torch.optim.Adam(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp["weight_decay"]
    )

    loaders = [
        _infinite_loader(env["train"], hp["batch_size"], augment=True, device=device)
        for env in train_envs
    ]

    update_count = 0
    for step in range(n_steps):
        model.train()
        batches = [next(loader) for loader in loaders]

        if algorithm == "ERM":
            all_x = torch.cat([x for x, y in batches])
            all_y = torch.cat([y for x, y in batches])
            loss  = F.cross_entropy(model(all_x), all_y)

        elif algorithm == "IRM":
            penalty_weight = (
                hp["irm_lambda"]
                if update_count >= hp["irm_penalty_anneal_iters"]
                else 1.0
            )
            all_x      = torch.cat([x for x, y in batches])
            all_logits = model(all_x)
            idx        = 0
            nll        = 0.0
            penalty    = 0.0
            for x, y in batches:
                logits   = all_logits[idx:idx + len(x)]
                idx     += len(x)
                nll     += F.cross_entropy(logits, y)
                penalty += irm_penalty(logits, y)
            nll     /= len(batches)
            penalty /= len(batches)
            loss     = nll + penalty_weight * penalty

            if update_count == hp["irm_penalty_anneal_iters"]:
                opt = torch.optim.Adam(
                    model.parameters(),
                    lr=hp["lr"],
                    weight_decay=hp["weight_decay"]
                )

        opt.zero_grad()
        loss.backward()
        opt.step()
        update_count += 1

    model.eval()
    return model


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def accuracy(model, dataset, device, batch_size=64):
    """Evaluate accuracy on a dataset."""
    transform = get_transforms(augment=False)
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    correct = total = 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        pred    = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += len(y)
    return correct / total if total > 0 else 0.0


# =============================================================================
# Data splits — DomainBed protocol (80% train, 20% val per domain)
# =============================================================================

def split_domain(domain_dataset, holdout_frac=0.2, seed=0):
    """
    Split domain into train (80%) and val (20%).
    Returns (train_dataset, val_dataset) — both are Subset objects
    sharing the same underlying PACSDataset.
    """
    n     = len(domain_dataset)
    rng   = np.random.RandomState(seed)
    perm  = rng.permutation(n)
    n_val = int(n * holdout_frac)

    val_dataset   = Subset(domain_dataset, perm[:n_val])
    train_dataset = Subset(domain_dataset, perm[n_val:])

    # Wrap subsets so they support .transform attribute
    val_dataset.dataset   = domain_dataset
    train_dataset.dataset = domain_dataset

    return train_dataset, val_dataset


# =============================================================================
# Sweep
# =============================================================================

def run_sweep(algorithm, domains, test_env_idx, n_hparams, n_trials,
              device, n_steps=5001):
    """
    Run n_hparams HP configs × n_trials seeds for one test environment.
    Returns list of averaged results per HP config.
    """
    rng = np.random.RandomState(42 + test_env_idx)

    # Split all domains into train/val
    # DomainBed uses seed=0 for split
    splits = []
    for d in domains:
        train_ds, val_ds = split_domain(d["dataset"], holdout_frac=0.2, seed=0)
        splits.append({"train": train_ds, "val": val_ds,
                        "full": d["dataset"], "name": d["name"]})

    # Training envs = all domains except test
    train_env_idxs = [i for i in range(len(domains)) if i != test_env_idx]

    all_results = []

    for hp_idx in range(n_hparams):
        hp         = sample_hparams(algorithm, rng)
        hp_results = []

        for trial in range(n_trials):
            seed  = hp_idx * 100 + trial
            train_envs = [splits[i] for i in train_env_idxs]

            model = train_model(algorithm, train_envs, hp, device, seed, n_steps)

            # Evaluate on all domains × both splits
            env_in_accs  = []   # train portion accuracy
            env_out_accs = []   # val portion accuracy

            for i, split in enumerate(splits):
                in_acc  = _subset_accuracy(model, split["train"], device)
                out_acc = _subset_accuracy(model, split["val"],   device)
                env_in_accs.append(in_acc)
                env_out_accs.append(out_acc)

            hp_results.append({
                "env_in_accs":  env_in_accs,
                "env_out_accs": env_out_accs,
            })

            test_name = domains[test_env_idx]["name"]
            print(f"  hp={hp_idx:02d} trial={trial} "
                  + " | ".join(
                      f"{splits[i]['name']} in={env_in_accs[i]:.3f} out={env_out_accs[i]:.3f}"
                      for i in range(len(domains))
                  )
                  + f" | test_ood={env_out_accs[test_env_idx]:.3f}")

        # Average across trials
        avg = {
            "hp":           hp,
            "hp_idx":       hp_idx,
            "env_in_accs":  [np.mean([r["env_in_accs"][i]  for r in hp_results])
                              for i in range(len(domains))],
            "env_out_accs": [np.mean([r["env_out_accs"][i] for r in hp_results])
                              for i in range(len(domains))],
        }
        all_results.append(avg)

    return all_results


@torch.no_grad()
def _subset_accuracy(model, subset, device, batch_size=64):
    """Evaluate accuracy on a Subset or Dataset."""
    transform = get_transforms(augment=False)
    underlying = subset.dataset if hasattr(subset, 'dataset') else subset
    original_transform = underlying.transform
    underlying.transform = transform
    loader  = DataLoader(subset, batch_size=batch_size,
                         shuffle=False, num_workers=2)
    correct = total = 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        pred    = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += len(y)
    underlying.transform = original_transform
    return correct / total if total > 0 else 0.0


# =============================================================================
# Selection — train-domain val only (IID)
# =============================================================================

def iid_selection(results, test_env_idx, n_envs):
    """
    IIDAccuracySelectionMethod:
    Pick HP with highest mean out_acc across training environments.
    """
    best_val  = -1
    best_test = None
    for r in results:
        val_acc = np.mean([
            r["env_out_accs"][i]
            for i in range(n_envs)
            if i != test_env_idx
        ])
        if val_acc > best_val:
            best_val  = val_acc
            best_test = r["env_out_accs"][test_env_idx]
    return best_test, best_val


def oracle_selection(results, test_env_idx):
    """Oracle: pick HP with highest out_acc on test domain."""
    best_val  = -1
    best_test = None
    for r in results:
        val_acc = r["env_out_accs"][test_env_idx]
        if val_acc > best_val:
            best_val  = val_acc
            best_test = r["env_out_accs"][test_env_idx]
    return best_test, best_val


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Path to PACS dataset root")
    parser.add_argument("--n_hparams",  type=int, default=20)
    parser.add_argument("--n_trials",   type=int, default=3)
    parser.add_argument("--n_steps",    type=int, default=5001)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./results_pacs")
    parser.add_argument("--algorithms", type=str, default="ERM,IRM")
    parser.add_argument("--test_envs",  type=str, default="0,1,2,3",
                        help="Which test envs to run (0=A,1=C,2=P,3=S)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device     = args.device if torch.cuda.is_available() else "cpu"
    algorithms = args.algorithms.split(",")
    test_envs  = [int(x) for x in args.test_envs.split(",")]

    print(f"Loading PACS from {args.data_dir}...")
    domains = load_pacs(args.data_dir)
    print(f"  {len(domains)} domains loaded\n")

    all_algo_results = {}
    t0 = time.time()

    for algorithm in algorithms:
        print(f"\n{'='*65}")
        print(f"  {algorithm} — {args.n_hparams} HP configs × {args.n_trials} seeds")
        print(f"  Backbone: CLIP ViT-B/32 (last block + head finetuned)")
        print(f"{'='*65}")

        algo_env_results = {}

        for test_env_idx in test_envs:
            env_name = PACS_DOMAIN_NAMES[test_env_idx]
            train_names = [PACS_DOMAIN_NAMES[i]
                          for i in range(len(domains))
                          if i != test_env_idx]
            print(f"\n  --- Test env: {env_name} | Train: {train_names} ---")

            results = run_sweep(
                algorithm, domains, test_env_idx,
                args.n_hparams, args.n_trials,
                device, args.n_steps
            )

            iid_test,  iid_val  = iid_selection(results, test_env_idx, len(domains))
            ora_test,  ora_val  = oracle_selection(results, test_env_idx)

            ref_iid = DOMAINBED_REF.get(algorithm, {}).get(env_name, None)
            ref_ora = DOMAINBED_ORACLE.get(algorithm, {}).get(env_name, None)

            print(f"\n  Results — {algorithm} test env {env_name}:")
            print(f"  {'Method':<30} {'Test acc':>9}  {'Val acc':>9}  {'DomainBed ref':>14}")
            print(f"  {'─'*66}")
            ref_str = f"{ref_iid:.1f}%" if ref_iid else "—"
            print(f"  {'Train-domain val (IID)':<30} {iid_test:>9.1%}  {iid_val:>9.1%}  {ref_str:>14}")
            ref_str = f"{ref_ora:.1f}%" if ref_ora else "—"
            print(f"  {'Oracle (reference only)':<30} {ora_test:>9.1%}  {ora_val:>9.1%}  {ref_str:>14}")

            algo_env_results[env_name] = {
                "iid":    {"test": iid_test,  "val": iid_val},
                "oracle": {"test": ora_test,  "val": ora_val},
            }

        all_algo_results[algorithm] = algo_env_results

    # Final summary table — DomainBed format
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — DomainBed format")
    print(f"  (train-domain val selection)")
    print(f"{'='*70}")
    print(f"  {'Algorithm':<10} {'A':>8} {'C':>8} {'P':>8} {'S':>8} {'Avg':>8}")
    print(f"  {'─'*46}")

    for alg, res in all_algo_results.items():
        vals = [res.get(e, {}).get("iid", {}).get("test", 0) * 100
                for e in PACS_DOMAIN_NAMES]
        avg  = np.mean([v for v in vals if v > 0])
        print(f"  {alg:<10} " + " ".join(f"{v:>7.1f}%" for v in vals) + f" {avg:>7.1f}%")

    print(f"\n  ResNet50 DomainBed reference (train-domain val):")
    print(f"  ERM        84.7%    80.8%    97.2%    79.3%    85.5%")
    print(f"  IRM        84.8%    76.4%    96.7%    76.1%    83.5%")
    print(f"  (CLIP numbers are novel — no published DomainBed baseline)")

    print(f"\n  Oracle comparison:")
    print(f"  {'Algorithm':<10} {'A':>8} {'C':>8} {'P':>8} {'S':>8} {'Avg':>8}")
    print(f"  {'─'*46}")
    for alg, res in all_algo_results.items():
        vals = [res.get(e, {}).get("oracle", {}).get("test", 0) * 100
                for e in PACS_DOMAIN_NAMES]
        avg  = np.mean([v for v in vals if v > 0])
        print(f"  {alg:<10} " + " ".join(f"{v:>7.1f}%" for v in vals) + f" {avg:>7.1f}%")

    print(f"\n  DomainBed oracle reference:")
    print(f"  ERM        86.5%    81.3%    96.2%    82.7%    86.7%")
    print(f"  IRM        84.2%    79.7%    95.9%    78.3%    84.5%")

    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")

    # Save results
    json_path = os.path.join(args.output_dir, "pacs_clip_reproduction.json")
    with open(json_path, "w") as f:
        json.dump(all_algo_results, f, indent=2)
    print(f"  Results → {json_path}")


if __name__ == "__main__":
    main()