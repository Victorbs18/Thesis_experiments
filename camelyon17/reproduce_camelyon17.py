"""
reproduce_camelyon17.py — ResNet50 + CLIP ViT-B/32 ERM/IRM baseline on Camelyon17
====================================================================================
Mirrors the structure of reproduce_pacs_clip.py exactly.

Camelyon17 (WILDS):
  Task:     Binary classification — tumor tissue in histopathology patches
  Domains:  5 hospitals (0-4), each is a distinct domain
  Images:   96×96 RGB patches
  Protocol: Train on hospitals {0,1,2,3}, test on hospital {2} — Env 2
             (Env 2 chosen because it has the highest negative ACL correlation
              in Salaudeen et al. 2025 — R=0.78, partially well-specified)

  Note on Env 2:
    The WILDS standard protocol tests on hospital 4 (id_test) and uses
    hospital 3 as OOD val. Here we follow the Salaudeen et al. convention
    where Env 2 = hospital index 2 as the OOD test environment.
    Train on the remaining hospitals {0,1,3,4}.

Why Env 2:
    From Salaudeen et al. (2025) Table 17, Camelyon17 Env 2 has:
      - Overall R=0.78 (misspecified by global measure)
      - But for high-accuracy models (>90%), negative correlation observed
      - This makes it the most interesting split — partially well-specified
    The hospital staining spurious correlation is genuine (different labs
    use different H&E staining protocols), making this a natural intervention
    in the sense of Salaudeen et al. Section 4.3.

Comparison table produced:
                        ERM      IRM     DomainBed ref (if available)
  ResNet50 (ID val)    ???%     ???%
  CLIP ViT-B/32 (ID val) ???%  ???%

Usage:
  # Quick check (5 configs, 1 seed, 2001 steps)
  python reproduce_camelyon17.py \\
    --data_dir /path/to/wilds \\
    --n_hparams 5 --n_trials 1 --n_steps 2001 \\
    --algorithms ERM IRM \\
    --backbones resnet50 clip \\
    --test_env 2 \\
    --device cuda \\
    --output_dir ./results_camelyon17

  # Download WILDS data first (requires wilds package):
  python -c "from wilds import get_dataset; get_dataset('camelyon17', download=True, root_dir='/path/to/wilds')"
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as tv_models

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")

# =============================================================================
# Constants
# =============================================================================

N_CLASSES    = 2   # tumor / non-tumor
N_HOSPITALS  = 5   # Camelyon17 hospitals 0-4
IMAGE_SIZE   = 96  # native Camelyon17 patch size

# DomainBed / WILDS reference numbers (from WILDS leaderboard, ResNet50 ERM)
# These are the standard WILDS protocol numbers (train 0,1,2 / test on 3,4)
# We note our split differs (test on Env 2) so direct comparison is approximate
WILDS_REF = {
    "ResNet50": {"ERM": 70.3, "IRM": 64.2},  # approximate, varies by run
}


# =============================================================================
# Data loading — WILDS Camelyon17
# =============================================================================

def get_camelyon17_splits(data_dir, test_env, seed=0):
    """
    Load Camelyon17 from WILDS and split into train/val/test environments.

    test_env: hospital index to hold out as OOD test (0-4)
    train envs: all hospitals except test_env
    val split: 20% holdout from each training hospital (ID val)

    Returns:
        train_subsets: list of (Subset, hospital_idx) for training
        id_val_subset: Subset for in-distribution validation (selection criterion)
        ood_test_subset: Subset for OOD test (final evaluation only)
    """
    try:
        from wilds import get_dataset
    except ImportError:
        raise ImportError(
            "Please install wilds: pip install wilds\n"
            "Then download: python -c \"from wilds import get_dataset; "
            "get_dataset('camelyon17', download=True, root_dir='.')\""
        )

    dataset = get_dataset(dataset='camelyon17', root_dir=data_dir, download=False)

    # WILDS provides metadata: hospital (0-4), slide, center
    # metadata_array column 0 = hospital index
    metadata = dataset.metadata_array  # (N, n_metadata_fields)
    hospitals = metadata[:, 0].numpy()

    train_hospitals = [h for h in range(N_HOSPITALS) if h != test_env]

    rng = np.random.RandomState(seed)

    train_subsets  = []
    id_val_indices = []

    for h in train_hospitals:
        h_indices = np.where(hospitals == h)[0]
        rng.shuffle(h_indices)
        n_val  = int(len(h_indices) * 0.2)
        n_train= len(h_indices) - n_val
        train_subsets.append(Subset(dataset, h_indices[:n_train]))
        id_val_indices.extend(h_indices[n_train:].tolist())

    id_val_subset  = Subset(dataset, id_val_indices)
    ood_test_subset= Subset(dataset, np.where(hospitals == test_env)[0].tolist())

    print(f"  Hospital splits (test_env={test_env}):")
    for h, sub in zip(train_hospitals, train_subsets):
        print(f"    Train hospital {h}: {len(sub)} patches")
    print(f"    ID val (all train hospitals): {len(id_val_subset)} patches")
    print(f"    OOD test hospital {test_env}: {len(ood_test_subset)} patches")

    return train_subsets, id_val_subset, ood_test_subset


def get_transform(backbone, augment=True):
    """Return appropriate transform for backbone."""
    if backbone == "clip":
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std= [0.26862954, 0.26130258, 0.27577711]
        )
        size = 224
    else:  # resnet50
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        )
        size = 224  # upsample from 96 to 224

    if augment:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize,
        ])


class WILDSSubsetWithTransform(torch.utils.data.Dataset):
    """Wrap a WILDS Subset to apply a transform to the image."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y, metadata = self.subset[idx]
        # x is a PIL Image from WILDS
        if self.transform:
            x = self.transform(x)
        return x, y.item()


def make_loader(subset, transform, batch_size, shuffle, num_workers=4):
    ds = WILDSSubsetWithTransform(subset, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=shuffle)


@torch.no_grad()
def eval_accuracy(model, subset, transform, device, batch_size=128):
    loader = make_loader(subset, transform, batch_size, shuffle=False, num_workers=2)
    correct = total = 0
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        pred   = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += len(y)
    return correct / total if total > 0 else 0.0


# =============================================================================
# Models
# =============================================================================

class ResNet50Model(nn.Module):
    """ResNet50 with ImageNet pretrained weights, last block + head finetuned."""
    def __init__(self, num_classes=2, dropout=0.0, full_finetune=False):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
        base    = tv_models.resnet50(weights=weights)

        # Freeze all by default
        for param in base.parameters():
            param.requires_grad = False

        if full_finetune:
            for param in base.parameters():
                param.requires_grad = True
        else:
            # Unfreeze layer4 (last residual block) + avgpool
            for param in base.layer4.parameters():
                param.requires_grad = True

        self.features  = nn.Sequential(*list(base.children())[:-1])  # up to avgpool
        self.n_outputs = 2048
        self.dropout   = nn.Dropout(dropout)
        self.classifier= nn.Linear(self.n_outputs, num_classes)

    def forward(self, x):
        feat = self.features(x).squeeze(-1).squeeze(-1)
        feat = self.dropout(feat)
        return self.classifier(feat)


class CLIPModel(nn.Module):
    """CLIP ViT-B/32 with last block + head finetuned."""
    def __init__(self, num_classes=2, dropout=0.0):
        super().__init__()
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.visual    = clip_model.visual
        self.n_outputs = 512

        for param in self.visual.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block + ln_post + projection
        for param in self.visual.transformer.resblocks[-1].parameters():
            param.requires_grad = True
        for param in self.visual.ln_post.parameters():
            param.requires_grad = True
        if self.visual.proj is not None:
            self.visual.proj.requires_grad = True

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.n_outputs, num_classes)

    def forward(self, x):
        features = self.visual(x.float())
        features = self.dropout(features)
        return self.classifier(features)


def build_model(backbone, device, dropout=0.0):
    if backbone == "resnet50":
        model = ResNet50Model(num_classes=N_CLASSES, dropout=dropout)
    elif backbone == "clip":
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP not installed")
        model = CLIPModel(num_classes=N_CLASSES, dropout=dropout)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {backbone}: {n_trainable:,} / {n_total:,} params trainable "
          f"({100*n_trainable/n_total:.1f}%)")
    return model.to(device)


# =============================================================================
# IRM penalty
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

def sample_hparams(algorithm, rng, backbone):
    """Sample HP config appropriate for the backbone."""
    hp = {}

    if backbone == "clip":
        hp["lr"]           = float(10 ** rng.uniform(-6, -4))   # [1e-6, 1e-4]
    else:
        hp["lr"]           = float(10 ** rng.uniform(-5, -3))   # [1e-5, 1e-3]

    hp["weight_decay"]     = float(10 ** rng.uniform(-6, -2))
    hp["batch_size"]       = int(rng.choice([32, 64]))
    hp["dropout"]          = float(rng.choice([0.0, 0.1, 0.5]))

    if algorithm == "IRM":
        hp["irm_lambda"]               = float(10 ** rng.uniform(-1, 5))
        hp["irm_penalty_anneal_iters"] = int(10 ** rng.uniform(0, 4))

    return hp


# =============================================================================
# Training
# =============================================================================

def train_one_config(algorithm, backbone, train_subsets, hp, device, seed, n_steps):
    """Train ERM or IRM for n_steps on Camelyon17 training hospitals."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(backbone, device, dropout=hp.get("dropout", 0.0))
    opt   = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=hp["lr"], weight_decay=hp["weight_decay"]
    )

    transform_train = get_transform(backbone, augment=True)

    # Infinite loaders — one per training hospital
    def make_infinite(subset):
        ds     = WILDSSubsetWithTransform(subset, transform_train)
        loader = DataLoader(ds, batch_size=hp["batch_size"],
                            shuffle=True, num_workers=2,
                            drop_last=True, pin_memory=True)
        while True:
            for batch in loader:
                yield batch

    loaders = [make_infinite(sub) for sub in train_subsets]

    update_count = 0
    for step in range(n_steps):
        model.train()
        batches = [next(loader) for loader in loaders]

        if algorithm == "ERM":
            all_x = torch.cat([x.to(device) for x, y in batches])
            all_y = torch.cat([y.to(device) for x, y in batches])
            loss  = F.cross_entropy(model(all_x), all_y)

        elif algorithm == "IRM":
            penalty_weight = (hp["irm_lambda"]
                              if update_count >= hp["irm_penalty_anneal_iters"]
                              else 1.0)
            all_x      = torch.cat([x.to(device) for x, y in batches])
            all_logits = model(all_x)
            idx = nll = penalty = 0
            for x, y in batches:
                y       = y.to(device)
                logits  = all_logits[idx:idx + len(x)]
                idx    += len(x)
                nll    += F.cross_entropy(logits, y)
                penalty+= irm_penalty(logits, y)
            nll     /= len(batches)
            penalty /= len(batches)
            loss     = nll + penalty_weight * penalty

            if update_count == hp["irm_penalty_anneal_iters"]:
                opt = torch.optim.Adam(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=hp["lr"], weight_decay=hp["weight_decay"]
                )

        opt.zero_grad()
        loss.backward()
        opt.step()
        update_count += 1

        if (step + 1) % 500 == 0:
            print(f"    step={step+1}/{n_steps}", flush=True)

    model.eval()
    return model


# =============================================================================
# Main experiment — one backbone × one algorithm
# =============================================================================

def run_backbone_algorithm(backbone, algorithm, train_subsets, id_val_subset,
                           ood_test_subset, n_hparams, n_trials, n_steps,
                           device, rng):
    """Run n_hparams HP configs × n_trials seeds, return results."""
    transform_eval = get_transform(backbone, augment=False)

    results  = []
    t0       = time.time()

    for hp_idx in range(n_hparams):
        hp = sample_hparams(algorithm, rng, backbone)

        trial_id_vals  = []
        trial_ood_accs = []

        for trial in range(n_trials):
            seed  = hp_idx * 100 + trial
            model = train_one_config(algorithm, backbone, train_subsets, hp,
                                     device, seed=seed, n_steps=n_steps)

            id_val_acc = eval_accuracy(model, id_val_subset,  transform_eval, device)
            ood_acc    = eval_accuracy(model, ood_test_subset, transform_eval, device)

            trial_id_vals.append(id_val_acc)
            trial_ood_accs.append(ood_acc)

            print(f"  hp={hp_idx:02d} trial={trial} "
                  f"id_val={id_val_acc:.3f} ood={ood_acc:.3f} "
                  f"| λ={hp.get('irm_lambda', 0):.0f}",
                  flush=True)

        results.append({
            "hp":             hp,
            "mean_id_val":    float(np.mean(trial_id_vals)),
            "mean_ood_acc":   float(np.mean(trial_ood_accs)),
            "best_ood_acc":   float(np.max(trial_ood_accs)),
            "trial_id_vals":  trial_id_vals,
            "trial_ood_accs": trial_ood_accs,
        })

    # Selection strategies
    iid_selected  = max(results, key=lambda r: r["mean_id_val"])
    oracle_selected = max(results, key=lambda r: r["mean_ood_acc"])

    elapsed = (time.time() - t0) / 60
    return {
        "results":         results,
        "iid":             {"test": iid_selected["mean_ood_acc"],
                            "val":  iid_selected["mean_id_val"]},
        "oracle":          {"test": oracle_selected["mean_ood_acc"]},
        "time_min":        elapsed,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str, required=True,
                        help="Path to WILDS data directory")
    parser.add_argument("--test_env",    type=int, default=2,
                        help="Hospital index to use as OOD test (0-4). Default=2")
    parser.add_argument("--n_hparams",   type=int, default=5)
    parser.add_argument("--n_trials",    type=int, default=1)
    parser.add_argument("--n_steps",     type=int, default=2001)
    parser.add_argument("--algorithms",  nargs="+", default=["ERM", "IRM"])
    parser.add_argument("--backbones",   nargs="+", default=["resnet50", "clip"])
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--output_dir",  type=str, default="./results_camelyon17")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(0)

    print(f"\nLoading Camelyon17 (test_env={args.test_env})...")
    train_subsets, id_val_subset, ood_test_subset = get_camelyon17_splits(
        args.data_dir, test_env=args.test_env
    )

    all_results = {}

    for backbone in args.backbones:
        all_results[backbone] = {}

        for algorithm in args.algorithms:
            print(f"\n{'='*65}")
            print(f"  {algorithm} — backbone: {backbone}")
            print(f"  {args.n_hparams} HP configs × {args.n_trials} seeds × "
                  f"{args.n_steps} steps")
            print(f"{'='*65}\n")

            result = run_backbone_algorithm(
                backbone, algorithm,
                train_subsets, id_val_subset, ood_test_subset,
                args.n_hparams, args.n_trials, args.n_steps,
                device, rng
            )
            all_results[backbone][algorithm] = result

            print(f"\n  Results — {backbone} {algorithm} (test hospital {args.test_env}):")
            print(f"  {'Method':<40} {'Test acc':>10}  {'Val acc':>10}")
            print(f"  {'─'*64}")
            print(f"  {'Train-domain val (IID)':<40} "
                  f"{result['iid']['test']:>10.1%}  {result['iid']['val']:>10.1%}")
            print(f"  {'Oracle (reference only)':<40} "
                  f"{result['oracle']['test']:>10.1%}  {'':>10}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — Camelyon17 (test hospital {args.test_env})")
    print(f"  (train-domain val selection, {args.n_hparams} configs × "
          f"{args.n_trials} seeds × {args.n_steps} steps)")
    print(f"{'='*70}")
    print(f"\n  {'Algorithm':<25} {'ERM':>10}  {'IRM':>10}")
    print(f"  {'─'*50}")

    for backbone in args.backbones:
        erm_acc = all_results[backbone].get("ERM", {}).get("iid", {}).get("test", None)
        irm_acc = all_results[backbone].get("IRM", {}).get("iid", {}).get("test", None)
        erm_str = f"{erm_acc:.1%}" if erm_acc is not None else "—"
        irm_str = f"{irm_acc:.1%}" if irm_acc is not None else "—"
        print(f"  {backbone:<25} {erm_str:>10}  {irm_str:>10}")

    print(f"\n  WILDS leaderboard reference (ResNet50, train-val selection, "
          f"standard WILDS split):")
    print(f"  {'ResNet50 ERM':<25} {WILDS_REF['ResNet50']['ERM']:>10.1f}%")
    print(f"  {'ResNet50 IRM':<25} {WILDS_REF['ResNet50']['IRM']:>10.1f}%")
    print(f"  (Note: WILDS standard protocol differs from our Env 2 split)")

    # Oracle comparison
    print(f"\n  Oracle comparison:")
    print(f"  {'Algorithm':<25} {'ERM':>10}  {'IRM':>10}")
    print(f"  {'─'*50}")
    for backbone in args.backbones:
        erm_oracle = all_results[backbone].get("ERM", {}).get("oracle", {}).get("test", None)
        irm_oracle = all_results[backbone].get("IRM", {}).get("oracle", {}).get("test", None)
        erm_str = f"{erm_oracle:.1%}" if erm_oracle is not None else "—"
        irm_str = f"{irm_oracle:.1%}" if irm_oracle is not None else "—"
        print(f"  {backbone:<25} {erm_str:>10}  {irm_str:>10}")

    # Save JSON
    total_time = sum(
        all_results[b][a].get("time_min", 0)
        for b in all_results
        for a in all_results[b]
    )

    output = {
        "experiment":   "Camelyon17 ResNet50+CLIP ERM/IRM baseline",
        "test_env":     args.test_env,
        "settings": {
            "n_hparams": args.n_hparams,
            "n_trials":  args.n_trials,
            "n_steps":   args.n_steps,
            "selection": "train-domain val (IID)",
            "note": (f"Test hospital {args.test_env}. "
                     f"Env 2 chosen per Salaudeen et al. (2025) — "
                     f"highest negative ACL correlation among Camelyon splits.")
        },
        "results":      all_results,
        "wilds_ref":    WILDS_REF,
        "total_time_min": total_time
    }

    json_path = os.path.join(args.output_dir, f"camelyon17_env{args.test_env}_baseline.json")
    with open(json_path, "w") as f:
        # Strip model objects from results before saving
        import copy
        clean = copy.deepcopy(output)
        json.dump(clean, f, indent=2)

    print(f"\n  Total time: {total_time:.1f} min")
    print(f"  Results → {json_path}")


if __name__ == "__main__":
    main()
