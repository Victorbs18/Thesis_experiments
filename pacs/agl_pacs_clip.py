"""
agl_pacs_clip.py — Agreement-Guided HP Search on PACS with CLIP ViT-B/32
=========================================================================
Applies the AGL/ACL framework to PACS using CLIP ViT-B/32 as backbone.

Same framework logic as agl_exp5_extension.py:
  Phase 1 — Coarse random search (ERM + IRM, n_coarse_trials configs)
  Phase 2 — Detection (AGL label-free + ACL labeled)
  Phase 3 — Targeted fine search in promising HP region
  Phase 4 — TTA boost on selected model

Key differences from ColoredMNIST version:
  - Backbone: CLIP ViT-B/32 (last block + head finetuned)
  - Data: PACS 224x224 RGB, 7 classes, 4 domains
  - Protocol: leave-one-domain-out (train 3, test 1)
  - OOD val split: held-out 20% of test domain (for labeled mode)
  - No oracle column (no separate OOD val in ColoredMNIST sense)

Results table format:
  Domain  DomainBed ERM  DomainBed IRM  CLIP ERM  CLIP IRM  New LF pre  New LF post
  A       84.7%         84.8%          91.7%     92.9%     ???         ???
  C       80.8%         76.4%          97.0%     94.9%     ???         ???
  P       97.2%         96.7%         100.0%     99.4%     ???         ???
  S       79.3%         76.1%          88.4%     90.2%     ???         ???

Usage:
  python agl_pacs_clip.py --data_dir /path/to/pacs --device cuda
  python agl_pacs_clip.py --data_dir /path/to/pacs --n_coarse_trials 5 --n_seeds 1 --test_envs 3
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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from scipy import stats
from PIL import Image

try:
    import clip
except ImportError:
    raise ImportError("Please install clip: pip install git+https://github.com/openai/CLIP.git")

# =============================================================================
# Constants
# =============================================================================

PACS_DOMAINS      = ["art_painting", "cartoon", "photo", "sketch"]
PACS_DOMAIN_NAMES = ["A", "C", "P", "S"]
PACS_CLASSES      = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
N_CLASSES         = 7

# DomainBed reference numbers (ResNet50, train-domain val)
DOMAINBED_REF = {
    "ERM": {"A": 84.7, "C": 80.8, "P": 97.2, "S": 79.3},
    "IRM": {"A": 84.8, "C": 76.4, "P": 96.7, "S": 76.1},
}

# Our CLIP baseline (from reproduce_pacs_clip.py, 5 configs x 1 seed x 2001 steps)
CLIP_BASELINE = {
    "ERM": {"A": 91.7, "C": 97.0, "P": 100.0, "S": 88.4},
    "IRM": {"A": 92.9, "C": 94.9, "P":  99.4, "S": 90.2},
}

# Framework constants
BETA                   = 0.5    # instability penalty in scoring
TOP_K                  = 3      # top coarse configs to identify HP region
DIVERGE_FRAC_THRESHOLD = 0.20   # fraction of IRM configs that must diverge
ACL_DEVIATION_THRESHOLD = 0.03  # IRM above ERM ACL line by this much = DIVERGE
TTA_LR                 = 1e-5   # smaller lr for CLIP TTA
TTA_STEPS              = 10     # TTA adaptation steps


# =============================================================================
# Dataset
# =============================================================================

class PACSDataset(Dataset):
    """PACS domain — preloads all images into RAM as PIL Images."""

    def __init__(self, domain_dir, transform=None):
        self.transform = transform
        self.images    = []
        self.labels    = []

        for class_idx, class_name in enumerate(PACS_CLASSES):
            class_dir = os.path.join(domain_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img = Image.open(os.path.join(class_dir, fname)).convert("RGB")
                    self.images.append(img)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_transform(augment=True):
    """CLIP preprocessing with optional augmentation."""
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
    """Load all PACS domains with preloading."""
    domains = []
    for domain_name, domain_short in zip(PACS_DOMAINS, PACS_DOMAIN_NAMES):
        domain_dir = os.path.join(data_dir, domain_name)
        if not os.path.isdir(domain_dir):
            for alt in [domain_name.replace("_", ""), domain_short.lower(),
                        domain_name.split("_")[0]]:
                alt_dir = os.path.join(data_dir, alt)
                if os.path.isdir(alt_dir):
                    domain_dir = alt_dir
                    break
        if not os.path.isdir(domain_dir):
            raise FileNotFoundError(f"Domain not found: {domain_dir}")
        print(f"  Preloading {domain_short} ({domain_name})...", end=" ", flush=True)
        ds = PACSDataset(domain_dir)
        print(f"{len(ds)} images")
        domains.append({"dataset": ds, "name": domain_short})
    return domains


def split_domain(dataset, holdout_frac=0.2, seed=0):
    """Split domain into train (80%) and val (20%) subsets."""
    n    = len(dataset)
    rng  = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(n * holdout_frac)
    return Subset(dataset, perm[n_val:]), Subset(dataset, perm[:n_val])


def make_loader(subset, batch_size, augment, device):
    """Infinite data loader."""
    underlying = subset.dataset if hasattr(subset, 'dataset') else subset
    underlying.transform = get_transform(augment=augment)
    loader = DataLoader(subset, batch_size=batch_size,
                        shuffle=True, num_workers=2,
                        drop_last=True, pin_memory=True)
    while True:
        for x, y in loader:
            yield x.to(device), y.to(device)


@torch.no_grad()
def eval_accuracy(model, subset, device, batch_size=64):
    """Evaluate accuracy on a Subset."""
    underlying = subset.dataset if hasattr(subset, 'dataset') else subset
    orig = underlying.transform
    underlying.transform = get_transform(augment=False)
    loader  = DataLoader(subset, batch_size=batch_size,
                         shuffle=False, num_workers=2)
    correct = total = 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        pred    = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += len(y)
    underlying.transform = orig
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def get_predictions(model, subset, device, batch_size=64):
    """Get softmax predictions on a Subset — for agreement computation."""
    underlying = subset.dataset if hasattr(subset, 'dataset') else subset
    orig = underlying.transform
    underlying.transform = get_transform(augment=False)
    loader = DataLoader(subset, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    preds = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.append(F.softmax(logits, dim=1).cpu())
    underlying.transform = orig
    return torch.cat(preds, dim=0)


# =============================================================================
# Model — CLIP ViT-B/32 last block + head
# =============================================================================

class CLIPModel(nn.Module):
    def __init__(self, num_classes=7, dropout=0.0):
        super().__init__()
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.visual    = clip_model.visual
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

        self.classifier = nn.Linear(self.n_outputs, num_classes)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        features = self.visual(x.float())
        features = self.dropout(features)
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

def sample_hparams(algorithm, rng, targeted=False,
                   lam_min=None, lam_max=None, lr_min=None, lr_max=None):
    """Sample HP config for CLIP finetuning."""
    hp = {}
    if targeted and lr_min and lr_max:
        hp["lr"] = float(10 ** rng.uniform(
            np.log10(max(lr_min, 1e-7)),
            np.log10(min(lr_max, 1e-3))))
    else:
        hp["lr"] = float(10 ** rng.uniform(-6, -4))  # [1e-6, 1e-4] for CLIP

    hp["weight_decay"] = float(10 ** rng.uniform(-6, -2))
    hp["batch_size"]   = int(rng.choice([16, 32]))
    hp["dropout"]      = float(rng.choice([0.0, 0.1, 0.5]))

    if algorithm == "IRM":
        if targeted and lam_min and lam_max:
            hp["irm_lambda"] = float(10 ** rng.uniform(
                np.log10(max(lam_min, 1e-1)),
                np.log10(min(lam_max, 1e5))))
        else:
            hp["irm_lambda"] = float(10 ** rng.uniform(-1, 5))
        hp["irm_penalty_anneal_iters"] = int(10 ** rng.uniform(0, 4))

    return hp


# =============================================================================
# Training
# =============================================================================

def train_model(algorithm, train_envs, hp, device, seed, n_steps=5001):
    """Train ERM or IRM on PACS training environments."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CLIPModel(num_classes=N_CLASSES, dropout=hp.get("dropout", 0.0)).to(device)
    opt   = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=hp["lr"], weight_decay=hp["weight_decay"]
    )

    loaders = [make_loader(env["train"], hp["batch_size"],
                           augment=True, device=device)
               for env in train_envs]

    update_count = 0
    for step in range(n_steps):
        model.train()
        batches = [next(loader) for loader in loaders]

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
                opt = torch.optim.Adam(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=hp["lr"], weight_decay=hp["weight_decay"]
                )

        opt.zero_grad()
        loss.backward()
        opt.step()
        update_count += 1

    model.eval()
    return model


# =============================================================================
# Agreement computation
# =============================================================================

def compute_agreement(preds_a, preds_b):
    """Fraction of examples where two models predict the same class."""
    return (preds_a.argmax(1) == preds_b.argmax(1)).float().mean().item()


def compute_entropy_from_preds(preds):
    """Mean prediction entropy."""
    eps = 1e-8
    return -(preds * torch.log(preds + eps)).sum(1).mean().item()


def fit_line(x, y):
    """Fit linear regression, return slope, intercept, R²."""
    if len(x) < 3:
        return 1.0, 0.0, 0.0
    s, i, r, _, _ = stats.linregress(x, y)
    return s, i, r**2


# =============================================================================
# TTA
# =============================================================================

def apply_tent(model, ref_subset, device, steps=TTA_STEPS, lr=TTA_LR):
    """TENT: minimize prediction entropy on unlabeled OOD data."""
    adapted = copy.deepcopy(model)
    # Only adapt trainable parameters
    params  = [p for p in adapted.parameters() if p.requires_grad]
    opt     = torch.optim.Adam(params, lr=lr)

    underlying = ref_subset.dataset if hasattr(ref_subset, 'dataset') else ref_subset
    orig = underlying.transform
    underlying.transform = get_transform(augment=False)
    loader = DataLoader(ref_subset, batch_size=32, shuffle=True, num_workers=0)

    for _ in range(steps):
        adapted.train()
        for x, _ in loader:
            x      = x.to(device)
            logits = adapted(x)
            probs  = F.softmax(logits, dim=1)
            eps    = 1e-8
            entropy = -(probs * torch.log(probs + eps)).sum(1).mean()
            opt.zero_grad()
            entropy.backward()
            opt.step()
            break  # one batch per step

    adapted.eval()
    underlying.transform = orig
    return adapted


# =============================================================================
# Main framework — one test environment
# =============================================================================

def run_test_env(test_env_idx, domains, n_coarse_trials, n_seeds,
                 fine_trials, n_steps, device, output_dir):

    env_name = PACS_DOMAIN_NAMES[test_env_idx]
    train_env_idxs = [i for i in range(len(domains)) if i != test_env_idx]
    train_names    = [PACS_DOMAIN_NAMES[i] for i in train_env_idxs]

    print(f"\n{'='*65}")
    print(f"  Test env: {env_name} | Train: {train_names}")
    print(f"  n_coarse={n_coarse_trials}  n_seeds={n_seeds}  "
          f"fine_trials={fine_trials}  n_steps={n_steps}")
    print(f"{'='*65}")

    t0  = time.time()
    rng = np.random.RandomState(42 + test_env_idx)

    # Split all domains
    splits = []
    for d in domains:
        train_ds, val_ds = split_domain(d["dataset"], holdout_frac=0.2, seed=0)
        splits.append({"train": train_ds, "val": val_ds,
                        "full": d["dataset"], "name": d["name"]})

    train_envs = [splits[i] for i in train_env_idxs]

    # Reference sets for agreement computation
    # id_ref  = val portion of first training domain (labeled)
    # ood_ref = val portion of test domain (for labeled mode + TTA validation)
    # ood_ref_images = ood_ref without labels (for label-free agreement)
    id_ref   = splits[train_env_idxs[0]]["val"]   # ID reference
    ood_ref  = splits[test_env_idx]["val"]         # OOD reference (val split — NOT full test)

    # Split ood_ref into adapt (for TTA) and validate (for labeled scoring)
    n_ood      = len(ood_ref)
    ood_perm   = np.random.RandomState(0).permutation(n_ood)
    ood_adapt  = Subset(ood_ref.dataset, ood_ref.indices[ood_perm[:n_ood//2]])
    ood_val    = Subset(ood_ref.dataset, ood_ref.indices[ood_perm[n_ood//2:]])

    # Full test domain for final evaluation only
    test_full  = splits[test_env_idx]["full"]

    # =========================================================================
    # Phase 1 — Coarse random search
    # =========================================================================
    print(f"\n[Phase 1] Coarse search ({n_coarse_trials} trials × {n_seeds} seeds)...")

    erm_configs = []
    irm_configs = []

    for trial in range(n_coarse_trials):
        # --- ERM ---
        erm_hp      = sample_hparams("ERM", rng)
        erm_results = []
        for seed in range(n_seeds):
            model    = train_model("ERM", train_envs, erm_hp, device,
                                   seed=trial*100+seed, n_steps=n_steps)
            id_val   = eval_accuracy(model, splits[train_env_idxs[0]]["val"], device)
            ood_acc  = eval_accuracy(model, ood_val, device)
            id_preds = get_predictions(model, id_ref, device)
            ood_preds= get_predictions(model, ood_adapt, device)
            erm_results.append({
                "id_val_acc": id_val, "ood_acc": ood_acc,
                "id_preds": id_preds, "ood_preds": ood_preds
            })

        erm_cfg = {
            "hp":             erm_hp,
            "mean_id_val_acc": np.mean([r["id_val_acc"] for r in erm_results]),
            "mean_ood_acc":    np.mean([r["ood_acc"]    for r in erm_results]),
            "std_ood_acc":     np.std( [r["ood_acc"]    for r in erm_results]),
            "mean_id_preds":   torch.stack([r["id_preds"]  for r in erm_results]).mean(0),
            "mean_ood_preds":  torch.stack([r["ood_preds"] for r in erm_results]).mean(0),
        }
        erm_configs.append(erm_cfg)

        # --- IRM ---
        irm_hp      = sample_hparams("IRM", rng)
        irm_results = []
        for seed in range(n_seeds):
            model    = train_model("IRM", train_envs, irm_hp, device,
                                   seed=trial*100+seed+50, n_steps=n_steps)
            id_val   = eval_accuracy(model, splits[train_env_idxs[0]]["val"], device)
            ood_acc  = eval_accuracy(model, ood_val, device)
            id_preds = get_predictions(model, id_ref, device)
            ood_preds= get_predictions(model, ood_adapt, device)
            irm_results.append({
                "id_val_acc": id_val, "ood_acc": ood_acc,
                "id_preds": id_preds, "ood_preds": ood_preds
            })

        irm_cfg = {
            "hp":             irm_hp,
            "mean_id_val_acc": np.mean([r["id_val_acc"] for r in irm_results]),
            "mean_ood_acc":    np.mean([r["ood_acc"]    for r in irm_results]),
            "std_ood_acc":     np.std( [r["ood_acc"]    for r in irm_results]),
            "mean_id_preds":   torch.stack([r["id_preds"]  for r in irm_results]).mean(0),
            "mean_ood_preds":  torch.stack([r["ood_preds"] for r in irm_results]).mean(0),
        }
        irm_configs.append(irm_cfg)

        print(f"  trial={trial:02d} | ERM val={erm_cfg['mean_id_val_acc']:.3f} "
              f"ood={erm_cfg['mean_ood_acc']:.3f} | "
              f"IRM val={irm_cfg['mean_id_val_acc']:.3f} "
              f"ood={irm_cfg['mean_ood_acc']:.3f} "
              f"λ={irm_hp.get('irm_lambda', 0):.0f}")

    # =========================================================================
    # Phase 2 — Detection
    # =========================================================================
    print(f"\n[Phase 2] Detection...")

    # Fit AGL line (ERM-ERM agreement pairs)
    id_agrs  = []
    ood_agrs = []
    for i, a in enumerate(erm_configs):
        for j, b in enumerate(erm_configs):
            if i >= j:
                continue
            id_agrs.append(compute_agreement(a["mean_id_preds"], b["mean_id_preds"]))
            ood_agrs.append(compute_agreement(a["mean_ood_preds"], b["mean_ood_preds"]))

    agl_slope, agl_intercept, agl_r2 = fit_line(np.array(id_agrs), np.array(ood_agrs))
    print(f"  AGL line: slope={agl_slope:.3f}  intercept={agl_intercept:.3f}  R²={agl_r2:.3f}")

    # Fit ACL line (ERM ID val vs OOD acc)
    acl_x = np.array([c["mean_id_val_acc"] for c in erm_configs])
    acl_y = np.array([c["mean_ood_acc"]    for c in erm_configs])
    acl_slope, acl_intercept, acl_r2 = fit_line(acl_x, acl_y)
    print(f"  ACL line: slope={acl_slope:.3f}  intercept={acl_intercept:.3f}  R²={acl_r2:.3f}")

    # ERM entropy (label-free struggling signal)
    erm_best      = max(erm_configs, key=lambda c: c["mean_id_val_acc"])
    id_entropy    = compute_entropy_from_preds(erm_best["mean_id_preds"])
    ood_entropy   = compute_entropy_from_preds(erm_best["mean_ood_preds"])
    entropy_delta = ood_entropy - id_entropy
    erm_struggling= entropy_delta > 0
    print(f"  ERM entropy: ID={id_entropy:.3f}  OOD={ood_entropy:.3f}  "
          f"ΔH={entropy_delta:+.3f}  struggling={erm_struggling}")

    # IRM divergence from AGL line
    n_div_lf = 0
    for irm_cfg in irm_configs:
        for erm_cfg in erm_configs:
            id_agr  = compute_agreement(irm_cfg["mean_id_preds"],  erm_cfg["mean_id_preds"])
            ood_agr = compute_agreement(irm_cfg["mean_ood_preds"], erm_cfg["mean_ood_preds"])
            expected = agl_slope * id_agr + agl_intercept
            if ood_agr < expected - 0.02:
                n_div_lf += 1
                break

    frac_div_lf = n_div_lf / len(irm_configs)
    irm_diverging = frac_div_lf >= DIVERGE_FRAC_THRESHOLD
    print(f"  IRM diverging: {n_div_lf}/{len(irm_configs)} configs "
          f"(frac={frac_div_lf:.2f}  threshold={DIVERGE_FRAC_THRESHOLD})")

    # IRM quality check — best diverging IRM vs best ERM
    div_configs = []
    for irm_cfg in irm_configs:
        for erm_cfg in erm_configs:
            id_agr  = compute_agreement(irm_cfg["mean_id_preds"],  erm_cfg["mean_id_preds"])
            ood_agr = compute_agreement(irm_cfg["mean_ood_preds"], erm_cfg["mean_ood_preds"])
            expected = agl_slope * id_agr + agl_intercept
            if ood_agr < expected - 0.02:
                div_configs.append(irm_cfg)
                break

    best_erm_val = max(c["mean_id_val_acc"] for c in erm_configs)
    best_irm_val = (max(c["mean_id_val_acc"] for c in div_configs)
                    if div_configs else 0.0)

    # Label-free decision
    if not erm_struggling:
        decision_lf = "AGREE_ERM_WORKS"
    elif not irm_diverging:
        decision_lf = "AGREE_NO_DIVERGE"
    elif best_irm_val < best_erm_val:
        decision_lf = "DIVERGE"
    else:
        decision_lf = "AGREE_IRM_FAILS"

    # Labeled decision (ACL-based)
    n_div_lb = sum(
        1 for c in irm_configs
        if c["mean_ood_acc"] > acl_slope * c["mean_id_val_acc"] + acl_intercept + ACL_DEVIATION_THRESHOLD
    )
    if n_div_lb / len(irm_configs) >= DIVERGE_FRAC_THRESHOLD:
        decision_labeled = "DIVERGE"
    elif any(c["mean_ood_acc"] < acl_slope * c["mean_id_val_acc"] + acl_intercept - ACL_DEVIATION_THRESHOLD
             for c in irm_configs):
        decision_labeled = "AGREE_IRM_FAILS"
    else:
        decision_labeled = "AGREE_ERM_WORKS"

    print(f"  Labeled decision:    {decision_labeled} ({n_div_lb}/{len(irm_configs)} IRM above ACL)")
    print(f"  Label-free decision: {decision_lf} "
          f"(erm_struggling={erm_struggling} ΔH={entropy_delta:+.3f}, "
          f"irm_diverging={irm_diverging})")

    # =========================================================================
    # Phase 3 — Targeted fine search
    # =========================================================================
    print(f"\n[Phase 3] Targeted fine search ({fine_trials} trials)...")

    def get_hp_region(configs, lam_key="irm_lambda", lr_key="lr"):
        top = sorted(configs, key=lambda c: c["mean_id_val_acc"], reverse=True)[:TOP_K]
        lams = [c["hp"].get(lam_key, 1.0) for c in top]
        lrs  = [c["hp"][lr_key] for c in top]
        return min(lams)/5, max(lams)*5, min(lrs)/3, max(lrs)*3

    # Label-free fine search
    if decision_lf == "DIVERGE":
        lam_min, lam_max, lr_min, lr_max = get_hp_region(div_configs or irm_configs)
        print(f"  [LF DIVERGE] λ=[{lam_min:.1f}, {lam_max:.1f}]  "
              f"lr=[{lr_min:.2e}, {lr_max:.2e}]")
        best_lf_score = -np.inf
        best_lf_model = None
        best_lf_acc   = 0.0

        for ft in range(fine_trials):
            hp    = sample_hparams("IRM", rng, targeted=True,
                                   lam_min=lam_min, lam_max=lam_max,
                                   lr_min=lr_min, lr_max=lr_max)
            model = train_model("IRM", train_envs, hp, device,
                                seed=1000+ft, n_steps=n_steps)
            id_val  = eval_accuracy(model, splits[train_env_idxs[0]]["val"], device)
            ood_acc = eval_accuracy(model, ood_val, device)

            # Label-free score: AGL deviation
            id_preds  = get_predictions(model, id_ref, device)
            ood_preds = get_predictions(model, ood_adapt, device)
            deviations = []
            for erm_cfg in erm_configs:
                id_agr  = compute_agreement(id_preds,  erm_cfg["mean_id_preds"])
                ood_agr = compute_agreement(ood_preds, erm_cfg["mean_ood_preds"])
                expected = agl_slope * id_agr + agl_intercept
                deviations.append(ood_agr - expected)
            score = -np.mean(deviations) - BETA * np.std(deviations)

            print(f"  [LF fine={ft:02d}] val={id_val:.3f} ood={ood_acc:.3f} "
                  f"score={score:+.3f} λ={hp['irm_lambda']:.0f}")

            if score > best_lf_score:
                best_lf_score = score
                best_lf_model = model
                best_lf_acc   = ood_acc

    elif decision_lf in ("AGREE_ERM_WORKS", "AGREE_IRM_FAILS"):
        lr_min, lr_max = (min(c["hp"]["lr"] for c in erm_configs[:TOP_K]),
                          max(c["hp"]["lr"] for c in erm_configs[:TOP_K]))
        print(f"  [LF ERM] lr=[{lr_min:.2e}, {lr_max:.2e}]")
        best_lf_score = -np.inf
        best_lf_model = None
        best_lf_acc   = 0.0

        for ft in range(fine_trials):
            hp    = sample_hparams("ERM", rng, targeted=True,
                                   lr_min=lr_min, lr_max=lr_max)
            model = train_model("ERM", train_envs, hp, device,
                                seed=1000+ft, n_steps=n_steps)
            id_val  = eval_accuracy(model, splits[train_env_idxs[0]]["val"], device)
            ood_acc = eval_accuracy(model, ood_val, device)

            # Label-free score: within-ERM OOD agreement
            ood_preds = get_predictions(model, ood_adapt, device)
            agrs = [compute_agreement(ood_preds, c["mean_ood_preds"])
                    for c in erm_configs]
            score = np.mean(agrs) - BETA * np.std(agrs)

            print(f"  [LF fine={ft:02d}] val={id_val:.3f} ood={ood_acc:.3f} "
                  f"score={score:+.3f}")

            if score > best_lf_score:
                best_lf_score = score
                best_lf_model = model
                best_lf_acc   = ood_acc

    else:  # AGREE_NO_DIVERGE
        print(f"  [LF AGREE_NO_DIVERGE] No fine search — using best coarse model")
        best_coarse   = max(erm_configs, key=lambda c: c["mean_id_val_acc"])
        best_lf_model = train_model("ERM", train_envs, best_coarse["hp"],
                                    device, seed=999, n_steps=n_steps)
        best_lf_acc   = eval_accuracy(best_lf_model, ood_val, device)

    # Labeled fine search (uses ood_val labels for scoring)
    if decision_labeled == "DIVERGE":
        top_irm = sorted(irm_configs,
                          key=lambda c: c["mean_ood_acc"] - (acl_slope * c["mean_id_val_acc"] + acl_intercept),
                          reverse=True)[:TOP_K]
        lam_min, lam_max, lr_min, lr_max = get_hp_region(top_irm)
        print(f"  [Lab DIVERGE] λ=[{lam_min:.1f}, {lam_max:.1f}]")
        best_lab_score = -np.inf
        best_lab_model = None
        best_lab_acc   = 0.0

        for ft in range(fine_trials):
            hp    = sample_hparams("IRM", rng, targeted=True,
                                   lam_min=lam_min, lam_max=lam_max,
                                   lr_min=lr_min, lr_max=lr_max)
            model = train_model("IRM", train_envs, hp, device,
                                seed=2000+ft, n_steps=n_steps)
            id_val  = eval_accuracy(model, splits[train_env_idxs[0]]["val"], device)
            ood_acc = eval_accuracy(model, ood_val, device)
            expected = acl_slope * id_val + acl_intercept
            score    = (ood_acc - expected) - BETA * 0.0

            print(f"  [Lab fine={ft:02d}] val={id_val:.3f} ood={ood_acc:.3f} "
                  f"score={score:+.3f} λ={hp['irm_lambda']:.0f}")

            if score > best_lab_score:
                best_lab_score = score
                best_lab_model = model
                best_lab_acc   = ood_acc

    else:
        top_erm = sorted(erm_configs, key=lambda c: c["mean_ood_acc"], reverse=True)[:TOP_K]
        lr_min  = min(c["hp"]["lr"] for c in top_erm)
        lr_max  = max(c["hp"]["lr"] for c in top_erm)
        print(f"  [Lab ERM] lr=[{lr_min:.2e}, {lr_max:.2e}]")
        best_lab_score = -np.inf
        best_lab_model = None
        best_lab_acc   = 0.0

        for ft in range(fine_trials):
            hp    = sample_hparams("ERM", rng, targeted=True,
                                   lr_min=lr_min, lr_max=lr_max)
            model = train_model("ERM", train_envs, hp, device,
                                seed=2000+ft, n_steps=n_steps)
            id_val  = eval_accuracy(model, splits[train_env_idxs[0]]["val"], device)
            ood_acc = eval_accuracy(model, ood_val, device)
            score   = ood_acc - BETA * 0.0
            print(f"  [Lab fine={ft:02d}] val={id_val:.3f} ood={ood_acc:.3f} "
                  f"score={score:+.3f}")
            if score > best_lab_score:
                best_lab_score = score
                best_lab_model = model
                best_lab_acc   = ood_acc

    # =========================================================================
    # Phase 4 — TTA
    # =========================================================================
    print(f"\n[Phase 4] TTA boost...")

    pre_lf  = eval_accuracy(best_lf_model,  test_full, device)
    pre_lab = eval_accuracy(best_lab_model, test_full, device)

    # TTA adapts on ood_adapt, validates on ood_val
    adapted_lf  = apply_tent(best_lf_model,  ood_adapt, device)
    adapted_lab = apply_tent(best_lab_model, ood_adapt, device)

    post_lf_ood_val  = eval_accuracy(adapted_lf,  ood_val, device)
    post_lab_ood_val = eval_accuracy(adapted_lab, ood_val, device)
    pre_lf_ood_val   = eval_accuracy(best_lf_model,  ood_val, device)
    pre_lab_ood_val  = eval_accuracy(best_lab_model, ood_val, device)

    keep_lf  = post_lf_ood_val  >= pre_lf_ood_val
    keep_lab = post_lab_ood_val >= pre_lab_ood_val

    final_lf  = eval_accuracy(adapted_lf  if keep_lf  else best_lf_model,  test_full, device)
    final_lab = eval_accuracy(adapted_lab if keep_lab else best_lab_model, test_full, device)

    print(f"  LF:  pre_test={pre_lf:.3f}  post_test={final_lf:.3f}  "
          f"TTA_kept={keep_lf}")
    print(f"  Lab: pre_test={pre_lab:.3f}  post_test={final_lab:.3f}  "
          f"TTA_kept={keep_lab}")

    # =========================================================================
    # Results
    # =========================================================================
    std_erm_acc = max(c["mean_ood_acc"] for c in erm_configs)
    std_irm_acc = max(c["mean_ood_acc"] for c in irm_configs)
    clip_erm_ref = CLIP_BASELINE["ERM"].get(env_name, 0) / 100
    clip_irm_ref = CLIP_BASELINE["IRM"].get(env_name, 0) / 100

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS — Test env {env_name}")
    print(f"{'='*65}")
    rows = [
        ("DomainBed ERM (ResNet50, ref)",   DOMAINBED_REF["ERM"].get(env_name,0)/100, None),
        ("DomainBed IRM (ResNet50, ref)",   DOMAINBED_REF["IRM"].get(env_name,0)/100, None),
        ("CLIP ERM baseline",               clip_erm_ref,  None),
        ("CLIP IRM baseline",               clip_irm_ref,  None),
        ("Coarse best ERM (id val)",        std_erm_acc,   std_erm_acc - clip_erm_ref),
        ("Coarse best IRM (id val)",        std_irm_acc,   std_irm_acc - clip_erm_ref),
        ("New Labeled★  pre-TTA  (LB1)",    pre_lab,       pre_lab  - clip_erm_ref),
        ("New Labeled★  post-TTA (LB2)",    final_lab,     final_lab - clip_erm_ref),
        ("New LabelFree★ pre-TTA  (LB1)",   pre_lf,        pre_lf   - clip_erm_ref),
        ("New LabelFree★ post-TTA (LB2)",   final_lf,      final_lf  - clip_erm_ref),
    ]
    print(f"  {'Strategy':<40} {'OOD acc':>9}  {'vs CLIP ERM':>12}")
    print(f"  {'─'*65}")
    for name, acc, gap in rows:
        gap_str = f"{gap:+.3f}" if gap is not None else "    —   "
        print(f"  {name:<40} {acc:>9.3f}  {gap_str:>12}")

    result = {
        "env": env_name,
        "detection": {"labeled": decision_labeled, "label_free": decision_lf},
        "agl": {"slope": agl_slope, "intercept": agl_intercept, "r2": agl_r2},
        "acl": {"slope": acl_slope, "intercept": acl_intercept, "r2": acl_r2},
        "entropy": {"id": id_entropy, "ood": ood_entropy, "delta": entropy_delta},
        "divergence": {"n_div_lf": n_div_lf, "frac_div_lf": frac_div_lf,
                       "n_div_lb": n_div_lb},
        "results": {
            "domainbed_erm":  DOMAINBED_REF["ERM"].get(env_name,0)/100,
            "domainbed_irm":  DOMAINBED_REF["IRM"].get(env_name,0)/100,
            "clip_erm_base":  clip_erm_ref,
            "clip_irm_base":  clip_irm_ref,
            "coarse_erm":     std_erm_acc,
            "coarse_irm":     std_irm_acc,
            "labeled_pre":    pre_lab,
            "labeled_post":   final_lab,
            "lf_pre":         pre_lf,
            "lf_post":        final_lf,
        },
        "time_min": (time.time() - t0) / 60
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"agl_pacs_clip_{env_name}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Time: {result['time_min']:.1f} min")
    print(f"  JSON → {json_path}")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        type=str, required=True)
    parser.add_argument("--n_coarse_trials", type=int, default=20)
    parser.add_argument("--n_seeds",         type=int, default=3)
    parser.add_argument("--fine_trials",     type=int, default=10)
    parser.add_argument("--n_steps",         type=int, default=5001)
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--output_dir",      type=str, default="./results_agl_pacs")
    parser.add_argument("--test_envs",       type=str, default="0,1,2,3",
                        help="0=A, 1=C, 2=P, 3=S")
    args = parser.parse_args()

    device    = args.device if torch.cuda.is_available() else "cpu"
    test_envs = [int(x) for x in args.test_envs.split(",")]

    print("Loading PACS...")
    domains = load_pacs(args.data_dir)

    all_results = {}
    for test_env_idx in test_envs:
        result = run_test_env(
            test_env_idx, domains,
            args.n_coarse_trials, args.n_seeds,
            args.fine_trials, args.n_steps,
            device, args.output_dir
        )
        all_results[PACS_DOMAIN_NAMES[test_env_idx]] = result

    # Final summary
    print(f"\n{'='*75}")
    print(f"  FINAL SUMMARY — AGL Framework on PACS CLIP ViT-B/32")
    print(f"{'='*75}")
    print(f"  {'Domain':<8} {'DB ERM':>8} {'DB IRM':>8} {'CLIP ERM':>10} "
          f"{'LF pre':>8} {'LF post':>9} {'Decision'}")
    print(f"  {'─'*75}")
    for env_name, res in all_results.items():
        r = res["results"]
        print(f"  {env_name:<8} {r['domainbed_erm']:>8.1%} {r['domainbed_irm']:>8.1%} "
              f"{r['clip_erm_base']:>10.1%} {r['lf_pre']:>8.1%} "
              f"{r['lf_post']:>9.1%}  {res['detection']['label_free']}")

    # Save combined results
    combined_path = os.path.join(args.output_dir, "agl_pacs_clip_all.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Combined results → {combined_path}")


if __name__ == "__main__":
    main()