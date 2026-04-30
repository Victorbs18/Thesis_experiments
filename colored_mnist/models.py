"""
models.py — MLP architecture and training for ERM and IRM
==========================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# Architecture
# =============================================================================

class MLP(nn.Module):
    """
    3-layer MLP matching the architecture from the IRM paper reproduction.
    Input: flattened 3-channel 28x28 image = 2352 dims
    Output: single logit for binary classification
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * 28 * 28, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# IRM penalty
# =============================================================================

def irm_penalty(logits, labels):
    """
    IRMv1 penalty: squared norm of gradient of a fixed scalar classifier.
    Forces the optimal classifier to be invariant across environments.
    """
    scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
    loss  = nn.BCEWithLogitsLoss()(logits * scale, labels)
    grad  = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


# =============================================================================
# Training
# =============================================================================

def train_model(envs, hp, method, device, seed, max_steps=None):
    """
    Train a model using ERM or IRM on the given environments.

    hp dict keys:
      hidden_dim, lr, l2_reg, steps,
      penalty_weight (IRM only), penalty_anneal_iters (IRM only)

    method: "erm" or "irm"
    """
    torch.manual_seed(seed)
    model     = MLP(hp["hidden_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=hp["lr"], weight_decay=hp["l2_reg"])
    loss_fn   = nn.BCEWithLogitsLoss()
    pw        = hp.get("penalty_weight", 0.0)
    pa        = hp.get("penalty_anneal_iters", 0)
    steps = min(hp["steps"], max_steps) if max_steps else hp["steps"]
    for step in range(steps):
        model.train()
        total   = torch.tensor(0.0, device=device)
        penalty = torch.tensor(0.0, device=device)

        for env in envs:
            x, y   = env["images"].to(device), env["labels"].to(device)
            logits  = model(x)
            total  += loss_fn(logits, y)
            if method == "irm":
                penalty += irm_penalty(logits, y)

        total   /= len(envs)
        penalty /= len(envs)

        if method == "irm" and pw > 0:
            w    = pw if step >= pa else 1.0
            loss = (total + w * penalty) / (w if w > 1 else 1)
        else:
            loss = total

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def get_predictions(model, data, device):
    """Binary predictions (0 or 1) for all examples in data."""
    model.eval()
    return (model(data["images"].to(device)) > 0).float().cpu()


def compute_accuracy(model, data, device):
    """Fraction of correct predictions."""
    preds  = get_predictions(model, data, device)
    labels = data["labels"].cpu()
    return (preds == labels).float().mean().item()


def worst_env_val_acc(model, val_envs, device):
    """
    Worst-case accuracy across validation environments.
    This is the non-oracle selection criterion from your practical work —
    no OOD labels used, just held-out training domain data.
    """
    return min(compute_accuracy(model, v, device) for v in val_envs)


# =============================================================================
# HP ranges — same as your original search.py
# =============================================================================

HP_RANGES = {
    "hidden_dim":           [32, 64, 80, 85, 88, 92, 95, 99, 100, 102, 103,
                             114, 117, 119, 138, 176, 178, 215, 237, 253,
                             254, 256, 258, 301, 309, 319, 340, 341, 423,
                             437, 486],
    "lr":                   [1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3],
    "l2_reg":               [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    "steps":                [101, 201, 301, 401, 501],
    "penalty_anneal_iters": [50, 80, 100, 150, 200, 250],
    "penalty_weight":       [10, 100, 500, 1000, 5000, 10000,
                             50000, 100000, 500000, 700000],
}


def sample_erm_hp(rng):
    return {
        "hidden_dim": int(rng.choice(HP_RANGES["hidden_dim"])),
        "lr":         float(rng.choice(HP_RANGES["lr"])),
        "l2_reg":     float(rng.choice(HP_RANGES["l2_reg"])),
        "steps":      int(rng.choice(HP_RANGES["steps"])),
    }


def sample_irm_hp(rng):
    hp = sample_erm_hp(rng)
    hp["penalty_weight"]       = float(rng.choice(HP_RANGES["penalty_weight"]))
    hp["penalty_anneal_iters"] = int(rng.choice(HP_RANGES["penalty_anneal_iters"]))
    return hp
