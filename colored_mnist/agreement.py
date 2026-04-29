"""
agreement.py — Agreement, entropy, and AGL line fitting
========================================================
All computations are label-free on the test side.
"""

import itertools
import numpy as np
import torch
from scipy import stats

from models import get_predictions


# =============================================================================
# Agreement
# =============================================================================

def compute_agreement(preds1, preds2):
    """Fraction of inputs where two models make the same prediction."""
    return (preds1 == preds2).float().mean().item()


def compute_pairwise_agreement(models, id_ref, ood_ref, device):
    """
    Compute ID and OOD agreement for all pairs in a list of models.
    Returns list of {id_agr, ood_agr} dicts — one per pair.
    """
    points = []
    for i, j in itertools.combinations(range(len(models)), 2):
        mi, mj = models[i], models[j]
        pi_id  = get_predictions(mi, id_ref,  device)
        pj_id  = get_predictions(mj, id_ref,  device)
        pi_ood = get_predictions(mi, ood_ref, device)
        pj_ood = get_predictions(mj, ood_ref, device)
        points.append({
            "id_agr":  compute_agreement(pi_id,  pj_id),
            "ood_agr": compute_agreement(pi_ood, pj_ood),
        })
    return points


def compute_cross_agreement(models_a, models_b, id_ref, ood_ref, device):
    """
    Compute ID and OOD agreement between paired models from two lists.
    models_a[i] is paired with models_b[i] (same seed, different method).
    Returns list of {id_agr, ood_agr} dicts.
    """
    assert len(models_a) == len(models_b)
    points = []
    for ma, mb in zip(models_a, models_b):
        pa_id  = get_predictions(ma, id_ref,  device)
        pb_id  = get_predictions(mb, id_ref,  device)
        pa_ood = get_predictions(ma, ood_ref, device)
        pb_ood = get_predictions(mb, ood_ref, device)
        points.append({
            "id_agr":  compute_agreement(pa_id,  pb_id),
            "ood_agr": compute_agreement(pa_ood, pb_ood),
        })
    return points


# =============================================================================
# Entropy
# =============================================================================

@torch.no_grad()
def compute_entropy(model, data, device):
    """
    Mean binary prediction entropy over all examples.
    H(p) = -p*log(p) - (1-p)*log(1-p)

    H ≈ 0.0   → model is confident
    H ≈ 0.693 → model is maximally uncertain (predicts ~0.5)

    High entropy on OOD data = model destabilized by shift.
    Used to detect Case B: both models agree but both fail.
    """
    model.eval()
    eps    = 1e-6
    logits = model(data["images"].to(device))
    p      = torch.sigmoid(logits).cpu()
    H      = -(p * torch.log(p + eps) + (1-p) * torch.log(1-p + eps))
    return H.mean().item()


def entropy_interpretation(H, threshold=0.4):
    """
    Simple interpretation of entropy value.
    threshold=0.4 is roughly halfway to maximum uncertainty (0.693).
    """
    if H < 0.2:
        return "confident"
    elif H < threshold:
        return "moderate"
    else:
        return "uncertain"


# =============================================================================
# AGL line fitting
# =============================================================================

def probit(p, eps=1e-6):
    """Map probability to probit (inverse normal CDF) space."""
    return float(stats.norm.ppf(np.clip(p, eps, 1 - eps)))


def inv_probit(z):
    """Map probit value back to probability."""
    return float(stats.norm.cdf(z))


def fit_agl_line(points):
    """
    Fit AGL line in probit space from a list of {id_agr, ood_agr} points.
    Returns (slope, intercept, R²).

    Higher R² = stronger linear structure = more reliable signal.
    """
    if len(points) < 3:
        raise ValueError(f"Need at least 3 points to fit AGL line, got {len(points)}. "
                         f"Increase n_trials or n_seeds.")
    x = np.array([probit(p["id_agr"])  for p in points])
    y = np.array([probit(p["ood_agr"]) for p in points])
    slope, intercept, r, _, _ = stats.linregress(x, y)
    return slope, intercept, r ** 2


def compute_deviation(id_agr, ood_agr, slope, intercept):
    """
    Signed deviation of a point from the AGL line in probit space.

    Negative = below the line = IRM diverged more than expected from ERM
    Zero     = on the line    = IRM behaves like ERM under shift
    Positive = above the line = IRM agrees more than expected (unusual)
    """
    return probit(ood_agr) - (slope * probit(id_agr) + intercept)


def predict_ood_agreement(id_agr, slope, intercept):
    """Predict OOD agreement from ID agreement using the fitted line."""
    return inv_probit(slope * probit(id_agr) + intercept)


# =============================================================================
# Decision rule
# =============================================================================

DEVIATION_THRESHOLD = -0.10   # probit units
ENTROPY_THRESHOLD   = 0.40    # nats


def make_decision(cross_points, erm_entropies, irm_entropies):
    """
    Full decision combining agreement deviation and entropy.

    Returns dict with:
      decision:  DIVERGE / AGREE / BOTH_FAIL
      signal:    what the agreement says
      entropy:   what the entropy says
      details:   full numbers
    """
    deviations   = [p["deviation"] for p in cross_points]
    n_below      = sum(1 for d in deviations if d < DEVIATION_THRESHOLD)
    frac_below   = n_below / len(deviations)
    mean_dev     = float(np.mean(deviations))
    std_dev      = float(np.std(deviations))

    mean_erm_H   = float(np.mean(erm_entropies))
    mean_irm_H   = float(np.mean(irm_entropies))

    # Agreement signal
    if frac_below >= 0.5:
        agr_signal = "DIVERGE"
    else:
        agr_signal = "AGREE"

    # Entropy signal
    both_uncertain  = (mean_erm_H > ENTROPY_THRESHOLD and
                       mean_irm_H > ENTROPY_THRESHOLD)
    irm_degenerate  = (mean_irm_H > ENTROPY_THRESHOLD and
                       mean_erm_H < ENTROPY_THRESHOLD)

    # Combined decision
    if agr_signal == "AGREE" and both_uncertain:
        decision = "BOTH_FAIL"
        explanation = ("IRM and ERM agree but both are uncertain. "
                       "Neither method handles this shift. "
                       "Consider TTA or flagging for human review.")
    elif agr_signal == "AGREE":
        decision = "AGREE"
        explanation = ("IRM and ERM agree and ERM is confident. "
                       "The shift is mild or IRM collapsed to ERM. "
                       "Stick with best ERM config.")
    elif irm_degenerate:
        decision = "IRM_DEGENERATE"
        explanation = ("IRM disagrees with ERM but IRM is uncertain. "
                       "IRM likely found a degenerate solution. "
                       "Discard this IRM config, try different HP.")
    else:
        decision = "DIVERGE"
        explanation = ("IRM diverges from ERM under shift and is confident. "
                       "IRM is doing real work. "
                       "Proceed to TTA verification and HP selection.")

    return {
        "decision":    decision,
        "explanation": explanation,
        "agreement": {
            "signal":     agr_signal,
            "n_below":    n_below,
            "frac_below": frac_below,
            "mean_dev":   mean_dev,
            "std_dev":    std_dev,
        },
        "entropy": {
            "mean_erm_H":      mean_erm_H,
            "mean_irm_H":      mean_irm_H,
            "erm_state":       entropy_interpretation(mean_erm_H),
            "irm_state":       entropy_interpretation(mean_irm_H),
            "both_uncertain":  both_uncertain,
            "irm_degenerate":  irm_degenerate,
        },
    }
