# analysis/scores.py
"""
Label-free surrogate scores used by odp_bench_comparison.py to rank models
without OOD ground truth.

  ATC (atc_threshold, atc_score) - Average Thresholded Confidence
                                    (Garg et al. 2022); sel: argmax
  DOC (doc_score)                - val_conf_mean - test_conf_mean
                                    (confidence gap); sel: argmin
  NucNorm (nuclear_norm_score)   - nuclear_norm(test_probs) / N
                                    (prediction matrix structure); sel: argmax
  MDE (mde_score)                - -mean(log(sum(exp(p/T))))
                                    (energy; more negative = peaked); sel: argmin
  Disp (dispersion_score)        - mean distance of pseudo-class centroids
                                    from the global centroid; sel: argmax
"""

import numpy as np


def atc_threshold(val_probs, val_acc):
    n = len(val_probs)
    k = max(1, min(n, int(round(n * float(val_acc)))))
    return float(np.sort(val_probs.max(axis=1))[::-1][k - 1])


def atc_score(test_probs, threshold):
    return float((test_probs.max(axis=1) > threshold).mean())


def doc_score(val_probs, test_probs):
    """Confidence gap: val_conf - test_conf. Lower = smaller gap = better."""
    return float(val_probs.max(axis=1).mean() - test_probs.max(axis=1).mean())


def nuclear_norm_score(test_probs):
    """Nuclear norm / N. Higher = more structured prediction matrix."""
    return float(np.linalg.norm(test_probs, 'nuc') / len(test_probs))


def mde_score(test_probs, T=1.0):
    """Mean Dispersion Energy. More negative = more peaked distributions."""
    return float(-np.mean(T * np.log(np.sum(np.exp(test_probs / T), axis=1))))


def dispersion_score(test_probs):
    """Mean distance of pseudo-label class centroids from global centroid."""
    pseudo   = test_probs.argmax(axis=1)
    centroid = test_probs.mean(axis=0)
    scores   = []
    for c in range(test_probs.shape[1]):
        mask = pseudo == c
        if mask.sum() == 0:
            continue
        scores.append(np.linalg.norm(test_probs[mask].mean(axis=0) - centroid))
    return float(np.mean(scores)) if scores else 0.0
