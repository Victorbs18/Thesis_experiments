# analysis/distance_metrics.py
"""
Distribution-distance metrics used by cross_algorithm_agreement.py
  - MMD: (Maximum Mean Discrepancy)
  - Wasserstein
  - PAD: (Proxy A-Distance)        
"""

import numpy as np
import ot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances


def _subsample(X, max_n=2000, seed=87):
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_n:
        X = X[rng.choice(X.shape[0], max_n, replace=False)]
    return X


def compute_mmd(X, Y):
    """
    Maximum Mean Discrepancy with an RBF kernel, averaged over three bandwidths.
    """
    X = _subsample(X)
    Y = _subsample(Y)
    XY = np.vstack([X, Y])
    sq = euclidean_distances(XY, XY, squared=True)
    # indices of the upper triangle excluding diagonal
    idx = np.triu_indices(len(XY), k=1)
    # meadian heuristic
    sigma = float(np.median(np.sqrt(sq[idx])))
    sigma = max(sigma, 1e-6)
    bandwidths = [sigma / 2, sigma, sigma * 2]

    n_x = len(X)
    mmd = 0.0
    for s in bandwidths:
        # RBF Kernel
        K   = np.exp(-sq / (2 * s ** 2))
        # empirical estimates of the 3 expectations
        Kxx = K[:n_x, :n_x].mean() 
        Kyy = K[n_x:, n_x:].mean()
        Kxy = K[:n_x, n_x:].mean()
        mmd += float(Kxx + Kyy - 2 * Kxy)
    return max(mmd / len(bandwidths), 0.0)


def compute_wasserstein(X, Y, n_iters=1000, reg=0.05):
    """
    Entropic-regularized (Sinkhorn) Wasserstein distance via POT, with a
    Euclidean ground cost and uniform marginals.
    """
    X = _subsample(X, max_n=1000)
    Y = _subsample(Y, max_n=1000)
    n, m = X.shape[0], Y.shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    M = ot.dist(X, Y, metric='euclidean')
    scale = M.max()
    if scale <= 0:
        return 0.0
    cost = ot.sinkhorn2(a, b, M / scale, reg, numItermax=n_iters)
    return float(max(cost, 0.0) * scale)


def compute_pad(X, Y, n_iters=1000):
    """
    Proxy A-Distance = 2 * (1 - 2 * error), where error is the training
    error rate of a linear classifier trained to separate X from Y.
    """
    X = _subsample(X)
    Y = _subsample(Y)
    data   = np.vstack([X, Y])
    labels = np.array([0] * len(X) + [1] * len(Y))

    clf = LogisticRegression(max_iter=n_iters)
    clf.fit(data, labels)
    error = 1.0 - clf.score(data, labels)
    return float(max(0.0, 2.0 * (1.0 - 2.0 * error)))
