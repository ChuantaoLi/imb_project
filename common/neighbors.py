"""common.neighbors — shared k-NN / SMOTE / safe-level utilities for the 14
data-resampling methods.

All helpers are ROBUST to tiny classes (the survey's KEEL sets include classes
with only a handful of samples, and a per-fold split can leave a minority class
with 2-3 samples): k is silently capped to len-1, and a class with a single
sample is duplicated rather than crashing SMOTE.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def split_classes(y):
    """Return (majority_label, minority_labels, counts) where majority is the
    single largest class and minorities are every smaller class."""
    from collections import Counter
    cnt = Counter(y.tolist())
    items = sorted(cnt.items(), key=lambda kv: -kv[1])
    maj = items[0][0]
    minors = [c for c, _ in items[1:]]
    return maj, minors, cnt


def knn_indices(X, k, exclude_self=True):
    """(n, kk) neighbour-index array; kk = min(k, n-1) (or n if not excluding
    self). Returns an (n,0) array when there is no valid neighbour."""
    n = len(X)
    want = (n - 1) if exclude_self else n
    kk = min(k, want)
    if kk <= 0:
        return np.empty((n, 0), dtype=int)
    nn = NearestNeighbors(n_neighbors=(kk + 1) if exclude_self else kk).fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:] if exclude_self else idx


def smote_generate(X_min, n_to_generate, k=5, rng=None):
    """Standard SMOTE: synthesize `n_to_generate` samples by linearly
    interpolating each seed with one of its k SAME-CLASS nearest neighbours.

    Robust to small classes:
      * k is capped at len(X_min)-1;
      * a single-sample class is duplicated (gap=0) rather than erroring.
    """
    rng = rng if rng is not None else np.random.RandomState(42)
    X_min = np.asarray(X_min, dtype=float)
    n = len(X_min)
    if n_to_generate <= 0:
        return np.empty((0, X_min.shape[1] if X_min.ndim == 2 else 1), dtype=float)
    if n == 0:
        return np.empty((0, X_min.shape[1]), dtype=float)
    if n == 1:
        return np.repeat(X_min, n_to_generate, axis=0)
    kk = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(X_min)
    _, idx = nn.kneighbors(X_min)
    idx = idx[:, 1:]                                   # exclude self
    d = X_min.shape[1]
    out = np.empty((n_to_generate, d), dtype=float)
    seeds = rng.randint(0, n, size=n_to_generate)
    nbs = idx[seeds, rng.randint(0, kk, size=n_to_generate)]
    gaps = rng.rand(n_to_generate, 1)
    out = X_min[seeds] + gaps * (X_min[nbs] - X_min[seeds])
    return out


def safe_level(X, y, k=5):
    """Safe Level (Bunkhumpornpat 2009 / SOUP): for each sample, fraction of
    its k nearest neighbours that share its class, in [0,1]. 1 = safe core,
    0 = boundary/outlier. Computed over the WHOLE set (mixed classes)."""
    n = len(X)
    kk = min(k, n - 1)
    if kk <= 0:
        return np.ones(n)
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]
    nb_lab = y[idx]
    same = (nb_lab == y[:, None]).sum(axis=1)
    return same / kk


def cross_class_nn_dist(X, y, c, k=5):
    """For every instance of class c, the distance to its k-th nearest
    OTHER-class neighbour (used by SHSampler / MC-CCR as the oversampling
    radius). Falls back to the single other-class NN distance when few exist."""
    Xc = X[y == c]
    Xoth = X[y != c]
    if len(Xoth) == 0 or len(Xc) == 0:
        return np.full(len(Xc), np.nan)
    kk = min(k, len(Xoth))
    nn = NearestNeighbors(n_neighbors=kk).fit(Xoth)
    dist, _ = nn.kneighbors(Xc)
    return dist[:, -1].astype(float)


def uniform_in_ball(seeds, n, radius, rng=None):
    """Generate `n` points uniformly inside d-dimensional balls of `radius`
    centred at rows of `seeds` (one seed per generated point, sampled with
    replacement). Used by the spherical-boundary oversamplers (MC-CCR /
    SHSampler)."""
    rng = rng if rng is not None else np.random.RandomState(42)
    seeds = np.asarray(seeds, dtype=float)
    d = seeds.shape[1]
    pick = rng.randint(0, len(seeds), size=n)
    # uniform direction on the sphere
    dirs = rng.normal(size=(n, d))
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    # uniform radius in a d-ball: r = R * U^(1/d)
    rho = radius * (rng.rand(n) ** (1.0 / d))
    return seeds[pick] + rho[:, None] * dirs


def median_nn_dist(Xc):
    """Median nearest-neighbour distance within a single class (default radius
    heuristic for the spherical oversamplers)."""
    if len(Xc) <= 1:
        return 1.0
    nn = NearestNeighbors(n_neighbors=min(2, len(Xc))).fit(Xc)
    dd, _ = nn.kneighbors(Xc)
    return float(np.median(dd[:, 1]))

