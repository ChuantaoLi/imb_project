"""QC-SMOTE: Quality-Controlled SMOTE for imbalanced classification.

The paper couples three decisions that plain SMOTE leaves implicit:
minority-seed trustworthiness, best-of-K candidate quality, and an adaptive
fallback to duplication when interpolation is unsafe.  This implementation keeps
that flow in the project-wide resampler contract and applies it one-vs-rest for
multiclass folds.
"""
import os
import sys
from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from common.resampler import Resampler
from common.neighbors import knn_indices
from common import smoke as _smoke

MODEL_KEY = "QC-SMOTE"


def _minmax01(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return v
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo + 1e-12:
        return np.ones_like(v)
    return (v - lo) / (hi - lo)


class QCSMOTE(Resampler):
    def __init__(
        self,
        k=5,
        k_low=1,
        k_mid=3,
        k_high=5,
        tau1=0.33,
        tau2=0.66,
        purity_threshold=0.3,
        clearance_threshold=0.05,
        eta=0.5,
        lambda_clearance=0.25,
        rho0=1.0,
        trust_weights=(0.4, 0.3, 0.3),
        rf_n_estimators=30,
        n_jobs=-1,
        random_state=42,
        **kw
    ):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        self.k = int(k)
        self.k_low = int(k_low)
        self.k_mid = int(k_mid)
        self.k_high = int(k_high)
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)
        self.purity_threshold = float(purity_threshold)
        self.clearance_threshold = float(clearance_threshold)
        self.eta = float(eta)
        self.lambda_clearance = float(lambda_clearance)
        self.rho0 = float(rho0)
        self.trust_weights = tuple(float(w) for w in trust_weights)

    def _global_neighbours(self, X):
        kk = min(self.k, len(X) - 1)
        if kk <= 0:
            return None, None, kk
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
        dist, idx = nn.kneighbors(X)
        return dist[:, 1:], idx[:, 1:], kk

    def _trustworthiness(self, X, y, cls, dist_all, idx_all, kk):
        idx_c = np.where(y == cls)[0]
        Xc = X[idx_c]
        if len(Xc) == 0:
            return idx_c, np.empty(0), np.empty(0)
        if kk <= 0:
            return idx_c, np.ones(len(Xc)), np.zeros(len(Xc))

        nb_lab = y[idx_all[idx_c]]
        support = np.mean(nb_lab == cls, axis=1)

        if len(Xc) <= 1:
            density = np.ones(len(Xc))
        else:
            n_same = min(self.k, len(Xc) - 1) + 1
            sd, _ = NearestNeighbors(n_neighbors=n_same).fit(Xc).kneighbors(Xc)
            density = 1.0 / (np.mean(sd[:, 1:], axis=1) + 1e-12)
            density = _minmax01(density)

        Xoth = X[y != cls]
        if len(Xoth) == 0:
            isolation = np.ones(len(Xc))
        else:
            od, _ = NearestNeighbors(n_neighbors=1).fit(Xoth).kneighbors(Xc)
            isolation = _minmax01(od[:, 0])

        w0, w1, w2 = self.trust_weights
        s = max(w0 + w1 + w2, 1e-12)
        trust = (w0 * support + w1 * density + w2 * isolation) / s
        trust = np.clip(trust, 0.0, None)
        return idx_c, trust, support

    def _local_purity(self, point, cls, nn_all, y):
        kk = min(self.k, len(y))
        if kk <= 0:
            return 1.0
        _, idx = nn_all.kneighbors(np.asarray(point).reshape(1, -1), n_neighbors=kk)
        return float(np.mean(y[idx[0]] == cls))

    def _clearance(self, point, X_other):
        if len(X_other) == 0:
            return np.inf
        d = np.linalg.norm(X_other - point, axis=1)
        return float(np.min(d))

    def _fallback_duplicate(self, seed_local, Xc, trust, same_nn):
        candidates = same_nn[seed_local]
        if len(candidates) == 0:
            return Xc[seed_local].copy()
        best = candidates[int(np.argmax(trust[candidates]))]
        return Xc[best].copy()

    def _generate_for_class(self, X, y, cls, n_need, rng, dist_all, idx_all, kk):
        idx_c, trust, support = self._trustworthiness(X, y, cls, dist_all, idx_all, kk)
        Xc = X[idx_c]
        if n_need <= 0 or len(Xc) == 0:
            return np.empty((0, X.shape[1]))
        if len(Xc) == 1:
            return np.repeat(Xc, n_need, axis=0)

        weights = trust + 1e-12
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(Xc), 1.0 / len(Xc))
        expected = n_need * weights
        allocations = np.floor(expected).astype(int)
        fractional = expected - allocations
        if allocations.sum() < n_need:
            missing = n_need - int(allocations.sum())
            if fractional.sum() > 0:
                add = rng.choice(np.arange(len(Xc)), size=missing, replace=True, p=fractional / fractional.sum())
            else:
                add = rng.choice(np.arange(len(Xc)), size=missing, replace=True, p=weights)
            for a in add:
                allocations[int(a)] += 1
        seeds = np.repeat(np.arange(len(Xc)), allocations)
        rng.shuffle(seeds)

        overlap = 1.0 - float(np.mean(support)) if len(support) else 0.0
        if overlap < self.tau1:
            n_candidates = max(1, self.k_low)
        elif overlap < self.tau2:
            n_candidates = max(self.k_low + 1, self.k_mid)
        else:
            n_candidates = max(self.k_mid + 1, self.k_high)
        rho_c = min(1.0, max(1e-12, self.rho0 * (1.0 - overlap)))

        same_nn = knn_indices(Xc, self.k)
        nn_all = NearestNeighbors(n_neighbors=min(max(self.k, 1), len(X))).fit(X)
        X_other = X[y != cls]
        if len(X_other):
            od, _ = NearestNeighbors(n_neighbors=1).fit(X_other).kneighbors(Xc)
            tau_clear = max(float(np.median(od[:, 0])), 1e-12)
        else:
            tau_clear = 1.0

        out = np.empty((n_need, X.shape[1]), dtype=float)
        for t, seed_local in enumerate(seeds):
            neighs = same_nn[seed_local]
            if len(neighs) == 0:
                out[t] = Xc[seed_local]
                continue
            nb_local = int(rng.choice(neighs))
            midpoint = 0.5 * (Xc[seed_local] + Xc[nb_local])
            mid_purity = self._local_purity(midpoint, cls, nn_all, y)

            best_x, best_ipq, best_clear, best_score = None, -np.inf, 0.0, -np.inf
            for _ in range(n_candidates):
                gap = rng.rand() * rho_c
                cand = Xc[seed_local] + gap * (Xc[nb_local] - Xc[seed_local])
                purity = self._local_purity(cand, cls, nn_all, y)
                ipq = self.eta * purity + (1.0 - self.eta) * mid_purity
                clear = self._clearance(cand, X_other)
                score = ipq + self.lambda_clearance * min(1.0, clear / tau_clear)
                if score > best_score:
                    best_x, best_ipq, best_clear, best_score = cand, ipq, clear, score

            if best_ipq >= self.purity_threshold and best_clear >= self.clearance_threshold * tau_clear:
                out[t] = best_x
            else:
                out[t] = self._fallback_duplicate(seed_local, Xc, trust, same_nn)
        return out

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        counts = Counter(y.tolist())
        target = int(max(counts.values()))
        dist_all, idx_all, kk = self._global_neighbours(X)

        parts_x, parts_y = [X], [y]
        for cls in sorted(counts):
            n_need = target - counts[cls]
            gen = self._generate_for_class(X, y, cls, n_need, rng, dist_all, idx_all, kk)
            if len(gen):
                parts_x.append(gen)
                parts_y.append(np.full(len(gen), cls, dtype=y.dtype))
        return np.vstack(parts_x), np.concatenate(parts_y)


def build(random_state=42, smoke=False, rf_n_estimators=30, n_jobs=-1, **kw):
    return QCSMOTE(
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
