"""GLOS — Global-LOcal information based Oversampling (Han et al., IJMLC 2023).

"Global-local information based oversampling for multi-class imbalanced data."

Mechanism: partitions each minority's instances by local difficulty and applies
a DIFFERENT oversampler to each group, producing more diverse synthetic samples:
  * SAFE instances (high same-class NN fraction)        -> MMO  (global): draw from the
    class Gaussian N(mu, Sigma) -- reflects global within-class variation (MDO-style).
  * BORDERLINE instances (mid same-class NN fraction)    -> AMBO (local): Borderline-SMOTE
    -- SMOTE toward a same-class NN (strengthen the boundary).
  * SMALL-DISJUNCT instances (low same-class NN fraction, isolated) -> RRO (radial):
    Gaussian noise around the seed (radial, fills isolated sub-regions).

Generation budget is allocated PER INSTANCE from the gap between class-level
and instance-level DID scores. The three sub-algorithms (MMO/AMBO/RRO) are
then dispatched according to each selected instance's local neighbourhood
pattern.
"""
import os
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common.neighbors import split_classes, knn_indices, median_nn_dist
from common import smoke as _smoke

MODEL_KEY = "glos"


class GLOS(Resampler):
    def __init__(self, k=5, shrinkage=1e-3, rf_n_estimators=30, n_jobs=-1,
                 random_state=42, **kw):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        self.k = k
        self.shrinkage = shrinkage

    def _mmo(self, Xc, n, rng):                  # global Gaussian (MDO-style)
        d = Xc.shape[1]
        if len(Xc) == 1:
            return np.repeat(Xc, n, axis=0)
        mu = Xc.mean(0)
        Sig = np.cov(Xc, rowvar=False) + self.shrinkage * np.eye(d)
        try:
            return rng.multivariate_normal(mu, Sig, size=n)
        except Exception:
            return Xc[rng.randint(0, len(Xc), n)]

    def _ambo(self, Xc, n, rng):                 # Borderline-SMOTE
        if len(Xc) == 1:
            return np.repeat(Xc, n, axis=0)
        nn = knn_indices(Xc, self.k); kk = nn.shape[1]
        s = rng.randint(0, len(Xc), n)
        nb = nn[s, rng.randint(0, kk, n)]
        g = rng.rand(n, 1)
        return Xc[s] + g * (Xc[nb] - Xc[s])

    def _rro(self, Xc, n, rng):                  # radial Gaussian (MC-RBO-style)
        sigma = median_nn_dist(Xc) if len(Xc) > 1 else 1.0
        s = rng.randint(0, len(Xc), n)
        return Xc[s] + rng.normal(0.0, sigma, size=(n, Xc.shape[1]))

    def _did_scores(self, X, y, c, global_idx=None, global_dist=None):
        """Return (instance_did, class_did) for class c.

        When global NN data is provided, derives same/other distances from mixed
        neighbourhood instead of fitting three separate trees.
        """
        Xc = X[y == c]
        if len(Xc) == 0:
            return np.empty(0, dtype=float), 0.0
        if len(Xc) == 1:
            return np.zeros(1, dtype=float), 0.0

        if global_idx is not None and global_dist is not None:
            # Derive from precomputed global NN to avoid extra NN fit
            c_mask = y == c
            idx_c = np.where(c_mask)[0]
            same_mask = y[global_idx[idx_c]] == c
            same_mean = np.where(
                same_mask.any(axis=1),
                (global_dist[idx_c] * same_mask).sum(axis=1) / np.maximum(same_mask.sum(axis=1), 1),
                np.zeros(len(Xc)))
            other_mask = ~same_mask
            if other_mask.any():
                other_mean = (global_dist[idx_c] * other_mask).sum(axis=1) / np.maximum(other_mask.sum(axis=1), 1)
            else:
                other_mean = np.full(len(Xc), same_mean.max() + 1.0, dtype=float)
        else:
            Xo = X[y != c]
            same_k = min(self.k + 1, len(Xc))
            same_nn = NearestNeighbors(n_neighbors=same_k).fit(Xc)
            same_dist, _ = same_nn.kneighbors(Xc)
            same_dist = same_dist[:, 1:]
            same_mean = same_dist.mean(axis=1) if same_dist.shape[1] > 0 else np.zeros(len(Xc))

            if len(Xo) == 0:
                other_mean = np.full(len(Xc), same_mean.max() + 1.0, dtype=float)
            else:
                other_k = min(self.k, len(Xo))
                other_nn = NearestNeighbors(n_neighbors=other_k).fit(Xo)
                other_dist, _ = other_nn.kneighbors(Xc)
                other_mean = other_dist.mean(axis=1)

        did = other_mean / np.clip(same_mean, 1e-12, None)
        return did, float(did.mean())

    def _local_groups(self, X, y, c, global_idx=None, global_kk=None):
        """safe / borderline / small-disjunct groups from mixed-neighbourhood
        composition for class c.  Accepts precomputed global NN to avoid re-fitting."""
        Xc = X[y == c]
        if len(Xc) == 0:
            return np.empty(0, dtype=object), None, None, None
        n = len(X)
        if global_idx is not None:
            idx = global_idx[np.where(y == c)[0]]
            kk = global_kk
        else:
            kk = min(self.k, n - 1)
            if kk <= 0:
                return np.full(len(Xc), "safe", dtype=object), None, None, None
            nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
            _, idx = nn.kneighbors(Xc)
            idx = idx[:, 1:]
        if kk <= 0:
            return np.full(len(Xc), "safe", dtype=object), None, None, None
        same = (y[idx] == c).sum(axis=1)
        groups = np.full(len(Xc), "small", dtype=object)
        groups[same == kk] = "safe"
        groups[(same > 0) & (same < kk)] = "border"
        return groups

    @staticmethod
    def _allocate_counts(weights, total):
        weights = np.asarray(weights, dtype=float)
        if total <= 0 or len(weights) == 0:
            return np.zeros(len(weights), dtype=int)
        weights = np.clip(weights, 0.0, None)
        if weights.sum() <= 0:
            weights = np.ones(len(weights), dtype=float)
        raw = weights / weights.sum() * total
        counts = np.floor(raw).astype(int)
        remain = total - counts.sum()
        if remain > 0:
            order = np.argsort(-(raw - counts))
            counts[order[:remain]] += 1
        return counts

    def _mmo_seeded(self, Xc, seed_idx, counts, rng):
        total = int(np.sum(counts))
        if total <= 0:
            return np.empty((0, Xc.shape[1]), dtype=float)
        if len(Xc) == 1:
            return np.repeat(Xc, total, axis=0)
        d = Xc.shape[1]
        mu = Xc.mean(0)
        Sig = np.cov(Xc, rowvar=False) + self.shrinkage * np.eye(d)
        out = []
        for idx, n in zip(seed_idx, counts):
            if n <= 0:
                continue
            seed = Xc[idx:idx + 1]
            try:
                global_draw = rng.multivariate_normal(mu, Sig, size=n)
            except Exception:
                global_draw = Xc[rng.randint(0, len(Xc), size=n)]
            gap = rng.rand(n, 1)
            out.append(seed + gap * (global_draw - seed))
        return np.vstack(out) if out else np.empty((0, Xc.shape[1]), dtype=float)

    def _ambo_seeded(self, Xc, seed_idx, counts, rng):
        total = int(np.sum(counts))
        if total <= 0:
            return np.empty((0, Xc.shape[1]), dtype=float)
        if len(Xc) == 1:
            return np.repeat(Xc, total, axis=0)
        nn = knn_indices(Xc, self.k)
        out = []
        for idx, n in zip(seed_idx, counts):
            if n <= 0:
                continue
            if nn.shape[1] <= 0:
                out.append(np.repeat(Xc[idx:idx + 1], n, axis=0))
                continue
            nbs = nn[idx]
            pick = rng.choice(nbs, size=n, replace=True)
            gap = rng.rand(n, 1)
            seed = np.repeat(Xc[idx:idx + 1], n, axis=0)
            out.append(seed + gap * (Xc[pick] - seed))
        return np.vstack(out) if out else np.empty((0, Xc.shape[1]), dtype=float)

    def _rro_seeded(self, Xc, seed_idx, counts, rng):
        total = int(np.sum(counts))
        if total <= 0:
            return np.empty((0, Xc.shape[1]), dtype=float)
        sigma = median_nn_dist(Xc) if len(Xc) > 1 else 1.0
        out = []
        for idx, n in zip(seed_idx, counts):
            if n <= 0:
                continue
            seed = np.repeat(Xc[idx:idx + 1], n, axis=0)
            out.append(seed + rng.normal(0.0, sigma, size=(n, Xc.shape[1])))
        return np.vstack(out) if out else np.empty((0, Xc.shape[1]), dtype=float)

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        maj, minors, cnt = split_classes(y)
        target = cnt[maj]
        Xl, yl = [X[y == maj]], [y[y == maj]]

        # Precompute global NN once for all classes (used by both _did_scores & _local_groups)
        n = len(X)
        kk_global = min(self.k, n - 1)
        if kk_global > 0:
            nn_global = NearestNeighbors(n_neighbors=kk_global + 1).fit(X)
            g_dist, g_idx = nn_global.kneighbors(X)
            g_dist, g_idx = g_dist[:, 1:], g_idx[:, 1:]  # exclude self
        else:
            g_dist = g_idx = None

        for c in minors:
            Xc = X[y == c]
            n_gen = target - len(Xc)
            if n_gen > 0 and len(Xc) >= 1:
                did_i, did_c = self._did_scores(X, y, c, global_idx=g_idx, global_dist=g_dist)
                groups = self._local_groups(X, y, c, global_idx=g_idx, global_kk=kk_global)
                hard = np.where(did_i < did_c)[0]
                if len(hard) == 0:
                    hard = np.argsort(did_i)[:max(1, min(len(Xc), self.k))]
                weights = np.maximum(did_c - did_i[hard], 0.0)
                counts = self._allocate_counts(weights, n_gen)

                gen = []
                for name, fn in (
                    ("safe", self._mmo_seeded),
                    ("border", self._ambo_seeded),
                    ("small", self._rro_seeded),
                ):
                    mask = groups[hard] == name
                    if mask.any():
                        gen.append(fn(Xc, hard[mask], counts[mask], rng))
                gen = np.vstack([g for g in gen if len(g) > 0]) if gen else np.empty((0, X.shape[1]))
                if len(gen) < n_gen:
                    extra = self._ambo_seeded(Xc, hard, self._allocate_counts(weights, n_gen - len(gen)), rng)
                    gen = np.vstack([gen, extra]) if len(extra) > 0 else gen
                Xl.append(gen[:n_gen])
                yl.append(np.full(min(len(gen), n_gen), c, dtype=y.dtype))
            Xl.append(Xc); yl.append(np.full(len(Xc), c, dtype=y.dtype))
        return np.vstack(Xl), np.concatenate(yl)


def build(random_state=42, smoke=False, k=5, rf_n_estimators=30, n_jobs=-1, **kw):
    return GLOS(
        k=k,
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
