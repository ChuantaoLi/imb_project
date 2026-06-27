"""SHSampler — Similarity-based Hybrid Sampling (Zheng et al., DSAA 2022).

"Combating mutuality with difficulty factors in multi-class imbalanced data:
a similarity-based hybrid sampling."

Mechanism (hybrid over+under):
  * sample SIMILARITY / DISSIMILARITY estimation identifies difficulty factors.
    We realise these two signals with safe level (similarity to own class) and
    cross-class k-NN distance (dissimilarity to other classes), which are the
    two quantities explicitly exposed by the paper abstract and downstream
    descriptions.
  * UNDERsampling (relative majority): ROULETTE-WHEEL deletion by difficulty.
    LOW-safe-level samples are more likely to be removed.
  * OVERsampling (relative minority): HIGH-safe-level seeds are strengthened by
    generating synthetic instances uniformly inside a seed-specific sphere whose
    radius is the seed's distance to its k-th OTHER-class neighbour.

Implementation note:
  * The full PDF is not machine-readable here, so the exact class-grouping
    threshold is not directly extractable. To instantiate the paper's
    "relative majority" / "relative minority" wording in a multi-class setting,
    we use the mean class size as the relative balance target: classes above the
    mean are weakened, classes below the mean are strengthened, and classes on
    the mean are left unchanged. This is explicit and avoids silently collapsing
    the method to a single-global-majority heuristic.
"""
import os
import sys
import numpy as np
from collections import Counter

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common.neighbors import safe_level, cross_class_nn_dist, median_nn_dist
from common import smoke as _smoke

MODEL_KEY = "shsampler"


class SHSampler(Resampler):
    def __init__(
        self,
        k=5,
        target_mode="mean",
        rf_n_estimators=30,
        n_jobs=-1,
        random_state=42,
        **kw
    ):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        self.k = int(k)
        self.target_mode = str(target_mode)

    def _target_size(self, counts):
        vals = np.asarray(list(counts.values()), dtype=float)
        if self.target_mode == "max":
            return int(np.max(vals))
        return int(round(float(np.mean(vals))))

    def _estimate_similarity(self, X, y):
        return safe_level(X, y, self.k)

    def _estimate_dissimilarity(self, X, y, c):
        radii = cross_class_nn_dist(X, y, c, self.k)
        fallback = median_nn_dist(X[y == c])
        return np.where(np.isfinite(radii) & (radii > 0), radii, fallback)

    def _roulette_delete(self, idx, sim, n_delete, rng):
        if n_delete <= 0 or len(idx) == 0:
            return idx
        n_delete = min(int(n_delete), len(idx))
        delete_w = (1.0 - np.asarray(sim, dtype=float)) + 1e-6
        delete_w /= delete_w.sum()
        drop_local = rng.choice(len(idx), size=n_delete, replace=False, p=delete_w)
        keep_mask = np.ones(len(idx), dtype=bool)
        keep_mask[drop_local] = False
        return idx[keep_mask]

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        cnt = Counter(y.tolist())
        target = self._target_size(cnt)
        sim = self._estimate_similarity(X, y)
        Xl, yl = [], []

        for c in np.unique(y):
            idx = np.where(y == c)[0]
            sc = sim[idx]
            if len(idx) > target:                       # relative majority -> roulette delete by difficulty
                keep = self._roulette_delete(idx, sc, len(idx) - target, rng)
                Xl.append(X[keep]); yl.append(y[keep])
            elif len(idx) < target:                     # relative minority -> spherical strengthening
                Xl.append(X[idx]); yl.append(y[idx])
                n_gen = target - len(idx)
                if n_gen > 0:
                    radii = self._estimate_dissimilarity(X, y, c)
                    seed_w = sc + 1e-6
                    seed_w /= seed_w.sum()
                    local_seeds = rng.choice(len(idx), size=n_gen, replace=True, p=seed_w)
                    seeds = idx[local_seeds]
                    r_per = radii[local_seeds]
                    gen = _uniform_per_seed(X[seeds], r_per, rng)
                    Xl.append(gen); yl.append(np.full(n_gen, c, dtype=y.dtype))
            else:
                Xl.append(X[idx]); yl.append(y[idx])
        return np.vstack(Xl), np.concatenate(yl)


def _uniform_per_seed(seeds, r_per, rng):
    seeds = np.asarray(seeds, float); d = seeds.shape[1]; n = len(seeds)
    dirs = rng.normal(size=(n, d)); dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    rho = r_per * (rng.rand(n) ** (1.0 / d))
    return seeds + rho[:, None] * dirs


def build(
    random_state=42,
    smoke=False,
    k=5,
    target_mode="mean",
    rf_n_estimators=30,
    n_jobs=-1,
    **kw
):
    return SHSampler(
        k=k,
        target_mode=target_mode,
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
