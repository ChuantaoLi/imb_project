"""SMOM — Synthetic Minority Oversampling for Multiclass (Zhu, Lin, Liu, PR 2017).

"Synthetic minority oversampling technique for multiclass imbalance problems."

Mechanism: extends SMOTE so it (a) considers the cluster/region structure of each
class and (b) weights the SELECTION of candidate neighbour directions for every
minority seed using whole-set neighbourhood information, aiming to EXPAND
minority region coverage while avoiding overlap with other classes.

Paper-faithful realisation (per minority class):
  * cluster each class first (the paper uses a neighbourhood-based clustering
    idea to characterise minority regions);
  * assign every instance an inherited SAFE / TRAPPED region label;
  * for each minority seed, compute a SELECTION WEIGHT FOR EACH CANDIDATE
    NEIGHBOUR DIRECTION, rather than weighting only the seed itself;
  * prefer directions whose midpoint remains surrounded by the same class,
    which operationalises the paper's "avoid over-generalisation" mechanism.

Implementation note:
  * the exact NBDOS formulas are not accessible here; we approximate the paper's
    neighbourhood-based clustering with per-class DBSCAN and derive SAFE /
    TRAPPED cluster labels from whole-set neighbourhood purity. This preserves
    the paper's explicit cluster-labelling -> direction-weighting flow instead
    of collapsing it to seed-only safe-level weighting.
"""
import os
import sys
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common.neighbors import safe_level, knn_indices, median_nn_dist
from common import smoke as _smoke

MODEL_KEY = "smom"
SAFE, TRAPPED = 1, 0


class SMOM(Resampler):
    def __init__(
        self,
        k=5,
        cluster_min_samples=3,
        trapped_penalty=0.2,
        midpoint_alpha=0.5,
        rf_n_estimators=30,
        n_jobs=-1,
        random_state=42,
        **kw
    ):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        self.k = int(k)
        self.cluster_min_samples = int(cluster_min_samples)
        self.trapped_penalty = float(trapped_penalty)
        self.midpoint_alpha = float(midpoint_alpha)

    def _target_size(self, y):
        cnt = Counter(np.asarray(y).tolist())
        return int(max(cnt.values())) if cnt else 0

    def _class_eps(self, Xc):
        if len(Xc) <= 1:
            return 1.0
        nn = NearestNeighbors(n_neighbors=min(self.k + 1, len(Xc))).fit(Xc)
        dist, _ = nn.kneighbors(Xc)
        kth = dist[:, -1]
        kth = kth[np.isfinite(kth) & (kth > 0)]
        if len(kth) == 0:
            return max(median_nn_dist(Xc), 1e-6)
        return max(float(np.median(kth)), 1e-6)

    def _cluster_and_label(self, X, y, cls, whole_safe):
        idx = np.where(y == cls)[0]
        Xc = X[idx]
        if len(Xc) == 0:
            return idx, np.empty(0, dtype=int), np.empty(0, dtype=int)
        if len(Xc) <= 2:
            cluster_ids = np.zeros(len(Xc), dtype=int)
            states = np.full(len(Xc), SAFE, dtype=int)
            return idx, cluster_ids, states

        eps = self._class_eps(Xc)
        min_samples = max(2, min(self.cluster_min_samples, len(Xc)))
        cluster_ids = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xc)
        states = np.full(len(Xc), TRAPPED, dtype=int)
        safe_local = whole_safe[idx]

        non_noise = cluster_ids[cluster_ids >= 0]
        if len(non_noise) == 0:
            cluster_ids = np.zeros(len(Xc), dtype=int)
            states[:] = np.where(safe_local >= 0.5, SAFE, TRAPPED)
            return idx, cluster_ids, states

        for cid in np.unique(non_noise):
            members = np.where(cluster_ids == cid)[0]
            cluster_purity = float(np.mean(safe_local[members]))
            states[members] = SAFE if cluster_purity >= 0.5 else TRAPPED

        noise = np.where(cluster_ids < 0)[0]
        if len(noise):
            states[noise] = TRAPPED
        return idx, cluster_ids, states

    def _seed_order(self, n, total, rng):
        if total <= 0 or n <= 0:
            return np.empty(0, dtype=int)
        reps = int(np.ceil(float(total) / float(n)))
        order = np.tile(np.arange(n, dtype=int), reps)[:total]
        rng.shuffle(order)
        return order

    def _midpoint_same_ratio(self, point, cls, nn_all, y):
        kk = min(self.k, len(y))
        if kk <= 0:
            return 1.0
        _, idx = nn_all.kneighbors(np.asarray(point, dtype=float).reshape(1, -1), n_neighbors=kk)
        return float(np.mean(y[idx[0]] == cls))

    def _direction_weights(self, seed_local, nn_same, Xc, cls, states, nn_all, y):
        candidates = nn_same[seed_local]
        if len(candidates) == 0:
            return np.empty(0, dtype=float)
        weights = np.zeros(len(candidates), dtype=float)
        seed_state = 1.0 if states[seed_local] == SAFE else np.sqrt(self.trapped_penalty)
        for j, nb_local in enumerate(candidates):
            midpoint = Xc[seed_local] + self.midpoint_alpha * (Xc[nb_local] - Xc[seed_local])
            same_ratio = self._midpoint_same_ratio(midpoint, cls, nn_all, y)
            nb_state = 1.0 if states[nb_local] == SAFE else self.trapped_penalty
            weights[j] = max(same_ratio, 1e-6) * seed_state * nb_state
        s = weights.sum()
        return (weights / s) if s > 0 else np.full(len(candidates), 1.0 / len(candidates))

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        target = self._target_size(y)
        whole_safe = safe_level(X, y, self.k)
        nn_all = NearestNeighbors(n_neighbors=min(max(self.k, 1), len(X))).fit(X)
        counts = Counter(np.asarray(y).tolist())
        Xl, yl = [], []

        for c in np.unique(y):
            idx = np.where(y == c)[0]
            Xc = X[idx]
            Xl.append(Xc)
            yl.append(np.full(len(Xc), c, dtype=y.dtype))
            if counts[int(c)] >= target:
                continue

            n_gen = target - len(Xc)
            if n_gen > 0:
                if len(Xc) == 1:
                    gen = np.repeat(Xc, n_gen, axis=0)
                else:
                    _, _, states = self._cluster_and_label(X, y, c, whole_safe)
                    nn_same = knn_indices(Xc, self.k)
                    seeds = self._seed_order(len(Xc), n_gen, rng)
                    gen = np.empty((n_gen, X.shape[1]), dtype=float)
                    for t, seed_local in enumerate(seeds):
                        candidates = nn_same[seed_local]
                        if len(candidates) == 0:
                            gen[t] = Xc[seed_local]
                            continue
                        w_dir = self._direction_weights(seed_local, nn_same, Xc, c, states, nn_all, y)
                        nb_local = rng.choice(candidates, p=w_dir)
                        gap = rng.rand()
                        gen[t] = Xc[seed_local] + gap * (Xc[nb_local] - Xc[seed_local])
                Xl.append(gen)
                yl.append(np.full(n_gen, c, dtype=y.dtype))
        return np.vstack(Xl), np.concatenate(yl)


def build(
    random_state=42,
    smoke=False,
    k=5,
    cluster_min_samples=3,
    trapped_penalty=0.2,
    midpoint_alpha=0.5,
    rf_n_estimators=30,
    n_jobs=-1,
    **kw
):
    return SMOM(
        k=k,
        cluster_min_samples=cluster_min_samples,
        trapped_penalty=trapped_penalty,
        midpoint_alpha=midpoint_alpha,
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
