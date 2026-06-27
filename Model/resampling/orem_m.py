"""OREM-M — Oversampling with Reliably Expanding Minority regions (Zhu, Liu, Zhu,
TKDE 2023).

"Oversampling with reliably expanding minority class regions for imbalanced
data learning."

Mechanism:
  1. For a binary subproblem, discover the candidate minority region (CMR)
     around every minority sample from the ordered same-class / other-class
     neighborhood list.
  2. Identify assistant seeds that correspond to clean subregions, i.e. local
     directions whose midpoint area is not blocked by majority samples.
  3. Generate synthetic samples only toward those assistant seeds; if an
     assistant seed is a majority sample, only move halfway toward it.
  4. Extend OREM to multiclass imbalance via iterative generation so later
     minority classes also avoid the synthetic samples already created for the
     previous classes.
"""
import os
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common import smoke as _smoke

MODEL_KEY = "orem_m"


class OREMM(Resampler):
    def __init__(
        self,
        k=None,
        consecutive_majority=5,
        iteration_order="ascending",
        rf_n_estimators=30,
        n_jobs=-1,
        random_state=42,
        **kw
    ):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        if iteration_order not in {"ascending", "descending"}:
            raise ValueError("iteration_order must be one of ascending/descending")
        # `k` is accepted only for backward compatibility with older wrappers.
        self.k = k
        self.consecutive_majority = int(consecutive_majority)
        self.iteration_order = iteration_order

    def _knn_all(self, data_u, data_s, include_self=False):
        data_u = np.asarray(data_u, dtype=float)
        data_s = np.asarray(data_s, dtype=float)
        if len(data_u) == 0 or len(data_s) == 0:
            return np.empty((len(data_u), 0), dtype=int), np.empty((len(data_u), 0), dtype=float)
        k = len(data_s) if not include_self else min(len(data_s), len(data_u) + len(data_s))
        if k <= 0:
            return np.empty((len(data_u), 0), dtype=int), np.empty((len(data_u), 0), dtype=float)
        nn = NearestNeighbors(n_neighbors=min(len(data_s), k), metric="euclidean").fit(data_s)
        dist, idx = nn.kneighbors(data_u, return_distance=True)
        if include_self:
            out_idx, out_dist = [], []
            for i in range(len(data_u)):
                mask = idx[i] != i
                out_idx.append(idx[i][mask])
                out_dist.append(dist[i][mask])
            max_len = max((len(row) for row in out_idx), default=0)
            idx_pad = np.full((len(data_u), max_len), -1, dtype=int)
            dist_pad = np.full((len(data_u), max_len), np.inf, dtype=float)
            for i, (row_idx, row_dist) in enumerate(zip(out_idx, out_dist)):
                idx_pad[i, :len(row_idx)] = row_idx
                dist_pad[i, :len(row_dist)] = row_dist
            return idx_pad, dist_pad
        return idx.astype(int), dist.astype(float)

    def _discover_cmr(self, X_pos, X_neg):
        np_pos = len(X_pos)
        if np_pos == 0:
            return []
        if np_pos == 1:
            return [[]]
        same_idx, same_dist = self._knn_all(X_pos, X_pos, include_self=True)
        neg_idx, neg_dist = self._knn_all(X_pos, X_neg, include_self=False)
        cas = []
        for i in range(np_pos):
            dist_i = np.concatenate([same_dist[i], neg_dist[i]])
            idx_i = np.concatenate([same_idx[i], neg_idx[i] + np_pos])
            valid = idx_i >= 0
            dist_i = dist_i[valid]
            idx_i = idx_i[valid]
            if len(dist_i) == 0:
                cas.append([])
                continue
            order = np.argsort(dist_i)
            sorted_idx = idx_i[order]
            count_break = 0
            chosen = sorted_idx.copy()
            for j, idx_val in enumerate(sorted_idx):
                if idx_val >= np_pos:
                    count_break += 1
                else:
                    count_break = 0
                if count_break >= self.consecutive_majority:
                    chosen = sorted_idx[:max(j - self.consecutive_majority + 1, 1)]
                    break
            cas.append(chosen.astype(int).tolist())
        return cas

    def _identify_clean_regions(self, X_pos, X_neg, cas):
        np_pos = len(X_pos)
        data = np.vstack([X_pos, X_neg]) if len(X_neg) else X_pos.copy()
        assistants = [[] for _ in range(np_pos)]
        for i in range(np_pos):
            for j, aid in enumerate(cas[i]):
                mean_ij = np.mean(np.vstack([X_pos[i], data[aid]]), axis=0)
                threshold = float(np.linalg.norm(X_pos[i] - mean_ij))
                prior = cas[i][:j]
                if len(prior) == 0:
                    assistants[i].append(int(aid))
                    continue
                prior_points = data[np.asarray(prior, dtype=int)]
                prior_dist = np.linalg.norm(prior_points - mean_ij.reshape(1, -1), axis=1)
                closer = prior_dist - threshold < 1e-5
                maj_count = int(np.sum(np.asarray(prior, dtype=int)[closer] >= np_pos))
                if maj_count == 0:
                    assistants[i].append(int(aid))
        return assistants

    def _generate_binary(self, X_pos, X_neg, n_gen, rng):
        X_pos = np.asarray(X_pos, dtype=float)
        X_neg = np.asarray(X_neg, dtype=float)
        if n_gen <= 0 or len(X_pos) == 0:
            return np.empty((0, X_pos.shape[1] if X_pos.ndim == 2 else 0), dtype=float)
        if len(X_pos) == 1:
            return np.repeat(X_pos, n_gen, axis=0)

        cas = self._discover_cmr(X_pos, X_neg)
        assistants = self._identify_clean_regions(X_pos, X_neg, cas)
        np_pos = len(X_pos)
        data = np.vstack([X_pos, X_neg]) if len(X_neg) else X_pos.copy()

        times = n_gen // np_pos
        seed_ids = np.repeat(np.arange(np_pos), times)
        remainder = n_gen - len(seed_ids)
        if remainder > 0:
            seed_ids = np.concatenate([seed_ids, rng.choice(np.arange(np_pos), size=remainder, replace=False)])
        rng.shuffle(seed_ids)

        generated = np.zeros((n_gen, X_pos.shape[1]), dtype=float)
        for i, sid in enumerate(seed_ids):
            sample = X_pos[sid]
            as_i = assistants[sid]
            if len(as_i) == 0:
                generated[i] = sample
                continue
            aid = int(as_i[rng.randint(len(as_i))])
            gap = rng.rand(X_pos.shape[1])
            if aid >= np_pos:
                gap = gap / 2.0
            generated[i] = sample + gap * (data[aid] - sample)
        return generated

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) <= 1:
            return X, y
        target = int(np.max(counts))
        order = np.argsort(counts)
        if self.iteration_order == "descending":
            order = order[::-1]
        work_X = X.copy()
        work_y = y.copy()
        original_order = classes.tolist()

        for cls in classes[order]:
            cls = int(cls)
            X_pos = work_X[work_y == cls]
            need = target - len(X_pos)
            if need <= 0:
                continue
            X_neg = work_X[work_y != cls]
            gen = self._generate_binary(X_pos, X_neg, need, rng)
            if len(gen) > 0:
                work_X = np.vstack([work_X, gen])
                work_y = np.concatenate([work_y, np.full(len(gen), cls, dtype=y.dtype)])

        X_parts, y_parts = [], []
        for cls in original_order:
            Xc = work_X[work_y == cls]
            X_parts.append(Xc)
            y_parts.append(np.full(len(Xc), cls, dtype=y.dtype))
        return np.vstack(X_parts), np.concatenate(y_parts)


def build(random_state=42, smoke=False, rf_n_estimators=30, n_jobs=-1, **kw):
    return OREMM(
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
