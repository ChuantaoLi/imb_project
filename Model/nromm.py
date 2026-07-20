"""NROMM — Noise-Robust Oversampling for imbalanced data (Liu et al., PR 2023).

"Noise-robust oversampling for imbalanced data classification."

Mechanism:
  1. Use an adapted one-versus-ensemble decomposition for multiclass imbalance:
     when oversampling a class, all larger classes are merged into the current
     majority side.
  2. Inside each binary subproblem, minority and majority samples are grouped
     into INLAND / BORDERLINE / TRAPPED regions via a CFSFDP-style density-peak
     clustering plus local cross-class safety analysis.
  3. Majority borderline / trapped samples are cleaned, while minority samples
     are generated with group-specific rules; trapped samples use adaptive
     embedding to move generation toward safer minority supports.
  4. A safe boundary rejects candidates that would fall closer to the majority
     side than to the minority manifold.
"""
import os
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common.neighbors import knn_indices
from common import smoke as _smoke

MODEL_KEY = "nromm"

INLAND, BORDERLINE, TRAPPED = 0, 1, 2


def _pairwise_distances(X):
    X = np.asarray(X, dtype=float)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


class NROMM(Resampler):
    def __init__(
        self,
        k=5,
        density_percentile=20.0,
        safe_margin=1.05,
        max_ratio=20,
        rf_n_estimators=30,
        n_jobs=-1,
        random_state=42,
        **kw
    ):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        self.k = int(k)
        self.density_percentile = float(density_percentile)
        self.safe_margin = float(safe_margin)
        self.max_ratio = int(max_ratio)

    def _density_peak_clusters(self, Xc):
        Xc = np.asarray(Xc, dtype=float)
        n = len(Xc)
        if n <= 2:
            return np.zeros(n, dtype=int), np.ones(n, dtype=float), np.ones(n, dtype=float)
        dist = _pairwise_distances(Xc)
        tri = dist[np.triu_indices(n, 1)]
        tri = tri[tri > 0]
        dc = np.percentile(tri, self.density_percentile) if len(tri) else 1.0
        dc = max(float(dc), 1e-12)
        rho = np.exp(-((dist / dc) ** 2)).sum(axis=1) - 1.0
        order = np.argsort(-rho)
        delta = np.zeros(n, dtype=float)
        parent = np.full(n, -1, dtype=int)
        delta[order[0]] = float(np.max(dist[order[0]])) if len(dist) else 1.0
        for rank in range(1, n):
            idx = order[rank]
            higher = order[:rank]
            d = dist[idx, higher]
            best = int(np.argmin(d))
            delta[idx] = float(d[best])
            parent[idx] = int(higher[best])

        gamma = rho * delta
        n_centers = max(1, int(np.ceil(np.sqrt(n / 2.0))))
        center_idx = np.argsort(-gamma)[:n_centers]
        labels = np.full(n, -1, dtype=int)
        for cid, idx in enumerate(center_idx):
            labels[idx] = cid
        for idx in order:
            if labels[idx] >= 0:
                continue
            p = parent[idx]
            while p >= 0 and labels[p] < 0:
                p = parent[p]
            labels[idx] = 0 if p < 0 else labels[p]
        return labels, rho, delta

    def _group_points(self, X_pos, X_neg):
        X_pos = np.asarray(X_pos, dtype=float)
        X_neg = np.asarray(X_neg, dtype=float)
        n_pos = len(X_pos)
        if n_pos == 0:
            empty = np.empty(0, dtype=int)
            return empty, {"purity": np.empty(0), "same_dist": np.empty(0), "other_dist": np.empty(0)}

        labels, _, _ = self._density_peak_clusters(X_pos)
        same_idx = knn_indices(X_pos, self.k)
        if same_idx.shape[1] == 0:
            same_dist = np.full(n_pos, 1.0, dtype=float)
        else:
            same_dist = np.linalg.norm(X_pos - X_pos[same_idx[:, 0]], axis=1)

        if len(X_neg) == 0:
            other_dist = np.full(n_pos, np.inf, dtype=float)
            purity = np.ones(n_pos, dtype=float)
        else:
            nn_other = NearestNeighbors(n_neighbors=1).fit(X_neg)
            other_dist = nn_other.kneighbors(X_pos, return_distance=True)[0][:, 0]

            Xmix = np.vstack([X_pos, X_neg])
            ymix = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(len(X_neg), dtype=int)])
            kk = min(max(1, self.k), len(Xmix) - 1)
            nn_mix = NearestNeighbors(n_neighbors=kk + 1).fit(Xmix)
            idx = nn_mix.kneighbors(X_pos, return_distance=False)[:, 1:]
            purity = np.mean(ymix[idx] == 1, axis=1)

        margin = other_dist - same_dist
        cluster_margin = {}
        for lab in np.unique(labels):
            cluster_margin[int(lab)] = float(np.mean(margin[labels == lab]))

        groups = np.full(n_pos, BORDERLINE, dtype=int)
        for i in range(n_pos):
            cm = cluster_margin[int(labels[i])]
            if purity[i] >= 0.75 and margin[i] > 0 and cm >= 0:
                groups[i] = INLAND
            elif purity[i] <= 0.25 or margin[i] <= 0 or (cm < 0 and purity[i] < 0.5):
                groups[i] = TRAPPED
            else:
                groups[i] = BORDERLINE
        return groups, {"purity": purity, "same_dist": same_dist, "other_dist": other_dist}

    def _project_within_boundary(self, seed, cand, X_neg):
        if len(X_neg) == 0:
            return cand
        radius = np.min(np.linalg.norm(X_neg - seed.reshape(1, -1), axis=1))
        radius = max(radius * 0.95, 1e-12)
        direction = cand - seed
        norm = np.linalg.norm(direction)
        if norm > radius:
            cand = seed + direction / norm * radius
        return cand

    def _candidate_ok(self, cand, X_pos, X_neg):
        if len(X_neg) == 0:
            return True
        d_neg = float(np.min(np.linalg.norm(X_neg - cand.reshape(1, -1), axis=1)))
        d_pos = float(np.min(np.linalg.norm(X_pos - cand.reshape(1, -1), axis=1)))
        return d_neg > self.safe_margin * d_pos

    def _adaptive_embed(self, seed, anchors, rng):
        if len(anchors) == 0:
            return seed.copy()
        anchor = anchors[rng.randint(len(anchors))]
        lam = 0.5 + 0.5 * rng.rand()
        return seed + lam * (anchor - seed)

    def _generate_for_seed(self, seed, group_id, X_pos, groups, X_neg, rng):
        same_idx = np.arange(len(X_pos))
        if len(X_pos) == 1:
            return seed.copy()

        inland = X_pos[groups == INLAND]
        non_trapped = X_pos[groups != TRAPPED]
        neighbors = X_pos[knn_indices(X_pos, self.k)[np.where(np.all(X_pos == seed, axis=1))[0][0]]]

        if group_id == TRAPPED:
            embed = self._adaptive_embed(seed, non_trapped if len(non_trapped) else X_pos, rng)
            support_pool = inland if len(inland) else non_trapped
            if len(support_pool) == 0:
                support_pool = X_pos
            support = support_pool[rng.randint(len(support_pool))]
            cand = embed + rng.rand() * (support - embed)
        elif group_id == BORDERLINE:
            support_pool = non_trapped if len(non_trapped) else X_pos
            support = support_pool[rng.randint(len(support_pool))]
            cand = seed + rng.rand() * (support - seed)
        else:
            support_pool = inland if len(inland) else X_pos
            support = support_pool[rng.randint(len(support_pool))]
            cand = seed + rng.rand() * (support - seed)

        if len(neighbors) > 0:
            nb = neighbors[rng.randint(len(neighbors))]
            cand = 0.5 * cand + 0.5 * (seed + rng.rand() * (nb - seed))
        return self._project_within_boundary(seed, cand, X_neg)

    def _oversample_minority(self, X_pos, X_neg, n_gen, rng):
        if n_gen <= 0 or len(X_pos) == 0:
            return np.empty((0, X_pos.shape[1] if X_pos.ndim == 2 else 0), dtype=float)
        if len(X_pos) == 1:
            return np.repeat(X_pos, n_gen, axis=0)

        groups, _ = self._group_points(X_pos, X_neg)
        weights = np.choose(groups, [0.5, 1.0, 1.5]).astype(float)
        weights = weights / np.sum(weights)
        generated = []
        attempts = 0
        max_attempts = n_gen * self.max_ratio + 50
        while len(generated) < n_gen and attempts < max_attempts:
            attempts += 1
            sid = rng.choice(np.arange(len(X_pos)), p=weights)
            cand = self._generate_for_seed(X_pos[sid], int(groups[sid]), X_pos, groups, X_neg, rng)
            if self._candidate_ok(cand, X_pos, X_neg):
                generated.append(cand)

        if len(generated) < n_gen:
            fall = []
            deficit = n_gen - len(generated)
            same_idx = knn_indices(X_pos, self.k)
            kk = same_idx.shape[1]
            if kk <= 0:
                fall = np.repeat(X_pos[:1], deficit, axis=0)
            else:
                seeds = rng.choice(np.arange(len(X_pos)), size=deficit, replace=True, p=weights)
                nbs = same_idx[seeds, rng.randint(0, kk, size=deficit)]
                gaps = rng.rand(deficit, 1)
                for s, nb, gap in zip(seeds, nbs, gaps):
                    cand = X_pos[s] + float(gap) * (X_pos[nb] - X_pos[s])
                    cand = self._project_within_boundary(X_pos[s], cand, X_neg)
                    fall.append(cand)
                fall = np.asarray(fall, dtype=float)
            generated.extend(list(fall))
        return np.asarray(generated[:n_gen], dtype=float)

    def _binary_clean_majority(self, X_maj, X_min):
        if len(X_maj) == 0:
            return X_maj
        maj_groups, _ = self._group_points(X_maj, X_min)
        keep = maj_groups == INLAND
        if not np.any(keep):
            return X_maj
        return X_maj[keep]

    def _remove_overlap_inland(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        keep = np.ones(len(X), dtype=bool)
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            other = np.where(y != cls)[0]
            if len(idx) == 0 or len(other) == 0:
                continue
            groups, stats = self._group_points(X[idx], X[other])
            overlap = stats["other_dist"] < stats["same_dist"]
            local_drop = (groups == INLAND) & overlap
            if np.all(local_drop):
                continue
            keep[idx[local_drop]] = False
        if np.sum(keep) < 2:
            return X, y
        return X[keep], y[keep]

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        classes, counts = np.unique(y, return_counts=True)
        order = np.argsort(counts)[::-1]
        classes = classes[order]
        counts = counts[order]
        target = int(counts[0])
        class_data = {int(c): X[y == c].copy() for c in classes}

        # one-versus-ensemble: oversample each class against all currently larger classes
        for i in range(1, len(classes)):
            cls = int(classes[i])
            X_pos = np.asarray(class_data[cls], dtype=float)
            need = target - len(X_pos)
            if need <= 0:
                continue
            larger = [int(c) for c in classes[:i]]
            neg_parts = []
            for major_cls in larger:
                cleaned = self._binary_clean_majority(np.asarray(class_data[major_cls], dtype=float), X_pos)
                neg_parts.append(cleaned)
            X_neg = np.vstack(neg_parts) if len(neg_parts) else np.empty((0, X.shape[1]), dtype=float)
            gen = self._oversample_minority(X_pos, X_neg, need, rng)
            if len(gen) > 0:
                class_data[cls] = np.vstack([X_pos, gen])

        X_parts, y_parts = [], []
        for cls in classes:
            cls = int(cls)
            arr = np.asarray(class_data[cls], dtype=float)
            X_parts.append(arr)
            y_parts.append(np.full(len(arr), cls, dtype=y.dtype))
        X_res = np.vstack(X_parts)
        y_res = np.concatenate(y_parts)
        return self._remove_overlap_inland(X_res, y_res)


def build(random_state=42, smoke=False, k=5, rf_n_estimators=30, n_jobs=-1, **kw):
    return NROMM(
        k=k,
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
