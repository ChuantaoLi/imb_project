"""MC-NRO — Multi-class Neighborhood Repartition Oversampling (Shen et al.,
Neurocomputing 2024).

"Neighborhood repartition-based oversampling algorithm for multiclass imbalanced
data with label noise."

Mechanism (label-noise robust; combines MC-CCR cleaning with MC-RBO generation):
  1. Use MC-CCR-style energy-driven neighbourhood expansion to translate
     other-class observations away from minority observations.
  2. Repartition local neighbourhoods after translation and delete translated
     observations that still behave as overlap / label-noise / outliers in the
     minority neighbourhood.
  3. Generate new minority observations radially, but keep only candidates that
     fall into favourable subregions according to the MUTUAL POTENTIAL of the
     current class and its local rival classes.
"""
import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common.neighbors import knn_indices
from common import smoke as _smoke

MODEL_KEY = "mc_nro"


class MCNRO(Resampler):
    def __init__(self, energy=0.25, spread=None, k=5, p_norm=1.0,
                 potential_tau=1.0, rf_n_estimators=30, n_jobs=-1,
                 random_state=42, **kw):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        self.energy = float(energy)
        self.spread = spread
        self.k = int(k)
        self.p_norm = float(p_norm)
        self.potential_tau = float(potential_tau)

    def _distance(self, x, y):
        return float(np.sum(np.abs(x - y) ** self.p_norm) ** (1.0 / self.p_norm))

    def _class_sigma(self, Xc):
        Xc = np.asarray(Xc, dtype=float)
        if len(Xc) <= 1:
            return 1.0
        nn = NearestNeighbors(n_neighbors=min(2, len(Xc))).fit(Xc)
        dd, _ = nn.kneighbors(Xc)
        return float(np.mean(dd[:, 1]))

    def _seed_sigmas(self, Xc):
        Xc = np.asarray(Xc, dtype=float)
        if len(Xc) <= 1:
            return np.ones(len(Xc), dtype=float)
        nn = NearestNeighbors(n_neighbors=min(2, len(Xc))).fit(Xc)
        dd, _ = nn.kneighbors(Xc)
        return np.clip(dd[:, 1].astype(float), 1e-12, None)

    def _translate_around_class(self, X, y, cls, rng):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        minority_idx = np.where(y == cls)[0]
        other_idx = np.where(y != cls)[0]
        if len(minority_idx) == 0 or len(other_idx) == 0:
            return X.copy()

        minority_points = X[minority_idx]
        other_points = X[other_idx].copy()
        distances = distance_matrix(minority_points, other_points, self.p_norm)
        translations = np.zeros_like(other_points, dtype=float)

        for i, minority_point in enumerate(minority_points):
            remaining_energy = self.energy
            radius = 0.0
            order = np.argsort(distances[i])
            n_inside = 0
            while True:
                if n_inside == len(other_points):
                    denom = (n_inside + 1) if n_inside == 0 else n_inside
                    radius += remaining_energy / denom
                    break
                radius_change = remaining_energy / (n_inside + 1)
                d_next = distances[i, order[n_inside]]
                if d_next >= radius + radius_change:
                    radius += radius_change
                    break
                last_distance = 0.0 if n_inside == 0 else distances[i, order[n_inside - 1]]
                radius_change = d_next - last_distance
                radius += radius_change
                remaining_energy -= radius_change * (n_inside + 1)
                n_inside += 1

            for j in range(n_inside):
                idx = order[j]
                other_point = other_points[idx].copy()
                d = distances[i, idx]
                while d < 1e-20:
                    jitter = 1e-6 * rng.rand(len(other_point)) + 1e-6
                    other_point = other_point + jitter * rng.choice([-1.0, 1.0], len(other_point))
                    d = self._distance(minority_point, other_point)
                translations[idx] += (radius - d) / d * (other_point - minority_point)

        X_new = X.copy()
        X_new[other_idx] = other_points + translations
        return X_new

    def _repartition_delete(self, X, y, focus_class):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        kk = min(self.k, max(0, len(X) - 1))
        if kk <= 0:
            return X, y
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
        _, idx = nn.kneighbors(X)
        idx = idx[:, 1:]
        keep = np.ones(len(X), dtype=bool)
        other_idx = np.where(y != focus_class)[0]
        for i in other_idx:
            nb = idx[i]
            nb_lab = y[nb]
            own = int((nb_lab == y[i]).sum())
            focus = int((nb_lab == focus_class).sum())
            closest_lab = y[nb[0]] if len(nb) else y[i]
            # Repartitioned neighbourhood still belongs to the minority region:
            # remove likely overlap / label-noise / outlier observations.
            if focus > own or (focus >= max(1, kk // 2) and closest_lab == focus_class):
                keep[i] = False
        return X[keep], y[keep]

    def _local_rival_class(self, x, cls, X, y):
        mask_other = y != cls
        X_other = X[mask_other]
        y_other = y[mask_other]
        if len(X_other) == 0:
            return None
        nn = NearestNeighbors(n_neighbors=1).fit(X_other)
        _, idx = nn.kneighbors(x.reshape(1, -1))
        return y_other[idx[0, 0]]

    def _mutual_potential(self, x, Xsame, Xother, sigma_same, sigma_other):
        sigma_same = max(float(sigma_same), 1e-12)
        sigma_other = max(float(sigma_other), 1e-12)
        ds = np.sum((Xsame - x.reshape(1, -1)) ** 2, axis=1)
        do = np.sum((Xother - x.reshape(1, -1)) ** 2, axis=1)
        ps = float(np.exp(-ds / (2.0 * sigma_same ** 2)).sum())
        po = float(np.exp(-do / (2.0 * sigma_other ** 2)).sum())
        return ps, po

    def _candidate_ok(self, cand, cls, X, y, sigma_same):
        rival = self._local_rival_class(cand, cls, X, y)
        if rival is None:
            return True
        Xsame = X[y == cls]
        Xrival = X[y == rival]
        sigma_rival = self._class_sigma(Xrival)
        ps, pr = self._mutual_potential(cand, Xsame, Xrival, sigma_same, sigma_rival)
        if ps <= self.potential_tau * pr:
            return False

        kk = min(self.k, len(X))
        if kk <= 0:
            return True
        nn = NearestNeighbors(n_neighbors=kk).fit(X)
        _, idx = nn.kneighbors(cand.reshape(1, -1))
        labs = y[idx[0]]
        same = int((labs == cls).sum())
        rival_n = int((labs == rival).sum())
        return same >= rival_n

    def _oversample_class(self, X, y, cls, n_gen, rng):
        Xc = np.asarray(X[y == cls], dtype=float)
        if n_gen <= 0 or len(Xc) == 0:
            return np.empty((0, X.shape[1]), dtype=float)
        if len(Xc) == 1:
            return np.repeat(Xc, n_gen, axis=0)

        nn_all = knn_indices(X, self.k)
        idx_cls = np.where(y == cls)[0]
        same_counts = np.zeros(len(idx_cls), dtype=float)
        for pos, global_idx in enumerate(idx_cls):
            nb = nn_all[global_idx]
            if len(nb) == 0:
                same_counts[pos] = float(self.k)
            else:
                same_counts[pos] = float((y[nb] == cls).sum())

        kk = max(1, min(self.k, max(1, len(X) - 1)))
        border = (same_counts > 0) & (same_counts < kk)
        safe = same_counts == kk
        weights = np.where(border, 2.0, np.where(safe, 1.0, 0.0))
        if weights.sum() <= 0:
            weights = np.ones(len(Xc), dtype=float)
        weights = weights / np.clip(weights.sum(), 1e-12, None)

        seed_sigma = self._seed_sigmas(Xc)
        class_sigma = self._class_sigma(Xc) if self.spread is None else float(self.spread)
        generated = []
        attempts = 0
        max_attempts = max(50, 20 * n_gen)
        while len(generated) < n_gen and attempts < max_attempts:
            attempts += 1
            sid = rng.choice(np.arange(len(Xc)), p=weights)
            sigma = seed_sigma[sid] if self.spread is None else float(self.spread)
            sigma = max(sigma, 1e-12)
            cand = Xc[sid] + rng.normal(0.0, sigma, size=X.shape[1])
            if self._candidate_ok(cand, cls, X, y, max(class_sigma, sigma)):
                generated.append(cand)

        if len(generated) < n_gen:
            fallback = Xc[rng.choice(np.arange(len(Xc)), size=n_gen - len(generated), replace=True)]
            for base in fallback:
                generated.append(base + rng.normal(0.0, class_sigma, size=X.shape[1]))
        return np.asarray(generated, dtype=float)

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        order = np.argsort(counts)[::-1]
        classes = classes[order]

        Xw, yw = X.copy(), y.copy()
        for cls in classes[1:]:
            Xw = self._translate_around_class(Xw, yw, cls, rng)
            Xw, yw = self._repartition_delete(Xw, yw, cls)

        counts_after = {c: int((yw == c).sum()) for c in np.unique(yw)}
        target = max(counts_after.values()) if counts_after else len(yw)
        X_parts = [Xw]
        y_parts = [yw]
        for cls in classes:
            current = counts_after.get(cls, 0)
            need = target - current
            if need > 0:
                gen = self._oversample_class(Xw, yw, cls, need, rng)
                if len(gen) > 0:
                    X_parts.append(gen)
                    y_parts.append(np.full(len(gen), cls, dtype=y.dtype))
        return np.vstack(X_parts), np.concatenate(y_parts)


def build(random_state=42, smoke=False, rf_n_estimators=30, n_jobs=-1, **kw):
    return MCNRO(
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
