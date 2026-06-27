"""MC-CCR — Multi-class Combined Cleaning & Resampling (Koziarski, Woźniak,
Krawczyk, KBS 2020).

"Combined cleaning and resampling algorithm for multi-class imbalanced data
with label noise."

Mechanism (handles label noise):
  1. For each minority observation, expand a p-norm sphere with a fixed ENERGY
     budget. Expansion costs 1 per unit radius before reaching majority points,
     then the marginal cost increases with each enclosed majority observation.
  2. CLEANING: majority observations enclosed by the sphere are either
     translated outside the sphere, removed, or ignored depending on the paper
     switch.
  3. OVERSAMPLING: synthetic minority observations are generated inside each
     minority sphere. The number of samples assigned to a sphere is
     proportional to 1 / radius (or drawn randomly from the available spheres).
  4. MULTI-CLASS: classes are processed in descending-size order using the
     dedicated `sampling` or `complete` strategy from the authors' reference
     implementation, preserving inter-class relationships better than a naive
     one-vs-rest pass.
"""

import os
import sys
import numpy as np
from scipy.spatial import distance_matrix

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common import smoke as _smoke

MODEL_KEY = "mc_ccr"


class MCCR(Resampler):
    def __init__(
        self, energy=0.25, cleaning_strategy="translate", selection_strategy="proportional", p_norm=1.0, method="sampling", rf_n_estimators=30, n_jobs=-1, random_state=42, **kw
    ):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        if cleaning_strategy not in {"ignore", "translate", "remove"}:
            raise ValueError("cleaning_strategy must be one of ignore/translate/remove")
        if selection_strategy not in {"proportional", "random"}:
            raise ValueError("selection_strategy must be one of proportional/random")
        if method not in {"sampling", "complete"}:
            raise ValueError("method must be one of sampling/complete")
        self.energy = float(energy)
        self.cleaning_strategy = cleaning_strategy
        self.selection_strategy = selection_strategy
        self.p_norm = float(p_norm)
        self.method = method

    def _distance(self, x, y):
        return float(np.sum(np.abs(x - y) ** self.p_norm) ** (1.0 / self.p_norm))

    def _sample_inside_sphere(self, d, radius, rng):
        if radius <= 0:
            return np.zeros(d, dtype=float)
        direction = (2.0 * rng.rand(d) - 1.0).astype(float)
        norm = self._distance(direction, np.zeros(d, dtype=float))
        while norm < 1e-20:
            direction = (2.0 * rng.rand(d) - 1.0).astype(float)
            norm = self._distance(direction, np.zeros(d, dtype=float))
        return direction / norm * rng.rand() * radius

    def _binary_ccr(self, X, y, minority_class, n, rng):
        minority_points = X[y == minority_class].astype(float).copy()
        majority_points = X[y != minority_class].astype(float).copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()
        if len(minority_points) == 0:
            return X.copy(), y.copy()
        if n <= 0:
            return X.copy(), y.copy()

        distances = distance_matrix(minority_points, majority_points, self.p_norm)
        radii = np.zeros(len(minority_points), dtype=float)
        translations = np.zeros_like(majority_points, dtype=float)
        kept = np.ones(len(majority_points), dtype=bool)

        for i, minority_point in enumerate(minority_points):
            remaining_energy = self.energy
            radius = 0.0
            order = np.argsort(distances[i])
            n_inside = 0
            while True:
                if n_inside == len(majority_points):
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
            radii[i] = radius

            for j in range(n_inside):
                idx = order[j]
                majority_point = majority_points[idx].copy()
                d = distances[i, idx]
                while d < 1e-20:
                    jitter = 1e-6 * rng.rand(len(majority_point)) + 1e-6
                    majority_point = majority_point + jitter * rng.choice([-1.0, 1.0], len(majority_point))
                    d = self._distance(minority_point, majority_point)
                translation = (radius - d) / d * (majority_point - minority_point)
                translations[idx] += translation
                kept[idx] = False

        if self.cleaning_strategy == "translate":
            majority_points = majority_points + translations
        elif self.cleaning_strategy == "remove":
            majority_points = majority_points[kept]
            majority_labels = majority_labels[kept]

        appended = []
        if self.selection_strategy == "proportional":
            inv = 1.0 / np.clip(radii, 1e-12, None)
            denom = np.clip(inv.sum(), 1e-12, None)
            for i, minority_point in enumerate(minority_points):
                n_syn = int(np.round(inv[i] / denom * n))
                for _ in range(n_syn):
                    appended.append(minority_point + self._sample_inside_sphere(X.shape[1], radii[i], rng))
        else:
            choice = rng.choice(np.arange(len(minority_points)), size=n, replace=True)
            for i in choice:
                appended.append(minority_points[i] + self._sample_inside_sphere(X.shape[1], radii[i], rng))

        if appended:
            appended = np.asarray(appended, dtype=float)
            points = np.concatenate([majority_points, minority_points, appended], axis=0)
            labels = np.concatenate(
                [
                    majority_labels,
                    minority_labels,
                    np.full(len(appended), minority_class, dtype=y.dtype),
                ]
            )
        else:
            points = np.concatenate([majority_points, minority_points], axis=0)
            labels = np.concatenate([majority_labels, minority_labels], axis=0)
        return points, labels

    @staticmethod
    def _unpack_observations(observations):
        pts = []
        lab = []
        for cls, arr in observations.items():
            arr = np.asarray(arr, dtype=float)
            if len(arr) == 0:
                continue
            pts.append(arr)
            lab.append(np.full(len(arr), cls))
        return np.concatenate(pts, axis=0), np.concatenate(lab, axis=0)

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        classes = np.unique(y)
        sizes = np.array([(y == c).sum() for c in classes], dtype=int)
        order = np.argsort(sizes)[::-1]
        classes = classes[order]
        observations = {c: X[y == c].copy() for c in classes}
        n_max = int(max(sizes))

        if self.method == "sampling":
            for i in range(1, len(classes)):
                current_class = int(classes[i])
                n = n_max - len(observations[current_class])
                if n <= 0:
                    continue
                used = {}
                unused = {}
                for j in range(i):
                    cls = int(classes[j])
                    arr = np.asarray(observations[cls], dtype=float)
                    all_idx = np.arange(len(arr))
                    take = min(len(arr), int(n_max / i))
                    if take > 0:
                        picked = rng.choice(all_idx, size=take, replace=False)
                    else:
                        picked = np.empty(0, dtype=int)
                    mask = np.zeros(len(arr), dtype=bool)
                    mask[picked] = True
                    used[cls] = arr[mask]
                    unused[cls] = arr[~mask]
                used[current_class] = np.asarray(observations[current_class], dtype=float)
                unused[current_class] = np.empty((0, X.shape[1]), dtype=float)
                for j in range(i + 1, len(classes)):
                    cls = int(classes[j])
                    used[cls] = np.empty((0, X.shape[1]), dtype=float)
                    unused[cls] = np.asarray(observations[cls], dtype=float)
                pts, lab = self._unpack_observations(used)
                pts_os, lab_os = self._binary_ccr(pts, lab, current_class, n, rng)
                observations = {}
                for cls in classes:
                    cls = int(cls)
                    over = pts_os[lab_os == cls]
                    rest = np.asarray(unused[cls], dtype=float)
                    if len(over) == 0:
                        observations[cls] = rest
                    elif len(rest) == 0:
                        observations[cls] = over
                    else:
                        observations[cls] = np.concatenate([over, rest], axis=0)
        else:
            for i in range(1, len(classes)):
                current_class = int(classes[i])
                n = n_max - len(observations[current_class])
                if n <= 0:
                    continue
                pts, lab = self._unpack_observations(observations)
                pts_os, lab_os = self._binary_ccr(pts, lab, current_class, n, rng)
                observations = {int(cls): pts_os[lab_os == cls] for cls in classes}
        return self._unpack_observations(observations)


def build(random_state=42, smoke=False, energy=0.25, rf_n_estimators=30, n_jobs=-1, **kw):
    return MCCR(
        energy=energy,
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
