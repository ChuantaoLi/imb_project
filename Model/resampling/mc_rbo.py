"""MC-RBO — Multi-class Radial-Based Oversampling (Krawczyk, Koziarski, Woźniak,
TNNLS 2020).

"Radial-based oversampling for multiclass imbalanced data classification."

Mechanism:
  1. Treat every minority observation as the centre of an RBF potential field.
  2. For each requested synthetic point, choose a minority seed and iteratively
     move it along axis-aligned directions so that the ABSOLUTE value of the
     mutual class potential decreases.
  3. The mutual class potential is computed from both minority and other-class
     observations, so generation is guided toward regions where the mutual class
     distribution is small.
  4. Multi-class handling follows the authors' dedicated `sampling` / `complete`
     strategies rather than a naive per-class Gaussian perturbation.
"""
import os
import sys
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common import smoke as _smoke

MODEL_KEY = "mc_rbo"


class MCRBO(Resampler):
    def __init__(
        self,
        gamma=0.05,
        step_size=0.001,
        n_steps=500,
        approximate_potential=True,
        n_nearest_neighbors=25,
        method="sampling",
        spread=None,
        rf_n_estimators=30,
        n_jobs=-1,
        random_state=42,
        **kw
    ):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        if method not in {"sampling", "complete"}:
            raise ValueError("method must be one of sampling/complete")
        # `spread` is kept as a compatibility alias for the paper's gamma width.
        if spread is not None:
            gamma = spread
        self.gamma = float(gamma)
        self.step_size = float(step_size)
        self.n_steps = int(n_steps)
        self.approximate_potential = bool(approximate_potential)
        self.n_nearest_neighbors = int(n_nearest_neighbors)
        self.method = method

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

    @staticmethod
    def _distance(x, y, p_norm=1.0):
        return float(np.sum(np.abs(x - y) ** p_norm) ** (1.0 / p_norm))

    @staticmethod
    def _rbf(distance_value, gamma):
        gamma = float(gamma)
        if gamma == 0.0:
            return 0.0
        return float(np.exp(-((distance_value / gamma) ** 2)))

    def _mutual_class_potential(self, point, other_points, minority_points):
        potential = 0.0
        for other_point in other_points:
            potential += self._rbf(self._distance(point, other_point), self.gamma)
        for minority_point in minority_points:
            potential -= self._rbf(self._distance(point, minority_point), self.gamma)
        return float(potential)

    def _possible_directions(self, n_dimensions, excluded_direction=None, rng=None):
        directions = []
        for dimension in range(n_dimensions):
            for sign in (-1.0, 1.0):
                if excluded_direction is None or excluded_direction != (dimension, sign):
                    directions.append((dimension, sign))
        if rng is None:
            rng = np.random.RandomState(self.random_state)
        rng.shuffle(directions)
        return directions

    def _local_support(self, point, X, y, minority_class, seed_global_idx=None):
        if not self.approximate_potential or len(X) <= 1:
            minority_points = X[y == minority_class]
            other_points = X[y != minority_class]
            return minority_points, other_points

        distances = np.sum(np.abs(X - point.reshape(1, -1)), axis=1).astype(float)
        if seed_global_idx is not None and 0 <= seed_global_idx < len(distances):
            distances[seed_global_idx] = np.inf
        else:
            same_point = np.where(np.all(np.isclose(X, point, atol=1e-12), axis=1))[0]
            if len(same_point) > 0:
                distances[same_point[0]] = np.inf
        k = min(self.n_nearest_neighbors + 1, len(X))
        idx = np.argsort(distances)[:k]
        closest_points = X[idx]
        closest_labels = y[idx]
        minority_points = closest_points[closest_labels == minority_class]
        other_points = closest_points[closest_labels != minority_class]
        if len(minority_points) == 0:
            minority_points = X[y == minority_class]
        if len(other_points) == 0:
            other_points = X[y != minority_class]
        return minority_points, other_points

    def _generate_from_seed(self, point, X, y, minority_class, rng, seed_global_idx=None):
        point = np.asarray(point, dtype=float)
        minority_points, other_points = self._local_support(
            point, X, y, minority_class, seed_global_idx=seed_global_idx
        )
        translation = np.zeros(len(point), dtype=float)
        potential = self._mutual_class_potential(point, other_points, minority_points)
        directions = self._possible_directions(len(point), rng=rng)

        for _ in range(self.n_steps):
            if len(directions) == 0:
                break
            dimension, sign = directions.pop()
            candidate_translation = translation.copy()
            candidate_translation[dimension] += sign * self.step_size
            candidate_potential = self._mutual_class_potential(
                point + candidate_translation, other_points, minority_points
            )
            if abs(candidate_potential) < abs(potential):
                translation = candidate_translation
                potential = candidate_potential
                directions = self._possible_directions(
                    len(point), excluded_direction=(dimension, -sign), rng=rng
                )
        return point + translation

    def _binary_rbo(self, X, y, minority_class, n, rng):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        minority_mask = y == minority_class
        minority_points = X[minority_mask]
        if len(minority_points) == 0 or n <= 0:
            return np.empty((0, X.shape[1]), dtype=float)

        minority_global_idx = np.where(minority_mask)[0]
        counts = np.zeros(len(minority_points), dtype=int)
        chosen = rng.choice(np.arange(len(minority_points)), size=n, replace=True)
        for idx in chosen:
            counts[idx] += 1

        appended = []
        for local_idx, point in enumerate(minority_points):
            if counts[local_idx] == 0:
                continue
            global_idx = int(minority_global_idx[local_idx])
            for _ in range(counts[local_idx]):
                appended.append(
                    self._generate_from_seed(
                        point, X, y, minority_class, rng, seed_global_idx=global_idx
                    )
                )
        if len(appended) == 0:
            return np.empty((0, X.shape[1]), dtype=float)
        return np.asarray(appended, dtype=float)

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        classes = np.unique(y)
        sizes = np.array([(y == c).sum() for c in classes], dtype=int)
        order = np.argsort(sizes)[::-1]
        classes = classes[order]
        observations = {int(c): X[y == c].copy() for c in classes}
        n_max = int(sizes[order][0]) if len(sizes) > 0 else len(y)

        if self.method == "sampling":
            for i in range(1, len(classes)):
                current_class = int(classes[i])
                current_obs = np.asarray(observations[current_class], dtype=float)
                n = n_max - len(current_obs)
                if n <= 0:
                    continue
                sampled = {current_class: current_obs}
                untouched = {current_class: np.empty((0, X.shape[1]), dtype=float)}
                for j in range(i):
                    cls = int(classes[j])
                    arr = np.asarray(observations[cls], dtype=float)
                    take = min(len(arr), int(n_max / i))
                    if take > 0:
                        picked = rng.choice(np.arange(len(arr)), size=take, replace=False)
                        mask = np.zeros(len(arr), dtype=bool)
                        mask[picked] = True
                        sampled[cls] = arr[mask]
                        untouched[cls] = arr[~mask]
                    else:
                        sampled[cls] = np.empty((0, X.shape[1]), dtype=float)
                        untouched[cls] = arr
                for j in range(i + 1, len(classes)):
                    cls = int(classes[j])
                    sampled[cls] = np.empty((0, X.shape[1]), dtype=float)
                    untouched[cls] = np.asarray(observations[cls], dtype=float)

                pts, lab = self._unpack_observations(sampled)
                appended = self._binary_rbo(pts, lab, current_class, n, rng)
                new_observations = {}
                for cls in classes:
                    cls = int(cls)
                    original = np.asarray(sampled[cls], dtype=float)
                    if cls == current_class and len(appended) > 0:
                        over = np.concatenate([original, appended], axis=0)
                    else:
                        over = original
                    rest = np.asarray(untouched[cls], dtype=float)
                    if len(over) == 0:
                        new_observations[cls] = rest
                    elif len(rest) == 0:
                        new_observations[cls] = over
                    else:
                        new_observations[cls] = np.concatenate([over, rest], axis=0)
                observations = new_observations
        else:
            for i in range(1, len(classes)):
                current_class = int(classes[i])
                n = n_max - len(observations[current_class])
                if n <= 0:
                    continue
                pts, lab = self._unpack_observations(observations)
                appended = self._binary_rbo(pts, lab, current_class, n, rng)
                if len(appended) > 0:
                    observations[current_class] = np.concatenate(
                        [np.asarray(observations[current_class], dtype=float), appended], axis=0
                    )
        return self._unpack_observations(observations)


def build(
    random_state=42,
    smoke=False,
    gamma=0.05,
    step_size=0.001,
    n_steps=500,
    rf_n_estimators=30,
    n_jobs=-1,
    **kw
):
    return MCRBO(
        gamma=gamma,
        step_size=step_size,
        n_steps=n_steps,
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
