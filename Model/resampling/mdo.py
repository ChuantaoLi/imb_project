"""MDO — Mahalanobis Distance Oversampling in PC space (Abdi & Hashemi, TKDE 2016).

"To combat multi-class imbalanced problems by means of over-sampling techniques."

Mechanism:
  1. Select minority seeds according to the paper's local-neighbourhood rule.
  2. Move the selected minority subset into principal-component space.
  3. For every generated sample, preserve the selected seed's MAHALANOBIS
     distance from the minority mean, i.e. generate on the same probability
     contour rather than sampling blindly from a global Gaussian.
  4. Transform the generated point back to the original feature space.
"""
import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common import smoke as _smoke

MODEL_KEY = "mdo"


class MDO(Resampler):
    def __init__(
        self,
        k2=5,
        k1_frac=0.5,
        variance_floor=1e-3,
        shrinkage=1e-3,
        rf_n_estimators=30,
        n_jobs=-1,
        random_state=42,
        **kw
    ):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        self.k2 = int(k2)
        self.k1_frac = float(k1_frac)
        self.variance_floor = float(variance_floor)
        # kept for backward compatibility with the previous simplified version
        self.shrinkage = float(shrinkage)

    def _generate_pc_sample(self, alpha, variances, rng):
        alpha_v = alpha * variances
        alpha_v = np.maximum(alpha_v, self.variance_floor)
        x_new = np.zeros_like(variances, dtype=float)
        accum = 0.0
        for j in range(len(variances) - 1):
            value = 2.0 * rng.rand() - 1.0
            value = value * np.sqrt(alpha_v[j])
            x_new[j] = value
            accum += (value ** 2) / alpha_v[j]
        last = np.sqrt(max((1.0 - accum) * alpha * variances[-1], 0.0)) * (accum <= 1.0)
        x_new[-1] = (2.0 * rng.rand() - 1.0) * last
        return x_new

    def _generate_samples(self, X_sel, weights, n_to_sample, rng):
        mu = np.mean(X_sel, axis=0)
        centered = X_sel - mu
        n_components = min(centered.shape[0], centered.shape[1])
        if n_components <= 0:
            return np.empty((0, X_sel.shape[1]), dtype=float)
        pca = PCA(n_components=n_components, random_state=self.random_state)
        transformed = pca.fit_transform(centered)
        variances = np.var(transformed, axis=0).astype(float)
        variances = np.maximum(variances, self.variance_floor)

        samples = []
        while len(samples) < n_to_sample:
            idx = rng.choice(np.arange(len(X_sel)), p=weights)
            alpha = float(np.sum((transformed[idx] ** 2) / variances))
            samples.append(self._generate_pc_sample(alpha, variances, rng))
        return pca.inverse_transform(np.vstack(samples)) + mu

    def _oversample_class(self, X, y, cls, target, rng):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        Xc = X[y == cls]
        n_to_sample = int(target - len(Xc))
        if n_to_sample <= 0 or len(Xc) == 0:
            return np.empty((0, X.shape[1]), dtype=float)
        if len(Xc) == 1:
            return np.repeat(Xc, n_to_sample, axis=0)

        k1 = max(1, int(self.k2 * self.k1_frac))
        k1 = min(k1, max(1, len(X) - 1))
        k2 = min(self.k2 + 1, len(X))
        nn = NearestNeighbors(n_neighbors=k2)
        nn.fit(X)
        ind = nn.kneighbors(Xc, return_distance=False)
        n_min = np.sum(y[ind[:, 1:]] == cls, axis=1)
        selected_mask = n_min >= k1
        X_sel = Xc[selected_mask]
        if len(X_sel) == 0:
            # paper implementation falls back to keeping data unchanged when
            # all minority samples are treated as noise; here we degrade
            # gracefully to the original class points as selectable seeds.
            X_sel = Xc
            n_min = np.maximum(n_min, 1)
            selected_mask = np.ones(len(Xc), dtype=bool)

        weights = n_min[selected_mask].astype(float) / max(float(k2), 1.0)
        if np.sum(weights) <= 0:
            weights = np.ones(len(X_sel), dtype=float)
        weights = weights / np.sum(weights)
        return self._generate_samples(X_sel, weights, n_to_sample, rng)

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        classes, counts = np.unique(y, return_counts=True)
        target = int(np.max(counts))
        majority_class = int(classes[np.argmax(counts)])
        Xl, yl = [X[y == majority_class]], [y[y == majority_class]]
        for c in classes:
            if int(c) == majority_class:
                continue
            Xc = X[y == c]
            gen = self._oversample_class(X, y, int(c), target, rng)
            if len(gen) > 0:
                Xl.append(gen)
                yl.append(np.full(len(gen), c, dtype=y.dtype))
            Xl.append(Xc)
            yl.append(np.full(len(Xc), c, dtype=y.dtype))
        return np.vstack(Xl), np.concatenate(yl)


def build(random_state=42, smoke=False, rf_n_estimators=30, n_jobs=-1, **kw):
    return MDO(rf_n_estimators=rf_n_estimators, n_jobs=n_jobs, random_state=random_state, **kw)


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
