"""ILMNN: Imbalance Large Margin Nearest Neighbor plus oversampling.

Faithful implementation of the paper's main ingredients:
sample reweighting from local projected distributions, full linear projection
L in M = L.T L, weighted LMNN pull/push gradients, KL distribution-preserving
cross entropy, Frobenius regularisation, and SMOTE in the learned space.
"""
import os
import sys
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from common.neighbors import smote_generate
from common.resampler import align_proba
from common import smoke as _smoke

MODEL_KEY = "ILMNN"


class ILMNN:
    def __init__(
        self,
        k=5,
        weight_k=4,
        mu1=10.0,
        mu2=0.5,
        n_iter=30,
        learning_rate=1e-3,
        margin=1.0,
        tol=1e-5,
        grad_clip=10.0,
        rf_n_estimators=30,
        n_jobs=-1,
        random_state=42,
        **kw
    ):
        self.k = int(k)
        self.weight_k = int(weight_k)
        self.mu1 = float(mu1)
        self.mu2 = float(mu2)
        self.n_iter = int(n_iter)
        self.learning_rate = float(learning_rate)
        self.margin = float(margin)
        self.tol = float(tol)
        self.grad_clip = float(grad_clip)
        self.rf_n_estimators = int(rf_n_estimators)
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.classes_ = None
        self.n_classes_ = None
        self.L_ = None
        self.clf_ = None

    def _transform_with(self, X, L):
        return np.asarray(X, dtype=float) @ L.T

    def _transform(self, X):
        return self._transform_with(X, self.L_)

    def _target_neighbours(self, X, y):
        """Use NearestNeighbors instead of manual O(n²d) distance loops."""
        n = len(y)
        targets = [np.empty(0, dtype=int) for _ in range(n)]
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            Xc = X[idx]
            if len(Xc) <= 1:
                continue
            kk = min(self.k, len(Xc) - 1)
            nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xc)
            _, nn_idx = nn.kneighbors(Xc)
            nn_idx = nn_idx[:, 1:]  # exclude self
            for j, i in enumerate(idx):
                targets[i] = idx[nn_idx[j]]
        return targets

    def _same_distribution(self, X, y):
        """Use NearestNeighbors to avoid O(n²d) manual distance computation."""
        n = len(y)
        P = np.zeros((n, n), dtype=float)
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            Xc = X[idx]
            if len(Xc) <= 1:
                continue
            kk = min(len(Xc) - 1, max(10, self.k * 3))
            nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xc)
            dist, nn_idx = nn.kneighbors(Xc)
            dist = dist[:, 1:] ** 2  # squared distances, exclude self
            nn_idx = nn_idx[:, 1:]
            for j, i in enumerate(idx):
                same_global = idx[nn_idx[j]]
                # Numerically stable softmax over negative squared distances
                e = np.exp(-dist[j] - np.max(-dist[j]))
                P[i, same_global] = e / (np.sum(e) + 1e-12)
        return P

    def _sample_weights(self, Xp, y):
        n = len(y)
        kk = min(max(1, self.weight_k), n - 1)
        if kk <= 0:
            return np.ones(n)
        dist, idx = NearestNeighbors(n_neighbors=kk + 1).fit(Xp).kneighbors(Xp)
        dist = dist[:, 1:] ** 2
        idx = idx[:, 1:]
        diff = y[idx] != y[:, None]
        e = np.exp(-dist)
        dsim = np.sum(e * diff, axis=1) / (np.sum(e, axis=1) + 1e-12)

        cnt = Counter(y.tolist())
        min_count = min(cnt.values())
        class_factor = np.array([min_count / cnt[int(c)] for c in y], dtype=float)
        w = (1.0 + dsim) * class_factor
        return w / (np.mean(w) + 1e-12)

    def _outer(self, a, b):
        d = a - b
        return np.outer(d, d)

    def _kl_gradient(self, X, Xp, y, L, P):
        n, d = X.shape
        accum = np.zeros((d, d), dtype=float)
        for i in range(n):
            same = np.where(P[i] > 0)[0]
            if len(same) == 0:
                continue
            dsq = np.sum((Xp[same] - Xp[i]) ** 2, axis=1)
            e = np.exp(-dsq - np.max(-dsq))
            q = e / (np.sum(e) + 1e-12)
            expected_c = np.zeros((d, d), dtype=float)
            for qv, l in zip(q, same):
                expected_c += qv * self._outer(X[i], X[l])
            for j in same:
                accum += P[i, j] * (self._outer(X[i], X[j]) - expected_c)
        return 2.0 * L @ accum / max(1, n)

    def _weighted_lmnn_gradient(self, X, Xp, y, L, targets, weights):
        n, d = X.shape
        accum = np.zeros((d, d), dtype=float)
        active = 0
        for i in range(n):
            diff_candidates = np.where(y != y[i])[0]
            if len(diff_candidates) == 0:
                continue
            diff_d = np.sum((Xp[diff_candidates] - Xp[i]) ** 2, axis=1)
            for j in targets[i]:
                c_ij = self._outer(X[i], X[j])
                d_ij = np.sum((Xp[i] - Xp[j]) ** 2)
                accum += weights[i] * c_ij
                violators = diff_candidates[self.margin + d_ij - diff_d > 0]
                for l in violators:
                    accum += weights[i] * (c_ij - self._outer(X[i], X[l]))
                    active += 1
        return 2.0 * L @ accum / max(1, n + active)

    def _loss_proxy(self, Xp, y, targets):
        total = 0.0
        for i in range(len(y)):
            diff = np.where(y != y[i])[0]
            if len(diff) == 0:
                continue
            diff_d = np.sum((Xp[diff] - Xp[i]) ** 2, axis=1)
            for j in targets[i]:
                d_ij = np.sum((Xp[i] - Xp[j]) ** 2)
                total += d_ij + np.sum(np.maximum(0.0, self.margin + d_ij - diff_d))
        return total / max(1, len(y))

    def _learn_projection(self, X, y):
        n, d = X.shape
        L = np.eye(d, dtype=float)
        targets = self._target_neighbours(X, y)
        P = self._same_distribution(X, y)
        prev_loss = np.inf
        for _ in range(max(1, self.n_iter)):
            Xp = self._transform_with(X, L)
            weights = self._sample_weights(Xp, y)
            grad = self._weighted_lmnn_gradient(X, Xp, y, L, targets, weights)
            grad += self.mu1 * self._kl_gradient(X, Xp, y, L, P)
            M = L.T @ L
            grad += 4.0 * self.mu2 * (L @ M)
            norm = np.linalg.norm(grad)
            if norm > self.grad_clip:
                grad = grad * (self.grad_clip / (norm + 1e-12))
            L_next = L - self.learning_rate * grad
            if not np.all(np.isfinite(L_next)):
                break
            loss = self._loss_proxy(self._transform_with(X, L_next), y, targets)
            if abs(prev_loss - loss) <= self.tol * max(1.0, abs(prev_loss)):
                L = L_next
                break
            prev_loss = loss
            L = L_next
        return L

    def _oversample(self, Xp, y, rng):
        cnt = Counter(y.tolist())
        target = max(cnt.values())
        parts_x, parts_y = [Xp], [y]
        for cls in sorted(cnt):
            need = target - cnt[cls]
            if need <= 0:
                continue
            gen = smote_generate(Xp[y == cls], need, k=self.k, rng=rng)
            if len(gen):
                parts_x.append(gen)
                parts_y.append(np.full(len(gen), cls, dtype=y.dtype))
        return np.vstack(parts_x), np.concatenate(parts_y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.max()) + 1 if len(self.classes_) else 0
        self.L_ = self._learn_projection(X, y)
        Xp = self._transform(X)
        Xr, yr = self._oversample(Xp, y, np.random.RandomState(self.random_state))
        self.clf_ = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.clf_.fit(Xr, yr)
        return self

    def predict_proba(self, X):
        raw = self.clf_.predict_proba(self._transform(X))
        return align_proba(raw, self.n_classes_, self.clf_.classes_)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def build(random_state=42, smoke=False, rf_n_estimators=30, n_jobs=-1, **kw):
    return ILMNN(
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
