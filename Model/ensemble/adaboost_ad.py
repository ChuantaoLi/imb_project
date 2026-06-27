"""AdaBoost.AD — data distribution + adaptive weight boosting (Li et al., TKDE 2024).

"Multi-class imbalance classification based on data distribution and adaptive
weights."

This module mirrors the paper and the authors' released MATLAB code:
1. compute the between-class imbalance ratio gamma(X, y);
2. compute the general density rho(X, y) from same-class neighbours located
   inside the global k-NN neighbourhood;
3. derive the inverse within-class density with class-wise normalization;
4. initialize sample weights uniformly as D^(1) = 1 / n;
5. train decision-tree base learners and compute beta_t = log(W_correct / W_wrong);
6. build the adaptive margin from the accumulated ensemble vote scores; and
7. update D^(t+1) = exp(-gamma * rho^{-1} * sigma^(t)) * D^(t) *
   exp(-|beta_t| * 1[h_t(x)=y]), followed by normalization.

The paper's prediction rule is weighted majority voting by beta_t. This project
also exposes `predict_proba()` as a softmax over the same vote scores so it can
plug into the common evaluation pipeline.
"""

import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common import smoke as _smoke

MODEL_KEY = "adaboost_ad"


class AdaBoostAD:
    def __init__(self, n_estimators=50, k=5, random_state=42, max_depth=None, density_lambda=1.0, **kw):
        self.n_estimators = int(n_estimators)
        self.k = int(k)
        self.random_state = random_state
        self.max_depth = max_depth
        self.density_lambda = float(density_lambda)

    @staticmethod
    def _softmax(scores):
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)

    def _general_density(self, X, y):
        n_samples = len(X)
        rho = np.zeros(n_samples, dtype=float)
        if n_samples <= 1:
            return rho

        # The supplementary MATLAB code sorts all pairwise distances before
        # selecting the first k non-zero neighbours.
        nn = NearestNeighbors(n_neighbors=n_samples).fit(X)
        distances, indices = nn.kneighbors(X)
        eps = 1e-12

        for i in range(n_samples):
            seen = 0
            same_class_inv_dist = 0.0
            same_class_count = 0
            for dist, idx in zip(distances[i], indices[i]):
                if idx == i or dist <= eps:
                    continue
                seen += 1
                if y[idx] == y[i]:
                    same_class_inv_dist += 1.0 / dist
                    same_class_count += 1
                if seen == self.k:
                    break
            if same_class_count > 0:
                rho[i] = same_class_inv_dist / same_class_count

        normalizer = rho.mean()
        if normalizer > eps:
            rho = rho / normalizer
        return rho

    def _inverse_within_class_density(self, rho, y):
        rho_inv = np.ones_like(rho, dtype=float)
        eps = 1e-12
        for class_idx in range(self.n_classes_):
            mask = y == class_idx
            if not np.any(mask):
                continue
            values = self._softmax((-rho[mask] * self.density_lambda)[:, None]).ravel()
            z_i = values.mean()
            rho_inv[mask] = values / max(z_i, eps)
        return rho_inv

    def _build_base_learner(self, seed):
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=seed,
        )

    def _vote_scores(self, X):
        X = np.asarray(X, float)
        scores = np.zeros((len(X), self.n_classes_), dtype=float)
        for clf, alpha in zip(self.clfs_, self.alphas_):
            weight = max(float(alpha), 0.0)
            if weight <= 0.0:
                continue
            pred = clf.predict(X)
            scores[np.arange(len(X)), pred] += weight
        return scores

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).ravel()
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        n_samples = len(X)

        class_counts = np.bincount(y_encoded, minlength=self.n_classes_).astype(float)
        min_class_size = max(class_counts[class_counts > 0].min(), 1.0)
        gamma = class_counts[y_encoded] / min_class_size
        rho = self._general_density(X, y_encoded)
        rho_inv = self._inverse_within_class_density(rho, y_encoded)

        w = np.full(n_samples, 1.0 / max(n_samples, 1), dtype=float)
        vote_scores = np.zeros((n_samples, self.n_classes_), dtype=float)
        eps = 1e-12

        self.gamma_ = gamma
        self.rho_ = rho
        self.rho_inv_ = rho_inv
        self.clfs_, self.alphas_ = [], []
        for m in range(self.n_estimators):
            clf = self._build_base_learner(self.random_state + m)
            clf.fit(X, y_encoded, sample_weight=w)
            pred = clf.predict(X)
            correct = pred == y_encoded

            sum_correct = w[correct].sum()
            sum_incorrect = w[~correct].sum()
            beta = np.log((sum_correct + eps) / (sum_incorrect + eps))

            self.clfs_.append(clf)
            self.alphas_.append(beta)

            effective_beta = max(beta, 0.0)
            vote_scores[np.arange(n_samples), pred] += effective_beta
            proba = self._softmax(vote_scores)

            if self.n_classes_ == 1:
                sigma = np.full(n_samples, 1.0, dtype=float)
            else:
                wrong_scores = proba.copy()
                wrong_scores[np.arange(n_samples), y_encoded] = -np.inf
                max_wrong = wrong_scores.max(axis=1)
                sigma = proba[np.arange(n_samples), y_encoded] - max_wrong + 1.0

            adaptive_factor = np.exp(-gamma * rho_inv * sigma)
            w = adaptive_factor * w
            w[correct] *= np.exp(-abs(beta))

            total = w.sum()
            if total <= 0 or not np.isfinite(total):
                w = np.full(n_samples, 1.0 / max(n_samples, 1), dtype=float)
            else:
                w /= total

        self.alphas_ = np.array(self.alphas_, dtype=float)
        self.positive_alphas_ = np.maximum(self.alphas_, 0.0)
        return self

    def predict_proba(self, X):
        return self._softmax(self._vote_scores(X))

    def predict(self, X):
        scores = self._vote_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]


def build(random_state=42, smoke=False, n_estimators=50, k=5, max_depth=None, density_lambda=1.0, **kw):
    if smoke:
        n_estimators = min(n_estimators, 8)
    return AdaBoostAD(
        n_estimators=n_estimators,
        k=k,
        random_state=random_state,
        max_depth=max_depth,
        density_lambda=density_lambda,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
