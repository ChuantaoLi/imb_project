"""OREMBoost — AdaBoost with OREM-M resampling per round (Zhu, Liu, Zhu, TKDE 2023).

"Oversampling with reliably expanding minority class regions for imbalanced data
learning." (the boosting variant.)

Mechanism: a boosting variant of OREM-M. Each AdaBoost (SAMME) round first
draws a weighted bootstrap from the current sample distribution, rebalances it
with OREM-M (reliable clean-subregion expansion), then trains a weak decision
tree. The round's weighted error is measured on the original weighted training
set to drive the SAMME alpha update.

Base learner = decision tree (default `max_depth=3`), N = 30.
"""

import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
# reuse the OREM-M resampler (reliable-region expansion)
from Model.resampling.orem_m import OREMM
from common import smoke as _smoke

MODEL_KEY = "oremboost"


class OREMBoost:
    def __init__(self, n_estimators=30, consecutive_majority=5, iteration_order="ascending", max_depth=3, random_state=42, **kw):
        self.n_estimators = int(n_estimators)
        self.consecutive_majority = int(consecutive_majority)
        self.iteration_order = iteration_order
        self.max_depth = max_depth
        self.random_state = int(random_state)

    def _build_base_learner(self, seed):
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=seed,
        )

    def _align_proba(self, clf, X):
        raw = np.asarray(clf.predict_proba(X), dtype=float)
        aligned = np.zeros((len(X), self.n_classes_), dtype=float)
        for j, cls in enumerate(np.asarray(clf.classes_, dtype=int)):
            aligned[:, self.class_to_index_[int(cls)]] = raw[:, j]
        return aligned

    def _vote_scores(self, X):
        X = np.asarray(X, float)
        scores = np.zeros((len(X), self.n_classes_), dtype=float)
        for clf, alpha in zip(self.clfs_, self.alphas_):
            pred = clf.predict(X)
            for i, cls in enumerate(pred):
                scores[i, self.class_to_index_[int(cls)]] += float(alpha)
        return scores

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int).ravel()
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.class_to_index_ = {int(c): i for i, c in enumerate(self.classes_)}
        self.index_to_class_ = {i: int(c) for i, c in enumerate(self.classes_)}
        K = len(self.classes_)
        self.n_classes_ = K
        rng = np.random.RandomState(self.random_state)
        w = np.full(len(X), 1.0 / len(X))

        self.clfs_, self.alphas_ = [], []
        for m in range(self.n_estimators):
            # OREM-M rebalance on a weighted bootstrap of the current distribution
            idx = rng.choice(len(X), size=len(X), replace=True, p=w)
            Xb, yb = OREMM(
                consecutive_majority=self.consecutive_majority,
                iteration_order=self.iteration_order,
                random_state=self.random_state + m,
            )._resample(X[idx], y[idx])
            clf = self._build_base_learner(self.random_state + m)
            clf.fit(Xb, yb)
            pred = clf.predict(X)  # error on original weighted set
            err = np.dot(w, (pred != y)) / w.sum()
            err = np.clip(err, 1e-6, 1 - 1e-6)
            alpha = np.log((1 - err) / err) + np.log(max(K - 1, 1))
            w *= np.exp(alpha * (pred != y))
            w /= w.sum()
            self.clfs_.append(clf)
            self.alphas_.append(alpha)
        self.alphas_ = np.array(self.alphas_)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        votes = self._vote_scores(X)
        votes -= votes.max(1, keepdims=True)
        e = np.exp(votes)
        return e / np.clip(e.sum(1, keepdims=True), 1e-12, None)

    def predict(self, X):
        scores = self._vote_scores(X)
        return np.asarray(
            [self.index_to_class_[i] for i in np.argmax(scores, axis=1)],
            dtype=int,
        )


def build(random_state=42, smoke=False, n_estimators=30, consecutive_majority=5, iteration_order="ascending", max_depth=3, **kw):
    if smoke:
        n_estimators = min(n_estimators, 6)
    return OREMBoost(
        n_estimators=n_estimators,
        consecutive_majority=consecutive_majority,
        iteration_order=iteration_order,
        max_depth=max_depth,
        random_state=random_state,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
