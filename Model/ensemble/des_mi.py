"""DES-MI — Dynamic Ensemble Selection for multiclass imbalance (García et al.,
Information Sciences 2018).

"Dynamic ensemble selection for multi-class imbalanced datasets."

This implementation follows the paper's two core steps:
1. train a pool of CART classifiers on random-balance bootstraps; and
2. for each query, estimate each classifier's competence on the region of
   competence using instance weights that favour minority-class neighbours,
   then select the top `pct_accuracy` fraction of the pool.

The project benchmark passes pre-encoded multiclass labels, so this module keeps
the paper's multiclass setting directly instead of decomposing into binaries.
"""
import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.neighbors import smote_generate
from common import smoke as _smoke

MODEL_KEY = "des_mi"


class DESMI:
    def __init__(self, n_estimators=30, k=7, pct_accuracy=0.4, alpha=0.9,
                 dsel_perc=0.5, n_select=None, random_state=42, **kw):
        self.n_estimators = int(n_estimators)
        self.k = int(k)
        self.pct_accuracy = float(pct_accuracy)
        self.alpha = float(alpha)
        self.dsel_perc = float(dsel_perc)
        self.n_select = n_select
        self.random_state = random_state
        self._validate_parameters()

    def _random_balance_subset(self, X, y, rng):
        classes = np.unique(y)
        total = len(y)
        props = rng.dirichlet(np.ones(len(classes)))
        targets = np.maximum(1, np.round(props * total).astype(int))
        Xl, yl = [], []
        for c, t in zip(classes, targets):
            idx = np.where(y == c)[0]
            if len(idx) >= t:
                sel = rng.choice(idx, size=t, replace=False)
                Xl.append(X[sel]); yl.append(y[sel])
            else:
                Xl.append(X[idx]); yl.append(y[idx])
                gen = smote_generate(X[idx], t - len(idx), self.k, rng)
                Xl.append(gen); yl.append(np.full(t - len(idx), c, dtype=y.dtype))
        return np.vstack(Xl), np.concatenate(yl)

    def _validate_parameters(self):
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not (0.0 < self.pct_accuracy <= 1.0):
            raise ValueError("pct_accuracy must be in (0, 1]")
        if not (0.0 < self.dsel_perc < 1.0):
            raise ValueError("dsel_perc must be in (0, 1)")

    def _compute_n_selected(self):
        if self.n_select is not None:
            return min(self.n_estimators, max(1, int(self.n_select)))
        return min(self.n_estimators, max(1, int(self.n_estimators * self.pct_accuracy)))

    def _split_pool_and_dsel(self, X, y):
        counts = np.bincount(y, minlength=int(y.max()) + 1)
        can_stratify = np.all(counts[counts > 0] >= 2)
        if can_stratify:
            try:
                return train_test_split(
                    X, y, test_size=self.dsel_perc, stratify=y,
                    random_state=self.random_state
                )
            except ValueError:
                pass
        # Fallback for ultra-small classes: reuse the full set for both steps.
        return X, X.copy(), y, y.copy()

    def _align_proba(self, clf, X):
        raw = clf.predict_proba(X)
        proba = np.zeros((len(X), self.n_classes_), dtype=float)
        for j, c in enumerate(clf.classes_):
            proba[:, int(c)] = raw[:, j]
        return proba

    def _estimate_competence(self, X):
        _, nb = self.nn_.kneighbors(X)
        targets = self.y_dsel_[nb]
        class_frequency = np.bincount(self.y_dsel_, minlength=self.n_classes_)
        num = class_frequency[targets]
        weight = 1.0 / (1.0 + np.exp(self.alpha * num))
        weight /= np.clip(weight.sum(axis=1, keepdims=True), 1e-12, None)

        correct = self.processed_dsel_[nb, :].astype(float) * weight[:, :, None]
        return correct.mean(axis=1)

    def _select(self, competence):
        n_sel = self._compute_n_selected()
        return np.argsort(competence, axis=1)[:, ::-1][:, :n_sel]

    def _majority_vote(self, votes):
        out = np.zeros(votes.shape[0], dtype=int)
        for i, row in enumerate(votes):
            out[i] = np.argmax(np.bincount(row, minlength=self.n_classes_))
        return out

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.max()) + 1

        self.X_pool_, self.X_dsel_, self.y_pool_, self.y_dsel_ = self._split_pool_and_dsel(X, y)
        rng = np.random.RandomState(self.random_state)
        self.clfs_ = []
        for i in range(self.n_estimators):
            Xb, yb = self._random_balance_subset(self.X_pool_, self.y_pool_, rng)
            clf = DecisionTreeClassifier(random_state=self.random_state + i)
            self.clfs_.append(clf.fit(Xb, yb))

        self.nn_ = NearestNeighbors(
            n_neighbors=min(self.k, len(self.X_dsel_))
        ).fit(self.X_dsel_)
        self.processed_dsel_ = np.column_stack([
            clf.predict(self.X_dsel_) == self.y_dsel_ for clf in self.clfs_
        ])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        competence = self._estimate_competence(X)
        selected = self._select(competence)
        probas = np.stack([self._align_proba(clf, X) for clf in self.clfs_], axis=1)
        out = probas[np.arange(len(X))[:, None], selected, :].mean(axis=1)
        return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)

    def predict(self, X):
        X = np.asarray(X, float)
        competence = self._estimate_competence(X)
        selected = self._select(competence)
        predictions = np.column_stack([clf.predict(X) for clf in self.clfs_])
        votes = predictions[np.arange(len(X))[:, None], selected]
        return self._majority_vote(votes)


def build(random_state=42, smoke=False, n_estimators=30, k=7,
          pct_accuracy=0.4, alpha=0.9, dsel_perc=0.5, **kw):
    if smoke:
        n_estimators = min(n_estimators, 8)
    return DESMI(
        n_estimators=n_estimators, k=k, pct_accuracy=pct_accuracy,
        alpha=alpha, dsel_perc=dsel_perc, random_state=random_state, **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
