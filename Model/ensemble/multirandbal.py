"""MultiRandBal — Random Balance Ensembles for multiclass imbalance (Rodríguez,
Díez-Pastor, Arnaiz-González, KBS 2020).

"Random balance ensembles for multiclass imbalance learning."

Mechanism: each base learner trains on a bootstrap whose CLASS DISTRIBUTION is
drawn at RANDOM (independent random per-class weights normalised to the original
sample size, subject to a minimum class size) -- so the ensemble sees many
different (even inverted) imbalance regimes. For every class, if its natural
count exceeds the drawn target it is under-sampled, otherwise it is
SMOTE-oversampled up to the target. The paper studies both bagging and boosting
variants; bagging aggregates by majority vote, while boosting uses AdaBoost-style
classifier weighting over random-balanced training rounds.
"""
import os
import sys
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.neighbors import smote_generate
from common import smoke as _smoke

MODEL_KEY = "multirandbal"


class MultiRandBal:
    def __init__(
        self,
        n_estimators=30,
        k=5,
        mechanism="bagging",
        base_estimator=None,
        min_samples=2,
        max_depth=None,
        random_state=42,
        **kw
    ):
        if mechanism not in {"bagging", "boosting"}:
            raise ValueError("mechanism must be one of bagging/boosting")
        self.n_estimators = int(n_estimators)
        self.k = int(k)
        self.mechanism = mechanism
        self.base_estimator = base_estimator
        self.min_samples = int(min_samples)
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_base_estimator(self, seed):
        if self.base_estimator is not None:
            return self.base_estimator(random_state=seed)
        return DecisionTreeClassifier(max_depth=self.max_depth, random_state=seed)

    def _random_balance_targets(self, rng):
        random_weights = np.array([rng.uniform(0.0, 1.0) for _ in self.classes_], dtype=float)
        weight_sum = float(random_weights.sum())
        if weight_sum <= 0:
            random_weights = np.ones(len(self.classes_), dtype=float)
            weight_sum = float(len(self.classes_))
        targets = {}
        for cls, weight in zip(self.classes_, random_weights):
            target = max(int(round(self.n_samples_total_ * weight / weight_sum)), self.min_samples)
            targets[cls] = target
        return targets

    def _resample_class(self, Xc, target, rng):
        Xc = np.asarray(Xc, dtype=float)
        if target <= len(Xc):
            picked = rng.choice(np.arange(len(Xc)), size=target, replace=False)
            return Xc[picked]
        generated = target - len(Xc)
        if len(Xc) == 0:
            return np.empty((0, self.n_features_), dtype=float)
        if generated <= 0:
            return Xc.copy()
        synth = smote_generate(Xc, generated, self.k, rng)
        return np.vstack([Xc, synth])

    def _random_balance_subset(self, X, y, rng):
        targets = self._random_balance_targets(rng)
        Xl, yl = [], []
        for cls in self.classes_:
            current = X[y == cls]
            target = targets[cls]
            subset = self._resample_class(current, target, rng)
            if len(subset) == 0:
                continue
            Xl.append(subset)
            yl.append(np.full(len(subset), cls, dtype=y.dtype))
        return np.vstack(Xl), np.concatenate(yl)

    def _fit_bagging(self, X, y, rng):
        self.clfs_ = []
        for i in range(self.n_estimators):
            Xb, yb = self._random_balance_subset(X, y, rng)
            clf = self._make_base_estimator(self.random_state + i)
            clf.fit(Xb, yb)
            self.clfs_.append(clf)

    def _fit_boosting(self, X, y, rng):
        n_samples = len(y)
        self.clfs_ = []
        self.alphas_ = []
        sample_weight = np.full(n_samples, 1.0 / max(n_samples, 1), dtype=float)
        n_classes = len(self.classes_)
        max_err = 1.0 - 1.0 / max(n_classes, 2)

        for i in range(self.n_estimators):
            draw = rng.choice(np.arange(n_samples), size=n_samples, replace=True, p=sample_weight)
            Xw, yw = X[draw], y[draw]
            Xb, yb = self._random_balance_subset(Xw, yw, rng)
            clf = self._make_base_estimator(self.random_state + i)
            clf.fit(Xb, yb)
            pred = clf.predict(X)
            miss = pred != y
            err = float(np.sum(sample_weight * miss))
            err = min(max(err, 1e-12), max_err)
            if err >= max_err:
                continue
            alpha = np.log((1.0 - err) / err) + np.log(max(n_classes - 1, 1))
            sample_weight *= np.exp(alpha * miss)
            sample_weight /= np.sum(sample_weight)
            self.clfs_.append(clf)
            self.alphas_.append(alpha)

        if len(self.clfs_) == 0:
            clf = self._make_base_estimator(self.random_state)
            Xb, yb = self._random_balance_subset(X, y, rng)
            clf.fit(Xb, yb)
            self.clfs_.append(clf)
            self.alphas_.append(1.0)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self.class_to_index_ = {int(c): i for i, c in enumerate(self.classes_)}
        self.index_to_class_ = {i: int(c) for i, c in enumerate(self.classes_)}
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.n_samples_per_class_ = {int(c): int((y == c).sum()) for c in self.classes_}
        self.n_samples_total_ = len(y)
        rng = np.random.RandomState(self.random_state)
        if self.mechanism == "bagging":
            self._fit_bagging(X, y, rng)
        else:
            self._fit_boosting(X, y, rng)
        return self

    def _align_proba(self, clf, X):
        raw = clf.predict_proba(X)
        p = np.zeros((len(X), self.n_classes_), dtype=float)
        for j, c in enumerate(clf.classes_):
            p[:, self.class_to_index_[int(c)]] = raw[:, j]
        return p

    def predict_proba(self, X):
        X = np.asarray(X, float)
        acc = np.zeros((len(X), self.n_classes_), dtype=float)
        if self.mechanism == "boosting":
            weights = np.asarray(self.alphas_, dtype=float)
            weights = weights / np.sum(weights)
            for alpha, clf in zip(weights, self.clfs_):
                acc += alpha * self._align_proba(clf, X)
        else:
            for clf in self.clfs_:
                acc += self._align_proba(clf, X)
            acc /= max(len(self.clfs_), 1)
        denom = np.clip(acc.sum(1, keepdims=True), 1e-12, None)
        return acc / denom

    def predict(self, X):
        X = np.asarray(X, float)
        if self.mechanism == "boosting":
            scores = np.zeros((len(X), self.n_classes_), dtype=float)
            for alpha, clf in zip(self.alphas_, self.clfs_):
                pred = clf.predict(X)
                for i, cls in enumerate(pred):
                    scores[i, self.class_to_index_[int(cls)]] += alpha
            return np.asarray([self.index_to_class_[i] for i in np.argmax(scores, axis=1)], dtype=int)

        votes = []
        for clf in self.clfs_:
            votes.append(clf.predict(X))
        votes = np.asarray(votes).T
        out = []
        for row in votes:
            out.append(Counter(row.tolist()).most_common(1)[0][0])
        return np.asarray(out, dtype=int)


def build(
    random_state=42,
    smoke=False,
    n_estimators=30,
    k=5,
    mechanism="bagging",
    min_samples=2,
    max_depth=None,
    **kw
):
    if smoke:
        n_estimators = min(n_estimators, 8)
    return MultiRandBal(
        n_estimators=n_estimators,
        k=k,
        mechanism=mechanism,
        min_samples=min_samples,
        max_depth=max_depth,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
