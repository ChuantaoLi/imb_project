"""PT-Bagging — Plug-in Bagging with threshold-moving (Collell, Prelec, Patil,
Neurocomputing 2018).

"A simple plug-in bagging ensemble based on threshold-moving for classifying
binary and multiclass imbalanced data."

Mechanism: standard bag-of-trees bagging (with-replacement bootstrap); aggregate
by averaging the per-class probabilities of all base learners; then APPLY the
paper's threshold-moving rule from Algorithm 1:
    score_k(x) = \hat{P}(y=k|x) / lambda_k
and predict the class with the highest score. For multiclass macro-accuracy, the
paper shows that the class priors provide the optimal thresholds, so the default
choice is lambda_k = P(y=k).
No training-time resampling is used -- imbalance is handled purely at the
decision stage. Base learner = decision tree (paper J48; sklearn CART).
"""
import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common import smoke as _smoke

MODEL_KEY = "pt_bagging"


class PTBagging:
    def __init__(
        self,
        n_estimators=30,
        max_depth=None,
        threshold_mode="macro_accuracy",
        thresholds=None,
        random_state=42,
        **kw
    ):
        if threshold_mode not in {"macro_accuracy", "manual"}:
            raise ValueError("threshold_mode must be one of macro_accuracy/manual")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.threshold_mode = threshold_mode
        self.thresholds = thresholds
        self.random_state = int(random_state)

    def _align_proba(self, clf, X):
        raw = np.asarray(clf.predict_proba(X), dtype=float)
        aligned = np.zeros((len(X), self.n_classes_), dtype=float)
        for j, c in enumerate(np.asarray(clf.classes_, dtype=int)):
            aligned[:, self.class_to_index_[int(c)]] = raw[:, j]
        return aligned

    def _decision_thresholds(self):
        if self.threshold_mode == "manual":
            if self.thresholds is None:
                raise ValueError("manual threshold_mode requires thresholds")
            thresholds = np.asarray(self.thresholds, dtype=float).ravel()
            if len(thresholds) != self.n_classes_:
                raise ValueError("thresholds length must equal number of classes")
            return np.clip(thresholds, 1e-12, None)
        return np.clip(self.class_priors_, 1e-12, None)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self.class_to_index_ = {int(c): i for i, c in enumerate(self.classes_)}
        self.index_to_class_ = {i: int(c) for i, c in enumerate(self.classes_)}
        self.n_classes_ = len(self.classes_)
        self.class_priors_ = np.array(
            [np.mean(y == c) for c in self.classes_],
            dtype=float,
        )
        self.lambda_ = self._decision_thresholds()
        rng = np.random.RandomState(self.random_state)
        self.clfs_ = []
        n = len(X)
        for i in range(self.n_estimators):
            idx = rng.randint(0, n, n)                  # with-replacement bootstrap
            clf = DecisionTreeClassifier(max_depth=self.max_depth,
                                         random_state=self.random_state + i)
            clf.fit(X[idx], y[idx])
            self.clfs_.append(clf)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        acc = np.zeros((len(X), self.n_classes_), dtype=float)
        for clf in self.clfs_:
            acc += self._align_proba(clf, X)
        acc /= max(len(self.clfs_), 1)
        return acc / np.clip(acc.sum(1, keepdims=True), 1e-12, None)

    def predict(self, X):
        proba = self.predict_proba(X)
        scores = proba / self.lambda_.reshape(1, -1)
        picked = np.argmax(scores, axis=1)
        return np.asarray([self.index_to_class_[i] for i in picked], dtype=int)


def build(random_state=42, smoke=False, n_estimators=30, max_depth=None,
          threshold_mode="macro_accuracy", thresholds=None, **kw):
    if smoke:
        n_estimators = min(n_estimators, 8)
    return PTBagging(
        n_estimators=n_estimators,
        max_depth=max_depth,
        threshold_mode=threshold_mode,
        thresholds=thresholds,
        random_state=random_state,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
