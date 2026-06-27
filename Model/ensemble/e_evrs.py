"""E-EVRS — Ensemble Evidential Re-Sampling (Grina, Elouedi, Lefevre, IJAR 2023).

"Re-sampling of multi-class imbalanced data using belief function theory and
ensemble learning."

Mechanism: a bagging ensemble where EACH bootstrap is resampled with the
MC-EVHS evidential hybrid resampler (oversample classes below the mean size,
priority-undersample classes above), a base learner is trained per bootstrap, and
the per-classifier predictions are aggregated with the DEMPSTER rule of
combination (belief-function fusion) rather than plain averaging.

The fusion stage is classifier-independent: any probabilistic base learner can
provide the soft posterior that is then turned into a belief assignment and
combined with Dempster's rule. This implementation exposes decision tree
(default) and SVM base learners and reuses the shared DS engine
(`common.dempster`) together with the MC-EVHS resampler.
"""
import os
import sys
from copy import deepcopy
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from Model.resampling.mc_evhs import MCEVHS
from common.dempster import mass_from_proba, dempster_combine, pignistic_proba
from common import smoke as _smoke

MODEL_KEY = "e_evrs"


class EEVRS:
    def __init__(self, n_estimators=30, k=5, alpha=1.0, t=5, base_kind="dt",
                 base_estimator=None, random_state=42, **kw):
        self.n_estimators = int(n_estimators)
        self.k = int(k)
        self.alpha = float(alpha)
        self.t = int(t)
        self.base_kind = base_kind
        self.base_estimator = base_estimator
        self.random_state = random_state

    def _make_base(self, seed):
        if self.base_estimator is not None:
            clf = deepcopy(self.base_estimator)
            params = clf.get_params()
            if "random_state" in params:
                clf.set_params(random_state=seed)
            return clf
        if self.base_kind == "dt":
            return DecisionTreeClassifier(random_state=seed)
        if self.base_kind == "svm":
            return SVC(kernel="rbf", probability=True, random_state=seed)
        raise ValueError(f"Unsupported E-EVRS base classifier: {self.base_kind}")

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.class_to_index_ = {int(c): i for i, c in enumerate(self.classes_)}
        rng = np.random.RandomState(self.random_state)
        self.clfs_ = []
        for i in range(self.n_estimators):
            idx = rng.randint(0, len(X), len(X))
            Xb, yb = MCEVHS(k=self.k, alpha=self.alpha, t=self.t,
                            random_state=self.random_state + i)._resample(X[idx], y[idx])
            clf = self._make_base(self.random_state + i)
            self.clfs_.append(clf.fit(Xb, yb))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        classes = list(self.classes_)
        out = np.zeros((len(X), self.n_classes_))
        for ti in range(len(X)):
            masses = []
            for clf in self.clfs_:
                raw = clf.predict_proba(X[ti:ti + 1])[0]
                p = np.zeros(self.n_classes_)
                for j, c in enumerate(clf.classes_):
                    p[self.class_to_index_[int(c)]] = raw[j]
                p = p / (p.sum() + 1e-12)
                masses.append(mass_from_proba(p, classes))
            combined = dempster_combine(masses)
            out[ti] = pignistic_proba(combined, classes)
        out /= out.sum(1, keepdims=True) + 1e-12
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


def build(random_state=42, smoke=False, n_estimators=30, k=5, alpha=1.0, t=5,
          base_kind="dt", base_estimator=None, **kw):
    if smoke:
        n_estimators = min(n_estimators, 8)
    return EEVRS(
        n_estimators=n_estimators,
        k=k,
        alpha=alpha,
        t=t,
        base_kind=base_kind,
        base_estimator=base_estimator,
        random_state=random_state,
        **kw,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
