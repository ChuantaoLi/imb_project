"""common.resampler — base class for the 14 data-resampling methods.

Every resampler follows the same pattern:
    fit(X, y):   resample (X, y) -> (X', y') via the subclass's _resample(),
                 then train ONE RandomForest(n_estimators=30) on (X', y').
    predict_proba(X):  delegate to the RF.

This guarantees all 14 resamplers differ ONLY in the resampling step and share
the identical downstream RF(30) head -> pure apples-to-apples comparison. The
subclass implements only `_resample(self, X, y) -> (X_res, y_res)`.

X / y arrive already StandardScaled + integer-encoded 0..C-1 (the runner's
preprocessing), so _resample operates on scaled features.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def align_proba(proba, n_classes, model_classes):
    """Expand a (N, k) proba to (N, n_classes) reindexing by `model_classes`
    (the RF's classes_). No-op when k == n_classes."""
    proba = np.asarray(proba, dtype=float)
    if proba.shape[1] == n_classes:
        return proba
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    for j, c in enumerate(model_classes):
        try:
            out[:, int(c)] = proba[:, j]
        except (ValueError, IndexError):
            continue
    return out


class Resampler:
    """Base for the 14 data-resampling survey methods.

    Subclass and override ``_resample(self, X, y)``. Set any method-specific
    hyper-params in the subclass __init__ (call super().__init__ first).
    """

    def __init__(self, rf_n_estimators=30, n_jobs=-1, random_state=42, **kw):
        self.rf_n_estimators = rf_n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.clf_ = None
        self.classes_ = None
        self.n_classes_ = None

    # -- subclass hook -------------------------------------------------------
    def _resample(self, X, y):
        """Return (X_resampled, y_resampled). Override in subclass."""
        raise NotImplementedError

    # -- shared fit / predict -----------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = (int(self.classes_.max()) + 1) if len(self.classes_) else 0
        Xr, yr = self._resample(X, y)
        Xr = np.asarray(Xr, dtype=float)
        yr = np.asarray(yr).astype(int).ravel()
        if len(np.unique(yr)) < 2:                      # degenerate guard
            yr = y; Xr = X
        self.clf_ = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            random_state=self.random_state, n_jobs=self.n_jobs)
        self.clf_.fit(Xr, yr)
        return self

    def predict_proba(self, X):
        raw = self.clf_.predict_proba(np.asarray(X, dtype=float))
        return align_proba(raw, self.n_classes_, self.clf_.classes_)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
