"""SPE with NROMM's noise-robust, boundary-safe oversampling layer."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from .nromm import NROMM
from .spe_boost_base import SPEBoostEnsemble


class NROMMSPE(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=42, n_estimators=14, rf_trees=22,
                 mix_original=0.25):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.rf_trees = rf_trees
        self.mix_original = mix_original

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        # NROMM is the safe-boundary data branch.  It is fitted once with a
        # full-depth RF, matching its paper-level downstream learner.
        self.nromm_ = NROMM(k=5, max_ratio=20, rf_n_estimators=self.rf_trees,
                            n_jobs=1, random_state=self.random_state)
        self.nromm_.fit(X, y)
        # SPE remains an independent, self-paced boosting branch.  A small
        # fixed contribution supplies diversity without re-sampling NROMM's
        # generated points a second time.
        self.spe_ = SPEBoostEnsemble(
            random_state=self.random_state + 7919, strategy="spe_rf",
            base_depth=None, rf_trees=14,
            n_estimators=self.n_estimators, learning_rate=0.55)
        self.spe_.fit(X, y)
        self.classes_ = self.nromm_.classes_
        return self

    def predict_proba(self, X):
        p_n = self.nromm_.predict_proba(X)
        p_s = self.spe_.predict_proba(X)
        p = (1.0 - self.mix_original) * p_n + self.mix_original * p_s
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
