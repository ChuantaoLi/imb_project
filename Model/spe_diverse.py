"""Diverse SPE aggregation used by the final V2 candidates."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from .spe_boost_base import SPEBoostEnsemble


class DiverseSPE(BaseEstimator, ClassifierMixin):
    """Two complementary SPE learners with fixed, interpretable mixing.

    The Balanced-RF branch emphasizes minority recall; the ExtraTrees branch
    provides decorrelated probability estimates.  The mixture is a model
    component, not a test-set selection rule.
    """
    def __init__(self, random_state=42, n_estimators=14, rf_trees=30,
                 extra_trees=30, mix=0.65):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.rf_trees = rf_trees
        self.extra_trees = extra_trees
        self.mix = mix

    def fit(self, X, y):
        self.brf_ = SPEBoostEnsemble(
            random_state=self.random_state, strategy="brf_full",
            base_depth=8, rf_trees=self.rf_trees,
            n_estimators=self.n_estimators)
        self.extra_ = SPEBoostEnsemble(
            random_state=self.random_state + 7919, strategy="extra",
            base_depth=None, rf_trees=self.extra_trees,
            n_estimators=self.n_estimators)
        self.brf_.fit(X, y)
        self.extra_.fit(X, y)
        self.classes_ = self.brf_.classes_
        return self

    def predict_proba(self, X):
        p = self.mix * self.brf_.predict_proba(X)
        p += (1.0 - self.mix) * self.extra_.predict_proba(X)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
