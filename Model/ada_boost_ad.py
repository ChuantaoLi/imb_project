# -*- coding: utf-8 -*-
"""AdaBoostAD — Adaptive Boosting with Adaptive Margin (Liu et al., 2024).

"AdaBoostAD: Adaptive Boosting with Adaptive Margin for Imbalanced Data."

Mechanism:
  * Between-class imbalance factor γ_i: class_size / min_class_size
  * Within-class inverse density factor ρ⁻¹_i: exp(-ρ_i) / Z_c,
    where ρ_i is the inverse average k-nearest-same-class-distance normalised
    by mean over all samples.
  * Adaptive margin: prob_correct - max_prob_incorrect + 1
  * Weight update: w_i ← exp(-γ_i · ρ⁻¹_i · margin) · w_i · exp(-β_t · correct_i)
  * The product γ_i · ρ⁻¹_i amplifies the weight of hard samples from dense,
    large classes while suppressing noisy small-class samples.
"""
import os
import sys
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from common import smoke as _smoke

MODEL_KEY = "AdaBoostAD"


class AdaBoostAD(BaseEstimator, ClassifierMixin):
    """Adaptive Boosting with Adaptive Margin.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of boosting rounds.
    base_estimator : estimator or None, default=None
        Base weak learner. None -> DecisionTreeClassifier(max_depth=1).
    random_state : int or None, default=None
        Random seed.
    k_neighbors : int, default=5
        Number of neighbours for within-class density estimation.
    """

    def __init__(self, n_estimators=50, base_estimator=None, random_state=None,
                 k_neighbors=5):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.k_neighbors = k_neighbors

        self.estimators_ = []
        self.estimator_weights_ = []
        self.classes_ = None
        self.n_classes_ = 0
        self.label_encoder_ = None
        self.class_to_idx_ = None

    def _compute_between_class_imbalance(self, y):
        n_samples = len(y)
        class_counts = Counter(y)
        min_class_size = min(class_counts.values())

        gamma = np.zeros(n_samples)
        for i in range(n_samples):
            gamma[i] = class_counts[y[i]] / min_class_size

        return gamma

    def _compute_inverse_within_class_density(self, X, y):
        n_samples = X.shape[0]

        k = self.k_neighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
        distances, indices = nn.kneighbors(X)

        rho_raw = np.ones(n_samples)

        for i in range(n_samples):
            neighbor_indices = indices[i, 1:]
            neighbor_distances = distances[i, 1:]
            neighbor_labels = y[neighbor_indices]

            same_class_mask = (neighbor_labels == y[i])
            same_class_distances = neighbor_distances[same_class_mask]

            k_i = len(same_class_distances)

            if k_i > 0:
                safe_distances = np.maximum(same_class_distances, 1e-10)
                rho_raw[i] = (1.0 / k_i) * np.sum(1.0 / safe_distances)

        T = np.mean(rho_raw)
        rho_general = rho_raw / T

        rho_inv_numerator = np.exp(-rho_general)
        rho_inverse = np.zeros(n_samples)

        for cls in self.classes_:
            class_mask = (y == cls)
            if np.any(class_mask):
                Z_c = np.mean(rho_inv_numerator[class_mask])
                rho_inverse[class_mask] = rho_inv_numerator[class_mask] / Z_c

        return rho_inverse

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.label_encoder_ = LabelEncoder().fit(y)
        y_encoded = self.label_encoder_.transform(y)
        self.class_to_idx_ = {cls: i for i, cls in
                              enumerate(self.label_encoder_.classes_)}

        n_samples = X.shape[0]

        if self.base_estimator is None:
            base_estimator = DecisionTreeClassifier(
                max_depth=1, random_state=self.random_state
            )
        else:
            base_estimator = self.base_estimator

        gamma = self._compute_between_class_imbalance(y)
        rho_inverse = self._compute_inverse_within_class_density(X, y)

        sample_weights = np.full(n_samples, 1.0 / n_samples)
        ensemble_outputs = np.zeros((n_samples, self.n_classes_))

        self.estimators_ = []
        self.estimator_weights_ = []

        for t in range(self.n_estimators):
            try:
                base_estimator.set_params(random_state=self.random_state)
            except ValueError:
                pass

            estimator = clone(base_estimator)
            estimator.fit(X, y, sample_weight=sample_weights)

            y_pred = estimator.predict(X)
            y_pred_encoded = self.label_encoder_.transform(y_pred)

            correct_mask = (y_pred == y)
            incorrect_mask = ~correct_mask

            numerator = np.sum(sample_weights[correct_mask])
            denominator = np.sum(sample_weights[incorrect_mask])

            beta_t = np.log(numerator / max(denominator, 1e-10))

            self.estimators_.append(estimator)
            self.estimator_weights_.append(beta_t)

            for i in range(n_samples):
                ensemble_outputs[i, y_pred_encoded[i]] += beta_t

            exp_outputs = np.exp(ensemble_outputs -
                                 np.max(ensemble_outputs, axis=1, keepdims=True))
            probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

            prob_correct = probabilities[np.arange(n_samples), y_encoded]

            probabilities_incorrect = probabilities.copy()
            probabilities_incorrect[np.arange(n_samples), y_encoded] = -np.inf
            prob_max_incorrect = np.max(probabilities_incorrect, axis=1)

            adaptive_margin = prob_correct - prob_max_incorrect + 1.0

            adaptive_factor = np.exp(-gamma * rho_inverse * adaptive_margin)

            is_correct_int = correct_mask.astype(int)
            sample_weights = (adaptive_factor * sample_weights *
                              np.exp(-beta_t * is_correct_int))

            sample_weights /= np.sum(sample_weights)

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        n_samples = X.shape[0]

        ensemble_outputs = np.zeros((n_samples, self.n_classes_))

        for beta_t, estimator in zip(self.estimator_weights_, self.estimators_):
            y_pred = estimator.predict(X)

            for i in range(n_samples):
                if y_pred[i] in self.class_to_idx_:
                    ensemble_outputs[i, self.class_to_idx_[y_pred[i]]] += beta_t

        exp_outputs = np.exp(ensemble_outputs -
                             np.max(ensemble_outputs, axis=1, keepdims=True))
        probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

        return probabilities

    def predict(self, X):
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return self.classes_[indices]


# =============================================================================
# Factory + smoke driver  (unified contract for run_all.py)
# =============================================================================

def build(random_state=42, smoke=False, n_estimators=50, k_neighbors=5, **kw):
    """Unified factory for AdaBoostAD."""
    if smoke:
        n_estimators = min(n_estimators, 5)
    return AdaBoostAD(
        n_estimators=n_estimators,
        k_neighbors=k_neighbors,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
