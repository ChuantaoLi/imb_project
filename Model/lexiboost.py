# -*- coding: utf-8 -*-
"""LexiBoost — Lexicographic Boosting for Multi-class Imbalanced Classification
(Shen & Lin, 2023).

"LexiBoost: A Lexicographic Boosting Framework for Multi-class Imbalanced
Classification."

Mechanism:
  * Trains T base classifiers via AdaBoost (SAMME).
  * Stage P: for each class j, solves an LP minimising the max margin violation
    over ω-weighted classifiers. This gives the optimal per-class loss opt_loss_j.
  * Stage Q: solves a global LP over all samples + class-level constraints to
    minimise the worst-class slack χ, producing the final weight vector α*.
  * The resulting ensemble prioritises classes where boosting alone performs
    worst — a lexicographic (worst-first) re-weighting of the base classifiers.
"""
import os
import sys
import numpy as np
from scipy.optimize import linprog
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from common import smoke as _smoke

MODEL_KEY = "LexiBoost"


class LexiBoost(BaseEstimator, ClassifierMixin):
    """Lexicographic Boosting for Multi-class Imbalanced Data.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of base estimators.
    base_estimator : estimator or None, default=None
        Base weak learner. None -> DecisionTreeClassifier(max_depth=1).
    random_state : int or None, default=None
        Random seed.
    """

    def __init__(self, n_estimators=50, base_estimator=None, random_state=None):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.random_state = random_state

        self.estimators_ = []
        self.estimator_weights_ = []
        self.classes_ = None
        self.label_encoder_ = None

    def _get_multi_class_margin_matrix(self, X, y_internal):
        n_samples = X.shape[0]
        T = len(self.estimators_)
        margin_matrix = np.zeros((n_samples, T))

        for t in range(T):
            f_t = self.estimators_[t]
            f_t_probas = f_t.predict_proba(X)

            if f_t_probas.shape[1] < len(self.classes_):
                aligned_probas = np.zeros((n_samples, len(self.classes_)))
                try:
                    est_class_indices = self.label_encoder_.transform(
                        f_t.classes_)
                    aligned_probas[:, est_class_indices] = f_t_probas
                    f_t_probas = aligned_probas
                except Exception:
                    continue

            margin_contribution = np.sum(y_internal * f_t_probas, axis=1)
            margin_matrix[:, t] = margin_contribution

        return margin_matrix

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.label_encoder_ = LabelEncoder().fit(self.classes_)
        y_encoded = self.label_encoder_.transform(y)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        T = self.n_estimators

        if self.base_estimator is None:
            base_est = DecisionTreeClassifier(max_depth=1)
        else:
            base_est = clone(self.base_estimator)

        try:
            base_est.set_params(random_state=self.random_state)
        except ValueError:
            pass

        ada_model = AdaBoostClassifier(
            estimator=base_est,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )

        ada_model.fit(X, y)
        self.estimators_ = ada_model.estimators_

        T = len(self.estimators_)
        if T == 0:
            raise ValueError("AdaBoost produced zero estimators")

        y_internal = -np.ones((n_samples, n_classes))
        y_internal[np.arange(n_samples), y_encoded] = 1

        M = self._get_multi_class_margin_matrix(X, y_internal)

        class_indices_list = [np.where(y_encoded == c_idx)[0]
                              for c_idx in range(n_classes)]
        opt_losses = np.zeros(n_classes)

        for j in range(n_classes):
            I_j = class_indices_list[j]
            n_j = len(I_j)
            if n_j == 0:
                opt_losses[j] = 0.0
                continue

            M_j = M[I_j, :]

            n_vars_P = T + n_j

            c_P = np.zeros(n_vars_P)
            c_P[T:] = 1.0 / n_j

            A_ub_P = np.zeros((n_j, n_vars_P))
            A_ub_P[:, :T] = -M_j
            A_ub_P[:, T:] = -np.eye(n_j)
            b_ub_P = np.full(n_j, -1.0)

            A_eq_P = np.zeros((1, n_vars_P))
            A_eq_P[0, :T] = 1.0
            b_eq_P = np.array([1.0])

            bounds_P = [(0, None)] * n_vars_P

            res_P = linprog(c_P, A_ub=A_ub_P, b_ub=b_ub_P, A_eq=A_eq_P,
                            b_eq=b_eq_P, bounds=bounds_P, method='highs')
            if not res_P.success:
                # Fallback: HiGHS can fail on ill-conditioned LPs
                res_P = linprog(c_P, A_ub=A_ub_P, b_ub=b_ub_P, A_eq=A_eq_P,
                                b_eq=b_eq_P, bounds=bounds_P,
                                method='interior-point')

            if res_P.success:
                opt_losses[j] = res_P.fun
            else:
                opt_losses[j] = 0.0

        n_vars_Q = T + n_samples + 1

        c_Q = np.zeros(n_vars_Q)
        c_Q[-1] = 1.0

        A_ub_Q_rows = []
        b_ub_Q_rows = []

        for j in range(n_classes):
            I_j = class_indices_list[j]
            n_j = len(I_j)
            if n_j == 0:
                continue

            row = np.zeros(n_vars_Q)
            for i_idx, sample_idx in enumerate(I_j):
                row[T + sample_idx] = 1.0 / n_j
            row[-1] = -1.0  # -chi
            A_ub_Q_rows.append(row)
            b_ub_Q_rows.append(opt_losses[j])

        for i in range(n_samples):
            row = np.zeros(n_vars_Q)
            row[:T] = -M[i, :]
            row[T + i] = -1.0
            A_ub_Q_rows.append(row)
            b_ub_Q_rows.append(-1.0)

        A_ub_Q = np.vstack(A_ub_Q_rows)
        b_ub_Q = np.hstack(b_ub_Q_rows)

        A_eq_Q = np.zeros((1, n_vars_Q))
        A_eq_Q[0, :T] = 1.0
        b_eq_Q = np.array([1.0])

        bounds_Q = [(0, None)] * n_vars_Q

        res_Q = linprog(c_Q, A_ub=A_ub_Q, b_ub=b_ub_Q, A_eq=A_eq_Q,
                        b_eq=b_eq_Q, bounds=bounds_Q, method='highs')
        if not res_Q.success:
            # Fallback: HiGHS can fail on ill-conditioned LPs
            res_Q = linprog(c_Q, A_ub=A_ub_Q, b_ub=b_ub_Q, A_eq=A_eq_Q,
                            b_eq=b_eq_Q, bounds=bounds_Q,
                            method='interior-point')

        if res_Q.success:
            alpha_star = res_Q.x[:T]
            alpha_star[alpha_star < 0] = 0
            alpha_sum = np.sum(alpha_star)
            if alpha_sum > 0:
                self.estimator_weights_ = alpha_star / alpha_sum
            else:
                self.estimator_weights_ = np.ones(T) / T
        else:
            self.estimator_weights_ = ada_model.estimator_weights_

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        n_samples = X.shape[0]

        final_probas = np.zeros((n_samples, len(self.classes_)))

        for alpha_t, f_t in zip(self.estimator_weights_, self.estimators_):
            if alpha_t == 0:
                continue

            f_t_probas = f_t.predict_proba(X)

            if f_t_probas.shape[1] < len(self.classes_):
                aligned_probas = np.zeros((n_samples, len(self.classes_)))
                try:
                    est_class_indices = self.label_encoder_.transform(
                        f_t.classes_)
                    aligned_probas[:, est_class_indices] = f_t_probas
                    f_t_probas = aligned_probas
                except Exception:
                    continue

            final_probas += alpha_t * f_t_probas

        proba_sum = np.sum(final_probas, axis=1, keepdims=True)
        proba_sum[proba_sum == 0] = 1

        return final_probas / proba_sum

    def predict(self, X):
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return self.label_encoder_.inverse_transform(indices)


# =============================================================================
# Factory + smoke driver  (unified contract for run_all.py)
# =============================================================================

def build(random_state=42, smoke=False, n_estimators=50, **kw):
    """Unified factory for LexiBoost."""
    if smoke:
        n_estimators = min(n_estimators, 5)
    return LexiBoost(
        n_estimators=n_estimators,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
