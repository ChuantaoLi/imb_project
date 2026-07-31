# -*- coding: utf-8 -*-
"""DualLexiBoost — Dual Lexicographic Boosting (Shen & Lin, 2023).

"DualLexiBoost: Lexicographic Boosting with Dual Formulation."

Mechanism (dual formulation of LexiBoost):
  * Stage A: trains T base classifiers via weighted sampling, updating the
    distribution D via the dual P' LP — each classifier's sample weights are
    the dual variables that minimise worst-class redundancy.
  * Stage P_j: computes per-class optimal losses λ*_j from the primal LP.
  * Stage B: trains a second set of T classifiers, updating D via the dual Q'
    LP which incorporates both the primal class-level losses and sample-level
    margin constraints.
  * Final LP (primal Q): solves for the ensemble weight vector α* that
    minimises the worst-class loss using the combined classifier pool.
  * Result: a lexicographically-optimised boosting ensemble that is explicitly
    shaped to perform well on the hardest class.
"""
import os
import sys
import numpy as np
from scipy.optimize import linprog
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from common import smoke as _smoke

MODEL_KEY = "DualLexiBoost"


def _safe_linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None,
                  verbose=False):
    """linprog with HiGHS → interior-point fallback for robustness.

    HiGHS (a C++ library) can segfault on ill-conditioned LP matrices.  When
    the process is protected by subprocess isolation (run_all.py) the crash is
    contained, but a fallback to the more stable interior-point method still
    increases the chance of getting a result on the first attempt.
    """
    for method in ('highs', 'interior-point'):
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method=method)
            if res.success:
                return res
        except Exception:
            if verbose:
                print(f"[linprog] {method} raised; trying fallback")
            continue
    # Last resort: return unsuccessful result from highs
    try:
        return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                       bounds=bounds, method='highs')
    except Exception:
        return None


class DualLexiBoost(BaseEstimator, ClassifierMixin):
    """Dual Lexicographic Boosting.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of base estimators per stage (A + B = 2×n_estimators total).
    max_depth : int, default=10
        Maximum depth of each decision-tree weak learner.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print LP solver diagnostics.
    """

    def __init__(self, n_estimators=50, max_depth=10, random_state=None,
                 verbose=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.base_classifiers_ = []
        self.alpha_ = None
        self.classes_ = None
        self.verbose = verbose

    def _train_weak_classifier(self, X, y, sample_weight):
        clf = DecisionTreeClassifier(max_depth=self.max_depth,
                                     random_state=self.random_state)
        clf.fit(X, y, sample_weight=sample_weight)
        return clf

    def _compute_signed_pred(self, clf, X, y_original):
        preds = clf.predict(X)
        return np.where(preds == y_original, 1, -1)

    def _solve_class_specific_lp(self, X, y, classifiers, class_label):
        class_indices = np.where(y == class_label)[0]
        n_class_samples = len(class_indices)
        if n_class_samples == 0:
            return np.ones(len(classifiers)) / max(1, len(classifiers)), 0.0

        n_classifiers = len(classifiers)
        n_vars = n_classifiers + n_class_samples

        c = np.zeros(n_vars)
        c[n_classifiers:] = 1.0 / n_class_samples

        A_ub = []
        b_ub = []

        for idx_pos, idx in enumerate(class_indices):
            row = np.zeros(n_vars)
            for t in range(n_classifiers):
                pred = classifiers[t].predict(X[idx].reshape(1, -1))[0]
                signed_pred = 1 if pred == y[idx] else -1
                row[t] = -signed_pred
            row[n_classifiers + idx_pos] = -1.0
            A_ub.append(row)
            b_ub.append(-1.0)

        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n_classifiers] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, None)] * n_vars

        try:
            res = _safe_linprog(c,
                                A_ub=np.array(A_ub) if A_ub else None,
                                b_ub=np.array(b_ub) if b_ub else None,
                                A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                                verbose=self.verbose)
            if res is not None and res.success:
                return res.x[:n_classifiers], res.fun
            else:
                if self.verbose:
                    print(f"[P_j LP] error {class_label}")
                return np.ones(n_classifiers) / n_classifiers, 0.0
        except Exception as e:
            if self.verbose:
                print(f"[P_j LP] ex: {e}")
            return np.ones(n_classifiers) / n_classifiers, 0.0

    def _solve_final_lp(self, X, y, classifiers, class_optimal_losses):
        n_samples = X.shape[0]
        n_classifiers = len(classifiers)
        n_classes = len(self.classes_)
        n_vars = n_classifiers + n_samples + 1

        c = np.zeros(n_vars)
        c[-1] = 1.0

        A_ub = []
        b_ub = []

        for j_idx, class_label in enumerate(self.classes_):
            class_indices = np.where(y == class_label)[0]
            n_class_samples = len(class_indices)
            if n_class_samples == 0:
                continue

            row = np.zeros(n_vars)
            for idx in class_indices:
                row[n_classifiers + idx] = 1.0 / n_class_samples
            row[-1] = -1.0
            A_ub.append(row)
            b_ub.append(class_optimal_losses[j_idx])

        for i in range(n_samples):
            row = np.zeros(n_vars)
            for t in range(n_classifiers):
                pred = classifiers[t].predict(X[i].reshape(1, -1))[0]
                signed_pred = 1 if pred == y[i] else -1
                row[t] = -signed_pred
            row[n_classifiers + i] = -1.0
            A_ub.append(row)
            b_ub.append(-1.0)

        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n_classifiers] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, None)] * n_vars

        try:
            res = _safe_linprog(c,
                                A_ub=np.array(A_ub) if A_ub else None,
                                b_ub=np.array(b_ub) if b_ub else None,
                                A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                                verbose=self.verbose)
            if res is not None and res.success:
                return res.x[:n_classifiers]
            else:
                if self.verbose:
                    print("[Q LP] error")
                return np.ones(n_classifiers) / n_classifiers
        except Exception as e:
            if self.verbose:
                print(f"[Q LP] ex: {e}")
            return np.ones(n_classifiers) / n_classifiers

    def _solve_dual_Pprime(self, X, y, classifiers, D_current, t_round,
                           class_idx_list):
        n = X.shape[0]
        F = np.zeros((t_round, n))
        for tau in range(t_round):
            F[tau, :] = self._compute_signed_pred(classifiers[tau], X, y)

        all_D_solutions = []

        for j, class_indices in enumerate(class_idx_list):
            nj = len(class_indices)
            if nj == 0:
                continue

            n_vars = n + 1

            c = np.zeros(n_vars)
            for i_in_class in class_indices:
                c[i_in_class] = -1.0
            c[-1] = 1.0

            A_ub = []
            b_ub = []

            for tau in range(t_round):
                row = np.zeros(n_vars)
                row[:n] = F[tau, :]
                row[-1] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)

            A_eq = np.zeros((1, n_vars))
            A_eq[0, :n] = 1.0
            b_eq = np.array([1.0])

            bounds = []
            for i in range(n):
                if i in class_indices:
                    bounds.append((0.0, 1.0 / float(nj)))
                else:
                    bounds.append((0.0, None))
            bounds.append((None, None))

            try:
                res = _safe_linprog(c,
                                    A_ub=np.array(A_ub) if A_ub else None,
                                    b_ub=np.array(b_ub) if b_ub else None,
                                    A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                                    verbose=self.verbose)
                if res is not None and res.success:
                    D_solution = res.x[:n]
                    if np.sum(D_solution) > 0:
                        all_D_solutions.append(D_solution)
                else:
                    if self.verbose:
                        print(f"[P'_j] error {j}")
            except Exception as e:
                if self.verbose:
                    print(f"[P'_j] ex: {e} for class {j}")

        if not all_D_solutions:
            return D_current

        combined_D = np.mean(all_D_solutions, axis=0)
        sum_D = np.sum(combined_D)

        if sum_D > 0:
            return combined_D / sum_D
        else:
            return D_current

    def _solve_dual_Qprime(self, X, y, classifiers, lambda_star_list, t_round,
                           class_idx_list):
        n = X.shape[0]
        k = len(class_idx_list)
        n_vars = n + k + 1

        sLambda = np.array([
            np.sum(lambda_star_list[j])
            if lambda_star_list[j] is not None and len(lambda_star_list[j]) > 0
            else 0.0
            for j in range(k)
        ])
        nj_list = np.array([len(class_idx_list[j]) for j in range(k)],
                           dtype=float)

        c = np.zeros(n_vars)
        c[:n] = -1.0
        for j in range(k):
            c[n + j] = sLambda[j] / nj_list[j] if nj_list[j] > 0 else 0.0
        c[-1] = 1.0

        A_ub = []
        b_ub = []

        F = np.zeros((t_round, n))
        for tau in range(t_round):
            F[tau, :] = self._compute_signed_pred(classifiers[tau], X, y)

        for tau in range(t_round):
            row = np.zeros(n_vars)
            row[:n] = F[tau, :]
            row[-1] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

        for j in range(k):
            nj = nj_list[j]
            if nj == 0:
                continue
            for i in class_idx_list[j]:
                row = np.zeros(n_vars)
                row[i] = 1.0
                row[n + j] = -1.0 / nj
                A_ub.append(row)
                b_ub.append(0.0)

        row = np.zeros(n_vars)
        row[n:n + k] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)

        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, None)] * (n + k) + [(None, None)]

        try:
            res = _safe_linprog(c,
                                A_ub=np.array(A_ub) if A_ub else None,
                                b_ub=np.array(b_ub) if b_ub else None,
                                A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                                verbose=self.verbose)
            if res is not None and res.success:
                D_sol = res.x[:n]
                return (D_sol / np.sum(D_sol)
                        if np.sum(D_sol) > 0
                        else np.ones(n) / float(n))
            else:
                if self.verbose:
                    print("[Q'] error")
                return np.ones(n) / float(n)
        except Exception as e:
            if self.verbose:
                print(f"[Q'] ex: {e}")
            return np.ones(n) / float(n)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.label_encoder_ = LabelEncoder().fit(y)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        class_idx_list = [list(np.where(y == cls)[0]) for cls in self.classes_]
        D = np.zeros(n_samples, dtype=float)
        for j, cls in enumerate(self.classes_):
            idxs = class_idx_list[j]
            nj = len(idxs)
            if nj > 0:
                D[idxs] = 1.0 / (n_classes * nj)
        D = (D / np.sum(D)
             if np.sum(D) > 0
             else np.ones(n_samples) / n_samples)

        # ---- Stage A: train classifiers with dual P' distribution ----
        self.base_classifiers_ = []
        for t in range(self.n_estimators):
            clf = self._train_weak_classifier(X, y, sample_weight=D)
            self.base_classifiers_.append(clf)

            preds = clf.predict(X)
            err = np.sum(D * (preds != y))
            if self.verbose:
                print(f"[DualLexiBoost A] round {t + 1}, err {err:.6f}")
            if err > (1.0 - 1.0 / float(n_classes)) or err == 0:
                if err > (1.0 - 1.0 / float(n_classes)):
                    self.base_classifiers_.pop()
                break

            D = self._solve_dual_Pprime(X, y, self.base_classifiers_, D,
                                        len(self.base_classifiers_),
                                        class_idx_list)
            D = (D / np.sum(D)
                 if np.sum(D) > 0
                 else np.ones(n_samples) / n_samples)

        if not self.base_classifiers_:
            if self.verbose:
                print("No classifiers in stage A, using fallback")
            clf = DecisionTreeClassifier(
                max_depth=1, random_state=self.random_state).fit(X, y)
            self.base_classifiers_.append(clf)

        # ---- Compute per-class primal losses ----
        class_optimal_losses = []
        lambda_star_list = []
        for j, cls in enumerate(self.classes_):
            _, opt_loss_j = self._solve_class_specific_lp(
                X, y, self.base_classifiers_, cls)
            class_optimal_losses.append(opt_loss_j)

            nj = len(class_idx_list[j])
            lambda_star_list.append(
                np.full(nj, opt_loss_j) if nj > 0 else np.array([]))

        # ---- Stage B: train more classifiers with dual Q' distribution ----
        D = np.zeros(n_samples, dtype=float)
        for j, cls in enumerate(self.classes_):
            idxs = class_idx_list[j]
            nj = len(idxs)
            if nj > 0:
                D[idxs] = 1.0 / (n_classes * nj)
        D = (D / np.sum(D)
             if np.sum(D) > 0
             else np.ones(n_samples) / n_samples)

        all_classifiers = []
        for t in range(self.n_estimators):
            clf = self._train_weak_classifier(X, y, sample_weight=D)
            all_classifiers.append(clf)

            preds = clf.predict(X)
            err = np.sum(D * (preds != y))
            if self.verbose:
                print(f"[DualLexiBoost B] round {t + 1}, err {err:.6f}")
            if err > (1.0 - 1.0 / float(n_classes)) or err == 0:
                if err > (1.0 - 1.0 / float(n_classes)):
                    all_classifiers.pop()
                break

            D = self._solve_dual_Qprime(X, y, all_classifiers,
                                        lambda_star_list,
                                        len(all_classifiers),
                                        class_idx_list)
            D = (D / np.sum(D)
                 if np.sum(D) > 0
                 else np.ones(n_samples) / float(n_samples))

        if not all_classifiers:
            all_classifiers = self.base_classifiers_
        if not all_classifiers:
            if self.verbose:
                print("No classifiers available, using fallback")
            clf = DecisionTreeClassifier(
                max_depth=1, random_state=self.random_state).fit(X, y)
            all_classifiers.append(clf)
            if not class_optimal_losses:
                class_optimal_losses = [0.0] * n_classes

        # ---- Final LP for ensemble weights ----
        alpha_final = self._solve_final_lp(X, y, all_classifiers,
                                           class_optimal_losses)

        alpha_final = np.maximum(alpha_final, 0.0)
        if np.sum(alpha_final) == 0:
            alpha_final = np.ones(len(all_classifiers)) / float(
                len(all_classifiers))
        else:
            alpha_final = alpha_final / np.sum(alpha_final)

        self.base_classifiers_ = all_classifiers
        self.alpha_ = alpha_final
        return self

    def predict(self, X):
        check_is_fitted(self)
        n_samples = X.shape[0]
        classes = self.classes_

        if len(classes) == 2:
            scores = np.zeros(n_samples)
            binary_labels = self.label_encoder_.transform(classes)
            for alpha_t, clf in zip(self.alpha_, self.base_classifiers_):
                preds = self.label_encoder_.transform(clf.predict(X))
                binary_preds = np.where(preds == binary_labels[1], 1, -1)
                scores += alpha_t * binary_preds
            return np.where(scores >= 0, classes[1], classes[0])
        else:
            class_scores = np.zeros((n_samples, len(classes)))
            for alpha_t, clf in zip(self.alpha_, self.base_classifiers_):
                preds = clf.predict(X)
                for j, cls in enumerate(classes):
                    class_scores[preds == cls, j] += alpha_t
            idx = np.argmax(class_scores, axis=1)
            return classes[idx]

    def predict_proba(self, X):
        check_is_fitted(self)
        n_samples = X.shape[0]
        classes = self.classes_

        if len(classes) == 2:
            scores = np.zeros(n_samples)
            binary_labels = self.label_encoder_.transform(classes)
            for alpha_t, clf in zip(self.alpha_, self.base_classifiers_):
                preds = self.label_encoder_.transform(clf.predict(X))
                binary_preds = np.where(preds == binary_labels[1], 1, -1)
                scores += alpha_t * binary_preds
            prob = 1.0 / (1.0 + np.exp(-2.0 * scores))
            return np.column_stack([1 - prob, prob])
        else:
            class_scores = np.zeros((n_samples, len(classes)))
            for alpha_t, clf in zip(self.alpha_, self.base_classifiers_):
                preds = clf.predict(X)
                for j, cls in enumerate(classes):
                    class_scores[preds == cls, j] += alpha_t

            exp_scores = np.exp(
                class_scores - np.max(class_scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return probs


# =============================================================================
# Factory + smoke driver  (unified contract for run_all.py)
# =============================================================================

def build(random_state=42, smoke=False, n_estimators=50, max_depth=10, **kw):
    """Unified factory for DualLexiBoost."""
    if smoke:
        n_estimators = min(n_estimators, 5)
    return DualLexiBoost(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
