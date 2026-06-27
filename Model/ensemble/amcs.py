"""AMCS — adaptive multiple classifier system (Li et al., KBS 2016).

Paper-faithful structure:
1. classify the dataset into one of 8 types using IR / dimensionality /
   number-of-classes thresholds;
2. choose the paper's fixed route for that type:
   FCBF or BPSO feature selection + AdaBoost.M1 (FiltEX resampling) +
   one base classifier among C4.5 / SVM / KNN / RBF-NN;
3. weight the weak learners with training-time normalized AUCarea and fuse
   their posterior outputs into the final class probabilities.

Unlike the earlier placeholder implementation, this module does not search an
extra resampling pool or an extra ensemble-rule pool at fit time.
"""

import os
import sys
import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common import smoke as _smoke

MODEL_KEY = "amcs"


TYPE_ROUTES = {
    1: ("fcbf", "c45"),
    2: ("bpso", "knn"),
    3: ("fcbf", "c45"),
    4: ("bpso", "svm"),
    5: ("bpso", "rbf"),
    6: ("fcbf", "svm"),
    7: ("bpso", "c45"),
    8: ("bpso", "svm"),
}


def _entropy_from_counts(counts):
    p = np.asarray(counts, dtype=float)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _discretize_vector(x, bins=10):
    x = np.asarray(x, dtype=float).ravel()
    if len(np.unique(x)) <= 1:
        return np.zeros_like(x, dtype=int)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(x, quantiles))
    if len(edges) <= 2:
        return np.zeros_like(x, dtype=int)
    return np.digitize(x, edges[1:-1], right=False)


def _symmetrical_uncertainty(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    counts_a = np.bincount(ia, minlength=len(ua))
    counts_b = np.bincount(ib, minlength=len(ub))
    joint = np.zeros((len(ua), len(ub)), dtype=float)
    np.add.at(joint, (ia, ib), 1.0)
    h_a = _entropy_from_counts(counts_a)
    h_b = _entropy_from_counts(counts_b)
    h_ab = _entropy_from_counts(joint.ravel())
    denom = h_a + h_b
    if denom <= 1e-12:
        return 0.0
    return float(2.0 * (h_a + h_b - h_ab) / denom)


def _full_fcbf(X, y, delta=0.0):
    """FCBF with SU relevance filtering and redundancy elimination."""
    if X.shape[1] == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    Xd = np.column_stack([_discretize_vector(X[:, j]) for j in range(X.shape[1])])
    yd = np.asarray(y, dtype=int).ravel()
    su_class = np.array([_symmetrical_uncertainty(Xd[:, j], yd) for j in range(Xd.shape[1])], dtype=float)

    relevant = np.where(su_class >= float(delta))[0]
    if relevant.size == 0 and su_class.size:
        relevant = np.array([int(np.argmax(su_class))], dtype=int)

    ordered = relevant[np.argsort(su_class[relevant])[::-1]]
    selected = []
    active = list(int(i) for i in ordered)
    while active:
        p = active.pop(0)
        selected.append(p)
        keep = []
        for q in active:
            if _symmetrical_uncertainty(Xd[:, p], Xd[:, q]) < su_class[q]:
                keep.append(q)
        active = keep

    mask = np.zeros(X.shape[1], dtype=bool)
    if selected:
        mask[np.array(selected, dtype=int)] = True
    return mask, su_class


def _reorder_auc_values(order):
    order = np.asarray(order, dtype=float)
    n = len(order)
    if n <= 3:
        return order.copy()
    r = np.empty(n, dtype=float)
    r[0] = order[0]
    half = n // 2
    for i in range(1, half + 1):
        r[i] = order[2 * i - 1]
    if n % 2 == 0:
        for i in range(half + 1, n):
            idx = n - 2 * (i - half - 1) - 2
            r[i] = order[idx]
    else:
        for i in range(half + 1, n):
            idx = n - 2 * (i - half - 1) - 1
            r[i] = order[idx]
    return r


def _normalized_aucarea(y_true, proba, n_classes):
    y_true = np.asarray(y_true, dtype=int).ravel()
    proba = np.asarray(proba, dtype=float)
    if n_classes <= 1:
        return 1.0

    aucs = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            mask = (y_true == i) | (y_true == j)
            if mask.sum() < 2:
                continue
            y_pair = y_true[mask]
            proba_pair = proba[mask]
            ni = int(np.sum(y_pair == i))
            nj = int(np.sum(y_pair == j))
            pos = i if ni > nj else j
            y_bin = (y_pair == pos).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            try:
                auc = roc_auc_score(y_bin, proba_pair[:, pos])
            except Exception:
                auc = 0.5
            aucs.append(float(auc))

    if not aucs:
        return 0.0

    order = np.sort(np.asarray(aucs, dtype=float))[::-1]
    arranged = _reorder_auc_values(order)
    q = len(arranged)
    if q == 0:
        return 0.0
    area = 0.5 * math.sin(2.0 * math.pi / q) * (np.sum(arranged[:-1] * arranged[1:]) + arranged[-1] * arranged[0])
    max_area = 0.5 * math.sin(2.0 * math.pi / q) * q
    if max_area <= 1e-12:
        return 0.0
    return float(area / max_area)


class _RBFNN:
    def __init__(self, n_centers=12, gamma=None, ridge=1e-2, random_state=42):
        self.n_centers = int(n_centers)
        self.gamma = gamma
        self.ridge = float(ridge)
        self.random_state = int(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_centers = min(max(2, self.n_centers), len(X))
        self.centers_ = KMeans(n_clusters=n_centers, n_init=3, random_state=self.random_state).fit(X).cluster_centers_
        gamma = self.gamma if self.gamma is not None else 1.0 / max(X.shape[1], 1)
        self.gamma_ = float(gamma)
        H = np.exp(-self.gamma_ * self._sqdist(X, self.centers_))
        Y = np.eye(self.n_classes_)[y]
        reg = self.ridge * np.eye(n_centers)
        self.W_ = np.linalg.solve(H.T @ H + reg, H.T @ Y)
        return self

    @staticmethod
    def _sqdist(a, b):
        return np.sum(a**2, axis=1)[:, None] + np.sum(b**2, axis=1)[None, :] - 2.0 * a @ b.T

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        H = np.exp(-self.gamma_ * self._sqdist(X, self.centers_))
        logits = H @ self.W_
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class AMCS:
    def __init__(
        self,
        n_estimators=20,
        bpso_particles=20,
        bpso_iters=10,
        random_state=42,
        **kw,
    ):
        self.n_estimators = int(n_estimators)
        self.bpso_particles = int(bpso_particles)
        self.bpso_iters = int(bpso_iters)
        self.random_state = int(random_state)
        self.rbf_centers = int(kw.get("rbf_centers", 12))
        self.rbf_ridge = float(kw.get("rbf_ridge", 1e-2))
        self.fcbf_delta = float(kw.get("fcbf_delta", 0.0))
        self.bpso_w = float(kw.get("bpso_w", 0.729))
        self.bpso_c1 = float(kw.get("bpso_c1", 1.49445))
        self.bpso_c2 = float(kw.get("bpso_c2", 1.49445))
        self.bpso_vmax = float(kw.get("bpso_vmax", 6.0))
        self.bpso_cv = int(kw.get("bpso_cv", 5))

    def _infer_type(self, X, y_enc):
        counts = np.bincount(y_enc, minlength=self.n_classes_)
        present = counts[counts > 0]
        ir = float(present.max() / max(present.min(), 1)) if present.size else 1.0
        dim = X.shape[1]
        high_ir = ir >= 10.0
        high_dim = dim >= 10
        many_classes = self.n_classes_ >= 6
        if high_ir and high_dim and many_classes:
            return 1
        if high_ir and (not high_dim) and many_classes:
            return 2
        if high_ir and high_dim and (not many_classes):
            return 3
        if high_ir and (not high_dim) and (not many_classes):
            return 4
        if (not high_ir) and high_dim and many_classes:
            return 5
        if (not high_ir) and (not high_dim) and many_classes:
            return 6
        if (not high_ir) and high_dim and (not many_classes):
            return 7
        return 8

    def _make_base_classifier(self, base_name, seed, y_fit=None):
        if base_name == "svm":
            return SVC(kernel="rbf", probability=True, random_state=seed)
        if base_name == "knn":
            min_count = 5
            if y_fit is not None and len(y_fit):
                counts = np.bincount(np.asarray(y_fit, dtype=int), minlength=self.n_classes_)
                positive = counts[counts > 0]
                if positive.size:
                    min_count = int(max(1, min(5, positive.min())))
            return KNeighborsClassifier(n_neighbors=min_count)
        if base_name == "c45":
            return DecisionTreeClassifier(criterion="entropy", random_state=seed)
        if base_name == "rbf":
            return _RBFNN(
                n_centers=self.rbf_centers,
                ridge=self.rbf_ridge,
                random_state=seed,
            )
        raise ValueError(f"Unsupported AMCS base classifier: {base_name}")

    def _align_proba(self, clf, X):
        raw = np.asarray(clf.predict_proba(X), dtype=float)
        out = np.zeros((len(X), self.n_classes_), dtype=float)
        for j, c in enumerate(np.asarray(clf.classes_, dtype=int)):
            out[:, int(c)] = raw[:, j]
        row_sum = out.sum(axis=1, keepdims=True)
        return out / np.clip(row_sum, 1e-12, None)

    def _cv_subset_score(self, X, y, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        Xsel = np.asarray(X, dtype=float)[:, mask]
        y = np.asarray(y, dtype=int).ravel()
        counts = np.bincount(y, minlength=self.n_classes_)
        min_count = int(counts[counts > 0].min()) if np.any(counts > 0) else 0
        if min_count <= 1:
            clf = self._make_base_classifier(self.base_name_, self.random_state, y_fit=y)
            clf.fit(Xsel, y)
            proba = self._align_proba(clf, Xsel)
            return _normalized_aucarea(y, proba, self.n_classes_)

        n_splits = min(self.bpso_cv, min_count)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(Xsel, y)):
            clf = self._make_base_classifier(self.base_name_, self.random_state + fold_id, y_fit=y[tr_idx])
            clf.fit(Xsel[tr_idx], y[tr_idx])
            proba = self._align_proba(clf, Xsel[va_idx])
            scores.append(_normalized_aucarea(y[va_idx], proba, self.n_classes_))
        return float(np.mean(scores)) if scores else 0.0

    def _bpso_select(self, X, y):
        d = X.shape[1]
        if d == 0:
            return np.zeros(0, dtype=bool)
        rng = np.random.RandomState(self.random_state)
        n_particles = max(1, self.bpso_particles)
        velocity = np.zeros((n_particles, d), dtype=float)
        position = (rng.rand(n_particles, d) >= 0.5).astype(int)
        for i in range(n_particles):
            if position[i].sum() == 0:
                position[i, rng.randint(0, d)] = 1

        personal_best = position.copy()
        personal_score = np.full(n_particles, -np.inf, dtype=float)
        global_best = position[0].copy()
        global_score = -np.inf

        for _ in range(max(1, self.bpso_iters)):
            for i in range(n_particles):
                score = self._cv_subset_score(X, y, position[i].astype(bool))
                if score > personal_score[i]:
                    personal_score[i] = score
                    personal_best[i] = position[i].copy()
                if score > global_score:
                    global_score = score
                    global_best = position[i].copy()

            for i in range(n_particles):
                r1 = rng.rand(d)
                r2 = rng.rand(d)
                velocity[i] = self.bpso_w * velocity[i] + self.bpso_c1 * r1 * (personal_best[i] - position[i]) + self.bpso_c2 * r2 * (global_best - position[i])
                velocity[i] = np.clip(velocity[i], -self.bpso_vmax, self.bpso_vmax)
                transfer = 1.0 / (1.0 + np.exp(-2.0 * velocity[i]))
                position[i] = (rng.rand(d) < transfer).astype(int)
                if position[i].sum() == 0:
                    position[i, rng.randint(0, d)] = 1

        best = np.asarray(global_best, dtype=bool)
        if best.sum() == 0:
            best[np.argmax([self._cv_subset_score(X, y, np.eye(d, dtype=bool)[j]) for j in range(d)])] = True
        return best

    def _select_features(self, X, y):
        if self.feature_method_ == "fcbf":
            mask, _ = _full_fcbf(X, y, delta=self.fcbf_delta)
            if mask.sum() == 0:
                mask = np.ones(X.shape[1], dtype=bool)
            return mask
        if self.feature_method_ == "bpso":
            return self._bpso_select(X, y)
        raise ValueError(f"Unsupported AMCS feature selector: {self.feature_method_}")

    @staticmethod
    def _filt_ex_sample(X, y, weights, rng):
        idx = rng.choice(len(y), size=len(y), replace=True, p=weights)
        return X[idx], y[idx]

    def _train_adaboost_m1(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        n_samples = len(y)
        weights = np.full(n_samples, 1.0 / max(n_samples, 1), dtype=float)
        models = []
        auc_weights = []
        betas = []
        eps = 1e-12

        for t in range(max(1, self.n_estimators)):
            Xb, yb = self._filt_ex_sample(X, y, weights, self.rng_)
            clf = self._make_base_classifier(self.base_name_, self.random_state + t, y_fit=yb)
            clf.fit(Xb, yb)

            train_proba = self._align_proba(clf, X)
            train_pred = np.argmax(train_proba, axis=1)
            incorrect = (train_pred != y).astype(float)
            err = float(np.dot(weights, incorrect))
            err = min(max(err, eps), 1.0 - eps)

            # AdaBoost.M1 requires weak learners better than chance.
            if err >= 0.5:
                break

            beta = err / max(1.0 - err, eps)
            weights *= np.power(beta, 1.0 - incorrect)
            weights /= np.clip(weights.sum(), eps, None)

            models.append(clf)
            betas.append(beta)
            auc_weights.append(max(_normalized_aucarea(y, train_proba, self.n_classes_), eps))

        if not models:
            clf = self._make_base_classifier(self.base_name_, self.random_state, y_fit=y)
            clf.fit(X, y)
            models = [clf]
            betas = [1.0]
            auc_weights = [max(_normalized_aucarea(y, self._align_proba(clf, X), self.n_classes_), eps)]

        auc_weights = np.asarray(auc_weights, dtype=float)
        auc_weights /= np.clip(auc_weights.sum(), eps, None)
        return models, np.asarray(betas, dtype=float), auc_weights

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.rng_ = np.random.RandomState(self.random_state)

        self.type_id_ = self._infer_type(X, y_enc)
        self.feature_method_, self.base_name_ = TYPE_ROUTES[self.type_id_]
        self.route_name_ = f"{self.feature_method_}-adaboost-{self.base_name_}"

        self.final_mask_ = self._select_features(X, y_enc)
        if self.final_mask_.sum() == 0:
            self.final_mask_ = np.ones(X.shape[1], dtype=bool)

        Xsel = X[:, self.final_mask_]
        self.pool_, self.beta_weights_, self.base_weights_ = self._train_adaboost_m1(Xsel, y_enc)
        return self

    def predict_proba(self, X):
        Xsel = np.asarray(X, dtype=float)[:, self.final_mask_]
        fused = np.zeros((len(Xsel), self.n_classes_), dtype=float)
        for weight, clf in zip(self.base_weights_, self.pool_):
            fused += float(weight) * self._align_proba(clf, Xsel)
        row_sum = fused.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            fused[row_sum.ravel() <= 0] = 1.0 / max(self.n_classes_, 1)
            row_sum = fused.sum(axis=1, keepdims=True)
        return fused / np.clip(row_sum, 1e-12, None)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build(random_state=42, smoke=False, n_estimators=20, bpso_particles=20, bpso_iters=10, **kw):
    if smoke:
        n_estimators = min(n_estimators, 5)
        bpso_particles = min(bpso_particles, 8)
        bpso_iters = min(bpso_iters, 4)
    return AMCS(
        n_estimators=n_estimators,
        bpso_particles=bpso_particles,
        bpso_iters=bpso_iters,
        random_state=random_state,
        **kw,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
