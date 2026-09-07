# -*- coding: utf-8 -*-
"""HRIELM: hybrid-resampling improved selective ensemble ELM.

Faithful components from Zhang et al., Journal of Building Engineering 70
(2023): HRT (SMOTE interpolation plus random under-sampling), one ELM per
benchmark class, and local KNN selection/ distance-weighted voting.  The
benchmark runner supplies a train fold; a stratified part of that fold is used
as the paper's verification set so test labels never affect selection.
"""
import os
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from common import smoke as _smoke

MODEL_KEY = "HRIELM"


def _sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _hrt(X, y, target, k, rng):
    """Paper HRT for one benchmark count, preserving original samples."""
    parts_x, parts_y = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        if len(idx) > target:
            idx = rng.choice(idx, size=target, replace=False)
            parts_x.append(X[idx]); parts_y.append(y[idx])
        elif len(idx) < target:
            base = X[idx]
            if len(base) == 0:
                continue
            need = target - len(base)
            if len(base) == 1:
                synth = np.repeat(base, need, axis=0)
            else:
                kk = min(max(1, k), len(base) - 1)
                # Querying the fitted points with n_neighbors equal to the
                # fit size is rejected by recent sklearn versions because
                # self-neighbours are removed.  A pairwise fallback is both
                # exact for this small minority subset and version-independent.
                neigh = np.empty((len(base), kk), dtype=int)
                for ii in range(len(base)):
                    d = np.sum((base - base[ii]) ** 2, axis=1)
                    d[ii] = np.inf
                    neigh[ii] = np.argsort(d)[:kk]
                synth = np.empty((need, X.shape[1]), dtype=float)
                for j in range(need):
                    i = int(rng.randint(len(base)))
                    q = int(rng.choice(neigh[i]))
                    synth[j] = base[i] + rng.rand() * (base[q] - base[i])
            parts_x.append(np.vstack([base, synth]))
            parts_y.append(np.full(target, cls, dtype=y.dtype))
        else:
            parts_x.append(X[idx]); parts_y.append(y[idx])
    if not parts_x:
        return X.copy(), y.copy()
    Xo, yo = np.vstack(parts_x), np.concatenate(parts_y)
    p = rng.permutation(len(yo))
    return Xo[p], yo[p]


class _ELM:
    def __init__(self, input_dim, hidden=2000, reg=0.01, seed=42):
        self.hidden = int(hidden)
        self.reg = float(reg)
        rng = np.random.RandomState(seed)
        self.W = rng.normal(0.0, 1.0, (input_dim, self.hidden))
        self.b = rng.normal(0.0, 1.0, self.hidden)

    def fit(self, X, y, n_classes):
        H = _sigmoid(X @ self.W + self.b)
        Y = np.zeros((len(y), n_classes), dtype=float)
        Y[np.arange(len(y)), y.astype(int)] = 1.0
        # Ridge-stabilised Moore-Penrose solution, the regularised ELM objective.
        # Use the smaller primal/dual system.  The paper uses 2000 hidden
        # neurons; solving a 2000x2000 system is unnecessary when a fold has
        # fewer samples and can make a benchmark infeasible.
        if H.shape[0] <= self.hidden:
            A = H @ H.T + self.reg * np.eye(H.shape[0])
            self.beta_ = H.T @ np.linalg.solve(A, Y)
        else:
            A = H.T @ H + self.reg * np.eye(self.hidden)
            self.beta_ = np.linalg.solve(A, H.T @ Y)
        self.n_classes_ = n_classes
        return self

    def predict_proba(self, X):
        logits = _sigmoid(X @ self.W + self.b) @ self.beta_
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)


class HRIELM:
    def __init__(self, hidden_neurons=2000, reg=0.01, smote_k_neighbors=5,
                 n_neighbors=5, val_ratio=0.2, random_state=42,
                 preencoded=True, **kwargs):
        self.hidden_neurons = int(hidden_neurons)
        self.reg = float(reg)
        self.smote_k = int(smote_k_neighbors)
        self.n_neighbors = int(n_neighbors)
        self.val_ratio = float(val_ratio)
        self.random_state = int(random_state)
        self.preencoded = preencoded

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.max()) + 1
        rng = np.random.RandomState(self.random_state)
        counts = Counter(y)
        # Split before fitting: the paper's verification set is used only for
        # local competence selection and must not be part of HRT/ELM training.
        try:
            tr, va = train_test_split(np.arange(len(y)), test_size=self.val_ratio,
                                      stratify=y, random_state=self.random_state)
        except ValueError:
            cut = max(1, int(len(y) * (1.0 - self.val_ratio)))
            p = rng.permutation(len(y)); tr, va = p[:cut], p[cut:]
        if len(va) == 0: va = tr
        X_fit, y_fit = X[tr], y[tr]
        # Each benchmark class creates one balanced HRT training set/model.
        self.models_ = []
        for j, cls in enumerate(self.classes_):
            # Each fault class is the benchmark in turn, hence its own count
            # is the target (HRIELM Section 4/5), rather than one global min.
            target = max(1, int(np.sum(y_fit == cls)))
            Xr, yr = _hrt(X_fit, y_fit, target, self.smote_k, rng)
            self.models_.append(_ELM(X.shape[1], self.hidden_neurons,
                                     self.reg, self.random_state + j).fit(
                                         Xr, yr, self.n_classes_))
        # Verification data is independent of model fitting and is used only
        # to select locally competent ELMs at prediction time.
        self.X_val_, self.y_val_ = X[va], y[va]
        self._k_query = min(self.n_neighbors, max(1, len(va) - 1))
        self.nn_ = NearestNeighbors(n_neighbors=self._k_query).fit(self.X_val_)
        self._train_idx_ = tr
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        all_p = np.stack([m.predict_proba(X) for m in self.models_], axis=1)
        out = np.empty((len(X), self.n_classes_), dtype=float)
        k = self._k_query
        _, idx = self.nn_.kneighbors(X, n_neighbors=k)
        val_p = np.stack([m.predict_proba(self.X_val_) for m in self.models_], axis=1)
        for i in range(len(X)):
            d = np.linalg.norm(self.X_val_[idx[i]] - X[i], axis=1)
            inv_d = 1.0 / np.maximum(d, 1e-12)
            acc = np.array([(np.argmax(val_p[idx[i], j], axis=1) == self.y_val_[idx[i]]).mean()
                            for j in range(len(self.models_))])
            selected = acc >= acc.mean()
            if not selected.any():
                selected[:] = True
            alpha = np.array([
                np.sum(inv_d * (np.argmax(val_p[idx[i], j], axis=1) == self.y_val_[idx[i]]))
                for j in range(len(self.models_))
            ])
            alpha[~selected] = 0.0
            if not np.isfinite(alpha).all() or alpha.sum() <= 0:
                alpha[:] = 1.0
            out[i] = (all_p[i] * alpha[:, None]).sum(axis=0)
            out[i] /= max(out[i].sum(), 1e-12)
        return out

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def build(random_state=42, smoke=False, hidden_neurons=2000, reg=0.01,
          smote_k_neighbors=5, n_neighbors=5, val_ratio=0.2, **kwargs):
    if smoke:
        hidden_neurons = min(hidden_neurons, 128)
    return HRIELM(hidden_neurons=hidden_neurons, reg=reg,
                  smote_k_neighbors=smote_k_neighbors, n_neighbors=n_neighbors,
                  val_ratio=val_ratio, random_state=random_state, **kwargs)


if __name__ == '__main__':
    _smoke.run_smoke(build, MODEL_KEY)
