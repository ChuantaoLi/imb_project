"""
SPE: Self-paced Ensemble for Highly Imbalanced Massive Data Classification
Complete reproduction of:
  "Self-paced Ensemble for Highly Imbalanced Massive Data Classification"
  Zhining Liu, Wei Cao, Zhifeng Gao, Jiang Bian, Hechang Chen, Yi Chang, Tie-Yan Liu
  2020 IEEE 36th International Conference on Data Engineering (ICDE)

Algorithm 1 (Self-paced Ensemble):
  1-2  Train f_0 on a randomly under-sampled balanced subset (|N'| = |P|).
  3-11 For i = 1..n-1:
         F_i(x)      = (1/i) * sum_{j<i} f_j(x)                  # ensemble so far
         Cut majority into k bins w.r.t. hardness H(x,y,F_i):
             B_l = {(x,y) | (l-1)/k <= H(x,y,F_i) < l/k},  H in [0,1]
         h_l         = (sum_{s in B_l} H(x_s,y_s,F_i)) / |B_l|   # avg hardness / bin
         alpha       = tan(i*pi / (2n))                          # self-paced factor
         p_l         = 1 / (h_l + alpha)                         # unnormalised bin weight
         sample from B_l : (p_l / sum_m p_m) * |P| samples
         train f_i on the newly under-sampled subset
  12   return F(x) = (1/n) * sum_m f_m(x)                        # averaged ensemble

Hardness (paper default, Sec. VI footnote 3): absolute error H(x,y,F) = |F(x)-y|,
k = 20 bins. For multi-class data we use the natural generalisation
H(x,y,F) = 1 - F(x)[y] (1 minus predicted probability of the true class), which
is EXACTLY |F(x)-y| in the binary case.

Migration note (this file):
  The model CLASS is unchanged (paper-faithful). The old per-module split/scale/
  fit/predict/evaluate harness (run_one/evaluate/main) was removed and replaced
  by the shared runner `common.base.run_model_on_dataset`. A `preencoded` flag
  lets the runner pass already-scaled + integer-encoded data, so the train-only
  scaling invariant is enforced in ONE place for all 30 models.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from common import smoke as _smoke


# ==============================================================================
# 1. Self-paced Ensemble  (class unchanged)
# ==============================================================================

class SelfPacedEnsemble:
    """Self-paced Ensemble (Liu et al., ICDE 2020, Algorithm 1).

    Multi-class generalisation: every over-represented class (count exceeds the
    minimum class size) is independently self-paced under-sampled down to the
    minority class size; all minority / already-balanced classes are kept. The
    final ensemble averages the per-classifier predicted probabilities
    (Algorithm 1, lines 4 & 12).
    """

    def __init__(self, base_estimator=None, n_estimators=30, n_bins=20,
                 hardness='absolute', random_state=42, preencoded=False):
        """
        Args:
            base_estimator: sklearn classifier used as base learner f.
                None -> RandomForestClassifier(n_estimators=30).
            n_estimators: n, ensemble size (UNIFIED = 30; paper default 10).
            n_bins: k, number of hardness bins (paper default 20).
            hardness: {'absolute','squared','entropy'} hardness function H.
            random_state: seed.
            preencoded: when True, the runner has already StandardScaled X and
                integer-encoded y to 0..C-1, so fit/predict_proba skip the
                internal scaler/label-encoder.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_bins = n_bins
        self.hardness = hardness
        self.random_state = random_state
        self.preencoded = preencoded

        self.label_encoder_ = LabelEncoder()
        self.scaler = StandardScaler()
        self.classifiers_ = []
        self.classes_ = None
        self.n_classes_ = None

    # ---- helpers -----------------------------------------------------------

    def _make_base(self, seed):
        if self.base_estimator is None:
            return RandomForestClassifier(n_estimators=30, random_state=seed, n_jobs=-1)
        clf = deepcopy(self.base_estimator)
        params = clf.get_params()
        if 'random_state' in params:
            clf.set_params(random_state=seed)
        return clf

    def _ensemble_predict_proba(self, X):
        probas = [clf.predict_proba(X) for clf in self.classifiers_]
        avg = np.mean(probas, axis=0)
        avg = np.clip(avg, 1e-12, 1.0)
        avg /= avg.sum(axis=1, keepdims=True)
        return avg

    def _hardness(self, X, y_enc):
        proba = self._ensemble_predict_proba(X)                       # (N, C)
        p_true = proba[np.arange(len(y_enc)), y_enc]                  # P(true class)
        if self.hardness == 'absolute':
            H = 1.0 - p_true
        elif self.hardness == 'squared':
            H = (1.0 - p_true) ** 2
        elif self.hardness == 'entropy':
            H = -np.log(p_true + 1e-12)
            H = H / (H.max() + 1e-12)
        else:
            raise ValueError(f'unknown hardness {self.hardness}')
        return np.clip(H, 0.0, 1.0)

    @staticmethod
    def _allocate_counts(weights, capacities, target):
        raw = weights * target
        counts = np.minimum(np.floor(raw).astype(int), capacities)
        remaining = target - int(counts.sum())
        guard = 0
        while remaining > 0 and guard < 20 * len(weights) + 20:
            cap_left = capacities - counts
            cand = np.where(cap_left > 0)[0]
            if len(cand) == 0:
                break
            order = cand[np.argsort(-weights[cand])]
            take = min(remaining, len(order))
            for j in range(take):
                counts[order[j]] += 1
                remaining -= 1
            guard += 1
        return counts

    def _random_balance_subset(self, X, y, target, rng):
        idx = []
        for c in np.unique(y):
            ic = np.where(y == c)[0]
            if len(ic) > target:
                ic = rng.choice(ic, size=target, replace=False)
            idx.append(ic)
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        return X[idx], y[idx]

    def _self_paced_subset(self, X, y, target, alpha, rng):
        H = self._hardness(X, y)
        keep = []
        for c in np.unique(y):
            ic = np.where(y == c)[0]
            if len(ic) <= target:
                keep.append(ic)
                continue
            hc = H[ic]
            bin_id = np.minimum(self.n_bins - 1, (hc * self.n_bins).astype(int))
            bins = np.unique(bin_id)
            h_bar = np.array([hc[bin_id == b].mean() for b in bins])
            weights = 1.0 / (h_bar + alpha)
            weights = weights / (weights.sum() + 1e-12)
            capacities = np.array([(bin_id == b).sum() for b in bins])
            counts = self._allocate_counts(weights, capacities, target)
            sel = []
            for j, b in enumerate(bins):
                if counts[j] <= 0:
                    continue
                members = ic[bin_id == b]
                m = int(min(counts[j], len(members)))
                if m > 0:
                    sel.append(rng.choice(members, size=m, replace=False))
            keep.append(np.concatenate(sel))
        keep = np.concatenate(keep)
        rng.shuffle(keep)
        return X[keep], y[keep]

    # ---- public API --------------------------------------------------------

    def fit(self, X, y):
        if self.preencoded:
            X_sc = np.asarray(X, dtype=float)
            y_enc = np.asarray(y).astype(int)
            self.classes_ = np.unique(y_enc)
            self.n_classes_ = int(self.classes_.max()) + 1 if len(self.classes_) else 0
        else:
            y_enc = self.label_encoder_.fit_transform(y)
            self.classes_ = self.label_encoder_.classes_
            self.n_classes_ = len(self.classes_)
            X_sc = self.scaler.fit_transform(X)

        counts = Counter(y_enc)
        target = min(counts.values())                                 # = |P|
        rng = np.random.RandomState(self.random_state)

        self.classifiers_ = []
        Xb, yb = self._random_balance_subset(X_sc, y_enc, target, rng)
        self.classifiers_.append(self._make_base(self.random_state))
        self.classifiers_[0].fit(Xb, yb)

        n = self.n_estimators
        denom = max(n - 1, 1)
        for i in range(1, n):
            alpha = np.tan(np.pi * i / (2.0 * denom))                 # line 7
            Xs, ys = self._self_paced_subset(X_sc, y_enc, target, alpha, rng)
            clf = self._make_base(self.random_state + i)
            clf.fit(Xs, ys)
            self.classifiers_.append(clf)
        return self

    def predict_proba(self, X):
        X_sc = np.asarray(X, dtype=float) if self.preencoded else self.scaler.transform(X)
        return self._ensemble_predict_proba(X_sc)

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)


# ==============================================================================
# 2. Factory + smoke driver  (unified contract for run_all.py)
# ==============================================================================

MODEL_KEY = "SPE"


def make_base_estimator(kind, rf_n_estimators, n_jobs):
    if kind == 'rf':
        return RandomForestClassifier(n_estimators=rf_n_estimators,
                                      random_state=42, n_jobs=n_jobs)
    elif kind == 'dt':
        return DecisionTreeClassifier(random_state=42)
    raise ValueError(f'unknown base estimator {kind}')


def build(random_state=42, smoke=False, base_kind='rf', n_estimators=30,
          n_bins=20, hardness='absolute', rf_n_estimators=30, n_jobs=-1, **kw):
    """Unified factory: downstream/base classifier = RandomForest(30); ensemble
    size n = 30 (paper default 10); k = 20 bins; hardness = absolute."""
    if smoke:
        n_estimators = min(n_estimators, 5)
    base = make_base_estimator(base_kind, rf_n_estimators, n_jobs)
    return SelfPacedEnsemble(base_estimator=base, n_estimators=n_estimators,
                             n_bins=n_bins, hardness=hardness,
                             random_state=random_state, preencoded=True)


if __name__ == '__main__':
    _smoke.run_smoke(build, MODEL_KEY)
