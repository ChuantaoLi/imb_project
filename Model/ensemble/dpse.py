"""DPSE — Differential Partition Sampling Ensemble (Gao et al., EAAI 2021).

"A multiclass classification using one-versus-all approach with the differential
partition sampling ensemble."

Mechanism (OvA + differential partition sampling + confidence weighting):
  * decompose the multiclass problem into C one-versus-all binary problems;
  * partition samples into safe / borderline / rare / outlier regions from the
    mixed-neighbourhood composition;
  * create several balanced binary training subsets using an arithmetic sequence
    of differential sampling sizes; each subset applies s-random undersampling
    to majority safe samples and br-SMOTE to minority borderline/rare samples;
  * aggregate the OvA heads by class-confidence weighting.
"""
import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.neighbors import smote_generate
from common import smoke as _smoke

MODEL_KEY = "dpse"


class DPSE:
    def __init__(self, n_subsets=3, k=5, random_state=42, **kw):
        self.n_subsets = n_subsets
        self.k = k
        self.random_state = random_state

    def _mixed_neighbors(self, X):
        from sklearn.neighbors import NearestNeighbors
        kk = min(self.k, max(len(X) - 1, 1))
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
        _, idx = nn.kneighbors(X)
        return idx[:, 1:]

    def _categorize_samples(self, X, yb):
        idx = self._mixed_neighbors(X)
        nb_lab = yb[idx]
        same = (nb_lab == yb[:, None]).sum(axis=1)
        kk = idx.shape[1]
        if kk <= 0:
            return np.full(len(X), "safe", dtype=object)

        # Paper categories are based on neighbourhood composition.
        cat = np.full(len(X), "borderline", dtype=object)
        cat[same == kk] = "safe"
        cat[same == 0] = "outlier"
        rare_hi = max(1, kk // 2 - 1)
        rare_mask = (same > 0) & (same <= rare_hi)
        cat[rare_mask] = "rare"
        border_mask = (same > rare_hi) & (same < kk)
        cat[border_mask] = "borderline"
        return cat

    def _arithmetic_targets(self, n_pos, n_neg):
        lo, hi = sorted((int(n_pos), int(n_neg)))
        if self.n_subsets <= 1 or lo == hi:
            return np.array([max(2, lo)], dtype=int)
        seq = np.linspace(lo, hi, self.n_subsets)
        return np.maximum(2, np.round(seq).astype(int))

    def _br_smote(self, X, seed_idx, n_gen, rng):
        if n_gen <= 0:
            return np.empty((0, X.shape[1]), dtype=float)
        if len(seed_idx) == 0:
            seed_idx = np.arange(len(X))
        X_seed = X[seed_idx]
        return smote_generate(X_seed, n_gen, self.k, rng)

    def _s_random_undersample(self, idx, cats, target, rng):
        safe = idx[cats[idx] == "safe"]
        hard = idx[cats[idx] != "safe"]
        keep = []
        keep_hard = min(len(hard), target)
        if keep_hard > 0:
            keep.extend(rng.choice(hard, size=keep_hard, replace=False).tolist())
        remain = target - len(keep)
        if remain > 0:
            src = safe if len(safe) > 0 else hard
            if len(src) > 0:
                take = min(len(src), remain)
                keep.extend(rng.choice(src, size=take, replace=False).tolist())
        if len(keep) < target:
            src = idx if len(idx) > 0 else np.array([], dtype=int)
            if len(src) > 0:
                extra = rng.choice(src, size=target - len(keep), replace=True)
                keep.extend(extra.tolist())
        return np.asarray(keep, dtype=int)

    def _ova_subset(self, X, yb, rng, target):
        """Build one resampled OvA binary subset at differential target size.

        Majority class uses s-random undersampling biased toward removing safe
        examples first. Minority class uses br-SMOTE, i.e. synthesis from the
        borderline/rare region before falling back to the full positive set.
        """
        idx_pos = np.where(yb == 1)[0]
        idx_neg = np.where(yb == 0)[0]
        cats = self._categorize_samples(X, yb)

        if len(idx_pos) >= target:
            keep_pos = rng.choice(idx_pos, size=target, replace=False)
            Xp = X[keep_pos]
        else:
            seed_idx = idx_pos[np.isin(cats[idx_pos], ["borderline", "rare"])]
            extra = self._br_smote(X, seed_idx, target - len(idx_pos), rng)
            Xp = np.vstack([X[idx_pos], extra]) if len(extra) else X[idx_pos]

        keep_neg = self._s_random_undersample(idx_neg, cats, target, rng)
        Xn = X[keep_neg]
        return np.vstack([Xp, Xn]), np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.max()) + 1
        rng = np.random.RandomState(self.random_state)
        self.heads_ = {c: [] for c in self.classes_}
        for ci, c in enumerate(self.classes_):
            yb = (y == c).astype(int)
            targets = self._arithmetic_targets(int((yb == 1).sum()), int((yb == 0).sum()))
            for s in range(self.n_subsets):
                target = int(targets[min(s, len(targets) - 1)])
                Xs, ys = self._ova_subset(X, yb, rng, target)
                self.heads_[c].append(DecisionTreeClassifier(
                    random_state=self.random_state + ci * 10 + s).fit(Xs, ys))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        scores = np.zeros((len(X), self.n_classes_))
        for ci, c in enumerate(self.classes_):
            ps = [h.predict_proba(X)[:, 1] if len(h.classes_) == 2 else
                  h.predict_proba(X)[:, list(h.classes_).index(1)] if 1 in h.classes_ else
                  np.zeros(len(X)) for h in self.heads_[c]]
            scores[:, ci] = np.mean(ps, axis=0)
        scores = np.clip(scores, 0, None)
        s = scores.sum(1, keepdims=True) + 1e-12
        return scores / s

    def predict(self, X):
        return np.argmax(self.predict_proba(X), 1)


def build(random_state=42, smoke=False, n_subsets=3, k=5, **kw):
    if smoke:
        n_subsets = 2
    return DPSE(n_subsets=n_subsets, k=k, random_state=random_state)


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
