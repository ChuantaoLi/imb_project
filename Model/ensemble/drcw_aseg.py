"""DRCW-ASEG — Distance-based Relative Competence Weighting with Adaptive
Synthetic Example Generation (Zhang et al., Neurocomputing 2018).

"DRCW-ASEG: one-versus-one distance-based relative competence weighting with
adaptive synthetic example generation for multi-class imbalanced datasets."

Mechanism:
  * decompose the multiclass task into one-versus-one binary sub-problems;
  * for each class pair, train a bagged pool of CART classifiers on the pairwise
    data via bootstrap sampling;
  * at test time, for each pairwise problem, find the local neighbours inside the
    pairwise training set and run ASEG to synthesize minority examples whenever
    the local region is imbalanced;
  * estimate each classifier's relative competence by its distance-weighted
    correctness on that pairwise local region, then aggregate the selected pair
    posterior into a global multiclass score.
"""

import os
import sys
from itertools import combinations
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.neighbors import smote_generate
from common import smoke as _smoke

MODEL_KEY = "drcw_aseg"


class DRCWASEG:
    def __init__(self, n_estimators=30, k=7, imb_thresh=1.5, random_state=42, **kw):
        self.n_estimators = int(n_estimators)
        self.k = int(k)
        self.imb_thresh = float(imb_thresh)
        self.random_state = random_state

    def _bootstrap_pair_indices(self, y_pair, rng):
        n = len(y_pair)
        if n <= 1:
            return np.arange(n)
        for _ in range(32):
            idx = rng.randint(0, n, n)
            if len(np.unique(y_pair[idx])) == 2:
                return idx
        # Rare safeguard for extremely skewed tiny pair sets.
        idx0 = np.where(y_pair == 0)[0]
        idx1 = np.where(y_pair == 1)[0]
        forced = [rng.choice(idx0), rng.choice(idx1)]
        extra = rng.randint(0, n, max(0, n - 2))
        return np.concatenate([np.asarray(forced, dtype=int), extra])

    def _align_binary_proba(self, clf, x):
        raw = clf.predict_proba(x)[0]
        proba = np.zeros(2, dtype=float)
        for j, c in enumerate(clf.classes_):
            proba[int(c)] = raw[j]
        return proba

    def _distance_weights(self, dist):
        w = np.exp(-np.asarray(dist, dtype=float))
        return w / np.clip(w.sum(), 1e-12, None)

    def _pair_neighbors(self, model, x):
        dist, nb = model["nn"].kneighbors(x.reshape(1, -1))
        nb = nb[0]
        dist = dist[0]
        return model["X"][nb], model["y"][nb], dist

    def _minority_seeds(self, model, neigh_X, neigh_y, minority_label, x):
        seeds = neigh_X[neigh_y == minority_label]
        if len(seeds) > 0:
            return seeds
        full = model["X"][model["y"] == minority_label]
        if len(full) == 0:
            return full
        order = np.argsort(np.linalg.norm(full - x.reshape(1, -1), axis=1))
        take = min(max(1, self.k), len(full))
        return full[order[:take]]

    def _augment_local_region(self, model, x, neigh_X, neigh_y, dist, rng):
        cnt = np.bincount(neigh_y, minlength=2)
        maj = int(np.argmax(cnt))
        minc = 1 - maj
        maj_n = int(cnt[maj])
        min_n = int(cnt[minc])
        if maj_n <= 0:
            return neigh_X, neigh_y, dist
        if maj_n <= self.imb_thresh * max(1, min_n):
            return neigh_X, neigh_y, dist

        need = maj_n - min_n
        seeds = self._minority_seeds(model, neigh_X, neigh_y, minc, x)
        if len(seeds) == 0:
            return neigh_X, neigh_y, dist
        gen = smote_generate(seeds, need, min(self.k, max(1, len(seeds))), rng)
        if len(gen) == 0:
            return neigh_X, neigh_y, dist
        gen_dist = np.linalg.norm(gen - x.reshape(1, -1), axis=1)
        augX = np.vstack([neigh_X, gen])
        augy = np.concatenate([neigh_y, np.full(len(gen), minc, dtype=neigh_y.dtype)])
        augd = np.concatenate([dist, gen_dist])
        return augX, augy, augd

    def _pair_competence(self, model, x, rng):
        neigh_X, neigh_y, dist = self._pair_neighbors(model, x)
        augX, augy, augd = self._augment_local_region(model, x, neigh_X, neigh_y, dist, rng)
        w_dist = self._distance_weights(augd)
        comp = np.array([np.dot((clf.predict(augX) == augy).astype(float), w_dist) for clf in model["clfs"]], dtype=float)
        if comp.sum() <= 0:
            comp = np.ones(len(model["clfs"]), dtype=float)
        return comp / np.clip(comp.sum(), 1e-12, None)

    def fit(self, X, y):
        self.X_ = np.asarray(X, float)
        self.y_ = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(self.y_)
        self.n_classes_ = len(self.classes_)
        self.class_to_index_ = {int(c): i for i, c in enumerate(self.classes_)}
        rng = np.random.RandomState(self.random_state)
        self.pairs_ = []
        self.models_ = {}

        for pair_id, (ca, cb) in enumerate(combinations(self.classes_, 2)):
            mask = np.isin(self.y_, [ca, cb])
            X_pair = self.X_[mask]
            y_pair = (self.y_[mask] == cb).astype(int)
            clfs = []
            for i in range(self.n_estimators):
                idx = self._bootstrap_pair_indices(y_pair, rng)
                clf = DecisionTreeClassifier(random_state=self.random_state + pair_id * 100 + i)
                clfs.append(clf.fit(X_pair[idx], y_pair[idx]))
            nn = NearestNeighbors(n_neighbors=min(self.k, len(X_pair))).fit(X_pair)
            pair = (int(ca), int(cb))
            self.pairs_.append(pair)
            self.models_[pair] = {"X": X_pair, "y": y_pair, "clfs": clfs, "nn": nn}
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        rng = np.random.RandomState(self.random_state)
        out = np.zeros((len(X), self.n_classes_))
        for ti, x in enumerate(X):
            score = np.zeros(self.n_classes_, dtype=float)
            for pair in self.pairs_:
                ca, cb = pair
                model = self.models_[pair]
                w = self._pair_competence(model, x, rng)
                pair_proba = np.zeros(2, dtype=float)
                for clf, wi in zip(model["clfs"], w):
                    pair_proba += wi * self._align_binary_proba(clf, x.reshape(1, -1))
                pair_proba /= np.clip(pair_proba.sum(), 1e-12, None)
                score[self.class_to_index_[ca]] += pair_proba[0]
                score[self.class_to_index_[cb]] += pair_proba[1]
            out[ti] = score
        out /= out.sum(1, keepdims=True) + 1e-12
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


def build(random_state=42, smoke=False, n_estimators=30, k=7, **kw):
    if smoke:
        n_estimators = min(n_estimators, 8)
    return DRCWASEG(
        n_estimators=n_estimators,
        k=k,
        random_state=random_state,
        **kw,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
