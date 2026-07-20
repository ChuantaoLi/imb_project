"""imDEF: Dynamic Ensemble Framework for imbalanced classification.

Python port of the authors' MATLAB implementation:
OREMpre/OREMgen generate floor(sqrt(n)) artificial datasets with class
distributions p**cf, SPSECE trains the classifier pool from classification-error
probabilities, SPSEEM constructs generational referee committees from negative
dynamic ensemble margins, and prediction is competence-weighted.
"""
import os
import sys
from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from common.resampler import align_proba
from common import smoke as _smoke

MODEL_KEY = "imDEF"


class _ConstantReferee:
    def __init__(self, value):
        self.value = float(value)

    def competence(self, X):
        return np.full(len(X), self.value, dtype=float)


class ImDEF:
    def __init__(
        self,
        n_estimators=100,
        q=5,
        n_bins=20,
        alpha1=None,
        alpha2=0.7,
        beta=25,
        max_depth=None,
        random_state=42,
        **kw
    ):
        self.n_estimators = int(n_estimators)
        self.q = int(q)
        self.n_bins = int(n_bins)
        self.alpha1 = alpha1
        self.alpha2 = float(alpha2)
        self.beta = int(beta)
        self.max_depth = max_depth
        self.random_state = random_state
        self.classes_ = None
        self.n_classes_ = None
        self.pool_ = []
        self.referees_ = []
        self.force_all_competent_ = False
        self.candidate_directions_ = None
        self.class_distribution_ = None

    def _train_tree(self, X, y, rng):
        clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=rng.randint(1 << 30))
        clf.fit(X, y)
        return clf

    def _tree_proba(self, clf, X):
        return align_proba(clf.predict_proba(X), self.n_classes_, clf.classes_)

    def _bootstrap_all_classes(self, y, rng):
        n = len(y)
        for _ in range(100):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            if len(np.unique(y[idx])) == len(self.classes_):
                return idx
        return np.arange(n)

    def _candidate_generation_direction(self, data_p, data_n, radius_ind):
        np_ = len(data_p)
        data = np.vstack([data_p, data_n]) if len(data_n) else data_p
        out = []
        for i in range(np_):
            dirs = []
            ordered = radius_ind[i]
            for j, cand in enumerate(ordered):
                previous = ordered[:j]
                previous_other = previous[previous >= np_]
                if len(previous_other) == 0:
                    dirs.append(int(cand))
                    continue
                mean_i = 0.5 * (data_p[i] + data[int(cand)])
                threshold = np.linalg.norm(data_p[i] - mean_i)
                d_prev = np.linalg.norm(data[previous_other] - mean_i, axis=1)
                if not np.any(d_prev - threshold < 1e-5):
                    dirs.append(int(cand))
            out.append(np.asarray(dirs, dtype=int))
        return out

    def _orem_pre(self, X, y):
        candidate_directions = {}
        sizes = []
        for cls in self.classes_:
            cls = int(cls)
            idx_p = np.where(y == cls)[0]
            idx_n = np.where(y != cls)[0]
            sizes.append(len(idx_p))
            data_p = X[idx_p]
            data_n = X[idx_n]
            np_, nn_ = len(data_p), len(data_n)
            if np_ == 0:
                candidate_directions[cls] = []
                continue
            if np_ <= 1:
                candidate_directions[cls] = [np.asarray([0], dtype=int) for _ in range(np_)]
                continue

            same_k = max(1, np_ - 1)
            same_dist, same_idx = NearestNeighbors(n_neighbors=same_k + 1).fit(data_p).kneighbors(data_p)
            same_dist = same_dist[:, 1:]
            same_idx = same_idx[:, 1:]
            if nn_:
                other_dist, other_idx = NearestNeighbors(n_neighbors=nn_).fit(data_n).kneighbors(data_p)
            else:
                other_dist = np.empty((np_, 0))
                other_idx = np.empty((np_, 0), dtype=int)

            radius_ind = []
            for i in range(np_):
                dis_i = np.concatenate([same_dist[i], other_dist[i]])
                ind_i = np.concatenate([same_idx[i], other_idx[i] + np_])
                order = np.argsort(dis_i)
                sorted_ind = ind_i[order]
                count_break = 0
                chosen = sorted_ind
                for pos, original_pos in enumerate(order):
                    if original_pos >= same_dist.shape[1]:
                        count_break += 1
                    else:
                        count_break = 0
                    if count_break >= self.q:
                        chosen = sorted_ind[: max(pos + 1 - self.q, 1)]
                        break
                radius_ind.append(np.asarray(chosen, dtype=int))
            candidate_directions[cls] = self._candidate_generation_direction(data_p, data_n, radius_ind)
        sizes = np.asarray(sizes, dtype=float)
        return candidate_directions, sizes / np.sum(sizes)

    def _focus_oversample(self, sample, data_p, data_n, direction, rng):
        if len(direction) == 0:
            return sample.copy()
        data = np.vstack([data_p, data_n]) if len(data_n) else data_p
        cand = data[int(rng.choice(direction))]
        gap = rng.rand(sample.shape[0])
        return sample + gap * (cand - sample)

    def _orem_gen_class(self, X, y, cls, ns, rng):
        idx_p = np.where(y == cls)[0]
        idx_n = np.where(y != cls)[0]
        data_p = X[idx_p]
        data_n = X[idx_n]
        if ns <= 0 or len(data_p) == 0:
            return np.empty((0, X.shape[1]))
        seed_idx = rng.choice(np.arange(len(data_p)), size=ns, replace=True)
        directions = self.candidate_directions_[int(cls)]
        out = np.empty((ns, X.shape[1]), dtype=float)
        for i, s in enumerate(seed_idx):
            out[i] = self._focus_oversample(data_p[int(s)], data_p, data_n, directions[int(s)], rng)
        return out

    def _orem_gen(self, X, y, power, rng):
        dist = np.power(np.maximum(self.class_distribution_, 1e-12), power)
        dist = dist / np.sum(dist)
        sizes = np.round(len(y) * dist).astype(int)
        parts_x, parts_y = [], []
        for cls, ns in zip(self.classes_, sizes):
            gen = self._orem_gen_class(X, y, int(cls), int(ns), rng)
            if len(gen):
                parts_x.append(gen)
                parts_y.append(np.full(len(gen), int(cls), dtype=y.dtype))
        Xg = np.vstack(parts_x)
        yg = np.concatenate(parts_y)
        order = rng.permutation(len(yg))
        return Xg[order], yg[order]

    def _difficulty(self, proba, y):
        return 1.0 - proba[np.arange(len(y)), y]

    def _cut_to_bins(self, h, b):
        if len(h) == 0:
            return [], np.array([])
        if np.max(h) <= np.min(h):
            return [np.arange(len(h))], np.array([float(np.mean(h))])
        bins, ahard = [], []
        edges = np.linspace(float(np.min(h)), float(np.max(h)), b + 1)
        edges[-1] += 0.01
        for i in range(b):
            idx = np.where((h < edges[i + 1]) & (h >= edges[i]))[0]
            bins.append(idx)
            ahard.append(np.nan if len(idx) == 0 else float(np.mean(h[idx])))
        return bins, np.asarray(ahard, dtype=float)

    def _sample_from_bins(self, bins, pl, n, y_required, rng, weights=None, use_ceil=False):
        valid = np.where(np.isfinite(pl) & (pl > 0))[0]
        if len(valid) == 0:
            return rng.choice(np.arange(len(y_required)), size=n, replace=True)
        for _ in range(100):
            chosen = []
            for bi in valid:
                count = int(np.ceil(pl[bi] * n)) if use_ceil else int(round(pl[bi] * n))
                if count <= 0 or len(bins[bi]) == 0:
                    continue
                p = None
                if weights is not None:
                    w = np.asarray(weights[bins[bi]], dtype=float)
                    if w.sum() > 0:
                        p = w / w.sum()
                chosen.append(rng.choice(bins[bi], size=count, replace=True, p=p))
            if not chosen:
                continue
            idx = np.concatenate(chosen)
            if len(np.unique(y_required[idx])) >= min(2, len(np.unique(y_required))):
                return idx
        return rng.choice(np.arange(len(y_required)), size=n, replace=True)

    def _spsece(self, X, y, nt, alpha1, rng):
        models = []
        n = len(y)
        init_idx = self._bootstrap_all_classes(y, rng)
        c0 = self._train_tree(X[init_idx], y[init_idx], rng)
        hardness = [self._difficulty(self._tree_proba(c0, X), y)]
        if nt <= 1:
            sr = np.ones(1)
        else:
            sr = 1.0 - np.arange(nt) / (alpha1 * (nt - 1))
        for t in range(nt):
            h = np.mean(np.abs(np.vstack(hardness)), axis=0)
            bins, ahard = self._cut_to_bins(h, self.n_bins)
            pl = np.power(1.0 / (1e-2 + ahard), sr[t])
            pl = pl / np.nansum(pl)
            idx = self._sample_from_bins(bins, pl, n, y, rng)
            clf = self._train_tree(X[idx], y[idx], rng)
            models.append(clf)
            hardness.append(self._difficulty(self._tree_proba(clf, X), y))
        return models

    def _spseem(self, X, y, cl, rng):
        n, t_count = cl.shape
        competent = []
        trainable_cols = []
        for t in range(t_count):
            rate = float(np.mean(cl[:, t]))
            if rate >= 0.99:
                competent.append([_ConstantReferee(1.0)])
            elif rate <= 0.01:
                competent.append([_ConstantReferee(0.0)])
            else:
                competent.append(None)
                trainable_cols.append(t)
        if not trainable_cols:
            return competent

        cl_sub = cl[:, trainable_cols]
        committees = self._do_spseem(X, y, cl_sub, rng)
        for col, committee in zip(trainable_cols, committees):
            competent[col] = committee
        return competent

    def _do_spseem(self, X, y, cl, rng):
        n, t_count = cl.shape
        temp = cl.astype(float).copy()
        temp[temp == 1] = -1.0
        temp[temp == 0] = 1.0
        hardness = [np.sum(temp, axis=1)]
        if self.beta <= 1:
            sg = np.ones(1)
        else:
            sg = 1.0 - np.arange(self.beta) / (self.alpha2 * (self.beta - 1))

        cnt = Counter(y.tolist())
        max_sqrt = max(np.sqrt(v) for v in cnt.values())
        sw = np.array([max_sqrt / np.sqrt(cnt[int(c)]) for c in y], dtype=float)
        committees = [[] for _ in range(t_count)]

        for r in range(self.beta):
            h = np.mean(np.vstack(hardness), axis=0)
            if np.max(h) > np.min(h):
                h = (h - np.min(h)) / (np.max(h) - np.min(h))
            bins, ahard = self._cut_to_bins(h, self.n_bins)
            pl = np.power(1.0 / (1e-2 + ahard), sg[r])
            pl = pl / np.nansum(pl)
            predict_prob_competent = np.zeros((n, t_count), dtype=float)
            for t in range(t_count):
                label_t = cl[:, t].astype(int)
                idx = self._sample_from_bins(bins, pl, n, label_t, rng, weights=sw, use_ceil=True)
                if len(np.unique(label_t[idx])) < 2:
                    ref = _ConstantReferee(float(np.mean(label_t)))
                else:
                    ref = self._train_tree(X[idx], label_t[idx], rng)
                committees[t].append(ref)
                if isinstance(ref, _ConstantReferee):
                    predict_prob_competent[:, t] = ref.competence(X)
                else:
                    predict_prob_competent[:, t] = align_proba(ref.predict_proba(X), 2, ref.classes_)[:, 1]
            hardness.append(np.sum(predict_prob_competent * temp, axis=1))
        return committees

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.max()) + 1 if len(self.classes_) else 0
        alpha1 = self.alpha1
        if alpha1 is None:
            alpha1 = 1.0 if len(self.classes_) == 2 else 0.8
        rng = np.random.RandomState(self.random_state)

        self.candidate_directions_, self.class_distribution_ = self._orem_pre(X, y)
        m = max(1, int(np.floor(np.sqrt(self.n_estimators))))
        powers = np.linspace(1.0, -1.0, m)
        base = self.n_estimators // m
        rem = self.n_estimators - m * base
        self.pool_ = []
        for r, power in enumerate(powers):
            nt = base + (1 if (r + 1) <= rem else 0)
            Xg, yg = self._orem_gen(X, y, power, rng)
            self.pool_.extend(self._spsece(Xg, yg, nt, alpha1, rng))
        self.pool_ = self.pool_[: self.n_estimators]

        pred_train = np.column_stack([clf.predict(X) for clf in self.pool_])
        cl = (pred_train == y[:, None]).astype(int)
        hc = np.sum(cl, axis=1)
        valid = np.where(hc <= self.n_estimators * 0.95)[0]
        if len(valid) == 0:
            self.force_all_competent_ = True
            self.referees_ = []
        else:
            self.force_all_competent_ = False
            self.referees_ = self._spseem(X[valid], y[valid], cl[valid], rng)
        return self

    def _competence_matrix(self, X):
        if self.force_all_competent_:
            return np.ones((len(X), len(self.pool_)), dtype=float)
        comp = np.zeros((len(X), len(self.pool_)), dtype=float)
        for t, committee in enumerate(self.referees_):
            vals = np.zeros(len(X), dtype=float)
            for ref in committee:
                if isinstance(ref, _ConstantReferee):
                    vals += ref.competence(X)
                else:
                    vals += align_proba(ref.predict_proba(X), 2, ref.classes_)[:, 1]
            comp[:, t] = vals / max(1, len(committee))
        return comp

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        comp = self._competence_matrix(X)
        p = np.zeros((len(X), self.n_classes_), dtype=float)
        for t, clf in enumerate(self.pool_):
            p += self._tree_proba(clf, X) * comp[:, [t]]
        sums = p.sum(axis=1, keepdims=True)
        p = np.divide(p, sums, out=np.full_like(p, 1.0 / max(1, self.n_classes_)), where=sums > 0)
        return p

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def build(random_state=42, smoke=False, **kw):
    return ImDEF(random_state=random_state, **kw)


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
