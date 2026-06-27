"""MC-EVHS — Evidential Hybrid re-Sampling (Grina, Elouedi, Lefevre, IPMU 2022).

"Evidential hybrid re-sampling for multi-class imbalanced data."

Mechanism (Dempster-Shafer evidence theory):
  * Each instance is assigned a SOFT EVIDENTIAL LABEL built from distances to
    class centers, eligible meta-class centers (overlapping regions), and an
    outlier proposition. This follows the paper's CCR-style evidential
    membership construction, rather than a k-NN vote.
  * Majority classes above the mean size are adaptively undersampled by
    removing the most ambiguous / noisy observations, while never going below
    the mean class size.
  * Minority classes below the mean size are oversampled only from borderline
    objects, i.e. samples with high meta-class support. New objects are then
    generated with a SMOTE-style interpolation restricted to these evidential
    border regions.

The evidential label construction follows the paper and its cited CCR rule:
Mahalanobis distances to class / meta-class centers, a bounded number of
meta-classes, and an explicit outlier proposition.
"""

import os
import sys
import numpy as np
from itertools import combinations
from collections import Counter

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common.neighbors import knn_indices
from common import smoke as _smoke

MODEL_KEY = "mc_evhs"


class MCEVHS(Resampler):
    def __init__(self, k=5, alpha=1.0, t=5, ridge=1e-6, rf_n_estimators=30, n_jobs=-1, random_state=42, **kw):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        self.k = k
        self.alpha = float(alpha)
        self.t = int(t)
        self.ridge = float(ridge)

    def _cov_inv(self, Xc):
        Xc = np.asarray(Xc, dtype=float)
        d = Xc.shape[1]
        if len(Xc) <= 1:
            return np.eye(d, dtype=float)
        cov = np.cov(Xc, rowvar=False)
        if np.ndim(cov) == 0:
            cov = np.array([[float(cov)]], dtype=float)
        cov = np.asarray(cov, dtype=float) + self.ridge * np.eye(d, dtype=float)
        return np.linalg.pinv(cov)

    def _mahalanobis_batch(self, X, center, inv_cov):
        diff = np.asarray(X, dtype=float) - np.asarray(center, dtype=float)
        d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        return np.sqrt(np.clip(d2, 0.0, None))

    def _build_focals(self, X, y, classes):
        classes = list(classes)
        centers = {}
        inv_covs = {}
        for c in classes:
            Xc = np.asarray(X[y == c], dtype=float)
            center = Xc.mean(axis=0)
            inv_cov = self._cov_inv(Xc)
            centers[(c,)] = center
            inv_covs[(c,)] = inv_cov

        focals = []
        max_size = min(max(2, self.t), len(classes))
        for size in range(2, max_size + 1):
            for U in combinations(classes, size):
                center_u = np.mean([centers[(c,)] for c in U], axis=0)
                member_dist = [np.linalg.norm(center_u - centers[(c,)]) for c in U]
                other = [c for c in classes if c not in U]
                other_dist = [np.linalg.norm(center_u - centers[(c,)]) for c in other]
                if other_dist and max(member_dist) >= min(other_dist):
                    continue
                Xu = np.asarray(X[np.isin(y, list(U))], dtype=float)
                centers[U] = center_u
                inv_covs[U] = self._cov_inv(Xu)
                focals.append(U)
        return centers, inv_covs, focals

    def _mass_functions(self, X, y, classes):
        classes = np.asarray(classes)
        centers, inv_covs, meta_focals = self._build_focals(X, y, classes)
        singleton_keys = [(c,) for c in classes]
        singleton_dist = {key: self._mahalanobis_batch(X, centers[key], inv_covs[key]) for key in singleton_keys}
        class_scales = []
        for key in singleton_keys:
            own = singleton_dist[key][y == key[0]]
            if len(own) > 0:
                class_scales.append(np.median(own))
        outlier_scale = float(np.median(class_scales)) if class_scales else 1.0
        outlier_scale = max(outlier_scale, 1e-12)

        masses = []
        for i in range(len(X)):
            score = {}
            own_singleton = {key: singleton_dist[key][i] for key in singleton_keys}
            for key, d in own_singleton.items():
                score[key] = 1.0 / np.clip(d, 1e-12, None) ** self.alpha
            for key in meta_focals:
                d_meta = float(self._mahalanobis_batch(X[i : i + 1], centers[key], inv_covs[key])[0])
                member_d = np.array([own_singleton[(c,)] for c in key], dtype=float)
                balance = 1.0 / (1.0 + np.std(member_d) / np.clip(member_d.mean(), 1e-12, None))
                score[key] = balance / np.clip(d_meta, 1e-12, None) ** self.alpha
            min_single = min(own_singleton.values())
            outlier_score = max(min_single / outlier_scale - 1.0, 0.0) ** self.alpha
            if outlier_score > 0:
                score[("outlier",)] = outlier_score
            total = sum(score.values())
            if total <= 0:
                total = 1.0
            masses.append({k: v / total for k, v in score.items()})
        return masses

    @staticmethod
    def _class_scores(bba, cls):
        own = float(bba.get((cls,), 0.0))
        meta = float(sum(v for k, v in bba.items() if len(k) > 1 and cls in k))
        out = float(bba.get(("outlier",), 0.0))
        return own, meta, out

    def _undersample_indices(self, idx, masses, cls, target):
        if len(idx) <= target:
            return idx
        scored = []
        for ii in idx:
            own, meta, out = self._class_scores(masses[ii], cls)
            ambiguity = meta + out - own
            scored.append((ii, ambiguity, own))
        scored.sort(key=lambda t: (t[1], -t[2]))
        ambiguous = [ii for ii, ambiguity, _ in scored if ambiguity > 0]
        n_remove = min(len(idx) - target, len(ambiguous))
        if n_remove <= 0:
            return idx
        remove = set(ambiguous[:n_remove])
        return np.array([ii for ii in idx if ii not in remove], dtype=int)

    def _oversample(self, Xc, masses_c, cls, n_gen, rng):
        if n_gen <= 0 or len(Xc) == 0:
            return np.empty((0, Xc.shape[1]), dtype=float)
        if len(Xc) == 1:
            return np.repeat(Xc, n_gen, axis=0)
        own = np.array([self._class_scores(bba, cls)[0] for bba in masses_c], dtype=float)
        meta = np.array([self._class_scores(bba, cls)[1] for bba in masses_c], dtype=float)
        out = np.array([self._class_scores(bba, cls)[2] for bba in masses_c], dtype=float)
        border = np.where((meta >= own) & (meta >= out))[0]
        if len(border) == 0:
            border = np.argsort(own)[: max(1, min(len(Xc), self.k))]
        w = np.clip(meta[border] + (1.0 - own[border]), 1e-12, None)
        w = w / np.clip(w.sum(), 1e-12, None)
        picked = rng.choice(border, size=n_gen, replace=True, p=w)
        nn_same = knn_indices(Xc, self.k)
        if nn_same.shape[1] <= 0:
            return Xc[picked].copy()
        neigh = nn_same[picked, rng.randint(0, nn_same.shape[1], size=n_gen)]
        gaps = rng.rand(n_gen, 1)
        return Xc[picked] + gaps * (Xc[neigh] - Xc[picked])

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        classes = np.unique(y)
        masses = self._mass_functions(X, y, classes)
        cnt = Counter(y.tolist())
        target = int(round(np.mean(list(cnt.values()))))
        Xl, yl = [], []
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) > target:
                keep = self._undersample_indices(idx, masses, c, target)
                Xl.append(X[keep])
                yl.append(y[keep])
            else:
                Xl.append(X[idx])
                yl.append(y[idx])
                n_gen = target - len(idx)
                if n_gen > 0:
                    gen = self._oversample(
                        X[idx],
                        [masses[ii] for ii in idx],
                        c,
                        n_gen,
                        rng,
                    )
                    Xl.append(gen)
                    yl.append(np.full(len(gen), c, dtype=y.dtype))
        return np.vstack(Xl), np.concatenate(yl)


def build(random_state=42, smoke=False, k=5, alpha=1.0, t=5, rf_n_estimators=30, n_jobs=-1, **kw):
    return MCEVHS(
        k=k,
        alpha=alpha,
        t=t,
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
