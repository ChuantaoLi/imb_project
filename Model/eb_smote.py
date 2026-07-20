"""EB-SMOTE: Expandable Borderline SMOTE.

The TKDE paper is binary; this implementation applies the method one-vs-rest
for each underrepresented class.  It selects borderline target and largest-class
samples, then generates convex-combination samples that expand both borderlines
toward each other.
"""
import os
import sys
from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from common.resampler import Resampler
from common import smoke as _smoke

MODEL_KEY = "EB-SMOTE"


class EBSMOTE(Resampler):
    def __init__(
        self,
        m_alpha=5,
        m_beta=5,
        lambda_border=2.0,
        v_alpha=5,
        w_alpha=5,
        v_beta=5,
        w_beta=5,
        expand_alpha=0.5,
        expand_beta=0.3,
        resampled_ratio=1.0,
        rf_n_estimators=30,
        n_jobs=-1,
        random_state=42,
        **kw
    ):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        self.m_alpha = int(m_alpha)
        self.m_beta = int(m_beta)
        self.lambda_border = float(lambda_border)
        self.v_alpha = int(v_alpha)
        self.w_alpha = int(w_alpha)
        self.v_beta = int(v_beta)
        self.w_beta = int(w_beta)
        self.expand_alpha = float(expand_alpha)
        self.expand_beta = float(expand_beta)
        self.resampled_ratio = float(resampled_ratio)

    def _border_indices(self, X, y, cls, opposite_cls, m):
        idx = np.where(y == cls)[0]
        if len(idx) == 0 or len(X) <= 1:
            return idx[:0]
        kk = min(m, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
        neigh = nn.kneighbors(X[idx], return_distance=False)[:, 1:]
        opp = np.sum(y[neigh] == opposite_cls, axis=1)
        lo = int(np.ceil(kk / max(self.lambda_border, 1e-12)))
        mask = (opp >= lo) & (opp < kk)
        return idx[mask]

    def _choose_nearest(self, src, pool, k, rng):
        if len(pool) == 0:
            return None
        kk = min(max(1, k), len(pool))
        _, idx = NearestNeighbors(n_neighbors=kk).fit(pool).kneighbors(src.reshape(1, -1))
        return pool[int(rng.choice(idx[0]))]

    def _minority_sample(self, pi, border_min, border_maj, rng):
        tj = self._choose_nearest(pi, border_maj, self.v_alpha, rng)
        if tj is None:
            return pi.copy()
        pk = self._choose_nearest(tj, border_min, self.w_alpha, rng)
        if pk is None:
            pk = pi
        lam = rng.rand()
        inner = lam * pi + (1.0 - lam) * pk
        return (1.0 - self.expand_alpha) * inner + self.expand_alpha * tj

    def _majority_sample(self, ti, border_min, border_maj, rng):
        pj = self._choose_nearest(ti, border_min, self.v_beta, rng)
        if pj is None:
            return ti.copy()
        tk = self._choose_nearest(pj, border_maj, self.w_beta, rng)
        if tk is None:
            tk = ti
        lam = rng.rand()
        inner = lam * ti + (1.0 - lam) * tk
        return (1.0 - self.expand_beta) * inner + self.expand_beta * pj

    def _resample_pair(self, X, y, cls, majority_cls, rng):
        border_min_idx = self._border_indices(X, y, cls, majority_cls, self.m_alpha)
        border_maj_idx = self._border_indices(X, y, majority_cls, cls, self.m_beta)
        border_min = X[border_min_idx]
        border_maj = X[border_maj_idx]
        Xc = X[y == cls]
        Xmaj = X[y == majority_cls]

        if len(border_min) == 0:
            return np.empty((0, X.shape[1])), np.empty((0, X.shape[1]))
        if len(border_min) == 0 or len(border_maj) == 0:
            return np.empty((0, X.shape[1])), np.empty((0, X.shape[1]))

        gen_maj = np.empty((len(border_maj), X.shape[1]), dtype=float)
        for i, ti in enumerate(border_maj):
            gen_maj[i] = self._majority_sample(ti, border_min, border_maj, rng)

        gnum = (self.resampled_ratio * (len(Xmaj) + len(border_maj)) - len(Xc)) / len(border_min)
        gnum = max(0, int(np.ceil(gnum)))
        gen_min = np.empty((len(border_min) * gnum, X.shape[1]), dtype=float)
        pos = 0
        for pi in border_min:
            if len(border_maj) <= self.v_alpha:
                chosen_majority = rng.choice(np.arange(len(border_maj)), size=gnum, replace=True)
            else:
                _, near = NearestNeighbors(n_neighbors=self.v_alpha).fit(border_maj).kneighbors(pi.reshape(1, -1))
                chosen_majority = rng.choice(near[0], size=gnum, replace=True)
            for mj in chosen_majority:
                tj = border_maj[int(mj)]
                pk = self._choose_nearest(tj, border_min, self.w_alpha, rng)
                if pk is None:
                    pk = pi
                lam = rng.rand()
                inner = lam * pi + (1.0 - lam) * pk
                gen_min[pos] = (1.0 - self.expand_alpha) * inner + self.expand_alpha * tj
                pos += 1
        return gen_min, gen_maj

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        counts = Counter(y.tolist())
        target = int(max(counts.values()))
        majority_cls = int(max(counts.items(), key=lambda kv: kv[1])[0])

        parts_x, parts_y = [X], [y]
        for cls in sorted(counts):
            if cls == majority_cls:
                continue
            gen_min, gen_maj = self._resample_pair(X, y, cls, majority_cls, rng)
            if len(gen_min):
                parts_x.append(gen_min)
                parts_y.append(np.full(len(gen_min), cls, dtype=y.dtype))
            if len(gen_maj):
                parts_x.append(gen_maj)
                parts_y.append(np.full(len(gen_maj), majority_cls, dtype=y.dtype))
        return np.vstack(parts_x), np.concatenate(parts_y)


def build(random_state=42, smoke=False, rf_n_estimators=30, n_jobs=-1, **kw):
    return EBSMOTE(
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
