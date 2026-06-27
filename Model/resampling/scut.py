"""SCUT — SMOTE + cluster-based undersampling (Agrawal et al., IC3K 2015).

"SCUT: Multi-Class Imbalanced Data Classification Using SMOTE and Cluster-based
Undersampling."

Mechanism (hybrid resampling, per class):
  * MAJORITY classes: cluster with EM / Gaussian mixtures, then draw a
    cluster-stratified random subset so every cluster remains represented after
    undersampling.
  * MINORITY classes: SMOTE oversample up to the common target size.
Target = mean class size (SCUT balances every class toward the average).

Implementation note:
  * The original paper states EM clustering but does not publish executable
    code. A later public `scutr` implementation realises SCUT with
    `Mclust(data, G=1:9)` and then performs cluster-stratified sampling. We
    mirror that behaviour as closely as sklearn allows by selecting the best
    GaussianMixture via BIC over covariance families and `k in [1, ..., 9]`
    (capped by class size).
"""
import os
import sys
import numpy as np
from collections import Counter
from sklearn.mixture import GaussianMixture

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common.neighbors import smote_generate
from common import smoke as _smoke

MODEL_KEY = "scut"


class SCUT(Resampler):
    def __init__(
        self,
        k=5,
        max_clusters=9,
        covariance_types=("full", "tied", "diag", "spherical"),
        rf_n_estimators=30,
        n_jobs=-1,
        random_state=42,
        **kw
    ):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        self.k = int(k)
        self.max_clusters = None if max_clusters is None else int(max_clusters)
        self.covariance_types = tuple(covariance_types)

    def _max_components(self, n_samples):
        if self.max_clusters is None:
            return n_samples
        return min(self.max_clusters, n_samples)

    def _em_cluster(self, Xc):
        """Choose the best sklearn GaussianMixture via BIC."""
        Xc = np.asarray(Xc, dtype=float)
        n_samples = len(Xc)
        if n_samples <= 1:
            return np.zeros(n_samples, dtype=int)
        best_bic = None
        best_model = None
        for cov in self.covariance_types:
            for kk in range(1, self._max_components(n_samples) + 1):
                try:
                    gm = GaussianMixture(
                        n_components=kk,
                        covariance_type=cov,
                        reg_covar=1e-6,
                        n_init=3,
                        random_state=self.random_state,
                    ).fit(Xc)
                    bic = gm.bic(Xc)
                    if best_bic is None or bic < best_bic:
                        best_bic = bic
                        best_model = gm
                except Exception:
                    continue
        if best_model is None:
            return np.zeros(n_samples, dtype=int)
        try:
            return best_model.predict(Xc).astype(int)
        except Exception:
            return np.zeros(n_samples, dtype=int)

    def _sample_clusters(self, labels, target, rng):
        """Mirror scutr::sample_classes() cluster-stratified sampling."""
        labels = np.asarray(labels).astype(int)
        if target <= 0 or len(labels) == 0:
            return np.empty(0, dtype=int)
        clusters = np.unique(labels)
        per_cluster = int(np.ceil(float(target) / float(len(clusters))))
        sampled = []
        for cl in clusters:
            members = np.where(labels == cl)[0]
            if len(members) == 1:
                picks = np.repeat(members, per_cluster)
            else:
                picks = rng.choice(
                    members,
                    size=per_cluster,
                    replace=(per_cluster > len(members)),
                )
            sampled.append(np.asarray(picks, dtype=int))
        grid = np.vstack(sampled).T.reshape(-1)
        return grid[:target]

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        cnt = Counter(y.tolist())
        target = int(round(np.mean(list(cnt.values()))))
        Xl, yl = [], []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            Xc = X[idx]
            if len(idx) > target:                       # majority -> EM cluster + stratified RUS
                labels = self._em_cluster(Xc)
                local_keep = self._sample_clusters(labels, target, rng)
                keep = idx[local_keep]
                Xl.append(X[keep]); yl.append(y[keep])
            else:                                        # minority -> SMOTE
                Xl.append(Xc); yl.append(y[idx])
                n_gen = target - len(idx)
                if n_gen > 0:
                    gen = smote_generate(Xc, n_gen, self.k, rng)
                    Xl.append(gen); yl.append(np.full(n_gen, c, dtype=y.dtype))
        return np.vstack(Xl), np.concatenate(yl)


def build(
    random_state=42,
    smoke=False,
    k=5,
    max_clusters=9,
    covariance_types=("full", "tied", "diag", "spherical"),
    rf_n_estimators=30,
    n_jobs=-1,
    **kw
):
    return SCUT(
        k=k,
        max_clusters=max_clusters,
        covariance_types=covariance_types,
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
