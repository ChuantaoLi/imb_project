"""SOUP — Similarity Oversampling and Undersampling Preprocessing
(Janicka, Lango, Stefanowski, AMCS 2019).

"Using information on class interrelations to improve classification of
multiclass imbalanced data: a new resampling algorithm."

Mechanism:
  * evaluate examples' DIFFICULTY from the class composition of their whole-set
    neighbourhoods and from the SIMILARITY of neighbouring classes;
  * Target class size = arithmetic MEAN of all class sizes;
  * Majority classes (> target): PRIORITY undersampling by the lowest
    similarity-aware safety scores;
  * Minority classes (< target): REPLICATION oversampling by the highest
    similarity-aware safety scores.

Implementation note:
  * the paper / author page expose SOUP SIM1..SIM6 and SOUP Heur variants, but
    the exact formulas are not directly readable here. We therefore instantiate
    the published "Heur" path with a heuristic class-similarity matrix derived
    from mutual neighbourhood mixing. This restores the paper's missing second
    signal -- neighbouring-class similarity -- instead of collapsing SOUP to a
    plain safe-level sorter.
"""
import os
import sys
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common import smoke as _smoke

MODEL_KEY = "soup"


class SOUP(Resampler):
    def __init__(
        self,
        k=5,
        similarity_mode="heur",
        update_difficulty=True,
        rf_n_estimators=30,
        n_jobs=-1,
        random_state=42,
        **kw
    ):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        self.k = int(k)
        self.similarity_mode = str(similarity_mode)
        self.update_difficulty = bool(update_difficulty)

    def _target_size(self, y):
        cnt = Counter(np.asarray(y).tolist())
        return int(round(np.mean(list(cnt.values())))) if cnt else 0

    def _class_maps(self, y):
        classes = np.unique(y)
        return classes, {int(c): i for i, c in enumerate(classes)}

    def _neighbor_labels(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = len(X)
        kk = min(self.k, n - 1)
        if kk <= 0:
            return np.empty((n, 0), dtype=int), np.empty((n, 0), dtype=y.dtype), 0, None
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
        _, idx = nn.kneighbors(X)
        idx = idx[:, 1:]
        return idx, y[idx], kk, nn

    def _class_similarity_matrix(self, X, y, nb_labels=None, kk=None):
        classes, class_to_index = self._class_maps(y)
        n_classes = len(classes)
        sim = np.eye(n_classes, dtype=float)
        if nb_labels is None:
            _, nb_labels, kk, _ = self._neighbor_labels(X, y)
        if kk is None:
            kk = nb_labels.shape[1]
        if kk <= 0 or n_classes <= 1:
            return classes, class_to_index, sim

        raw = np.zeros((n_classes, n_classes), dtype=float)
        for cls in classes:
            row = class_to_index[int(cls)]
            mask = (y == cls)
            if not np.any(mask):
                continue
            neigh = nb_labels[mask]
            for other in classes:
                col = class_to_index[int(other)]
                raw[row, col] = float(np.mean(neigh == other))

        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                if self.similarity_mode == "identity":
                    val = 0.0
                else:
                    val = 0.5 * (raw[i, j] + raw[j, i])
                sim[i, j] = sim[j, i] = float(np.clip(val, 0.0, 1.0))
        return classes, class_to_index, sim

    def _similarity_safe_level(self, X, y, cached_nn=None):
        """Compute safe-level scores. Pass (idx, nb_labels, kk) to reuse a single NN call."""
        if cached_nn is not None:
            _, nb_labels, kk = cached_nn
        else:
            _, nb_labels, kk, _ = self._neighbor_labels(X, y)
        classes, class_to_index, sim = self._class_similarity_matrix(X, y, nb_labels=nb_labels, kk=kk)
        if kk <= 0:
            return np.ones(len(y), dtype=float)
        scores = np.zeros(len(y), dtype=float)
        for i, cls in enumerate(y):
            row = class_to_index[int(cls)]
            cols = np.asarray([class_to_index[int(lb)] for lb in nb_labels[i]], dtype=int)
            scores[i] = float(np.mean(sim[row, cols])) if len(cols) else 1.0
        return scores

    def _priority_order(self, values, descending, rng):
        values = np.asarray(values, dtype=float)
        noise = rng.uniform(0.0, 1e-9, size=len(values))
        base = -values if descending else values
        return np.argsort(base + noise)

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        Xw = np.asarray(X, dtype=float)
        yw = np.asarray(y).copy()
        target = self._target_size(yw)

        def _compute_nn_cache():
            """Compute global NN once; reuse until dataset changes."""
            idx, nb, kk, _ = self._neighbor_labels(Xw, yw)
            return idx, nb, kk

        # Priority undersampling: repeatedly remove the hardest majority samples.
        while True:
            cnt = Counter(yw.tolist())
            oversized = [c for c, n in cnt.items() if n > target]
            if not oversized:
                break
            nn_cache = _compute_nn_cache()
            scores = self._similarity_safe_level(Xw, yw, cached_nn=nn_cache)
            for c in sorted(oversized):
                idx = np.where(yw == c)[0]
                n_drop = cnt[c] - target
                if n_drop <= 0:
                    continue
                local = self._priority_order(scores[idx], descending=False, rng=rng)
                drop_idx = idx[local[:n_drop]]
                keep_mask = np.ones(len(yw), dtype=bool)
                keep_mask[drop_idx] = False
                Xw = Xw[keep_mask]
                yw = yw[keep_mask]
                if self.update_difficulty:
                    break
            if not self.update_difficulty:
                break

        # Priority replication: repeatedly duplicate the safest minority samples.
        while True:
            cnt = Counter(yw.tolist())
            undersized = [c for c, n in cnt.items() if n < target]
            if not undersized:
                break
            nn_cache = _compute_nn_cache()
            scores = self._similarity_safe_level(Xw, yw, cached_nn=nn_cache)
            changed = False
            for c in sorted(undersized):
                idx = np.where(yw == c)[0]
                n_add = target - len(idx)
                if n_add <= 0:
                    continue
                local = self._priority_order(scores[idx], descending=True, rng=rng)
                rep_idx = idx[local[:1 if self.update_difficulty else n_add]]
                Xw = np.vstack([Xw, Xw[rep_idx]])
                yw = np.concatenate([yw, yw[rep_idx]])
                changed = True
                if self.update_difficulty:
                    break
            if not changed or not self.update_difficulty:
                break
        return Xw, yw


def build(
    random_state=42,
    smoke=False,
    k=5,
    similarity_mode="heur",
    update_difficulty=True,
    rf_n_estimators=30,
    n_jobs=-1,
    **kw
):
    return SOUP(
        k=k,
        similarity_mode=similarity_mode,
        update_difficulty=update_difficulty,
        rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **kw
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
