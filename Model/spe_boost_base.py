"""Research core for the SPE/boosting iteration chain.

The class keeps SPE's self-paced majority selection, then adds a single
stage-wise boosting state. Version modules select one documented strategy.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors


class SPEBoostEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=42, n_estimators=18, n_bins=12,
                 base_depth=2, rf_trees=12, learning_rate=0.8,
                 strategy="boost", preencoded=True):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_bins = n_bins
        self.base_depth = base_depth
        self.rf_trees = rf_trees
        self.learning_rate = learning_rate
        self.strategy = strategy
        self.preencoded = preencoded

    def _margin_hardness(self, proba, y):
        true = proba[np.arange(len(y)), y]
        other = proba.copy()
        other[np.arange(len(y)), y] = -np.inf
        return np.clip((1.0 - true + np.max(other, axis=1)) / 2.0, 0.0, 1.0)

    def _density(self, X):
        if len(X) < 3:
            return np.ones(len(X))
        k = min(7, len(X) - 1)
        d = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
            X, return_distance=True)[0][:, 1:]
        rho = 1.0 / (np.mean(d, axis=1) + 1e-9)
        lo, hi = np.percentile(rho, 5), np.percentile(rho, 95)
        return np.clip((rho - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    def _hardness(self, X, y, cumulative):
        h = self._margin_hardness(cumulative, y)
        if self.strategy in {"density", "boundary", "density_cost"}:
            h = np.clip(h * (0.55 + 0.45 * self._density(X)), 0.0, 1.0)
        if self.strategy in {"margin", "boundary", "entropy"}:
            if self.strategy == "entropy":
                ent = -(cumulative * np.log(cumulative + 1e-12)).sum(axis=1)
                h = 0.5 * h + 0.5 * ent / np.log(cumulative.shape[1])
            else:
                h = np.clip(h ** 0.75, 0.0, 1.0)
        return h

    def _self_paced_indices(self, y, hardness, weights, rng, round_id):
        classes, counts = np.unique(y, return_counts=True)
        if self.strategy == "brf_full":
            # The SPE curriculum is used for the first half of rounds; after
            # that, Balanced RF sees the full stream and supplies its own
            # per-tree class-balanced bootstrap.
            if round_id >= max(1, self.n_estimators // 2):
                idx = np.arange(len(y), dtype=int)
                rng.shuffle(idx)
                return idx
        target = int(max(2, counts.min()))
        selected = []
        for c, count in zip(classes, counts):
            idx = np.flatnonzero(y == c)
            if count <= target:
                selected.extend(idx.tolist())
                continue
            hc = hardness[idx]
            bins = np.minimum(self.n_bins - 1, (hc * self.n_bins).astype(int))
            avail = np.array([(bins == b).sum() for b in range(self.n_bins)])
            mean_h = np.array([hc[bins == b].mean() if avail[b] else 0.0
                               for b in range(self.n_bins)])
            pace = np.tan(np.pi * min(round_id, self.n_estimators - 1) /
                          (2.0 * max(self.n_estimators - 1, 1)))
            score = 1.0 / (mean_h + 0.15 + pace)
            if self.strategy in {"boost", "cost", "density_cost", "margin", "entropy", "boundary"}:
                score *= 0.65 + 0.35 * np.arange(self.n_bins) / max(self.n_bins - 1, 1)
            score[avail == 0] = 0.0
            score /= score.sum() + 1e-12
            raw = score * target
            take = np.minimum(np.floor(raw).astype(int), avail)
            rem = target - int(take.sum())
            order = np.argsort(-(raw - take) - (mean_h * 1e-3))
            for b in order:
                if rem <= 0:
                    break
                if take[b] < avail[b]:
                    take[b] += 1
                    rem -= 1
            for b, n_take in enumerate(take):
                if n_take:
                    members = idx[bins == b]
                    selected.extend(rng.choice(members, size=int(n_take), replace=False).tolist())
        selected = np.asarray(selected, dtype=int)
        rng.shuffle(selected)
        return selected

    def _make_estimator(self, seed):
        if self.strategy == "brf_full":
            return BalancedRandomForestClassifier(
                n_estimators=self.rf_trees, max_depth=self.base_depth,
                random_state=seed, n_jobs=1, replacement=True,
                sampling_strategy="all")
        if self.strategy == "extra":
            return ExtraTreesClassifier(n_estimators=self.rf_trees, max_depth=self.base_depth,
                                        random_state=seed, n_jobs=1, class_weight=None)
        if self.strategy == "tree":
            return DecisionTreeClassifier(max_depth=self.base_depth, random_state=seed,
                                          class_weight=None)
        return RandomForestClassifier(n_estimators=self.rf_trees, max_depth=self.base_depth,
                                      max_features="sqrt", random_state=seed,
                                      n_jobs=1, class_weight=None)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        self.classes_ = np.unique(y)
        n, c = len(y), len(self.classes_)
        counts = np.bincount(y, minlength=c).astype(float)
        # SPE already supplies class balance through its per-round subset.
        # Starting with uniform mass avoids double-counting the minority when
        # sample_weight is applied inside an already balanced subset.
        w = np.ones(n, dtype=float)
        w /= w.sum()
        self.estimators_, self.alphas_, self.round_sizes_ = [], [], []
        cumulative = np.full((n, c), 1.0 / c)
        rng = np.random.RandomState(self.random_state)
        for t in range(self.n_estimators):
            hardness = np.full(n, 0.5) if t == 0 else self._hardness(X, y, cumulative)
            idx = self._self_paced_indices(y, hardness, w, rng, t)
            fit_w = w[idx].copy()
            if self.strategy in {"cost", "density_cost"}:
                local = np.bincount(y[idx], minlength=c).astype(float)
                fit_w *= 1.0 / np.maximum(local[y[idx]], 1.0)
            fit_w /= fit_w.sum() + 1e-12
            clf = self._make_estimator(int(rng.randint(0, 2**31 - 1)))
            clf.fit(X[idx], y[idx], sample_weight=fit_w)
            p = np.asarray(clf.predict_proba(X), dtype=float)
            aligned = np.zeros((n, c), dtype=float)
            for j, cls in enumerate(getattr(clf, "classes_", [])):
                aligned[:, int(cls)] = p[:, j]
            pred = np.argmax(aligned, axis=1)
            err = float(np.clip(np.sum(w * (pred != y)), 1e-4, 1.0 - 1e-4))
            alpha = float(np.clip(self.learning_rate * 0.5 * np.log((1.0 - err) / err),
                                  0.02, 2.5))
            if self.strategy == "brf_full":
                alpha = 1.0
            update = np.where(pred == y, np.exp(-alpha), np.exp(alpha))
            if self.strategy in {"density", "boundary", "density_cost"}:
                rho = self._density(X)
                update = np.where(pred != y, np.exp(alpha * (0.65 + 0.35 * rho)), update)
            w *= update
            w += 1e-12
            w /= w.sum()
            self.estimators_.append(clf)
            self.alphas_.append(alpha)
            self.round_sizes_.append(int(len(idx)))
            cumulative += alpha * aligned
            cumulative /= cumulative.sum(axis=1, keepdims=True)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        c = len(self.classes_)
        scores = np.zeros((len(X), c), dtype=float)
        total = 0.0
        for clf, alpha in zip(self.estimators_, self.alphas_):
            p = clf.predict_proba(X)
            aligned = np.zeros((len(X), c), dtype=float)
            for j, cls in enumerate(getattr(clf, "classes_", [])):
                aligned[:, int(cls)] = p[:, j]
            scores += alpha * aligned
            total += alpha
        if total <= 0:
            return np.full((len(X), c), 1.0 / max(c, 1))
        scores = np.clip(scores / total, 1e-12, None)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
