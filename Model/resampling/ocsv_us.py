"""OCSV-US — Undersampling with Support Vectors (Krawczyk, Bellinger, Corizzo,
Japkowicz, IJCNN 2021).

"Undersampling with support vectors for multi-class imbalanced data
classification."

Mechanism (undersampling only):
  1. Per class, fit a ONE-CLASS SVM and keep its SUPPORT VECTORS (the instances
     that define the class boundary -- the most informative, non-redundant
     representatives).
  2. GENETIC ALGORITHM selects the most significant SV subset (binary mask over
     SVs) maximising balanced accuracy on a held-out validation split; the rest
     are discarded. The result is a much smaller, low-overlap dataset.

DEAP realises the GA (mechanism preserved; budget reduced for single-threaded
compute -- see code review for paper vs used budget). Downstream = RF(30).
"""

import os
import sys
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.resampler import Resampler
from common import smoke as _smoke

MODEL_KEY = "ocsv_us"


class OCSVUS(Resampler):
    def __init__(self, nu=0.1, gamma="scale", ga_pop=12, ga_gen=8, val_size=0.25, target_strategy="min", rf_n_estimators=30, n_jobs=-1, random_state=42, **kw):
        super().__init__(rf_n_estimators, n_jobs, random_state)
        if target_strategy not in {"min", "mean"}:
            raise ValueError("target_strategy must be one of min/mean")
        self.nu = float(nu)
        self.gamma = gamma
        self.ga_pop = int(ga_pop)
        self.ga_gen = int(ga_gen)
        self.val_size = float(val_size)
        self.target_strategy = target_strategy

    def _support_vectors(self, X, y):
        sv_X, sv_y, sv_sig = [], [], []
        for c in np.unique(y):
            Xc = X[y == c]
            if len(Xc) < 2:
                sv_X.append(Xc)
                sv_y.append(np.full(len(Xc), c, dtype=y.dtype))
                sv_sig.append(np.ones(len(Xc), dtype=float))
                continue
            try:
                oc = OneClassSVM(nu=self.nu, gamma=self.gamma).fit(Xc)
                sv_idx = oc.support_
                strength = np.abs(oc.dual_coef_).ravel()
                if len(strength) != len(sv_idx):
                    strength = np.ones(len(sv_idx), dtype=float)
                sv_X.append(Xc[sv_idx])
                sv_y.append(np.full(len(sv_idx), c, dtype=y.dtype))
                sv_sig.append(np.maximum(strength, 1e-12))
            except Exception:
                sv_X.append(Xc)
                sv_y.append(np.full(len(Xc), c, dtype=y.dtype))
                sv_sig.append(np.ones(len(Xc), dtype=float))
        return np.vstack(sv_X), np.concatenate(sv_y), np.concatenate(sv_sig)

    def _target_count(self, ysv):
        _, counts = np.unique(ysv, return_counts=True)
        if len(counts) == 0:
            return 0
        if self.target_strategy == "mean":
            target = int(round(np.mean(counts)))
        else:
            target = int(np.min(counts))
        return max(1, target)

    def _split_pool_val(self, Xsv, ysv, sig):
        classes, counts = np.unique(ysv, return_counts=True)
        if len(classes) < 2 or np.min(counts) < 2:
            return Xsv, np.empty((0, Xsv.shape[1])), ysv, np.empty(0, dtype=ysv.dtype), sig
        Xpool, Xval, ypool, yval, spool, _ = train_test_split(
            Xsv,
            ysv,
            sig,
            test_size=self.val_size,
            stratify=ysv,
            random_state=self.random_state,
        )
        if len(np.unique(ypool)) < 2 or len(np.unique(yval)) < 2:
            return Xsv, np.empty((0, Xsv.shape[1])), ysv, np.empty(0, dtype=ysv.dtype), sig
        return Xpool, Xval, ypool, yval, spool

    def _build_fixed_and_variable(self, ypool):
        target = self._target_count(ypool)
        classes, counts = np.unique(ypool, return_counts=True)
        fixed, variable, class_target = [], [], {}
        for cls, cnt in zip(classes, counts):
            idx = np.where(ypool == cls)[0]
            cnt = int(cnt)
            want = min(cnt, target)
            class_target[int(cls)] = want
            if cnt <= want:
                fixed.extend(idx.tolist())
            else:
                variable.extend(idx.tolist())
        return np.asarray(fixed, dtype=int), np.asarray(variable, dtype=int), class_target

    def _decode_mask(self, ind, ypool, sigpool, fixed_idx, var_idx, class_target):
        selected = set(fixed_idx.tolist())
        if len(var_idx) == 0:
            return np.asarray(sorted(selected), dtype=int)
        chosen_global = var_idx[np.asarray(ind, dtype=bool)]
        for cls, target in class_target.items():
            idx_cls = np.where(ypool == cls)[0]
            if len(idx_cls) <= target:
                selected.update(idx_cls.tolist())
                continue
            chosen_cls = chosen_global[ypool[chosen_global] == cls]
            remain_cls = idx_cls[~np.isin(idx_cls, chosen_cls)]
            if len(chosen_cls) > target:
                order = np.argsort(sigpool[chosen_cls])[::-1][:target]
                chosen_cls = chosen_cls[order]
            elif len(chosen_cls) < target:
                need = target - len(chosen_cls)
                if need > 0:
                    order = np.argsort(sigpool[remain_cls])[::-1][:need]
                    chosen_cls = np.concatenate([chosen_cls, remain_cls[order]])
            selected.update(np.asarray(chosen_cls, dtype=int).tolist())
        return np.asarray(sorted(selected), dtype=int)

    def _ga_select(self, Xsv, ysv, sig, rng):
        """GA over removable support vectors while preserving class-balance targets."""
        from deap import base, creator, tools, algorithms

        if len(Xsv) <= 8 or self.ga_gen <= 0:
            return Xsv, ysv
        Xpool, Xval, ypool, yval, sigpool = self._split_pool_val(Xsv, ysv, sig)
        if len(Xval) == 0:
            return Xsv, ysv
        fixed_idx, var_idx, class_target = self._build_fixed_and_variable(ypool)
        if len(var_idx) == 0:
            decoded = self._decode_mask([], ypool, sigpool, fixed_idx, var_idx, class_target)
            return Xpool[decoded], ypool[decoded]
        n = len(var_idx)

        def fit(ind):
            picked = self._decode_mask(ind, ypool, sigpool, fixed_idx, var_idx, class_target)
            if len(np.unique(ypool[picked])) < 2:
                return (0.0,)
            try:
                clf = RandomForestClassifier(
                    n_estimators=min(self.rf_n_estimators, 15),
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
                clf.fit(Xpool[picked], ypool[picked])
                return (balanced_accuracy_score(yval, clf.predict(Xval)),)
            except Exception:
                return (0.0,)

        if not hasattr(creator, "FitMaxOCSV"):
            creator.create("FitMaxOCSV", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndOCSV"):
            creator.create("IndOCSV", list, fitness=creator.FitMaxOCSV)
        tb = base.Toolbox()
        class_weights = sigpool[var_idx]
        class_weights = class_weights / np.clip(class_weights.max(), 1e-12, None)
        tb.register("attr", lambda p=None: int(rng.rand() < p), 0.5)

        def init_individual():
            genes = [int(rng.rand() < (0.25 + 0.5 * w)) for w in class_weights]
            return creator.IndOCSV(genes)

        tb.register("ind", tools.initRepeat, creator.IndOCSV, tb.attr, n=n)
        tb.unregister("ind")
        tb.register("ind", init_individual)
        tb.register("pop", tools.initRepeat, list, tb.ind)
        tb.register("mate", tools.cxTwoPoint)
        tb.register("mutate", tools.mutFlipBit, indpb=0.1)
        tb.register("select", tools.selTournament, tournsize=3)
        tb.register("evaluate", fit)
        pop = tb.pop(n=self.ga_pop)
        algorithms.eaSimple(pop, tb, cxpb=0.5, mutpb=0.2, ngen=self.ga_gen, verbose=False)
        best = tools.selBest(pop, k=1)[0]
        picked = self._decode_mask(best, ypool, sigpool, fixed_idx, var_idx, class_target)
        if len(picked) < len(np.unique(ypool)) * 2:
            return Xsv, ysv
        return Xpool[picked], ypool[picked]

    def _resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        Xsv, ysv, sig = self._support_vectors(X, y)
        Xs, ys = self._ga_select(Xsv, ysv, sig, rng)
        if len(Xs) < len(np.unique(y)) * 2:  # safety: keep all SVs
            Xs, ys = Xsv, ysv
        return Xs, ys


def build(random_state=42, smoke=False, nu=0.1, ga_pop=12, ga_gen=8, rf_n_estimators=30, n_jobs=-1, **kw):
    if smoke:
        ga_pop, ga_gen = 6, 3
    return OCSVUS(nu=nu, ga_pop=ga_pop, ga_gen=ga_gen, rf_n_estimators=rf_n_estimators, n_jobs=n_jobs, random_state=random_state, **kw)


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
