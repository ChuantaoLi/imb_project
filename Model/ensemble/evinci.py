"""EVINCI — Evolutionary Inversion of class distribution (Fernandes & de
Carvalho, Information Sciences 2019).

"Evolutionary inversion of class distribution in overlapping areas for
multi-class imbalanced learning."

Mechanism: build a pool of candidate TRAINING SUBSETS with RANDOM-BALANCE
sampling and VARIABLE imbalance ratios, then use an EVOLUTIONARY algorithm
(NSGA-II) to evolve the retained samples inside each subset. The two objectives
are:
  * maximize validation balanced accuracy;
  * reduce the concentration of less-representative majority instances in
    overlapping areas.

Each evolved subset trains one CART base learner, and the final ensemble
aggregates the evolved learners. Each class never exceeds its original count
(no oversampling beyond the natural data).
"""

import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.neighbors import safe_level
from common import smoke as _smoke

MODEL_KEY = "evinci"


class EVINCI:
    def __init__(self, n_estimators=30, ga_pop=12, ga_gen=8, val_size=0.3, random_state=42, **kw):
        self.n_estimators = int(n_estimators)
        self.ga_pop = int(ga_pop)
        self.ga_gen = int(ga_gen)
        self.val_size = float(val_size)
        self.random_state = random_state

    def _random_subset(self, X, y, rng):
        """Random-balance subset, never exceeding a class's original count."""
        classes = np.unique(y)
        total = len(y)
        props = rng.dirichlet(np.ones(len(classes)))
        targets = np.minimum(np.maximum(1, np.round(props * total).astype(int)), [(y == c).sum() for c in classes])
        Xl, yl = [], []
        for c, t in zip(classes, targets):
            idx = np.where(y == c)[0]
            sel = rng.choice(idx, size=t, replace=False)
            Xl.append(X[sel])
            yl.append(y[sel])
        return np.vstack(Xl), np.concatenate(yl)

    def _subset_overlap_penalty(self, X, y):
        """Penalty for retaining less-representative majority instances in
        overlapping areas. Lower is better."""
        if len(X) <= 1:
            return 0.0
        sl = safe_level(X, y, k=5)
        counts = np.array([(y == c).sum() for c in np.unique(y)], dtype=float)
        maj_thresh = float(np.max(counts)) if len(counts) else 0.0
        penalty = 0.0
        kept = 0
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            is_majority = (y == c).sum() >= maj_thresh
            if not is_majority:
                continue
            overlap = 1.0 - sl[idx]
            penalty += float(overlap.sum())
            kept += len(idx)
        if kept == 0:
            return 0.0
        return penalty / kept

    def _fit_tree(self, X, y, seed):
        return DecisionTreeClassifier(random_state=seed).fit(X, y)

    def _align_proba(self, clf, X):
        raw = clf.predict_proba(X)
        p = np.zeros((len(X), self.n_classes_), dtype=float)
        for j, c in enumerate(clf.classes_):
            p[:, self.class_to_index_[int(c)]] = raw[:, j]
        return p

    def _evolve_subset(self, Xcand, ycand, Xva, yva, rng, seed):
        from deap import base, creator, tools, algorithms

        n = len(Xcand)
        if n <= max(8, len(np.unique(ycand)) * 2) or self.ga_gen <= 0:
            return Xcand, ycand
        sl = safe_level(Xcand, ycand, k=5)
        retain_prob = np.clip(0.25 + 0.75 * sl, 0.1, 0.95)

        def evaluate(ind):
            mask = np.array(ind, dtype=bool)
            if mask.sum() < len(self.classes_):
                return (0.0, -1.0)
            Xs = Xcand[mask]
            ys = ycand[mask]
            if len(np.unique(ys)) < len(self.classes_):
                return (0.0, -1.0)
            clf = self._fit_tree(Xs, ys, seed)
            pred = clf.predict(Xva)
            acc = balanced_accuracy_score(yva, pred)
            penalty = self._subset_overlap_penalty(Xs, ys)
            return (acc, -penalty)

        fit_name = f"FitMultiEV_{seed}"
        ind_name = f"IndEV_{seed}"
        creator.create(fit_name, base.Fitness, weights=(1.0, 1.0))
        creator.create(ind_name, list, fitness=getattr(creator, fit_name))
        tb = base.Toolbox()

        def init_ind():
            return getattr(creator, ind_name)([int(rng.rand() < retain_prob[j]) for j in range(n)])

        tb.register("ind", init_ind)
        tb.register("pop", tools.initRepeat, list, tb.ind)
        tb.register("mate", tools.cxTwoPoint)
        tb.register("mutate", tools.mutFlipBit, indpb=min(0.2, 5.0 / max(n, 1)))
        tb.register("select", tools.selNSGA2)
        tb.register("evaluate", evaluate)
        pop = tb.pop(n=self.ga_pop)
        algorithms.eaMuPlusLambda(
            pop,
            tb,
            mu=self.ga_pop,
            lambda_=self.ga_pop,
            cxpb=0.5,
            mutpb=0.2,
            ngen=self.ga_gen,
            verbose=False,
        )
        front = tools.sortLogNondominated(pop, len(pop))[0]
        best = max(front, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1], -sum(ind)))
        for name in (fit_name, ind_name):
            if name in creator.__dict__:
                del creator.__dict__[name]
        mask = np.array(best, dtype=bool)
        if mask.sum() < len(self.classes_) or len(np.unique(ycand[mask])) < len(self.classes_):
            return Xcand, ycand
        return Xcand[mask], ycand[mask]

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.class_to_index_ = {int(c): i for i, c in enumerate(self.classes_)}
        rng = np.random.RandomState(self.random_state)

        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=self.val_size, stratify=y, random_state=self.random_state)
        self.pool_ = []
        self.subsets_ = []
        for i in range(self.n_estimators):
            Xcand, ycand = self._random_subset(Xtr, ytr, rng)
            Xevo, yevo = self._evolve_subset(Xcand, ycand, Xva, yva, rng, self.random_state + i)
            self.subsets_.append((Xevo, yevo))
            self.pool_.append(self._fit_tree(Xevo, yevo, self.random_state + i))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        acc = np.zeros((len(X), self.n_classes_))
        for clf in self.pool_:
            acc += self._align_proba(clf, X)
        acc /= max(len(self.pool_), 1)
        return acc / acc.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


def build(random_state=42, smoke=False, n_estimators=30, ga_pop=12, ga_gen=8, **kw):
    if smoke:
        n_estimators, ga_pop, ga_gen = 12, 6, 3
    return EVINCI(
        n_estimators=n_estimators,
        ga_pop=ga_pop,
        ga_gen=ga_gen,
        random_state=random_state,
        **kw,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
