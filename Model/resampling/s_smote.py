"""S-SMOTE — dynamic over-sampling based on sensitivity + memetic RBFNN
(Fernández-Navarro, Hervás-Martínez, Gutiérrez, Pattern Recognition 2011).

"A dynamic over-sampling procedure based on sensitivity for multi-class problems."

Paper-faithful main flow:
  1. STAGE-1 OVER-SAMPLING: partially rebalance only the global minority class
     by duplicating its size with SMOTE.
  2. DYNAMIC OVER-SAMPLING DURING EVOLUTION: after evaluating the current
     population, identify the minimum-sensitivity class of the best individual
     and duplicate that class with SMOTE for the next generations.
  3. MEMETIC RBFNN: GA searches the RBFNN structure / width / ridge, while the
     output weights are solved by least squares as the local refinement step.

This is an embedded-classifier method: the final head is the GA-optimised RBFNN,
not the project-wide RF(30).
"""
import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common.neighbors import smote_generate
from common import smoke as _smoke

MODEL_KEY = "s_smote"


class RBFNN:
    """Gaussian RBF network: H = exp(-gamma ||x - C||^2); logits = H @ W (ridge LS)."""
    def __init__(self, n_centers, gamma, ridge):
        self.n_centers = int(n_centers)
        self.gamma = float(gamma)
        self.ridge = float(ridge)

    def fit(self, X, y, n_classes):
        self.n_classes = int(n_classes)
        k = min(self.n_centers, len(X))
        self.centers = KMeans(n_clusters=k, n_init=3,
                              random_state=0).fit(X).cluster_centers_
        H = np.exp(-self.gamma * self._sqdist(X, self.centers))
        Y = np.eye(n_classes)[y]
        self.W = np.linalg.solve(H.T @ H + self.ridge * np.eye(k), H.T @ Y)
        return self

    def _sqdist(self, A, B):
        return (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2 * A @ B.T)

    def predict_proba(self, X):
        H = np.exp(-self.gamma * self._sqdist(X, self.centers))
        logits = H @ self.W
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        return e / np.clip(e.sum(1, keepdims=True), 1e-12, None)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), 1)


class SSMOTE:
    def __init__(self, k=5, ga_pop=12, ga_gen=8, max_centers=40,
                 ridge=1e-2, random_state=42, smoke=False):
        self.k = int(k)
        self.ga_pop = int(ga_pop)
        self.ga_gen = int(ga_gen)
        self.max_centers = int(max_centers)
        self.ridge = float(ridge)
        self.random_state = int(random_state)
        self.smoke = bool(smoke)

    def _encode_y(self, y):
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.class_to_index_ = {int(c): i for i, c in enumerate(self.classes_)}
        self.index_to_class_ = {i: int(c) for i, c in enumerate(self.classes_)}
        self.n_classes_ = len(self.classes_)
        return y_enc.astype(int)

    def _class_counts(self, y):
        if len(y) == 0:
            return np.zeros(self.n_classes_, dtype=int)
        return np.bincount(y, minlength=self.n_classes_).astype(int)

    def _duplicate_class_with_smote(self, X, y, cls, rng):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            return X, y
        gen = smote_generate(X[idx], len(idx), self.k, rng)
        X_new = np.vstack([X, gen])
        y_new = np.concatenate([y, np.full(len(gen), cls, dtype=int)])
        return X_new, y_new

    def _initial_stage_resample(self, X, y, rng):
        counts = self._class_counts(y)
        positive = counts[counts > 0]
        if len(positive) == 0:
            return X, y
        min_count = int(positive.min())
        minority_classes = np.where(counts == min_count)[0]
        Xw, yw = np.asarray(X, float), np.asarray(y).astype(int)
        for cls in minority_classes:
            Xw, yw = self._duplicate_class_with_smote(Xw, yw, int(cls), rng)
        return Xw, yw

    def _per_class_sensitivity(self, y_true, y_pred):
        sens = np.zeros(self.n_classes_, dtype=float)
        for cls in range(self.n_classes_):
            mask = (y_true == cls)
            if np.any(mask):
                sens[cls] = float(np.mean(y_pred[mask] == cls))
        return sens

    def _minimum_sensitivity_class(self, model, Xva, yva, current_y):
        pred = model.predict(np.asarray(Xva, float))
        sens = self._per_class_sensitivity(yva, pred)
        counts = self._class_counts(current_y)
        present = np.where(counts > 0)[0]
        if len(present) == 0:
            return None, sens
        worst_score = np.min(sens[present])
        tied = present[np.isclose(sens[present], worst_score)]
        if len(tied) > 1:
            tied_counts = counts[tied]
            tied = tied[np.argmin(tied_counts)]
            return int(tied), sens
        return int(tied[0]), sens

    def _build_rbf(self, params, Xtr, ytr):
        n_c = max(2, int(round(params[0])))
        gamma = max(self.base_gamma_ * float(params[1]), 1e-8)
        ridge = max(self.ridge * float(params[2]), 1e-8)
        try:
            return RBFNN(n_c, gamma, ridge).fit(Xtr, ytr, self.n_classes_)
        except Exception:
            return None

    def _evaluate_individual(self, ind, Xtr, ytr, Xva, yva):
        model = self._build_rbf(ind, Xtr, ytr)
        if model is None:
            return 0.0, None, None, None
        pred = model.predict(Xva)
        score = float(balanced_accuracy_score(yva, pred))
        worst_cls, sens = self._minimum_sensitivity_class(model, Xva, yva, ytr)
        return score, model, worst_cls, sens

    def _init_population(self, rng, low_nc, high_nc):
        pop = []
        for _ in range(self.ga_pop):
            pop.append([
                float(rng.uniform(low_nc, high_nc)),
                float(rng.uniform(0.2, 5.0)),
                float(rng.uniform(0.1, 10.0)),
            ])
        return pop

    def _mutate(self, ind, rng, low_nc, high_nc):
        out = list(ind)
        for i, sigma in enumerate((2.0, 0.5, 1.0)):
            out[i] += float(rng.normal(0.0, sigma))
        out[0] = float(np.clip(out[0], low_nc, high_nc))
        out[1] = float(np.clip(out[1], 0.05, 10.0))
        out[2] = float(np.clip(out[2], 0.01, 20.0))
        return out

    def _crossover(self, a, b, rng, low_nc, high_nc):
        alpha = float(rng.uniform(-0.5, 1.5))
        c1 = [alpha * x + (1.0 - alpha) * y for x, y in zip(a, b)]
        c2 = [alpha * y + (1.0 - alpha) * x for x, y in zip(a, b)]
        for child in (c1, c2):
            child[0] = float(np.clip(child[0], low_nc, high_nc))
            child[1] = float(np.clip(child[1], 0.05, 10.0))
            child[2] = float(np.clip(child[2], 0.01, 20.0))
        return c1, c2

    def _tournament_select(self, pop, fitness, rng, tournsize=3):
        idx = rng.choice(len(pop), size=min(tournsize, len(pop)), replace=False)
        best = idx[np.argmax([fitness[i] for i in idx])]
        return list(pop[int(best)])

    def fit(self, X, y):
        X = np.asarray(X, float)
        y_raw = np.asarray(y).astype(int).ravel()
        y = self._encode_y(y_raw)
        rng = np.random.RandomState(self.random_state)
        X_stage1, y_stage1 = self._initial_stage_resample(X, y, rng)
        Xtr, Xva, ytr, yva = train_test_split(
            X_stage1,
            y_stage1,
            test_size=0.25,
            stratify=y_stage1,
            random_state=self.random_state,
        )
        Xtr_dyn, ytr_dyn = np.asarray(Xtr, float), np.asarray(ytr).astype(int)
        Xfit_dyn, yfit_dyn = np.asarray(X_stage1, float), np.asarray(y_stage1).astype(int)
        self.base_gamma_ = 1.0 / max(X.shape[1], 1)
        low_nc = 2.0
        high_nc = float(max(low_nc + 1.0, min(self.max_centers, len(Xtr_dyn))))
        pop = self._init_population(rng, low_nc, high_nc)
        self.dynamic_history_ = []
        best_score = -np.inf
        best_ind = [min(10.0, high_nc), 1.0, 1.0]

        for gen in range(self.ga_gen):
            fitness = []
            gen_models = []
            gen_worst = []
            for ind in pop:
                score, model, worst_cls, _ = self._evaluate_individual(ind, Xtr_dyn, ytr_dyn, Xva, yva)
                fitness.append(score)
                gen_models.append(model)
                gen_worst.append(worst_cls)
                if score > best_score:
                    best_score = score
                    best_ind = list(ind)

            best_idx = int(np.argmax(fitness))
            focus_cls = gen_worst[best_idx]
            self.dynamic_history_.append(
                {
                    "generation": gen,
                    "best_score": float(fitness[best_idx]),
                    "focus_class": None if focus_cls is None else int(self.classes_[focus_cls]),
                }
            )
            if focus_cls is not None:
                Xtr_dyn, ytr_dyn = self._duplicate_class_with_smote(Xtr_dyn, ytr_dyn, focus_cls, rng)
                Xfit_dyn, yfit_dyn = self._duplicate_class_with_smote(Xfit_dyn, yfit_dyn, focus_cls, rng)

            if gen == self.ga_gen - 1:
                break

            offspring = []
            while len(offspring) < self.ga_pop:
                p1 = self._tournament_select(pop, fitness, rng)
                p2 = self._tournament_select(pop, fitness, rng)
                if rng.rand() < 0.5:
                    c1, c2 = self._crossover(p1, p2, rng, low_nc, high_nc)
                else:
                    c1, c2 = list(p1), list(p2)
                if rng.rand() < 0.3:
                    c1 = self._mutate(c1, rng, low_nc, high_nc)
                if rng.rand() < 0.3:
                    c2 = self._mutate(c2, rng, low_nc, high_nc)
                offspring.extend([c1, c2])
            pop = offspring[:self.ga_pop]

        self.best_params_ = {
            "n_centers": max(2, int(round(best_ind[0]))),
            "gamma": max(self.base_gamma_ * float(best_ind[1]), 1e-8),
            "ridge": max(self.ridge * float(best_ind[2]), 1e-8),
        }
        self.rbf_ = RBFNN(
            self.best_params_["n_centers"],
            self.best_params_["gamma"],
            self.best_params_["ridge"],
        ).fit(Xfit_dyn, yfit_dyn, self.n_classes_)
        return self

    def predict_proba(self, X):
        return self.rbf_.predict_proba(np.asarray(X, float))

    def predict(self, X):
        pred = self.rbf_.predict(np.asarray(X, float))
        return np.asarray([self.index_to_class_[int(i)] for i in pred], dtype=int)


def build(random_state=42, smoke=False, k=5, ga_pop=12, ga_gen=8,
          max_centers=40, ridge=1e-2, **kw):
    if smoke:
        ga_pop, ga_gen = 6, 3
    return SSMOTE(
        k=k,
        ga_pop=ga_pop,
        ga_gen=ga_gen,
        max_centers=max_centers,
        ridge=ridge,
        random_state=random_state,
        smoke=smoke,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
