"""Easy-BPNN — one-versus-one EasyEnsemble with a BPNN base (Zhang et al., KBS 2016).

"Empowering one-vs-one decomposition with ensemble learning for multi-class
imbalanced data."

Mechanism:
  * decompose the multiclass task into one-versus-one binary sub-problems;
  * for each class pair, build an EasyEnsemble by repeatedly random
    under-sampling the local majority class down to the minority size;
  * train one BPNN on each balanced pairwise subset;
  * fuse the pairwise ensemble posteriors back into a multiclass score.

Base BPNN = single hidden layer (50 units), sigmoid activations, softmax
output, cross-entropy backprop (numpy, no DL framework for a 50-unit model).
"""
import os
import sys
from itertools import combinations
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from common import smoke as _smoke

MODEL_KEY = "easy_bpnn"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z); return e / e.sum(1, keepdims=True)


class BPNN:
    """Single-hidden-layer BP classifier (numpy)."""
    def __init__(self, n_hidden=50, lr=0.1, epochs=120, l2=1e-4, random_state=0):
        self.n_hidden = n_hidden; self.lr = lr; self.epochs = epochs
        self.l2 = l2; self.random_state = random_state

    def fit(self, X, y, n_classes):
        self.n_classes = n_classes
        rng = np.random.RandomState(self.random_state)
        d = X.shape[1]
        self.W1 = rng.randn(d, self.n_hidden) * np.sqrt(1.0 / d)
        self.b1 = np.zeros(self.n_hidden)
        self.W2 = rng.randn(self.n_hidden, n_classes) * np.sqrt(1.0 / self.n_hidden)
        self.b2 = np.zeros(n_classes)
        Y = np.eye(n_classes)[y]
        n = len(X)
        for _ in range(self.epochs):
            p = rng.permutation(n)
            for s in range(0, n, 64):
                idx = p[s:s + 64]
                Xb, Yb = X[idx], Y[idx]
                h = _sigmoid(Xb @ self.W1 + self.b1)
                o = _softmax(h @ self.W2 + self.b2)
                do = (o - Yb) / len(idx)
                gW2 = h.T @ do + self.l2 * self.W2
                gb2 = do.sum(0)
                dh = do @ self.W2.T * h * (1 - h)
                gW1 = Xb.T @ dh + self.l2 * self.W1
                gb1 = dh.sum(0)
                self.W2 -= self.lr * gW2; self.b2 -= self.lr * gb2
                self.W1 -= self.lr * gW1; self.b1 -= self.lr * gb1
        return self

    def predict_proba(self, X):
        h = _sigmoid(X @ self.W1 + self.b1)
        return _softmax(h @ self.W2 + self.b2)


class EasyBPNN:
    def __init__(self, n_estimators=30, n_hidden=50, lr=0.1, epochs=120,
                 random_state=42, **kw):
        self.n_estimators = int(n_estimators)
        self.n_hidden = int(n_hidden)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.random_state = random_state

    def _pair_balanced_subset(self, X_pair, y_pair, rng):
        idx0 = np.where(y_pair == 0)[0]
        idx1 = np.where(y_pair == 1)[0]
        target = min(len(idx0), len(idx1))
        sel0 = rng.choice(idx0, size=target, replace=False)
        sel1 = rng.choice(idx1, size=target, replace=False)
        idx = np.concatenate([sel0, sel1])
        rng.shuffle(idx)
        return X_pair[idx], y_pair[idx]

    @staticmethod
    def _align_binary_proba(net, X):
        raw = net.predict_proba(X)
        out = np.zeros((len(X), 2), dtype=float)
        out[:, :raw.shape[1]] = raw
        return out

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.class_to_index_ = {int(c): i for i, c in enumerate(self.classes_)}
        rng = np.random.RandomState(self.random_state)
        self.pairs_ = []
        self.models_ = {}
        for pair_id, (ca, cb) in enumerate(combinations(self.classes_, 2)):
            mask = np.isin(y, [ca, cb])
            X_pair = X[mask]
            y_pair = (y[mask] == cb).astype(int)
            nets = []
            for i in range(self.n_estimators):
                Xb, yb = self._pair_balanced_subset(X_pair, y_pair, rng)
                net = BPNN(
                    self.n_hidden,
                    self.lr,
                    self.epochs,
                    random_state=self.random_state + pair_id * 100 + i,
                ).fit(Xb, yb, 2)
                nets.append(net)
            pair = (int(ca), int(cb))
            self.pairs_.append(pair)
            self.models_[pair] = {"nets": nets}
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        out = np.zeros((len(X), self.n_classes_), dtype=float)
        for pair in self.pairs_:
            ca, cb = pair
            model = self.models_[pair]
            pair_acc = np.zeros((len(X), 2), dtype=float)
            for net in model["nets"]:
                pair_acc += self._align_binary_proba(net, X)
            pair_acc /= max(len(model["nets"]), 1)
            pair_acc /= np.clip(pair_acc.sum(1, keepdims=True), 1e-12, None)
            out[:, self.class_to_index_[ca]] += pair_acc[:, 0]
            out[:, self.class_to_index_[cb]] += pair_acc[:, 1]
        out /= np.clip(out.sum(1, keepdims=True), 1e-12, None)
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


def build(random_state=42, smoke=False, n_estimators=30, epochs=120, **kw):
    if smoke:
        n_estimators, epochs = min(n_estimators, 5), 40
    return EasyBPNN(
        n_estimators=n_estimators,
        epochs=epochs,
        random_state=random_state,
        **kw,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
