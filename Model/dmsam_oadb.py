# -*- coding: utf-8 -*-
"""DMSAM-OAdB (Zhu & Zhang, IEEE TIM 2026).

Implements the paper's multiscale sparse attention module (parallel 1-D
convolutions, adaptive soft-threshold attention) and optimized AdaBoost
(SAMME.R alpha, 10% correctly-classified majority zeroing, capped by the
minority count).  The source benchmark contains engineered feature vectors,
so vectors are treated as a one-channel sequence; raw furnace images/signals
are not fabricated.
"""
import os
import sys
import numpy as np
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from common import smoke as _smoke
MODEL_KEY = "DMSAM-OAdB"


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - only used on minimal CPU installs
    torch = None


if torch is not None:
    class _MSAM(nn.Module):
        def __init__(self, channels=16, kernels=(3, 5, 7)):
            super().__init__(); self.reduce = nn.Conv1d(1, channels, 1)
            self.branches = nn.ModuleList([nn.Conv1d(channels, channels, k, padding=k//2) for k in kernels])
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.scale = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())

        def forward(self, x):
            u = self.reduce(x)
            fused = sum(b(u) for b in self.branches)
            z = self.gap(torch.abs(fused)).squeeze(-1)
            # Eq. (7): channel threshold = sigmoid(FC(GAP)) * mean(abs(feature)).
            tau = self.scale(z).unsqueeze(-1) * torch.mean(torch.abs(fused), dim=2, keepdim=True)
            soft = torch.sign(fused) * F.relu(torch.abs(fused) - tau)
            return fused * torch.sigmoid(soft)


    class _DMSAMNet(nn.Module):
        def __init__(self, n_classes, channels=16, dropout=0.1):
            super().__init__(); self.msam = _MSAM(channels)
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                      nn.Linear(channels, 64), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(64, n_classes))

        def forward(self, x): return self.head(self.msam(x))


class DMSAMOAdB:
    def __init__(self, n_estimators=4, channels=16, epochs=50, batch_size=64,
                 lr=1e-3, gamma=2.0, zero_ratio=0.1, weight_decay=0.0,
                 device=None, random_state=42, **kwargs):
        self.n_estimators = int(n_estimators); self.channels = int(channels)
        self.epochs = int(epochs); self.batch_size = int(batch_size); self.lr = float(lr)
        self.gamma = float(gamma); self.zero_ratio = float(zero_ratio); self.weight_decay = float(weight_decay)
        self.device = device or ('cuda' if torch is not None and torch.cuda.is_available() else 'cpu')
        self.random_state = int(random_state)

    def _loss(self, logits, y, sample_w, class_w):
        p = torch.softmax(logits, dim=1)
        beta = class_w / class_w.max().clamp_min(1e-8)
        target = F.one_hot(y, num_classes=logits.shape[1]).float()
        # Eq. (21): positive and negative class terms with tanh focal
        # modulation.  beta is inverse-frequency class balancing.
        pos = beta[None, :] * torch.tanh(1.0 - p).pow(self.gamma) * torch.log(p + 1e-8)
        neg = (1.0 - beta[None, :]) * torch.tanh(p).pow(self.gamma) * torch.log(1.0 - p + 1e-8)
        loss = -(target * pos + (1.0 - target) * neg).sum(dim=1)
        return (loss * sample_w).sum() / sample_w.sum().clamp_min(1e-8)

    def _train_one(self, X, y, weights, seed):
        if torch is None:
            from sklearn.neural_network import MLPClassifier
            m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=max(100, self.epochs), random_state=seed)
            m.fit(X, y); return m
        torch.manual_seed(seed); np.random.seed(seed)
        net = _DMSAMNet(self.n_classes_, self.channels).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.device).unsqueeze(1)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        wt = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        cw = 1.0 / np.maximum(counts, 1.0); cw /= cw.mean()
        class_w = torch.as_tensor(cw, dtype=torch.float32, device=self.device)
        net.train()
        # Keep OAdB's per-sample distribution in every loss while using
        # bounded mini-batches.  Full-batch convolution on the largest
        # bearing fold can retain close to a gigabyte of CPU allocator state.
        batch_size = max(1, min(self.batch_size, len(xt)))
        order_rng = torch.Generator(device="cpu").manual_seed(int(seed))
        for _ in range(max(1, self.epochs)):
            order = torch.randperm(len(xt), generator=order_rng, device="cpu")
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size].to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = self._loss(net(xt[idx]), yt[idx], wt[idx], class_w)
                loss.backward(); opt.step()
        return net

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int).ravel()
        self.classes_ = np.unique(y); self.n_classes_ = int(self.classes_.max()) + 1
        rng = np.random.RandomState(self.random_state); n = len(y)
        D = np.full(n, 1.0 / max(1, n)); self.models_ = []; self.estimator_weights_ = []
        counts = Counter(y); minority = min(counts.values()); maj = max(counts, key=counts.get)
        for t in range(max(1, self.n_estimators)):
            active = D > 0
            if active.sum() < self.n_classes_:
                active[:] = True; D = np.full(n, 1.0 / n)
            model = self._train_one(X, y, D, self.random_state + t)
            p = self._predict_model(model, X); pred = np.argmax(p, axis=1)
            err = float(np.sum(D[pred != y])); err = np.clip(err, 1e-8, 1.0 - 1e-8)
            alpha = 0.5 * np.log((1.0 - err) / err) + 0.5 * np.log(max(1, self.n_classes_ - 1))
            self.models_.append(model); self.estimator_weights_.append(max(0.0, float(alpha)))
            D *= np.where(pred == y, np.exp(-alpha), np.exp(alpha)); D /= max(D.sum(), 1e-12)
            # OAdB Eq. (16): randomly remove at most 10% of correctly
            # classified majority points, capped by the minority count.
            good = np.flatnonzero((y == maj) & (pred == y) & (D > 0))
            k = min(len(good), minority, int(np.floor(self.zero_ratio * len(good))))
            if k > 0: D[rng.choice(good, size=k, replace=False)] = 0.0
            if D.sum() <= 0: D = np.full(n, 1.0 / n)
            else: D /= D.sum()
        self.estimator_weights_ = np.asarray(self.estimator_weights_, dtype=float)
        if not np.any(self.estimator_weights_): self.estimator_weights_[:] = 1.0
        return self

    @staticmethod
    def _predict_model(model, X):
        if torch is not None and isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                z = model(torch.as_tensor(X, dtype=torch.float32, device=next(model.parameters()).device).unsqueeze(1))
                return torch.softmax(z, dim=1).cpu().numpy()
        return model.predict_proba(X)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float); scores = np.zeros((len(X), self.n_classes_))
        for a, m in zip(self.estimator_weights_, self.models_):
            p = np.clip(self._predict_model(m, X), 1e-8, 1.0)
            scores += float(a) * np.log(p)
        scores -= scores.max(axis=1, keepdims=True); p = np.exp(scores)
        return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, X): return np.argmax(self.predict_proba(X), axis=1)


def build(random_state=42, smoke=False, n_estimators=4, channels=16,
          epochs=50, batch_size=64, lr=1e-3, gamma=2.0, zero_ratio=0.1,
          device=None, **kwargs):
    if smoke: n_estimators = min(n_estimators, 2); epochs = min(epochs, 3)
    return DMSAMOAdB(n_estimators=n_estimators, channels=channels, epochs=epochs,
                     batch_size=batch_size, lr=lr, gamma=gamma,
                     zero_ratio=zero_ratio, device=device, random_state=random_state)


if __name__ == '__main__': _smoke.run_smoke(build, MODEL_KEY)
