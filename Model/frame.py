"""FRAME: Feature Rectification for Class Imbalance Learning.

Fully-supervised FRAME for the benchmark interface.  The implementation follows
the paper's tabular setting: a single-layer MLP embedding network with 128
filters, self-attentive feature interaction to learn multiple class centroids,
and a distance-based softmax classifier trained end-to-end by negative log
likelihood.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from common import smoke as _smoke

MODEL_KEY = "FRAME"


class _FrameNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_centroids=3):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_centroids = int(n_centroids)
        self.embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.z_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.centroid_queries = nn.Parameter(torch.randn(n_centroids, hidden_dim) * 0.02)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim))

    def features(self, x):
        return self.embed(x)

    def class_centroids(self, h_cls):
        n = h_cls.shape[0]
        if n == 0:
            return h_cls.new_zeros((0, self.hidden_dim))
        z = self.z_proj(h_cls)
        k = self.k_proj(z)
        v = self.v_proj(z)
        k_cent = max(1, min(self.n_centroids, n))
        queries = self.centroid_queries[:k_cent]
        attn = torch.softmax((queries @ k.T) / np.sqrt(self.hidden_dim), dim=1)
        pooled = attn @ v
        residual = z.mean(dim=0, keepdim=True)
        return self.c_proj(pooled + residual)

    def centroids(self, h, y, classes):
        return {int(c): self.class_centroids(h[y == int(c)]) for c in classes}

    def logits_from_centroids(self, h, centroids, n_classes):
        scores = h.new_full((h.shape[0], n_classes), -1e9)
        for cls, c in centroids.items():
            if c.numel() == 0:
                continue
            d2 = torch.sum((h[:, None, :] - c[None, :, :]) ** 2, dim=2)
            scores[:, int(cls)] = torch.logsumexp(-d2, dim=1) - np.log(c.shape[0])
        return scores


class FRAME:
    def __init__(
        self,
        n_centroids=3,
        hidden_dim=128,
        lr=1e-4,
        epochs=1000,
        weight_decay=0.0,
        random_state=42,
        device=None,
        **kw
    ):
        self.n_centroids = int(n_centroids)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = None
        self.n_classes_ = None
        self.imputer_ = None
        self.net_ = None
        self.centroids_ = None

    def _tensor(self, X):
        return torch.as_tensor(X, dtype=torch.float32, device=self.device)

    def fit(self, X, y):
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.max()) + 1 if len(self.classes_) else 0
        self.imputer_ = SimpleImputer(strategy="mean")
        X_imp = self.imputer_.fit_transform(X)

        xt = self._tensor(X_imp)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.net_ = _FrameNet(X_imp.shape[1], self.hidden_dim, self.n_centroids).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.net_.train()
        for _ in range(max(1, self.epochs)):
            opt.zero_grad()
            h = self.net_.features(xt)
            centroids = self.net_.centroids(h, yt, self.classes_)
            logits = self.net_.logits_from_centroids(h, centroids, self.n_classes_)
            loss = F.cross_entropy(logits, yt)
            loss.backward()
            opt.step()

        self.net_.eval()
        with torch.no_grad():
            h = self.net_.features(xt)
            self.centroids_ = {
                cls: c.detach().clone()
                for cls, c in self.net_.centroids(h, yt, self.classes_).items()
            }
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X_imp = self.imputer_.transform(X)
        with torch.no_grad():
            h = self.net_.features(self._tensor(X_imp))
            logits = self.net_.logits_from_centroids(h, self.centroids_, self.n_classes_)
            p = torch.softmax(logits, dim=1).cpu().numpy()
        return p

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def build(random_state=42, smoke=False, **kw):
    return FRAME(random_state=random_state, **kw)


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
