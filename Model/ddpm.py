# -*- coding: utf-8 -*-
"""DDPM fault-data augmentation (Zhao et al., EAAI 139 (2025) 109520).

The paper converts raw vibration windows to CWT images, trains a conditional
DDPM with a U-Net denoiser, then diagnoses with a 2-D CNN.  This benchmark's
input contract is a fixed table of 24 engineered features, so the faithful
data-level operations are implemented in feature space: a conditional MLP
denoiser replaces the spatial U-Net and a one-channel CNN replaces the 2-D
CNN.  The forward/reverse Gaussian transitions, epsilon objective, class-wise
augmentation, Adam optimisation and classifier cross-entropy remain unchanged.
"""
import os
import sys
from collections import Counter
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from common import smoke as _smoke
MODEL_KEY = "DDPM"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None


if torch is not None:
    class _Denoiser(nn.Module):
        def __init__(self, dim, n_classes, hidden=128, t_dim=64, diffusion_steps=1000):
            super().__init__()
            self.time = nn.Embedding(max(1, int(diffusion_steps)), t_dim)
            self.label = nn.Embedding(max(1, n_classes), 32)
            self.net = nn.Sequential(nn.Linear(dim + t_dim + 32, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dim))

        def forward(self, x, t, y):
            return self.net(torch.cat([x, self.time(t), self.label(y)], dim=1))


    class _FeatureCNN(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 6, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(6, 1, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.fc = nn.Sequential(nn.Flatten(), nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, n_classes))

        def forward(self, x): return self.fc(self.features(x))


class DDPM:
    def __init__(self, diffusion_steps=1000, diffusion_epochs=3000,
                 classifier_epochs=300, batch_size=64, lr=1e-3,
                 max_generated_per_class=500, balance=True, denoiser_hidden=128,
                 device=None, random_state=42, **kwargs):
        self.T = int(diffusion_steps); self.diffusion_epochs = int(diffusion_epochs)
        self.classifier_epochs = int(classifier_epochs); self.batch_size = int(batch_size)
        self.lr = float(lr); self.max_generated = int(max_generated_per_class)
        self.balance = bool(balance); self.denoiser_hidden = int(denoiser_hidden)
        self.device = device or ('cuda' if torch is not None and torch.cuda.is_available() else 'cpu')
        self.random_state = int(random_state)

    def _schedule(self):
        # Ho et al. linear beta schedule, beta_1=1e-4, beta_T=0.02.
        b = torch.linspace(1e-4, 2e-2, self.T, device=self.device)
        a = 1.0 - b; abar = torch.cumprod(a, dim=0)
        return b, a, abar

    def _q_sample(self, x0, t, abar):
        noise = torch.randn_like(x0)
        at = abar[t].sqrt().unsqueeze(1); st = (1.0 - abar[t]).sqrt().unsqueeze(1)
        return at * x0 + st * noise, noise

    def _train_diffusion(self, X, y):
        if torch is None: return None
        net = _Denoiser(X.shape[1], self.n_classes_, self.denoiser_hidden,
                        t_dim=64, diffusion_steps=self.T).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        b, a, abar = self._schedule()
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        net.train(); n = len(X)
        for _ in range(max(1, self.diffusion_epochs)):
            idx = torch.randint(0, n, (min(self.batch_size, n),), device=self.device)
            t = torch.randint(0, self.T, (len(idx),), device=self.device)
            noisy, noise = self._q_sample(xt[idx], t, abar)
            loss = F.mse_loss(net(noisy, t, yt[idx]), noise)
            opt.zero_grad(); loss.backward(); opt.step()
        return net

    def _sample(self, net, n, cls, dim):
        if n <= 0: return np.empty((0, dim), dtype=float)
        b, a, abar = self._schedule(); x = torch.randn((n, dim), device=self.device)
        y = torch.full((n,), int(cls), dtype=torch.long, device=self.device)
        net.eval()
        with torch.no_grad():
            for ti in range(self.T - 1, -1, -1):
                t = torch.full((n,), ti, dtype=torch.long, device=self.device)
                eps = net(x, t, y); alpha = a[ti]; beta = b[ti]; ab = abar[ti]
                mean = (x - beta / torch.sqrt(1.0 - ab) * eps) / torch.sqrt(alpha)
                if ti > 0: x = mean + torch.sqrt(beta) * torch.randn_like(x)
                else: x = mean
        return x.cpu().numpy()

    def _train_classifier(self, X, y):
        if torch is None:
            from sklearn.neural_network import MLPClassifier
            m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=max(100, self.classifier_epochs), random_state=self.random_state)
            m.fit(X, y); return m
        torch.manual_seed(self.random_state)
        net = _FeatureCNN(self.n_classes_).to(self.device); opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.device).unsqueeze(1)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        cw = 1.0 / np.maximum(counts, 1.0); cw /= cw.mean()
        weight = torch.as_tensor(cw, dtype=torch.float32, device=self.device)
        net.train()
        n = len(X)
        for _ in range(max(1, self.classifier_epochs)):
            # The paper trains the CNN with mini-batches (64 in the reported
            # experiments), rather than silently turning batch_size into a
            # no-op full-batch setting.
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, max(1, self.batch_size)):
                idx = order[start:start + max(1, self.batch_size)]
                opt.zero_grad(); loss = F.cross_entropy(net(xt[idx]), yt[idx], weight=weight)
                loss.backward(); opt.step()
        return net

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int).ravel()
        self.classes_ = np.unique(y); self.n_classes_ = int(self.classes_.max()) + 1
        np.random.seed(self.random_state)
        if torch is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.random_state)
        self.diffusion_ = self._train_diffusion(X, y)
        X_aug, y_aug = [X], [y]
        if self.diffusion_ is not None and self.balance:
            counts = Counter(y); target = max(counts.values())
            for c in self.classes_:
                need = min(self.max_generated, max(0, target - counts[int(c)]))
                if need:
                    X_aug.append(self._sample(self.diffusion_, need, int(c), X.shape[1]))
                    y_aug.append(np.full(need, int(c), dtype=int))
        self.X_aug_ = np.vstack(X_aug); self.y_aug_ = np.concatenate(y_aug)
        self.classifier_ = self._train_classifier(self.X_aug_, self.y_aug_)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if torch is not None and isinstance(self.classifier_, nn.Module):
            self.classifier_.eval()
            with torch.no_grad():
                z = self.classifier_(torch.as_tensor(X, dtype=torch.float32, device=self.device).unsqueeze(1))
                return torch.softmax(z, dim=1).cpu().numpy()
        return self.classifier_.predict_proba(X)

    def predict(self, X): return np.argmax(self.predict_proba(X), axis=1)


def build(random_state=42, smoke=False, diffusion_steps=1000, diffusion_epochs=3000,
          classifier_epochs=300, batch_size=64, lr=1e-3, max_generated_per_class=500,
          balance=True, denoiser_hidden=128, device=None, **kwargs):
    if smoke:
        diffusion_steps = min(diffusion_steps, 20); diffusion_epochs = min(diffusion_epochs, 3); classifier_epochs = min(classifier_epochs, 3)
    return DDPM(diffusion_steps=diffusion_steps, diffusion_epochs=diffusion_epochs,
                classifier_epochs=classifier_epochs, batch_size=batch_size, lr=lr,
                max_generated_per_class=max_generated_per_class, balance=balance,
                denoiser_hidden=denoiser_hidden, device=device, random_state=random_state)


if __name__ == '__main__': _smoke.run_smoke(build, MODEL_KEY)
