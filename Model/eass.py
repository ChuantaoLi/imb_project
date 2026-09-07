# -*- coding: utf-8 -*-
"""EASS: Ensemble Automatic Sample Selector with Soft Actor-Critic.

The available article text specifies an SAC sampler whose state is the
training/validation error density and whose action is a low-complexity sample
weight.  This module implements that contract with a compact Gaussian actor,
twin critics and entropy regularisation, then trains the selected ensemble.
    The diagnosis head is intentionally configurable because EASS is a data-level
    method; the paper's experiments use decision trees.  ``dt`` is the default
    and RF/SVM/MLP remain available for controlled comparisons.
"""
import os
import sys
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from common import smoke as _smoke
MODEL_KEY = "EASS"


def _make_clf(name, seed):
    name = str(name).lower()
    if name == 'dt': return DecisionTreeClassifier(random_state=seed)
    if name == 'svm': return SVC(probability=True, class_weight='balanced', random_state=seed)
    if name == 'rf': return RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed, n_jobs=1)
    return MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, early_stopping=True, random_state=seed)


class _SAC:
    def __init__(self, dim, action_dim, hidden=64, lr=3e-4, gamma=0.99,
                 tau=0.005, alpha=0.2, seed=42, backend='torch'):
        self.rng = np.random.RandomState(seed); self.gamma = gamma
        self.tau = tau; self.alpha = alpha; self.action_dim = int(action_dim)
        try:
            if str(backend).lower() in ('numpy', 'cpu_numpy'):
                raise ImportError
            import torch
            import torch.nn as nn
            torch.manual_seed(seed); self.torch = torch
            self.actor = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                       nn.Linear(hidden, 2 * self.action_dim))
            self.q1 = nn.Sequential(nn.Linear(dim + self.action_dim, hidden),
                                    nn.ReLU(), nn.Linear(hidden, 1))
            # Independent initialization is required for the clipped double-Q
            # estimate; copying q1 would make both critics evolve identically.
            self.q2 = nn.Sequential(nn.Linear(dim + self.action_dim, hidden),
                                    nn.ReLU(), nn.Linear(hidden, 1))
            self.q1_target = copy.deepcopy(self.q1)
            self.q2_target = copy.deepcopy(self.q2)
            self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        except ImportError:
            self.torch = None

    def _sample(self, states):
        torch = self.torch
        params = self.actor(states).reshape(-1, 2, self.action_dim)
        mean, log_std = params[:, 0], params[:, 1].clamp(-5.0, 2.0)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample()
        action = torch.sigmoid(raw).clamp(1e-5, 1.0 - 1e-5)
        logp = (normal.log_prob(raw) - torch.log(action) - torch.log1p(-action)).sum(dim=1, keepdim=True)
        return action, logp

    def act(self, state):
        if self.torch is None:
            mu = self.rng.uniform(0.0, 1.0, self.action_dim)
            sigma = self.rng.uniform(0.05, 0.25, self.action_dim)
            return np.clip(mu + sigma * self.rng.randn(self.action_dim), 0.0, 1.0)
        with self.torch.no_grad():
            s = self.torch.as_tensor(np.asarray(state)[None], dtype=self.torch.float32)
            action, _ = self._sample(s)
            return action.squeeze(0).cpu().numpy()

    def update(self, states, actions, rewards, next_states=None, steps=1):
        if self.torch is None or len(states) == 0: return
        torch = self.torch
        s = torch.as_tensor(states, dtype=torch.float32)
        a = torch.as_tensor(actions, dtype=torch.float32).reshape(-1, self.action_dim)
        r = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, 1)
        ns = s if next_states is None else torch.as_tensor(next_states, dtype=torch.float32)
        for _ in range(max(1, steps)):
            with torch.no_grad():
                na, nlogp = self._sample(ns)
                q_next = torch.minimum(self.q1_target(torch.cat([ns, na], 1)),
                                       self.q2_target(torch.cat([ns, na], 1)))
                target = r + self.gamma * (q_next - self.alpha * nlogp)
            self.opt_q.zero_grad(set_to_none=True)
            q1 = self.q1(torch.cat([s, a], 1)); q2 = self.q2(torch.cat([s, a], 1))
            qloss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()
            qloss.backward(); self.opt_q.step()
            self.opt_a.zero_grad(set_to_none=True)
            pa, logp = self._sample(s)
            aloss = (self.alpha * logp - torch.minimum(
                self.q1(torch.cat([s, pa], 1)), self.q2(torch.cat([s, pa], 1)))).mean()
            aloss.backward(); self.opt_a.step()
            with torch.no_grad():
                for target_net, online_net in ((self.q1_target, self.q1), (self.q2_target, self.q2)):
                    for tp, p in zip(target_net.parameters(), online_net.parameters()):
                        tp.mul_(1.0 - self.tau).add_(self.tau * p)


class EASS:
    def __init__(self, classifier='dt', n_estimators=5, hidden_dim=64,
                 episodes=5, updates=2, keep_ratio=0.7, backend='torch', random_state=42, **kwargs):
        self.classifier_name = classifier; self.n_estimators = int(n_estimators)
        self.hidden_dim = int(hidden_dim); self.episodes = int(episodes); self.updates = int(updates)
        self.keep_ratio = float(keep_ratio); self.random_state = int(random_state)
        self.backend = backend

    @staticmethod
    def _errors(X, y, ensemble, n_classes):
        if not ensemble:
            return np.full(len(y), 1.0 - 1.0 / max(1, n_classes), dtype=float)
        p = np.mean([m.predict_proba(X) for m in ensemble], axis=0)
        return np.clip(1.0 - p[np.arange(len(y)), y], 0.0, 1.0)

    @classmethod
    def _state(cls, Xtr, ytr, Xval, yval, ensemble, classes, bins=10):
        """Eq. (20)-(25): class-wise error-density coordination state."""
        out = []
        for X, y in ((Xtr, ytr), (Xval, yval)):
            e = cls._errors(X, y, ensemble, len(classes))
            rho = []
            for c in classes:
                ec = e[y == c]
                if len(ec) == 0:
                    rho.append(0.0); continue
                # Histogram density weighted by hard-example error.  rho=N/ED
                # is bounded so it remains numerically useful to SAC.
                hist, edges = np.histogram(ec, bins=bins, range=(0.0, 1.0))
                centers = (edges[:-1] + edges[1:]) * 0.5
                ed = float(np.sum(hist * np.maximum(centers, 1e-3)))
                rho.append(float(len(ec) / max(ed, 1e-6)))
            rho = np.asarray(rho, dtype=float)
            out.append(rho / max(float(np.max(rho)), 1.0))
        return np.concatenate(out)

    @staticmethod
    def _sample_indices(X, y, errors, action, classes, rng):
        """Eq. (17)-(18): class-normalized Gaussian hard-example weights."""
        selected = []
        for c, params in zip(classes, np.asarray(action).reshape(-1, 2)):
            idx = np.flatnonzero(y == c)
            if not len(idx):
                continue
            mu = float(np.clip(params[0], 0.0, 1.0))
            sigma = float(np.clip(abs(params[1]), 0.03, 1.0))
            e = errors[idx]
            w = np.exp(-0.5 * ((e - mu) / sigma) ** 2) / sigma
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            w += 1e-12; w /= w.sum()
            selected.append((idx, w))
        if not selected:
            return np.empty(0, dtype=int)
        target = min(len(idx) for idx, _ in selected)
        picked = [rng.choice(idx, size=target, replace=False, p=w) for idx, w in selected]
        return np.concatenate(picked)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int).ravel()
        self.classes_ = np.unique(y); self.n_classes_ = int(self.classes_.max()) + 1
        rng = np.random.RandomState(self.random_state)
        try: tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=self.random_state)
        except ValueError: tr, va = np.arange(len(y)), np.arange(len(y))
        self.actor_ = _SAC(2 * self.n_classes_, 2 * self.n_classes_, self.hidden_dim,
                           lr=1e-3, seed=self.random_state, backend=self.backend)
        best_models, best_score, best_selected = [], -np.inf, None
        for ep in range(max(1, self.episodes)):
            episode_models = []
            for step in range(max(1, self.n_estimators)):
                st = self._state(X[tr], y[tr], X[va], y[va], episode_models, self.classes_)
                action = self.actor_.act(st)
                errors = self._errors(X[tr], y[tr], episode_models, self.n_classes_)
                selected_local = self._sample_indices(X[tr], y[tr], errors, action,
                                                      self.classes_, rng)
                if len(selected_local) == 0:
                    selected_local = np.arange(len(tr))
                clf = _make_clf(self.classifier_name, self.random_state + ep * self.n_estimators + step)
                clf.fit(X[tr][selected_local], y[tr][selected_local])
                episode_models.append(clf)
                score = f1_score(y[va], np.argmax(np.mean(
                    [m.predict_proba(X[va]) for m in episode_models], axis=0), axis=1),
                    average='macro') if len(va) else 0.0
                next_st = self._state(X[tr], y[tr], X[va], y[va], episode_models, self.classes_)
                self.actor_.update(st[None, :], np.asarray(action)[None, :], [score],
                                    next_states=next_st[None, :], steps=self.updates)
            if episode_models:
                final_score = f1_score(y[va], np.argmax(np.mean(
                    [m.predict_proba(X[va]) for m in episode_models], axis=0), axis=1),
                    average='macro') if len(va) else 0.0
                if final_score > best_score:
                    best_score = final_score; best_models = episode_models
                    best_selected = selected_local
        self.models_ = best_models
        self.selected_ = best_selected
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(np.asarray(X, dtype=float)) for m in self.models_], axis=0)
        return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    def predict(self, X): return np.argmax(self.predict_proba(X), axis=1)


def build(random_state=42, smoke=False, classifier='dt', n_estimators=5,
          hidden_dim=64, episodes=5, updates=2, keep_ratio=0.7, backend='torch', **kwargs):
    if smoke: episodes = min(episodes, 2); updates = min(updates, 1); n_estimators = min(n_estimators, 2)
    return EASS(classifier=classifier, n_estimators=n_estimators,
                hidden_dim=hidden_dim, episodes=episodes, updates=updates,
                keep_ratio=keep_ratio, backend=backend, random_state=random_state)


if __name__ == '__main__': _smoke.run_smoke(build, MODEL_KEY)
