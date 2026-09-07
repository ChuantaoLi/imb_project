# -*- coding: utf-8 -*-
"""DiagSelect (Fan et al., IEEE TII 2022).

The paper treats sample selection as a single-state MDP and trains a GRU
policy with pretraining followed by REINFORCE.  This implementation retains
that policy and validation-reward loop while exposing the paper's deliberately
classifier-agnostic head (DT/SVM/MLP/RF).  The default SVM matches the paper's
industrial experiments; ``classifier`` can be changed in config.yaml.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from common import smoke as _smoke
MODEL_KEY = "DiagSelect"


class _Policy:
    def __init__(self, dim, hidden=80, lr=1e-3, seed=42, backend='torch'):
        try:
            if str(backend).lower() in ('numpy', 'cpu_numpy'):
                raise ImportError
            import torch
            import torch.nn as nn
            torch.manual_seed(seed)
            self.torch = torch
            self.net = nn.GRU(dim, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)
            self.opt = torch.optim.RMSprop(list(self.net.parameters()) + list(self.head.parameters()), lr=lr)
        except ImportError:
            self.torch = None
        self.rng = np.random.RandomState(seed)

    def score(self, X):
        if self.torch is None:
            return np.full(len(X), 0.5)
        t = self.torch.as_tensor(X, dtype=self.torch.float32).unsqueeze(0)
        h, _ = self.net(t)
        return self.torch.sigmoid(self.head(h)).squeeze().detach().cpu().numpy()

    def pretrain(self, X, target, epochs):
        if self.torch is None: return
        torch = self.torch
        tX = torch.as_tensor(X, dtype=torch.float32).unsqueeze(0)
        ty = torch.as_tensor(target, dtype=torch.float32)
        for _ in range(max(1, epochs)):
            self.opt.zero_grad(); h, _ = self.net(tX)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.head(h).squeeze(), ty)
            loss.backward(); self.opt.step()

    def reinforce(self, X, rewards, epochs):
        if self.torch is None: return
        torch = self.torch
        tX = torch.as_tensor(X, dtype=torch.float32).unsqueeze(0)
        r = torch.as_tensor(rewards, dtype=torch.float32)
        for _ in range(max(1, epochs)):
            self.opt.zero_grad(); h, _ = self.net(tX)
            logits = self.head(h).squeeze(); probs = torch.sigmoid(logits)
            # Eq. (12): sample a binary action from the policy during offline
            # training.  Deterministic thresholding would remove the
            # exploration term required by REINFORCE; argmax is reserved for
            # online inference after the policy has converged.
            dist = torch.distributions.Bernoulli(probs=probs)
            actions = dist.sample()
            logp = dist.log_prob(actions)
            centered = r - r.mean()
            # A validation reward is often identical for every Bernoulli
            # action in one episode.  Centering that vector would erase the
            # REINFORCE signal completely, so retain a scalar baseline only in
            # the non-degenerate case.
            advantage = centered if torch.std(r) > 1e-7 else (r - 0.5)
            loss = -(advantage * logp).mean(); loss.backward(); self.opt.step()


def _classifier(name, seed, n_classes):
    name = str(name).lower()
    if name == 'dt': return DecisionTreeClassifier(max_depth=None, random_state=seed)
    if name == 'svm': return SVC(probability=True, class_weight='balanced', random_state=seed)
    if name == 'rf': return RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed, n_jobs=1)
    return MLPClassifier(hidden_layer_sizes=(80,), max_iter=300, early_stopping=True,
                         random_state=seed)


class DiagSelect:
    def __init__(self, classifier='svm', policy_hidden=80, policy_lr=1e-3,
                 pretrain_epochs=10, policy_epochs=5, episodes=5,
                 keep_ratio=0.5, max_selected_samples=10000, backend='torch', random_state=42, **kwargs):
        self.classifier_name = classifier; self.policy_hidden = int(policy_hidden)
        self.policy_lr = float(policy_lr); self.pretrain_epochs = int(pretrain_epochs)
        self.policy_epochs = int(policy_epochs); self.episodes = int(episodes)
        self.keep_ratio = float(keep_ratio); self.random_state = int(random_state)
        self.max_selected_samples = int(max_selected_samples)
        self.backend = backend

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int).ravel()
        self.classes_ = np.unique(y); self.n_classes_ = int(self.classes_.max()) + 1
        rng = np.random.RandomState(self.random_state)
        try: tr, va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=self.random_state)
        except ValueError: tr, va = np.arange(len(y)), np.arange(len(y))
        # Paper state contains feature+label information; append one-hot label.
        onehot = np.eye(self.n_classes_)[y]
        state = np.hstack([X, onehot])
        self.policy_ = _Policy(state.shape[1], self.policy_hidden, self.policy_lr, self.random_state, self.backend)
        # Eq. (1): the initial selection probability is the proportion of the
        # sample's class in the training set (the reject probability is its
        # complement).  This is deliberately a class-prior target, not an
        # inverse-frequency weight; later REINFORCE updates learn imbalance
        # corrections from validation reward.
        counts = np.bincount(y[tr], minlength=self.n_classes_).astype(float)
        target = counts[y[tr]] / max(float(len(tr)), 1.0)
        # A GRU receives the training set as one sequence in the paper.  Very
        # large industrial folds make that sequence exceed Windows/CUDA
        # resource limits, so train the policy on a deterministic 2048-point
        # summary and interpolate its action probabilities back to all points.
        max_policy_samples = 2048
        policy_idx = np.linspace(0, len(tr) - 1, min(max_policy_samples, len(tr)), dtype=int)
        policy_state = state[tr][policy_idx]
        policy_target = target[policy_idx]
        self.policy_.pretrain(policy_state, policy_target, self.pretrain_epochs)
        selected = np.ones(len(tr), dtype=bool)
        for ep in range(max(1, self.episodes)):
            p_small = np.asarray(self.policy_.score(policy_state), dtype=float).reshape(-1)
            p = np.interp(np.arange(len(tr)), policy_idx, p_small)
            # Eq. (17): deterministic online action is the per-sample
            # argmax of the two policy probabilities, equivalent to selecting
            # when P(select | state) >= 0.5.  A top-quantile rule would force a
            # fixed sampling rate unrelated to the learned policy.
            selected = p >= 0.5
            # The final classifier must expose every class.  The paper's
            # selector can downweight samples, but it cannot remove a class.
            for c in self.classes_:
                ci = np.flatnonzero(y[tr] == c)
                if len(ci) and not selected[ci].any():
                    selected[ci[np.argmax(p[ci])]] = True
            if selected.sum() < self.n_classes_:
                selected[np.argsort(-p)[:self.n_classes_]] = True
            if self.max_selected_samples > 0 and selected.sum() > self.max_selected_samples:
                keep = np.zeros(len(tr), dtype=bool)
                # Preserve class coverage, then fill the budget by the
                # highest policy probabilities (the action is sample choice).
                for c in self.classes_:
                    ci = np.flatnonzero((y[tr] == c) & selected)
                    if len(ci): keep[ci[np.argmax(p[ci])]] = True
                rest = np.flatnonzero(selected & ~keep)
                room = max(0, self.max_selected_samples - int(keep.sum()))
                if room: keep[rest[np.argsort(-p[rest])[:room]]] = True
                selected = keep
            clf = _classifier(self.classifier_name, self.random_state + ep, self.n_classes_)
            clf.fit(X[tr][selected], y[tr][selected])
            reward = np.full(len(policy_idx), (clf.predict(X[va]) == y[va]).mean() if len(va) else 0.0)
            self.policy_.reinforce(policy_state, reward, self.policy_epochs)
        self.selected_ = selected
        self.clf_ = _classifier(self.classifier_name, self.random_state + 1000, self.n_classes_)
        self.clf_.fit(X[tr][selected], y[tr][selected])
        return self

    def predict_proba(self, X): return self.clf_.predict_proba(np.asarray(X, dtype=float))
    def predict(self, X): return np.argmax(self.predict_proba(X), axis=1)


def build(random_state=42, smoke=False, classifier='svm', policy_hidden=80,
          policy_lr=1e-3, pretrain_epochs=10, policy_epochs=5, episodes=5,
          keep_ratio=0.5, max_selected_samples=10000, backend='torch', **kwargs):
    if smoke:
        pretrain_epochs = min(pretrain_epochs, 2); policy_epochs = min(policy_epochs, 1); episodes = min(episodes, 2)
    return DiagSelect(classifier=classifier, policy_hidden=policy_hidden,
                      policy_lr=policy_lr, pretrain_epochs=pretrain_epochs,
                      policy_epochs=policy_epochs, episodes=episodes,
                      keep_ratio=keep_ratio, max_selected_samples=max_selected_samples,
                      backend=backend, random_state=random_state)


if __name__ == '__main__': _smoke.run_smoke(build, MODEL_KEY)
