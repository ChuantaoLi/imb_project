"""
DEAHS: Dynamic Ensemble fault diagnosis framework with Adaptive Hierarchical Sampling strategy
Complete reproduction of:
  "Dynamic ensemble fault diagnosis framework with adaptive hierarchical sampling strategy
   for industrial imbalanced and overlapping data"
  Haoyan Dong, Chuang Peng, Lei Chen, Kuangrong Hao
  Reliability Engineering and System Safety, 2025
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
# SMOTE import removed: safe-region oversampling uses equivalent manual interpolation

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Relocatable paths: resolved from THIS file so the module is importable and
# callable from any working directory (no hardcoded absolute paths).
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # D:\imb_project
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from common import smoke as _smoke

# ==============================================================================
# 1. SAC-based Adaptive Undersampler (Outer Layer)
# ==============================================================================

class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SACActor(nn.Module):
    """SAC Actor: outputs mean and log_std for Gaussian policy -> [mu, sigma]."""
    def __init__(self, state_dim, hidden_dim=64, action_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = -4.0
        self.log_std_max = -0.5

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.sigmoid(self.mean_layer(x))
        log_std = torch.clamp(self.log_std_layer(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        if deterministic:
            return mean, None
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.sigmoid(z)
        log_prob = normal.log_prob(z) - torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACCritic(nn.Module):
    """SAC Twin Q-Network."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SACAgent:
    """Soft Actor-Critic for adaptive undersampling parameter selection."""
    def __init__(self, state_dim, action_dim=2, hidden_dim=64,
                 lr=3e-4, gamma=0.99, tau=0.005, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.actor = SACActor(state_dim, hidden_dim, action_dim).to(device)
        self.critic = SACCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=5000)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, state, deterministic=False):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a, _ = self.actor.sample(s, deterministic)
        return a.cpu().numpy()[0]

    def update(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            na, nlp = self.actor.sample(next_state)
            q1t, q2t = self.critic_target(next_state, na)
            qt = torch.min(q1t, q2t) - self.alpha * nlp
            q_backup = reward + (1 - done) * self.gamma * qt

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_a, log_prob = self.actor.sample(state)
        q1n, q2n = self.critic(state, new_a)
        actor_loss = (self.alpha * log_prob - torch.min(q1n, q2n)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)


# ==============================================================================
# 2. Inner Layer: Membership Entropy + Weighted Oversampling
# ==============================================================================

def compute_membership_and_entropy(X, y, n_fuzzy=2):
    """Compute membership degree (Eq. 9) and membership entropy (Eq. 10).

    m_i^j = 1 / sum_k (d_i^j / d_i^k)^{2/(n-1)}
    H(x_i) = -sum_k m_i^k * log(m_i^k)
    """
    classes = np.unique(y)
    n_samples = X.shape[0]
    n_classes = len(classes)
    exponent = 2.0 / (n_fuzzy - 1)  # =2 when n_fuzzy=2

    # Class centers
    centers = np.array([X[y == c].mean(axis=0) for c in classes])

    # Distances: (n_samples, n_classes)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, C, d)
    distances = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-10)  # (N, C)

    # Membership degrees (Eq. 9)
    membership = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        for j in range(n_classes):
            ratios = distances[i, j] / (distances[i, :] + 1e-10)
            membership[i, j] = 1.0 / (np.sum(ratios ** exponent) + 1e-10)

    # Entropy (Eq. 10)
    entropy = -np.sum(membership * np.log(membership + 1e-10), axis=1)
    return membership, entropy, classes


def divide_regions(entropy):
    """Divide into safe/overlap regions (Eq. 11-12). T_ov = mu_H + sigma_H."""
    T_ov = np.mean(entropy) + np.std(entropy)
    overlap_mask = entropy > T_ov
    safe_mask = ~overlap_mask
    return safe_mask, overlap_mask


def compute_between_class_scatter(X, y):
    """Between-class scatter matrix (Eq. 13)."""
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    n_features = X.shape[1]
    S_B = np.zeros((n_features, n_features))
    for c in classes:
        mask = y == c
        n_k = mask.sum()
        class_mean = X[mask].mean(axis=0)
        diff = (class_mean - overall_mean).reshape(-1, 1)
        S_B += n_k * (diff @ diff.T)
    return S_B


def weighted_oversample(X_min, y_min, weights, n_target, k=5, rng=None):
    """Weighted SMOTE oversampling for overlapping region minority samples (Eq. 14-15).

    alpha_i = (x_i - mu_j)^T S_B (x_i - mu_j)  -> contribution
    w_i = alpha_i / sum(alpha_j)                  -> sampling weight
    """
    if rng is None:
        rng = np.random.RandomState(42)
    n = X_min.shape[0]
    if n <= 1 or n_target <= n:
        return X_min.copy(), y_min.copy()

    n_synthetic = n_target - n
    w_norm = weights / (weights.sum() + 1e-10)
    counts = np.maximum(1, np.round(w_norm * n_synthetic).astype(int))

    # Adjust counts to match n_synthetic
    diff = n_synthetic - counts.sum()
    if diff > 0:
        top = np.argsort(w_norm)[-diff:]
        counts[top] += 1
    elif diff < 0:
        sorted_idx = np.argsort(w_norm)
        for idx in sorted_idx:
            if diff >= 0:
                break
            reduce = min(counts[idx] - 1, -diff)
            counts[idx] -= reduce
            diff += reduce

    actual_k = min(k, n - 1)
    if actual_k < 1:
        actual_k = 1

    syn_X, syn_y = [], []
    for i in range(n):
        if counts[i] <= 0:
            continue
        dists = np.sqrt(np.sum((X_min - X_min[i]) ** 2, axis=1))
        dists[i] = np.inf
        nn_idx = np.argsort(dists)[:actual_k]
        for _ in range(counts[i]):
            nn = rng.choice(nn_idx)
            gap = rng.uniform(0, 1)
            syn_X.append(X_min[i] + gap * (X_min[nn] - X_min[i]))
            syn_y.append(y_min[i])

    if not syn_X:
        return X_min.copy(), y_min.copy()
    return np.vstack([X_min, np.array(syn_X)]), np.concatenate([y_min, np.array(syn_y)])


# ==============================================================================
# 3. DEAHS Framework
# ==============================================================================

class DEAHS:
    """Dynamic Ensemble fault diagnosis with Adaptive Hierarchical Sampling."""

    def __init__(self, n_base_classifiers=30, n_histogram_bins=10,
                 alpha_gaussian=0.95, sac_hidden_dim=64, sac_lr=3e-4,
                 sac_batch_size=32, sac_update_steps=5,
                 smote_k_neighbors=5, fuzzification_param=2,
                 val_ratio=0.2, test_ratio=0.2,
                 rf_n_estimators=30, n_jobs=-1,
                 device='cpu', random_state=42, preencoded=False):
        self.preencoded = preencoded
        self.n_base = n_base_classifiers
        self.n_bins = n_histogram_bins
        self.alpha_gauss = alpha_gaussian
        self.sac_hidden = sac_hidden_dim
        self.sac_lr = sac_lr
        self.sac_batch = sac_batch_size
        self.sac_steps = sac_update_steps
        self.smote_k = smote_k_neighbors
        self.n_fuzzy = fuzzification_param
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rf_n_est = rf_n_estimators
        self.n_jobs = n_jobs
        self.device = device
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.base_classifiers = []
        self.classes_ = None
        self.majority_class_ = None
        self.n_classes_ = None

    def _ensemble_predict_proba(self, X):
        """Ensemble prediction using averaging (Eq. 2)."""
        if not self.base_classifiers:
            return None
        probas = [clf.predict_proba(X) for clf in self.base_classifiers]
        avg = np.mean(probas, axis=0)
        return avg / avg.sum(axis=1, keepdims=True)

    def _ensemble_predict(self, X):
        p = self._ensemble_predict_proba(X)
        if p is None:
            return None
        return np.argmax(p, axis=1)

    def _get_state(self, X_train, y_train, X_val, y_val, maj_mask, n_cls):
        """Compute state vector (Eq. 5): [hist_train, hist_val, Acc_minority].

        Eq. (4): Histogram bins range [0,1] with b bins: bin j covers [(j-1)/b, j/b].
        Eq. (5): s_t = [L_hat_N_tr, L_hat_N_val, Acc_P] in R^{2b+(C-1)}.
        Both histograms are computed on MAJORITY samples only (N).
        """
        proba_train = self._ensemble_predict_proba(X_train)
        if proba_train is None:
            proba_train = np.ones((X_train.shape[0], n_cls)) / n_cls

        # Cross-entropy loss on majority training samples (Eq. 3-4)
        y_maj = y_train[maj_mask]
        p_maj = proba_train[maj_mask]
        y_oh = np.zeros((len(y_maj), n_cls))
        y_oh[np.arange(len(y_maj)), y_maj.astype(int)] = 1.0
        L_maj = -np.sum(y_oh * np.log(p_maj + 1e-10), axis=1)

        # Normalize losses to [0,1] for histogram binning (Eq. 4: range 0~1)
        L_maj_min, L_maj_max = L_maj.min(), L_maj.max()
        L_maj_norm = (L_maj - L_maj_min) / (L_maj_max - L_maj_min + 1e-10)
        h_train, _ = np.histogram(L_maj_norm, bins=self.n_bins, range=(0, 1))
        h_train = h_train / (len(L_maj_norm) + 1e-10)

        # Cross-entropy loss on validation MAJORITY samples (Eq. 4-5)
        val_maj_mask = y_val == self.majority_class_
        proba_val = self._ensemble_predict_proba(X_val)
        if proba_val is None:
            proba_val = np.ones((X_val.shape[0], n_cls)) / n_cls

        p_val_maj = proba_val[val_maj_mask]
        y_val_maj = y_val[val_maj_mask]
        y_val_oh = np.zeros((len(y_val_maj), n_cls))
        y_val_oh[np.arange(len(y_val_maj)), y_val_maj.astype(int)] = 1.0
        L_val = -np.sum(y_val_oh * np.log(p_val_maj + 1e-10), axis=1)

        # Normalize losses to [0,1] for histogram binning (Eq. 4: range 0~1)
        L_val_min, L_val_max = L_val.min(), L_val.max()
        L_val_norm = (L_val - L_val_min) / (L_val_max - L_val_min + 1e-10)
        h_val, _ = np.histogram(L_val_norm, bins=self.n_bins, range=(0, 1))
        h_val = h_val / (len(L_val_norm) + 1e-10)

        # Minority class accuracy on validation (Eq. 5)
        acc_min = np.zeros(n_cls - 1)
        if len(self.base_classifiers) > 0:
            y_pred_val = self._ensemble_predict(X_val)
            for idx, c in enumerate(range(n_cls)):
                if c == self.majority_class_:
                    continue
                actual_idx = c - 1 if c > self.majority_class_ else c
                mask = y_val == c
                if mask.sum() > 0:
                    acc_min[actual_idx] = (y_pred_val[mask] == c).mean()

        return np.concatenate([h_train, h_val, acc_min]).astype(np.float32)

    def _compute_metrics(self, X_val, y_val, n_cls):
        """Compute validation metrics [BA, G-mean, F1_macro, F1_micro] for reward."""
        if not self.base_classifiers:
            return [0.0] * 4
        y_pred = self._ensemble_predict(X_val)
        recalls = []
        for c in range(n_cls):
            m = y_val == c
            if m.sum() > 0:
                recalls.append((y_pred[m] == c).mean())
            else:
                recalls.append(0.0)
        ba = np.mean(recalls)
        gmean = np.prod([r for r in recalls if r > 0]) ** (1.0 / max(1, sum(1 for r in recalls if r > 0))) if any(r > 0 for r in recalls) else 0.0
        f1m = f1_score(y_val, y_pred, average='macro', zero_division=0)
        f1mi = f1_score(y_val, y_pred, average='micro', zero_division=0)
        return [ba, gmean, f1m, f1mi]

    def _outer_undersample(self, X_maj, y_maj, losses, mu, sigma, n_target, rng):
        """Outer layer adaptive undersampling (Eq. 8)."""
        n = X_maj.shape[0]
        if n <= n_target:
            return X_maj, y_maj

        # Normalize losses to [0,1]
        lmin, lmax = losses.min(), losses.max()
        l_norm = (losses - lmin) / (lmax - lmin + 1e-10)

        # Eq. 8: mixture of Gaussian and uniform
        gauss = (1.0 / (sigma * np.sqrt(2 * np.pi) + 1e-10)) * \
                np.exp(-0.5 * ((l_norm - mu) / (sigma + 1e-10)) ** 2)
        uniform = 1.0 / n
        weights = self.alpha_gauss * gauss + (1 - self.alpha_gauss) * uniform
        weights = np.maximum(weights, 1e-10)
        weights /= weights.sum()

        idx = rng.choice(n, size=n_target, replace=False, p=weights)
        return X_maj[idx], y_maj[idx]

    def _inner_oversample(self, X, y, n_cls, rng):
        """Inner layer: overlap identification + weighted oversampling (Algorithm 1 lines 10-17)."""
        # Step 1: Membership & entropy (Eq. 9-10)
        membership, entropy, classes = compute_membership_and_entropy(X, y, self.n_fuzzy)

        # Step 2: Region division (Eq. 11-12)
        safe_mask, overlap_mask = divide_regions(entropy)

        # Step 3: Between-class scatter (Eq. 13)
        S_B = compute_between_class_scatter(X, y)

        # Identify majority/minority
        maj_c = self.majority_class_

        # Target count per minority class
        n_maj = (y == maj_c).sum()
        target_per_class = n_maj  # balance each minority to majority level

        result_X = [X[y == maj_c]]
        result_y = [y[y == maj_c]]

        for c in range(n_cls):
            if c == maj_c:
                continue
            min_mask = y == c
            X_min = X[min_mask]
            y_min = y[min_mask]
            n_min = X_min.shape[0]

            if n_min == 0:
                continue

            min_global_idx = np.where(min_mask)[0]

            # Separate safe/overlap
            min_safe = safe_mask[min_global_idx]
            min_overlap = overlap_mask[min_global_idx]

            X_safe = X_min[min_safe]
            X_over = X_min[min_overlap]
            y_safe = y_min[min_safe]
            y_over = y_min[min_overlap]

            # Number of synthetic samples needed
            n_need = max(0, target_per_class - n_min)
            if n_need <= 0:
                result_X.append(X_min)
                result_y.append(y_min)
                continue

            os_X, os_y = [X_min], [y_min]

            # Determine split between safe and overlap synthetic samples
            n_ov = X_over.shape[0]
            n_sf = X_safe.shape[0]

            if n_ov > 0 and n_sf > 0:
                ov_frac = n_ov / (n_ov + n_sf)
                n_need_ov = max(1, int(n_need * ov_frac * 1.5))
                n_need_sf = max(0, n_need - n_need_ov)
            elif n_ov > 0:
                n_need_ov = n_need
                n_need_sf = 0
            elif n_sf > 0:
                n_need_ov = 0
                n_need_sf = n_need
            else:
                result_X.append(X_min)
                result_y.append(y_min)
                continue

            # Safe region: SMOTE-like interpolation (Algorithm 1 line 15)
            # Standard SMOTE requires multi-class input, so we use equivalent
            # manual interpolation (k-NN + random gap) with uniform weights.
            if n_need_sf > 0 and X_safe.shape[0] >= 2:
                k = min(self.smote_k, X_safe.shape[0] - 1)
                target_safe = X_safe.shape[0] + n_need_sf
                uniform_w = np.ones(X_safe.shape[0])
                Xs, ys = weighted_oversample(X_safe, y_safe, uniform_w,
                                              target_safe, k=k, rng=rng)
                if Xs.shape[0] > X_safe.shape[0]:
                    os_X.append(Xs[X_safe.shape[0]:])
                    os_y.append(ys[X_safe.shape[0]:])

            # Overlap region: weighted oversampling (lines 12-13)
            if n_need_ov > 0 and X_over.shape[0] >= 1:
                ov_global_idx = min_global_idx[min_overlap]
                # Compute contributions (Eq. 14)
                class_center = X[y == c].mean(axis=0)
                contributions = np.array([
                    (x - class_center) @ S_B @ (x - class_center)
                    for x in X_over
                ])
                # Ensure non-negative
                contributions = np.maximum(contributions, 1e-10)
                # Weights (Eq. 15)
                w = contributions / (contributions.sum() + 1e-10)

                target_ov = X_over.shape[0] + n_need_ov
                Xo, yo = weighted_oversample(X_over, y_over, w, target_ov,
                                              k=self.smote_k, rng=rng)
                os_X.append(Xo[X_over.shape[0]:])
                os_y.append(yo[X_over.shape[0]:])

            X_os = np.vstack(os_X)
            y_os = np.concatenate(os_y)
            result_X.append(X_os)
            result_y.append(y_os)

        return np.vstack(result_X), np.concatenate(result_y)

    def fit(self, X, y):
        """Train DEAHS (Algorithm 1).

        Args:
            X: feature array (n_samples, n_features)
            y: label array (n_samples,) - string or int labels
        """
        # Encode labels (runner already encoded when preencoded=True)
        if self.preencoded:
            y_enc = np.asarray(y).astype(int)
            self.classes_ = np.unique(y_enc)
            self.n_classes_ = int(self.classes_.max()) + 1 if len(self.classes_) else 0
        else:
            y_enc = self.label_encoder.fit_transform(y)
            self.n_classes_ = len(self.label_encoder.classes_)
            self.classes_ = self.label_encoder.classes_

        # Identify majority class (most samples)
        counts = Counter(y_enc)
        self.majority_class_ = max(counts, key=counts.get)

        # Scale features (runner already scaled when preencoded=True)
        X_sc = np.asarray(X, dtype=float) if self.preencoded else self.scaler.fit_transform(X)

        # Split: train:val = 6:2 (test is handled externally by caller, total 6:2:2)
        # Input X is ~80% of original data (after caller's test split).
        # val_adj = 0.2/0.8 = 0.25, so train:val = 75:25 of input = 60:20 of original
        val_adj = self.val_ratio / (1 - self.test_ratio)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_sc, y_enc, test_size=val_adj,
                stratify=y_enc, random_state=self.random_state)
        except ValueError as e:
            if "least populated class" in str(e) or "too few" in str(e):
                import warnings
                warnings.warn(
                    f"DEAHS: stratified split failed ({e}). "
                    f"Falling back to unstratified split."
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_sc, y_enc, test_size=val_adj,
                    random_state=self.random_state)
            else:
                raise

        # Identify majority/minority in training set
        maj_mask = y_train == self.majority_class_
        X_maj = X_train[maj_mask]
        y_maj = y_train[maj_mask]
        X_min_all = X_train[~maj_mask]
        y_min_all = y_train[~maj_mask]

        n_maj = X_maj.shape[0]
        n_min = X_min_all.shape[0]
        n_min_per_class = {c: (y_min_all == c).sum() for c in range(self.n_classes_) if c != self.majority_class_}
        min_target = max(n_min_per_class.values()) if n_min_per_class else n_min

        rng = np.random.RandomState(self.random_state)

        # Initialize SAC (Eq. 5 state dim = 2*b + (C-1))
        state_dim = 2 * self.n_bins + (self.n_classes_ - 1)
        sac = SACAgent(state_dim=state_dim, action_dim=2,
                       hidden_dim=self.sac_hidden, lr=self.sac_lr, device=self.device)

        # Step 3: First base classifier with random balanced subset
        n_sample = min(n_maj, n_min)
        maj_idx = rng.choice(n_maj, size=n_sample, replace=False)
        X_init = np.vstack([X_maj[maj_idx], X_min_all])
        y_init = np.concatenate([y_maj[maj_idx], y_min_all])
        shuf = rng.permutation(len(y_init))
        X_init, y_init = X_init[shuf], y_init[shuf]

        clf1 = RandomForestClassifier(n_estimators=self.rf_n_est, random_state=self.random_state, n_jobs=self.n_jobs)
        clf1.fit(X_init, y_init)
        self.base_classifiers = [clf1]

        # Boosting loop (Algorithm 1 lines 4-20)
        for t in range(1, self.n_base):
            # Line 6: State
            state = self._get_state(X_train, y_train, X_val, y_val, maj_mask, self.n_classes_)

            # Metrics before
            metrics_before = self._compute_metrics(X_val, y_val, self.n_classes_)

            # Line 7: Action from SAC
            action = sac.select_action(state)
            mu_t, sigma_t = action[0], max(action[1], 0.01)

            # Compute losses on majority samples for weighting
            p_maj = self._ensemble_predict_proba(X_maj)
            y_moh = np.zeros((len(y_maj), self.n_classes_))
            y_moh[np.arange(len(y_maj)), y_maj.astype(int)] = 1.0
            losses = -np.sum(y_moh * np.log(p_maj + 1e-10), axis=1)

            # Line 8-9: Outer undersampling
            X_maj_s, y_maj_s = self._outer_undersample(
                X_maj, y_maj, losses, mu_t, sigma_t, n_sample, rng)

            X_sub = np.vstack([X_maj_s, X_min_all])
            y_sub = np.concatenate([y_maj_s, y_min_all])

            # Lines 10-17: Inner oversampling
            try:
                X_bal, y_bal = self._inner_oversample(X_sub, y_sub, self.n_classes_, rng)
            except Exception:
                X_bal, y_bal = X_sub, y_sub

            # Line 19: Train new base classifier
            shuf = rng.permutation(len(y_bal))
            X_bal, y_bal = X_bal[shuf], y_bal[shuf]

            new_clf = RandomForestClassifier(
                n_estimators=self.rf_n_est,
                random_state=self.random_state + t, n_jobs=self.n_jobs)
            new_clf.fit(X_bal, y_bal)
            self.base_classifiers.append(new_clf)

            # Reward computation
            metrics_after = self._compute_metrics(X_val, y_val, self.n_classes_)
            weights_r = np.array([0.25, 0.25, 0.25, 0.25])
            reward = float(np.sum(weights_r * (np.array(metrics_after) - np.array(metrics_before))))

            next_state = self._get_state(X_train, y_train, X_val, y_val, maj_mask, self.n_classes_)
            done = 1.0 if t == self.n_base - 1 else 0.0
            sac.store_transition(state, action, reward, next_state, done)
            for _ in range(self.sac_steps):
                sac.update(batch_size=self.sac_batch)

        return self

    def predict(self, X):
        """Predict labels (returns original labels; ints when preencoded)."""
        X_sc = np.asarray(X, dtype=float) if self.preencoded else self.scaler.transform(X)
        y_enc = self._ensemble_predict(X_sc)
        if self.preencoded:
            return y_enc
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X):
        """Predict probabilities."""
        X_sc = np.asarray(X, dtype=float) if self.preencoded else self.scaler.transform(X)
        return self._ensemble_predict_proba(X_sc)


# ==============================================================================
# 4. Factory + smoke driver  (unified contract for run_all.py)
# ==============================================================================

MODEL_KEY = "DEAHS"


def build(random_state=42, smoke=False, n_base_classifiers=30, n_histogram_bins=10,
          alpha_gaussian=0.95, sac_hidden_dim=64, sac_lr=3e-4, sac_batch_size=32,
          sac_update_steps=5, smote_k_neighbors=5, fuzzification_param=2,
          val_ratio=0.2, test_ratio=0.2, rf_n_estimators=30, n_jobs=-1, **kw):
    """Unified factory. SAC-based undersampling + safe-region weighted oversampling
    + 30-RF ensemble (paper Algorithm 1). smoke reduces the RL budget."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if smoke:
        n_base_classifiers = min(n_base_classifiers, 5)
        sac_update_steps = min(sac_update_steps, 1)
        sac_batch_size = min(sac_batch_size, 16)
    return DEAHS(n_base_classifiers=n_base_classifiers, n_histogram_bins=n_histogram_bins,
                 alpha_gaussian=alpha_gaussian, sac_hidden_dim=sac_hidden_dim,
                 sac_lr=sac_lr, sac_batch_size=sac_batch_size, sac_update_steps=sac_update_steps,
                 smote_k_neighbors=smote_k_neighbors, fuzzification_param=fuzzification_param,
                 val_ratio=val_ratio, test_ratio=test_ratio,
                 rf_n_estimators=rf_n_estimators, n_jobs=n_jobs,
                 device=dev, random_state=random_state, preencoded=True)


if __name__ == '__main__':
    _smoke.run_smoke(build, MODEL_KEY)
