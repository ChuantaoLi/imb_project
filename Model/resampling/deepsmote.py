"""
DeepSMOTE: Fusing Deep Learning and SMOTE for Imbalanced Data
Complete reproduction of:
  "DeepSMOTE: Fusing Deep Learning and SMOTE for Imbalanced Data"
  Damien Dablain, Bartosz Krawczyk, Nitesh V. Chawla
  IEEE Transactions on Neural Networks and Learning Systems, 2023

Algorithm 1 (DeepSMOTE):
  Train the Encoder/Decoder:
    for epoch:
      for batch b in B:
        E_b = encode(b)
        D_b = decode(E_b)
        RL  = MSE(D_b, b)                         # reconstruction loss
        c   = randomly sample a class from C
        cb  = randomly sample |b| instances from c   # same-class batch
        E_s = encode(cb)
        P_e = permute order(E_s)                    # -> injects variance
        D_p = decode(P_e)
        PL  = MSE(D_p, cb)                          # penalty loss
        TL  = RL + PL
        theta -= alpha * d(TL)/d(theta)
  Generate Samples:
    for each minority class m:
      E_m = encode(X_m)
      G_m = SMOTE(E_m)                              # interpolate in latent space
      S_m = decode(G_m)                             # synthetic instances
    -> balanced training set, then train a classifier on it.

Two things make DeepSMOTE different from "SMOTE after a plain autoencoder":
  1. The PENALTY LOSS: a class is sampled, |b| of its instances are encoded and
     their order PERMUTED before decoding. The decoder must therefore reconstruct
     a *different* same-class instance for each slot -- this trains it to decode
     the in-between embeddings that SMOTE produces at inference time, which is
     exactly why DeepSMOTE needs no discriminator (unlike GAN/WAE oversampling).
  2. SMOTE is applied in the LEARNED LATENT space (not the raw feature space) and
     the interpolated embeddings are decoded back to the input domain.

Tabular adaptation: the paper's DCGAN convolutional encoder/decoder (Sec. IV.C:
encoder = 4 conv layers + BatchNorm + LeakyReLU -> dense latent; decoder =
mirrored conv-transpose + BatchNorm + ReLU, Tanh output) is designed for images
in [-1, 1]. The bearing benchmarks here are StandardScaler-normalised tabular
features, so we keep the SAME design choices (encoder: BatchNorm + LeakyReLU;
decoder: BatchNorm + ReLU) but use an MLP with a LINEAR output head (Tanh would
clip the standardised features). Latent SMOTE and the penalty-loss training are
implemented exactly as written in Algorithm 1.

The experimental protocol (18 bearing datasets under IR5/10/20, the 80/20
stratified split, 5 repeats with seeds 1..5, the 6 evaluation metrics, and the
output CSV schema) is identical to the DEAHS.py / SPE.py benchmark harness so
the methods are directly comparable. The downstream classifier is RandomForest
(n_estimators=30) by default -- the UNIFIED cross-method setting (DeepSMOTE is
classifier-independent per paper Sec. III, so swapping the head does not alter
the DeepSMOTE oversampler itself). Pass --classifier mlp for the paper-faithful
deep MLP head (analogous to the paper's ResNet-18).
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Relocatable paths: resolved from THIS file so the module is importable and
# callable from any working directory (no hardcoded absolute paths).
# ---------------------------------------------------------------------------
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # D:\imb_project
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from common import smoke as _smoke


# ==============================================================================
# 1. Encoder / Decoder  (Algorithm 1 "Train the Encoder / Decoder")
# ==============================================================================

class Encoder(nn.Module):
    """MLP encoder mirroring the DCGAN-encoder design choices
    (BatchNorm + LeakyReLU), funnelled to a latent embedding."""

    def __init__(self, n_features, hidden_dims, latent_dim, neg_slope=0.2):
        super().__init__()
        dims = [n_features] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.LeakyReLU(neg_slope),
            ]
        self.body = nn.Sequential(*layers)
        self.fc_out = nn.Linear(dims[-1], latent_dim)

    def forward(self, x):
        return self.fc_out(self.body(x))


class Decoder(nn.Module):
    """Mirrored MLP decoder (BatchNorm + ReLU). The output activation matches the
    input normalisation, following the paper's design (Sec. IV.C: "except for the
    final layer, which uses Tanh"):
      - out_activation='tanh'    -> Tanh, used when inputs are scaled to [-1, 1]
                                    (the paper's image-domain setting; --norm minmax).
      - out_activation='linear'  -> identity, used when inputs are StandardScaler'd
                                    (roughly mean 0, std 1, tails beyond +/-1; Tanh
                                    would clip them)."""

    def __init__(self, n_features, hidden_dims, latent_dim, out_activation='linear'):
        super().__init__()
        dims = [latent_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
            ]
        self.body = nn.Sequential(*layers)
        self.fc_out = nn.Linear(dims[-1], n_features)
        if out_activation == 'tanh':
            self.out_act = nn.Tanh()
        else:
            self.out_act = nn.Identity()

    def forward(self, z):
        return self.out_act(self.fc_out(self.body(z)))


class MLPClassifier(nn.Module):
    """Downstream classifier trained on the DeepSMOTE-balanced set
    (tabular analogue of the paper's ResNet-18 head)."""

    def __init__(self, n_features, n_classes, hidden_dims=(128, 64), dropout=0.3):
        super().__init__()
        dims = [n_features] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.body = nn.Sequential(*layers)
        self.fc_out = nn.Linear(dims[-1], n_classes)

    def forward(self, x):
        return self.fc_out(self.body(x))


# ==============================================================================
# 2. Helpers: derangement (penalty-loss permutation) + latent-space SMOTE
# ==============================================================================

def derangement(n, device):
    """A permutation of {0..n-1} with NO fixed points (perm[i] != i for all i).

    Algorithm 1 permutes the order of the encoded same-class batch; using a
    derangement guarantees every decoded slot is paired with a *different*
    instance, so the penalty loss is never trivially zero. For n <= 1 there is
    no valid derangement and we return a no-op index (the caller only hits this
    path for degenerate batches, which BatchNorm rules out in practice)."""
    if n <= 1:
        return torch.zeros(n, dtype=torch.long, device=device)
    perm = torch.randperm(n, device=device)
    fixed = perm == torch.arange(n, device=device)
    if fixed.any():
        perm = perm.clone()
        perm[fixed] = (perm[fixed] + 1) % n          # shift every fixed point
    return perm


def latent_smote(Z, n_target, k=5, rng=None):
    """Standard SMOTE interpolation carried out in the latent/embedding space.

    Z        : (n, d) embeddings of a single minority class.
    n_target : desired number of instances for this class (>= n).
    Returns  : (n_target, d) -- the original n embeddings plus (n_target-n)
               convex combinations of each point and one of its k nearest
               same-class neighbours (gap ~ U[0, 1]), exactly as in SMOTE.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    n = Z.shape[0]
    if n >= n_target:
        return Z.copy()

    kk = max(1, min(k, n - 1))                        # usable neighbour count
    # k-NN graph within this class (Euclidean distance in latent space)
    diff = Z[:, None, :] - Z[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    nn_idx = np.argsort(dist, axis=1)[:, :kk]

    n_synth = n_target - n
    synth = np.empty((n_synth, Z.shape[1]), dtype=Z.dtype)
    for s in range(n_synth):
        i = rng.randint(n)
        j = nn_idx[i, rng.randint(kk)]
        gap = rng.uniform(0.0, 1.0)
        synth[s] = Z[i] + gap * (Z[j] - Z[i])
    return np.vstack([Z, synth])


# ==============================================================================
# 3. DeepSMOTE Framework
# ==============================================================================

class DeepSMOTE:
    """DeepSMOTE oversampler + downstream classifier (Algorithm 1)."""

    def __init__(self,
                 # encoder/decoder
                 hidden_dims=(128, 64), latent_dim=16,
                 ae_epochs=200, ae_batch=128, ae_lr=2e-4,
                 ae_min_epochs=50, ae_patience=25,
                 norm='standard',
                 # generation
                 smote_k=5,
                 # downstream classifier
                 classifier='rf',
                 clf_hidden=(128, 64), clf_epochs=100, clf_lr=1e-3,
                 clf_batch=128, clf_dropout=0.3,
                 rf_n_estimators=30, n_jobs=-1,
                 # misc
                 device='cpu', random_state=42, preencoded=False):
        self.preencoded = preencoded
        self.hidden_dims = tuple(hidden_dims)
        self.latent_dim = latent_dim
        self.ae_epochs = ae_epochs              # max epoch cap (paper: up to 350)
        self.ae_batch = ae_batch
        self.ae_lr = ae_lr
        self.ae_min_epochs = ae_min_epochs      # paper floor ("50-350 epochs")
        self.ae_patience = ae_patience          # plateau patience for early stop
        self.norm = norm                        # 'standard' -> Linear out; 'minmax' -> Tanh out
        self.smote_k = smote_k
        self.classifier = classifier
        self.clf_hidden = tuple(clf_hidden)
        self.clf_epochs = clf_epochs
        self.clf_lr = clf_lr
        self.clf_batch = clf_batch
        self.clf_dropout = clf_dropout
        self.rf_n_est = rf_n_estimators
        self.n_jobs = n_jobs
        self.device = device
        self.random_state = random_state

        # Input scaling matches the decoder output activation, per the paper's
        # design (images in [-1,1] -> Tanh). 'standard' keeps the harness
        # convention; 'minmax' is the paper-faithful image-domain analog.
        from sklearn.preprocessing import MinMaxScaler
        if self.norm == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
            self.dec_out_act = 'tanh'
        else:
            self.scaler = StandardScaler()
            self.dec_out_act = 'linear'
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.rng = np.random.RandomState(random_state)

        self.encoder = None
        self.decoder = None
        self.net = None           # MLP classifier
        self.rf = None            # RandomForest classifier

    # ---- Algorithm 1: train the encoder/decoder with RL + PL ----------------

    def _train_autoencoder(self, X, y):
        n = X.shape[0]
        Xg = torch.from_numpy(X).float().to(self.device)

        self.encoder = Encoder(self.n_features_, self.hidden_dims,
                               self.latent_dim).to(self.device)
        self.decoder = Decoder(self.n_features_, list(reversed(self.hidden_dims)),
                               self.latent_dim,
                               out_activation=self.dec_out_act).to(self.device)
        opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.ae_lr)

        bs = max(2, min(self.ae_batch, n))             # BatchNorm needs >= 2
        classes = np.unique(y)
        class_idx = {int(c): np.where(y == c)[0] for c in classes}

        # Sec. V.A.8: "We train the models for 50-350 epochs, depending on when
        # the training loss plateaus." -> early-stop on the mean total loss once
        # it plateaus, after a minimum number of epochs (ae_min_epochs), with the
        # hard cap ae_epochs. patience=None disables early stopping (fixed runs).
        use_es = self.ae_patience is not None
        best_loss = np.inf
        bad = 0

        self.encoder.train()
        self.decoder.train()
        for ep in range(self.ae_epochs):
            order = np.random.permutation(n)
            ep_tl = 0.0
            n_batches = 0
            for start in range(0, n - bs + 1, bs):     # drop_last semantics
                bidx = order[start:start + bs]
                xb = Xg[bidx]

                # Reconstruction loss: RL = MSE(decode(encode(b)), b)
                rb = self.decoder(self.encoder(xb))
                RL = F.mse_loss(rb, xb)

                # Penalty loss: sample ONE class, |b| of its instances, encode,
                # PERMUTE the order, decode, and penalise against the originals.
                c = int(classes[np.random.randint(len(classes))])
                pool = class_idx[c]
                if len(pool) >= bs:
                    sel = np.random.choice(pool, size=bs, replace=False)
                else:
                    sel = np.random.choice(pool, size=bs, replace=True)
                cb = Xg[sel]
                zc = self.encoder(cb)
                perm = derangement(bs, self.device)
                dp = self.decoder(zc[perm])
                PL = F.mse_loss(dp, cb)

                TL = RL + PL
                opt.zero_grad()
                TL.backward()
                opt.step()

                ep_tl += TL.item()
                n_batches += 1

            if use_es:
                mean_tl = ep_tl / max(1, n_batches)
                if mean_tl < best_loss - 1e-6:
                    best_loss = mean_tl
                    bad = 0
                else:
                    bad += 1
                if ep + 1 >= self.ae_min_epochs and bad >= self.ae_patience:
                    break     # training loss has plateaued

    # ---- Algorithm 1: generate samples (encode -> SMOTE -> decode) ----------

    @torch.no_grad()
    def _balance(self, X, y):
        self.encoder.eval()
        self.decoder.eval()

        counts = Counter(y)
        target = max(counts.values())                 # balance every class up to the largest

        parts_X, parts_y = [], []
        for c in range(self.n_classes_):
            mask = y == c
            Xc = X[mask]
            nc = Xc.shape[0]
            parts_X.append(Xc)                        # keep the real instances
            parts_y.append(np.full(nc, c, dtype=int))

            if nc >= target:
                continue                              # already a largest class

            # E_m = encode(X_m); G_m = SMOTE(E_m); S_m = decode(G_m)
            Zc = self.encoder(torch.from_numpy(Xc).float().to(self.device)) \
                     .cpu().numpy()
            Zs = latent_smote(Zc, target, k=self.smote_k, rng=self.rng)
            syn_z = torch.from_numpy(Zs[nc:]).float().to(self.device)
            Sx = self.decoder(syn_z).cpu().numpy()
            parts_X.append(Sx)
            parts_y.append(np.full(Sx.shape[0], c, dtype=int))

        X_bal = np.vstack(parts_X)
        y_bal = np.concatenate(parts_y)
        # shuffle so classes are intermixed for training
        s = self.rng.permutation(len(y_bal))
        return X_bal[s], y_bal[s]

    # ---- downstream classifier ---------------------------------------------

    def _train_classifier(self, X, y):
        if self.classifier == 'rf':
            self.rf = RandomForestClassifier(
                n_estimators=self.rf_n_est,
                random_state=self.random_state, n_jobs=self.n_jobs)
            self.rf.fit(X, y)
            return

        n = X.shape[0]
        Xg = torch.from_numpy(X).float().to(self.device)
        yg = torch.from_numpy(y).long().to(self.device)
        self.net = MLPClassifier(self.n_features_, self.n_classes_,
                                 self.clf_hidden, self.clf_dropout).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.clf_lr)
        bs = max(2, min(self.clf_batch, n))
        self.net.train()
        for _ in range(self.clf_epochs):
            order = np.random.permutation(n)
            for start in range(0, n - bs + 1, bs):
                bidx = order[start:start + bs]
                logits = self.net(Xg[bidx])
                loss = F.cross_entropy(logits, yg[bidx])
                opt.zero_grad()
                loss.backward()
                opt.step()

    # ---- public API --------------------------------------------------------

    def fit(self, X, y):
        """Train DeepSMOTE (Algorithm 1) and the downstream classifier.

        Args:
            X: feature array (n_samples, n_features)
            y: label array (string or int labels)
        """
        if self.preencoded:
            y_enc = np.asarray(y).astype(int)
            self.classes_ = np.unique(y_enc)
            self.n_classes_ = int(self.classes_.max()) + 1 if len(self.classes_) else 0
        else:
            y_enc = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
            self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        X_sc = np.asarray(X, dtype=float) if self.preencoded else self.scaler.fit_transform(X)

        # Phase 1: encoder/decoder with reconstruction + penalty loss
        self._train_autoencoder(X_sc, y_enc)
        # Phase 2: oversample minorities in latent space, decode -> balanced set
        X_bal, y_bal = self._balance(X_sc, y_enc)
        # Phase 3: train the classifier on the balanced set
        self._train_classifier(X_bal, y_bal)
        return self

    def predict_proba(self, X):
        X_sc = np.asarray(X, dtype=float) if self.preencoded else self.scaler.transform(X)
        if self.classifier == 'rf':
            raw = self.rf.predict_proba(X_sc)
            proba = np.zeros((X_sc.shape[0], self.n_classes_), dtype=float)
            for j, c in enumerate(self.rf.classes_):
                proba[:, int(c)] = raw[:, j]
            return proba

        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(X_sc).float().to(self.device))
            return F.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return idx if self.preencoded else self.label_encoder.inverse_transform(idx)


# ==============================================================================
# 4. Factory + smoke driver  (unified contract for run_all.py)
# ==============================================================================

MODEL_KEY = "DeepSMOTE"


def build(random_state=42, smoke=False, classifier='rf', rf_n_estimators=30,
          n_jobs=-1, hidden_dims=(128, 64), latent_dim=16, smote_k=5, **kw):
    """Unified factory. Encoder/decoder (RL+PL) + latent SMOTE + RF(30) downstream.
    smoke: ae_epochs=50 ; full: ae_epochs=200 (paper up to 350)."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if smoke:
        ae_epochs, ae_min_epochs, ae_patience, clf_epochs = 50, 10, 10, 50
    else:
        ae_epochs, ae_min_epochs, ae_patience, clf_epochs = 200, 50, 25, 100
    return DeepSMOTE(hidden_dims=tuple(hidden_dims), latent_dim=latent_dim,
                     ae_epochs=ae_epochs, ae_batch=128, ae_lr=2e-4,
                     ae_min_epochs=ae_min_epochs, ae_patience=ae_patience, norm='standard',
                     smote_k=smote_k, classifier=classifier,
                     clf_hidden=(128, 64), clf_epochs=clf_epochs, clf_lr=1e-3,
                     clf_batch=128, clf_dropout=0.3, rf_n_estimators=rf_n_estimators,
                     n_jobs=n_jobs, device=dev, random_state=random_state, preencoded=True)


if __name__ == '__main__':
    _smoke.run_smoke(build, MODEL_KEY)
