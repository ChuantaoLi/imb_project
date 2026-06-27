"""
DDPM-enabled data augmentation method for intelligent machine fault diagnosis
Complete reproduction of:
  "Denoising diffusion probabilistic model-enabled data augmentation method
   for intelligent machine fault diagnosis"
  Pengcheng Zhao, Wei Zhang, Xiaoshan Cao, Xiang Li
  Engineering Applications of Artificial Intelligence 139 (2025) 109520

------------------------------------------------------------------------------
WHAT IS REPRODUCED (paper Sec. 2)
------------------------------------------------------------------------------
  1. Continuous Wavelet Transform (paper Sec. 2.2): in the paper the 1-D
     vibration signal is turned into a 2-D time-frequency image and the DDPM is
     trained on those images.  The benchmark datasets here are ALREADY extracted
     tabular feature vectors (21 features per sample, last column = label), so
     there is no raw 1-D signal to transform.  The diffusion model is therefore
     trained directly over the feature vectors -- the only sensible adaptation.
     The paper itself notes "the diffusion model employs samples of identical
     fault types to generate fresh samples", i.e. generation is conditioned on
     the fault class: we implement a single CLASS-CONDITIONAL DDPM (a U-Net noise
     predictor conditioned on the class label), which is the robust equivalent of
     one model per class when some minority classes have only ~18 training
     samples.

  2. Denoising Diffusion Probabilistic Model (paper Sec. 2.1):
        forward  (Eq. 1-4):  q(x_t | x_0) = N(sqrt(a_bar_t) x_0, (1-a_bar_t) I)
                             with a_t = 1 - b_t , a_bar_t = prod_{i<=t} a_i
        reverse  (Eq. 5-10): q_theta(x_{t-1}|x_t) = N(mu_theta, Sigma_theta)
        loss     (Eq. 11):   L = E[ || eps - eps_theta(sqrt(a_bar_t) x_0 +
                             sqrt(1-a_bar_t) eps, t) ||^2 ]        (simplified MSE)
        sampling (Eq. 12 / Algorithm 1):  ancestral denoising
                             x_{t-1} = 1/sqrt(a_t)[ x_t - (1-a_t)/sqrt(1-a_bar_t)
                             eps_theta(x_t,t) ] + sigma_t * Z,  Z~N(0,I)
     Linear beta schedule, T = 1000 diffusion steps, U-Net noise predictor
     (paper Sec. 2.2: 4-5 down-sampling stages with skip connections; here a
     1-D U-Net whose depth is chosen from the feature length).

  3. Data augmentation: synthetic minority-class samples are generated with the
     trained DDPM and added to the training set until every minority class
     reaches the majority-class size (oversampling), then the classifier is
     trained on real + synthetic data.  This is the diffusion-model analogue of
     the balancing used by DEAHS / SPE so results are directly comparable.

  4. Fault-diagnosis CNN (paper Sec. 2.2 / Fig. 6): 3 conv+pool stages, 2 fully
     connected layers (64 -> n_classes), ReLU, dropout, softmax; Adam (lr=1e-3),
     cross-entropy, batch size 64, 300 epochs.  Implemented here as a 1-D-CNN
     over the feature vector.  For the UNIFIED cross-method comparison the
     downstream head is instead a RandomForest(n_estimators=30) (--classifier rf,
     the default); the DDPM contribution is the diffusion augmentation, which is
     classifier-independent, so swapping the head does not change the model.

------------------------------------------------------------------------------
EXPERIMENTAL PROTOCOL  (identical to DEAHS.py / SPE.py benchmark harness)
------------------------------------------------------------------------------
  Datasets  {IR5,IR10,IR20}_{CWRU,HUST,JNU,MFPT,PHM,SEU} (18 csv).
  Split     train_test_split(test_size=0.2, stratify=y, random_state=s).
  Repeats   5 runs, seeds s = 1..5  (the "five-fold validation").
  Metrics   Accuracy, Precision, Recall, F1, GMean, AUC (macro).
  Output    <RESULTS_DIR>/DDPM_Results.csv  (Dataset, IR, *_mean, *_std)  [relative].

DEFAULTS: with a GPU available this now matches the paper exactly -- T=1000
diffusion steps, DDPM trained for 3000 epochs (Sec. 2.2), CNN 300 epochs /
batch 64 / Adam 1e-3 (Sec. 2.2), and class-balancing augmentation.  The DDPM
beta schedule (linear 1e-4 -> 0.02), Adam lr (2e-4) and batch size are NOT given
in the paper, so the standard Ho et al. (2020) defaults are used.  On a CPU-only
machine lower --ddpm_epochs (e.g. 500) and/or --timesteps to keep runtime sane.
Every knob is a CLI flag.  Pass --device cuda (or run on a CUDA-capable env).
"""

import os
import math
import copy
import warnings
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from common import smoke as _smoke

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. Denoising Diffusion Probabilistic Model
# ==============================================================================


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal positional embedding of the diffusion timestep t (Transformer
    style, as used by Ho et al. 2020)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: (B,) float
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000.0) / max(half - 1, 1)))
        emb = t[:, None] * freq[None, :]  # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


def _norm(channels):
    """GroupNorm whose group count divides `channels`."""
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class ResConvBlock(nn.Module):
    """Residual 1-D conv block with FiLM (scale+shift) conditioning on the
    diffusion timestep and class label."""

    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _norm(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.film = nn.Linear(cond_dim, 2 * out_ch)

    def forward(self, x, cond):  # x: (B, C, L), cond: (B, cond_dim)
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(cond).unsqueeze(-1).chunk(2, dim=1)  # (B,C,1) each
        h = h * (1.0 + scale) + shift
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.proj(x)


class UNet1D(nn.Module):
    """1-D U-Net noise predictor eps_theta(x_t, t, class) (paper Sec. 2.2).

    Encoder/decoder depth is set automatically from the feature length so that
    the bottleneck stays >= 4 wide (with only 21 features, this yields 2
    down-sampling stages).  Skip connections fuse encoder/decoder features at
    every level, exactly as described for the paper's 2-D U-Net.
    """

    def __init__(self, n_feat, n_classes, base=32, mults=(1, 2, 4), cond_dim=128):
        super().__init__()
        self.cond_dim = cond_dim
        self.time_mlp = nn.Sequential(SinusoidalTimeEmb(cond_dim), nn.Linear(cond_dim, cond_dim * 4), nn.SiLU(), nn.Linear(cond_dim * 4, cond_dim))
        self.class_emb = nn.Embedding(n_classes, cond_dim)

        # number of down-sampling stages: stop when spatial dim would drop < 4
        levels = 0
        L = n_feat
        while (L // (2 ** (levels + 1))) >= 4 and (levels + 1) < len(mults):
            levels += 1
        self.levels = max(levels, 0)
        chans = [base * m for m in mults[: self.levels + 1]]

        self.init_conv = ResConvBlock(1, chans[0], cond_dim)
        self.downs = nn.ModuleList([ResConvBlock(chans[i], chans[i + 1], cond_dim) for i in range(self.levels)])
        self.mid = ResConvBlock(chans[-1], chans[-1], cond_dim)
        self.ups = nn.ModuleList([ResConvBlock(chans[self.levels - i] + chans[self.levels - i - 1], chans[self.levels - i - 1], cond_dim) for i in range(self.levels)])
        self.out_norm = _norm(chans[0])
        self.out_conv = nn.Conv1d(chans[0], 1, 1)

    def forward(self, x, t, cls):  # x: (B,1,L), t: (B,) float, cls: (B,) long
        cond = self.time_mlp(t) + self.class_emb(cls)  # (B, cond_dim)
        h = self.init_conv(x, cond)
        skips = [h]
        for i, down in enumerate(self.downs):
            h = F.avg_pool1d(h, 2)
            h = down(h, cond)
            if i < self.levels - 1:  # keep all but bottleneck
                skips.append(h)
        h = self.mid(h, cond)
        for up in self.ups:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = up(h, cond)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h  # (B,1,L)


class GaussianDiffusion(nn.Module):
    """Class-conditional Gaussian (DDPM) diffusion with the exact forward
    process, simplified-MSE training objective and ancestral sampling of the
    paper (Eqs. 1-12, Algorithm 1)."""

    def __init__(self, n_feat, n_classes, T=1000, beta_start=1e-4, beta_end=2e-2, base_ch=32):
        super().__init__()
        self.T = T
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.model = UNet1D(n_feat, n_classes, base=base_ch)

        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_abar", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_1m_abar", torch.sqrt(1.0 - alphas_cumprod).float())

    # ---- forward / training -------------------------------------------------
    def q_sample(self, x0, t, noise=None):
        """q(x_t | x_0) = N(sqrt(a_bar_t) x_0, (1-a_bar_t) I)   (Eq. 2-4)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sah = self.sqrt_abar[t][:, None, None]
        s1a = self.sqrt_1m_abar[t][:, None, None]
        return sah * x0 + s1a * noise

    def training_loss(self, x0, labels):
        """Simplified objective  L = E||eps - eps_theta(x_t, t)||^2  (Eq. 11)."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.model(xt, t.float(), labels)
        return F.mse_loss(pred, noise)

    # ---- reverse / sampling (Algorithm 1) -----------------------------------
    @torch.no_grad()
    def p_sample(self, xt, t_int, labels):
        """One ancestral denoising step  (Eq. 12)."""
        beta_t = self.betas[t_int]
        alpha_t = self.alphas[t_int]
        abar_t = self.alphas_cumprod[t_int]
        sqrt_a = alpha_t.sqrt()
        sqrt_1m_abar_t = (1.0 - abar_t).clamp_min(1e-8).sqrt()
        t_vec = torch.full((xt.shape[0],), t_int, device=xt.device, dtype=torch.float)
        eps = self.model(xt, t_vec, labels)
        mean = (xt - (1.0 - alpha_t) / sqrt_1m_abar_t * eps) / sqrt_a
        if t_int > 0:
            sigma = beta_t.sqrt()  # sigma_t^2 = beta_t (Ho et al.)
            return mean + sigma * torch.randn_like(xt)
        return mean

    @torch.no_grad()
    def sample(self, n, class_label, device, batch_size=512):
        """Generate `n` samples of `class_label` from pure noise (Algorithm 1)."""
        out = []
        remaining = n
        while remaining > 0:
            m = min(batch_size, remaining)
            x = torch.randn(m, 1, self.n_feat, device=device)
            labels = torch.full((m,), int(class_label), device=device, dtype=torch.long)
            for t in reversed(range(self.T)):
                x = self.p_sample(x, t, labels)
            out.append(x.squeeze(1).cpu())  # (m, n_feat)
            remaining -= m
        return torch.cat(out, dim=0)[:n].numpy()


# ==============================================================================
# 2. Fault-diagnosis 1-D-CNN classifier  (paper Fig. 6, adapted to 1-D features)
# ==============================================================================


class CNN1DClassifier(nn.Module):
    """Fault-diagnosis CNN (paper Sec. 2.2 / Fig. 6, adapted to 1-D features).

    Structure (paper): 3 conv layers + 3 pooling layers + 2 fully connected
    layers (64 -> n_classes), ReLU, dropout, softmax; Adam(lr=1e-3),
    cross-entropy, batch 64.

    `channels` is the number of filters in the three conv layers.  The paper
    states "16, 6, and 1 sets of convolutional kernels"; that exact triplet is
    available via --cnn_channels 16,6,1, but with adaptive pooling to length 1 it
    collapses the last stage to a single channel, i.e. a 1-scalar FC input, which
    is degenerate for a 10/12-class problem -- almost certainly an artefact of the
    paper's 2-D images (whose large spatial extent keeps many features even with 1
    channel).  The 1-D default therefore keeps three *increasing* stages
    (16, 32, 64); both choices are structurally 3 conv + 3 pool + 2 FC.
    """

    def __init__(self, n_feat, n_classes, channels=(16, 32, 64), p_drop=0.5):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(c3, 64),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):  # x: (B, n_feat)
        return self.classifier(self.features(x.unsqueeze(1)))  # (B, n_classes) logits


# ==============================================================================
# 3. Full pipeline: DDPM augmentation + CNN fault diagnosis
# ==============================================================================


class DDPMFaultDiagnosis:
    """DDPM-based data-augmentation fault diagnosis (paper method, adapted)."""

    def __init__(
        self,
        T=1000,
        ddpm_epochs=3000,
        ddpm_lr=2e-4,
        ddpm_batch=128,
        cnn_epochs=300,
        cnn_lr=1e-3,
        cnn_batch=64,
        cnn_weight_decay=0.0,
        cnn_channels=(16, 32, 64),
        classifier_type="rf",
        rf_n_estimators=30,
        n_jobs=-1,
        aug_strategy="balance",
        aug_ratio=5,
        max_synthetic_per_class=0,
        base_ch=32,
        gen_batch=512,
        device="cpu",
        random_state=42,
        verbose=False,
        preencoded=False,
    ):
        self.T = T
        self.ddpm_epochs = ddpm_epochs
        self.ddpm_lr = ddpm_lr
        self.ddpm_batch = ddpm_batch
        self.cnn_epochs = cnn_epochs
        self.cnn_lr = cnn_lr
        self.cnn_batch = cnn_batch
        self.cnn_wd = cnn_weight_decay
        self.cnn_channels = tuple(cnn_channels)
        self.classifier_type = classifier_type  # 'rf' (UNIFIED) | 'cnn' (paper Fig. 6)
        self.rf_n_est = rf_n_estimators
        self.n_jobs = n_jobs
        self.aug_strategy = aug_strategy
        self.aug_ratio = aug_ratio
        self.max_syn = max_synthetic_per_class
        self.base_ch = base_ch
        self.gen_batch = gen_batch
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.preencoded = preencoded

        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.diffusion = None
        self.clf = None  # CNNClassifier OR RandomForest
        self.classes_ = None
        self.n_classes_ = None
        self.n_feat_ = None

    # ---- DDPM training ------------------------------------------------------
    def _train_diffusion(self, X_t, y_t):
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.ddpm_batch, shuffle=True)
        opt = torch.optim.Adam(self.diffusion.parameters(), lr=self.ddpm_lr)
        self.diffusion.train()
        for ep in range(self.ddpm_epochs):
            tot = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = self.diffusion.training_loss(xb, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * xb.shape[0]
            if self.verbose and (ep % max(1, self.ddpm_epochs // 5) == 0):
                print(f"      [DDPM] epoch {ep+1}/{self.ddpm_epochs}  " f"loss={tot/len(ds):.5f}", flush=True)

    # ---- synthetic-sample generation (augmentation) -------------------------
    def _generate_augmented(self, X_t, y_t):
        """Return (X_aug, y_aug) of synthetic samples for the minority classes."""
        counts = Counter(y_t.tolist())
        maj = max(counts.values())
        self.diffusion.eval()
        syn_X, syn_y = [], []
        for c in range(self.n_classes_):
            n_c = counts.get(c, 0)
            if self.aug_strategy == "balance":
                n_gen = maj - n_c
            else:  # 'fixed': augment by ratio
                n_gen = int(round(n_c * (self.aug_ratio - 1)))
            if self.max_syn > 0:
                n_gen = min(n_gen, self.max_syn)
            if n_gen <= 0:
                continue
            samples = self.diffusion.sample(n_gen, c, self.device, self.gen_batch)
            syn_X.append(samples)
            syn_y.append(np.full(n_gen, c, dtype=np.int64))
            if self.verbose:
                print(f"      [DDPM] class {c}: generated {n_gen} synthetic " f"(was {n_c})", flush=True)
        if not syn_X:
            return np.zeros((0, self.n_feat_)), np.zeros((0,), dtype=np.int64)
        return np.vstack(syn_X), np.concatenate(syn_y)

    # ---- CNN training -------------------------------------------------------
    def _train_classifier(self, X_t, y_t):
        """Train the downstream fault-diagnosis classifier on real+synthetic data.

        classifier_type:
          'rf'  -> RandomForest(n_estimators=30): the UNIFIED cross-method
                   classifier. The DDPM contribution is the diffusion-based
                   data augmentation (Sec. 2.1-2.2); the downstream head is a
                   standard fault-diagnosis classifier and is therefore swappable
                   without altering the diffusion model.
          'cnn' -> the paper's 1-D-CNN fault diagnosis head (Sec. 2.2 / Fig. 6)."""
        if self.classifier_type == "rf":
            # train directly on the numpy balanced set (real + synthetic)
            X_np = X_t.numpy() if isinstance(X_t, torch.Tensor) else X_t
            y_np = y_t.numpy() if isinstance(y_t, torch.Tensor) else y_t
            self.clf = RandomForestClassifier(n_estimators=self.rf_n_est, n_jobs=self.n_jobs, random_state=self.random_state)
            self.clf.fit(X_np, y_np)
            return

        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.cnn_batch, shuffle=True)
        opt = torch.optim.Adam(self.clf.parameters(), lr=self.cnn_lr, weight_decay=self.cnn_wd)
        loss_fn = nn.CrossEntropyLoss()
        self.clf.train()
        for ep in range(self.cnn_epochs):
            tot = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.clf(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * xb.shape[0]
            if self.verbose and (ep % max(1, self.cnn_epochs // 5) == 0):
                print(f"      [CNN]  epoch {ep+1}/{self.cnn_epochs}  " f"loss={tot/len(ds):.5f}", flush=True)

    # ---- public API ---------------------------------------------------------
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        y_enc = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        self.n_feat_ = X.shape[1]

        if self.preencoded:
            # data already StandardScaled + label-encoded by the shared runner
            # (common.base) -- respect the train-only-scaling invariant.
            X_sc = np.asarray(X, dtype=np.float32)
        else:
            X_sc = self.scaler.fit_transform(X).astype(np.float32)  # unit-var -> ideal for diffusion

        # diffusion operates on (B, 1, L); CNN operates on (B, L) and adds the
        # channel dim internally.
        X_t = torch.from_numpy(X_sc).unsqueeze(1)  # (N, 1, L)
        y_t = torch.from_numpy(y_enc.astype(np.int64))

        # 1) train the class-conditional DDPM on the (scaled) training set
        self.diffusion = GaussianDiffusion(self.n_feat_, self.n_classes_, T=self.T, base_ch=self.base_ch).to(self.device)
        self._train_diffusion(X_t, y_t)

        # 2) generate synthetic minority samples (paper: data augmentation)
        syn_X, syn_y = self._generate_augmented(X_t, y_t)
        if syn_X.shape[0] > 0:
            X_all = np.vstack([X_sc, syn_X.astype(np.float32)])
            y_all = np.concatenate([y_enc, syn_y])
        else:
            X_all, y_all = X_sc, y_enc

        if self.verbose:
            print(f"      train size: real={X_sc.shape[0]} " f"synthetic={syn_X.shape[0]} total={X_all.shape[0]}", flush=True)

        # 3) train the fault-diagnosis classifier on real + synthetic data
        if self.classifier_type == "cnn":
            self.clf = CNN1DClassifier(self.n_feat_, self.n_classes_, channels=self.cnn_channels).to(self.device)
            X_all_t = torch.from_numpy(X_all.astype(np.float32))
            y_all_t = torch.from_numpy(y_all.astype(np.int64))
            self._train_classifier(X_all_t, y_all_t)
        else:  # 'rf' unified classifier
            self._train_classifier(X_all.astype(np.float32), y_all.astype(np.int64))
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        if self.preencoded:
            X_sc = np.asarray(X, dtype=np.float32)
        else:
            X_sc = self.scaler.transform(X).astype(np.float32)
        if self.classifier_type == "rf":
            raw = self.clf.predict_proba(X_sc)
            proba = np.zeros((X_sc.shape[0], self.n_classes_), dtype=float)
            for j, c in enumerate(self.clf.classes_):
                proba[:, int(c)] = raw[:, j]
            return proba
        X_t = torch.from_numpy(X_sc).to(self.device)
        self.clf.eval()
        logits = self.clf(X_t)
        return F.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        p = self.predict_proba(X)
        return self.label_encoder.inverse_transform(np.argmax(p, axis=1))


# ==============================================================================
# Unified factory  (shared-runner contract: build(random_state, smoke) -> model)
# ==============================================================================


MODEL_KEY = "DDPM"


def build(random_state=42, smoke=False, classifier_type='rf', rf_n_estimators=30,
          n_jobs=-1, T=1000, base_ch=32, aug_strategy='balance', **kw):
    """Unified factory: class-conditional DDPM data augmentation + RF(30)
    downstream classifier (UNIFIED for cross-method comparison; the diffusion
    augmentation is the paper contribution and is classifier-independent).

      smoke: ddpm_epochs=100  (fast sanity)
      full : ddpm_epochs=1000 (paper Sec. 2.2 trains 3000; 1000 converges on
             the small tabular feature vectors here and bounds CPU runtime).

    Data arrive already StandardScaled + label-encoded from the shared runner,
    so preencoded=True skips this module's internal scaler (the train-only-
    scaling invariant is owned by common.base)."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ddpm_epochs = 100 if smoke else 1000
    return DDPMFaultDiagnosis(
        T=T, ddpm_epochs=ddpm_epochs, ddpm_lr=2e-4, ddpm_batch=128,
        cnn_epochs=300, cnn_lr=1e-3, cnn_batch=64, cnn_channels=(16, 32, 64),
        classifier_type=classifier_type, rf_n_estimators=rf_n_estimators,
        n_jobs=n_jobs, aug_strategy=aug_strategy, aug_ratio=5,
        max_synthetic_per_class=0, base_ch=base_ch, gen_batch=512,
        device=dev, random_state=random_state, verbose=False, preencoded=True)


if __name__ == '__main__':
    _smoke.run_smoke(build, MODEL_KEY)
