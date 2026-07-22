"""
DBCF: Deep Balanced Cascade Forest for imbalanced fault diagnosis
Complete reproduction of:
  "Deep balanced cascade forest: A novel fault diagnosis method for data
   imbalance"
  Hao Chen, Chaoshun Li, Wenxian Yang, Jie Liu, Xueli An, Yujie Zhao
  ISA Transactions 126 (2022) 428-439

The model fuses a DATA-level method and an ALGORITHM-level method:

  (A) Up-down Sampling (Sec. 3.1)  -- data level
      Reception rate (Eq. 1):
          r_i = sum_{j in same-class-among-N-nn} d(x_i, x_j)
                / sum_{k in N-nn}          d(x_i, x_k)
      A high r_i means the same-class neighbours dominate the N-neighbourhood
      => the sample sits in the CORE of its class; a low r_i means other-class
      neighbours intrude => the sample sits on the BOUNDARY.

      Up-sampling (Eq. 2): synthesise a minority sample between two same-class
      minority parents, weighted toward the higher-reception (core) parent:
          x' = (x_i * r_i + x_j * r_j) / (r_i + r_j)

      Down-sampling: discard each sample with probability proportional to the
      reciprocal of importance (= 1/r_i), i.e. boundary samples are dropped and
      core samples retained ("makes the boundary clear").

  (B) Balanced Forest (Sec. 3.2) -- algorithm level, the basic classifier
      Each tree splits with a BALANCE INFORMATION ENTROPY impurity (Eq. 3):
          ratio_k   = x_k / X_k                  (x_k in node, X_k globally)
          p_k       = ratio_k / sum_j ratio_j
          Bent(D)   = -sum_k p_k * log2(p_k)
      Unlike ordinary entropy (p_k = x_k/|D|), Bent measures each class by the
      FRACTION OF ITS GLOBAL POPULATION that reached the node, so a minority
      class contributes on equal footing with the majority.  The split criterion
      is the information gain under Bent (Eq. 4, binary splits V=2):
          Gain(D,a) = Bent(D) - sum_v (|D_v|/|D|) Bent(D_v)
      (The paper labels Eq. 4 "Info-Gain Ratio" but writes plain gain; we follow
      the written formula.)  Bent cannot be injected into sklearn's compiled
      tree, so the tree is implemented from scratch.

  (C) Multi-channel cascade forest (Sec. 3.3) -- the deep structure
      Each of the C channels (= 2) is fed an INDEPENDENT balanced dataset produced
      by one run of Up-down Sampling (bagging-style diversity).  Inside a channel
      a gcForest-style cascade is grown: every layer holds 4 Balanced Forests; per
      gcForest each forest is trained with k-fold cross-validation so the class-
      distribution vectors fed to the next layer are out-of-fold (leak-free), and
      the 4 forests' C-dim class vectors are CONCATENATED (4C extra dims) with the
      ORIGINAL features and passed to the next layer.  The cascade grows on a
      "growing set" until accuracy on a held-out "estimating set" stops improving
      (adaptive depth, max 5).  The last layer averages all 4 forests for the
      channel output, and the 2 channels are averaged for the prediction.

==============================================================================
EXPERIMENTAL PROTOCOL (kept identical to DEAHS.py / SPE.py for comparability):
  - Datasets: D:/.../Bearing_Datasets/{ImbalanceRatio_5,10,20}/*.csv
              (18 sets, last column = label, rest = features).
  - Split: train_test_split(test_size=0.2, stratify=y), 5 repeats seeds 1..5
           (the user's "five-fold verification" = 5 repeated random 80/20 splits).
  - Preprocessing: StandardScaler fit on train (helps Up-down's Euclidean step).
  - Metrics: Accuracy, Precision, Recall, F1 (macro), GMean, AUC (ovr macro).
  - Output: D:/.../DBCF_Results.csv  (Dataset, IR, {metric}_mean, {metric}_std).

Paper-faithful parameter choices used here:
  * Tree count per forest: default n_trees=200, matching the paper's main setting
    (Case I).  When reproducing the paper's second case, pass n_trees=100
    explicitly.

    NOTE on the unified "classifier = RandomForest(30)" protocol: unlike SPE /
    DeepSMOTE / DDPM, DBCF does NOT swap its head for a standard RandomForest.
    DBCF's classifier IS the Balanced Cascade Forest -- the Bent balance-
    information-entropy impurity (Eq. 3-4) and the multi-channel cascade are the
    paper's core algorithmic contribution and its namesake.  Replacing the
    Balanced Forest with sklearn RandomForest would delete Eq. 3-4 and gut the
    model.
  * Split thresholds: default None = evaluate every midpoint between sorted
    unique values (faithful CART).  --n_thresholds N caps to N quantiles (speed).
  * Up-down Sampling's neighbourhood size N (Eq.1, default 7) and the down-sampling
    discard-probability normalisation are not pinned down numerically by the paper;
    see the function docstrings for the choices made.
  * "Info-Gain Ratio" (Sec. 3.2): the paper labels Eq. 4 as gain ratio but the
    written Eq. 4 is plain Bent information gain; we follow the written formula.
  * Applicability: the paper studies the single-majority setting.  The code now
    enforces that assumption instead of silently generalising it to multi-majority
    data.
==============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Use joblib's native Parallel + delayed (NOT sklearn's wrappers).
# _fit_balanced_tree only uses numpy/scipy — it does not need sklearn's
# thread-local config propagation, so joblib is the right choice and avoids
# the sklearn "delayed should be used with Parallel" advisory entirely.
from joblib import Parallel, delayed


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
# 1. Up-down Sampling  (Sec. 3.1)
# ==============================================================================


def reception_rate(X, y, n_neighbors=7, min_dist=1e-12):
    """Reception rate r_i of every sample (Eq. 1).

    For sample x_i, take its N nearest neighbours (excluding itself).  Let
    n_i of them share x_i's class.  Then
        r_i = (sum of distances to those n_i same-class neighbours)
              / (sum of distances to all N neighbours).
    High r_i  -> same-class neighbours dominate the neighbourhood -> CORE sample.
    Low  r_i  -> other-class neighbours intrude                  -> BOUNDARY sample.

    Implemented with a k-d / ball tree (sklearn NearestNeighbors) so it scales
    to the ~30k-sample balanced sets (a full distance matrix would not).
    Returns r in (0, 1]; r=0 when no same-class neighbour is among the N nearest.
    """
    n = X.shape[0]
    k = min(n_neighbors + 1, n)  # +1 because the query point is its own NN
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X)
    dists, idx = nn.kneighbors(X)  # both (n, k); column 0 is self (dist 0)
    dists = dists[:, 1:]  # drop self
    idx = idx[:, 1:]
    y_nb = y[idx]  # (n, N) neighbour labels
    same = y_nb == y[:, None]  # (n, N) same-class mask
    sum_same = np.where(same, dists, 0.0).sum(axis=1)
    sum_all = dists.sum(axis=1) + min_dist
    r = sum_same / sum_all
    return r


def _weighted_pair_interpolate(X_cls, r_cls, n_need, rng):
    """Generate n_need synthetic minority samples (Eq. 2).

    x' = (x_i * r_i + x_j * r_j) / (r_i + r_j), with parents drawn at random
    among the class members.  Weights lean toward the higher-reception (core)
    parent, so synthesised points stay toward the class interior.
    """
    m = X_cls.shape[0]
    if m < 2 or n_need <= 0:
        return np.empty((0, X_cls.shape[1]))
    i = rng.randint(0, m, size=n_need)
    j = rng.randint(0, m, size=n_need)
    # avoid trivially identical parents where possible
    same = i == j
    if same.any():
        j[same] = (j[same] + 1 + rng.randint(0, m - 1, size=same.sum())) % m
    ri = r_cls[i] + 1e-12
    rj = r_cls[j] + 1e-12
    w_i = (ri / (ri + rj))[:, None]
    return X_cls[i] * w_i + X_cls[j] * (1.0 - w_i)


def up_down_sample(X, y, n_neighbors=7, rng=None, down_sample=True, strict_single_majority=True):
    """One realisation of Up-down Sampling -> a balanced channel dataset.

    Up-sampling: each class with fewer than the majority count is synthesised up
    to the majority count using reception-rate-weighted interpolation (Eq. 2).
    Down-sampling: reception rates are recomputed on the post-upsampling set and
    each sample is discarded with probability proportional to 1/r_i (reciprocal
    of importance, normalised to [0,1]).  The paper does not define an
    additional class-count floor, so this implementation keeps only a minimal
    safety rescue to avoid collapsing a class entirely.

    Stochasticity (random parents, random discards) gives channel-to-channel
    diversity when DBCF calls this once per channel.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    classes, counts = np.unique(y, return_counts=True)
    maj_count = counts.max()
    majority_classes = classes[counts == maj_count]
    if strict_single_majority and len(majority_classes) != 1:
        raise ValueError("DBCF assumes a single majority class; received " f"{len(majority_classes)} classes tied for the maximum count.")
    majority_class = majority_classes[0]
    # --- Up-sampling -------------------------------------------------------
    r = reception_rate(X, y, n_neighbors=n_neighbors)
    parts_X, parts_y = [X], [y]
    for c in classes:
        mask = y == c
        n_c = int(mask.sum())
        if c == majority_class:
            continue
        n_need = maj_count - n_c
        synth = _weighted_pair_interpolate(X[mask], r[mask], n_need, rng)
        parts_X.append(synth)
        parts_y.append(np.full(synth.shape[0], c, dtype=y.dtype))

    X_bal = np.vstack(parts_X)
    y_bal = np.concatenate(parts_y)

    if not down_sample or len(y_bal) <= len(classes):
        return X_bal, y_bal

    # --- Down-sampling -----------------------------------------------------
    r_bal = reception_rate(X_bal, y_bal, n_neighbors=n_neighbors)
    inv = 1.0 / (r_bal + 1e-12)  # reciprocal of importance
    p_discard = inv / inv.max()  # normalise to [0,1]
    roll = rng.rand(len(y_bal))
    discard = roll < p_discard

    keep_mask = ~discard
    for c in classes:
        cls_idx = np.where(y_bal == c)[0]
        kept_idx = cls_idx[keep_mask[cls_idx]]
        if len(kept_idx) > 0:
            continue

        # Safety only: if stochastic rejection removes a class completely,
        # rescue its most representative sample.
        rescue = cls_idx[np.argmin(p_discard[cls_idx])]
        keep_mask[rescue] = True

    return X_bal[keep_mask], y_bal[keep_mask]


# ==============================================================================
# 2. Balanced Forest  (Sec. 3.2): Bent-impurity tree + bagging
# ==============================================================================


def _bent_from_counts(class_counts, X_global):
    """Balance information entropy (Eq. 3) from a per-node class-count vector."""
    nz = class_counts > 0
    if not nz.any():
        return 0.0
    ratio = class_counts[nz] / X_global[nz]
    s = ratio.sum()
    if s <= 0:
        return 0.0
    p = ratio / s
    return float(-np.sum(p * np.log2(p)))


class BalancedTree:
    """A binary decision tree whose splits maximise Bent-based information gain
    (Eq. 3-4).  X_global_counts is the per-class total of the CHANNEL training
    set seen by the current balanced forest and defines the balance correction
    everywhere."""

    def __init__(self, X_global_counts, max_depth=12, min_samples_split=4, min_samples_leaf=2, max_features=None, n_thresholds=None, max_depth_re_tip=None, rng=None):
        self.Xg = X_global_counts.astype(np.float64)
        self.C = len(self.Xg)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.rng = rng if rng is not None else np.random.RandomState(0)
        # node storage (filled during fit, then frozen into arrays)
        self._n_feat = None
        self._feat = []
        self._thr = []
        self._left = []
        self._right = []
        self._leaf = []
        self._counts = []
        # frozen arrays used at predict time
        self.feat = self.thr = self.left = self.right = None
        self.is_leaf = None
        self.leaf_counts = None

    # ---- training ---------------------------------------------------------
    def fit(self, X, y):
        self._n_feat = X.shape[1]
        if self.max_features is None:
            self.mtry = max(1, int(np.sqrt(self._n_feat)))
        else:
            self.mtry = min(self._n_feat, max(1, self.max_features))
        self._build(X, y, 0)
        self._finalize()
        return self

    def _new_node(self):
        nid = len(self._feat)
        self._feat.append(0)
        self._thr.append(0.0)
        self._left.append(-1)
        self._right.append(-1)
        self._leaf.append(False)
        self._counts.append(None)
        return nid

    def _make_leaf(self, y):
        nid = self._new_node()
        self._leaf[nid] = True
        self._counts[nid] = np.bincount(y, minlength=self.C).astype(np.float64)
        return nid

    def _build(self, X, y, depth):
        n = len(y)
        yc = np.bincount(y, minlength=self.C)
        if depth >= self.max_depth or n < self.min_samples_split or np.count_nonzero(yc) <= 1:
            return self._make_leaf(y)
        split = self._best_split(X, y, yc)
        if split is None:
            return self._make_leaf(y)
        f, thr, lm, rm = split
        nid = self._new_node()
        self._feat[nid] = f
        self._thr[nid] = thr
        self._left[nid] = self._build(X[lm], y[lm], depth + 1)
        self._right[nid] = self._build(X[rm], y[rm], depth + 1)
        return nid

    def _best_split(self, X, y, parent_counts):
        """Find the binary split maximising Bent information gain (Eq. 4).

        Evaluates EVERY midpoint between consecutive sorted values of each
        candidate feature (faithful CART) cheaply: from Eq. 3,
            Bent(D) = log2(S) - T/S,  S = sum_k x_k/X_k,  T = sum_k (x_k/X_k) log2(x_k/X_k),
        so Bent for ALL split positions follows from a single cumulative class
        count over the sorted order (O(n*C), one cumsum per feature).
        """
        n = len(y)
        parent_bent = _bent_from_counts(parent_counts, self.Xg)
        feat_pool = self.rng.choice(self._n_feat, size=self.mtry, replace=False)
        Xg = self.Xg
        onehot = np.zeros((n, self.C), dtype=np.float64)  # reused across features
        onehot[np.arange(n), y] = 1.0
        msl = self.min_samples_leaf

        best_gain, best = -np.inf, None
        for f in feat_pool:
            xv = X[:, f]
            order = np.argsort(xv, kind="quicksort")
            xs = xv[order]
            # valid split positions i (0-indexed): value must change (xs[i]<xs[i+1])
            # and both children must respect min_samples_leaf.
            change = xs[:-1] < xs[1:]  # length n-1
            i_range = np.arange(n - 1)
            valid = change & (i_range >= msl - 1) & (i_range <= n - msl - 1)
            if not valid.any():
                continue

            cum = np.cumsum(onehot[order], axis=0)  # (n, C) running class counts
            total = cum[-1]
            # Bent over ALL positions via the closed form log2(S) - T/S
            rl = cum / Xg  # (n, C)  ratio per class
            with np.errstate(divide="ignore", invalid="ignore"):
                rl_log = np.where(rl > 0, rl * np.log2(rl), 0.0)
            Sl = rl.sum(axis=1)
            Tl = rl_log.sum(axis=1)
            Sl_s = np.where(Sl > 0, Sl, 1.0)
            Bl = np.log2(Sl_s) - Tl / Sl_s  # Bent(left),  (n,)

            rr = (total - cum) / Xg  # right child ratios
            with np.errstate(divide="ignore", invalid="ignore"):
                rr_log = np.where(rr > 0, rr * np.log2(rr), 0.0)
            Sr = rr.sum(axis=1)
            Tr = rr_log.sum(axis=1)
            Sr_s = np.where(Sr > 0, Sr, 1.0)
            Br = np.log2(Sr_s) - Tr / Sr_s  # Bent(right), (n,)

            nl = np.arange(1, n + 1)  # |left| at each position
            gains = parent_bent - (nl / n) * Bl - ((n - nl) / n) * Br
            gains_pos = gains[:-1][valid]
            thr_pos = (0.5 * (xs[:-1] + xs[1:]))[valid]
            # optional cap (speed only); None keeps all faithful midpoints
            if self.n_thresholds is not None and len(gains_pos) > self.n_thresholds:
                sel = np.random.RandomState(self.rng.randint(1 << 30)).choice(len(gains_pos), size=self.n_thresholds, replace=False)
                gains_pos = gains_pos[sel]
                thr_pos = thr_pos[sel]

            j = int(np.argmax(gains_pos))
            if gains_pos[j] > best_gain:
                best_gain = gains_pos[j]
                thr = float(thr_pos[j])
                lm = np.where(xv <= thr)[0]
                rm = np.where(xv > thr)[0]
                best = (int(f), thr, lm, rm)
        return best

    def _finalize(self):
        self.feat = np.array(self._feat, dtype=np.int64)
        self.thr = np.array(self._thr, dtype=np.float64)
        self.left = np.array(self._left, dtype=np.int64)
        self.right = np.array(self._right, dtype=np.int64)
        self.is_leaf = np.array(self._leaf, dtype=bool)
        # internal nodes stored None -> place zero vectors so the array is 2-D
        leaf_counts = np.zeros((len(self._counts), self.C), dtype=np.float64)
        for i, c in enumerate(self._counts):
            if c is not None:
                leaf_counts[i] = c
        self.leaf_counts = leaf_counts

    # ---- prediction (fully vectorised routing) ----------------------------
    def predict_proba(self, X):
        n = X.shape[0]
        node = np.zeros(n, dtype=np.int64)
        while True:
            internal = ~self.is_leaf[node]
            if not internal.any():
                break
            idx = np.where(internal)[0]
            cur = node[idx]
            col = self.feat[cur]
            val = X[idx, col]
            node[idx] = np.where(val <= self.thr[cur], self.left[cur], self.right[cur])
        counts = self.leaf_counts[node]
        s = counts.sum(axis=1, keepdims=True)
        proba = np.where(s > 0, counts / np.where(s > 0, s, 1.0), 1.0 / self.C)
        return proba


def _fit_balanced_tree(X, y, Xg, max_depth, max_features, n_thresholds, seed):
    """Module-level worker: fit one bootstrapped BalancedTree (picklable for joblib)."""
    rng = np.random.RandomState(seed)
    boot = rng.randint(0, len(y), size=len(y))
    tree = BalancedTree(Xg, max_depth=max_depth, max_features=max_features, n_thresholds=n_thresholds, rng=np.random.RandomState(rng.randint(1 << 30)))
    tree.fit(X[boot], y[boot])
    return tree


class BalancedForest:
    """Bagged ensemble of BalancedTrees (= one Balanced Forest)."""

    def __init__(self, X_global_counts, n_trees=30, max_depth=12, max_features=None, n_thresholds=None, n_jobs=1, rng=None):
        self.Xg = X_global_counts
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.n_jobs = n_jobs
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.trees = []

    def fit(self, X, y):
        seeds = [self.rng.randint(1 << 30) for _ in range(self.n_trees)]
        args = (X, y, self.Xg, self.max_depth, self.max_features, self.n_thresholds)
        if self.n_jobs and self.n_jobs != 1:
            # trees are independent -> embarrassingly parallel; the loky pool is
            # reused across the many forests of a run, so only the first call pays
            # the worker startup cost.
            self.trees = Parallel(n_jobs=self.n_jobs, backend="loky")(delayed(_fit_balanced_tree)(*args, s) for s in seeds)
        else:
            self.trees = [_fit_balanced_tree(*args, s) for s in seeds]
        return self

    def predict_proba(self, X):
        p = np.zeros((X.shape[0], self.Xg.shape[0]))
        for tree in self.trees:
            p += tree.predict_proba(X)
        p /= len(self.trees)
        return p


# ==============================================================================
# 3. Cascade forest channel + DBCF wrapper  (Sec. 3.3)
# ==============================================================================


class CascadeForestChannel:
    """One gcForest-style cascade of Balanced Forests with adaptive depth.

    Each layer holds ``n_forests`` (= 4, the paper's "4 BFs per layer") Balanced
    Forests.  Following gcForest (Zhou & Feng 2019, on which DBCF is built), each
    forest is trained with k-fold cross-validation so that the class-distribution
    vectors fed to the next layer are OUT-OF-FOLD (leak-free): fold i is predicted
    by the forest trained on the other k-1 folds, and the k sub-forests are
    retained and averaged at test time.  The n_forests C-dim class vectors are
    CONCATENATED (-> n_forests*C = 4C extra dims) with the ORIGINAL features and
    passed to the next layer (Sec. 2: "class distributions of 4 random forests are
    concatenated with the original data to be the input to the next layer").  The
    cascade grows on a "growing set" until accuracy on a held-out "estimating set"
    stops improving (adaptive depth, max_layers).  The last layer averages all
    n_forests for the channel output (Sec. 3.3: "The last level BFs' outputs ...
    are averaged to obtain the class distribution").
    """

    def __init__(
        self, n_classes, X_global_counts, n_forests=4, k_fold=3, n_trees=30, max_layers=5, max_depth=12, max_features=None, n_thresholds=None, grow_ratio=0.75, n_jobs=1, rng=None
    ):
        self.C = n_classes
        self.Xg = X_global_counts
        self.n_forests = n_forests
        self.k_fold = k_fold
        self.n_trees = n_trees
        self.max_layers = max_layers
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.grow_ratio = grow_ratio
        self.n_jobs = n_jobs
        self.rng = rng if rng is not None else np.random.RandomState(0)
        # layers[l] = list of n_forests slots; each slot = list of k_fold sub-forests
        self.layers = []

    def _new_forest(self, rng):
        return BalancedForest(self.Xg, n_trees=self.n_trees, max_depth=self.max_depth, max_features=self.max_features, n_thresholds=self.n_thresholds, n_jobs=self.n_jobs, rng=rng)

    def _slot_proba(self, sub_forests, X):
        """A forest's prediction = average of its k_fold sub-forests."""
        p = np.zeros((X.shape[0], self.C))
        for f in sub_forests:
            p += f.predict_proba(X)
        return p / len(sub_forests)

    def fit(self, X, y):
        # --- stratified growing / estimating split for adaptive-depth early
        #     stopping.  Only attempt stratification when every class has ≥2
        #     samples; otherwise fall back to a plain (non-stratified) split or
        #     use the full set for both.
        min_cls_full = np.min(np.bincount(y)) if len(y) else 0
        can_stratify = min_cls_full >= 2
        if self.grow_ratio < 1.0 and len(np.unique(y)) > 1 and can_stratify:
            Xg, Xe, yg, ye = train_test_split(
                X, y, train_size=self.grow_ratio, stratify=y,
                random_state=self.rng.randint(1 << 30))
        elif self.grow_ratio < 1.0 and len(np.unique(y)) > 1:
            # Not enough samples per class for stratification — plain split
            Xg, Xe, yg, ye = train_test_split(
                X, y, train_size=self.grow_ratio, stratify=None,
                random_state=self.rng.randint(1 << 30))
        else:
            Xg, Xe, yg, ye = X, X, y, y

        # --- k stratified folds over the growing set; cap k by the smallest
        #     class.  When a class has < 2 samples stratified CV is impossible,
        #     so we fall back to training each forest on ALL growing data (no
        #     OOF leak-free guarantee, which is acceptable for such tiny sets).
        min_cls = np.min(np.bincount(yg)) if len(yg) else 1
        use_cv = min_cls >= 2
        if use_cv:
            k = max(2, min(self.k_fold, min_cls))
            skf = StratifiedKFold(n_splits=k, shuffle=True,
                                  random_state=self.rng.randint(1 << 30))
            folds = [te for _, te in skf.split(Xg, yg)]

        best_acc, best_depth = -1.0, 0
        layers = []
        grow_feat = Xg.copy()  # input to the next layer
        for L in range(self.max_layers):
            layer_slots, slot_oof = [], []
            for _ in range(self.n_forests):
                sub_forests = []
                oof = np.zeros((len(yg), self.C))
                if use_cv:
                    for fi in range(k):
                        val = folds[fi]
                        tr = np.concatenate([folds[j] for j in range(k) if j != fi])
                        forest = self._new_forest(
                            np.random.RandomState(self.rng.randint(1 << 30)))
                        forest.fit(grow_feat[tr], yg[tr])
                        sub_forests.append(forest)
                        oof[val] = forest.predict_proba(grow_feat[val])
                else:
                    # Tiny dataset: train one forest on all growing data
                    forest = self._new_forest(
                        np.random.RandomState(self.rng.randint(1 << 30)))
                    forest.fit(grow_feat, yg)
                    sub_forests.append(forest)
                    oof = forest.predict_proba(grow_feat)
                layer_slots.append(sub_forests)
                slot_oof.append(oof)
            layers.append(layer_slots)
            v_grow = np.hstack(slot_oof)  # (ng, n_forests*C) = 4C
            grow_feat = np.hstack([Xg, v_grow])  # next-layer input

            acc = accuracy_score(ye, np.argmax(
                self._predict_proba(Xe, layers), axis=1))
            if acc > best_acc + 1e-6:
                best_acc, best_depth = acc, L + 1
            else:
                break  # accuracy stopped improving
        self.layers = layers[:best_depth] or layers[:1]
        return self

    def _predict_proba(self, X, layers):
        """Forward X through the cascade; return the last layer's averaged proba."""
        Xcur = X
        last = None
        for layer in layers:
            slot_probas = [self._slot_proba(sfs, Xcur) for sfs in layer]
            v = np.hstack(slot_probas)  # n_forests*C
            Xcur = np.hstack([X, v])  # original + this layer's vectors
            last = slot_probas
        return np.mean(last, axis=0)  # average over n_forests -> (n, C)

    def predict_proba(self, X):
        return self._predict_proba(X, self.layers)


class DBCF:
    """Deep Balanced Cascade Forest (Chen et al., ISA Transactions 2022)."""

    def __init__(
        self,
        n_channels=2,
        n_forests=4,
        k_fold=3,
        n_trees=200,
        max_layers=5,
        max_depth=12,
        max_features=None,
        n_thresholds=None,
        n_neighbors=7,
        grow_ratio=0.75,
        down_sample=True,
        max_samples_per_channel=None,
        n_jobs=1,
        random_state=42,
        preencoded=False,
        standardize=True,
        strict_single_majority=True,
    ):
        self.preencoded = preencoded
        self.standardize = standardize
        self.strict_single_majority = strict_single_majority
        self.n_channels = n_channels
        self.n_forests = n_forests
        self.k_fold = k_fold
        self.n_trees = n_trees
        self.max_layers = max_layers
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.n_neighbors = n_neighbors
        self.grow_ratio = grow_ratio
        self.down_sample = down_sample
        self.max_samples_per_channel = max_samples_per_channel
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.label_encoder_ = LabelEncoder()
        self.scaler = StandardScaler()
        self.channels_ = []
        self.classes_ = None
        self.n_classes_ = None
        self.original_counts_ = None

    def _maybe_cap(self, X, y, rng):
        """Optional stratified cap on a channel's balanced set (speed knob)."""
        if not self.max_samples_per_channel:
            return X, y
        cap = self.max_samples_per_channel
        if len(y) <= cap:
            return X, y
        keep = []
        for c in np.unique(y):
            ic = np.where(y == c)[0]
            per = max(1, int(round(cap * len(ic) / len(y))))
            per = min(per, len(ic))
            keep.append(rng.choice(ic, size=per, replace=False))
        keep = np.concatenate(keep)
        return X[keep], y[keep]

    def fit(self, X, y):
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        X_arr = np.asarray(X, dtype=float)
        X_sc = self.scaler.fit_transform(X_arr) if self.standardize else X_arr

        self.original_counts_ = np.bincount(y_enc, minlength=self.n_classes_).astype(np.float64)
        rng = np.random.RandomState(self.random_state)

        self.channels_ = []
        for ch in range(self.n_channels):
            ch_rng = np.random.RandomState(rng.randint(1 << 30))
            X_bal, y_bal = up_down_sample(X_sc, y_enc, n_neighbors=self.n_neighbors, rng=ch_rng, down_sample=self.down_sample, strict_single_majority=self.strict_single_majority)
            X_bal, y_bal = self._maybe_cap(X_bal, y_bal, ch_rng)
            channel_counts = np.bincount(y_bal, minlength=self.n_classes_).astype(np.float64)
            channel = CascadeForestChannel(
                n_classes=self.n_classes_,
                X_global_counts=channel_counts,
                n_forests=self.n_forests,
                k_fold=self.k_fold,
                n_trees=self.n_trees,
                max_layers=self.max_layers,
                max_depth=self.max_depth,
                max_features=self.max_features,
                n_thresholds=self.n_thresholds,
                grow_ratio=self.grow_ratio,
                n_jobs=self.n_jobs,
                rng=np.random.RandomState(ch_rng.randint(1 << 30)),
            )
            channel.fit(X_bal, y_bal)
            self.channels_.append(channel)
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        X_sc = self.scaler.transform(X_arr) if self.standardize else X_arr
        p = np.zeros((X_sc.shape[0], self.n_classes_))
        for ch in self.channels_:
            p += ch.predict_proba(X_sc)
        p /= len(self.channels_)
        p = np.clip(p, 1e-12, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        return p

    def predict(self, X):
        y_enc = np.argmax(self.predict_proba(X), axis=1)
        return self.label_encoder_.inverse_transform(y_enc)


# ==============================================================================
# 4. Factory + smoke driver  (unified contract for run_all.py)
# ==============================================================================

MODEL_KEY = "DBCF"


def build(random_state=42, smoke=False, n_channels=2, n_forests=4, k_fold=3, n_trees=200, max_depth=12, n_neighbors=7, down_sample=True, n_jobs=-1, **kw):
    """Unified factory. DBCF keeps its Balanced Cascade Forest (Bent-impurity
    trees) -- its classifier IS the paper's model, NOT replaced by RF. The
    paper uses 200 trees / BF in its main setting and 100 in Case II; pass
    n_trees explicitly to reproduce the latter. smoke caps depth and tree count
    for quick validation only."""
    max_layers = 2 if smoke else 5
    if smoke:
        n_trees = min(n_trees, 10)
    return DBCF(
        n_channels=n_channels,
        n_forests=n_forests,
        k_fold=k_fold,
        n_trees=n_trees,
        max_layers=max_layers,
        max_depth=max_depth,
        n_neighbors=n_neighbors,
        down_sample=down_sample,
        n_jobs=n_jobs,
        random_state=random_state,
        preencoded=True,
        standardize=True,
        strict_single_majority=True,
    )


if __name__ == "__main__":
    _smoke.run_smoke(build, MODEL_KEY)
