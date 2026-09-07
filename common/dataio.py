"""common.dataio — unified dataset / fold IO.

Every one of the 30 models sees ONLY ``DatasetDescriptor`` + ``Fold`` objects:
it never touches the filesystem and never branches on KEEL-vs-Bearing.

    KEEL    : folds are PRE-SPLIT on disk (fold{i}_{train,test}.csv); read directly.
    Bearing : a single constructed IR CSV is re-split with a deterministic
              StratifiedKFold(5, shuffle=True, random_state=42) so ALL models
              see identical folds (critical for fair comparison).

Bearing IR{5,10,20,30} CSVs are built once by ``build_bearing_ir_datasets``:
keep the majority (largest) class full-size, subsample every minority class to
floor(N_majority / IR) with a floor of 1.
"""
import os
import json
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import paths

BEARING_NAMES = ("CWRU", "Gearbox", "HUST", "JNU", "MFPT", "Ottawa", "SEU", "XJTU")
DEFAULT_IRS = (5, 10, 20, 30)
N_FOLDS = 5


@dataclass(frozen=True)
class DatasetDescriptor:
    name: str                                  # display name: ecoli1 | IR20_CWRU | NSL-KDD | 10Ydata
    source: str                                # 'keel' | 'bearing' | 'software_defect' | 'heart'
    keel_name: Optional[str] = None            # e.g. 'ecoli1'
    bearing_name: Optional[str] = None         # e.g. 'CWRU'
    ir: Optional[int] = None                   # e.g. 20
    dataset_name: Optional[str] = None         # csv stem for software_defect/heart, e.g. 'AR', 'Framingham'


@dataclass
class Fold:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    fold_id: int


# --------------------------------------------------------------------------- I/O

def _read_keel_csv(path):
    """KEEL fold CSV: header present, last column = Class (int or str)."""
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values
    return X, y


def _read_bearing_csv(path):
    """Bearing CSV: feature columns are all 'feat_*'; label column is 'label'."""
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if str(c).startswith("feat_")]
    X = df[feat_cols].values.astype(float)
    y = df["label"].values
    return X, y


def _read_generic_csv(path):
    """Generic CSV: all columns except the last are features; last column = label."""
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values
    return X, y


def load_folds(desc, random_state=42):
    """Return (list of N_FOLDS Fold objects, sorted-unique classes array)."""
    if desc.source == "keel":
        d = os.path.join(paths.KEEL_5FOLD, desc.keel_name)
        folds, all_y = [], []
        for i in range(1, N_FOLDS + 1):
            Xtr, ytr = _read_keel_csv(os.path.join(d, f"fold{i}_train.csv"))
            Xte, yte = _read_keel_csv(os.path.join(d, f"fold{i}_test.csv"))
            folds.append(Fold(Xtr, ytr, Xte, yte, i))
            all_y.append(ytr); all_y.append(yte)
        classes = np.array(sorted(set(np.concatenate(all_y).tolist())))
        return folds, classes

    if desc.source == "keel_complete":
        # Whole dataset from Dataset/KEEL/complete/<name>.csv → ONE deterministic
        # stratified holdout split (no cross-validation). All models see the
        # exact same split: test_size=0.3, random_state=42.
        path = os.path.join(paths.KEEL_COMPLETE, f"{desc.keel_name}.csv")
        X, y = _read_keel_csv(path)
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42)
        classes = np.array(sorted(set(y.tolist())))
        return [Fold(Xtr, ytr, Xte, yte, 1)], classes

    if desc.source == "bearing":
        path = os.path.join(paths.BEARING_IR, f"IR{desc.ir}", f"{desc.bearing_name}.csv")
        X, y = _read_bearing_csv(path)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
        folds = []
        for i, (tr, te) in enumerate(skf.split(X, y), start=1):
            folds.append(Fold(X[tr], y[tr], X[te], y[te], i))
        classes = np.array(sorted(set(y.tolist())))
        return folds, classes

    if desc.source in ("software_defect", "heart"):
        root = paths.SOFTWARE if desc.source == "software_defect" else paths.HEART_RAW
        path = os.path.join(root, f"{desc.dataset_name}.csv")
        X, y = _read_generic_csv(path)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
        folds = []
        for i, (tr, te) in enumerate(skf.split(X, y), start=1):
            folds.append(Fold(X[tr], y[tr], X[te], y[te], i))
        classes = np.array(sorted(set(y.tolist())))
        return folds, classes

    raise ValueError(f"unknown source {desc.source}")


# --------------------------------------------------------------- Bearing IR build

def build_bearing_ir_datasets(irs=DEFAULT_IRS, random_state=42, out_root=None,
                              overwrite=False, names=BEARING_NAMES):
    """Construct the imbalanced Bearing CSVs.

    For each Bearing dataset and each IR: keep the majority class (largest)
    full-size; subsample every minority class to floor(N_majority / IR) with a
    floor of 1. Writes Dataset/Bearing_IR/IR<ir>/<name>.csv (feat_*,label) and a
    _manifest.json. Idempotent (skips existing files unless overwrite=True)."""
    out_root = out_root or paths.BEARING_IR
    manifest = {}
    for name in names:
        raw = os.path.join(paths.BEARING_RAW, f"{name}.csv")
        if not os.path.isfile(raw):
            print(f"[build_bearing_ir] SKIP missing {raw}")
            continue
        X, y = _read_bearing_csv(raw)
        feat_cols = [c for c in pd.read_csv(raw, nrows=0).columns
                     if str(c).startswith("feat_")]
        counts = Counter(y.tolist())
        maj_label, n_maj = counts.most_common(1)[0]
        manifest[name] = {"majority": str(maj_label), "n_majority": int(n_maj), "irs": {}}
        for ir in irs:
            out_dir = os.path.join(out_root, f"IR{ir}")
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, f"{name}.csv")
            if os.path.isfile(out_csv) and not overwrite:
                manifest[name]["irs"][int(ir)] = {"path": out_csv, "skipped": True}
                continue
            target = max(1, int(n_maj // ir))
            rng = np.random.RandomState(random_state + hash(name) % 10000 + ir * 97)
            keep_idx = []
            per_class = {}
            for c in sorted(counts):
                idx = np.where(y == c)[0]
                if c == maj_label:
                    keep_idx.append(idx)
                    per_class[str(c)] = int(len(idx))
                else:
                    k = min(target, len(idx))
                    sub = rng.choice(idx, size=k, replace=False)
                    keep_idx.append(sub)
                    per_class[str(c)] = int(k)
            keep_idx = np.concatenate(keep_idx)
            df_out = pd.DataFrame(X[keep_idx], columns=feat_cols)
            df_out["label"] = y[keep_idx]
            df_out.to_csv(out_csv, index=False)
            achieved = max(per_class.values()) / max(1, min(
                v for k, v in per_class.items() if k != str(maj_label)) or [1])
            manifest[name]["irs"][int(ir)] = {
                "path": out_csv,
                "majority_kept_full": True,
                "minority_target": int(target),
                "per_class_size": per_class,
                "achieved_IR_maj_min": round(float(achieved), 3),
                "n_total": int(len(keep_idx)),
            }
            print(f"[build_bearing_ir] {name} IR{ir}: maj={n_maj} full, "
                  f"minorities->{target}, total={len(keep_idx)}")
    man_path = os.path.join(out_root, "_manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build_bearing_ir] manifest -> {man_path}")
    return manifest


# --------------------------------------------------------------------- discovery

def discover_all_datasets(keel_filter=None, bearing_irs=DEFAULT_IRS,
                          bearing_names=None, sdp_names=None, heart_names=None,
                          keel_root=None, bearing_ir_root=None):
    """Walk KEEL_5FOLD|KEEL_COMPLETE + Bearing_IR/IR<ir>/* + Software/* + Heart/* → descriptors.

    The KEEL root may use either layout:
      * 5fold:     <name>/fold{i}_{train,test}.csv      → source="keel"
      * complete:  <name>.csv (whole dataset)           → source="keel_complete"
    """
    keel_root = keel_root or paths.KEEL_5FOLD
    bearing_ir_root = bearing_ir_root or paths.BEARING_IR
    out = []

    if os.path.isdir(keel_root):
        for name in sorted(os.listdir(keel_root)):
            d = os.path.join(keel_root, name)
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "fold1_train.csv")):
                if keel_filter is not None and name not in keel_filter:
                    continue
                out.append(DatasetDescriptor(name=name, source="keel", keel_name=name))
            elif os.path.isfile(d) and name.endswith(".csv"):
                stem = name[:-4]
                if keel_filter is not None and stem not in keel_filter:
                    continue
                out.append(DatasetDescriptor(name=stem, source="keel_complete",
                                             keel_name=stem))

    names = set(bearing_names) if bearing_names is not None else None
    for ir in bearing_irs:
        d = os.path.join(bearing_ir_root, f"IR{ir}")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".csv"):
                continue
            bname = f.replace(".csv", "")
            if names is not None and bname not in names:
                continue
            out.append(DatasetDescriptor(
                name=f"IR{ir}_{bname}", source="bearing",
                bearing_name=bname, ir=int(ir)))

    # Software Defect: raw CSVs, label last column, no IR construction
    sdp_filter = set(sdp_names) if sdp_names is not None else None
    if os.path.isdir(paths.SOFTWARE):
        for f in sorted(os.listdir(paths.SOFTWARE)):
            if not f.endswith(".csv"):
                continue
            dname = f.replace(".csv", "")
            if sdp_filter is not None and dname not in sdp_filter:
                continue
            out.append(DatasetDescriptor(
                name=dname, source="software_defect", dataset_name=dname))

    # Heart: raw CSVs, label last column, no IR construction
    heart_filter = set(heart_names) if heart_names is not None else None
    if os.path.isdir(paths.HEART_RAW):
        for f in sorted(os.listdir(paths.HEART_RAW)):
            if not f.endswith(".csv"):
                continue
            dname = f.replace(".csv", "")
            if heart_filter is not None and dname not in heart_filter:
                continue
            out.append(DatasetDescriptor(
                name=dname, source="heart", dataset_name=dname))

    return out
