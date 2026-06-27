"""common.metrics — the 10 imbalance metrics + confusion-matrix rendering.

Metric set (mean +/- std over 5 folds; Runtime tracked separately by the runner):
    Accuracy, Precision(macro), Recall(macro), F1(macro),
    GMean (geometric mean of per-class recall over test-present classes;
           small zero-correction so one missed class does not zero the score),
    AUC (ovr macro; binary path when n_classes==2),
    MCC (Matthews correlation coefficient),
    AUPRC (macro average-precision; binary path when n_classes==2),
    IBA  (Index of Balanced Accuracy, dominance T=0.5),
    Kappa (Cohen's kappa).

Confusion matrices are pooled (summed) across the 5 folds, row-normalized so
each true-class row is a recall distribution (fair under imbalance), rendered at
dpi=600 with Times New Roman and English axis labels.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless; safe on servers / no display
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix,
)

METRIC_KEYS = (
    "Accuracy", "Precision", "Recall", "F1", "GMean", "AUC",
    "MCC", "AUPRC", "IBA", "Kappa",
)
IBA_T = 0.5      # dominance parameter for IBA (paper default 0.5)
GMEAN_ZERO_CORRECTION = 1e-3
# When a per-class recall is 0 (a present class the model never predicts), the
# strict geometric mean collapses the whole GMean to exactly 0, which makes it
# uninformative next to the other 9 (arithmetic) metrics. We floor such zeros to
# this small value so GMean degrades gracefully -- standard imbalance-learning
# practice (cf. imbalanced-learn's geometric_mean_score `correction`). The score
# stays on [0, 1] and the ranking signal is preserved (a model that misses a
# class is still heavily penalised, just not to exactly 0). Set to 0.0 for the
# strict textbook geometric mean.


def compute_metrics(y_true, y_pred, y_proba, n_classes, classes):
    """All 10 metrics. y_true / y_pred are integer-encoded 0..C-1; y_proba is
    (N, C) aligned to `classes`. Robust: AUC/AUPRC/MCC fall back to 0.0 when a
    metric is undefined (e.g. a class absent from the test fold)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_proba = np.asarray(y_proba)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # GMean: geometric mean of per-class recall, but ONLY over classes actually
    # present in the test fold (confusion-matrix row support > 0). A class with
    # no test samples cannot be misclassified, so counting it as recall 0 (as the
    # old np.all(per_recall>0) gate did) spuriously zeroed the whole score on
    # high-IR folds where a tiny minority happened to be absent from the test
    # split. Excluding absent classes removes that artifact. A *present* class the
    # model never predicts still has recall 0; we floor it to
    # GMEAN_ZERO_CORRECTION so the score degrades gracefully instead of
    # collapsing to exactly 0. exp(mean(log(.))) is the numerically-stable form.
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    with np.errstate(all="ignore"):
        row_support = cm.sum(axis=1)
        per_recall = np.diag(cm) / np.maximum(row_support, 1)
    per_recall = np.nan_to_num(per_recall, nan=0.0)
    recalls = per_recall[row_support > 0]
    if recalls.size == 0:
        gmean = 0.0
    else:
        recalls = np.where(recalls > 0, recalls, GMEAN_ZERO_CORRECTION)
        gmean = float(np.exp(np.mean(np.log(recalls))))

    # AUC / AUPRC
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
            auprc = average_precision_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            # macro AUPRC via one-hot y
            y_oh = np.zeros_like(y_proba)
            y_oh[np.arange(len(y_true)), np.clip(y_true, 0, n_classes - 1)] = 1.0
            auprc = average_precision_score(y_oh, y_proba, average="macro")
    except Exception:
        auc, auprc = 0.0, 0.0

    # MCC
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0

    # IBA(T=0.5) macro: per-class (1 + T(TPr - TNr)) * TPr
    iba_vals = []
    total = cm.sum()
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        iba_vals.append((1.0 + IBA_T * (tpr - tnr)) * tpr)
    iba = float(np.mean(iba_vals)) if iba_vals else 0.0

    # Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        "Accuracy": float(acc), "Precision": float(prec), "Recall": float(rec),
        "F1": float(f1), "GMean": float(gmean), "AUC": float(auc),
        "MCC": float(mcc), "AUPRC": float(auprc), "IBA": float(iba),
        "Kappa": float(kappa),
    }


def _apply_tnr_font():
    """Force Times New Roman (Windows system font) with a serif fallback."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"


def save_confusion_matrix(y_true_per_fold, y_pred_per_fold, classes, out_path,
                          title=None, cmap="Blues", dpi=600):
    """Pool (sum) the per-fold confusion matrices into one, row-normalize, and
    render at dpi=`dpi` with Times New Roman + English labels. Returns out_path."""
    n = len(classes)
    cm = np.zeros((n, n), dtype=float)
    for yt, yp in zip(y_true_per_fold, y_pred_per_fold):
        cm += confusion_matrix(np.asarray(yt).ravel(), np.asarray(yp).ravel(),
                               labels=list(range(n)))
    # row-normalize (recall per true class); guard zero rows
    row_sums = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float),
                    where=row_sums > 0)

    _apply_tnr_font()
    fig, ax = plt.subplots(figsize=(max(4, 0.55 * n + 2), max(3.5, 0.55 * n + 1.5)))
    im = ax.imshow(cmn, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    # integer ticks labelled with original class values
    tick_labels = [str(c) for c in classes]
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels)

    # annotate with percentage
    thresh = cmn.max() / 2.0 if cmn.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            v = cmn[i, j]
            if row_sums[i, 0] > 0:
                ax.text(j, i, f"{v*100:.1f}", ha="center", va="center",
                        color="white" if v > thresh else "black", fontsize=max(5, 9 - n // 4))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
