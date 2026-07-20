"""common.base — the ONE fit/predict loop every model runs through.

``run_model_on_dataset`` is classifier-agnostic: it only calls the model's
``fit`` / ``predict_proba`` (the uniform contract defined in the plan). It owns:

  * train-fold-only StandardScaler + LabelEncoder (the leakage invariant),
  * per-fold fit timing (Runtime = fit seconds, prediction excluded),
  * the 10 metrics per fold + mean/std aggregation,
  * pooled confusion-matrix PNG export.

This centralization replaces the 30-fold duplicated split/scale/fit/predict/
evaluate loops that existed in the old per-model harnesses.
"""
import time
import numpy as np

from .metrics import compute_metrics, save_confusion_matrix, save_confusion_matrix_json, METRIC_KEYS
from .preprocessing import Preprocessor


def _align_proba(proba, model, n_classes):
    """Map a model's predict_proba columns onto the full 0..C-1 class index.

    With stratified folds every class normally appears in the train fold, so a
    Random Forest's predict_proba already has C columns. This is a safety net for
    the rare case where a class is absent from one train fold: it reindexes using
    the model's exposed `classes_` (the model's own or its embedded clf_'s)."""
    proba = np.asarray(proba, dtype=float)
    C = proba.shape[1]
    if C == n_classes:
        return proba
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    mc = getattr(model, "classes_", None)
    if mc is None:
        clf = getattr(model, "clf_", None)
        mc = getattr(clf, "classes_", None) if clf is not None else None
    if mc is None:
        out[:, :min(C, n_classes)] = proba[:, :min(C, n_classes)]
        return out
    for j, c in enumerate(mc):
        try:
            out[:, int(c)] = proba[:, j]
        except (ValueError, IndexError):
            continue
    # renormalize rows just in case
    s = out.sum(axis=1, keepdims=True)
    out = np.divide(out, s, out=out, where=s > 0)
    return out


def run_model_on_dataset(model_factory, desc, folds, classes, fig_dir=None,
                         model_key="model", scale=True, base_seed=42,
                         save_cm=True, verbose=True, cm_json_dir=None):
    """Run one model across all folds of one dataset.

    Args:
        model_factory: callable(seed=int) -> fresh model with fit/predict_proba.
        desc:          DatasetDescriptor (used for output filenames).
        folds:         list[Fold] (typically 5).
        classes:       sorted unique label array (confusion-matrix ticks).
        fig_dir:       directory for the confusion PNG (None disables).
        model_key:     model name, used in filenames + prints.
        scale, base_seed, save_cm, verbose: as named.
        cm_json_dir:   directory for confusion-matrix JSON (None disables).
    Returns:
        flat dict {<metric>_mean, <metric>_std} for METRIC_KEYS + Runtime.
    """
    n_classes = len(classes)
    keys = METRIC_KEYS + ("Runtime",)
    per_fold = []
    cm_true, cm_pred = [], []

    for fold in folds:
        pre = Preprocessor(scale=scale)
        Xtr, ytr = pre.fit_transform(fold.X_train, fold.y_train)
        Xte = pre.transform_X(fold.X_test)
        yte = pre.transform_y(fold.y_test)

        model = model_factory(seed=base_seed + fold.fold_id)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_secs = time.perf_counter() - t0

        proba = _align_proba(model.predict_proba(Xte), model, n_classes)
        ypred = np.argmax(proba, axis=1)

        m = compute_metrics(yte, ypred, proba, n_classes, classes)
        m["Runtime"] = fit_secs
        per_fold.append(m)
        cm_true.append(yte); cm_pred.append(ypred)
        if verbose:
            print(f"    fold{fold.fold_id}: "
                  f"Acc={m['Accuracy']:.4f} F1={m['F1']:.4f} "
                  f"GMean={m['GMean']:.4f} AUC={m['AUC']:.4f} "
                  f"({fit_secs:.1f}s)", flush=True)

    out = {}
    for k in keys:
        vals = np.array([m[k] for m in per_fold], dtype=float)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))

    if save_cm and cm_json_dir:
        import os
        out_path = os.path.join(cm_json_dir, f"{model_key}__{desc.name}.json")
        save_confusion_matrix_json(cm_true, cm_pred, classes,
                                   model_key, desc.name, out_path)
    elif save_cm and fig_dir:
        import os
        out_path = os.path.join(fig_dir, f"{model_key}__{desc.name}.png")
        save_confusion_matrix(cm_true, cm_pred, classes, out_path,
                              title=f"{model_key} | {desc.name}")
    return out
