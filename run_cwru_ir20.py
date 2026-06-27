"""run_cwru_ir20.py — full 5-fold benchmark of ALL 30 models on ONE dataset:
Dataset/Bearing_IR/IR20/CWRU.csv  (Bearing, imbalance ratio IR=20).

A focused companion to run_all.py: instead of sweeping all 147 datasets it runs
every available model on just IR{ir}/{bearing_name} (default CWRU @ IR20) with the
FULL, paper-faithful configuration (smoke=False):
  * 5-fold StratifiedKFold(shuffle=True, random_state=42) -- identical folds for
    every model (fair comparison; the split is owned by common.dataio.load_folds);
  * train-fold-only StandardScaler + LabelEncoder (no leakage);
  * the 10 imbalance metrics + Runtime, mean/std over folds;
  * pooled confusion matrices (dpi=600, Times New Roman) under Result/Figures/.

It reuses run_all.MODEL_REGISTRY + run_all.resolve_models and common.base's
run_model_on_dataset, so it stays in lock-step with the main benchmark: no
duplicated registry and no duplicated split/scale/eval code. A cell that errors
is recorded with an "Error" column and the run continues. Output is written
atomically (per-cell .tmp + os.replace) so a crash mid-run never discards
finished cells.

Usage:
    python run_cwru_ir20.py                     # all models, 5 folds, full configs
    python run_cwru_ir20.py --models soup,mdo   # restrict to a subset of models
    python run_cwru_ir20.py --n-folds 5
    python run_cwru_ir20.py --smoke             # 1 fold, smoke configs (fast sanity)
    python run_cwru_ir20.py --no-cm             # skip confusion-matrix PNG export
    python run_cwru_ir20.py --bearing-name SEU --ir 10   # any single Bearing/IR set
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch libiomp5md guard

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from common import paths
from common.dataio import DatasetDescriptor, load_folds
from common.base import run_model_on_dataset
# NOTE: DDPM is commented out in run_all.MODEL_REGISTRY, so it is excluded here too.
# run_cwru_ir20 reuses the shared registry -- there is no separate DDPM import; to
# re-enable DDPM, uncomment its line in run_all.py's MODEL_REGISTRY.
from run_all import MODEL_REGISTRY, resolve_models, _atomic_to_csv

# The dataset under test (overridable via CLI for any single Bearing/IR set).
DEFAULT_BEARING_NAME = "CWRU"
DEFAULT_IR = 20


def main():
    ap = argparse.ArgumentParser(
        description="Full 5-fold benchmark of all 30 models on one Bearing IR dataset (default IR20/CWRU)")
    ap.add_argument("--models", default=None,
                    help="Comma list of model keys (default: all available)")
    ap.add_argument("--n-folds", type=int, default=5,
                    help="Folds per run (default 5 = full; ignored when --smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke=True model configs + 1 fold (fast sanity, NOT a benchmark)")
    ap.add_argument("--no-cm", action="store_true", help="Skip confusion-matrix PNG export")
    ap.add_argument("--bearing-name", default=DEFAULT_BEARING_NAME,
                    help=f"Bearing dataset name (default {DEFAULT_BEARING_NAME})")
    ap.add_argument("--ir", type=int, default=DEFAULT_IR,
                    help=f"Imbalance ratio (default {DEFAULT_IR})")
    ap.add_argument("--out", default=None, help="Output CSV path")
    args = ap.parse_args()

    smoke = args.smoke
    n_folds = 1 if smoke else args.n_folds

    models = [x.strip() for x in args.models.split(",")] if args.models else None
    avail = resolve_models(models)
    if not avail:
        raise RuntimeError("no runnable models found (check --models / imports)")

    desc = DatasetDescriptor(name=f"IR{args.ir}_{args.bearing_name}", source="bearing",
                             bearing_name=args.bearing_name, ir=args.ir)
    folds, classes = load_folds(desc)          # deterministic 5-fold split
    folds_run = folds[:n_folds]

    csv_path = args.out or os.path.join(paths.RESULTS_DIR,
                                        f"IR{args.ir}_{args.bearing_name}_Results.csv")
    paths.ensure_dirs()

    mode = "SMOKE" if smoke else "FULL"
    print(f"\n{'='*94}\n run_cwru_ir20 [{mode}] dataset={desc.name} "
          f"(C={len(classes)}, models={len(avail)}, n_folds={n_folds})\n{'='*94}")
    print(" fold sizes: " + " ".join(f"f{i+1}={f.X_train.shape[0]}+{f.X_test.shape[0]}"
                                     for i, f in enumerate(folds_run)))
    # train class distribution of fold 1 (context for the imbalance)
    counts = np.bincount(np.unique(folds_run[0].y_train, return_inverse=True)[1])
    print(f" fold1 train class counts: {dict(zip(classes.tolist(), counts.tolist()))}")

    rows = []
    for key, mod in avail.items():
        t0 = time.time()
        print(f"\n--- {key:12s} | {desc.name} ---", flush=True)
        try:
            factory = (lambda m: (lambda seed: m.build(random_state=seed, smoke=smoke)))(mod)
            res = run_model_on_dataset(factory, desc, folds_run, classes,
                                       fig_dir=paths.FIGURES_DIR if not args.no_cm else None,
                                       model_key=key, base_seed=42,
                                       save_cm=not args.no_cm, verbose=True)
            row = {"Model": key, "Dataset": desc.name,
                   "IR": f"IR{args.ir}", "Source": "bearing"}
            row.update(res)
            print(f"    -> done in {time.time()-t0:.1f}s "
                  f"(Acc={res['Accuracy_mean']:.3f} F1={res['F1_mean']:.3f} "
                  f"GMean={res['GMean_mean']:.3f})", flush=True)
        except Exception as e:
            import traceback
            print(f"    -> ERROR: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            row = {"Model": key, "Dataset": desc.name,
                   "IR": f"IR{args.ir}", "Source": "bearing", "Error": str(e)}
        rows.append(row)
        _atomic_to_csv(pd.DataFrame(rows), csv_path)
        print(f"    [incremental save] {len(rows)}/{len(avail)} cells -> "
              f"{os.path.basename(csv_path)}", flush=True)

    df = pd.DataFrame(rows)
    _atomic_to_csv(df, csv_path)
    print(f"\n{'='*94}\n{len(rows)} cells complete -> {csv_path}\n{'='*94}")

    # ranked summary (guard against an all-error run)
    metric_cols = ["Accuracy_mean", "F1_mean", "GMean_mean", "AUC_mean", "Runtime_mean"]
    show = [c for c in metric_cols if c in df.columns]
    ok = df[df["F1_mean"].notna()] if "F1_mean" in df.columns else df.iloc[0:0]
    if len(ok) and show:
        print("\nRanked by F1_mean:")
        print(ok.sort_values("F1_mean", ascending=False)[["Model"] + show].to_string(index=False))
    elif df["Error"].notna().any() if "Error" in df.columns else False:
        print("\nNo successful runs; see Error column in the CSV.")

    return df


if __name__ == "__main__":
    main()
