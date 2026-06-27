"""run_all.py — unified entry for the 30-model multiclass imbalance benchmark.

Covers the migrated old-5 (SPE/DeepSMOTE/DDPM/DBCF/DEAHS) plus 25 newly
reproduced survey methods (14 data-resampling + 11 ensemble) on a shared
benchmark: 115 KEEL 5-fold datasets + 32 constructed Bearing IR{5,10,20,30}
datasets = 147 datasets, 5-fold CV, 10 imbalance metrics + runtime, pooled
confusion matrices (dpi=600, Times New Roman).

Design:
  * All paths RELATIVE (resolved via PROJECT_ROOT); project is relocatable.
  * Models are discovered DYNAMICALLY: the registry maps a model key to a module
    path; at run time each module is imported and, if it exposes ``build()``,
    included. A model that is not yet implemented (or fails to import) is simply
    skipped with a warning, so this script stays runnable as models are added.
  * Every model conforms to the uniform contract: ``build(random_state, smoke)``
    -> an object with ``fit(X,y)`` / ``predict_proba(X)``. The shared runner
    ``common.base.run_model_on_dataset`` drives the fold loop, scaling, metrics,
    timing and confusion-matrix export. No per-model split/scale/eval code here.
  * Incremental atomic CSV write (write to .tmp then os.replace) after every
    (dataset, model) cell, so a crash mid-run never discards finished cells.

Modes:
  smoke (DEFAULT): 1 dataset, 1 fold, smoke=True model configs. Fast end-to-end
                   sanity check. NOT a benchmark.
  full  (--full):  all datasets, 5 folds, smoke=False (paper-faithful configs).

Usage:
    python run_all.py                                # smoke on ecoli1
    python run_all.py --full                         # full benchmark (slow)
    python run_all.py --datasets ecoli,glass         # restrict KEEL names
    python run_all.py --models SPE,soup              # restrict models
    python run_all.py --bearing-names CWRU --irs 5,20
"""
import os
import sys
import time
import argparse
import importlib
import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch libiomp5md guard

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from common import paths
from common.dataio import discover_all_datasets, load_folds, DatasetDescriptor
from common.base import run_model_on_dataset
from common.metrics import METRIC_KEYS

# ----- model registry: key -> dotted module path (canonical order) ------------
MODEL_REGISTRY = {
    # migrated old-5 (filed by algorithm: ensemble={SPE,DBCF,DEAHS}, resampling={DeepSMOTE,DDPM})
    "SPE": "Model.ensemble.spe", "DeepSMOTE": "Model.resampling.deepsmote",
    # "DDPM": "Model.resampling.ddpm",   # DDPM commented out -- excluded from all runs
    "DBCF": "Model.ensemble.dbcf", "DEAHS": "Model.ensemble.deahs",
    # 14 data-resampling
    "s_smote": "Model.resampling.s_smote", "scut": "Model.resampling.scut",
    "mdo": "Model.resampling.mdo", "smom": "Model.resampling.smom",
    "soup": "Model.resampling.soup", "mc_rbo": "Model.resampling.mc_rbo",
    "mc_ccr": "Model.resampling.mc_ccr", "ocsv_us": "Model.resampling.ocsv_us",
    "shsampler": "Model.resampling.shsampler", "mc_evhs": "Model.resampling.mc_evhs",
    "nromm": "Model.resampling.nromm", "orem_m": "Model.resampling.orem_m",
    "glos": "Model.resampling.glos", "mc_nro": "Model.resampling.mc_nro",
    # 11 ensemble
    "easy_bpnn": "Model.ensemble.easy_bpnn", "amcs": "Model.ensemble.amcs",
    "des_mi": "Model.ensemble.des_mi", "drcw_aseg": "Model.ensemble.drcw_aseg",
    "pt_bagging": "Model.ensemble.pt_bagging", "evinci": "Model.ensemble.evinci",
    "multirandbal": "Model.ensemble.multirandbal", "dpse": "Model.ensemble.dpse",
    "oremboost": "Model.ensemble.oremboost", "e_evrs": "Model.ensemble.e_evrs",
    "adaboost_ad": "Model.ensemble.adaboost_ad",
}


def resolve_models(requested):
    """Import requested model modules; keep those exposing build(). Returns
    dict key -> module. A missing/broken module is skipped with a warning."""
    keys = list(MODEL_REGISTRY) if requested is None else list(requested)
    avail = {}
    for key in keys:
        path = MODEL_REGISTRY.get(key)
        if path is None:
            print(f"[run_all] WARNING: unknown model key '{key}' (ignored)")
            continue
        try:
            mod = importlib.import_module(path)
        except Exception as e:
            print(f"[run_all] skip '{key}' (import failed: {type(e).__name__}: {e})")
            continue
        if not hasattr(mod, "build"):
            print(f"[run_all] skip '{key}' (no build() yet)")
            continue
        avail[key] = mod
    return avail


def _atomic_to_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_all(smoke=True, datasets=None, models=None, irs=(5, 10, 20, 30),
            bearing_names=None, n_folds=None, save_cm=True, out_path=None,
            base_seed=42):
    """Run every (model x dataset) cell; return a combined results DataFrame."""
    if n_folds is None:
        n_folds = 1 if smoke else 5

    avail = resolve_models(models)
    if not avail:
        raise RuntimeError("no runnable models found")

    # dataset discovery: smoke defaults to a small representative subset covering
    # binary (ecoli1) + multiclass (glass) KEEL and one multiclass Bearing IR set.
    if smoke and datasets is None and bearing_names is None:
        datasets = ["ecoli1", "glass"]
        bearing_names = ["CWRU"]
        irs = (5,)

    ds_list = discover_all_datasets(
        keel_filter=(set(datasets) if datasets else None),
        bearing_irs=irs,
        bearing_names=(set(bearing_names) if bearing_names else None))
    if not ds_list:
        raise FileNotFoundError("no datasets found for the given filters")

    mode = "SMOKE" if smoke else "FULL"
    print(f"\n{'='*94}\n run_all [{mode}] models={list(avail)} "
          f"n_datasets={len(ds_list)} n_folds={n_folds}\n{'='*94}")

    if out_path is None:
        out_path = os.path.join(paths.RESULTS_DIR,
                                "Smoke_Results.csv" if smoke else "All_Models_Results.csv")
    paths.ensure_dirs()

    rows = []
    for ds in ds_list:
        try:
            folds, classes = load_folds(ds)
        except Exception as e:
            print(f"\n--- LOAD FAIL {ds.name}: {e}")
            continue
        folds_run = folds[:n_folds]
        for key, mod in avail.items():
            t0 = time.time()
            print(f"\n--- {key:12s} | {ds.name} (C={len(classes)}, "
                  f"fold={folds_run[0].X_train.shape[0]}x{folds_run[0].X_test.shape[0]}) ---",
                  flush=True)
            try:
                factory = (lambda m: (lambda seed: m.build(random_state=seed, smoke=smoke)))(mod)
                res = run_model_on_dataset(factory, ds, folds_run, classes,
                                           fig_dir=paths.FIGURES_DIR if save_cm else None,
                                           model_key=key, base_seed=base_seed,
                                           save_cm=save_cm, verbose=smoke)
                row = {"Model": key, "Dataset": ds.name,
                       "IR": (f"IR{ds.ir}" if ds.source == "bearing" else "-"),
                       "Source": ds.source}
                row.update(res)
                print(f"    -> done in {time.time()-t0:.1f}s "
                      f"(Acc={res['Accuracy_mean']:.3f} F1={res['F1_mean']:.3f} "
                      f"GMean={res['GMean_mean']:.3f})", flush=True)
            except Exception as e:
                import traceback
                print(f"    -> ERROR: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                row = {"Model": key, "Dataset": ds.name,
                       "IR": (f"IR{ds.ir}" if ds.source == "bearing" else "-"),
                       "Source": ds.source, "Error": str(e)}
            rows.append(row)
            _atomic_to_csv(pd.DataFrame(rows), out_path)
            print(f"    [incremental save] {len(rows)} cells -> "
                  f"{os.path.basename(out_path)}", flush=True)

    df = pd.DataFrame(rows)
    _atomic_to_csv(df, out_path)
    print(f"\n{'='*94}\n{len(rows)} cells complete -> {out_path}\n{'='*94}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Unified 30-model imbalance benchmark runner")
    ap.add_argument("--full", action="store_true",
                    help="Full benchmark (all datasets, 5 folds). Default = smoke.")
    ap.add_argument("--models", default=None,
                    help="Comma list of model keys (default: all available)")
    ap.add_argument("--datasets", default=None,
                    help="Comma list of KEEL names to restrict to")
    ap.add_argument("--bearing-names", default=None,
                    help="Comma list of Bearing names (default all 8)")
    ap.add_argument("--irs", default="5,10,20,30",
                    help="Comma list of Bearing IRs (default 5,10,20,30)")
    ap.add_argument("--n-folds", type=int, default=None,
                    help="Folds per dataset (default smoke=1, full=5)")
    ap.add_argument("--no-cm", action="store_true", help="Skip confusion-matrix export")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--list-models", action="store_true", help="Print registry and exit")
    args = ap.parse_args()

    if args.list_models:
        print("Model registry (key -> module):")
        for k, v in MODEL_REGISTRY.items():
            print(f"  {k:12s} -> {v}")
        return

    smoke = not args.full
    models = [x.strip() for x in args.models.split(",")] if args.models else None
    datasets = [x.strip() for x in args.datasets.split(",")] if args.datasets else None
    bearing_names = [x.strip() for x in args.bearing_names.split(",")] if args.bearing_names else None
    irs = tuple(int(x) for x in args.irs.split(","))

    run_all(smoke=smoke, datasets=datasets, models=models, irs=irs,
            bearing_names=bearing_names, n_folds=args.n_folds,
            save_cm=not args.no_cm, out_path=args.out)


if __name__ == "__main__":
    main()
