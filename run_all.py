"""run_all.py — Unified benchmark runner for multiclass imbalance learning.

Auto-discovers models from Model/ and datasets from Dataset/.
Configure via config.yaml (optional) — without it, runs all models on all datasets.
Resume is automatic: existing results in the output xlsx are skipped.
Exports:
  - Results as xlsx (metrics per model×dataset, one sheet)
  - Confusion matrices as JSON (one file per model×dataset cell)

Usage:
    python run_all.py                         # smoke mode with config.yaml
    python run_all.py --config my_conf.yaml   # use a different config
    python run_all.py --full                  # full benchmark (overrides config)
    python run_all.py --models soup,spe       # restrict models (overrides config)
    python run_all.py --resume               # resume from last interruption
    python run_all.py --list-models           # print discovered models and exit
"""
import os
import sys
import time
import argparse
import importlib
import traceback
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch libiomp5md guard

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from common import paths
from common.dataio import discover_all_datasets, load_folds
from common.base import run_model_on_dataset


# =============================================================================
# YAML config
# =============================================================================

def _load_yaml(path):
    """Load a YAML file. Returns {} if the file is missing."""
    import yaml as _yaml
    path = os.path.join(PROJECT_ROOT, path)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return _yaml.safe_load(f) or {}


# =============================================================================
# Model discovery
# =============================================================================

def discover_models():
    """Scan Model/*.py, import each module, and return those exposing build().

    Returns:
        dict: {model_key: module}  where model_key is the module's MODEL_KEY attr.
    """
    model_dir = os.path.join(PROJECT_ROOT, "Model")
    avail = {}
    if not os.path.isdir(model_dir):
        print("[run_all] WARNING: Model/ directory not found")
        return avail

    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        mod_name = fname[:-3]
        try:
            mod = importlib.import_module(f"Model.{mod_name}")
        except Exception as e:
            print(f"[run_all] skip '{mod_name}' (import failed: {type(e).__name__}: {e})")
            continue
        if not hasattr(mod, "build"):
            print(f"[run_all] skip '{mod_name}' (no build() yet)")
            continue
        model_key = getattr(mod, "MODEL_KEY", mod_name)
        avail[model_key] = mod
    return avail


def build_paper_name_map():
    """Match MODEL_KEY to paper PDF names from Paper/.

    Normalises both sides (lowercase, strip separators) for matching.
    Returns {model_key: paper_name}.
    """
    paper_dir = os.path.join(PROJECT_ROOT, "Paper")
    paper_names = []
    if os.path.isdir(paper_dir):
        for fname in os.listdir(paper_dir):
            if fname.lower().endswith(".pdf"):
                paper_names.append(fname[:-4])

    def _norm(s):
        return s.lower().replace("-", "").replace("_", "")

    norm_to_paper = {_norm(p): p for p in paper_names}

    avail = discover_models()
    mapping = {}
    for key in avail:
        paper = norm_to_paper.get(_norm(key))
        if paper:
            mapping[key] = paper
        else:
            # fallback: use MODEL_KEY as-is
            mapping[key] = key
    return mapping


# =============================================================================
# Config loading
# =============================================================================

def _resolve_list(yaml_cfg, key, cli_val):
    """Resolve a list config value: CLI > YAML.

    YAML semantics (when CLI is not provided):
      - key missing               → None   (auto-discover all)
      - key present, value []     → []     (explicitly exclude all)
      - key present, value [a,b]  → [a,b]  (only those)

    Uses ``in`` to distinguish "key missing" from "key present with []"
    — avoiding the Python ``or`` trap where ``[] or fallback`` evaluates
    to ``fallback`` because empty lists are falsy.
    """
    if cli_val is not None:
        return cli_val
    if key in yaml_cfg:
        return yaml_cfg[key] or []
    return None


def load_config(config_path="config.yaml", cli_args=None):
    """Load YAML config and merge with CLI arguments (CLI wins).

    Returns a dict with all settings resolved.

    Sentinel convention for list fields:
      None  = "all" (auto-discover / default)
      []    = "none" (skip this category entirely)
      [...] = "only these"
    """
    yaml_cfg = _load_yaml(config_path)
    cli = cli_args or {}

    # --- mode ---
    mode = yaml_cfg.get("mode", "full")
    if cli.get("full"):
        mode = "full"
    smoke = (mode != "full")

    # --- models ---
    # models: [] or missing → all (you always need at least one model;
    # auto-discovery is the convenient default here)
    config_models = cli.get("models") if cli.get("models") is not None else (yaml_cfg.get("models") or None)

    # --- datasets (KEEL), bearing, nids, heart ---
    # Strict: [] really means "none"; missing key means "all".
    config_datasets = _resolve_list(yaml_cfg, "datasets", cli.get("datasets"))

    bearing_cfg = yaml_cfg.get("bearing") or {}
    config_bearing_names = _resolve_list(bearing_cfg, "names", cli.get("bearing_names"))
    config_irs        = _resolve_list(bearing_cfg, "irs",   cli.get("irs"))

    nids_cfg = yaml_cfg.get("nids") or {}
    config_nids_names = _resolve_list(nids_cfg, "names", cli.get("nids_names"))

    heart_cfg = yaml_cfg.get("heart") or {}
    config_heart_names = _resolve_list(heart_cfg, "names", cli.get("heart_names"))

    # --- folds ---
    n_folds = cli.get("n_folds") or yaml_cfg.get("n_folds") or (1 if smoke else 5)

    # --- output ---
    output_cfg = yaml_cfg.get("output") or {}
    save_cm = output_cfg.get("save_cm", True)
    out_file = cli.get("out") or output_cfg.get("file") or None

    # --- seed ---
    base_seed = yaml_cfg.get("base_seed", 42)

    # --- model params ---
    model_params = yaml_cfg.get("model_params") or {}

    # --- resume ---
    resume = cli.get("resume", False)

    return {
        "smoke": smoke,
        "mode": mode,
        "models": config_models,              # None=all, []=all (convenience), [...]=only
        "datasets": config_datasets,          # None=all, []=none, [...]=only
        "bearing_names": config_bearing_names,# None=all, []=none, [...]=only
        "irs": tuple(config_irs) if config_irs else (),
        "nids_names": config_nids_names,      # None=all, []=none, [...]=only
        "heart_names": config_heart_names,    # None=all, []=none, [...]=only
        "n_folds": n_folds,
        "save_cm": save_cm,
        "out_file": out_file,
        "base_seed": base_seed,
        "model_params": model_params,
        "resume": resume,
    }


# =============================================================================
# Atomic write helpers
# =============================================================================

def _atomic_to_xlsx(df, path):
    """Write DataFrame to xlsx atomically (tmp then replace)."""
    import openpyxl
    from openpyxl.utils.dataframe import dataframe_to_rows
    tmp = path + ".tmp"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)
    wb.save(tmp)
    os.replace(tmp, path)


# =============================================================================
# Main runner
# =============================================================================

def run_benchmark(config):
    """Run the benchmark. Returns the results DataFrame."""
    smoke = config["smoke"]
    mode = config["mode"]
    resume = config["resume"]
    save_cm = config["save_cm"]
    base_seed = config["base_seed"]
    n_folds = config["n_folds"]
    model_params = config["model_params"]

    # --- discover models ---
    all_avail = discover_models()
    if not all_avail:
        raise RuntimeError("no runnable models found in Model/")

    # filter by config
    if config["models"]:
        avail = {k: m for k, m in all_avail.items() if k in config["models"]}
        missing = set(config["models"]) - set(avail)
        if missing:
            print(f"[run_all] WARNING: requested models not found: {missing}")
    else:
        # None or [] → auto-discover all
        avail = dict(all_avail)

    if not avail:
        raise RuntimeError("no models selected (check config models list)")

    # --- paper name mapping ---
    paper_names = build_paper_name_map()

    # --- resolve datasets with strict semantics ---
    # None = not configured → all; [] = explicitly empty → none; [...] = only those
    def _filter_set(val):
        """None → None (no filter / all); [] → empty set (exclude all); [...] → set(...)."""
        if val is None:
            return None
        return set(val) if val else set()

    datasets      = config["datasets"]
    bearing_names = config["bearing_names"]
    irs           = config["irs"]
    nids_names    = config["nids_names"]
    heart_names   = config["heart_names"]

    # --- smoke mode: if nothing is configured, provide a sensible default ---
    all_empty = (
        (datasets is None or len(datasets) == 0) and
        (bearing_names is None or len(bearing_names) == 0) and
        (nids_names is None or len(nids_names) == 0) and
        (heart_names is None or len(heart_names) == 0)
    )
    if smoke and all_empty:
        datasets = ["ecoli"]
        bearing_names = []
        irs = ()
        nids_names = []
        heart_names = []
        n_folds = 5

    # --- discover datasets ---
    ds_list = discover_all_datasets(
        keel_filter=_filter_set(datasets),
        bearing_irs=irs if irs else (5, 10, 20, 30),
        bearing_names=_filter_set(bearing_names),
        nids_names=_filter_set(nids_names),
        heart_names=_filter_set(heart_names))
    if not ds_list:
        raise FileNotFoundError("no datasets found for the given filters")

    # --- resolve output path ---
    out_file = config["out_file"]
    if out_file is None:
        out_file = "Smoke_Results.xlsx" if smoke else "benchmark_results.xlsx"
    out_path = os.path.join(paths.RESULTS_DIR, out_file)
    paths.ensure_dirs()

    # --- load existing results for resume ---
    rows = []
    completed = set()
    if resume and os.path.isfile(out_path):
        try:
            existing_df = pd.read_excel(out_path, sheet_name="Results", engine="openpyxl")
            rows = existing_df.to_dict("records")
            for row in rows:
                m = str(row.get("Model", ""))
                d = str(row.get("Dataset", ""))
                if m and d:
                    completed.add((m, d))
            print(f"[run_all] loaded {len(completed)} existing cells from "
                  f"{os.path.basename(out_path)}")
        except Exception as e:
            print(f"[run_all] could not read existing xlsx ({e}); starting fresh")

    # --- report ---
    total_cells = len(ds_list) * len(avail)
    new_cells = total_cells - len(completed)
    print(f"\n{'=' * 94}\n"
          f" run_all [{mode.upper()}]  models={len(avail)}  datasets={len(ds_list)}"
          f"  n_folds={n_folds}\n"
          f" existing={len(completed)}  new={new_cells}  total={total_cells}\n"
          f" output={out_path}\n"
          f"{'=' * 94}")

    # Confusion-matrix JSON dir
    cm_json_dir = os.path.join(paths.CM_JSON_DIR) if save_cm else None
    if cm_json_dir:
        os.makedirs(cm_json_dir, exist_ok=True)

    # --- main loop ---
    for ds in ds_list:
        try:
            folds, classes = load_folds(ds)
        except Exception as e:
            print(f"\n--- LOAD FAIL {ds.name}: {e}")
            continue
        folds_run = folds[:n_folds]
        for key, mod in avail.items():
            paper_name = paper_names.get(key, key)

            if (paper_name, ds.name) in completed:
                print(f"\n--- skip {paper_name:12s} | {ds.name} (already in xlsx) ---",
                      flush=True)
                continue

            t0 = time.time()
            print(f"\n--- {paper_name:12s} | {ds.name} (C={len(classes)}, "
                  f"fold={folds_run[0].X_train.shape[0]}x{folds_run[0].X_test.shape[0]}) ---",
                  flush=True)

            try:
                # Build model factory with optional parameter overrides
                extra_params = model_params.get(key, {})
                factory = (lambda m, ep: (lambda seed: m.build(
                    random_state=seed, smoke=smoke, **ep)))(mod, extra_params)
                res = run_model_on_dataset(
                    factory, ds, folds_run, classes,
                    model_key=paper_name, base_seed=base_seed,
                    save_cm=save_cm, cm_json_dir=cm_json_dir, verbose=smoke)
                row = {
                    "Model": paper_name,
                    "Dataset": ds.name,
                    "IR": (f"IR{ds.ir}" if ds.source == "bearing" else "-"),
                    "Source": ds.source,
                }
                row.update(res)
                metric_str = ""
                if "Accuracy_mean" in res:
                    metric_str = (f"Acc={res['Accuracy_mean']:.3f} "
                                  f"F1={res['F1_mean']:.3f} "
                                  f"GMean={res['GMean_mean']:.3f}")
                print(f"    -> done in {time.time() - t0:.1f}s ({metric_str})",
                      flush=True)
            except Exception as e:
                print(f"    -> ERROR: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                row = {
                    "Model": paper_name,
                    "Dataset": ds.name,
                    "IR": (f"IR{ds.ir}" if ds.source == "bearing" else "-"),
                    "Source": ds.source,
                    "Error": str(e),
                }

            rows.append(row)
            completed.add((paper_name, ds.name))
            _atomic_to_xlsx(pd.DataFrame(rows), out_path)
            print(f"    [incremental save] {len(completed)}/{total_cells} cells -> "
                  f"{os.path.basename(out_path)}", flush=True)

    # --- final save ---
    df = pd.DataFrame(rows)
    _atomic_to_xlsx(df, out_path)
    print(f"\n{'=' * 94}\n{len(rows)} cells total -> {out_path}\n{'=' * 94}")
    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Unified imbalance benchmark runner (config-driven, auto-resume)")
    ap.add_argument("--config", default="config.yaml",
                    help="Path to YAML config file (default: config.yaml)")
    ap.add_argument("--full", action="store_true",
                    help="Full benchmark (overrides config mode)")
    ap.add_argument("--models", default=None,
                    help="Comma list of MODEL_KEYs (overrides config)")
    ap.add_argument("--datasets", default=None,
                    help="Comma list of KEEL dataset names (overrides config)")
    ap.add_argument("--bearing-names", default=None,
                    help="Comma list of Bearing names (overrides config)")
    ap.add_argument("--irs", default=None,
                    help="Comma list of Bearing IRs, e.g. 5,20 (overrides config)")
    ap.add_argument("--nids-names", default=None,
                    help="Comma list of NIDS dataset names (overrides config)")
    ap.add_argument("--heart-names", default=None,
                    help="Comma list of Heart dataset names (overrides config)")
    ap.add_argument("--n-folds", type=int, default=None,
                    help="Folds per dataset (overrides config)")
    ap.add_argument("--out", default=None,
                    help="Output xlsx path (overrides config)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last run (skip completed cells)")
    ap.add_argument("--list-models", action="store_true",
                    help="Print discovered models with paper names and exit")
    args = ap.parse_args()

    # --- list-models is a read-only action ---
    if args.list_models:
        avail = discover_models()
        paper_names = build_paper_name_map()
        print(f"{'MODEL_KEY':16s} {'Paper name':16s} {'Module'}")
        print("-" * 52)
        for key, mod in sorted(avail.items()):
            pname = paper_names.get(key, key)
            print(f"  {key:14s}  {pname:14s}  Model.{mod.__name__.split('.')[-1]}.py")
        print(f"\n{len(avail)} models discovered")
        return

    # --- build CLI overrides dict ---
    cli = {
        "full": args.full,
        "models": ([x.strip() for x in args.models.split(",")]
                   if args.models else None),
        "datasets": ([x.strip() for x in args.datasets.split(",")]
                     if args.datasets else None),
        "bearing_names": ([x.strip() for x in args.bearing_names.split(",")]
                          if args.bearing_names else None),
        "irs": (tuple(int(x) for x in args.irs.split(","))
                if args.irs else None),
        "nids_names": ([x.strip() for x in args.nids_names.split(",")]
                       if args.nids_names else None),
        "heart_names": ([x.strip() for x in args.heart_names.split(",")]
                        if args.heart_names else None),
        "n_folds": args.n_folds,
        "out": args.out,
        "resume": args.resume,
    }

    config = load_config(args.config, cli)
    run_benchmark(config)


if __name__ == "__main__":
    main()
