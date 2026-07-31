# -*- coding: utf-8 -*-
"""_run_cell.py — Standalone single-cell runner invoked via subprocess.

Usage:
    python _run_cell.py --model-key Ours --module-name Model.ours \\
        --dataset-json '{"name":"ecoli1","source":"keel",...}' \\
        --n-folds 5 --paper-name Ours --base-seed 42 \\
        [--smoke] [--save-cm] [--cm-json-dir path]

Prints one JSON line to stdout on success:
    {"status":"success","Accuracy_mean":0.9,...}

Prints one JSON line to stdout on error:
    {"status":"error","error":"message"}

Exit code is 0 on success (even if the model raised a Python exception — the
JSON encodes the error).  A C-level crash (segfault) produces a non-zero exit
code and no valid JSON on stdout — the parent detects this via the exit code.
"""
import os
import sys
import json
import argparse

# ---- thread limits (must be set BEFORE any numpy/scipy import) ----------
for _tv in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]:
    os.environ[_tv] = "1"

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---- path -----------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--module-name", required=True)
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--n-folds", type=int, required=True)
    ap.add_argument("--paper-name", required=True)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--save-cm", action="store_true")
    ap.add_argument("--cm-json-dir", default=None)
    ap.add_argument("--model-params-json", default="{}")
    args = ap.parse_args()

    try:
        import importlib
        from common.dataio import DatasetDescriptor, load_folds
        from common.base import run_model_on_dataset

        # Reconstruct dataset descriptor
        desc_dict = json.loads(args.dataset_json)
        desc = DatasetDescriptor(**desc_dict)

        # Load folds
        folds, classes = load_folds(desc)
        folds_run = folds[:args.n_folds]

        # Import model
        mod = importlib.import_module(args.module_name)

        # Build factory
        model_params = json.loads(args.model_params_json)
        extra_params = model_params.get(args.model_key, {})
        factory = (lambda m, ep: (lambda seed: m.build(
            random_state=seed, smoke=args.smoke, **ep)))(mod, extra_params)

        res = run_model_on_dataset(
            factory, desc, folds_run, classes,
            model_key=args.paper_name, base_seed=args.base_seed,
            save_cm=args.save_cm, cm_json_dir=args.cm_json_dir,
            verbose=args.smoke)

        result = {"status": "success"}
        result.update(res)
        print(json.dumps(result), flush=True)
        sys.exit(0)

    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}), flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
