"""common.paths — relocatable project path constants.

All paths are resolved from THIS file's location so the whole project is
relocatable and runnable from any working directory (no hardcoded absolute
paths). common/ lives directly under the project root, so the project root is
one level up from this file.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # D:\imb_project

DATA_ROOT    = os.path.join(PROJECT_ROOT, "Dataset")
KEEL_5FOLD   = os.path.join(DATA_ROOT, "KEEL", "5fold")        # <name>/fold{i}_{train,test}.csv
KEEL_COMPLETE = os.path.join(DATA_ROOT, "KEEL", "complete")    # <name>.csv (whole dataset, single split)
BEARING_RAW = os.path.join(DATA_ROOT, "Bearing")             # <name>.csv  (raw, header w/ feat_* + label)
BEARING_IR  = os.path.join(DATA_ROOT, "Bearing_IR")          # IR{5,10,20,30}/<name>.csv  (constructed)
SOFTWARE    = os.path.join(DATA_ROOT, "Software")          # AR.csv, CM.csv, KC.csv, MC.csv (features + defects label)
HEART_RAW   = os.path.join(DATA_ROOT, "Heart")             # <name>.csv  (raw, header w/ features + label_last_col)

RESULTS_DIR  = os.path.join(PROJECT_ROOT, "Result")
# FIGURES_DIR  = os.path.join(RESULTS_DIR, "Figures", "Confusion")
CM_JSON_DIR  = os.path.join(RESULTS_DIR, "ConfusionMatrices")
# REPORTS_DIR  = os.path.join(RESULTS_DIR, "Reports")
# PERMODEL_DIR = os.path.join(RESULTS_DIR, "PerModel")


def ensure_dirs():
    """Lazily create the output directory tree (idempotent)."""
    for d in (RESULTS_DIR, CM_JSON_DIR,
              # FIGURES_DIR, REPORTS_DIR, PERMODEL_DIR,
              BEARING_IR,
              *[os.path.join(BEARING_IR, f"IR{ir}") for ir in (5, 10, 20, 30)]):
        os.makedirs(d, exist_ok=True)
