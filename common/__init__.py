"""common — shared benchmark infrastructure for the 30-model multiclass
imbalance benchmark.

Public surface:
    paths       relocatable path constants
    preprocessing.Preprocessor        train-fit StandardScaler + LabelEncoder
    dataio.{DatasetDescriptor, Fold, load_folds, build_bearing_ir_datasets,
            discover_all_datasets}     unified dataset / fold IO
    metrics.{METRIC_KEYS, compute_metrics, save_confusion_matrix}
    base.run_model_on_dataset          the ONE fit/predict loop all models use
    smoke.run_smoke                    per-model smoke-test driver
    dempster                           Dempster-Shafer evidence combination
    IS_WINDOWS                         True on Windows, False otherwise
"""

import platform as _platform

IS_WINDOWS = _platform.system() == "Windows"

# ---------------------------------------------------------------------------
# On Windows, force joblib to prefer the threading backend over loky
# (multiprocessing).  The loky backend uses `spawn` on Windows (no fork()),
# and repeated pool creation/destruction across model×dataset cells causes
# worker crashes whose exceptions are not reliably propagated to the main
# process — the parent exits silently.  Sklearn's RandomForestClassifier and
# numpy-heavy custom trees both release the GIL during C-level work, so the
# threading backend is just as fast here and avoids process-spawn issues
# entirely.
#
# Must run BEFORE any sklearn/joblib import, because sklearn's internal
# joblib.Parallel reads the default backend at call time (not import time).
# "common" is the first project module imported by every entry-point, so this
# is the earliest reliable hook.
# ---------------------------------------------------------------------------
if IS_WINDOWS:
    try:
        import joblib.parallel as _jlp
        _jlp.DEFAULT_BACKEND = "threading"
    except Exception:
        pass  # joblib not installed yet — no-op; stays safe

# ---------------------------------------------------------------------------
# Silence scikit-learn's advisory parallel-config UserWarnings, project-wide.
#
# Root cause (read directly from sklearn 1.7's sklearn/utils/parallel.py):
# `_FuncWrapper.__call__` warns "`sklearn.utils.parallel.delayed` should be used
# with `sklearn.utils.parallel.Parallel` ..." whenever a task created with
# `sklearn.utils.parallel.delayed` is dispatched through PLAIN `joblib.Parallel`
# -- the scikit-learn config + warning filters are then empty on the worker, so
# the advisory fires. This project's own code (Model/ensemble/dbcf.py) imports
# BOTH Parallel and delayed from sklearn.utils.parallel, so it is *not* the
# source; the warning is raised inside third-party estimators (sklearn /
# imbalanced-learn internals). It is purely advisory (it only means sklearn's
# thread-local config won't reach that specific batch of workers) and does NOT
# affect any benchmark result, so we silence it.
#
# Registered here, on first import of `common`, so EVERY entry point
# (run_all.py, run_cwru_ir20.py, build_bearing_ir.py) -- and any loky worker
# forked afterwards -- inherits the filter before any model fits.
# ---------------------------------------------------------------------------
import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with"
            r" `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
)
# Sibling advisory from the same module (joblib.delayed fed to sklearn.Parallel).
_warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.Parallel` needs to be used in"
            r" conjunction with `sklearn\.utils\.parallel\.delayed`.*",
    category=UserWarning,
)
# joblib loky resource_tracker: stale temp memmap files were cleaned up externally
# (common on Windows). Harmless — the tracker just can't delete what's already gone.
_warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker:",
    category=UserWarning,
)
