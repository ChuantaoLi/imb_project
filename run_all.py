"""
run_all.py — Unified benchmark runner for multiclass imbalance learning.

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
    python run_all.py --resume               # deprecated — resume is automatic
    python run_all.py --list-models           # print discovered models and exit
"""
import os
import sys
import time
import gc
import json
import subprocess
import argparse
import importlib
import traceback
import pandas as pd
from dataclasses import asdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch libiomp5md guard

# =============================================================================
# Thread-limiting environment variables — prevent BLAS/OpenMP threading conflicts
# that cause Segmentation fault (core dumped) on some hardware.
# numpy / scipy / scikit-learn link against multi-threaded BLAS libraries
# (OpenBLAS, MKL, etc.). When multiple threads compete for the same BLAS
# resources, or when nested parallelism occurs (BLAS threads inside Python
# threads or multiprocessing workers), the C-level libraries can segfault.
# Setting each to 1 forces single-threaded BLAS throughout, which eliminates
# these crashes at a negligible throughput cost for this workload.
# =============================================================================
_THREAD_LIMIT_VARS = [
    "OMP_NUM_THREADS",           # OpenMP (used by many BLAS backends)
    "OPENBLAS_NUM_THREADS",      # OpenBLAS
    "MKL_NUM_THREADS",           # Intel MKL
    "NUMEXPR_NUM_THREADS",       # NumExpr (used by pandas)
    "VECLIB_MAXIMUM_THREADS",    # macOS Accelerate
    "BLIS_NUM_THREADS",          # BLIS
]
for _tv in _THREAD_LIMIT_VARS:
    os.environ[_tv] = "1"

# Suppress loky resource_tracker cleanup warnings — harmless on Windows where
# temp memmap files are often cleaned externally. PYTHONWARNINGS is used (not
# warnings.filterwarnings) because the resource_tracker runs in a spawned child
# process that does not inherit the parent's in-process warning filters.
_warn_extra = "ignore::UserWarning:joblib.externals.loky.backend.resource_tracker"
_cur = os.environ.get("PYTHONWARNINGS", "")
if _warn_extra not in _cur:
    os.environ["PYTHONWARNINGS"] = f"{_cur},{_warn_extra}" if _cur else _warn_extra

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

# -----------------------------------------------------------------------------
# VERSION_REGISTRY — explicit registration of every ours_vN improvement version.
# Each entry: MODEL_KEY -> (module, short description). Entries are merged into
# discovery (auto-discovery of Model/*.py also picks them up; the registry makes
# the version chain explicit and documents what changed between versions).
# -----------------------------------------------------------------------------
VERSION_REGISTRY = {
    # ours_v1 最终设计 (冻结版):
    #   1) 类预算权重: 每轮从累计集成的软边际重导出权重, 每类总预算相等 (1/C),
    #      取代 v0 的 exp(-delta*(1-rho)) 稀疏惩罚 (后者饿死少数类导致坍缩)。
    #   2) 验证式混合采样: 每类配额 = (target-n_c)*O_t*need_c, need_c 来自当前
    #      集成的类平均边际 (采样<->集成协同反馈); 配额累加器解决小类取整为 0;
    #      安全种子SMOTE / 边界种子中点球验证推 / 孤立种子不生成; 接受率门控
    #      使不可分 blob 自限为零采样; 多数类在权重层平衡 (数据层欠采样经消融
    #      证明有害已移除); 噪声清理经消融证明有害/中性, 默认关闭。
    #   3) mini-forest 弱学习器 (2 棵树, sqrt 特征, bootstrap): 单树的叶子概率
    #      在类预算下饱和 (0.9+), 毁掉概率排序 (AUC 0.881/第18); mini-forest
    #      修复排序 (AUC 0.924/第2, AUPRC 0.739/第7) 且保住 GMean/F1。
    #   Complete 协议 113 数据集: F1 0.812, GMean 0.740, AUC 0.924, AUPRC 0.737
    #   -> Avg_Rank_excl_Runtime 7.25 (22 模型中第 4, 进入前 5)。
    "ours_v1": ("Model.ours_v1",
               "v1 最终版: 类预算权重 + 验证式混合采样(配额累加器/自适应k/"
               "接受率门控) + mini-forest 弱学习器; 113 数据集 AvgRank 7.25 (第4)"),
    "ours_v2": ("Model.ours_v2",
               "v2: 平衡目标梯度权重 w_i=(1/n_c)·σ(-m_i) (每类总质量自动有界 "
               "[0.27,0.73], 无显式预算/裁剪/阈值) + 赤字驱动配额 "
               "(def_c=1-m̄_c, 累加器, 隐式调度) + 信任度分派生成 "
               "(种子π∝σ(-m)·τ, 步长λ·τ); 5fold 协议, 关注 GMean"),
    "ours_v3": ("Model.ours_v3",
               "v3: 干净超球统一全链路 — 每个样本的干净半径 "
               "r_i=min_{j:y_j≠y_i}‖x_i-x_j‖ 驱动: 权重 w_i=σ(-m_i)·r_i/C_c "
               "(类干净空间占比) + 配额接受率门控 η_c (中点球验证, 嵌入类自限) "
               "+ 生成 (种子π∝w_i, 干净对中点球验证, 无干净对tiny类按干净半径"
               "复制); 移除信任度概念"),
    "ours_v4": ("Model.ours_v4",
               "v4: v3 + sqrt 干净份额权重 w_i=σ(-m_i)·√r_i/Σ√r_j — 边际梯度"
               "恢复主导 (AUC 恢复), 超球因子降半幂; 类级归一化保持质量有界"),
    # Research iteration chain; each module is retained for reproducibility.
    "ours_v1_2": ("Model.ours_v1_2", "V1.2 margin hardness"),
    "ours_v1_3": ("Model.ours_v1_3", "V1.3 density-aware pacing"),
    "ours_v1_4": ("Model.ours_v1_4", "V1.4 inverse-frequency cost sensitivity"),
    "ours_v1_5": ("Model.ours_v1_5", "V1.5 entropy hardness"),
    "ours_v1_6": ("Model.ours_v1_6", "V1.6 density-cost coupling"),
    "ours_v1_7": ("Model.ours_v1_7", "V1.7 boundary pacing"),
    "ours_v1_8": ("Model.ours_v1_8", "V1.8 shallow-tree ablation"),
    "ours_v1_9": ("Model.ours_v1_9", "V1.9 ExtraTrees weak learner"),
    "ours_v1_10": ("Model.ours_v1_10", "V1.10 density-cost final V1"),
    "ours_v2_1": ("Model.ours_v2_1", "V2.1 bootstrap diversity"),
    "ours_v2_2": ("Model.ours_v2_2", "V2.2 calibrated prior"),
    "ours_v2_3": ("Model.ours_v2_3", "V2.3 SMOM-style margin focus"),
    "ours_v2_4": ("Model.ours_v2_4", "V2.4 SOUP-style diversity"),
    "ours_v2_5": ("Model.ours_v2_5", "V2.5 DEAHS-style adaptive pace"),
    "ours_v2_6": ("Model.ours_v2_6", "V2.6 SHSampler boundary quota"),
    "ours_v2_7": ("Model.ours_v2_7", "V2.7 focal boosting"),
    "ours_v2_8": ("Model.ours_v2_8", "V2.8 temperature aggregation"),
    "ours_v2_9": ("Model.ours_v2_9", "V2.9 robust ExtraTrees"),
    "ours_v2_10": ("Model.ours_v2_10", "V2.10 SOUP-style diverse SPE"),
    "ours_v2_final": ("Model.ours_v2_final", "V2 final selected composition"),
}

def discover_models():
    """Scan Model/*.py (+ VERSION_REGISTRY), import each module, and return
    those exposing build().

    Returns:
        dict: {model_key: module}  where model_key is the module's MODEL_KEY attr.
        Also sets a ``_module_name`` attr on each value for subprocess use.
    """
    model_dir = os.path.join(PROJECT_ROOT, "Model")
    avail = {}
    if not os.path.isdir(model_dir):
        print("[run_all] WARNING: Model/ directory not found")
        return avail

    mod_names = sorted(fname[:-3] for fname in os.listdir(model_dir)
                       if fname.endswith(".py") and not fname.startswith("_"))
    for key, (mod_name, _desc) in VERSION_REGISTRY.items():
        if mod_name.startswith("Model."):
            mod_name = mod_name[len("Model."):]
        if mod_name not in mod_names:
            print(f"[run_all] WARNING: registry entry '{key}' -> {mod_name} "
                  f"not found in Model/; skipped")
        else:
            mod_names.append(mod_name)

    for mod_name in dict.fromkeys(mod_names):   # dict.fromkeys dedups, keeps order
        try:
            mod = importlib.import_module(f"Model.{mod_name}")
        except Exception as e:
            print(f"[run_all] skip '{mod_name}' (import failed: {type(e).__name__}: {e})")
            continue
        if not hasattr(mod, "build"):
            print(f"[run_all] skip '{mod_name}' (no build() yet)")
            continue
        model_key = getattr(mod, "MODEL_KEY", mod_name)
        mod._module_name = f"Model.{mod_name}"   # for subprocess re-import
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
# Subprocess isolation — each model×dataset cell runs in a child process so a
# C-level crash (segfault) kills only that cell, not the whole benchmark.
#
# We use ``subprocess.run`` (not ``multiprocessing``) to avoid any risk of
# multiprocessing-internal state corruption leaking into the main process.
# The child is a standalone ``_run_cell.py`` script that prints one JSON line
# to stdout and exits.  The main process only does lightweight bookkeeping.
# =============================================================================

CELL_TIMEOUT = 3600  # max seconds per model×dataset cell (1 hour)


def _run_cell_isolated(model_key, module_name, desc, n_folds, paper_name,
                       model_params, smoke, base_seed, save_cm, cm_json_dir,
                       timeout=CELL_TIMEOUT):
    """Spawn a child process for one model×dataset cell; return (status, data).

    Uses ``subprocess.run`` to invoke ``_run_cell.py`` — a standalone script
    that loads data, trains, evaluates, and prints one JSON result line.
    This approach keeps the main process completely free of heavy C-extension
    usage (numpy/scipy/sklearn), so it cannot segfault itself.

    Returns:
        ('success', metrics_dict)  on success
        ('error', error_message)   on failure / timeout / crash
    """
    desc_json = json.dumps(asdict(desc), ensure_ascii=False)
    model_params_json = json.dumps(model_params, ensure_ascii=False)

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "_run_cell.py"),
        "--model-key", model_key,
        "--module-name", module_name,
        "--dataset-json", desc_json,
        "--n-folds", str(n_folds),
        "--paper-name", paper_name,
        "--base-seed", str(base_seed),
        "--model-params-json", model_params_json,
    ]
    if smoke:
        cmd.append("--smoke")
    if save_cm and cm_json_dir:
        cmd.extend(["--save-cm", "--cm-json-dir", cm_json_dir])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )
    except subprocess.TimeoutExpired:
        return ('error', f'Timeout after {timeout}s')
    except Exception as e:
        return ('error', f'Subprocess launch failed: {e}')

    # Non-zero exit code → C-level crash (segfault)
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-200:]
        return ('error', f'Process crashed (exit {proc.returncode})'
                f'{": " + stderr_tail if stderr_tail else ""}'
                f' — likely segfault')

    # Parse JSON result from stdout
    stdout = (proc.stdout or "").strip()
    if not stdout:
        return ('error', 'No output from subprocess')

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return ('error', f'Invalid JSON from subprocess: {stdout[:200]}')

    if result.get("status") == "success":
        # Remove status key and return metrics
        result.pop("status", None)
        return ('success', result)
    else:
        return ('error', result.get("error", "Unknown error"))


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

    # --- keel layout: '5fold' (pre-split folds, 5-CV) | 'complete' (whole
    # datasets, single deterministic holdout split) ---
    keel_folds = cli.get("keel_folds") or yaml_cfg.get("keel_folds", "5fold")
    if keel_folds not in ("5fold", "complete"):
        raise ValueError(f"keel_folds must be '5fold' or 'complete', got {keel_folds!r}")

    # --- datasets (KEEL), bearing, software_defect, heart ---
    # Strict: [] really means "none"; missing key means "all".
    config_datasets = _resolve_list(yaml_cfg, "datasets", cli.get("datasets"))

    bearing_cfg = yaml_cfg.get("bearing") or {}
    config_bearing_names = _resolve_list(bearing_cfg, "names", cli.get("bearing_names"))
    config_irs        = _resolve_list(bearing_cfg, "irs",   cli.get("irs"))

    sdp_cfg = yaml_cfg.get("software_defect") or {}
    config_sdp_names = _resolve_list(sdp_cfg, "names", cli.get("sdp_names"))

    heart_cfg = yaml_cfg.get("heart") or {}
    config_heart_names = _resolve_list(heart_cfg, "names", cli.get("heart_names"))

    # --- folds ---
    n_folds = cli.get("n_folds") or yaml_cfg.get("n_folds") or (1 if smoke else 5)
    if keel_folds == "complete" and cli.get("n_folds") is None:
        n_folds = 1          # complete datasets have exactly one holdout split

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

    # --- isolation ---
    no_isolation = cli.get("no_isolation", False)

    return {
        "smoke": smoke,
        "mode": mode,
        "keel_folds": keel_folds,             # '5fold' | 'complete'
        "models": config_models,              # None=all, []=all (convenience), [...]=only
        "datasets": config_datasets,          # None=all, []=none, [...]=only
        "bearing_names": config_bearing_names,# None=all, []=none, [...]=only
        "irs": tuple(config_irs) if config_irs else (),
        "sdp_names": config_sdp_names,         # None=all, []=none, [...]=only
        "heart_names": config_heart_names,    # None=all, []=none, [...]=only
        "n_folds": n_folds,
        "save_cm": save_cm,
        "out_file": out_file,
        "base_seed": base_seed,
        "model_params": model_params,
        "resume": resume,
        "no_isolation": no_isolation,
    }


# =============================================================================
# Atomic write helpers
# =============================================================================

# =============================================================================
# Save helpers — CSV is the durable mid-run state (avoids openpyxl C-ext
# crashes in the main process).  It is only an internal working file and is
# removed once the final xlsx has been written.
# =============================================================================

_SAVE_BATCH_SIZE = 5


def _save_rows_csv(rows, path):
    """Save rows to CSV (stable, no C extensions)."""
    tmp = path + ".tmp"
    df = pd.DataFrame(rows)
    df.to_csv(tmp, index=False, encoding="utf-8")
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    os.replace(tmp, path)


def _load_csv(path):
    """Load saved rows from CSV. Returns list of dicts."""
    df = pd.read_csv(path, encoding="utf-8")
    return df.to_dict("records")


def _csv_to_xlsx(csv_path, xlsx_path):
    """Convert final CSV to xlsx."""
    import openpyxl
    from openpyxl.utils.dataframe import dataframe_to_rows
    df = pd.read_csv(csv_path, encoding="utf-8")
    tmp = xlsx_path + ".tmp"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)
    wb.save(tmp)
    if os.path.exists(xlsx_path):
        try:
            os.remove(xlsx_path)
        except OSError:
            pass
    os.replace(tmp, xlsx_path)
    print(f"[run_all] xlsx written to {os.path.basename(xlsx_path)}")


# =============================================================================
# Main runner
# =============================================================================

def run_benchmark(config):
    """Run the benchmark. Returns the results DataFrame."""
    smoke = config["smoke"]
    mode = config["mode"]
    no_isolation = config["no_isolation"]
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
    sdp_names     = config["sdp_names"]
    heart_names   = config["heart_names"]

    # --- smoke mode: if nothing is configured, provide a sensible default ---
    all_empty = (
        (datasets is None or len(datasets) == 0) and
        (bearing_names is None or len(bearing_names) == 0) and
        (sdp_names is None or len(sdp_names) == 0) and
        (heart_names is None or len(heart_names) == 0)
    )
    if smoke and all_empty:
        datasets = ["ecoli1"]
        bearing_names = []
        irs = ()
        sdp_names = []
        heart_names = []
        n_folds = 5

    # --- discover datasets ---
    keel_root = (paths.KEEL_COMPLETE if config["keel_folds"] == "complete"
                 else paths.KEEL_5FOLD)
    ds_list = discover_all_datasets(
        keel_filter=_filter_set(datasets),
        bearing_irs=irs if irs else (5, 10, 20, 30),
        bearing_names=_filter_set(bearing_names),
        sdp_names=_filter_set(sdp_names),
        heart_names=_filter_set(heart_names),
        keel_root=keel_root)
    if not ds_list:
        raise FileNotFoundError("no datasets found for the given filters")

    # --- resolve output paths ---
    out_file = config["out_file"]
    if out_file is None:
        base = "Smoke_Results" if smoke else "benchmark_results"
    else:
        base = out_file.replace(".xlsx", "").replace(".csv", "")
    csv_path = os.path.join(paths.RESULTS_DIR, base + ".csv")
    xlsx_path = os.path.join(paths.RESULTS_DIR, base + ".xlsx")
    paths.ensure_dirs()

    # --- always load existing results and continue appending to them ---
    # (CSV preferred; fallback to xlsx).  Cells already present are skipped
    # regardless of --resume, so a re-run never overwrites/rebuilds the file
    # from scratch — old rows are merged back in at the final save.
    rows = []
    completed = set()
    load_path = None
    if csv_path and os.path.isfile(csv_path):
        load_path = csv_path
    elif os.path.isfile(xlsx_path):
        load_path = xlsx_path

    if load_path:
        try:
            if load_path.endswith(".csv"):
                existing_df = pd.read_csv(load_path, encoding="utf-8")
            else:
                existing_df = pd.read_excel(load_path, sheet_name="Results",
                                            engine="openpyxl")
            rows = existing_df.to_dict("records")
            for row in rows:
                m = str(row.get("Model", ""))
                d = str(row.get("Dataset", ""))
                # error rows are NOT "completed" — they get retried next run.
                # NB: an empty Error column becomes NaN after xlsx conversion,
                # and `not NaN` is False, so NaN must be treated as "no error".
                err = row.get("Error")
                has_err = not (err is None or pd.isna(err)
                               or str(err).strip() == "")
                if m and d and not has_err:
                    completed.add((m, d))
            print(f"[run_all] loaded {len(completed)} existing cells from "
                  f"{os.path.basename(load_path)}")
        except Exception as e:
            print(f"[run_all] could not read existing file ({e}); starting fresh")

    # --- report (existing counts only cells inside the current grid, so
    # 'new' can never go negative from other models' / datasets' rows) ---
    sel_models = {paper_names.get(k, k) for k in avail}
    sel_datasets = {ds.name for ds in ds_list}
    existing_in_grid = sum(1 for (m, d) in completed
                           if m in sel_models and d in sel_datasets)
    total_cells = len(ds_list) * len(avail)
    new_cells = total_cells - existing_in_grid
    print(f"\n{'=' * 94}\n"
          f" run_all [{mode.upper()}]  models={len(avail)}  datasets={len(ds_list)}"
          f"  n_folds={n_folds}\n"
          f" existing={existing_in_grid}  new={new_cells}  total={total_cells}\n"
          f" save={csv_path}\n"
          f"{'=' * 94}")

    # Confusion-matrix JSON dir
    cm_json_dir = os.path.join(paths.CM_JSON_DIR) if save_cm else None
    if cm_json_dir:
        os.makedirs(cm_json_dir, exist_ok=True)

    # --- traversal order ---
    # models_outer (used for the complete protocol): iterate models in speed
    # order × datasets, so the fast models finish all datasets first and a
    # partial ranking exists early.  Cells are identical either way (resume
    # still skips completed ones).
    # Approximation of per-dataset fit cost (seconds, complete protocol);
    # unknown models are treated as fast so they never block the rest.
    _MODEL_SPEED = {
        "AdaBoostAD": 0.1, "LexiBoost": 0.2, "DualLexiBoost": 1.0,
        "glos": 0.1, "mc_ccr": 0.1, "mdo": 0.1, "shsampler": 0.1,
        "Ours": 2.0, "ILMNN": 0.4, "NROMM": 2.6, "FRAME": 6.0,
        "EB-SMOTE": 0.6, "OREM-M": 0.2, "SOUP": 9.0, "MC-RBO": 12.0,
        "QC-SMOTE": 12.0, "SPE": 11.0, "DBCF": 15.0, "imDEF": 14.0,
        "SMOM": 26.0, "DEAHS": 80.0,
    }
    models_outer = config.get("keel_folds") == "complete"
    iter_pairs = []
    for ds in ds_list:
        # Pre-flight: verify the dataset can actually be loaded
        try:
            _, classes_pre = load_folds(ds)
        except Exception as e:
            print(f"\n--- LOAD FAIL {ds.name}: {e}")
            continue
        for key, mod in avail.items():
            iter_pairs.append((ds, key, mod, len(classes_pre)))
    if models_outer:
        iter_pairs.sort(key=lambda p: (_MODEL_SPEED.get(p[1], 0.1), p[1]))
    else:
        iter_pairs.sort(key=lambda p: (p[0].name, p[1]))
    n_cells_done = 0

    # --- main loop (each model×dataset can run in an isolated subprocess) ---
    for ds, key, mod, n_classes in iter_pairs:
        paper_name = paper_names.get(key, key)
        module_name = getattr(mod, "_module_name", f"Model.{key}")

        if (paper_name, ds.name) in completed:
            print(f"\n--- skip {paper_name:12s} | {ds.name} (already in xlsx) ---",
                  flush=True)
            continue

        t0 = time.time()
        tag = "[direct]" if no_isolation else "[isolated]"
        print(f"\n--- {paper_name:12s} | {ds.name} (C={n_classes})"
              f" {tag} ---", flush=True)

        if no_isolation:
            # --------------------------------------------------------------
            # In-process path (original behaviour; for debugging)
            # --------------------------------------------------------------
            try:
                folds, classes = load_folds(ds)
                folds_run = folds[:n_folds]
                extra_params = model_params.get(key, {})
                factory = (lambda m, ep: (lambda seed: m.build(
                    random_state=seed, smoke=smoke, **ep)))(mod, extra_params)
                result = run_model_on_dataset(
                    factory, ds, folds_run, classes,
                    model_key=paper_name, base_seed=base_seed,
                    save_cm=save_cm, cm_json_dir=cm_json_dir, verbose=smoke)
                status, result_or_err = 'success', result
            except Exception as e:
                status, result_or_err = 'error', str(e)
        else:
            # --------------------------------------------------------------
            # Subprocess isolation: a C-level crash kills only this cell
            # --------------------------------------------------------------
            status, result_or_err = _run_cell_isolated(
                model_key=key,
                module_name=module_name,
                desc=ds,
                n_folds=n_folds,
                paper_name=paper_name,
                model_params=model_params,
                smoke=smoke,
                base_seed=base_seed,
                save_cm=save_cm,
                cm_json_dir=cm_json_dir,
            )

        elapsed = time.time() - t0

        if status == 'success':
            row = {
                "Model": paper_name,
                "Dataset": ds.name,
                "IR": (f"IR{ds.ir}" if ds.source == "bearing" else "-"),
                "Source": ds.source,
            }
            row.update(result_or_err)
            # drop any stale error row for this cell (now retried) so the
            # file stays deduped
            rows = [r for r in rows
                    if not (str(r.get("Model", "")) == paper_name
                            and str(r.get("Dataset", "")) == ds.name)]
            metric_str = ""
            if "Accuracy_mean" in result_or_err:
                metric_str = (f"Acc={result_or_err['Accuracy_mean']:.3f} "
                              f"F1={result_or_err['F1_mean']:.3f} "
                              f"GMean={result_or_err['GMean_mean']:.3f}")
            print(f"    -> done in {elapsed:.1f}s ({metric_str})",
                  flush=True)
        else:
            print(f"    -> CRASH/ERROR: {result_or_err}", flush=True)
            traceback.print_exc()
            row = {
                "Model": paper_name,
                "Dataset": ds.name,
                "IR": (f"IR{ds.ir}" if ds.source == "bearing" else "-"),
                "Source": ds.source,
                "Error": str(result_or_err),
            }

        rows.append(row)
        if status == 'success':
            completed.add((paper_name, ds.name))

        # --- release memory ---
        gc.collect()

        # --- periodic save to CSV (stable, no C extensions) ---
        n_cells_done += 1
        if n_cells_done % 25 == 0:
            _save_rows_csv(rows, csv_path)
            print(f"    [save] {len(completed)}/{total_cells} cells -> "
                  f"{os.path.basename(csv_path)}", flush=True)

    # --- final CSV save + xlsx conversion ---
    _save_rows_csv(rows, csv_path)
    print(f"\n{'=' * 94}\n{len(rows)} cells total -> {os.path.basename(csv_path)}"
          f"\n{'=' * 94}")
    # Convert to xlsx and drop the interim CSV — the xlsx is the only deliverable
    try:
        _csv_to_xlsx(csv_path, xlsx_path)
        try:
            os.remove(csv_path)
            print(f"[run_all] interim {os.path.basename(csv_path)} removed "
                  f"(xlsx is the only deliverable)")
        except OSError:
            pass  # leftover CSV is harmless — next run re-reads and merges it
    except Exception as e:
        print(f"[run_all] xlsx conversion failed ({e}); CSV kept at {csv_path}")
    return pd.DataFrame(rows)


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
    ap.add_argument("--keel-folds", default=None, choices=["5fold", "complete"],
                    help="KEEL layout: '5fold' (pre-split 5-CV) or "
                         "'complete' (whole datasets, single holdout)")
    ap.add_argument("--bearing-names", default=None,
                    help="Comma list of Bearing names (overrides config)")
    ap.add_argument("--irs", default=None,
                    help="Comma list of Bearing IRs, e.g. 5,20 (overrides config)")
    ap.add_argument("--sdp-names", default=None,
                    help="Comma list of Software Defect dataset names, e.g. AR,KC (overrides config)")
    ap.add_argument("--heart-names", default=None,
                    help="Comma list of Heart dataset names (overrides config)")
    ap.add_argument("--n-folds", type=int, default=None,
                    help="Folds per dataset (overrides config)")
    ap.add_argument("--out", default=None,
                    help="Output xlsx path (overrides config)")
    ap.add_argument("--resume", action="store_true",
                    help="(deprecated) resume is automatic — existing cells "
                         "are always loaded and skipped")
    ap.add_argument("--no-isolation", action="store_true",
                    help="Run cells in-process instead of isolated subprocesses")
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
        "keel_folds": args.keel_folds,
        "bearing_names": ([x.strip() for x in args.bearing_names.split(",")]
                          if args.bearing_names else None),
        "irs": (tuple(int(x) for x in args.irs.split(","))
                if args.irs else None),
        "sdp_names": ([x.strip() for x in args.sdp_names.split(",")]
                       if args.sdp_names else None),
        "heart_names": ([x.strip() for x in args.heart_names.split(",")]
                        if args.heart_names else None),
        "n_folds": args.n_folds,
        "out": args.out,
        "resume": args.resume,
        "no_isolation": args.no_isolation,
    }

    config = load_config(args.config, cli)
    run_benchmark(config)


if __name__ == "__main__":
    main()
