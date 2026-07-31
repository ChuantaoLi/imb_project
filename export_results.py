# -*- coding: utf-8 -*-
"""export_results.py — 结果整理脚本: 切分 benchmark_results.xlsx + 生成均值/排名表。

数据源: Result/benchmark_results.xlsx (run_all.py 的输出)

输出:
  Result/Ranking_Overall.xlsx            总体 Mean Values + Mean Ranking + Notes
  Result/KEEL_Result/Result_KEEL.xlsx    按数据集类型切分的结果明细
  Result/KEEL_Result/Ranking_KEEL.xlsx   该类型下的均值/排名
  Result/Bearing_Result_{5,10,20,30}/... 轴承按 IR 再细分
  Result/Heart_Result/...
  Result/Software_Result/...

规则:
  * Mean Values: 每个模型对该组全部数据集取 *_mean 的算术平均
    (Avg_F1_mean / Avg_GMean_mean / Avg_AUC_mean / Avg_AUPRC_mean / Avg_Runtime_mean)。
  * Mean Ranking: F1/GMean/AUC/AUPRC 越大越好、Runtime 越小越好, 并列取平均名次;
    Avg_Rank_excl_Runtime = 前 4 项排名的平均, Avg_Rank_incl_Runtime = 含 Runtime 5 项全平均。
  * 完整性: 模型必须在组内"全部数据集"上都有有效结果(指标非 NaN、无 Error), 否则
    不参与均值与排名, 仅在 Model 列保留模型名称, 缺失情况写入 Notes sheet。
  * 模型名规范化: 同一模型在不同批次运行中可能以论文名或小写 MODEL_KEY 写入,
    统一映射到论文名形式(见 MODEL_NAME_NORMALIZE), 避免同一模型被拆成两个。

Usage:
    python export_results.py                      # 默认读 Result/benchmark_results.xlsx
    python export_results.py --source xxx.xlsx    # 指定数据源
"""
import os
import re
import argparse

# ---- 线程限制(必须在任何 numpy/pandas import 之前设置) ----
# 与 run_all.py 一致: 避免 BLAS/OpenMP 线程竞争导致 Segmentation fault (core dumped)
for _tv in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]:
    os.environ[_tv] = "1"

import pandas as pd
import openpyxl
from openpyxl.styles import Font

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(PROJECT_ROOT, "Result")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
METRIC_COLS = ["F1_mean", "GMean_mean", "AUC_mean", "AUPRC_mean", "Runtime_mean"]
AVG_HEADERS = ["Avg_F1_mean", "Avg_GMean_mean", "Avg_AUC_mean", "Avg_AUPRC_mean",
               "Avg_Runtime_mean"]
RANK_HEADERS = ["Rank_F1_mean", "Rank_GMean_mean", "Rank_AUC_mean", "Rank_AUPRC_mean",
                "Rank_Runtime_mean"]
RANK_EXCL_HEADERS = ["Avg_Rank_excl_Runtime", "Avg_Rank_incl_Runtime"]
# 各指标排名方向: higher = 越大越好(名次1为最佳), lower = 越小越好
RANK_DIRECTION = {"F1_mean": "higher", "GMean_mean": "higher", "AUC_mean": "higher",
                  "AUPRC_mean": "higher", "Runtime_mean": "lower"}

# 同一模型两种写入名称 → 统一为论文名形式
MODEL_NAME_NORMALIZE = {
    "glos": "GLOS", "mc_ccr": "MC-CCR", "mc_rbo": "MC-RBO", "mdo": "MDO",
    "nromm": "NROMM", "orem_m": "OREM-M", "shsampler": "SHSampler",
    "smom": "SMOM", "soup": "SOUP",
}

BEARING_IR_RE = re.compile(r"^IR(\d+)_")

# 数据源 → (输出文件夹, Result 文件名, Ranking 文件名); bearing 特殊: 按 IR 细分
TYPE_SPEC = {
    "keel":            ("KEEL_Result",     "Result_KEEL",     "Ranking_KEEL"),
    "heart":           ("Heart_Result",    "Result_Heart",    "Ranking_Heart"),
    "software_defect": ("Software_Result", "Result_Software", "Ranking_Software"),
}

# 表格样式(与现有 Ranking_*.xlsx 保持一致)
HEADER_FONT = Font(name="Times New Roman", size=11, bold=True)
BODY_FONT = Font(name="Times New Roman", size=11)
MV_NUM_FMT = {i: "0.000_ " for i in range(2, 7)}        # Mean Values 数据列 B..F
MR_NUM_FMT = {2: "0.0_ ", 3: "0.0_ "}                    # Mean Ranking: 两列 Avg_Rank


def load_results(path):
    """读数据源, 规范化模型名。返回 (df, 规范化统计列表)。"""
    df = pd.read_excel(path)
    need = ["Model", "Dataset", "Source"] + METRIC_COLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"[export_results] 数据源缺少列: {missing}")
    df = df.copy()
    df["_orig_model"] = df["Model"]          # 保留原名, 用于 Notes 说明
    df["Model"] = df["Model"].replace(MODEL_NAME_NORMALIZE)
    return df


def metric_failed_mask(df):
    """标记"未跑出结果"的行: 任一指标 NaN 或 Error 列非空。"""
    fail = df[METRIC_COLS].isna().any(axis=1)
    if "Error" in df.columns:
        err = df["Error"].fillna("").astype(str).str.strip()
        fail = fail | (err != "")
    return fail


def split_groups(df):
    """按 Source 切分, 返回 [(group_key, sub_df), ...]; 总体在第一位。"""
    groups = [("Overall", df)]
    for src, g in df.groupby("Source", dropna=False):
        if src == "bearing":
            g = g.copy()
            g["_ir"] = g["Dataset"].map(
                lambda n: (m.group(1) if (m := BEARING_IR_RE.match(str(n))) else None))
            for ir, gg in g.groupby("_ir", dropna=False):
                key = f"Bearing_{ir}" if ir else "Bearing_other"
                groups.append((key, gg.drop(columns="_ir")))
        else:
            groups.append((str(src), g))    # 与 TYPE_SPEC 的键一致: keel / heart / software_defect
    return groups


def _missing_by_source(df, missing):
    """缺失数据集按 Source 计数 → '32 个 bearing, 4 个 heart' 文本。"""
    src_of = df.drop_duplicates("Dataset").set_index("Dataset")["Source"]
    cnt = src_of[src_of.index.isin(missing)].value_counts().to_dict()
    return ", ".join(f"{n} 个 {s}" for s, n in cnt.items())


def compute_rankings(sub_df):
    """计算一个组(总体或某类型)的均值/排名。

    返回 (mv_df, rank_df, notes_df):
      mv_df    每模型一行; 完整模型含均值, 不完整模型仅 Model 列。
      rank_df  完整模型的各项排名 + 平均排名; 不完整模型仅 Model 列。
      notes_df Notes sheet 内容(排除说明 + 失败单元 + 名称规范化)。
    """
    required = set(sub_df["Dataset"].unique())
    fail = metric_failed_mask(sub_df)
    valid = sub_df[~fail]
    failed_cells = sub_df.loc[fail, ["_orig_model", "Dataset"]].drop_duplicates()

    complete, excluded = [], []
    for model, g in valid.groupby("Model"):
        covered = set(g["Dataset"])
        missing = required - covered
        if missing:
            excluded.append({
                "Model": model, "required": len(required), "covered": len(covered),
                "missing": len(missing), "detail": _missing_by_source(sub_df, missing),
            })
        else:
            complete.append({"Model": model, **g[METRIC_COLS].mean().to_dict()})

    mv = pd.DataFrame(complete, columns=["Model"] + METRIC_COLS)
    rank = pd.DataFrame(complete, columns=["Model"])
    if len(mv) > 1:
        for m, direction in RANK_DIRECTION.items():
            r = mv[m].rank(ascending=(direction == "lower"), method="average")
            if (r == r.round()).all():
                r = r.astype(int)                    # 无并列时保持整数
            rank["Rank_" + m] = r.values
        rank[RANK_EXCL_HEADERS[0]] = rank[RANK_HEADERS[:4]].mean(axis=1)
        rank[RANK_EXCL_HEADERS[1]] = rank[RANK_HEADERS].mean(axis=1)
        rank = rank.sort_values(
            [RANK_EXCL_HEADERS[0], "Model"], key=lambda s: s.str.lower()
            if s.dtype == object else s)
        # 列顺序与旧文件一致: Model, Avg_Rank_excl_Runtime, Avg_Rank_incl_Runtime, Rank_*
        rank = rank[["Model", RANK_EXCL_HEADERS[0], RANK_EXCL_HEADERS[1]] + RANK_HEADERS]
    mv = mv.sort_values("Model", key=lambda s: s.str.lower())
    mv = mv.reset_index(drop=True)
    rank = rank.reset_index(drop=True)

    # ---- 不完整模型: 仅保留 Model 列, 追加到末尾 ----
    if excluded:
        names = sorted(e["Model"] for e in excluded)
        mv = pd.concat([mv, pd.DataFrame({"Model": names})], ignore_index=True)
        rank = pd.concat([rank, pd.DataFrame({"Model": names})], ignore_index=True)

    # ---- Notes ----
    notes = []
    if excluded:
        notes.append(("汇总", f"参与均值与排名的完整模型 {len(complete)} 个; "
                              f"数据集覆盖不全未参与 {len(excluded)} 个"))
        for e in sorted(excluded, key=lambda x: x["Model"].lower()):
            notes.append(("排除模型", f"{e['Model']}: 要求 {e['required']} 个数据集, "
                         f"有效结果 {e['covered']} 个, 缺失 {e['missing']} 个 "
                         f"({e['detail']}); 不参与均值与排名, 仅保留模型名"))
    elif complete:
        notes.append(("汇总", f"本组 {len(complete)} 个模型均在全部 "
                              f"{len(required)} 个数据集上跑出结果, 全部参与均值与排名"))
    for _, r in failed_cells.iterrows():
        notes.append(("失败单元", f"{r['_orig_model']} × {r['Dataset']}: "
                                  f"指标为 NaN/Error, 视为未跑出结果, 不参与聚合"))
    for new_name, old_counts in _rename_stats(sub_df).items():
        notes.append(("名称规范化",
                      f"模型名统一: {', '.join(f'{old}({n} 行)' for old, n in old_counts)} "
                      f"→ {new_name}"))
    notes_df = pd.DataFrame(notes, columns=["类别", "说明"])
    return mv, rank, notes_df


def _rename_stats(sub_df):
    """按 _orig_model 统计被规范化的行数 → {新名: [(原名, 行数), ...]}。"""
    renamed = sub_df.loc[sub_df["_orig_model"] != sub_df["Model"], ["_orig_model", "Model"]]
    out = {}
    for old, new in renamed.drop_duplicates().itertuples(index=False):
        out.setdefault(new, []).append((old, int((renamed["_orig_model"] == old).sum())))
    return out


# ---------------------------------------------------------------------------
# 写 Excel
# ---------------------------------------------------------------------------

def _style_sheet(ws, widths, num_fmt=None):
    """表头加粗 TNR 11, 数据 TNR 11, 数字格式与列宽。"""
    num_fmt = num_fmt or {}
    for cell in ws[1]:
        cell.font = HEADER_FONT
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if cell.value is not None:
                cell.font = BODY_FONT
                fmt = num_fmt.get(cell.column)
                if fmt and isinstance(cell.value, (int, float)):
                    cell.number_format = fmt
    for col, w in widths.items():
        ws.column_dimensions[col].width = w


def write_ranking_file(path, mv, rank, notes):
    """写 Ranking_<Type>.xlsx: Mean Values + Mean Ranking + Notes 三个 sheet。"""
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        mv.to_excel(xw, sheet_name="Mean Values", index=False)
        rank.to_excel(xw, sheet_name="Mean Ranking", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
    wb = openpyxl.load_workbook(path)
    _style_sheet(wb["Mean Values"], {"A": 18.625}, MV_NUM_FMT)
    # 排名列若出现平均名次(小数)则用 0.0 格式, 否则整数格式, 与旧文件一致
    # 只检查 5 个单项排名列, Avg_Rank 列本身就有小数
    has_frac = False
    if len(rank):
        vals = rank[RANK_HEADERS].astype(float).values
        has_frac = bool((vals % 1 > 0).any())
    rfmt = {i: ("0.0_ " if has_frac else "0_ ") for i in range(4, 9)}
    mr_fmt = {**MR_NUM_FMT, **rfmt}
    _style_sheet(wb["Mean Ranking"], {"A": 22.625}, mr_fmt)
    _style_sheet(wb["Notes"], {"A": 12, "B": 110})
    wb.save(path)


def write_result_file(path, sub_df):
    """写 Result_<Type>.xlsx: 切分出的明细行(原始列, 模型名已规范化)。"""
    cols = [c for c in sub_df.columns if not c.startswith("_")]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        sub_df[cols].to_excel(xw, sheet_name="Sheet1", index=False)
    wb = openpyxl.load_workbook(path)
    _style_sheet(wb["Sheet1"], {"A": 18.625})
    wb.save(path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="结果整理: 切分 + 均值 + 排名")
    ap.add_argument("--source", default=os.path.join(RESULT_DIR, "benchmark_results.xlsx"),
                    help="数据源 xlsx (默认 Result/benchmark_results.xlsx)")
    args = ap.parse_args()

    df = load_results(args.source)
    print(f"[export_results] 读取 {args.source}: {len(df)} 行, "
          f"{df['Model'].nunique()} 个模型(规范化后), {df['Dataset'].nunique()} 个数据集")

    for key, g in split_groups(df):
        if len(g) < 2:
            print(f"[export_results] 跳过 {key}: 数据不足")
            continue
        mv, rank, notes = compute_rankings(g)
        if key == "Overall":
            out_dir = RESULT_DIR
            rk_file = os.path.join(out_dir, "Ranking_Overall.xlsx")
        else:
            if key.startswith("Bearing_"):
                ir = key[len("Bearing_"):]
                folder, rk_stem = f"Bearing_Result_{ir}", f"Ranking_Bearing_{ir}"
                rs_stem = f"Result_Bearing_{ir}"
            else:
                folder, rs_stem, rk_stem = TYPE_SPEC[key]
            out_dir = os.path.join(RESULT_DIR, folder)
            os.makedirs(out_dir, exist_ok=True)
            write_result_file(os.path.join(out_dir, rs_stem + ".xlsx"), g)
            rk_file = os.path.join(out_dir, rk_stem + ".xlsx")
        write_ranking_file(rk_file, mv, rank, notes)
        n_excl = int((notes["类别"] == "排除模型").sum()) if len(notes) else 0
        n_rank = len(mv) - n_excl
        print(f"[export_results] {key:12s} 参与排名 {n_rank:2d} 个模型, "
              f"未参与 {n_excl} 个  -> {os.path.relpath(rk_file, RESULT_DIR)}")


if __name__ == "__main__":
    main()
