# -*- coding: utf-8 -*-
"""rank_complete.py — 从 complete 协议的结果文件计算均值/排名, 并与 5fold 旧排名对照。

数据源:
  * Result/benchmark_complete.csv(xlsx)  — complete 协议主结果 (run_all.py 输出)
  * Result/benchmark_ours_vN.xlsx        — 每个 ours 版本的独立结果文件 (可选)

规则 (与 export_results.py 一致):
  * 每个模型必须在其所在集合的"全部数据集"上有有效结果 (指标非 NaN 且无 Error),
    否则不参与均值与排名 (用于电池未跑完时的部分排名)。
  * F1/GMean/AUC/AUPRC 越大越好; Avg_Rank_excl_Runtime = 四项排名的平均。

Usage:
    python rank_complete.py [--extra ours_v2.xlsx ours_v3.xlsx ...]
"""
import os
import sys
import argparse

for _tv in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]:
    os.environ[_tv] = "1"

import pandas as pd

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Result")
METRIC_COLS = ["F1_mean", "GMean_mean", "AUC_mean", "AUPRC_mean", "Runtime_mean"]

# ---------------------------------------------------------------------------
# 基准模型均值表: 来自 Result/KEEL_Result/Ranking_KEEL.xlsx "Mean Values" sheet
# (用户指示: 基线模型已有结果, 直接用这些均值对比, 不再重跑基线)
# ---------------------------------------------------------------------------
BASELINE_MEANS = {
    "imDEF":       (0.826620, 0.785904, 0.924965, 0.757964),
    "GLOS":        (0.818394, 0.738539, 0.913981, 0.741400),
    "SMOM":        (0.816839, 0.739864, 0.914412, 0.737223),
    "ILMNN":       (0.819072, 0.761859, 0.910727, 0.733174),
    "OREM-M":      (0.817979, 0.742719, 0.913280, 0.731980),
    "SHSampler":   (0.805404, 0.804485, 0.915334, 0.730297),
    "QC-SMOTE":    (0.813756, 0.716018, 0.913969, 0.742386),
    "MDO":         (0.810412, 0.717436, 0.913575, 0.744960),
    "SOUP":        (0.814191, 0.752670, 0.913441, 0.731857),
    "DBCF":        (0.806212, 0.731196, 0.914624, 0.735421),
    "NROMM":       (0.816622, 0.729635, 0.911864, 0.736195),
    "FRAME":       (0.817877, 0.734125, 0.899494, 0.733334),
    "MC-CCR":      (0.812355, 0.709753, 0.910447, 0.742792),
    "DEAHS":       (0.755338, 0.846988, 0.911965, 0.725536),
    "MC-RBO":      (0.813466, 0.709309, 0.910173, 0.740252),
    "SPE":         (0.741054, 0.834540, 0.902776, 0.711464),
    "EB-SMOTE":    (0.796866, 0.675642, 0.907739, 0.726652),
    "DualLexiBoost": (0.782815, 0.717674, 0.793031, 0.536187),
    "Ours":        (0.662080, 0.461101, 0.849311, 0.619209),
    "LexiBoost":   (0.681185, 0.617157, 0.806734, 0.474475),
    "AdaBoostAD":  (0.585638, 0.672716, 0.808345, 0.469757),
}


def load_source(path):
    if str(path).endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8")


def compute(sub_df, label, n_baselines=21):
    """Rank the models in sub_df against the FIXED baseline means.

    规则: 基线 21 个模型用 Ranking_KEEL.xlsx 的固定均值; 文件中出现的其他模型
    (ours_vN 等) 用其在本文件 113 个数据集上的均值; 每个模型必须覆盖全部 113 个
    数据集才参与排名。
    """
    required = set(sub_df["Dataset"].unique())
    fail = sub_df[METRIC_COLS].isna().any(axis=1)
    if "Error" in sub_df.columns:
        fail = fail | (sub_df["Error"].fillna("").astype(str).str.strip() != "")
    valid = sub_df[~fail]

    rows = []
    excluded = []
    for model, g in valid.groupby("Model"):
        covered = set(g["Dataset"])
        missing = required - covered
        if missing:
            excluded.append((model, len(required), len(covered), len(missing)))
            continue
        means = g[METRIC_COLS].mean()
        rows.append({"Model": model, "F1_mean": means["F1_mean"],
                     "GMean_mean": means["GMean_mean"],
                     "AUC_mean": means["AUC_mean"],
                     "AUPRC_mean": means["AUPRC_mean"],
                     "Runtime_mean": means["Runtime_mean"]})
    # 固定基线均值 (来自 Ranking_KEEL.xlsx "Mean Values")
    for name, (f1, gm, auc, auprc) in BASELINE_MEANS.items():
        if name not in {r["Model"] for r in rows}:
            rows.append({"Model": name, "F1_mean": f1, "GMean_mean": gm,
                         "AUC_mean": auc, "AUPRC_mean": auprc,
                         "Runtime_mean": float("nan")})

    print(f"\n{'='*100}\n{label}")
    print(f"数据集数: {len(required)}  参与排名模型: {len(rows)}  "
          f"文件内未完整(不参与): {len(excluded)}")
    for m, req, cov, mis in sorted(excluded):
        print(f"  - {m}: 需 {req} 个数据集, 已有 {cov}, 缺 {mis}")
    if len(rows) < 2:
        print("模型不足, 无法排名")
        return None
    mv = pd.DataFrame(rows)
    for m in METRIC_COLS[:4]:
        mv[f"Rank_{m}"] = mv[m].rank(ascending=False, method="average")
    mv["Avg_Rank_excl_Runtime"] = mv[[f"Rank_{m}" for m in METRIC_COLS[:4]]].mean(axis=1)
    mv["Avg_Rank_incl_Runtime"] = mv[[f"Rank_{m}" for m in METRIC_COLS[:4]]].mean(axis=1)
    if mv["Runtime_mean"].notna().sum() > 1:
        mv["Rank_Runtime_mean"] = mv["Runtime_mean"].rank(ascending=True, method="average")
        mv["Avg_Rank_incl_Runtime"] = mv[[f"Rank_{m}" for m in METRIC_COLS[:4]] + ["Rank_Runtime_mean"]].mean(axis=1)
    mv = mv.sort_values("Avg_Rank_excl_Runtime").reset_index(drop=True)
    print(f"\n{'Model':18s} {'F1':>6s} {'GMean':>6s} {'AUC':>6s} {'AUPRC':>6s} "
          f"{'AvgRank':>8s}  {'F1rk':>4s} {'GMrk':>4s} {'AUCrk':>4s} {'APrk':>4s}")
    for _, r in mv.iterrows():
        print(f"{r['Model']:18s} {r['F1_mean']:6.3f} {r['GMean_mean']:6.3f} "
              f"{r['AUC_mean']:6.3f} {r['AUPRC_mean']:6.3f} "
              f"{r['Avg_Rank_excl_Runtime']:8.2f}  "
              f"{r['Rank_F1_mean']:4.0f} {r['Rank_GMean_mean']:4.0f} "
              f"{r['Rank_AUC_mean']:4.0f} {r['Rank_AUPRC_mean']:4.0f}")
    return mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=None)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="保存完整排名 xlsx")
    args = ap.parse_args()

    # prefer the durable CSV over the (possibly stale) xlsx
    if args.main is None:
        csv_p = os.path.join(RESULT_DIR, "benchmark_complete.csv")
        args.main = csv_p if os.path.isfile(csv_p) else os.path.join(
            RESULT_DIR, "benchmark_complete.xlsx")
    frames = [load_source(args.main)]
    for e in args.extra:
        p = e if os.path.isabs(e) else os.path.join(RESULT_DIR, e)
        frames.append(load_source(p))
    df = pd.concat(frames, ignore_index=True)
    df["Dataset"] = df["Dataset"].astype(str)
    df["Model"] = df["Model"].astype(str)
    compute(df, f"对比排名 ({os.path.basename(args.main)} + {len(args.extra)} 个额外文件)")
    if args.out:
        df.to_excel(args.out, index=False)
        print(f"\n合并明细已保存: {args.out}")


if __name__ == "__main__":
    main()
