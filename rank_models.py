#!/usr/bin/env python3
"""
Rank models for each metric based on All_Models_Results.csv.

For each metric (mean columns), ranks models per dataset (1 = best),
then computes the average rank across all datasets.
Outputs:
  1. avg_rankings.csv          — average rank of each model per metric
  2. avg_rankings_per_dataset.csv — per-dataset rankings (pivoted)
  3. summary_top5.txt          — top-5 models for each metric (by avg rank)
  4. model_win_counts.csv      — how many datasets each model ranks #1 per metric

Scoring: higher-is-better for all metrics EXCEPT Runtime_mean (lower-is-better).
"""

import pandas as pd
import numpy as np
import os

# ── Config ──────────────────────────────────────────────────────────────
INPUT_CSV = os.path.join(os.path.dirname(__file__), "All_Models_Results.csv")
OUT_DIR = os.path.dirname(__file__)

# Metrics to rank (only _mean columns)
METRIC_COLS = [
    "Accuracy_mean",
    "Precision_mean",
    "Recall_mean",
    "F1_mean",
    "GMean_mean",
    "AUC_mean",
    "MCC_mean",
    "AUPRC_mean",
    "IBA_mean",
    "Kappa_mean",
    "Runtime_mean",  # lower is better
]

# Metrics where lower is better
LOWER_IS_BETTER = {"Runtime_mean"}

# ── Load data ───────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows, {df['Model'].nunique()} models, {df['Dataset'].nunique()} datasets")

# Ensure metric columns are numeric
for col in METRIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Rank models per dataset, per metric ─────────────────────────────────
rank_dfs = []  # to store per-dataset ranking DataFrames

for dataset, grp in df.groupby("Dataset"):
    for metric in METRIC_COLS:
        valid = grp[["Model", metric]].dropna().copy()
        if valid.empty:
            continue
        ascending = metric in LOWER_IS_BETTER  # True for runtime
        valid["rank"] = valid[metric].rank(ascending=ascending, method="average")
        valid["Dataset"] = dataset
        valid["Metric"] = metric
        rank_dfs.append(valid[["Dataset", "Model", "Metric", "rank"]])

all_ranks = pd.concat(rank_dfs, ignore_index=True)
print(f"Ranking records: {len(all_ranks)}")

# ── Average rank per model per metric (across all datasets) ─────────────
avg_ranks = (
    all_ranks.groupby(["Model", "Metric"])["rank"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
avg_ranks.columns = ["Model", "Metric", "AvgRank", "StdRank", "NumDatasets"]
avg_ranks["AvgRank"] = avg_ranks["AvgRank"].round(2)
avg_ranks["StdRank"] = avg_ranks["StdRank"].round(2)

# Pivot: rows=models, columns=metrics
pivot = avg_ranks.pivot_table(
    index="Model", columns="Metric", values="AvgRank", aggfunc="first"
)

# Sort by average across all metrics (optional: gives an overall ranking)
metric_cols_for_sort = [m for m in METRIC_COLS if m in pivot.columns]
pivot["OverallAvgRank"] = pivot[metric_cols_for_sort].mean(axis=1).round(2)
pivot = pivot.sort_values("OverallAvgRank")

# Save
out1 = os.path.join(OUT_DIR, "avg_rankings.csv")
pivot.to_csv(out1)
print(f"Saved: {out1}")

# ── Per-dataset ranking table (pivoted wide) ────────────────────────────
per_dataset_pivot = all_ranks.pivot_table(
    index=["Dataset", "Model"], columns="Metric", values="rank", aggfunc="first"
)
per_dataset_pivot = per_dataset_pivot.round(2)
out2 = os.path.join(OUT_DIR, "avg_rankings_per_dataset.csv")
per_dataset_pivot.to_csv(out2)
print(f"Saved: {out2}")

# ── Summary: top-5 models per metric ────────────────────────────────────
with open(os.path.join(OUT_DIR, "summary_top5.txt"), "w", encoding="utf-8") as f:
    for metric in METRIC_COLS:
        direction = "↓ lower better" if metric in LOWER_IS_BETTER else "↑ higher better"
        f.write(f"\n{'=' * 70}\n")
        f.write(f"  {metric}  ({direction})\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"{'Rank':<6} {'Model':<20} {'AvgRank':<10} {'Std':<8} {'#Datasets'}\n")
        f.write(f"{'-' * 70}\n")

        subset = avg_ranks[avg_ranks["Metric"] == metric].sort_values("AvgRank")
        for i, (_, row) in enumerate(subset.iterrows(), 1):
            f.write(
                f"{i:<6} {row['Model']:<20} {row['AvgRank']:<10} {row['StdRank']:<8} {row['NumDatasets']}\n"
            )

    # Overall ranking
    f.write(f"\n{'=' * 70}\n")
    f.write(f"  OVERALL (average rank across all metrics)\n")
    f.write(f"{'=' * 70}\n")
    f.write(f"{'Rank':<6} {'Model':<20} {'OverallAvgRank':<15}\n")
    f.write(f"{'-' * 70}\n")
    for i, (model, row) in enumerate(pivot.iterrows()):
        f.write(f"{i+1:<6} {model:<20} {row['OverallAvgRank']:<15}\n")

print(f"Saved: summary_top5.txt")

# ── Win counts: how often each model ranks #1 per metric ────────────────
win_data = []
for (metric, dataset), grp in all_ranks.groupby(["Metric", "Dataset"]):
    best_rank = grp["rank"].min()
    winners = grp[grp["rank"] == best_rank]["Model"].tolist()
    for w in winners:
        win_data.append({"Metric": metric, "Dataset": dataset, "Model": w})

win_df = pd.DataFrame(win_data)
win_counts = (
    win_df.groupby(["Model", "Metric"]).size().reset_index(name="WinCount")
)
win_pivot = win_counts.pivot_table(
    index="Model", columns="Metric", values="WinCount", aggfunc="sum", fill_value=0
)
# Sort by total wins
win_pivot["TotalWins"] = win_pivot.sum(axis=1)
win_pivot = win_pivot.sort_values("TotalWins", ascending=False)

out4 = os.path.join(OUT_DIR, "model_win_counts.csv")
win_pivot.to_csv(out4)
print(f"Saved: {out4}")

# ── Print quick summary to console ──────────────────────────────────────
print("\n" + "=" * 70)
print("  OVERALL TOP 10 MODELS (avg rank across all metrics)")
print("=" * 70)
for i, (model, row) in enumerate(pivot.head(10).iterrows()):
    print(f"  {i+1:>3}. {model:<20}  avg_rank={row['OverallAvgRank']:.2f}")

print("\nDone.")
