"""build_bearing_ir.py — one-shot constructor for the imbalanced Bearing datasets.

For each of the 8 raw Bearing CSVs (Dataset/Bearing/<name>.csv) and each IR in
{5, 10, 20, 30}, builds a constructed dataset where the majority (largest) class
is kept full-size and every minority class is sub-sampled to floor(N_majority /
IR) (floor = 1). Output: Dataset/Bearing_IR/IR<ir>/<name>.csv (feat_*,label) plus
a _manifest.json recording per-IR per-class sizes.

The 5-fold split is NOT baked into the file; common.dataio.load_folds applies a
deterministic StratifiedKFold(5, random_state=42) at load time so all 30 models
see identical folds.

Usage:
    python build_bearing_ir.py            # build all 8 x 4 = 32 (idempotent)
    python build_bearing_ir.py --smoke    # build only IR5_CWRU (quick check)
    python build_bearing_ir.py --force    # rebuild even if files exist
"""
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from common import dataio, paths


def main():
    ap = argparse.ArgumentParser(description="Build imbalanced Bearing IR datasets")
    ap.add_argument("--smoke", action="store_true",
                    help="Build only IR5_CWRU (quick sanity check)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing constructed CSVs")
    ap.add_argument("--irs", default="5,10,20,30",
                    help="Comma list of IRs (default 5,10,20,30)")
    args = ap.parse_args()

    paths.ensure_dirs()
    irs = tuple(int(x) for x in args.irs.split(","))

    if args.smoke:
        names = ("CWRU",)
        irs = (5,)
        print("[build_bearing_ir] SMOKE: building only IR5_CWRU")
    else:
        names = dataio.BEARING_NAMES

    manifest = dataio.build_bearing_ir_datasets(
        irs=irs, random_state=42, overwrite=args.force, names=names)

    # verify the smoke target
    if args.smoke:
        import numpy as np, pandas as pd
        p = os.path.join(paths.BEARING_IR, "IR5", "CWRU.csv")
        df = pd.read_csv(p)
        feat = [c for c in df.columns if c.startswith("feat_")]
        from collections import Counter
        cnt = Counter(df["label"].tolist())
        maj, n_maj = max(cnt.items(), key=lambda kv: kv[1])
        minor_targets = {k: v for k, v in cnt.items() if k != maj}
        print(f"\n[verify] {p}")
        print(f"  features={len(feat)}  total={len(df)}")
        print(f"  majority '{maj}' kept full = {n_maj}")
        print(f"  minority sizes = {minor_targets}")
        exp = max(1, n_maj // 5)
        assert all(v == exp for v in minor_targets.values()), "minority size != N_maj/5"
        assert n_maj == 348, f"CWRU majority expected 348, got {n_maj}"
        print(f"  OK: majority full ({n_maj}), minorities = floor({n_maj}/5) = {exp}")


if __name__ == "__main__":
    main()
