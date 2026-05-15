#!/usr/bin/env python3
"""Aggregate metrics.json files into mean/std CSV tables."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

METRICS = ["accuracy", "macro_f1", "balanced_acc", "kappa", "qwk", "mave", "eer", "aer", "svr", "loss"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="summary_orah.csv")
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for mf in root.glob("*/*/*/seed_*/metrics.json"):
        # root/port/backbone/method/seed_x/metrics.json
        rel = mf.relative_to(root).parts
        if len(rel) < 5:
            continue
        port, backbone, method, seed_dir = rel[0], rel[1], rel[2], rel[3]
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = data.get(args.split, {})
        row = {"port": port, "backbone": backbone, "method": method, "seed": seed_dir.replace("seed_", "")}
        for k in METRICS:
            row[k] = m.get(k, None)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No metrics found under {root}")
    df.to_csv(root / f"raw_{args.out}", index=False)
    agg = df.groupby(["port", "backbone", "method"])[METRICS].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(root / args.out, index=False)
    print(f"wrote {root / args.out}")
    print(f"wrote {root / ('raw_' + args.out)}")

if __name__ == "__main__":
    main()
