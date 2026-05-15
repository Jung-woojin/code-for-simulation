#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# make_splits_6000.py — 새 데이터셋(daesan/incheon/mokpo) splits.csv 생성
# 위치: /home/wj/seafog/src/make_splits_6000.py
#
# 사용법:
#   python make_splits_6000.py \
#       --data_root /data1/wj/seafog/data_6000/seafog_6000_per_port \
#       --output /data1/wj/seafog/data_6000/splits_6000.csv

import argparse
import random
from pathlib import Path

import pandas as pd


CLASS_MAP = {
    "0_normal": "normal",
    "1_lowvis": "lowvis",
    "2_seafog": "seafog",
}


def make_splits(data_root: str, output: str, seed: int = 42):
    random.seed(seed)
    data_root = Path(data_root)
    rows = []

    ports = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    print(f"발견된 항만: {ports}")

    for port in ports:
        port_dir = data_root / port
        for cls_dir_name, class_label in CLASS_MAP.items():
            cls_dir = port_dir / cls_dir_name
            if not cls_dir.exists():
                print(f"  [WARN] 없음: {cls_dir}")
                continue

            imgs = sorted([
                p for p in cls_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ])
            random.shuffle(imgs)

            n     = len(imgs)
            n_tr  = int(n * 0.7)
            n_val = int(n * 0.1)

            for i, img in enumerate(imgs):
                if i < n_tr:
                    split = "train"
                elif i < n_tr + n_val:
                    split = "valid"
                else:
                    split = "test"

                rows.append({
                    "filepath":    str(img),
                    "port":        port,
                    "class_label": class_label,
                    "split":       split,
                })

            print(f"  {port}/{class_label}: {n}장 "
                  f"(train={n_tr}, valid={n_val}, test={n - n_tr - n_val})")

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output} ({len(df)}행)")

    print("\n=== 요약 ===")
    print(df.groupby(["port", "class_label", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str,
                   default="/data1/wj/seafog/data_6000/seafog_6000_per_port")
    p.add_argument("--output", type=str,
                   default="/data1/wj/seafog/data_6000/splits_6000.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    make_splits(args.data_root, args.output, args.seed)
