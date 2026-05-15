# -*- coding: utf-8 -*-
# 서버 경로 기준 splits.csv 재생성
# 실행: python /data1/wj/seafog/src/rebuild_splits.py

import re
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATA_DIR   = Path("/data1/wj/seafog/data")
SPLITS_CSV = DATA_DIR / "splits.csv"

PORTS   = ["daesan", "gunsan", "yeosu", "haeundae"]
CLASSES = ["normal", "lowvis", "seafog"]

TRAIN_N = 2100
VALID_N = 300
TEST_N  = 600

DATETIME_PATTERN = re.compile(r'(\d{14})')

def parse_dt(filename):
    m = DATETIME_PATTERN.search(filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None

def split_exact(rows):
    date_buckets = defaultdict(list)
    for row in rows:
        dt = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
        date_buckets[dt.date()].append(row)

    sorted_dates = sorted(date_buckets.keys())
    train_dates, valid_dates, test_dates = set(), set(), set()
    cumulative = 0

    for date_key in sorted_dates:
        n = len(date_buckets[date_key])
        if cumulative < TRAIN_N:
            train_dates.add(date_key)
        elif cumulative < TRAIN_N + VALID_N:
            valid_dates.add(date_key)
        else:
            test_dates.add(date_key)
        cumulative += n

    assert train_dates.isdisjoint(valid_dates), "train-valid 날짜 겹침!"
    assert train_dates.isdisjoint(test_dates),  "train-test 날짜 겹침!"
    assert valid_dates.isdisjoint(test_dates),  "valid-test 날짜 겹침!"

    result = []
    for row in rows:
        dt = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
        d = dt.date()
        if d in train_dates:
            row["split"] = "train"
        elif d in valid_dates:
            row["split"] = "valid"
        else:
            row["split"] = "test"
        result.append(row)

    # 장수 보정 (날짜 경계 오차 1~2장)
    for split_name, target, next_s in [
        ("train", TRAIN_N, "valid"),
        ("valid", VALID_N, "test"),
    ]:
        current = sum(1 for r in result if r["split"] == split_name)
        diff = current - target
        if diff > 0:
            moved = 0
            for r in reversed(result):
                if moved >= diff: break
                if r["split"] == split_name:
                    r["split"] = next_s
                    moved += 1
        elif diff < 0:
            moved = 0
            for r in reversed(result):
                if moved >= -diff: break
                if r["split"] == next_s:
                    r["split"] = split_name
                    moved += 1

    return result

if __name__ == "__main__":
    print("splits.csv 재생성 시작")
    print("  경로: {}".format(DATA_DIR))
    print("  목표: train={} / valid={} / test={}".format(TRAIN_N, VALID_N, TEST_N))

    rows = []
    print("\n  파일 스캔 중...")

    for port in PORTS:
        for cls in CLASSES:
            cls_dir = DATA_DIR / port / cls
            if not cls_dir.exists():
                print("  [경고] 없음: {}".format(cls_dir))
                continue
            files = sorted([
                f for f in cls_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])
            for f in files:
                dt = parse_dt(f.name)
                if dt:
                    rows.append({
                        "port":        port,
                        "class_label": cls,
                        "filepath":    str(f.resolve()),
                        "datetime":    dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "year":        dt.year,
                        "month":       dt.month,
                    })
            print("  {:10s} {:8s}: {:,}장".format(port, cls, len(files)))

    output_rows = []
    print("\n  분할 결과:")
    print("  {:<12} {:<10} {:>7} {:>7} {:>7} {:>7}".format(
        "항만", "클래스", "전체", "train", "valid", "test"))
    print("  " + "-"*52)

    for port in PORTS:
        for cls in CLASSES:
            subset = [r for r in rows
                      if r["port"] == port and r["class_label"] == cls]
            if not subset:
                continue
            split_rows = split_exact(subset)
            n_tr = sum(1 for r in split_rows if r["split"] == "train")
            n_va = sum(1 for r in split_rows if r["split"] == "valid")
            n_te = sum(1 for r in split_rows if r["split"] == "test")
            print("  {:<12} {:<10} {:>7,} {:>7,} {:>7,} {:>7,}".format(
                port, cls, len(split_rows), n_tr, n_va, n_te))
            output_rows.extend(split_rows)

    fieldnames = ["port","class_label","filepath","datetime","year","month","split"]
    with open(str(SPLITS_CSV), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(output_rows)

    total = len(output_rows)
    n_tr  = sum(1 for r in output_rows if r["split"] == "train")
    n_va  = sum(1 for r in output_rows if r["split"] == "valid")
    n_te  = sum(1 for r in output_rows if r["split"] == "test")

    print("\n  " + "="*52)
    print("  전체  : {:,}장".format(total))
    print("  train : {:,}장".format(n_tr))
    print("  valid : {:,}장".format(n_va))
    print("  test  : {:,}장".format(n_te))
    print("  저장  : {}".format(SPLITS_CSV))
