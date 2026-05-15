from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


PYTHON = "/usr/local/miniconda3/bin/python"
TRAIN_PY = "/home/wj/seafog/src/train.py"
DATA_CSV = "/data1/wj/seafog/data/splits.csv"

CNN_BACKBONES = [
    "xception",
    "resnet101",
    "tf_efficientnetv2_m",
    "convnext_base",
]

PORTS = ["daesan", "gunsan", "yeosu", "haeundae"]

SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
IMG_SIZE = 512
NUM_WORKERS = 16
PATIENCE = 15
USE_AMP = True

RESULT_ROOT = Path("/data1/wj/seafog/results/track_a")
LOG_ROOT = Path("/data1/wj/seafog/logs/track_a")

SKIP_COMPLETED = False  # True면 이미 끝난 조합(best.pth + test_metrics.json 존재)은 건너뜀


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_jobs() -> List[Tuple[str, str]]:
    jobs: List[Tuple[str, str]] = []
    for backbone in CNN_BACKBONES:
        for port in PORTS:
            jobs.append((backbone, port))
    return jobs


def is_completed(backbone: str, port: str) -> bool:
    out_dir = RESULT_ROOT / backbone / port
    return (out_dir / "best.pth").exists() and (out_dir / "test_metrics.json").exists()


def count_completed_jobs(jobs: List[Tuple[str, str]]) -> int:
    return sum(1 for backbone, port in jobs if is_completed(backbone, port))


def stream_process(cmd: List[str], log_file: Path) -> int:
    ensure_dir(log_file.parent)
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"[{ts()}] CMD: {' '.join(cmd)}\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)

        return process.wait()


def run_one(backbone: str, port: str, job_idx: int, total_jobs: int) -> None:
    out_dir = RESULT_ROOT / backbone / port
    log_file = LOG_ROOT / backbone / f"{port}.log"
    ensure_dir(out_dir)
    ensure_dir(log_file.parent)

    if SKIP_COMPLETED and is_completed(backbone, port):
        print(
            f"[{ts()}] SKIP | job={job_idx}/{total_jobs} | backbone={backbone} | port={port} | already completed",
            flush=True,
        )
        return

    cmd = [
        PYTHON,
        TRAIN_PY,
        "--backbone", backbone,
        "--port", port,
        "--data_csv", DATA_CSV,
        "--output", str(out_dir),
        "--seed", str(SEED),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--img_size", str(IMG_SIZE),
        "--num_workers", str(NUM_WORKERS),
        "--patience", str(PATIENCE),
        "--job_idx", str(job_idx),
        "--total_jobs", str(total_jobs),
    ]
    if USE_AMP:
        cmd.append("--use_amp")
    else:
        cmd.append("--no_amp")

    print("=" * 100, flush=True)
    print(
        f"[{ts()}] START | job={job_idx}/{total_jobs} | backbone={backbone} | port={port}",
        flush=True,
    )
    print(f"output: {out_dir}", flush=True)
    print(f"log   : {log_file}", flush=True)
    print("=" * 100, flush=True)

    ret = stream_process(cmd, log_file)
    if ret != 0:
        raise RuntimeError(
            f"Training failed | backbone={backbone} | port={port} | job={job_idx}/{total_jobs} | log={log_file}"
        )

    print("=" * 100, flush=True)
    print(
        f"[{ts()}] DONE  | job={job_idx}/{total_jobs} | backbone={backbone} | port={port}",
        flush=True,
    )
    print("=" * 100, flush=True)
    print(flush=True)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_backbone(backbone: str) -> Dict:
    rows = []
    for port in PORTS:
        out_dir = RESULT_ROOT / backbone / port
        metrics_path = out_dir / "test_metrics.json"
        summary_path = out_dir / "run_summary.json"

        if not metrics_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Missing result files for backbone={backbone}, port={port}")

        metrics = load_json(metrics_path)
        summary = load_json(summary_path)

        rows.append(
            {
                "backbone": backbone,
                "port": port,
                "best_epoch": summary["best_epoch"],
                "params_m": summary["num_params_m"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_precision": metrics["weighted_precision"],
                "weighted_recall": metrics["weighted_recall"],
                "weighted_f1": metrics["weighted_f1"],
                "fogbdry_precision": metrics["fogbdry_precision"],
                "fogbdry_recall": metrics["fogbdry_recall"],
                "fogbdry_f1": metrics["fogbdry_f1"],
                "normal_precision": metrics["normal_precision"],
                "lowvis_precision": metrics["lowvis_precision"],
                "seafog_precision": metrics["seafog_precision"],
                "normal_recall": metrics["normal_recall"],
                "lowvis_recall": metrics["lowvis_recall"],
                "seafog_recall": metrics["seafog_recall"],
                "normal_f1": metrics["normal_f1"],
                "lowvis_f1": metrics["lowvis_f1"],
                "seafog_f1": metrics["seafog_f1"],
                "aber_normal_to_lowvis": metrics["aber_normal_to_lowvis"],
                "aber_lowvis_to_normal": metrics["aber_lowvis_to_normal"],
                "aber_lowvis_to_seafog": metrics["aber_lowvis_to_seafog"],
                "aber_seafog_to_lowvis": metrics["aber_seafog_to_lowvis"],
                "aber_seafog_to_normal": metrics["aber_seafog_to_normal"],
                "aber_normal_to_seafog": metrics["aber_normal_to_seafog"],
            }
        )

    df = pd.DataFrame(rows)

    return {
        "backbone": backbone,
        "ports_completed": len(df),
        "params_m": float(df["params_m"].mean()),
        "macro_precision": float(df["macro_precision"].mean()),
        "macro_recall": float(df["macro_recall"].mean()),
        "macro_f1": float(df["macro_f1"].mean()),
        "weighted_precision": float(df["weighted_precision"].mean()),
        "weighted_recall": float(df["weighted_recall"].mean()),
        "weighted_f1": float(df["weighted_f1"].mean()),
        "fogbdry_precision": float(df["fogbdry_precision"].mean()),
        "fogbdry_recall": float(df["fogbdry_recall"].mean()),
        "fogbdry_f1": float(df["fogbdry_f1"].mean()),
        "normal_precision": float(df["normal_precision"].mean()),
        "lowvis_precision": float(df["lowvis_precision"].mean()),
        "seafog_precision": float(df["seafog_precision"].mean()),
        "normal_recall": float(df["normal_recall"].mean()),
        "lowvis_recall": float(df["lowvis_recall"].mean()),
        "seafog_recall": float(df["seafog_recall"].mean()),
        "normal_f1": float(df["normal_f1"].mean()),
        "lowvis_f1": float(df["lowvis_f1"].mean()),
        "seafog_f1": float(df["seafog_f1"].mean()),
        "port_std": float(df["macro_f1"].std(ddof=0)),
        "aber_normal_to_lowvis": float(df["aber_normal_to_lowvis"].mean()),
        "aber_lowvis_to_normal": float(df["aber_lowvis_to_normal"].mean()),
        "aber_lowvis_to_seafog": float(df["aber_lowvis_to_seafog"].mean()),
        "aber_seafog_to_lowvis": float(df["aber_seafog_to_lowvis"].mean()),
        "aber_seafog_to_normal": float(df["aber_seafog_to_normal"].mean()),
        "aber_normal_to_seafog": float(df["aber_normal_to_seafog"].mean()),
    }


def upsert_backbone_summary(row: Dict) -> None:
    summary_path = RESULT_ROOT / "summary_table_a.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        df = pd.DataFrame()

    row_df = pd.DataFrame([row])

    if df.empty:
        merged = row_df
    else:
        if "backbone" in df.columns:
            df = df[df["backbone"].astype(str) != str(row["backbone"])].copy()
        merged = pd.concat([df, row_df], ignore_index=True)

    merged.to_csv(summary_path, index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dir(RESULT_ROOT)
    ensure_dir(LOG_ROOT)

    jobs = build_jobs()
    total_jobs = len(jobs)
    completed_before = count_completed_jobs(jobs)

    print("=" * 100, flush=True)
    print("Track A | CNN 4 backbones x 4 ports sequential run", flush=True)
    print(f"python      : {PYTHON}", flush=True)
    print(f"train.py    : {TRAIN_PY}", flush=True)
    print(f"data_csv    : {DATA_CSV}", flush=True)
    print(f"backbones   : {CNN_BACKBONES}", flush=True)
    print(f"ports       : {PORTS}", flush=True)
    print(f"seed        : {SEED}", flush=True)
    print(f"epochs      : {EPOCHS}", flush=True)
    print(f"batch_size  : {BATCH_SIZE}", flush=True)
    print(f"img_size    : {IMG_SIZE}", flush=True)
    print(f"num_workers : {NUM_WORKERS}", flush=True)
    print(f"patience    : {PATIENCE}", flush=True)
    print(f"use_amp     : {USE_AMP}", flush=True)
    print(f"skip_done   : {SKIP_COMPLETED}", flush=True)
    print(f"start_time  : {ts()}", flush=True)
    print(f"completed_before_start : {completed_before}/{total_jobs}", flush=True)
    print("=" * 100, flush=True)
    print(flush=True)

    for i, (backbone, port) in enumerate(jobs, start=1):
        print(
            f"[{ts()}] PIPELINE PROGRESS | next_job={i}/{total_jobs} | backbone={backbone} | port={port}",
            flush=True,
        )
        run_one(backbone, port, job_idx=i, total_jobs=total_jobs)

        completed_now = count_completed_jobs(jobs)
        print(
            f"[{ts()}] PIPELINE STATUS | completed={completed_now}/{total_jobs}",
            flush=True,
        )
        print(flush=True)

        if port == PORTS[-1]:
            row = aggregate_backbone(backbone)
            upsert_backbone_summary(row)

            print("#" * 100, flush=True)
            print(f"[{ts()}] BACKBONE DONE  | {backbone}", flush=True)
            print(
                f"macro_p={row['macro_precision']:.4f} | "
                f"macro_r={row['macro_recall']:.4f} | "
                f"macro_f1={row['macro_f1']:.4f} | "
                f"fog_p={row['fogbdry_precision']:.4f} | "
                f"fog_r={row['fogbdry_recall']:.4f} | "
                f"fog_f1={row['fogbdry_f1']:.4f} | "
                f"port_std={row['port_std']:.4f}",
                flush=True,
            )
            print("#" * 100, flush=True)
            print(flush=True)

    print("=" * 100, flush=True)
    print(f"[{ts()}] ALL CNN RUNS FINISHED", flush=True)
    print(f"summary_table_a.csv      : {RESULT_ROOT / 'summary_table_a.csv'}", flush=True)
    print(f"summary_table_a_ports.csv: {RESULT_ROOT / 'summary_table_a_ports.csv'}", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()