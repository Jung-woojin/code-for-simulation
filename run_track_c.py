from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple

import pandas as pd


# ------------------------------------------------------------------ #
#  설정                                                              #
# ------------------------------------------------------------------ #
PYTHON = "/usr/local/miniconda3/bin/python"
TRAIN_PY = "/home/wj/seafog/src/train_c.py"
DATA_CSV = "/data1/wj/seafog/data/splits.csv"

BACKBONES = [
    "resnet101",
    "tf_efficientnetv2_m",
    "xception",
]

STRATEGIES = [
    "ordinal_context",
]

PORTS = ["daesan", "gunsan", "yeosu", "haeundae"]
SEEDS = [42]

EPOCHS_S1 = 20
EPOCHS_S2 = 30
EPOCHS_S3 = 40

PATIENCE_S1 = 7
PATIENCE_S2 = 7
PATIENCE_S3 = 10

NO_EARLY_STOP_FIRST_N = 3

BATCH_SIZE = 32
IMG_SIZE = 512
NUM_WORKERS = 16
USE_AMP = True

RESULT_ROOT = Path("/data1/wj/seafog/results/track_c")
LOG_ROOT = Path("/data1/wj/seafog/logs/track_c")

SKIP_COMPLETED = False


# ------------------------------------------------------------------ #
#  Job 정의                                                          #
# ------------------------------------------------------------------ #
class Job(NamedTuple):
    backbone: str
    strategy: str
    port: str
    seed: int


def build_jobs() -> List[Job]:
    jobs: List[Job] = []
    for backbone in BACKBONES:
        for strategy in STRATEGIES:
            for port in PORTS:
                for seed in SEEDS:
                    jobs.append(Job(backbone, strategy, port, seed))
    return jobs


def job_output_dir(job: Job) -> Path:
    return RESULT_ROOT / job.backbone / job.strategy / job.port / f"seed{job.seed}"


def is_completed(job: Job) -> bool:
    d = job_output_dir(job)
    return (d / "best.pth").exists() and (d / "test_metrics.json").exists()


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
#  프로세스 실행                                                     #
# ------------------------------------------------------------------ #
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


def run_one(job: Job, job_idx: int, total_jobs: int) -> None:
    out_dir = job_output_dir(job)
    log_file = LOG_ROOT / job.backbone / job.strategy / job.port / f"seed{job.seed}.log"
    ensure_dir(out_dir)

    if SKIP_COMPLETED and is_completed(job):
        print(
            f"[{ts()}] SKIP | {job_idx}/{total_jobs} | "
            f"{job.backbone}/{job.strategy}/{job.port}/seed{job.seed}",
            flush=True,
        )
        return

    cmd = [
        PYTHON,
        TRAIN_PY,
        "--backbone", job.backbone,
        "--strategy", job.strategy,
        "--port", job.port,
        "--data_csv", DATA_CSV,
        "--output", str(out_dir),
        "--seed", str(job.seed),
        "--img_size", str(IMG_SIZE),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", str(NUM_WORKERS),
        "--epochs_s1", str(EPOCHS_S1),
        "--epochs_s2", str(EPOCHS_S2),
        "--epochs_s3", str(EPOCHS_S3),
        "--patience_s1", str(PATIENCE_S1),
        "--patience_s2", str(PATIENCE_S2),
        "--patience_s3", str(PATIENCE_S3),
        "--no_early_stop_first_n", str(NO_EARLY_STOP_FIRST_N),
        "--job_idx", str(job_idx),
        "--total_jobs", str(total_jobs),
    ]

    if USE_AMP:
        cmd.append("--use_amp")
    else:
        cmd.append("--no_amp")

    print("=" * 100, flush=True)
    print(
        f"[{ts()}] START | {job_idx}/{total_jobs} | "
        f"{job.backbone} | {job.strategy} | {job.port} | seed{job.seed}",
        flush=True,
    )
    print(f"output : {out_dir}", flush=True)
    print(f"log    : {log_file}", flush=True)
    print("=" * 100, flush=True)

    ret = stream_process(cmd, log_file)
    if ret != 0:
        raise RuntimeError(
            f"Failed | {job.backbone}/{job.strategy}/{job.port}/seed{job.seed} | log={log_file}"
        )

    print("=" * 100, flush=True)
    print(
        f"[{ts()}] DONE | {job_idx}/{total_jobs} | "
        f"{job.backbone}/{job.strategy}/{job.port}/seed{job.seed}",
        flush=True,
    )
    print("=" * 100, flush=True)


# ------------------------------------------------------------------ #
#  집계                                                              #
# ------------------------------------------------------------------ #
def aggregate_results() -> None:
    rows = []

    for backbone in BACKBONES:
        for strategy in STRATEGIES:
            for port in PORTS:
                seed_rows = []

                for seed in SEEDS:
                    job = Job(backbone, strategy, port, seed)
                    metrics_path = job_output_dir(job) / "test_metrics.json"
                    if not metrics_path.exists():
                        continue

                    with metrics_path.open("r", encoding="utf-8") as f:
                        m = json.load(f)
                    seed_rows.append(m)

                if not seed_rows:
                    continue

                df = pd.DataFrame(seed_rows)
                row: Dict[str, object] = {
                    "backbone": backbone,
                    "strategy": strategy,
                    "port": port,
                    "n_seeds": len(df),
                }

                metric_cols = [
                    "macro_f1",
                    "macro_precision",
                    "macro_recall",
                    "normal_f1",
                    "lowvis_f1",
                    "seafog_f1",
                    "normal_precision",
                    "lowvis_precision",
                    "seafog_precision",
                    "normal_recall",
                    "lowvis_recall",
                    "seafog_recall",
                    "aber_normal_to_lowvis",
                    "aber_lowvis_to_normal",
                    "aber_lowvis_to_seafog",
                    "aber_seafog_to_lowvis",
                    "aber_seafog_to_normal",
                    "aber_normal_to_seafog",
                ]

                for col in metric_cols:
                    if col in df.columns:
                        row[f"{col}_mean"] = float(df[col].mean())
                        row[f"{col}_std"] = float(df[col].std(ddof=0))

                rows.append(row)

    if rows:
        summary_path = RESULT_ROOT / "summary_table_c.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n[집계 완료] {summary_path}", flush=True)


# ------------------------------------------------------------------ #
#  메인                                                              #
# ------------------------------------------------------------------ #
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_completed", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    cli = ap.parse_args()

    global SKIP_COMPLETED
    if cli.skip_completed:
        SKIP_COMPLETED = True

    if cli.aggregate_only:
        aggregate_results()
        return

    ensure_dir(RESULT_ROOT)
    ensure_dir(LOG_ROOT)

    jobs = build_jobs()
    total_jobs = len(jobs)
    done_before = sum(1 for j in jobs if is_completed(j))

    print("=" * 100, flush=True)
    print("Track C | ordinal_context sequential pipeline", flush=True)
    print(f"python     : {PYTHON}", flush=True)
    print(f"train.py   : {TRAIN_PY}", flush=True)
    print(f"data_csv   : {DATA_CSV}", flush=True)
    print(f"backbones  : {BACKBONES}", flush=True)
    print(f"strategies : {STRATEGIES}", flush=True)
    print(f"ports      : {PORTS}", flush=True)
    print(f"seeds      : {SEEDS}", flush=True)
    print(f"total_jobs : {total_jobs}", flush=True)
    print(f"completed  : {done_before}/{total_jobs}", flush=True)
    print(f"skip_done  : {SKIP_COMPLETED}", flush=True)
    print(f"start_time : {ts()}", flush=True)
    print("=" * 100, flush=True)

    for i, job in enumerate(jobs, start=1):
        print(
            f"\n[{ts()}] PROGRESS | {i}/{total_jobs} | "
            f"{job.backbone}/{job.strategy}/{job.port}/seed{job.seed}",
            flush=True,
        )
        run_one(job, job_idx=i, total_jobs=total_jobs)
        done_now = sum(1 for j in jobs if is_completed(j))
        print(f"[{ts()}] STATUS | completed={done_now}/{total_jobs}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print(f"[{ts()}] ALL DONE", flush=True)
    print("=" * 100, flush=True)

    aggregate_results()


if __name__ == "__main__":
    main()