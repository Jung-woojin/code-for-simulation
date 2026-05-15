"""
run_scratch.py
Scratch CNN 파이프라인: small kernel vs large kernel ERF 비교

2 strategy × 4 port × 1 seed = 8 jobs
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple

import pandas as pd


PYTHON      = "/usr/local/miniconda3/bin/python"
TRAIN_PY    = "/home/wj/seafog/src/train_scratch.py"
DATA_CSV    = "/data1/wj/seafog/data/splits.csv"

STRATEGIES  = ["scratch_small", "scratch_large"]
PORTS       = ["daesan", "gunsan", "yeosu", "haeundae"]
SEEDS       = [42]

EPOCHS      = 100
PATIENCE    = 15
BATCH_SIZE  = 32
IMG_SIZE    = 512
NUM_WORKERS = 16
LR          = 1e-3
MID_CHANNELS = 128
FEAT_DIM    = 512
USE_AMP     = True

RESULT_ROOT = Path("/data1/wj/seafog/results/scratch")
LOG_ROOT    = Path("/data1/wj/seafog/logs/scratch")

SKIP_COMPLETED = False


class Job(NamedTuple):
    strategy: str
    port: str
    seed: int


def build_jobs() -> List[Job]:
    return [Job(s, p, seed)
            for s in STRATEGIES
            for p in PORTS
            for seed in SEEDS]


def job_output_dir(job: Job) -> Path:
    return RESULT_ROOT / job.strategy / job.port / f"seed{job.seed}"


def is_completed(job: Job) -> bool:
    d = job_output_dir(job)
    return (d / "best.pth").exists() and (d / "test_metrics.json").exists()


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stream_process(cmd: List[str], log_file: Path) -> int:
    ensure_dir(log_file.parent)
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"[{ts()}] CMD: {' '.join(cmd)}\n\n")
        f.flush()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
        return process.wait()


def run_one(job: Job, job_idx: int, total_jobs: int) -> None:
    out_dir  = job_output_dir(job)
    log_file = LOG_ROOT / job.strategy / job.port / f"seed{job.seed}.log"
    ensure_dir(out_dir)

    if SKIP_COMPLETED and is_completed(job):
        print(f"[{ts()}] SKIP | {job_idx}/{total_jobs} | "
              f"{job.strategy}/{job.port}/seed{job.seed}", flush=True)
        return

    cmd = [
        PYTHON, TRAIN_PY,
        "--strategy",    job.strategy,
        "--port",        job.port,
        "--data_csv",    DATA_CSV,
        "--output",      str(out_dir),
        "--seed",        str(job.seed),
        "--img_size",    str(IMG_SIZE),
        "--batch_size",  str(BATCH_SIZE),
        "--num_workers", str(NUM_WORKERS),
        "--epochs",      str(EPOCHS),
        "--patience",    str(PATIENCE),
        "--lr",          str(LR),
        "--mid_channels",str(MID_CHANNELS),
        "--feat_dim",    str(FEAT_DIM),
        "--job_idx",     str(job_idx),
        "--total_jobs",  str(total_jobs),
    ]
    if USE_AMP:
        cmd.append("--use_amp")
    else:
        cmd.append("--no_amp")

    print("=" * 100, flush=True)
    print(f"[{ts()}] START | {job_idx}/{total_jobs} | "
          f"{job.strategy} | {job.port} | seed{job.seed}", flush=True)
    print(f"output : {out_dir}", flush=True)
    print(f"log    : {log_file}", flush=True)
    print("=" * 100, flush=True)

    ret = stream_process(cmd, log_file)
    if ret != 0:
        raise RuntimeError(
            f"Failed | {job.strategy}/{job.port}/seed{job.seed} | log={log_file}"
        )

    print("=" * 100, flush=True)
    print(f"[{ts()}] DONE | {job_idx}/{total_jobs} | "
          f"{job.strategy}/{job.port}/seed{job.seed}", flush=True)
    print("=" * 100, flush=True)


def aggregate_results() -> None:
    rows = []
    for strategy in STRATEGIES:
        port_rows = []
        for port in PORTS:
            for seed in SEEDS:
                job = Job(strategy, port, seed)
                mp  = job_output_dir(job) / "test_metrics.json"
                if not mp.exists():
                    continue
                with mp.open() as f:
                    m = json.load(f)
                port_rows.append({"strategy": strategy, "port": port,
                                   "seed": seed, **m})
        if not port_rows:
            continue
        df = pd.DataFrame(port_rows)
        rows.append({
            "strategy": strategy,
            "n_ports": len(df),
            "macro_f1_mean":   df["macro_f1"].mean(),
            "macro_f1_std":    df["macro_f1"].std(),
            "fogbdry_f1_mean": df["fogbdry_f1"].mean(),
            "fogbdry_f1_std":  df["fogbdry_f1"].std(),
            "port_std":        df["macro_f1"].std(ddof=0),
        })

    if rows:
        out = RESULT_ROOT / "summary_scratch.csv"
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n[집계 완료] {out}", flush=True)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)


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

    jobs       = build_jobs()
    total_jobs = len(jobs)
    done       = sum(1 for j in jobs if is_completed(j))

    print("=" * 100, flush=True)
    print("Scratch CNN | small kernel vs large kernel", flush=True)
    print(f"strategies  : {STRATEGIES}", flush=True)
    print(f"ports       : {PORTS}", flush=True)
    print(f"seeds       : {SEEDS}", flush=True)
    print(f"total_jobs  : {total_jobs}", flush=True)
    print(f"completed   : {done}/{total_jobs}", flush=True)
    print(f"start_time  : {ts()}", flush=True)
    print("=" * 100, flush=True)

    for i, job in enumerate(jobs, start=1):
        print(f"\n[{ts()}] PROGRESS | {i}/{total_jobs} | "
              f"{job.strategy}/{job.port}/seed{job.seed}", flush=True)
        run_one(job, job_idx=i, total_jobs=total_jobs)
        done_now = sum(1 for j in jobs if is_completed(j))
        print(f"[{ts()}] STATUS | completed={done_now}/{total_jobs}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print(f"[{ts()}] ALL DONE", flush=True)
    print("=" * 100, flush=True)

    aggregate_results()


if __name__ == "__main__":
    main()