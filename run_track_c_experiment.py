"""
Experiment runner for Track C.

Example:
  python /home/wj/seafog/src/run_track_c_experiment.py --skip_completed
  python /home/wj/seafog/src/run_track_c_experiment.py --strategies parallel stageaware_A
  python /home/wj/seafog/src/run_track_c_experiment.py --backbones resnet101 xception --seeds 42
  python /home/wj/seafog/src/run_track_c_experiment.py --aggregate_only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


PYTHON = "/usr/local/miniconda3/bin/python"
TRAIN_PY = "/home/wj/seafog/src/train_c_experiment.py"
DATA_CSV = "/data1/wj/seafog/data/splits.csv"

DEFAULT_BACKBONES = [
    "resnet101",
    "tf_efficientnetv2_m",
    "xception",
    "convnext_base",
]

DEFAULT_STRATEGIES = [
    "baseline",
    "parallel",
    "stageaware_A",
    "stageaware_B",
    "stageaware_C",
]

DEFAULT_PORTS = ["daesan", "gunsan", "yeosu", "haeundae"]
DEFAULT_SEEDS = [42, 123, 777]

EPOCHS_S1 = 30
EPOCHS_S2 = 30
EPOCHS_S3 = 40
PATIENCE_S1 = 5
PATIENCE_S2 = 7
PATIENCE_S3 = 10
NO_EARLY_STOP_FIRST_N = 3

BATCH_SIZE = 32
IMG_SIZE = 512
NUM_WORKERS = 16
USE_AMP = True

RESULT_ROOT = Path("/data1/wj/seafog/results/experiment")
LOG_ROOT = RESULT_ROOT / "logs"


@dataclass(frozen=True)
class Job:
    backbone: str
    strategy_label: str
    port: str
    seed: int


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_strategy(strategy_label: str) -> Tuple[str, Optional[str]]:
    if strategy_label == "baseline":
        return "baseline", None
    if strategy_label == "parallel":
        return "parallel", None
    if strategy_label == "stageaware_A":
        return "stageaware", "3,5,7,9"
    if strategy_label == "stageaware_B":
        return "stageaware", "5,5,7,9"
    if strategy_label == "stageaware_C":
        return "stageaware", "5,7,9,9"
    raise ValueError(f"Unknown strategy label: {strategy_label}")


def build_jobs(
    backbones: Sequence[str],
    strategies: Sequence[str],
    ports: Sequence[str],
    seeds: Sequence[int],
) -> List[Job]:
    jobs: List[Job] = []
    for backbone in backbones:
        for strategy in strategies:
            for port in ports:
                for seed in seeds:
                    jobs.append(Job(backbone, strategy, port, seed))
    return jobs


def job_output_dir(job: Job) -> Path:
    return RESULT_ROOT / job.backbone / job.strategy_label / job.port / f"seed{job.seed}"


def job_log_file(job: Job) -> Path:
    return LOG_ROOT / job.backbone / job.strategy_label / job.port / f"seed{job.seed}.log"


def is_completed(job: Job) -> bool:
    output_dir = job_output_dir(job)
    return (output_dir / "best.pth").exists() and (output_dir / "test_metrics.json").exists()


def stream_process(cmd: List[str], log_file: Path) -> int:
    ensure_dir(log_file.parent)
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(f"[{ts()}] CMD: {' '.join(cmd)}\n\n")
        handle.flush()

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
            handle.write(line)
        return process.wait()


def run_one(job: Job, job_idx: int, total_jobs: int, skip_completed: bool) -> None:
    output_dir = job_output_dir(job)
    log_file = job_log_file(job)
    ensure_dir(output_dir)

    if skip_completed and is_completed(job):
        print(
            f"[{ts()}] SKIP | {job_idx}/{total_jobs} | "
            f"{job.backbone}/{job.strategy_label}/{job.port}/seed{job.seed}",
            flush=True,
        )
        return

    model_strategy, stage_kernels = resolve_strategy(job.strategy_label)
    cmd = [
        PYTHON,
        TRAIN_PY,
        "--backbone",
        job.backbone,
        "--strategy_label",
        job.strategy_label,
        "--model_strategy",
        model_strategy,
        "--port",
        job.port,
        "--data_csv",
        DATA_CSV,
        "--output",
        str(output_dir),
        "--result_root",
        str(RESULT_ROOT),
        "--seed",
        str(job.seed),
        "--img_size",
        str(IMG_SIZE),
        "--batch_size",
        str(BATCH_SIZE),
        "--num_workers",
        str(NUM_WORKERS),
        "--epochs_s1",
        str(EPOCHS_S1),
        "--epochs_s2",
        str(EPOCHS_S2),
        "--epochs_s3",
        str(EPOCHS_S3),
        "--patience_s1",
        str(PATIENCE_S1),
        "--patience_s2",
        str(PATIENCE_S2),
        "--patience_s3",
        str(PATIENCE_S3),
        "--no_early_stop_first_n",
        str(NO_EARLY_STOP_FIRST_N),
        "--job_idx",
        str(job_idx),
        "--total_jobs",
        str(total_jobs),
    ]
    if stage_kernels:
        cmd.extend(["--stage_kernels", stage_kernels])
    if USE_AMP:
        cmd.append("--use_amp")
    else:
        cmd.append("--no_amp")

    print("=" * 100, flush=True)
    print(
        f"[{ts()}] START | {job_idx}/{total_jobs} | "
        f"{job.backbone} | {job.strategy_label} | {job.port} | seed{job.seed}",
        flush=True,
    )
    print(f"output : {output_dir}", flush=True)
    print(f"log    : {log_file}", flush=True)
    if stage_kernels:
        print(f"stage_kernels : {stage_kernels}", flush=True)
    print("=" * 100, flush=True)

    ret = stream_process(cmd, log_file)
    if ret != 0:
        raise RuntimeError(
            f"Failed | {job.backbone}/{job.strategy_label}/{job.port}/seed{job.seed} | log={log_file}"
        )

    print("=" * 100, flush=True)
    print(
        f"[{ts()}] DONE | {job_idx}/{total_jobs} | "
        f"{job.backbone}/{job.strategy_label}/{job.port}/seed{job.seed}",
        flush=True,
    )
    print("=" * 100, flush=True)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_results(
    backbones: Sequence[str],
    strategies: Sequence[str],
    ports: Sequence[str],
    seeds: Sequence[int],
) -> None:
    rows = []
    for backbone in backbones:
        for strategy_label in strategies:
            run_rows = []
            for port in ports:
                for seed in seeds:
                    job = Job(backbone, strategy_label, port, seed)
                    output_dir = job_output_dir(job)
                    metrics_path = output_dir / "test_metrics.json"
                    summary_path = output_dir / "run_summary.json"
                    if not metrics_path.exists() or not summary_path.exists():
                        continue
                    metrics = load_json(metrics_path)
                    summary = load_json(summary_path)
                    run_rows.append(
                        {
                            "port": port,
                            "seed": seed,
                            "best_epoch": summary.get("best_epoch"),
                            "params_m": summary.get("num_params_m"),
                            "macro_f1": metrics.get("macro_f1"),
                            "fogbdry_f1": metrics.get("fogbdry_f1"),
                            "lowvis_f1": metrics.get("lowvis_f1"),
                            "seafog_f1": metrics.get("seafog_f1"),
                            "aber_lowvis_to_seafog": metrics.get("aber_lowvis_to_seafog"),
                            "aber_seafog_to_lowvis": metrics.get("aber_seafog_to_lowvis"),
                        }
                    )

            if not run_rows:
                continue

            df = pd.DataFrame(run_rows)
            row = {
                "backbone": backbone,
                "strategy": strategy_label,
                "n_runs": len(df),
                "n_ports": int(df["port"].nunique()),
                "n_seeds": int(df["seed"].nunique()),
                "params_m_mean": float(df["params_m"].mean()) if "params_m" in df else None,
                "best_epoch_mean": float(df["best_epoch"].mean()) if "best_epoch" in df else None,
            }
            for col in [
                "macro_f1",
                "fogbdry_f1",
                "lowvis_f1",
                "seafog_f1",
                "aber_lowvis_to_seafog",
                "aber_seafog_to_lowvis",
            ]:
                row[f"{col}_mean"] = float(df[col].mean())
                row[f"{col}_std"] = float(df[col].std(ddof=0))
            rows.append(row)

    summary_path = RESULT_ROOT / "summary_table_c.csv"
    if rows:
        pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[{ts()}] aggregate_results -> {summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--ports", nargs="+", default=DEFAULT_PORTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(RESULT_ROOT)
    ensure_dir(LOG_ROOT)

    jobs = build_jobs(args.backbones, args.strategies, args.ports, args.seeds)
    total_jobs = len(jobs)
    completed_before = sum(1 for job in jobs if is_completed(job))

    if args.aggregate_only:
        aggregate_results(args.backbones, args.strategies, args.ports, args.seeds)
        return

    print("=" * 100, flush=True)
    print("Track C Experiment | staged fine-tuning pipeline", flush=True)
    print(f"result_root : {RESULT_ROOT}", flush=True)
    print(f"log_root    : {LOG_ROOT}", flush=True)
    print(f"backbones   : {args.backbones}", flush=True)
    print(f"strategies  : {args.strategies}", flush=True)
    print(f"ports       : {args.ports}", flush=True)
    print(f"seeds       : {args.seeds}", flush=True)
    print(f"total_jobs  : {total_jobs}", flush=True)
    print(f"completed   : {completed_before}/{total_jobs}", flush=True)
    print(f"skip_done   : {args.skip_completed}", flush=True)
    print(f"start_time  : {ts()}", flush=True)
    print("=" * 100, flush=True)

    for idx, job in enumerate(jobs, start=1):
        print(
            f"[{ts()}] PROGRESS | {idx}/{total_jobs} | "
            f"{job.backbone}/{job.strategy_label}/{job.port}/seed{job.seed}",
            flush=True,
        )
        run_one(job=job, job_idx=idx, total_jobs=total_jobs, skip_completed=args.skip_completed)
        done_now = sum(1 for item in jobs if is_completed(item))
        print(f"[{ts()}] STATUS | completed={done_now}/{total_jobs}", flush=True)

    print("=" * 100, flush=True)
    print(f"[{ts()}] ALL DONE", flush=True)
    print("=" * 100, flush=True)

    aggregate_results(args.backbones, args.strategies, args.ports, args.seeds)


if __name__ == "__main__":
    main()
