"""
train_scratch.py
Scratch CNN 학습 — small kernel vs large kernel ERF 비교
Track A train.py 구조 완전 동일 (단일 stage, early stop)

사용:
  python train_scratch.py \
      --strategy scratch_large \
      --port daesan \
      --data_csv /data1/wj/seafog/data/splits.csv \
      --output /data1/wj/seafog/results/scratch/scratch_large/daesan/seed42 \
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from scratch_model import build_scratch_model
from datasets import SeafogDataset


CLASS_NAMES = ["normal", "lowvis", "seafog"]


# ------------------------------------------------------------------ #
#  Track A train.py 와 동일한 유틸 / 지표                             #
# ------------------------------------------------------------------ #
@dataclass
class EvalResult:
    loss: float
    metrics: Dict[str, float]
    probs: np.ndarray
    labels: np.ndarray
    preds: np.ndarray
    sample_paths: List[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv_dicts(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def upsert_csv_row(path: Path, row: dict, key_cols: Sequence[str]) -> None:
    ensure_dir(path.parent)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()
    row_df = pd.DataFrame([row])
    if df.empty:
        merged = row_df
    else:
        if not all(col in df.columns for col in key_cols):
            merged = pd.concat([df, row_df], ignore_index=True)
        else:
            mask = pd.Series([True] * len(df))
            for col in key_cols:
                mask &= df[col].astype(str) == str(row[col])
            df = df.loc[~mask].copy()
            merged = pd.concat([df, row_df], ignore_index=True)
    merged.to_csv(path, index=False, encoding="utf-8-sig")


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    if isinstance(batch, dict):
        image_key = next((k for k in ["image", "img", "images"] if k in batch), None)
        label_key = next((k for k in ["label", "labels", "target", "targets"] if k in batch), None)
        path_key  = next((k for k in ["path", "paths", "filepath", "filepaths",
                                       "img_path", "img_paths"] if k in batch), None)
        if image_key is None or label_key is None:
            raise KeyError("Batch dict must contain image and label keys.")
        imgs   = batch[image_key]
        labels = batch[label_key]
        paths  = [str(x) for x in batch[path_key]] if path_key else [""] * len(labels)
        return imgs, labels, paths
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            imgs, labels = batch
            return imgs, labels, [""] * len(labels)
        if len(batch) >= 3:
            imgs, labels, raw_paths = batch[0], batch[1], batch[2]
            return imgs, labels, [str(x) for x in raw_paths]
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], average="weighted", zero_division=0)
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], average=None, zero_division=0)
    fog_mask = np.isin(labels, [1, 2])
    fog_labels, fog_preds = labels[fog_mask], preds[fog_mask]
    if len(fog_labels) > 0:
        fog_p, fog_r, fog_f1, _ = precision_recall_fscore_support(
            fog_labels, fog_preds, labels=[1, 2], average="macro", zero_division=0)
    else:
        fog_p = fog_r = fog_f1 = 0.0

    def aber(a: int, b: int) -> float:
        denom = int((labels == a).sum())
        return float(((labels == a) & (preds == b)).sum()) / denom if denom else 0.0

    return {
        "macro_precision": float(macro_p), "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p), "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "fogbdry_precision": float(fog_p), "fogbdry_recall": float(fog_r),
        "fogbdry_f1": float(fog_f1),
        "normal_precision": float(per_p[0]), "lowvis_precision": float(per_p[1]),
        "seafog_precision": float(per_p[2]),
        "normal_recall": float(per_r[0]), "lowvis_recall": float(per_r[1]),
        "seafog_recall": float(per_r[2]),
        "normal_f1": float(per_f1[0]), "lowvis_f1": float(per_f1[1]),
        "seafog_f1": float(per_f1[2]),
        "normal_support": int(per_sup[0]), "lowvis_support": int(per_sup[1]),
        "seafog_support": int(per_sup[2]),
        "aber_normal_to_lowvis": aber(0, 1), "aber_lowvis_to_normal": aber(1, 0),
        "aber_lowvis_to_seafog": aber(1, 2), "aber_seafog_to_lowvis": aber(2, 1),
        "aber_seafog_to_normal": aber(2, 0), "aber_normal_to_seafog": aber(0, 2),
    }


def evaluate_model(model, loader, device, criterion, use_amp) -> EvalResult:
    model.eval()
    total_loss, total_count = 0.0, 0
    probs_list, labels_list, paths_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs, labels, paths = unpack_batch(batch)
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            bs = imgs.size(0)
            total_loss  += loss.item() * bs
            total_count += bs
            probs_list.append(probs.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
            paths_list.extend(paths)
    probs_np  = np.concatenate(probs_list,  axis=0)
    labels_np = np.concatenate(labels_list, axis=0)
    preds_np  = probs_np.argmax(axis=1)
    return EvalResult(
        loss=float(total_loss / max(total_count, 1)),
        metrics=compute_metrics(labels_np, preds_np),
        probs=probs_np, labels=labels_np, preds=preds_np,
        sample_paths=paths_list,
    )


def train_one_epoch(model, loader, device, criterion, optimizer, scaler, use_amp) -> float:
    model.train()
    total_loss, total_count = 0.0, 0
    for batch in loader:
        imgs, labels, _ = unpack_batch(batch)
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        bs = imgs.size(0)
        total_loss  += loss.item() * bs
        total_count += bs
    return total_loss / max(total_count, 1)


def save_test_outputs(output_dir: Path, eval_result: EvalResult) -> None:
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"test_loss": eval_result.loss, **eval_result.metrics},
                  f, ensure_ascii=False, indent=2)
    cm = confusion_matrix(eval_result.labels, eval_result.preds, labels=[0, 1, 2])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pd.DataFrame(cm_norm / row_sums, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix_normalized.csv", encoding="utf-8-sig")
    class_rows = [
        {"class_id": i, "class_name": n,
         "precision": eval_result.metrics[f"{n}_precision"],
         "recall":    eval_result.metrics[f"{n}_recall"],
         "f1":        eval_result.metrics[f"{n}_f1"],
         "support":   eval_result.metrics[f"{n}_support"]}
        for i, n in enumerate(CLASS_NAMES)
    ]
    pd.DataFrame(class_rows).to_csv(
        output_dir / "class_metrics.csv", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------ #
#  메인                                                                #
# ------------------------------------------------------------------ #
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    ensure_dir(output_dir)
    scratch_root = output_dir.parents[2]  # results/scratch/

    print("=" * 100, flush=True)
    print(f"Scratch | job={args.job_idx}/{args.total_jobs} | "
          f"strategy={args.strategy} | port={args.port} | seed={args.seed}", flush=True)
    print(f"device={device} | output={output_dir}", flush=True)
    print("=" * 100, flush=True)

    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=True, drop_last=False,
                  persistent_workers=(args.num_workers > 0))
    train_loader = DataLoader(
        SeafogDataset(args.data_csv, args.port, "train", img_size=args.img_size),
        shuffle=True, **common)
    valid_loader = DataLoader(
        SeafogDataset(args.data_csv, args.port, "valid", img_size=args.img_size),
        shuffle=False, **common)
    test_loader  = DataLoader(
        SeafogDataset(args.data_csv, args.port, "test",  img_size=args.img_size),
        shuffle=False, **common)

    model = build_scratch_model(
        strategy=args.strategy,
        num_classes=3,
        mid_channels=args.mid_channels,
        feat_dim=args.feat_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"num_params  : {num_params:,} ({num_params/1e6:.2f}M)", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = (torch.cuda.amp.GradScaler(enabled=args.use_amp)
              if device.type == "cuda"
              else torch.cuda.amp.GradScaler(enabled=False))

    best_fogbdry  = -1.0
    best_epoch    = -1
    patience_cnt  = 0
    val_history   = []
    final_epoch   = 0

    for epoch in range(1, args.epochs + 1):
        final_epoch  = epoch
        train_loss   = train_one_epoch(model, train_loader, device, criterion,
                                       optimizer, scaler, args.use_amp)
        val_result   = evaluate_model(model, valid_loader, device, criterion, args.use_amp)
        scheduler.step()

        fogbdry = val_result.metrics["fogbdry_f1"]
        macro   = val_result.metrics["macro_f1"]

        val_history.append({
            "epoch": epoch, "train_loss": float(train_loss),
            "val_loss": val_result.loss,
            "val_macro_f1": macro, "val_fogbdry_f1": fogbdry,
            "val_lowvis_f1": val_result.metrics["lowvis_f1"],
            "val_seafog_f1": val_result.metrics["seafog_f1"],
        })

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train={train_loss:.4f} | "
            f"val_loss={val_result.loss:.4f} | "
            f"val_macro_p={val_result.metrics['macro_precision']:.4f} | "
            f"val_macro_r={val_result.metrics['macro_recall']:.4f} | "
            f"val_macro_f1={macro:.4f} | "
            f"val_fog_p={val_result.metrics['fogbdry_precision']:.4f} | "
            f"val_fog_r={val_result.metrics['fogbdry_recall']:.4f} | "
            f"val_fog_f1={fogbdry:.4f}",
            flush=True,
        )

        if fogbdry > best_fogbdry:
            best_fogbdry = fogbdry
            best_epoch   = epoch
            patience_cnt = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_fogbdry_f1": best_fogbdry,
                "args": vars(args),
            }, output_dir / "best.pth")
            print(f"  -> best.pth updated | epoch={epoch} | "
                  f"best_val_fogbdry_f1={best_fogbdry:.4f}", flush=True)
        else:
            patience_cnt += 1
            print(f"  -> no improvement | patience {patience_cnt}/{args.patience}",
                  flush=True)

        if patience_cnt >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}", flush=True)
            break

    save_csv_dicts(val_history, output_dir / "val_history.csv")
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)},
               output_dir / "last.pth")

    # 테스트
    print(f"\nReload best checkpoint: {output_dir / 'best.pth'}", flush=True)
    ckpt = torch.load(output_dir / "best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_result = evaluate_model(model, test_loader, device, criterion, args.use_amp)
    save_test_outputs(output_dir, test_result)

    run_summary = {
        "strategy": args.strategy,
        "port": args.port, "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_fogbdry_f1": best_fogbdry,
        "num_params": num_params,
        "num_params_m": float(num_params) / 1_000_000.0,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    # 전체 summary 업데이트
    row = {
        "strategy": args.strategy, "port": args.port, "seed": args.seed,
        "best_epoch": best_epoch, "params_m": float(num_params) / 1e6,
        **test_result.metrics,
    }
    upsert_csv_row(scratch_root / "summary_scratch_ports.csv", row,
                   key_cols=["strategy", "port", "seed"])

    print("=" * 100, flush=True)
    print("Scratch training finished.", flush=True)
    print(f"best_epoch={best_epoch} | best_val_fogbdry_f1={best_fogbdry:.4f}", flush=True)
    print(f"test_macro_p={test_result.metrics['macro_precision']:.4f} | "
          f"test_macro_r={test_result.metrics['macro_recall']:.4f} | "
          f"test_macro_f1={test_result.metrics['macro_f1']:.4f}", flush=True)
    print(f"test_fog_p={test_result.metrics['fogbdry_precision']:.4f} | "
          f"test_fog_r={test_result.metrics['fogbdry_recall']:.4f} | "
          f"test_fog_f1={test_result.metrics['fogbdry_f1']:.4f}", flush=True)
    print("=" * 100, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",    required=True,
                   choices=["scratch_small", "scratch_large"])
    p.add_argument("--port",        required=True)
    p.add_argument("--data_csv",    required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--img_size",    type=int,   default=512)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--num_workers", type=int,   default=16)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--patience",    type=int,   default=15)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--mid_channels",type=int,   default=128)
    p.add_argument("--feat_dim",    type=int,   default=512)
    p.add_argument("--use_amp",     action="store_true", default=True)
    p.add_argument("--no_amp",      dest="use_amp", action="store_false")
    p.add_argument("--job_idx",     type=int,   default=0)
    p.add_argument("--total_jobs",  type=int,   default=0)
    # backbone 인자 (호환성용, 사용 안 함)
    p.add_argument("--backbone",    default="dummy")
    return p.parse_args()


if __name__ == "__main__":
    main()