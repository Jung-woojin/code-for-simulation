"""
Experiment-only Track C trainer.

This file keeps the original staged fine-tuning flow, but writes all outputs
under `/data1/wj/seafog/results/experiment` and imports the experiment-only
context model implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from context_branch_experiment import build_context_model
from datasets import SeafogDataset


CLASS_NAMES = ["normal", "lowvis", "seafog"]


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
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
            merged = pd.concat([df.loc[~mask].copy(), row_df], ignore_index=True)

    merged.to_csv(path, index=False, encoding="utf-8-sig")


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    if isinstance(batch, dict):
        image_key = next((k for k in ["image", "img", "images"] if k in batch), None)
        label_key = next((k for k in ["label", "labels", "target", "targets"] if k in batch), None)
        path_key = next(
            (k for k in ["path", "paths", "filepath", "filepaths", "img_path", "img_paths"] if k in batch),
            None,
        )
        if image_key is None or label_key is None:
            raise KeyError("Batch dict must contain image and label keys.")
        images = batch[image_key]
        labels = batch[label_key]
        paths = [str(x) for x in batch[path_key]] if path_key else [""] * len(labels)
        return images, labels, paths

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            images, labels = batch
            return images, labels, [""] * len(labels)
        if len(batch) >= 3:
            images, labels, raw_paths = batch[0], batch[1], batch[2]
            return images, labels, [str(x) for x in raw_paths]

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0,
    )
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    fog_mask = np.isin(labels, [1, 2])
    fog_labels = labels[fog_mask]
    fog_preds = preds[fog_mask]
    if len(fog_labels) > 0:
        fog_p, fog_r, fog_f1, _ = precision_recall_fscore_support(
            fog_labels,
            fog_preds,
            labels=[1, 2],
            average="macro",
            zero_division=0,
        )
    else:
        fog_p = fog_r = fog_f1 = 0.0

    def aber(a: int, b: int) -> float:
        denom = int((labels == a).sum())
        return float(((labels == a) & (preds == b)).sum()) / denom if denom else 0.0

    return {
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "fogbdry_precision": float(fog_p),
        "fogbdry_recall": float(fog_r),
        "fogbdry_f1": float(fog_f1),
        "normal_precision": float(per_p[0]),
        "lowvis_precision": float(per_p[1]),
        "seafog_precision": float(per_p[2]),
        "normal_recall": float(per_r[0]),
        "lowvis_recall": float(per_r[1]),
        "seafog_recall": float(per_r[2]),
        "normal_f1": float(per_f1[0]),
        "lowvis_f1": float(per_f1[1]),
        "seafog_f1": float(per_f1[2]),
        "normal_support": int(per_sup[0]),
        "lowvis_support": int(per_sup[1]),
        "seafog_support": int(per_sup[2]),
        "aber_normal_to_lowvis": aber(0, 1),
        "aber_lowvis_to_normal": aber(1, 0),
        "aber_lowvis_to_seafog": aber(1, 2),
        "aber_seafog_to_lowvis": aber(2, 1),
        "aber_seafog_to_normal": aber(2, 0),
        "aber_normal_to_seafog": aber(0, 2),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_count = 0
    probs_list = []
    labels_list = []
    paths_list: List[str] = []

    with torch.no_grad():
        for batch in loader:
            images, labels, paths = unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            probs_list.append(probs.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
            paths_list.extend(paths)

    probs_np = np.concatenate(probs_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)
    preds_np = probs_np.argmax(axis=1)
    metrics = compute_metrics(labels_np, preds_np)

    return EvalResult(
        loss=float(total_loss / max(total_count, 1)),
        metrics=metrics,
        probs=probs_np,
        labels=labels_np,
        preds=preds_np,
        sample_paths=paths_list,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        images, labels, _ = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def save_test_outputs(output_dir: Path, eval_result: EvalResult) -> None:
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"test_loss": eval_result.loss, **eval_result.metrics},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    cm = confusion_matrix(eval_result.labels, eval_result.preds, labels=[0, 1, 2])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pd.DataFrame(cm_norm / row_sums, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix_normalized.csv",
        encoding="utf-8-sig",
    )

    class_rows = [
        {
            "class_id": idx,
            "class_name": class_name,
            "precision": eval_result.metrics[f"{class_name}_precision"],
            "recall": eval_result.metrics[f"{class_name}_recall"],
            "f1": eval_result.metrics[f"{class_name}_f1"],
            "support": eval_result.metrics[f"{class_name}_support"],
        }
        for idx, class_name in enumerate(CLASS_NAMES)
    ]
    class_rows.extend(
        [
            {
                "class_id": -1,
                "class_name": "macro_avg",
                "precision": eval_result.metrics["macro_precision"],
                "recall": eval_result.metrics["macro_recall"],
                "f1": eval_result.metrics["macro_f1"],
                "support": int(len(eval_result.labels)),
            },
            {
                "class_id": -2,
                "class_name": "weighted_avg",
                "precision": eval_result.metrics["weighted_precision"],
                "recall": eval_result.metrics["weighted_recall"],
                "f1": eval_result.metrics["weighted_f1"],
                "support": int(len(eval_result.labels)),
            },
            {
                "class_id": -3,
                "class_name": "fog_boundary_macro",
                "precision": eval_result.metrics["fogbdry_precision"],
                "recall": eval_result.metrics["fogbdry_recall"],
                "f1": eval_result.metrics["fogbdry_f1"],
                "support": int(
                    eval_result.metrics["lowvis_support"] + eval_result.metrics["seafog_support"]
                ),
            },
        ]
    )
    pd.DataFrame(class_rows).to_csv(
        output_dir / "class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_rows = [
        {
            "index": idx,
            "filepath": eval_result.sample_paths[idx] if idx < len(eval_result.sample_paths) else "",
            "gt": int(eval_result.labels[idx]),
            "gt_name": CLASS_NAMES[int(eval_result.labels[idx])],
            "pred": int(eval_result.preds[idx]),
            "pred_name": CLASS_NAMES[int(eval_result.preds[idx])],
            "prob_normal": float(eval_result.probs[idx, 0]),
            "prob_lowvis": float(eval_result.probs[idx, 1]),
            "prob_seafog": float(eval_result.probs[idx, 2]),
        }
        for idx in range(len(eval_result.labels))
    ]
    pd.DataFrame(pred_rows).to_csv(
        output_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )


def update_port_summary(
    result_root: Path,
    backbone: str,
    strategy: str,
    model_strategy: str,
    port: str,
    seed: int,
    best_epoch: int,
    num_params: int,
    stage_kernels: Optional[List[int]],
    alpha_values: List[float],
    test_result: EvalResult,
) -> None:
    row = {
        "backbone": backbone,
        "strategy": strategy,
        "model_strategy": model_strategy,
        "port": port,
        "seed": seed,
        "best_epoch": best_epoch,
        "params_m": float(num_params) / 1_000_000.0,
        "stage_kernels": ",".join(str(v) for v in stage_kernels) if stage_kernels else "",
        "alpha_values": ",".join(f"{v:.6f}" for v in alpha_values),
        **{
            key: test_result.metrics[key]
            for key in [
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
                "fogbdry_precision",
                "fogbdry_recall",
                "fogbdry_f1",
                "normal_precision",
                "lowvis_precision",
                "seafog_precision",
                "normal_recall",
                "lowvis_recall",
                "seafog_recall",
                "normal_f1",
                "lowvis_f1",
                "seafog_f1",
                "normal_support",
                "lowvis_support",
                "seafog_support",
                "aber_normal_to_lowvis",
                "aber_lowvis_to_normal",
                "aber_lowvis_to_seafog",
                "aber_seafog_to_lowvis",
                "aber_seafog_to_normal",
                "aber_normal_to_seafog",
            ]
        },
    }
    upsert_csv_row(
        result_root / "summary_table_c_ports.csv",
        row,
        key_cols=["backbone", "strategy", "port", "seed"],
    )


def run_stage(
    stage_idx: int,
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
    output_dir: Path,
    args: argparse.Namespace,
    max_epochs: int,
    patience: int,
    no_early_stop_first_n: int = 0,
    prev_best_macro_f1: float = -1.0,
) -> Tuple[float, int, List[dict]]:
    if stage_idx == 1:
        param_groups = model.get_param_groups_stage1(
            branch_lr=args.branch_lr_s1,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
    elif stage_idx == 2:
        param_groups = model.get_param_groups_stage2(
            trunk_last_lr=args.trunk_lr_s2,
            branch_lr=args.branch_lr_s2,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
    else:
        param_groups = model.get_param_groups_stage3(
            lr_head=args.head_lr,
            lr_last=args.trunk_lr_s3_last,
            lr_mid=args.trunk_lr_s3_mid,
            lr_first=args.trunk_lr_s3_first,
            branch_lr=args.branch_lr_s3,
            weight_decay=args.weight_decay,
        )

    active_groups = [group for group in param_groups if any(p.requires_grad for p in group["params"])]
    if not active_groups:
        print(f"  [Stage {stage_idx}] no trainable parameters, skip", flush=True)
        return prev_best_macro_f1, -1, []

    optimizer = AdamW(active_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-7)
    scaler = (
        torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
        if device.type == "cuda"
        else torch.cuda.amp.GradScaler(enabled=False)
    )

    best_macro_f1 = prev_best_macro_f1
    best_epoch = -1
    patience_count = 0
    history: List[dict] = []
    ckpt_path = output_dir / f"stage{stage_idx}_best.pth"

    print("=" * 100, flush=True)
    print(
        f"[Stage {stage_idx}] backbone={args.backbone} | "
        f"strategy={args.strategy_label} | model_strategy={args.model_strategy} | "
        f"port={args.port} | seed={args.seed} | max_epochs={max_epochs} | patience={patience}",
        flush=True,
    )
    for group in active_groups:
        count = sum(p.numel() for p in group["params"] if p.requires_grad)
        print(f"  {group['name']:25s} lr={group['lr']:.0e} params={count:,}", flush=True)
    print("=" * 100, flush=True)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_result = evaluate_model(
            model=model,
            loader=valid_loader,
            device=device,
            criterion=criterion,
            use_amp=use_amp,
        )
        scheduler.step()

        alphas = model.get_alphas()
        gate_str = ""
        if hasattr(model, "get_gate_value"):
            try:
                sample_batch = next(iter(valid_loader))
                sample_images = unpack_batch(sample_batch)[0][:4].to(device)
                gate_val = model.get_gate_value(sample_images)
                gate_str = f" gate={gate_val:.4f}"
            except Exception:
                gate_str = ""

        alpha_str = " ".join(f"alpha[{idx}]={value:.4f}" for idx, value in enumerate(alphas))
        macro_f1 = val_result.metrics["macro_f1"]
        fogbdry = val_result.metrics["fogbdry_f1"]

        history.append(
            {
                "stage": stage_idx,
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": val_result.loss,
                "val_macro_precision": val_result.metrics["macro_precision"],
                "val_macro_recall": val_result.metrics["macro_recall"],
                "val_macro_f1": val_result.metrics["macro_f1"],
                "val_fogbdry_precision": val_result.metrics["fogbdry_precision"],
                "val_fogbdry_recall": val_result.metrics["fogbdry_recall"],
                "val_fogbdry_f1": val_result.metrics["fogbdry_f1"],
                "val_lowvis_f1": val_result.metrics["lowvis_f1"],
                "val_seafog_f1": val_result.metrics["seafog_f1"],
                **{f"alpha_{idx}": value for idx, value in enumerate(alphas)},
            }
        )

        print(
            f"[S{stage_idx} Ep{epoch:03d}/{max_epochs:03d}] "
            f"train={train_loss:.4f} | val_loss={val_result.loss:.4f} | "
            f"val_macro_f1={val_result.metrics['macro_f1']:.4f} | "
            f"val_fog_f1={fogbdry:.4f} | "
            f"val_lowvis_f1={val_result.metrics['lowvis_f1']:.4f} | "
            f"val_seafog_f1={val_result.metrics['seafog_f1']:.4f} | "
            f"{alpha_str}{gate_str}",
            flush=True,
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "stage": stage_idx,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_macro_f1": best_macro_f1,
                    "best_val_fogbdry_f1": fogbdry,
                    "alphas": alphas,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(
                f"  -> best.pth updated | epoch={epoch} | "
                f"best_val_macro_f1={best_macro_f1:.4f} | "
                f"best_val_fogbdry_f1={fogbdry:.4f}",
                flush=True,
            )
        else:
            patience_count += 1
            print(f"  -> no improvement | patience {patience_count}/{patience}", flush=True)

        if patience_count >= patience and epoch > no_early_stop_first_n:
            print(f"Early stopping triggered at epoch {epoch}", flush=True)
            break

    return best_macro_f1, best_epoch, history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    result_root = Path(args.result_root)
    ensure_dir(output_dir)
    ensure_dir(result_root)

    stage_kernels = (
        [int(token) for token in args.stage_kernels.split(",")]
        if args.stage_kernels
        else None
    )

    print("=" * 100, flush=True)
    print(
        f"Track C Experiment | job={args.job_idx}/{args.total_jobs} | "
        f"backbone={args.backbone} | strategy={args.strategy_label} | "
        f"model_strategy={args.model_strategy} | port={args.port} | seed={args.seed}",
        flush=True,
    )
    print(f"device={device} | output={output_dir}", flush=True)
    if stage_kernels:
        print(f"stage_kernels={stage_kernels}", flush=True)
    print("=" * 100, flush=True)

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        SeafogDataset(args.data_csv, args.port, "train", img_size=args.img_size),
        shuffle=True,
        **common,
    )
    valid_loader = DataLoader(
        SeafogDataset(args.data_csv, args.port, "valid", img_size=args.img_size),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        SeafogDataset(args.data_csv, args.port, "test", img_size=args.img_size),
        shuffle=False,
        **common,
    )

    model = build_context_model(
        backbone=args.backbone,
        strategy=args.model_strategy,
        num_classes=3,
        img_size=args.img_size,
        stage_kernel_sizes=stage_kernels,
    ).to(device)

    num_params = sum(param.numel() for param in model.parameters())
    print(f"num_params: {num_params:,} ({num_params / 1_000_000.0:.2f}M)", flush=True)

    criterion = nn.CrossEntropyLoss()
    all_history: List[dict] = []

    best_macro_f1, stage1_best_epoch, hist1 = run_stage(
        stage_idx=1,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        criterion=criterion,
        use_amp=args.use_amp,
        output_dir=output_dir,
        args=args,
        max_epochs=args.epochs_s1,
        patience=args.patience_s1,
    )
    all_history.extend(hist1)

    s1_ckpt = output_dir / "stage1_best.pth"
    if s1_ckpt.exists():
        ckpt = torch.load(s1_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_macro_f1 = float(ckpt["best_val_macro_f1"])
        print(f"[Stage 1 best loaded] macro_f1={best_macro_f1:.4f}", flush=True)

    best_macro_f1, stage2_best_epoch, hist2 = run_stage(
        stage_idx=2,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        criterion=criterion,
        use_amp=args.use_amp,
        output_dir=output_dir,
        args=args,
        max_epochs=args.epochs_s2,
        patience=args.patience_s2,
        prev_best_macro_f1=best_macro_f1,
    )
    all_history.extend(hist2)

    s2_ckpt = output_dir / "stage2_best.pth"
    if s2_ckpt.exists():
        ckpt = torch.load(s2_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_macro_f1 = float(ckpt["best_val_macro_f1"])
        print(f"[Stage 2 best loaded] macro_f1={best_macro_f1:.4f}", flush=True)

    best_macro_f1, stage3_best_epoch, hist3 = run_stage(
        stage_idx=3,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        criterion=criterion,
        use_amp=args.use_amp,
        output_dir=output_dir,
        args=args,
        max_epochs=args.epochs_s3,
        patience=args.patience_s3,
        no_early_stop_first_n=args.no_early_stop_first_n,
        prev_best_macro_f1=best_macro_f1,
    )
    all_history.extend(hist3)

    final_best_macro_f1 = best_macro_f1
    final_best_fogbdry = None
    final_best_epoch = -1
    best_checkpoint_name = ""
    for ckpt_name in ["stage3_best.pth", "stage2_best.pth", "stage1_best.pth"]:
        ckpt_path = output_dir / ckpt_name
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            final_best_macro_f1 = float(ckpt["best_val_macro_f1"])
            final_best_fogbdry = float(ckpt.get("best_val_fogbdry_f1", -1.0))
            final_best_epoch = int(ckpt.get("epoch", -1))
            shutil.copy(ckpt_path, output_dir / "best.pth")
            best_checkpoint_name = ckpt_name
            print(
                f"[Final best from {ckpt_name}] "
                f"epoch={final_best_epoch} | "
                f"macro_f1={final_best_macro_f1:.4f} | "
                f"fogbdry_f1={final_best_fogbdry:.4f}",
                flush=True,
            )
            break

    save_csv_dicts(all_history, output_dir / "val_history.csv")
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, output_dir / "last.pth")

    print(f"Reload best checkpoint: {output_dir / 'best.pth'}", flush=True)
    ckpt = torch.load(output_dir / "best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_result = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        use_amp=args.use_amp,
    )
    save_test_outputs(output_dir, test_result)

    alpha_values = model.get_alphas()
    run_summary = {
        "backbone": args.backbone,
        "strategy": args.strategy_label,
        "model_strategy": args.model_strategy,
        "port": args.port,
        "seed": args.seed,
        "stage_kernels": stage_kernels,
        "schedule_info": {
            "strategy_label": args.strategy_label,
            "model_strategy": args.model_strategy,
            "stage_kernels": stage_kernels,
        },
        "stage_best_epochs": {
            "stage1": stage1_best_epoch,
            "stage2": stage2_best_epoch,
            "stage3": stage3_best_epoch,
        },
        "best_epoch": final_best_epoch,
        "selection_metric": "macro_f1",
        "best_val_macro_f1": final_best_macro_f1,
        "best_val_fogbdry_f1": final_best_fogbdry,
        "num_params": num_params,
        "num_params_m": float(num_params) / 1_000_000.0,
        "best_checkpoint": str(output_dir / "best.pth"),
        "best_checkpoint_source": best_checkpoint_name,
        "test_metrics_file": str(output_dir / "test_metrics.json"),
        "alpha": alpha_values,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)

    update_port_summary(
        result_root=result_root,
        backbone=args.backbone,
        strategy=args.strategy_label,
        model_strategy=args.model_strategy,
        port=args.port,
        seed=args.seed,
        best_epoch=final_best_epoch,
        num_params=num_params,
        stage_kernels=stage_kernels,
        alpha_values=alpha_values,
        test_result=test_result,
    )

    print("=" * 100, flush=True)
    print("Training finished.", flush=True)
    print(f"Saved to: {output_dir}", flush=True)
    print(
        f"best_epoch={final_best_epoch} | "
        f"best_val_macro_f1={final_best_macro_f1:.4f} | "
        f"best_val_fogbdry_f1={final_best_fogbdry:.4f}",
        flush=True,
    )
    print(
        f"test_macro_f1={test_result.metrics['macro_f1']:.4f} | "
        f"test_fogbdry_f1={test_result.metrics['fogbdry_f1']:.4f} | "
        f"test_lowvis_f1={test_result.metrics['lowvis_f1']:.4f} | "
        f"test_seafog_f1={test_result.metrics['seafog_f1']:.4f}",
        flush=True,
    )
    print(f"alphas={alpha_values}", flush=True)
    print("=" * 100, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--strategy_label", required=True)
    parser.add_argument(
        "--model_strategy",
        required=True,
        choices=["baseline", "parallel", "stageaware"],
    )
    parser.add_argument("--port", required=True)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--stage_kernels", default=None)
    parser.add_argument("--epochs_s1", type=int, default=30)
    parser.add_argument("--epochs_s2", type=int, default=30)
    parser.add_argument("--epochs_s3", type=int, default=40)
    parser.add_argument("--patience_s1", type=int, default=5)
    parser.add_argument("--patience_s2", type=int, default=7)
    parser.add_argument("--patience_s3", type=int, default=10)
    parser.add_argument("--no_early_stop_first_n", type=int, default=3)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--branch_lr_s1", type=float, default=1e-3)
    parser.add_argument("--branch_lr_s2", type=float, default=2e-4)
    parser.add_argument("--branch_lr_s3", type=float, default=1e-4)
    parser.add_argument("--trunk_lr_s2", type=float, default=5e-5)
    parser.add_argument("--trunk_lr_s3_last", type=float, default=2e-5)
    parser.add_argument("--trunk_lr_s3_mid", type=float, default=1e-5)
    parser.add_argument("--trunk_lr_s3_first", type=float, default=1e-6)
    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--total_jobs", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
