from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

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


def build_model(backbone: str, num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def build_dataloaders(
    data_csv: str,
    port: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SeafogDataset(
        splits_csv=data_csv,
        port=port,
        split="train",
        img_size=img_size,
    )
    valid_ds = SeafogDataset(
        splits_csv=data_csv,
        port=port,
        split="valid",
        img_size=img_size,
    )
    test_ds = SeafogDataset(
        splits_csv=data_csv,
        port=port,
        split="test",
        img_size=img_size,
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    valid_loader = DataLoader(valid_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, valid_loader, test_loader


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Supports:
    1) dict batch:
       {"image": ..., "label": ..., optional "path"/"filepath"/"img_path"}
    2) tuple/list batch:
       (images, labels) or (images, labels, paths)
    """
    if isinstance(batch, dict):
        image_key = next((k for k in ["image", "img", "images"] if k in batch), None)
        label_key = next((k for k in ["label", "labels", "target", "targets"] if k in batch), None)
        path_key = next(
            (k for k in ["path", "paths", "filepath", "filepaths", "img_path", "img_paths"] if k in batch),
            None,
        )

        if image_key is None or label_key is None:
            raise KeyError("Batch dict must contain image and label keys.")

        imgs = batch[image_key]
        labels = batch[label_key]

        if path_key is None:
            paths = [""] * len(labels)
        else:
            raw_paths = batch[path_key]
            if isinstance(raw_paths, (list, tuple)):
                paths = [str(x) for x in raw_paths]
            else:
                paths = [str(x) for x in raw_paths]

        return imgs, labels, paths

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            imgs, labels = batch
            return imgs, labels, [""] * len(labels)
        if len(batch) >= 3:
            imgs, labels, raw_paths = batch[0], batch[1], batch[2]
            if isinstance(raw_paths, (list, tuple)):
                paths = [str(x) for x in raw_paths]
            else:
                paths = [str(x) for x in raw_paths]
            return imgs, labels, paths

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def infer_group_name(backbone: str, param_name: str) -> str:
    # Head
    if (
        param_name.startswith("fc")
        or param_name.startswith("classifier")
        or param_name.startswith("head")
    ):
        return "head"

    # ResNet family
    if backbone.startswith("resnet"):
        if param_name.startswith(("conv1", "bn1", "layer1")):
            return "first"
        if param_name.startswith(("layer2", "layer3")):
            return "mid"
        if param_name.startswith("layer4"):
            return "last"
        return "mid"

    # Xception
    if backbone == "xception":
        if param_name.startswith(("conv1", "bn1", "conv2", "bn2")):
            return "first"
        if param_name.startswith(("block1", "block2", "block3", "block4")):
            return "first"
        if param_name.startswith(("block5", "block6", "block7", "block8")):
            return "mid"
        if param_name.startswith(
            ("block9", "block10", "block11", "block12", "conv3", "bn3", "conv4", "bn4")
        ):
            return "last"
        return "mid"

    # EfficientNetV2
    if backbone.startswith("tf_efficientnetv2"):
        if param_name.startswith(("conv_stem", "bn1", "blocks.0", "blocks.1")):
            return "first"
        if param_name.startswith(("blocks.2", "blocks.3", "blocks.4")):
            return "mid"
        if param_name.startswith(("blocks.5", "blocks.6", "conv_head", "bn2")):
            return "last"
        return "mid"

    # ConvNeXt
    if backbone.startswith("convnext"):
        if param_name.startswith(("stem", "downsample_layers.0", "stages.0")):
            return "first"
        if param_name.startswith(
            ("downsample_layers.1", "stages.1", "downsample_layers.2", "stages.2")
        ):
            return "mid"
        if param_name.startswith(("downsample_layers.3", "stages.3", "norm_pre")):
            return "last"
        return "mid"

    return "mid"


def build_llrd_optimizer(
    model: nn.Module,
    backbone: str,
    lr_head: float = 1e-3,
    lr_last: float = 1e-4,
    lr_mid: float = 1e-5,
    lr_first: float = 1e-6,
    weight_decay: float = 1e-4,
) -> AdamW:
    grouped: Dict[str, List[nn.Parameter]] = {
        "first": [],
        "mid": [],
        "last": [],
        "head": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_name = infer_group_name(backbone, name)
        grouped[group_name].append(param)

    param_groups = []
    if grouped["first"]:
        param_groups.append({"params": grouped["first"], "lr": lr_first, "weight_decay": weight_decay})
    if grouped["mid"]:
        param_groups.append({"params": grouped["mid"], "lr": lr_mid, "weight_decay": weight_decay})
    if grouped["last"]:
        param_groups.append({"params": grouped["last"], "lr": lr_last, "weight_decay": weight_decay})
    if grouped["head"]:
        param_groups.append({"params": grouped["head"], "lr": lr_head, "weight_decay": weight_decay})

    return AdamW(param_groups)


def get_model_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0,
    )

    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            labels,
            preds,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
    )

    fog_mask = np.isin(labels, [1, 2])
    fog_labels = labels[fog_mask]
    fog_preds = preds[fog_mask]

    if len(fog_labels) > 0:
        fog_precision, fog_recall, fog_f1, _ = precision_recall_fscore_support(
            fog_labels,
            fog_preds,
            labels=[1, 2],
            average="macro",
            zero_division=0,
        )
    else:
        fog_precision, fog_recall, fog_f1 = 0.0, 0.0, 0.0

    def aber(a: int, b: int) -> float:
        denom = int((labels == a).sum())
        if denom == 0:
            return 0.0
        num = int(((labels == a) & (preds == b)).sum())
        return float(num) / float(denom)

    return {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "fogbdry_precision": float(fog_precision),
        "fogbdry_recall": float(fog_recall),
        "fogbdry_f1": float(fog_f1),
        "normal_precision": float(per_class_precision[0]),
        "lowvis_precision": float(per_class_precision[1]),
        "seafog_precision": float(per_class_precision[2]),
        "normal_recall": float(per_class_recall[0]),
        "lowvis_recall": float(per_class_recall[1]),
        "seafog_recall": float(per_class_recall[2]),
        "normal_f1": float(per_class_f1[0]),
        "lowvis_f1": float(per_class_f1[1]),
        "seafog_f1": float(per_class_f1[2]),
        "normal_support": int(per_class_support[0]),
        "lowvis_support": int(per_class_support[1]),
        "seafog_support": int(per_class_support[2]),
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
            imgs, labels, paths = unpack_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_count += bs

            probs_list.append(probs.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
            paths_list.extend(paths)

    probs_np = np.concatenate(probs_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)
    preds_np = probs_np.argmax(axis=1)

    metrics = compute_metrics(labels_np, preds_np)
    mean_loss = total_loss / max(total_count, 1)

    return EvalResult(
        loss=float(mean_loss),
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
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        imgs, labels, _ = unpack_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_val_fogbdry_f1: float,
    args: argparse.Namespace,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_fogbdry_f1": best_val_fogbdry_f1,
            "args": vars(args),
        },
        path,
    )


def save_test_outputs(output_dir: Path, eval_result: EvalResult) -> None:
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"test_loss": eval_result.loss, **eval_result.metrics},
            f,
            ensure_ascii=False,
            indent=2,
        )

    cm = confusion_matrix(eval_result.labels, eval_result.preds, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_norm / row_sums
    cm_norm_df = pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_norm_df.to_csv(output_dir / "confusion_matrix_normalized.csv", encoding="utf-8-sig")

    class_rows = []
    for idx, class_name in enumerate(CLASS_NAMES):
        class_rows.append(
            {
                "class_id": idx,
                "class_name": class_name,
                "precision": eval_result.metrics[f"{class_name}_precision"],
                "recall": eval_result.metrics[f"{class_name}_recall"],
                "f1": eval_result.metrics[f"{class_name}_f1"],
                "support": eval_result.metrics[f"{class_name}_support"],
            }
        )

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

    rows = []
    for i in range(len(eval_result.labels)):
        rows.append(
            {
                "index": i,
                "filepath": eval_result.sample_paths[i] if i < len(eval_result.sample_paths) else "",
                "gt": int(eval_result.labels[i]),
                "gt_name": CLASS_NAMES[int(eval_result.labels[i])],
                "pred": int(eval_result.preds[i]),
                "pred_name": CLASS_NAMES[int(eval_result.preds[i])],
                "prob_normal": float(eval_result.probs[i, 0]),
                "prob_lowvis": float(eval_result.probs[i, 1]),
                "prob_seafog": float(eval_result.probs[i, 2]),
            }
        )

    pd.DataFrame(rows).to_csv(
        output_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )


def update_port_summary(
    track_a_root: Path,
    backbone: str,
    port: str,
    best_epoch: int,
    num_params: int,
    test_result: EvalResult,
) -> None:
    row = {
        "backbone": backbone,
        "port": port,
        "best_epoch": best_epoch,
        "params_m": float(num_params) / 1_000_000.0,
        "macro_precision": test_result.metrics["macro_precision"],
        "macro_recall": test_result.metrics["macro_recall"],
        "macro_f1": test_result.metrics["macro_f1"],
        "weighted_precision": test_result.metrics["weighted_precision"],
        "weighted_recall": test_result.metrics["weighted_recall"],
        "weighted_f1": test_result.metrics["weighted_f1"],
        "fogbdry_precision": test_result.metrics["fogbdry_precision"],
        "fogbdry_recall": test_result.metrics["fogbdry_recall"],
        "fogbdry_f1": test_result.metrics["fogbdry_f1"],
        "normal_precision": test_result.metrics["normal_precision"],
        "lowvis_precision": test_result.metrics["lowvis_precision"],
        "seafog_precision": test_result.metrics["seafog_precision"],
        "normal_recall": test_result.metrics["normal_recall"],
        "lowvis_recall": test_result.metrics["lowvis_recall"],
        "seafog_recall": test_result.metrics["seafog_recall"],
        "normal_f1": test_result.metrics["normal_f1"],
        "lowvis_f1": test_result.metrics["lowvis_f1"],
        "seafog_f1": test_result.metrics["seafog_f1"],
        "normal_support": test_result.metrics["normal_support"],
        "lowvis_support": test_result.metrics["lowvis_support"],
        "seafog_support": test_result.metrics["seafog_support"],
        "aber_normal_to_lowvis": test_result.metrics["aber_normal_to_lowvis"],
        "aber_lowvis_to_normal": test_result.metrics["aber_lowvis_to_normal"],
        "aber_lowvis_to_seafog": test_result.metrics["aber_lowvis_to_seafog"],
        "aber_seafog_to_lowvis": test_result.metrics["aber_seafog_to_lowvis"],
        "aber_seafog_to_normal": test_result.metrics["aber_seafog_to_normal"],
        "aber_normal_to_seafog": test_result.metrics["aber_normal_to_seafog"],
    }
    upsert_csv_row(track_a_root / "summary_table_a_ports.csv", row, key_cols=["backbone", "port"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--total_jobs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    ensure_dir(output_dir)
    track_a_root = output_dir.parents[1]

    print("=" * 100, flush=True)
    print(
        f"Track A | job={args.job_idx}/{args.total_jobs} | "
        f"backbone={args.backbone} | port={args.port}",
        flush=True,
    )
    print(f"device      : {device}", flush=True)
    print(f"output_dir  : {output_dir}", flush=True)
    print(f"epochs      : {args.epochs}", flush=True)
    print(f"batch_size  : {args.batch_size}", flush=True)
    print(f"img_size    : {args.img_size}", flush=True)
    print(f"num_workers : {args.num_workers}", flush=True)
    print(f"patience    : {args.patience}", flush=True)
    print("=" * 100, flush=True)

    train_loader, valid_loader, test_loader = build_dataloaders(
        data_csv=args.data_csv,
        port=args.port,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.backbone, num_classes=3, pretrained=True).to(device)
    num_params = get_model_num_params(model)

    print(f"num_params  : {num_params:,} ({num_params / 1_000_000.0:.2f}M)", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_llrd_optimizer(
        model=model,
        backbone=args.backbone,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)
    else:
        scaler = torch.amp.GradScaler("cpu", enabled=False)

    train_history: List[dict] = []
    val_history: List[dict] = []

    best_val_fogbdry_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    final_epoch = 0

    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch

        print(
            f"[JOB {args.job_idx}/{args.total_jobs}] "
            f"[{args.backbone} | {args.port}] "
            f"Epoch {epoch}/{args.epochs} START",
            flush=True,
        )

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=args.use_amp,
        )

        val_result = evaluate_model(
            model=model,
            loader=valid_loader,
            device=device,
            criterion=criterion,
            use_amp=args.use_amp,
        )

        scheduler.step()

        lr_first = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else None
        lr_mid = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else None
        lr_last = optimizer.param_groups[2]["lr"] if len(optimizer.param_groups) > 2 else None
        lr_head = optimizer.param_groups[3]["lr"] if len(optimizer.param_groups) > 3 else None

        train_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "lr_first": lr_first,
                "lr_mid": lr_mid,
                "lr_last": lr_last,
                "lr_head": lr_head,
            }
        )

        val_history.append(
            {
                "epoch": epoch,
                "val_loss": float(val_result.loss),
                "val_macro_precision": val_result.metrics["macro_precision"],
                "val_macro_recall": val_result.metrics["macro_recall"],
                "val_macro_f1": val_result.metrics["macro_f1"],
                "val_fogbdry_precision": val_result.metrics["fogbdry_precision"],
                "val_fogbdry_recall": val_result.metrics["fogbdry_recall"],
                "val_fogbdry_f1": val_result.metrics["fogbdry_f1"],
                "val_normal_precision": val_result.metrics["normal_precision"],
                "val_normal_recall": val_result.metrics["normal_recall"],
                "val_normal_f1": val_result.metrics["normal_f1"],
                "val_lowvis_precision": val_result.metrics["lowvis_precision"],
                "val_lowvis_recall": val_result.metrics["lowvis_recall"],
                "val_lowvis_f1": val_result.metrics["lowvis_f1"],
                "val_seafog_precision": val_result.metrics["seafog_precision"],
                "val_seafog_recall": val_result.metrics["seafog_recall"],
                "val_seafog_f1": val_result.metrics["seafog_f1"],
            }
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_result.loss:.4f} | "
            f"val_macro_p={val_result.metrics['macro_precision']:.4f} | "
            f"val_macro_r={val_result.metrics['macro_recall']:.4f} | "
            f"val_macro_f1={val_result.metrics['macro_f1']:.4f} | "
            f"val_fog_p={val_result.metrics['fogbdry_precision']:.4f} | "
            f"val_fog_r={val_result.metrics['fogbdry_recall']:.4f} | "
            f"val_fog_f1={val_result.metrics['fogbdry_f1']:.4f}",
            flush=True,
        )

        improved = val_result.metrics["fogbdry_f1"] > best_val_fogbdry_f1
        if improved:
            best_val_fogbdry_f1 = val_result.metrics["fogbdry_f1"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                output_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_fogbdry_f1=best_val_fogbdry_f1,
                args=args,
            )
            print(
                f"  -> best.pth updated | epoch={epoch} | "
                f"best_val_fogbdry_f1={best_val_fogbdry_f1:.4f}",
                flush=True,
            )
        else:
            patience_counter += 1
            print(
                f"  -> no improvement | patience {patience_counter}/{args.patience}",
                flush=True,
            )

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}", flush=True)
            break

    save_checkpoint(
        output_dir / "last.pth",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=final_epoch,
        best_val_fogbdry_f1=best_val_fogbdry_f1,
        args=args,
    )

    save_csv_dicts(train_history, output_dir / "train_history.csv")
    save_csv_dicts(val_history, output_dir / "val_history.csv")

    print(f"\nReload best checkpoint: {output_dir / 'best.pth'}", flush=True)
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

    run_summary = {
        "backbone": args.backbone,
        "port": args.port,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_ran": final_epoch,
        "best_epoch": best_epoch,
        "best_val_fogbdry_f1": best_val_fogbdry_f1,
        "num_params": num_params,
        "num_params_m": float(num_params) / 1_000_000.0,
        "best_checkpoint": str(output_dir / "best.pth"),
        "test_metrics_file": str(output_dir / "test_metrics.json"),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    update_port_summary(
        track_a_root=track_a_root,
        backbone=args.backbone,
        port=args.port,
        best_epoch=best_epoch,
        num_params=num_params,
        test_result=test_result,
    )

    print("=" * 100, flush=True)
    print("Training finished.", flush=True)
    print(f"Saved to: {output_dir}", flush=True)
    print(f"best_epoch={best_epoch}, best_val_fogbdry_f1={best_val_fogbdry_f1:.4f}", flush=True)
    print(
        f"test_macro_p={test_result.metrics['macro_precision']:.4f} | "
        f"test_macro_r={test_result.metrics['macro_recall']:.4f} | "
        f"test_macro_f1={test_result.metrics['macro_f1']:.4f}",
        flush=True,
    )
    print(
        f"test_fog_p={test_result.metrics['fogbdry_precision']:.4f} | "
        f"test_fog_r={test_result.metrics['fogbdry_recall']:.4f} | "
        f"test_fog_f1={test_result.metrics['fogbdry_f1']:.4f}",
        flush=True,
    )
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()