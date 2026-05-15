# -*- coding: utf-8 -*-
# train_erf.py — ERF 실험 fine-tune (해무 분류)
# 위치: /home/wj/seafog/src/train_erf.py
#
# 사용법:
#   python train_erf.py \
#     --backbone convnext --mode base \
#     --port daesan \
#     --pretrain_ckpt /data1/wj/seafog/pretrain_ckpt/convnext_base/best.pth

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import SeafogDataset
from models_erf import build_erf_model, load_pretrained_for_finetune

from fvcore.nn import FlopCountAnalysis

CLASS_NAMES = ["normal", "lowvis", "seafog"]


# ── 유틸 ──────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def upsert_csv_row(path: Path, row: dict, key_cols: List[str]) -> None:
    ensure_dir(path.parent)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    row_df = pd.DataFrame([row])
    if not df.empty and all(c in df.columns for c in key_cols):
        mask = pd.Series([True] * len(df))
        for col in key_cols:
            mask &= df[col].astype(str) == str(row[col])
        df = df.loc[~mask].copy()
    pd.concat([df, row_df], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


# ── FLOPs / Params 측정 ───────────────────────────────────────
def get_model_info(model: nn.Module, img_size: int = 512):
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    try:
        device = next(model.parameters()).device
        dummy  = torch.zeros(1, 3, img_size, img_size).to(device)
        flops  = FlopCountAnalysis(model, dummy).total()
        flops_g = flops / 1e9
    except Exception:
        flops_g = -1.0

    return round(params_m, 2), round(flops_g, 2)


# ── 메트릭 ────────────────────────────────────────────────────
def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], average=None, zero_division=0
    )

    def aber(a, b):
        denom = int((labels == a).sum())
        return float(((labels == a) & (preds == b)).sum()) / denom if denom else 0.0

    return {
        "macro_precision":       float(macro_p),
        "macro_recall":          float(macro_r),
        "macro_f1":              float(macro_f1),
        "weighted_precision":    float(weighted_p),
        "weighted_recall":       float(weighted_r),
        "weighted_f1":           float(weighted_f1),
        "normal_precision":      float(per_p[0]),
        "normal_recall":         float(per_r[0]),
        "normal_f1":             float(per_f1[0]),
        "normal_support":        int(per_sup[0]),
        "lowvis_precision":      float(per_p[1]),
        "lowvis_recall":         float(per_r[1]),
        "lowvis_f1":             float(per_f1[1]),
        "lowvis_support":        int(per_sup[1]),
        "seafog_precision":      float(per_p[2]),
        "seafog_recall":         float(per_r[2]),
        "seafog_f1":             float(per_f1[2]),
        "seafog_support":        int(per_sup[2]),
        "aber_normal_to_lowvis": aber(0, 1),
        "aber_normal_to_seafog": aber(0, 2),
        "aber_lowvis_to_normal": aber(1, 0),
        "aber_lowvis_to_seafog": aber(1, 2),
        "aber_seafog_to_lowvis": aber(2, 1),
        "aber_seafog_to_normal": aber(2, 0),
    }


# ── 평가 ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, total_n = 0., 0
    probs_list, labels_list = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        probs_list.append(torch.softmax(logits, 1).cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        total_loss += loss.item() * imgs.size(0)
        total_n    += imgs.size(0)

    probs  = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)
    preds  = probs.argmax(1)

    return total_loss / total_n, compute_metrics(labels, preds), probs, labels, preds


# ── 학습 1 에폭 ───────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss, total_n = 0., 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        total_n    += imgs.size(0)

    return total_loss / total_n


# ── LLRD optimizer ────────────────────────────────────────────
def build_optimizer(model: nn.Module, backbone: str, args) -> AdamW:
    def group(name: str) -> str:
        if any(name.startswith(k) for k in ["head", "fc", "classifier"]):
            return "head"
        if backbone.startswith("convnext"):
            if name.startswith(("stem", "stages.0", "downsample_layers.0")):
                return "first"
            if name.startswith(("stages.1", "stages.2",
                                 "downsample_layers.1", "downsample_layers.2")):
                return "mid"
            return "last"
        if backbone.startswith("tf_efficientnetv2"):
            if name.startswith(("conv_stem", "bn1", "blocks.0", "blocks.1")):
                return "first"
            if name.startswith(("blocks.2", "blocks.3", "blocks.4")):
                return "mid"
            return "last"
        if backbone == "xception":
            if name.startswith(("conv1", "bn1", "conv2", "bn2",
                                 "block1", "block2", "block3", "block4")):
                return "first"
            if name.startswith(("block5", "block6", "block7", "block8")):
                return "mid"
            return "last"
        if backbone.startswith("mobilenet"):
            if name.startswith(("features.0", "features.1", "features.2")):
                return "first"
            if name.startswith(("features.3", "features.4", "features.5",
                                 "features.6", "features.7", "features.8")):
                return "mid"
            return "last"
        return "mid"

    grouped: Dict[str, list] = {"first": [], "mid": [], "last": [], "head": []}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grouped[group(name)].append(param)

    param_groups = []
    lr_map = {
        "first": args.lr * 0.01,
        "mid":   args.lr * 0.1,
        "last":  args.lr * 0.5,
        "head":  args.lr,
    }
    for g, params in grouped.items():
        if params:
            param_groups.append({"params": params, "lr": lr_map[g],
                                  "weight_decay": args.weight_decay})

    return AdamW(param_groups)


# ── 결과 저장 ─────────────────────────────────────────────────
def save_test_outputs(
    output_dir: Path,
    metrics: dict,
    probs: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
) -> None:
    ensure_dir(output_dir)

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix_normalized.csv", encoding="utf-8-sig"
    )

    rows = [
        {
            "index": i,
            "gt": int(labels[i]), "gt_name": CLASS_NAMES[int(labels[i])],
            "pred": int(preds[i]), "pred_name": CLASS_NAMES[int(preds[i])],
            "prob_normal": float(probs[i, 0]),
            "prob_lowvis": float(probs[i, 1]),
            "prob_seafog": float(probs[i, 2]),
        }
        for i in range(len(labels))
    ]
    pd.DataFrame(rows).to_csv(
        output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )


def update_summary(
    summary_path: Path,
    backbone: str,
    mode: str,
    port: str,
    params_m: float,
    flops_g: float,
    best_epoch: int,
    metrics: dict,
) -> None:
    row = {
        "backbone":   backbone,
        "mode":       mode,
        "port":       port,
        "params_m":   params_m,
        "flops_g":    flops_g,
        "best_epoch": best_epoch,
        # macro
        "macro_precision":    metrics["macro_precision"],
        "macro_recall":       metrics["macro_recall"],
        "macro_f1":           metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall":    metrics["weighted_recall"],
        "weighted_f1":        metrics["weighted_f1"],
        # normal
        "normal_precision":   metrics["normal_precision"],
        "normal_recall":      metrics["normal_recall"],
        "normal_f1":          metrics["normal_f1"],
        "normal_support":     metrics["normal_support"],
        # lowvis
        "lowvis_precision":   metrics["lowvis_precision"],
        "lowvis_recall":      metrics["lowvis_recall"],
        "lowvis_f1":          metrics["lowvis_f1"],
        "lowvis_support":     metrics["lowvis_support"],
        # seafog
        "seafog_precision":   metrics["seafog_precision"],
        "seafog_recall":      metrics["seafog_recall"],
        "seafog_f1":          metrics["seafog_f1"],
        "seafog_support":     metrics["seafog_support"],
        # confusion
        "aber_normal_to_lowvis": metrics["aber_normal_to_lowvis"],
        "aber_normal_to_seafog": metrics["aber_normal_to_seafog"],
        "aber_lowvis_to_normal": metrics["aber_lowvis_to_normal"],
        "aber_lowvis_to_seafog": metrics["aber_lowvis_to_seafog"],
        "aber_seafog_to_lowvis": metrics["aber_seafog_to_lowvis"],
        "aber_seafog_to_normal": metrics["aber_seafog_to_normal"],
    }
    upsert_csv_row(summary_path, row, key_cols=["backbone", "mode", "port"])


# ── 메인 ──────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",      type=str, required=True)
    p.add_argument("--mode",          type=str, required=True)
    p.add_argument("--port",          type=str, required=True)
    p.add_argument("--data_csv",      type=str,
                   default="/data1/wj/seafog/data/splits.csv")
    p.add_argument("--pretrain_ckpt", type=str, required=True)
    p.add_argument("--output_root",   type=str,
                   default="/data1/wj/seafog/results/erf")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--img_size",      type=int,   default=512)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--num_workers",   type=int,   default=8)
    p.add_argument("--use_amp",       action="store_true", default=True)
    p.add_argument("--no_amp",        dest="use_amp", action="store_false")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    run_name   = f"{args.backbone}_{args.mode}"
    output_dir = Path(args.output_root) / run_name / args.port
    ensure_dir(output_dir)

    summary_path = Path(args.output_root) / "summary_erf.csv"

    # 이미 완료된 경우 스킵
    done_flag = output_dir / "done.txt"
    if done_flag.exists():
        print(f"[SKIP] {run_name} / {args.port} 이미 완료됨", flush=True)
        return

    print("=" * 80, flush=True)
    print(f"Fine-tune | {run_name} | port={args.port}", flush=True)
    print(f"device={device} | epochs={args.epochs} | batch={args.batch_size}", flush=True)
    print("=" * 80, flush=True)

    # 모델 로드
    model = load_pretrained_for_finetune(
        backbone=args.backbone,
        mode=args.mode,
        pretrain_ckpt=args.pretrain_ckpt,
        num_classes=3,
    ).to(device)

    params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    flops_g = -1.0
    print(f"params={params_m}M | FLOPs=TBD", flush=True)

    # 데이터로더
    train_ds = SeafogDataset(args.data_csv, args.port, "train",  img_size=args.img_size)
    valid_ds = SeafogDataset(args.data_csv, args.port, "valid",  img_size=args.img_size)
    test_ds  = SeafogDataset(args.data_csv, args.port, "test",   img_size=args.img_size)

    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=True, drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    valid_loader = DataLoader(valid_ds, shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)

    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.backbone, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.cuda.amp.GradScaler()

    best_val_f1    = -1.
    best_epoch     = -1
    patience_count = 0
    val_history    = []
    train_history  = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.use_amp
        )
        val_loss, val_metrics, _, _, _ = evaluate(
            model, valid_loader, criterion, device, args.use_amp
        )
        scheduler.step()

        val_f1 = val_metrics["macro_f1"]

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_p={val_metrics['macro_precision']:.4f} | "
            f"val_macro_r={val_metrics['macro_recall']:.4f} | "
            f"val_macro_f1={val_f1:.4f} | "
            f"val_normal_p={val_metrics['normal_precision']:.4f} "
            f"r={val_metrics['normal_recall']:.4f} "
            f"f1={val_metrics['normal_f1']:.4f} | "
            f"val_lowvis_p={val_metrics['lowvis_precision']:.4f} "
            f"r={val_metrics['lowvis_recall']:.4f} "
            f"f1={val_metrics['lowvis_f1']:.4f} | "
            f"val_seafog_p={val_metrics['seafog_precision']:.4f} "
            f"r={val_metrics['seafog_recall']:.4f} "
            f"f1={val_metrics['seafog_f1']:.4f}",
            flush=True,
        )

        train_history.append({"epoch": epoch, "train_loss": train_loss})
        val_history.append({"epoch": epoch, "val_loss": val_loss, **val_metrics})

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            best_epoch     = epoch
            patience_count = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_macro_f1": val_f1},
                       output_dir / "best.pth")
            print(f"  -> best.pth saved (val_macro_f1={val_f1:.4f})", flush=True)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    # 히스토리 저장
    pd.DataFrame(train_history).to_csv(output_dir / "train_history.csv",
                                        index=False, encoding="utf-8-sig")
    pd.DataFrame(val_history).to_csv(output_dir / "val_history.csv",
                                      index=False, encoding="utf-8-sig")

    # 테스트
    print(f"\nReload best.pth (epoch={best_epoch})", flush=True)
    ckpt = torch.load(output_dir / "best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_metrics, probs, labels, preds = evaluate(
        model, test_loader, criterion, device, args.use_amp
    )

    save_test_outputs(output_dir, test_metrics, probs, labels, preds)
    update_summary(summary_path, args.backbone, args.mode, args.port,
                   params_m, flops_g, best_epoch, test_metrics)

    done_flag.write_text(f"best_epoch={best_epoch} val_macro_f1={best_val_f1:.6f}\n")

    print("=" * 80, flush=True)
    print(f"완료: {run_name} / {args.port}", flush=True)
    print(
        f"test_macro_p={test_metrics['macro_precision']:.4f} | "
        f"test_macro_r={test_metrics['macro_recall']:.4f} | "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}",
        flush=True,
    )
    print(
        f"normal  p={test_metrics['normal_precision']:.4f} "
        f"r={test_metrics['normal_recall']:.4f} "
        f"f1={test_metrics['normal_f1']:.4f}",
        flush=True,
    )
    print(
        f"lowvis  p={test_metrics['lowvis_precision']:.4f} "
        f"r={test_metrics['lowvis_recall']:.4f} "
        f"f1={test_metrics['lowvis_f1']:.4f}",
        flush=True,
    )
    print(
        f"seafog  p={test_metrics['seafog_precision']:.4f} "
        f"r={test_metrics['seafog_recall']:.4f} "
        f"f1={test_metrics['seafog_f1']:.4f}",
        flush=True,
    )
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()