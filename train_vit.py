#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_vit.py — SwinT / ViT 해무 finetune
# 위치: /home/wj/seafog/src/train_vit.py
#
# 사용법:
#   python train_vit.py --backbone swin --port haeundae \
#       --pretrain_ckpt /data1/wj/seafog/pretrain_ckpt/swin_base/best.pth

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
import timm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import SeafogDataset

CLASS_NAMES = ["normal", "lowvis", "seafog"]

VIT_MODEL_MAP = {
    'swin': 'swin_base_patch4_window7_224',
    'vit':  'vit_base_patch16_224',
}

IMG_SIZE_MAP = {
    'swin': 512,
    'vit':  512,
}


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
    df     = pd.read_csv(path) if path.exists() else pd.DataFrame()
    row_df = pd.DataFrame([row])
    if not df.empty and all(c in df.columns for c in key_cols):
        mask = pd.Series([True] * len(df))
        for col in key_cols:
            mask &= df[col].astype(str) == str(row[col])
        df = df.loc[~mask].copy()
    pd.concat([df, row_df], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


# ── 모델 빌더 ─────────────────────────────────────────────────
def build_vit_for_finetune(backbone: str, pretrain_ckpt: str, num_classes: int = 3) -> nn.Module:
    """pretrain ckpt에서 weight 로드 후 head 교체"""
    timm_name = VIT_MODEL_MAP[backbone]
    img_size  = IMG_SIZE_MAP[backbone]

    # pretrain과 동일 구조로 먼저 로드 (224)
    model_pretrain = timm.create_model(timm_name, pretrained=False, num_classes=100, img_size=224)
    ckpt  = torch.load(pretrain_ckpt, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model_pretrain.load_state_dict(state, strict=False)

    # finetune용 모델 (img_size 지정 → timm이 pos_embed interpolation 자동 처리)
    model = timm.create_model(timm_name, pretrained=False, num_classes=100, img_size=img_size)
    model.load_state_dict(model_pretrain.state_dict(), strict=False)
    del model_pretrain
    print(f"  Loaded pretrain ckpt: {pretrain_ckpt}")

    # head 교체 (timm 구조별)
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head  = nn.Linear(in_features, num_classes)
    elif hasattr(model, "head") and hasattr(model.head, "fc"):
        in_features   = model.head.fc.in_features
        model.head.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):
        in_features      = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc    = nn.Linear(in_features, num_classes)

    print(f"  Head replaced → num_classes={num_classes}")
    return model


# ── LLRD optimizer ────────────────────────────────────────────
def build_optimizer(model: nn.Module, backbone: str, args) -> AdamW:
    def group(name: str) -> str:
        if any(name.startswith(k) for k in ["head", "fc", "classifier"]):
            return "head"
        if backbone == "swin":
            if name.startswith(("patch_embed", "layers.0")):
                return "first"
            if name.startswith(("layers.1", "layers.2")):
                return "mid"
            return "last"
        if backbone == "vit":
            if name.startswith(("patch_embed", "blocks.0", "blocks.1", "blocks.2", "blocks.3")):
                return "first"
            if name.startswith(("blocks.4", "blocks.5", "blocks.6", "blocks.7")):
                return "mid"
            return "last"
        return "mid"

    grouped: Dict[str, list] = {"first": [], "mid": [], "last": [], "head": []}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grouped[group(name)].append(param)

    lr_map = {
        "first": args.lr * 0.01,
        "mid":   args.lr * 0.1,
        "last":  args.lr * 0.5,
        "head":  args.lr,
    }
    param_groups = []
    for g, params in grouped.items():
        if params:
            param_groups.append({"params": params, "lr": lr_map[g],
                                  "weight_decay": args.weight_decay})
    return AdamW(param_groups)


# ── 메트릭 ────────────────────────────────────────────────────
def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0,1,2], average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0,1,2], average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        labels, preds, labels=[0,1,2], average=None, zero_division=0
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
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
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

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        total_n    += imgs.size(0)

    return total_loss / total_n


# ── 결과 저장 ─────────────────────────────────────────────────
def save_test_outputs(output_dir, metrics, probs, labels, preds):
    ensure_dir(output_dir)

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(labels, preds, labels=[0,1,2])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix_normalized.csv", encoding="utf-8-sig"
    )
    rows = [{"index": i,
             "gt": int(labels[i]), "gt_name": CLASS_NAMES[int(labels[i])],
             "pred": int(preds[i]), "pred_name": CLASS_NAMES[int(preds[i])],
             "prob_normal": float(probs[i,0]),
             "prob_lowvis": float(probs[i,1]),
             "prob_seafog": float(probs[i,2])}
            for i in range(len(labels))]
    pd.DataFrame(rows).to_csv(output_dir / "test_predictions.csv",
                               index=False, encoding="utf-8-sig")


def update_summary(summary_path, backbone, mode, port, params_m, best_epoch, metrics):
    row = {
        "backbone": backbone, "mode": mode, "port": port,
        "params_m": params_m, "flops_g": -1, "best_epoch": best_epoch,
        **{k: metrics[k] for k in [
            "macro_precision","macro_recall","macro_f1",
            "weighted_precision","weighted_recall","weighted_f1",
            "normal_precision","normal_recall","normal_f1","normal_support",
            "lowvis_precision","lowvis_recall","lowvis_f1","lowvis_support",
            "seafog_precision","seafog_recall","seafog_f1","seafog_support",
            "aber_normal_to_lowvis","aber_normal_to_seafog",
            "aber_lowvis_to_normal","aber_lowvis_to_seafog",
            "aber_seafog_to_lowvis","aber_seafog_to_normal",
        ]}
    }
    upsert_csv_row(summary_path, row, key_cols=["backbone","mode","port"])


# ── 메인 ──────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",      type=str, required=True, choices=['swin','vit'])
    p.add_argument("--port",          type=str, required=True)
    p.add_argument("--data_csv",      type=str, default="/data1/wj/seafog/data/splits.csv")
    p.add_argument("--pretrain_ckpt", type=str, required=True)
    p.add_argument("--output_root",   type=str, default="/data1/wj/seafog/results/erf")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=32)
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

    img_size   = IMG_SIZE_MAP[args.backbone]
    run_name   = f"{args.backbone}_base"
    output_dir = Path(args.output_root) / run_name / args.port
    ensure_dir(output_dir)

    summary_path = Path(args.output_root) / "summary_erf.csv"

    done_flag = output_dir / "done.txt"
    if done_flag.exists():
        print(f"[SKIP] {run_name} / {args.port} 이미 완료됨")
        return

    print("=" * 80)
    print(f"Fine-tune | {run_name} | port={args.port}")
    print(f"device={device} | epochs={args.epochs} | batch={args.batch_size} | img_size={img_size}")
    print("=" * 80)

    model = build_vit_for_finetune(args.backbone, args.pretrain_ckpt, num_classes=3).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={params_m:.2f}M | FLOPs=TBD")

    # 데이터로더 (img_size 512)
    train_ds = SeafogDataset(args.data_csv, args.port, "train", img_size=img_size)
    valid_ds = SeafogDataset(args.data_csv, args.port, "valid", img_size=img_size)
    test_ds  = SeafogDataset(args.data_csv, args.port, "test",  img_size=img_size)

    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=True, drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    valid_loader = DataLoader(valid_ds, shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)

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
            f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_p={val_metrics['macro_precision']:.4f} | "
            f"val_macro_r={val_metrics['macro_recall']:.4f} | "
            f"val_macro_f1={val_f1:.4f} | "
            f"val_normal_p={val_metrics['normal_precision']:.4f} "
            f"r={val_metrics['normal_recall']:.4f} f1={val_metrics['normal_f1']:.4f} | "
            f"val_lowvis_p={val_metrics['lowvis_precision']:.4f} "
            f"r={val_metrics['lowvis_recall']:.4f} f1={val_metrics['lowvis_f1']:.4f} | "
            f"val_seafog_p={val_metrics['seafog_precision']:.4f} "
            f"r={val_metrics['seafog_recall']:.4f} f1={val_metrics['seafog_f1']:.4f}",
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

    pd.DataFrame(train_history).to_csv(output_dir / "train_history.csv",
                                        index=False, encoding="utf-8-sig")
    pd.DataFrame(val_history).to_csv(output_dir / "val_history.csv",
                                      index=False, encoding="utf-8-sig")

    print(f"\nReload best.pth (epoch={best_epoch})", flush=True)
    ckpt = torch.load(output_dir / "best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_metrics, probs, labels, preds = evaluate(
        model, test_loader, criterion, device, args.use_amp
    )

    save_test_outputs(output_dir, test_metrics, probs, labels, preds)
    update_summary(summary_path, args.backbone, "base", args.port,
                   round(params_m, 2), best_epoch, test_metrics)

    done_flag.write_text(f"best_epoch={best_epoch} val_macro_f1={best_val_f1:.6f}\n")

    print("=" * 80)
    print(f"완료: {run_name} / {args.port}")
    print(f"test_macro_p={test_metrics['macro_precision']:.4f} | "
          f"test_macro_r={test_metrics['macro_recall']:.4f} | "
          f"test_macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"normal  p={test_metrics['normal_precision']:.4f} "
          f"r={test_metrics['normal_recall']:.4f} f1={test_metrics['normal_f1']:.4f}")
    print(f"lowvis  p={test_metrics['lowvis_precision']:.4f} "
          f"r={test_metrics['lowvis_recall']:.4f} f1={test_metrics['lowvis_f1']:.4f}")
    print(f"seafog  p={test_metrics['seafog_precision']:.4f} "
          f"r={test_metrics['seafog_recall']:.4f} f1={test_metrics['seafog_f1']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
