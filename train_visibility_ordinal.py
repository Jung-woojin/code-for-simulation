#!/usr/bin/env python3
"""
train_visibility_ordinal.py

Single-image port CCTV visibility-state training script.
Supports:
  - CE softmax baseline
  - CDW-CE distance-weighted softmax baseline
  - ordinal BCE head
  - ORAH/CORV conditional ordinal risk-aware head

Expected CSV columns, detected automatically when possible:
  path column: image_path / path / filepath / file / filename
  label column: label / class / target / y / gt
  split column: split, values train/val/test (optional)
  port column: port / harbor / site (optional)

Label mapping:
  0 normal / 1 lowvis / 2 seafog
  Also accepts common strings: normal, clear, lowvis, reduced, seafog, fog, etc.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError as e:
    raise SystemExit("timm is required. Install with: pip install timm") from e

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        cohen_kappa_score,
        f1_score,
    )
    from sklearn.model_selection import train_test_split
except ImportError as e:
    raise SystemExit("scikit-learn is required. Install with: pip install scikit-learn") from e


LABEL_MAP = {
    "0": 0,
    "normal": 0,
    "clear": 0,
    "good": 0,
    "visibility_normal": 0,
    "보통": 0,
    "보통시정": 0,
    "1": 1,
    "lowvis": 1,
    "low_visibility": 1,
    "reduced": 1,
    "reduced_visibility": 1,
    "degraded": 1,
    "저시정": 1,
    "2": 2,
    "seafog": 2,
    "sea_fog": 2,
    "fog": 2,
    "foggy": 2,
    "해무": 2,
}

MODEL_ALIASES = {
    "resnet50": "resnet50",
    "convnext_tiny": "convnext_tiny",
    "convnext": "convnext_tiny",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet": "efficientnet_b3",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "mobilenetv3_large_100": "mobilenetv3_large_100",
    "mobilenet": "mobilenetv3_large_100",
    "xception": "xception",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", required=True)
    p.add_argument("--port", default="all", help="Port/site name to filter. Use 'all' to disable filtering.")
    p.add_argument("--backbone", required=True, help="resnet50, convnext_tiny, efficientnet_b3, mobilenetv3_large, xception, or timm model name")
    p.add_argument("--method", required=True, choices=["ce", "cdwce", "ordinal", "orah"])
    p.add_argument("--output_root", required=True)
    p.add_argument("--pretrained", action="store_true", help="Use timm ImageNet pretrained weights.")
    p.add_argument("--checkpoint", default="", help="Optional local checkpoint to load into backbone/model with strict=False.")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3, help="Freeze backbone for this many epochs.")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--risk_lambda", type=float, default=0.3)
    p.add_argument("--risk_power", type=float, default=2.0, help="1 for abs distance, 2 for squared distance.")
    p.add_argument("--cdw_alpha", type=float, default=2.0)
    p.add_argument("--val_ratio", type=float, default=0.15, help="Used only if split column is missing.")
    p.add_argument("--test_ratio", type=float, default=0.15, help="Used only if split column is missing.")
    p.add_argument("--image_root", default="", help="Optional root prepended to relative paths.")
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--monitor", default="macro_f1", choices=["macro_f1", "qwk", "mave", "eer"])
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def parse_label(v) -> int:
    if pd.isna(v):
        raise ValueError("NaN label")
    s = str(v).strip().lower()
    if s in LABEL_MAP:
        return LABEL_MAP[s]
    try:
        i = int(float(s))
        if i in (0, 1, 2):
            return i
    except Exception:
        pass
    raise ValueError(f"Unknown label value: {v!r}")


def load_dataframe(args: argparse.Namespace) -> Tuple[pd.DataFrame, str, str, Optional[str], Optional[str]]:
    df = pd.read_csv(args.data_csv)
    path_col = find_col(df, ["image_path", "path", "filepath", "file", "filename", "img", "image"])
    label_col = find_col(df, ["label", "class", "class_label", "target", "y", "gt", "true_label"])
    split_col = find_col(df, ["split", "set", "subset"])
    port_col = find_col(df, ["port", "harbor", "site", "location"])
    if path_col is None or label_col is None:
        raise ValueError(f"Could not detect path/label columns. Columns={list(df.columns)}")

    if args.port.lower() != "all" and port_col is not None:
        df = df[df[port_col].astype(str).str.lower() == args.port.lower()].copy()
    elif args.port.lower() != "all" and port_col is None:
        print("[WARN] --port was specified but no port column was found; no port filtering applied.")

    df["__label__"] = df[label_col].apply(parse_label)
    df["__path__"] = df[path_col].astype(str)
    if args.image_root:
        root = Path(args.image_root)
        df["__path__"] = df["__path__"].apply(lambda p: str(root / p) if not os.path.isabs(p) else p)
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows after filtering.")
    return df, path_col, label_col, split_col, port_col


def make_splits(df: pd.DataFrame, split_col: Optional[str], args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    if split_col is not None:
        split_values = df[split_col].astype(str).str.lower()
        out = {
            "train": df[split_values == "train"].copy(),
            "val": df[split_values.isin(["val", "valid", "validation"])].copy(),
            "test": df[split_values == "test"].copy(),
        }
        if len(out["train"]) and len(out["val"]) and len(out["test"]):
            return out
        print("[WARN] split column found but train/val/test incomplete; creating stratified split.")

    train_df, tmp_df = train_test_split(
        df,
        test_size=args.val_ratio + args.test_ratio,
        random_state=args.seed,
        stratify=df["__label__"],
    )
    rel_test = args.test_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=rel_test,
        random_state=args.seed,
        stratify=tmp_df["__label__"],
    )
    return {"train": train_df.copy(), "val": val_df.copy(), "test": test_df.copy()}


class VisibilityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["__path__"]
        y = int(row["__label__"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {path}") from e
        if self.transform:
            img = self.transform(img)
        return img, y, path


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02)], p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, eval_tf


class VisibilityModel(nn.Module):
    def __init__(self, backbone_name: str, method: str, pretrained: bool, checkpoint: str = ""):
        super().__init__()
        self.method = method
        timm_name = MODEL_ALIASES.get(backbone_name, backbone_name)
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = int(self.backbone.num_features)
        out_dim = 3 if method in ("ce", "cdwce") else 2
        self.head = nn.Linear(feat_dim, out_dim)
        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, checkpoint: str) -> None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        # Accept checkpoints with or without model/backbone prefixes.
        cleaned = {}
        for k, v in state.items():
            nk = k
            for prefix in ["module.", "model."]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        print(f"[CKPT] loaded {checkpoint}")
        print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)


def ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    return torch.stack([(y >= 1).float(), (y >= 2).float()], dim=1)


def probs_from_orah_logits(logits: torch.Tensor) -> torch.Tensor:
    ab = torch.sigmoid(logits)
    a = ab[:, 0]
    b = ab[:, 1]
    p0 = 1.0 - a
    p1 = a * (1.0 - b)
    p2 = a * b
    probs = torch.stack([p0, p1, p2], dim=1)
    return torch.clamp(probs, 1e-6, 1.0)


def probs_from_ordinal_logits(logits: torch.Tensor) -> torch.Tensor:
    q = torch.sigmoid(logits)
    q1 = q[:, 0]
    q2 = torch.minimum(q[:, 1], q1)  # monotonic repair for inference/loss probability reconstruction
    p0 = 1.0 - q1
    p1 = q1 - q2
    p2 = q2
    probs = torch.stack([p0, p1, p2], dim=1)
    return torch.clamp(probs, 1e-6, 1.0)


def cdwce_loss(logits: torch.Tensor, y: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    n = probs.shape[1]
    idx = torch.arange(n, device=logits.device).view(1, -1)
    dist = (idx - y.view(-1, 1)).abs().float().pow(alpha)
    # Penalize probability assigned to wrong distant classes.
    loss = -torch.log(torch.clamp(1.0 - probs, min=1e-6)) * dist
    return loss.sum(dim=1).mean()


def loss_fn(method: str, logits: torch.Tensor, y: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if method == "ce":
        return F.cross_entropy(logits, y)
    if method == "cdwce":
        return cdwce_loss(logits, y, alpha=args.cdw_alpha)

    t = ordinal_targets(y)
    if method == "ordinal":
        return F.binary_cross_entropy_with_logits(logits, t)

    if method == "orah":
        # Conditional boundary loss.
        a_logit = logits[:, 0]
        b_logit = logits[:, 1]
        t1 = (y >= 1).float()
        t2 = (y >= 2).float()
        l1 = F.binary_cross_entropy_with_logits(a_logit, t1)
        mask = t1 > 0.5
        if mask.any():
            l2 = F.binary_cross_entropy_with_logits(b_logit[mask], t2[mask])
        else:
            l2 = torch.zeros((), device=logits.device)
        probs = probs_from_orah_logits(logits)
        idx = torch.arange(3, device=logits.device).view(1, -1)
        dist = (idx - y.view(-1, 1)).abs().float().pow(args.risk_power)
        risk = (probs * dist).sum(dim=1).mean()
        return l1 + l2 + args.risk_lambda * risk

    raise ValueError(method)


def logits_to_probs(method: str, logits: torch.Tensor) -> torch.Tensor:
    if method in ("ce", "cdwce"):
        return F.softmax(logits, dim=1)
    if method == "ordinal":
        return probs_from_ordinal_logits(logits)
    if method == "orah":
        return probs_from_orah_logits(logits)
    raise ValueError(method)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = np.abs(y_pred - y_true)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mave": float(np.mean(diff)),
        "eer": float(np.mean(diff == 2)),
        "aer": float(np.mean(diff == 1)),
        "svr": float(np.mean(diff ** 2)),
    }
    return metrics


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace):
    model.eval()
    losses, y_true, y_pred, paths_all, probs_all = [], [], [], [], []
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(args.method, logits, y, args)
            probs = logits_to_probs(args.method, logits)
            pred = probs.argmax(dim=1)
            losses.append(float(loss.item()) * len(y))
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            paths_all.extend(list(paths))
            probs_all.append(probs.cpu().numpy())
    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)
    metrics = compute_metrics(y_true_np, y_pred_np)
    metrics["loss"] = float(np.sum(losses) / max(1, len(y_true_np)))
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1, 2])
    probs_np = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, 3))
    return metrics, cm, y_true_np, y_pred_np, probs_np, paths_all


def make_optimizer(model: VisibilityModel, args: argparse.Namespace, train_backbone: bool) -> torch.optim.Optimizer:
    for p in model.backbone.parameters():
        p.requires_grad = train_backbone
    for p in model.head.parameters():
        p.requires_grad = True
    params = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr_backbone},
        {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": args.lr_head},
    ]
    return torch.optim.AdamW(params, weight_decay=args.weight_decay)


def is_better(current: Dict[str, float], best: Optional[Dict[str, float]], monitor: str) -> bool:
    if best is None:
        return True
    if monitor in ("mave", "eer"):
        return current[monitor] < best[monitor]
    return current[monitor] > best[monitor]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, _, _, split_col, _ = load_dataframe(args)
    splits = make_splits(df, split_col, args)
    train_tf, eval_tf = build_transforms(args.img_size)
    datasets = {
        "train": VisibilityDataset(splits["train"], train_tf),
        "val": VisibilityDataset(splits["val"], eval_tf),
        "test": VisibilityDataset(splits["test"], eval_tf),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
    }

    out_dir = Path(args.output_root) / args.port / args.backbone / args.method / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    for name, sdf in splits.items():
        sdf.to_csv(out_dir / f"split_{name}.csv", index=False)

    print(f"[INFO] device={device}")
    print(f"[INFO] output={out_dir}")
    print(f"[INFO] sizes train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])}")

    model = VisibilityModel(args.backbone, args.method, pretrained=args.pretrained, checkpoint=args.checkpoint).to(device)
    optimizer = make_optimizer(model, args, train_backbone=(args.warmup_epochs <= 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_metrics = None
    best_epoch = -1
    wait = 0
    history = []
    current_train_backbone = args.warmup_epochs <= 0

    for epoch in range(1, args.epochs + 1):
        if (not current_train_backbone) and epoch > args.warmup_epochs:
            print(f"[INFO] Unfreezing backbone at epoch {epoch}")
            current_train_backbone = True
            optimizer = make_optimizer(model, args, train_backbone=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch + 1))

        model.train()
        total_loss = 0.0
        total_n = 0
        t0 = time.time()
        for x, y, _ in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)
                loss = loss_fn(args.method, logits, y, args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(y)
            total_n += len(y)
        scheduler.step()

        val_metrics, val_cm, *_ = evaluate(model, loaders["val"], device, args)
        train_loss = total_loss / max(1, total_n)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | val_qwk={val_metrics['qwk']:.4f} | "
            f"val_mave={val_metrics['mave']:.4f} | val_eer={val_metrics['eer']:.4f} | {elapsed:.1f}s"
        )

        if is_better(val_metrics, best_metrics, args.monitor):
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            wait = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "val_metrics": val_metrics,
            }, out_dir / "best.pth")
            np.savetxt(out_dir / "best_val_confusion_matrix.csv", val_cm, fmt="%d", delimiter=",")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[INFO] Early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break

    ckpt = torch.load(out_dir / "best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    results = {"best_epoch": best_epoch, "best_val": best_metrics}
    for split_name in ["val", "test"]:
        metrics, cm, y_true, y_pred, probs, paths = evaluate(model, loaders[split_name], device, args)
        results[split_name] = metrics
        np.savetxt(out_dir / f"{split_name}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
        if args.save_preds:
            pred_df = pd.DataFrame({
                "path": paths,
                "y_true": y_true,
                "y_pred": y_pred,
                "p_normal": probs[:, 0],
                "p_lowvis": probs[:, 1],
                "p_seafog": probs[:, 2],
            })
            pred_df.to_csv(out_dir / f"{split_name}_predictions.csv", index=False)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_dir / "done.txt", "w", encoding="utf-8") as f:
        f.write("done\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
