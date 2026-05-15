# -*- coding: utf-8 -*-
# pretrain.py — ImageNet-100 pretrain
# 위치: /home/wj/seafog/src/pretrain.py
#
# 사용법:
#   python pretrain.py --backbone convnext --mode base
#   python pretrain.py --backbone convnext --mode typeA_7

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models_erf import build_erf_model


# ── 고정값 ────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 100


# ── 시드 ──────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────
class ImageNet100Dataset(Dataset):
    """
    폴더 구조 (Kaggle imagenet100):
      root/train.X1/n0XXXXXXX/*.JPEG
      root/train.X2/n0XXXXXXX/*.JPEG
      root/train.X3/n0XXXXXXX/*.JPEG
      root/train.X4/n0XXXXXXX/*.JPEG
      root/val.X/n0XXXXXXX/*.JPEG
    """
    def __init__(self, root: str, split: str = "train", transform=None):
        self.transform = transform
        self.samples   = []
        root = Path(root)

        # split 폴더 목록 결정
        if split == "train":
            split_dirs = sorted(root.glob("train.X*"))
        else:
            split_dirs = [root / "val.X"]

        # 전체 클래스 목록 (train.X1 기준으로 수집)
        all_classes = set()
        for sd in split_dirs:
            for d in sd.iterdir():
                if d.is_dir():
                    all_classes.add(d.name)
        classes = sorted(all_classes)
        cls2idx = {c: i for i, c in enumerate(classes)}

        # 샘플 수집
        for sd in split_dirs:
            for cls in classes:
                cls_dir = sd / cls
                if not cls_dir.exists():
                    continue
                for p in cls_dir.iterdir():
                    if p.suffix.lower() in (".jpeg", ".jpg", ".png"):
                        self.samples.append((str(p), cls2idx[cls]))

        print(f"  ImageNet100 [{split}]: {len(self.samples):,}장 / {len(classes)}클래스")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_loaders(data_dir: str, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = ImageNet100Dataset(data_dir, "train", train_tf)
    val_ds   = ImageNet100Dataset(data_dir, "val",   val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ── 학습 루프 ─────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0., 0, 0

    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

        if i % 100 == 0:
            print(f"  [epoch {epoch:03d}/{total_epochs}] "
                  f"step {i}/{len(loader)} | "
                  f"loss={total_loss/total:.4f} | "
                  f"acc={correct/total*100:.2f}%", flush=True)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0., 0, 0

    with torch.cuda.amp.autocast():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ── 메인 ──────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",     type=str, required=True)
    p.add_argument("--mode",         type=str, required=True)
    p.add_argument("--data_dir",     type=str, default="/data1/wj/seafog/data/imagenet100")
    p.add_argument("--save_dir",     type=str, default="/data1/wj/seafog/pretrain_ckpt")
    p.add_argument("--epochs",       type=int, default=90)
    p.add_argument("--batch_size",   type=int, default=512)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # 저장 경로
    run_name = f"{args.backbone}_{args.mode}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 이미 완료된 경우 스킵
    done_flag = save_dir / "pretrain_done.txt"
    if done_flag.exists():
        print(f"[SKIP] {run_name} 이미 완료됨", flush=True)
        return

    print("=" * 80, flush=True)
    print(f"Pretrain | backbone={args.backbone} | mode={args.mode}", flush=True)
    print(f"device={device} | epochs={args.epochs} | batch={args.batch_size}", flush=True)
    print("=" * 80, flush=True)

    # 모델
    model = build_erf_model(
        args.backbone, args.mode,
        num_classes=NUM_CLASSES, pretrained=False,
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {total_params:.1f}M", flush=True)

    # 데이터
    train_loader, val_loader = get_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    # 학습 설정
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.
    logs     = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args.epochs
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
            f"time={elapsed:.0f}s",
            flush=True,
        )

        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
        }
        logs.append(log)

        # best 저장
        if val_acc > best_acc:
            best_acc = val_acc
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(
                {"model_state_dict": state, "epoch": epoch, "val_acc": val_acc},
                save_dir / "best.pth",
            )
            print(f"  -> best.pth saved (val_acc={val_acc*100:.2f}%)", flush=True)

    # last 저장
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(
        {"model_state_dict": state, "epoch": args.epochs},
        save_dir / "last.pth",
    )

    # 로그 저장
    with (save_dir / "pretrain_log.json").open("w") as f:
        json.dump({"args": vars(args), "logs": logs, "best_val_acc": best_acc}, f, indent=2)

    # 완료 플래그
    done_flag.write_text(f"best_val_acc={best_acc:.6f}\n")

    print("=" * 80, flush=True)
    print(f"Pretrain 완료: {run_name} | best_val_acc={best_acc*100:.2f}%", flush=True)
    print(f"저장 경로: {save_dir}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()