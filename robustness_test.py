# -*- coding: utf-8 -*-
# robustness_test.py — Texture Perturbation & Structure Disruption Test
# 위치: /home/wj/seafog/src/robustness_test.py
#
# 사용법:
#   python robustness_test.py --test texture --port haeundae
#   python robustness_test.py --test structure --port haeundae
#   python robustness_test.py --test all --port haeundae
#   python robustness_test.py --test all --port haeundae --port2 yeosu

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models_erf import load_pretrained_for_finetune

# ── 설정 ──────────────────────────────────────────────────────
CLASS_NAMES   = ["normal", "lowvis", "seafog"]
DATA_CSV      = "/data1/wj/seafog/data/splits.csv"
RESULT_ROOT   = "/data1/wj/seafog/results/erf"
PRETRAIN_ROOT = "/data1/wj/seafog/pretrain_ckpt"
OUTPUT_ROOT   = "/data1/wj/seafog/results/robustness"

# 대상 모델 쌍 (backbone, mode)
TARGET_MODELS = [
    ("xception",     "base"),
    ("xception",     "typeB_7"),
    ("mobilenet",    "base"),
    ("mobilenet",    "typeA_11"),
    ("convnext",     "base"),       # negative control
    ("convnext",     "typeA_15"),   # negative control
    ("efficientnet", "base"),
    ("efficientnet", "typeA_7"),
]

# Texture Perturbation 설정
TEXTURE_PERTURBS = {
    "gaussian_noise": [0.05, 0.10, 0.20],
    "jpeg_compression": [30, 20, 10],
    "gaussian_blur": [3, 5, 9],
    "contrast_jitter": [0.3, 0.5, 0.8],
}

# Structure Disruption 설정
STRUCTURE_DISRUPT = {
    "patch_shuffle": [4, 8],
    "hband_shuffle": [4, 8],
    "horizon_mask": ["mid"],       # 중간 1/3 마스킹
    "top_mask": ["top"],           # 상단 1/3 마스킹
    "center_crop": [0.5, 0.75],    # 중앙 N% crop 후 resize
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Perturbation 함수들 ────────────────────────────────────────

def apply_gaussian_noise(img_np: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian noise 추가 (sigma: 0~1 범위 기준)"""
    noise = np.random.randn(*img_np.shape).astype(np.float32) * sigma * 255
    out   = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def apply_jpeg_compression(img_np: np.ndarray, quality: int) -> np.ndarray:
    """JPEG 압축 열화"""
    _, enc = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def apply_gaussian_blur(img_np: np.ndarray, kernel: int) -> np.ndarray:
    """Gaussian blur"""
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(img_np, (k, k), 0)


def apply_contrast_jitter(img_np: np.ndarray, factor: float) -> np.ndarray:
    """Local contrast jitter: 픽셀을 mean 방향으로 수렴"""
    mean = img_np.mean()
    out  = img_np.astype(np.float32) * (1 - factor) + mean * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_patch_shuffle(img_np: np.ndarray, n: int) -> np.ndarray:
    """이미지를 n×n 패치로 나눠서 무작위 섞기"""
    H, W, C = img_np.shape
    ph, pw   = H // n, W // n
    patches  = []
    for i in range(n):
        for j in range(n):
            patches.append(img_np[i*ph:(i+1)*ph, j*pw:(j+1)*pw].copy())
    random.shuffle(patches)
    out = img_np.copy()
    idx = 0
    for i in range(n):
        for j in range(n):
            out[i*ph:(i+1)*ph, j*pw:(j+1)*pw] = patches[idx]
            idx += 1
    return out


def apply_hband_shuffle(img_np: np.ndarray, n: int) -> np.ndarray:
    """수평 band를 n등분하여 무작위 섞기"""
    H, W, C = img_np.shape
    bh       = H // n
    bands    = [img_np[i*bh:(i+1)*bh].copy() for i in range(n)]
    random.shuffle(bands)
    return np.concatenate(bands, axis=0)


def apply_region_mask(img_np: np.ndarray, region: str) -> np.ndarray:
    """특정 영역을 mean 값으로 마스킹"""
    H, W, C = img_np.shape
    out      = img_np.copy().astype(np.float32)
    mean_val = img_np.mean()
    if region == "mid":
        out[H//3 : 2*H//3, :] = mean_val
    elif region == "top":
        out[:H//3, :]          = mean_val
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_center_crop(img_np: np.ndarray, ratio: float, target_size: int = 512) -> np.ndarray:
    """중앙 ratio 비율만큼 crop 후 target_size로 resize"""
    H, W, C = img_np.shape
    ch, cw   = int(H * ratio), int(W * ratio)
    sh, sw   = (H - ch) // 2, (W - cw) // 2
    crop     = img_np[sh:sh+ch, sw:sw+cw]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def apply_perturbation(img_np: np.ndarray, perturb_type: str, strength) -> np.ndarray:
    if perturb_type == "gaussian_noise":
        return apply_gaussian_noise(img_np, strength)
    elif perturb_type == "jpeg_compression":
        return apply_jpeg_compression(img_np, int(strength))
    elif perturb_type == "gaussian_blur":
        return apply_gaussian_blur(img_np, int(strength))
    elif perturb_type == "contrast_jitter":
        return apply_contrast_jitter(img_np, strength)
    elif perturb_type == "patch_shuffle":
        return apply_patch_shuffle(img_np, int(strength))
    elif perturb_type == "hband_shuffle":
        return apply_hband_shuffle(img_np, int(strength))
    elif perturb_type == "horizon_mask":
        return apply_region_mask(img_np, "mid")
    elif perturb_type == "top_mask":
        return apply_region_mask(img_np, "top")
    elif perturb_type == "center_crop":
        return apply_center_crop(img_np, strength)
    else:
        raise ValueError(f"Unknown perturbation: {perturb_type}")


# ── Dataset ───────────────────────────────────────────────────
class PerturbedDataset(Dataset):
    def __init__(self, data_csv: str, port: str, split: str = "test",
                 img_size: int = 512,
                 perturb_type: str = None, strength=None):
        df = pd.read_csv(data_csv)
        df = df[(df["port"] == port) & (df["split"] == split)]

        self.samples      = list(zip(df["filepath"].tolist(),
                                     df["class_label"].tolist()))
        self.label_map    = {"normal": 0, "lowvis": 1, "seafog": 2}
        self.img_size     = img_size
        self.perturb_type = perturb_type
        self.strength     = strength

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        print(f"  Dataset | port={port} split={split} | {len(self.samples)}장 "
              f"| perturb={perturb_type}({strength})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_str = self.samples[idx]
        label = self.label_map[label_str]

        img = Image.open(path).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        img_np = np.array(img)

        if self.perturb_type is not None:
            img_np = apply_perturbation(img_np, self.perturb_type, self.strength)

        tensor = self.normalize(Image.fromarray(img_np))
        return tensor, label


# ── 모델 로드 ─────────────────────────────────────────────────
def load_model(backbone: str, mode: str, port: str, device: torch.device) -> nn.Module:
    pretrain_ckpt = f"{PRETRAIN_ROOT}/{backbone}_{mode}/best.pth"
    finetune_ckpt = f"{RESULT_ROOT}/{backbone}_{mode}/{port}/best.pth"

    model = load_pretrained_for_finetune(
        backbone=backbone, mode=mode,
        pretrain_ckpt=pretrain_ckpt, num_classes=3,
    )
    ckpt = torch.load(finetune_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model.to(device).eval()


# ── 평가 ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_loader(model, loader, device) -> Dict:
    preds_all, labels_all = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds_all.append(logits.argmax(1).cpu().numpy())
        labels_all.append(labels.numpy())

    preds  = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0,1,2], average="macro", zero_division=0
    )
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0,1,2], average=None, zero_division=0
    )
    return {
        "macro_f1":      float(macro_f1),
        "macro_p":       float(macro_p),
        "macro_r":       float(macro_r),
        "normal_f1":     float(per_f1[0]),
        "lowvis_f1":     float(per_f1[1]),
        "seafog_f1":     float(per_f1[2]),
        "seafog_recall": float(per_r[2]),
    }


# ── 메인 실험 ─────────────────────────────────────────────────
def run_test(args, perturb_config: Dict, test_name: str):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ports     = [args.port]
    if args.port2:
        ports.append(args.port2)

    output_dir = Path(OUTPUT_ROOT) / test_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for port in ports:
        print(f"\n{'='*60}")
        print(f"Port: {port} | Test: {test_name}")
        print(f"{'='*60}")

        for backbone, mode in TARGET_MODELS:
            model_name = f"{backbone}_{mode}"
            print(f"\n--- {model_name} ---")

            # 모델 로드
            try:
                model = load_model(backbone, mode, port, device)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue

            # Clean 기준 성능
            clean_ds = PerturbedDataset(DATA_CSV, port, img_size=512)
            clean_loader = DataLoader(clean_ds, batch_size=32, num_workers=4,
                                      pin_memory=True, shuffle=False)
            clean_metrics = evaluate_loader(model, clean_loader, device)
            print(f"  Clean macro_f1={clean_metrics['macro_f1']:.4f} "
                  f"seafog_recall={clean_metrics['seafog_recall']:.4f}")

            # 기준 행
            all_rows.append({
                "port": port, "backbone": backbone, "mode": mode,
                "model": model_name,
                "perturb_type": "clean", "strength": 0,
                **{f"clean_{k}": clean_metrics[k] for k in clean_metrics},
                **{f"delta_{k}": 0.0 for k in clean_metrics},
                **clean_metrics,
            })

            # Perturbation 루프
            for perturb_type, strengths in perturb_config.items():
                for strength in strengths:
                    ds = PerturbedDataset(DATA_CSV, port, img_size=512,
                                          perturb_type=perturb_type, strength=strength)
                    loader = DataLoader(ds, batch_size=32, num_workers=4,
                                        pin_memory=True, shuffle=False)
                    metrics = evaluate_loader(model, loader, device)

                    delta_f1     = metrics["macro_f1"]     - clean_metrics["macro_f1"]
                    delta_seafog = metrics["seafog_recall"] - clean_metrics["seafog_recall"]

                    print(f"  {perturb_type}({strength}): "
                          f"macro_f1={metrics['macro_f1']:.4f} "
                          f"(Δ={delta_f1:+.4f}) | "
                          f"seafog_recall={metrics['seafog_recall']:.4f} "
                          f"(Δ={delta_seafog:+.4f})")

                    row = {
                        "port": port,
                        "backbone": backbone,
                        "mode": mode,
                        "model": model_name,
                        "perturb_type": perturb_type,
                        "strength": strength,
                        **{f"clean_{k}": clean_metrics[k] for k in clean_metrics},
                        **{f"delta_{k}": metrics[k] - clean_metrics[k] for k in metrics},
                        **metrics,
                    }
                    all_rows.append(row)

            del model
            torch.cuda.empty_cache()

    # 저장
    df = pd.DataFrame(all_rows)
    out_path = output_dir / f"{test_name}_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {out_path}")

    # 요약 출력
    print_summary(df, test_name)
    return df


def print_summary(df: pd.DataFrame, test_name: str):
    """모델 쌍별 Δ macro_f1 평균 비교"""
    print(f"\n{'='*60}")
    print(f"요약: {test_name}")
    print(f"{'='*60}")

    pairs = [
        ("xception_base",     "xception_typeB_7"),
        ("mobilenet_base",    "mobilenet_typeA_11"),
        ("convnext_base",     "convnext_typeA_15"),
        ("efficientnet_base", "efficientnet_typeA_7"),
    ]

    clean_df = df[df["perturb_type"] != "clean"]
    if clean_df.empty:
        return

    for base_model, ext_model in pairs:
        base_delta = clean_df[clean_df["model"] == base_model]["delta_macro_f1"].mean()
        ext_delta  = clean_df[clean_df["model"] == ext_model]["delta_macro_f1"].mean()
        diff       = ext_delta - base_delta

        direction = "ERF 확장이 더 안정적 ✅" if diff > 0 else "ERF 확장이 더 불안정 ⚠️"
        print(f"{base_model:25s} Δ={base_delta:+.4f}")
        print(f"{ext_model:25s} Δ={ext_delta:+.4f}  ({direction})")
        print()


# ── 인자 파싱 ─────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test",  type=str, default="all",
                   choices=["texture", "structure", "all"])
    p.add_argument("--port",  type=str, default="haeundae")
    p.add_argument("--port2", type=str, default=None,
                   help="추가 항만 (선택)")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    if args.test in ("texture", "all"):
        run_test(args, TEXTURE_PERTURBS, "texture_perturbation")

    if args.test in ("structure", "all"):
        run_test(args, STRUCTURE_DISRUPT, "structure_disruption")

    print("\n전체 완료!")
