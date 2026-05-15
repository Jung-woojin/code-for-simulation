# -*- coding: utf-8 -*-
# gradcam_compare.py — 두 모델 Grad-CAM 비교 시각화
# 위치: /home/wj/seafog/src/gradcam_compare.py
#
# 사용법:
#   python gradcam_compare.py --port haeundae --num_images 5

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from models_erf import load_pretrained_for_finetune

CLASS_NAMES = ["normal", "lowvis", "seafog"]

# ── 모델 설정 ─────────────────────────────────────────────────
MODEL_A = {
    "backbone": "convnext",
    "mode":     "base",
    "label":    "ConvNeXt-B (7×7)",
}
MODEL_B = {
    "backbone": "xception",
    "mode":     "base",
    "label":    "Xception (3×3)",
}

PRETRAIN_ROOT = "/data1/wj/seafog/pretrain_ckpt"
RESULT_ROOT   = "/data1/wj/seafog/results/erf"
DATA_ROOT     = "/data1/wj/seafog/data"
SAVE_DIR      = "/data1/wj/seafog/results/gradcam"


# ── Transform ─────────────────────────────────────────────────
def get_transform(img_size=512):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ── Grad-CAM ──────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class):
        self.model.eval()
        output = self.model(input_tensor)
        probs  = torch.softmax(output, dim=1)[0]

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (512, 512))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, probs.cpu().detach().numpy()


def get_target_layer(model, backbone):
    """backbone별 마지막 feature layer 반환"""
    if backbone == "convnext":
        return model.stages[-1].blocks[-1].conv_dw
    elif backbone == "xception":
        return model.block12.rep[-1]
    elif backbone == "efficientnet":
        return model.blocks[-1][-1].conv_dw
    elif backbone == "mobilenet":
        return model.blocks[-1][0].conv_dw
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


# ── 모델 로드 ─────────────────────────────────────────────────
def load_model(backbone, mode, port, device):
    pretrain_ckpt = f"{PRETRAIN_ROOT}/{backbone}_{mode}/best.pth"
    finetune_ckpt = f"{RESULT_ROOT}/{backbone}_{mode}/{port}/best.pth"

    model = load_pretrained_for_finetune(
        backbone=backbone,
        mode=mode,
        pretrain_ckpt=pretrain_ckpt,
        num_classes=3,
    )

    ckpt  = torch.load(finetune_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device).eval()

    return model


# ── 이미지 수집 ───────────────────────────────────────────────
def get_seafog_images(port, num_images=5):
    seafog_dir = Path(DATA_ROOT) / port / "seafog"
    images     = list(seafog_dir.glob("*.jpg"))
    random.shuffle(images)
    return images[:num_images]


# ── 시각화 ────────────────────────────────────────────────────
def overlay_cam(img_np, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (img_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return overlay


def visualize(img_paths, cam_a, cam_b, probs_a, probs_b, save_path):
    n = len(img_paths)
    fig = plt.figure(figsize=(18, n * 5))
    gs  = gridspec.GridSpec(n, 5, figure=fig,
                            hspace=0.4, wspace=0.3)

    for i, img_path in enumerate(img_paths):
        img_pil = Image.open(img_path).convert("RGB").resize((512, 512))
        img_np  = np.array(img_pil)

        overlay_a = overlay_cam(img_np, cam_a[i])
        overlay_b = overlay_cam(img_np, cam_b[i])

        # 원본
        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(img_np)
        ax0.set_title("Original\n(seafog)", fontsize=10)
        ax0.axis("off")

        # Model A CAM
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.imshow(overlay_a)
        ax1.set_title(f"{MODEL_A['label']}\nGrad-CAM", fontsize=10)
        ax1.axis("off")

        # Model A Softmax
        ax2 = fig.add_subplot(gs[i, 2])
        bars = ax2.barh(CLASS_NAMES, probs_a[i],
                        color=["steelblue", "orange", "tomato"])
        ax2.set_xlim(0, 1)
        ax2.set_title(f"{MODEL_A['label']}\nSoftmax", fontsize=10)
        for bar, prob in zip(bars, probs_a[i]):
            ax2.text(prob + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.3f}", va="center", fontsize=9)

        # Model B CAM
        ax3 = fig.add_subplot(gs[i, 3])
        ax3.imshow(overlay_b)
        ax3.set_title(f"{MODEL_B['label']}\nGrad-CAM", fontsize=10)
        ax3.axis("off")

        # Model B Softmax
        ax4 = fig.add_subplot(gs[i, 4])
        bars = ax4.barh(CLASS_NAMES, probs_b[i],
                        color=["steelblue", "orange", "tomato"])
        ax4.set_xlim(0, 1)
        ax4.set_title(f"{MODEL_B['label']}\nSoftmax", fontsize=10)
        for bar, prob in zip(bars, probs_b[i]):
            ax4.text(prob + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.3f}", va="center", fontsize=9)

    plt.suptitle(
        f"Grad-CAM Comparison | Port: {args.port} | Class: seafog",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"저장: {save_path}")


# ── 메인 ──────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",       type=str, default="haeundae")
    p.add_argument("--num_images", type=int, default=5)
    p.add_argument("--img_size",   type=int, default=512)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform(args.img_size)

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    print(f"모델 로드 중...")
    model_a = load_model(MODEL_A["backbone"], MODEL_A["mode"], args.port, device)
    model_b = load_model(MODEL_B["backbone"], MODEL_B["mode"], args.port, device)

    print(f"Target layer 설정 중...")
    target_a = get_target_layer(model_a, MODEL_A["backbone"])
    target_b = get_target_layer(model_b, MODEL_B["backbone"])

    gcam_a = GradCAM(model_a, target_a)
    gcam_b = GradCAM(model_b, target_b)

    print(f"이미지 수집 중... (port={args.port}, class=seafog)")
    img_paths = get_seafog_images(args.port, args.num_images)
    print(f"  {len(img_paths)}장 선택됨")

    cams_a, probs_a = [], []
    cams_b, probs_b = [], []

    seafog_idx = CLASS_NAMES.index("seafog")

    for i, img_path in enumerate(img_paths):
        print(f"  [{i+1}/{len(img_paths)}] {img_path.name}")

        img_pil    = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        img_tensor.requires_grad = True

        cam_a, prob_a = gcam_a.generate(img_tensor.clone(), seafog_idx)
        cam_b, prob_b = gcam_b.generate(img_tensor.clone(), seafog_idx)

        cams_a.append(cam_a)
        cams_b.append(cam_b)
        probs_a.append(prob_a)
        probs_b.append(prob_b)

        print(f"    ConvNeXt: {dict(zip(CLASS_NAMES, prob_a.round(3)))}")
        print(f"    Xception: {dict(zip(CLASS_NAMES, prob_b.round(3)))}")

    save_path = Path(SAVE_DIR) / f"gradcam_{args.port}_seafog.png"
    visualize(img_paths, cams_a, cams_b, probs_a, probs_b, save_path)

    print("\n완료!")