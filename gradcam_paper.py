# -*- coding: utf-8 -*-
# gradcam_paper.py — 논문용 Grad-CAM 배치 생성
# 위치: /home/wj/seafog/src/gradcam_paper.py
#
# 사용법:
#   python gradcam_paper.py --pair xception --port haeundae --num_images 10
#   python gradcam_paper.py --pair mobilenet --port yeosu --num_images 10
#   python gradcam_paper.py --pair convnext --port haeundae --num_images 10
#   python gradcam_paper.py --pair all --port haeundae --num_images 10

import argparse
import random
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from models_erf import load_pretrained_for_finetune

CLASS_NAMES   = ["normal", "lowvis", "seafog"]
PRETRAIN_ROOT = "/data1/wj/seafog/pretrain_ckpt"
RESULT_ROOT   = "/data1/wj/seafog/results/erf"
DATA_CSV      = "/data1/wj/seafog/data/splits.csv"
OUTPUT_ROOT   = "/data1/wj/seafog/results/gradcam_paper"

# ── 모델 쌍 정의 ──────────────────────────────────────────────
MODEL_PAIRS = {
    "xception": {
        "A": {"backbone": "xception", "mode": "base",   "label": "Xception base (3×3)"},
        "B": {"backbone": "xception", "mode": "typeB_7","label": "Xception typeB_7"},
        "role": "primary",
        "note": "haeundae +0.124 핵심 사례",
    },
    "mobilenet": {
        "A": {"backbone": "mobilenet", "mode": "base",    "label": "MobileNet base (3×3)"},
        "B": {"backbone": "mobilenet", "mode": "typeA_11","label": "MobileNet typeA_11"},
        "role": "primary",
        "note": "전체 평균 1위",
    },
    "convnext": {
        "A": {"backbone": "convnext", "mode": "base",    "label": "ConvNeXt base (7×7)"},
        "B": {"backbone": "convnext", "mode": "typeA_15","label": "ConvNeXt typeA_15"},
        "role": "negative_control",
        "note": "과확장 negative control",
    },
    "efficientnet": {
        "A": {"backbone": "efficientnet", "mode": "base",   "label": "EfficientNet base (3×3)"},
        "B": {"backbone": "efficientnet", "mode": "typeA_7","label": "EfficientNet typeA_7"},
        "role": "supplementary",
        "note": "보조 사례",
    },
}


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
        target_layer.register_forward_hook(self._register_grad_hook)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _register_grad_hook(self, module, input, output):
        if output.requires_grad:
            output.register_hook(
                lambda grad: setattr(self, 'gradients', grad.detach().clone())
            )

    def generate(self, input_tensor, target_class):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = probs.argmax().item()

        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)

        if self.gradients is None:
            # fallback: gradient 없을 경우 activation만 사용
            cam = self.activations.mean(dim=1, keepdim=True)
        else:
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam     = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = torch.relu(cam).squeeze().cpu().numpy()
        if cam.ndim == 0:
            cam = np.zeros((16, 16))
        cam = cv2.resize(cam, (512, 512))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, probs.cpu().detach().numpy(), pred


def get_target_layer(model, backbone):
    if backbone == "convnext":
        return model.stages[-1].blocks[-1].conv_dw
    elif backbone == "xception":
        return model.conv3.conv1
    elif backbone == "efficientnet":
        return model.blocks[-1][-1].conv_dw
    elif backbone == "mobilenet":
        return model.blocks[5][2].conv_dw
    raise ValueError(f"Unknown backbone: {backbone}")


def overlay_cam(img_np, cam, alpha=0.5):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (img_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)


# ── 모델 로드 ─────────────────────────────────────────────────
def load_model(backbone, mode, port, device):
    pretrain_ckpt = f"{PRETRAIN_ROOT}/{backbone}_{mode}/best.pth"
    finetune_ckpt = f"{RESULT_ROOT}/{backbone}_{mode}/{port}/best.pth"
    model = load_pretrained_for_finetune(
        backbone=backbone, mode=mode,
        pretrain_ckpt=pretrain_ckpt, num_classes=3,
    )
    ckpt = torch.load(finetune_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model.to(device).eval()


# ── 이미지 수집 ───────────────────────────────────────────────
def get_images(port, class_label="seafog", num_images=10, seed=42):
    random.seed(seed)
    rows = []
    with open(DATA_CSV) as f:
        for row in csv.DictReader(f):
            if (row["port"] == port and
                row["split"] == "test" and
                row["class_label"] == class_label):
                rows.append(Path(row["filepath"]))
    random.shuffle(rows)
    return rows[:num_images]


# ── 단일 이미지 figure 생성 ───────────────────────────────────
def make_figure(img_np, cam_a, cam_b, prob_a, prob_b, pred_a, pred_b,
                label_a, label_b, img_name, port, role):

    fig = plt.figure(figsize=(16, 5.5))
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.3,
                            left=0.03, right=0.97, top=0.88, bottom=0.08)

    colors = ["#4472C4", "#ED7D31", "#A9D18E"]

    # 원본
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_np)
    ax0.set_title("Original\n(GT: seafog)", fontsize=11, fontweight="bold")
    ax0.axis("off")

    def add_cam_axes(ax_cam, ax_bar, cam, prob, pred, label):
        color = "green" if pred == 2 else "red"
        ax_cam.imshow(overlay_cam(img_np, cam))
        ax_cam.set_title(f"{label}\nGrad-CAM", fontsize=10)
        ax_cam.set_xlabel(f"Pred: {CLASS_NAMES[pred]}",
                          color=color, fontsize=11, fontweight="bold")
        ax_cam.axis("off")

        bars = ax_bar.barh(CLASS_NAMES, prob,
                           color=colors, edgecolor="gray", linewidth=0.5)
        ax_bar.set_xlim(0, 1.15)
        ax_bar.set_title(f"{label}\nSoftmax", fontsize=10)
        ax_bar.axvline(x=0.5, color="gray", linestyle="--", alpha=0.4)
        for bar, p in zip(bars, prob):
            ax_bar.text(p + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{p:.3f}", va="center", fontsize=9, fontweight="bold")

    add_cam_axes(fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                 cam_a, prob_a, pred_a, label_a)
    add_cam_axes(fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[0, 4]),
                 cam_b, prob_b, pred_b, label_b)

    role_str = {"primary": "★ Primary", "negative_control": "⚠ Negative Control",
                "supplementary": "Supplementary"}.get(role, "")
    fig.suptitle(
        f"[{role_str}]  Port: {port}  |  {img_name}",
        fontsize=11, y=0.97
    )
    return fig


# ── 배치 실행 ─────────────────────────────────────────────────
def run_pair(pair_name, port, num_images, device, transform, args):
    pair   = MODEL_PAIRS[pair_name]
    cfg_a  = pair["A"]
    cfg_b  = pair["B"]
    role   = pair["role"]

    print(f"\n{'='*60}")
    print(f"[{pair_name}] {cfg_a['label']} vs {cfg_b['label']}")
    print(f"Port: {port} | Role: {role} | {pair['note']}")
    print(f"{'='*60}")

    try:
        model_a = load_model(cfg_a["backbone"], cfg_a["mode"], port, device)
        model_b = load_model(cfg_b["backbone"], cfg_b["mode"], port, device)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return

    target_a = get_target_layer(model_a, cfg_a["backbone"])
    target_b = get_target_layer(model_b, cfg_b["backbone"])
    gcam_a   = GradCAM(model_a, target_a)
    gcam_b   = GradCAM(model_b, target_b)

    out_dir = Path(OUTPUT_ROOT) / pair_name / port
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for class_label in CLASS_NAMES:
        img_paths = get_images(port, class_label=class_label,
                               num_images=num_images, seed=args.seed)
        print(f"  [{class_label}] {len(img_paths)}장 선택")

        target_idx = CLASS_NAMES.index(class_label)

        for i, img_path in enumerate(img_paths):
            img_pil    = Image.open(img_path).convert("RGB").resize((512, 512))
            img_np     = np.array(img_pil)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            cam_a, prob_a, pred_a = gcam_a.generate(img_tensor.clone(), target_idx)
            cam_b, prob_b, pred_b = gcam_b.generate(img_tensor.clone(), target_idx)

        print(f"  [{i+1:02d}/{len(img_paths)}] {img_path.name}")
        print(f"    {cfg_a['label']:30s} seafog={prob_a[2]:.3f} pred={CLASS_NAMES[pred_a]}")
        print(f"    {cfg_b['label']:30s} seafog={prob_b[2]:.3f} pred={CLASS_NAMES[pred_b]}")

        fig = make_figure(
            img_np, cam_a, cam_b, prob_a, prob_b, pred_a, pred_b,
            cfg_a["label"], cfg_b["label"],
            img_path.name, port, role
        )

        save_path = out_dir / f"{class_label}_{i:03d}_{img_path.stem}.png"
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        summary_rows.append({
            "idx": i, "img": img_path.name, "port": port, "gt": class_label,
            "model_a": f"{cfg_a['backbone']}_{cfg_a['mode']}",
            "model_b": f"{cfg_b['backbone']}_{cfg_b['mode']}",
            "pred_a": CLASS_NAMES[pred_a], "seafog_prob_a": round(float(prob_a[2]), 4),
            "pred_b": CLASS_NAMES[pred_b], "seafog_prob_b": round(float(prob_b[2]), 4),
            "a_correct": int(pred_a == 2), "b_correct": int(pred_b == 2),
        })

    # 요약 CSV
    pd_rows = __import__("pandas").DataFrame(summary_rows)
    pd_rows.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    correct_a = sum(r["a_correct"] for r in summary_rows)
    correct_b = sum(r["b_correct"] for r in summary_rows)
    print(f"\n  정확도 | {cfg_a['label']}: {correct_a}/{len(summary_rows)} "
          f"| {cfg_b['label']}: {correct_b}/{len(summary_rows)}")
    print(f"  저장: {out_dir}")

    del model_a, model_b
    torch.cuda.empty_cache()


# ── 메인 ──────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", type=str, default="all",
                   choices=["xception", "mobilenet", "convnext", "efficientnet", "all"])
    p.add_argument("--port", type=str, default="haeundae")
    p.add_argument("--num_images", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args      = parse_args()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform(img_size=512)

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    pairs = list(MODEL_PAIRS.keys()) if args.pair == "all" else [args.pair]

    # primary → negative_control → supplementary 순서로
    order = {"primary": 0, "negative_control": 1, "supplementary": 2}
    pairs = sorted(pairs, key=lambda x: order.get(MODEL_PAIRS[x]["role"], 9))

    for pair_name in pairs:
        run_pair(pair_name, args.port, args.num_images, device, transform, args)

    print("\n전체 완료!")
    print(f"결과 저장: {OUTPUT_ROOT}")
