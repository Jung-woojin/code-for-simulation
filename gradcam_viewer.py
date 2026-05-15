# -*- coding: utf-8 -*-
# gradcam_viewer.py — 대화형 Grad-CAM 뷰어
import matplotlib
matplotlib.use("Agg")
# 위치: /home/wj/seafog/src/gradcam_viewer.py
#
# 사용법:
#   python gradcam_viewer.py --port haeundae
#
# 조작:
#   → 또는 d : 다음 이미지
#   ← 또는 a : 이전 이미지
#   s         : 현재 이미지 저장
#   q         : 종료

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models_erf import load_pretrained_for_finetune

CLASS_NAMES    = ["normal", "lowvis", "seafog"]
PRETRAIN_ROOT  = "/data1/wj/seafog/pretrain_ckpt"
RESULT_ROOT    = "/data1/wj/seafog/results/erf"
DATA_ROOT      = "/data1/wj/seafog/data"
SAVE_DIR       = "/data1/wj/seafog/results/gradcam"

MODEL_A = {"backbone": "convnext", "mode": "base",    "label": "ConvNeXt-B (7×7)"}
MODEL_B = {"backbone": "xception", "mode": "base",    "label": "Xception (3×3)"}


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
            output.register_hook(lambda grad: setattr(self, 'gradients', grad.detach().clone()))

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()

    def generate(self, input_tensor, target_class):
        self.model.eval()
        output = self.model(input_tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = probs.argmax().item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().cpu().numpy()
        cam     = cv2.resize(cam, (512, 512))
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, probs.cpu().detach().numpy(), pred


def get_target_layer(model, backbone):
    if backbone == "convnext":
        return model.stages[-1].blocks[-1].conv_dw
    elif backbone == "xception":
        return model.conv3.conv1
    elif backbone == "efficientnet":
        return model.blocks[-1][-1].conv_dw
    elif backbone == "mobilenet":
        return model.blocks[-1][0].conv_dw
    raise ValueError(f"Unknown backbone: {backbone}")


def overlay_cam(img_np, cam, alpha=0.5):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (img_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)


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


# ── 뷰어 ──────────────────────────────────────────────────────
class Viewer:
    def __init__(self, img_paths, gcam_a, gcam_b, transform, device, port):
        self.img_paths = img_paths
        self.gcam_a    = gcam_a
        self.gcam_b    = gcam_b
        self.transform = transform
        self.device    = device
        self.port      = port
        self.idx       = 0
        self.n         = len(img_paths)

        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

        self.fig = plt.figure(figsize=(18, 7))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.render()
        plt.show()

    def process(self, img_path):
        img_pil    = Image.open(img_path).convert("RGB").resize((512, 512))
        img_np     = np.array(img_pil)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        seafog_idx = CLASS_NAMES.index("seafog")
        cam_a, prob_a, pred_a = self.gcam_a.generate(img_tensor.clone(), seafog_idx)
        cam_b, prob_b, pred_b = self.gcam_b.generate(img_tensor.clone(), seafog_idx)

        return img_np, cam_a, cam_b, prob_a, prob_b, pred_a, pred_b

    def render(self):
        self.fig.clear()
        img_path = self.img_paths[self.idx]
        img_np, cam_a, cam_b, prob_a, prob_b, pred_a, pred_b = self.process(img_path)

        overlay_a = overlay_cam(img_np, cam_a)
        overlay_b = overlay_cam(img_np, cam_b)

        gs = gridspec.GridSpec(1, 5, figure=self.fig,
                               wspace=0.35, left=0.04, right=0.98,
                               top=0.88, bottom=0.08)

        # 원본
        ax0 = self.fig.add_subplot(gs[0, 0])
        ax0.imshow(img_np)
        ax0.set_title("Original\n(GT: seafog)", fontsize=11, fontweight="bold")
        ax0.axis("off")

        # ConvNeXt CAM
        ax1 = self.fig.add_subplot(gs[0, 1])
        ax1.imshow(overlay_a)
        pred_label_a = CLASS_NAMES[pred_a]
        color_a = "green" if pred_a == 2 else "red"
        ax1.set_title(f"{MODEL_A['label']}\nGrad-CAM", fontsize=10)
        ax1.set_xlabel(f"Pred: {pred_label_a}", color=color_a,
                       fontsize=11, fontweight="bold")
        ax1.axis("off")

        # ConvNeXt Softmax
        ax2 = self.fig.add_subplot(gs[0, 2])
        colors = ["#4472C4", "#ED7D31", "#A9D18E"]
        bars = ax2.barh(CLASS_NAMES, prob_a, color=colors, edgecolor="gray", linewidth=0.5)
        ax2.set_xlim(0, 1.15)
        ax2.set_title(f"{MODEL_A['label']}\nSoftmax", fontsize=10)
        for bar, prob in zip(bars, prob_a):
            ax2.text(prob + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.3f}", va="center", fontsize=10, fontweight="bold")
        ax2.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

        # Xception CAM
        ax3 = self.fig.add_subplot(gs[0, 3])
        ax3.imshow(overlay_b)
        pred_label_b = CLASS_NAMES[pred_b]
        color_b = "green" if pred_b == 2 else "red"
        ax3.set_title(f"{MODEL_B['label']}\nGrad-CAM", fontsize=10)
        ax3.set_xlabel(f"Pred: {pred_label_b}", color=color_b,
                       fontsize=11, fontweight="bold")
        ax3.axis("off")

        # Xception Softmax
        ax4 = self.fig.add_subplot(gs[0, 4])
        bars = ax4.barh(CLASS_NAMES, prob_b, color=colors, edgecolor="gray", linewidth=0.5)
        ax4.set_xlim(0, 1.15)
        ax4.set_title(f"{MODEL_B['label']}\nSoftmax", fontsize=10)
        for bar, prob in zip(bars, prob_b):
            ax4.text(prob + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.3f}", va="center", fontsize=10, fontweight="bold")
        ax4.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

        self.fig.suptitle(
            f"Port: {self.port}  |  [{self.idx+1}/{self.n}]  {img_path.name}\n"
            f"← a / d → : 이전/다음    s : 저장    q : 종료",
            fontsize=11
        )
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key in ("right", "d"):
            self.idx = (self.idx + 1) % self.n
            self.render()
        elif event.key in ("left", "a"):
            self.idx = (self.idx - 1) % self.n
            self.render()
        elif event.key == "s":
            save_path = Path(SAVE_DIR) / f"gradcam_{self.port}_{self.idx:03d}.png"
            self.fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"저장: {save_path}")
        elif event.key == "q":
            plt.close()


# ── 메인 ──────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",     type=str, default="haeundae")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--split",    type=str, default="test",
                   choices=["train", "valid", "test"])
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)

    print("모델 로드 중...")
    model_a = load_model(MODEL_A["backbone"], MODEL_A["mode"], args.port, device)
    model_b = load_model(MODEL_B["backbone"], MODEL_B["mode"], args.port, device)

    target_a = get_target_layer(model_a, MODEL_A["backbone"])
    target_b = get_target_layer(model_b, MODEL_B["backbone"])

    gcam_a = GradCAM(model_a, target_a)
    gcam_b = GradCAM(model_b, target_b)

    # splits.csv에서 test seafog 이미지 수집
    import csv
    img_paths = []
    with open("/data1/wj/seafog/data/splits.csv") as f:
        for row in csv.DictReader(f):
            if (row["port"] == args.port and
                row["split"] == args.split and
                row["class_label"] == "seafog"):
                img_paths.append(Path(row["filepath"]))

    random.shuffle(img_paths)
    print(f"{args.port} {args.split} seafog: {len(img_paths)}장")

    transform = get_transform(args.img_size)

    # X11 forwarding 필요 - 없으면 저장 모드로
    print("저장 모드로 실행 (첫 10장 저장)")
    import matplotlib.pyplot as plt

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    for i, img_path in enumerate(img_paths[:10]):
        print(f"  [{i+1}/10] {img_path.name}")
        img_pil    = Image.open(img_path).convert("RGB").resize((512, 512))
        img_np     = np.array(img_pil)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        seafog_idx = CLASS_NAMES.index("seafog")

        cam_a, prob_a, pred_a = gcam_a.generate(img_tensor.clone(), seafog_idx)
        cam_b, prob_b, pred_b = gcam_b.generate(img_tensor.clone(), seafog_idx)

        fig = plt.figure(figsize=(18, 7))
        gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(img_np); ax0.set_title("Original\n(GT: seafog)"); ax0.axis("off")

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(overlay_cam(img_np, cam_a))
        ax1.set_title(f"{MODEL_A['label']}\nGrad-CAM"); ax1.axis("off")
        ax1.set_xlabel(f"Pred: {CLASS_NAMES[pred_a]}",
                       color="green" if pred_a==2 else "red", fontweight="bold")

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.barh(CLASS_NAMES, prob_a, color=["#4472C4","#ED7D31","#A9D18E"])
        ax2.set_xlim(0, 1.15); ax2.set_title(f"{MODEL_A['label']}\nSoftmax")
        for bar, p in zip(ax2.patches, prob_a):
            ax2.text(p+0.02, bar.get_y()+bar.get_height()/2,
                     f"{p:.3f}", va="center", fontsize=9)

        ax3 = fig.add_subplot(gs[0, 3])
        ax3.imshow(overlay_cam(img_np, cam_b))
        ax3.set_title(f"{MODEL_B['label']}\nGrad-CAM"); ax3.axis("off")
        ax3.set_xlabel(f"Pred: {CLASS_NAMES[pred_b]}",
                       color="green" if pred_b==2 else "red", fontweight="bold")

        ax4 = fig.add_subplot(gs[0, 4])
        ax4.barh(CLASS_NAMES, prob_b, color=["#4472C4","#ED7D31","#A9D18E"])
        ax4.set_xlim(0, 1.15); ax4.set_title(f"{MODEL_B['label']}\nSoftmax")
        for bar, p in zip(ax4.patches, prob_b):
            ax4.text(p+0.02, bar.get_y()+bar.get_height()/2,
                     f"{p:.3f}", va="center", fontsize=9)

        fig.suptitle(f"Port: {args.port} | {img_path.name}", fontsize=11)
        save_path = Path(SAVE_DIR) / f"gradcam_{args.port}_{i:03d}.png"
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"    저장: {save_path}")

    print(f"\n완료! 저장 경로: {SAVE_DIR}")