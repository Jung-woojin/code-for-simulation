# -*- coding: utf-8 -*-
# gradcam_failure.py — 오분류 샘플 Grad-CAM Mining
# 위치: /home/wj/seafog/src/gradcam_failure.py
#
# 사용법:
#   # 1순위: xception haeundae
#   python gradcam_failure.py --pair xception --port haeundae
#
#   # 2순위: mobilenet yeosu
#   python gradcam_failure.py --pair mobilenet --port yeosu
#
#   # 3순위: convnext (negative control)
#   python gradcam_failure.py --pair convnext --port haeundae
#
#   # 전체
#   python gradcam_failure.py --pair all --port haeundae

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from models_erf import load_pretrained_for_finetune

CLASS_NAMES   = ["normal", "lowvis", "seafog"]
PRETRAIN_ROOT = "/data1/wj/seafog/pretrain_ckpt"
RESULT_ROOT   = "/data1/wj/seafog/results/erf"
DATA_CSV      = "/data1/wj/seafog/data/splits.csv"
OUTPUT_ROOT   = "/data1/wj/seafog/results/gradcam_misclassified"

MODEL_PAIRS = {
    "convnext_shrink": {
        "A": {"backbone": "convnext", "mode": "typeA_3", "label": "ConvNeXt typeA_3 (3×3)"},
        "B": {"backbone": "convnext", "mode": "base",    "label": "ConvNeXt base (7×7)"},
        "role": "primary",
    },
    "xception": {
        "A": {"backbone": "xception", "mode": "base",    "label": "Xception base"},
        "B": {"backbone": "xception", "mode": "typeB_7", "label": "Xception typeB_7"},
        "role": "primary",
    },
    "mobilenet": {
        "A": {"backbone": "mobilenet", "mode": "base",     "label": "MobileNet base"},
        "B": {"backbone": "mobilenet", "mode": "typeA_11", "label": "MobileNet typeA_11"},
        "role": "primary",
    },
    "convnext": {
        "A": {"backbone": "convnext", "mode": "base",     "label": "ConvNeXt base"},
        "B": {"backbone": "convnext", "mode": "typeA_15", "label": "ConvNeXt typeA_15"},
        "role": "negative_control",
    },
    "efficientnet": {
        "A": {"backbone": "efficientnet", "mode": "base",   "label": "EfficientNet base"},
        "B": {"backbone": "efficientnet", "mode": "typeA_7","label": "EfficientNet typeA_7"},
        "role": "supplementary",
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
                lambda grad: setattr(self, "gradients", grad.detach().clone())
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


# ── 전체 테스트셋 추론 ────────────────────────────────────────
@torch.no_grad()
def run_inference(model, img_paths, label_list, transform, device):
    """전체 테스트셋에 대해 예측 결과 반환"""
    results = []
    label_map = {"normal": 0, "lowvis": 1, "seafog": 2}

    for path, label_str in zip(img_paths, label_list):
        gt = label_map[label_str]
        img_pil    = Image.open(path).convert("RGB").resize((512, 512))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        output = model(img_tensor)
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred   = int(probs.argmax())

        results.append({
            "path":      path,
            "gt":        gt,
            "gt_name":   label_str,
            "pred":      pred,
            "pred_name": CLASS_NAMES[pred],
            "correct":   int(pred == gt),
            "probs":     probs,
        })

    return results


# ── 오분류 케이스 분류 ────────────────────────────────────────
def classify_failure_case(gt_name, pred_name):
    """어떤 종류의 오분류인지 반환"""
    if gt_name == pred_name:
        return None
    return f"GT_{gt_name}__PRED_{pred_name}"


# ── Figure 생성 ───────────────────────────────────────────────
def make_comparison_figure(
    img_np,
    cam_a_pred, cam_a_gt,   # Model A CAM (pred class, gt class)
    cam_b_pred, cam_b_gt,   # Model B CAM
    prob_a, prob_b,
    pred_a, pred_b, gt,
    label_a, label_b,
    img_name, port, role, case_type
):
    fig = plt.figure(figsize=(20, 6))
    gs  = gridspec.GridSpec(1, 7, figure=fig, wspace=0.28,
                            left=0.02, right=0.98, top=0.88, bottom=0.08)

    colors = ["#4472C4", "#ED7D31", "#A9D18E"]

    def add_cam_ax(ax, cam, title, xlabel=None, color="black"):
        ax.imshow(overlay_cam(img_np, cam))
        ax.set_title(title, fontsize=9)
        if xlabel:
            ax.set_xlabel(xlabel, color=color, fontsize=10, fontweight="bold")
        ax.axis("off")

    def add_prob_ax(ax, prob, pred, title):
        bars = ax.barh(CLASS_NAMES, prob, color=colors,
                       edgecolor="gray", linewidth=0.5)
        ax.set_xlim(0, 1.15)
        ax.set_title(title, fontsize=9)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.4)
        # GT 표시
        ax.get_yticklabels()[gt].set_fontweight("bold")
        for bar, p in zip(bars, prob):
            ax.text(p + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{p:.3f}", va="center", fontsize=8)

    # 원본
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_np)
    ax0.set_title(f"Original\nGT: {CLASS_NAMES[gt]}", fontsize=10, fontweight="bold")
    ax0.axis("off")

    # Model A: pred-class CAM
    pred_color_a = "red" if pred_a != gt else "green"
    add_cam_ax(fig.add_subplot(gs[0, 1]),
               cam_a_pred,
               f"{label_a}\nCAM (pred: {CLASS_NAMES[pred_a]})",
               f"{'✗ WRONG' if pred_a != gt else '✓ CORRECT'}",
               pred_color_a)

    # Model A: gt-class CAM
    add_cam_ax(fig.add_subplot(gs[0, 2]),
               cam_a_gt,
               f"{label_a}\nCAM (GT: {CLASS_NAMES[gt]})")

    # Model A: Softmax
    add_prob_ax(fig.add_subplot(gs[0, 3]), prob_a, pred_a,
                f"{label_a}\nSoftmax")

    # Model B: pred-class CAM
    pred_color_b = "red" if pred_b != gt else "green"
    add_cam_ax(fig.add_subplot(gs[0, 4]),
               cam_b_pred,
               f"{label_b}\nCAM (pred: {CLASS_NAMES[pred_b]})",
               f"{'✗ WRONG' if pred_b != gt else '✓ CORRECT'}",
               pred_color_b)

    # Model B: gt-class CAM
    add_cam_ax(fig.add_subplot(gs[0, 5]),
               cam_b_gt,
               f"{label_b}\nCAM (GT: {CLASS_NAMES[gt]})")

    # Model B: Softmax
    add_prob_ax(fig.add_subplot(gs[0, 6]), prob_b, pred_b,
                f"{label_b}\nSoftmax")

    role_str = {"primary": "★", "negative_control": "⚠", "supplementary": ""}.get(role, "")
    fig.suptitle(
        f"{role_str} [{case_type}]  Port: {port}  |  {img_name}",
        fontsize=10, y=0.97
    )
    return fig


# ── 메인 실험 ─────────────────────────────────────────────────
def run_pair(pair_name, port, device, transform, args):
    pair  = MODEL_PAIRS[pair_name]
    cfg_a = pair["A"]
    cfg_b = pair["B"]
    role  = pair["role"]

    print(f"\n{'='*65}")
    print(f"[{pair_name}] {cfg_a['label']} vs {cfg_b['label']}")
    print(f"Port: {port} | Role: {role}")
    print(f"{'='*65}")

    # 모델 로드
    try:
        model_a = load_model(cfg_a["backbone"], cfg_a["mode"], port, device)
        model_b = load_model(cfg_b["backbone"], cfg_b["mode"], port, device)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return

    # 테스트셋 수집
    all_paths, all_labels = [], []
    with open(DATA_CSV) as f:
        for row in csv.DictReader(f):
            if row["port"] == port and row["split"] == "test":
                all_paths.append(Path(row["filepath"]))
                all_labels.append(row["class_label"])

    print(f"  테스트셋: {len(all_paths)}장")

    # 전체 추론
    print("  Model A 추론 중...")
    results_a = run_inference(model_a, all_paths, all_labels, transform, device)
    print("  Model B 추론 중...")
    results_b = run_inference(model_b, all_paths, all_labels, transform, device)

    acc_a = sum(r["correct"] for r in results_a) / len(results_a)
    acc_b = sum(r["correct"] for r in results_b) / len(results_b)
    print(f"  정확도 | A: {acc_a:.4f} | B: {acc_b:.4f}")

    # Grad-CAM 준비
    target_a = get_target_layer(model_a, cfg_a["backbone"])
    target_b = get_target_layer(model_b, cfg_b["backbone"])
    gcam_a   = GradCAM(model_a, target_a)
    gcam_b   = GradCAM(model_b, target_b)

    # 관심 케이스 정의
    # role에 따라 우선 케이스 다름
    if role == "primary":
        # A 틀리고 B 맞는 케이스 (ERF 개선 사례)
        interest_cases = [
            {"a_wrong": True,  "b_wrong": False, "tag": "A_fail_B_correct"},
            # A 맞고 B 틀린 케이스 (역전 사례, 드물지만 중요)
            {"a_wrong": False, "b_wrong": True,  "tag": "A_correct_B_fail"},
            # 둘 다 틀린 케이스 (공통 failure)
            {"a_wrong": True,  "b_wrong": True,  "tag": "both_fail"},
        ]
    else:  # negative_control
        # A 맞고 B 틀린 케이스 (과확장 역효과)
        interest_cases = [
            {"a_wrong": False, "b_wrong": True,  "tag": "A_correct_B_fail"},
            {"a_wrong": True,  "b_wrong": True,  "tag": "both_fail"},
            {"a_wrong": True,  "b_wrong": False, "tag": "A_fail_B_correct"},
        ]

    # 케이스별 분류
    case_buckets = {c["tag"]: [] for c in interest_cases}
    for ra, rb in zip(results_a, results_b):
        a_wrong = ra["pred"] != ra["gt"]
        b_wrong = rb["pred"] != rb["gt"]
        for case in interest_cases:
            if a_wrong == case["a_wrong"] and b_wrong == case["b_wrong"]:
                failure_type = classify_failure_case(ra["gt_name"], ra["pred_name"])
                case_buckets[case["tag"]].append({
                    "path":    ra["path"],
                    "gt":      ra["gt"],
                    "gt_name": ra["gt_name"],
                    "pred_a":  ra["pred"],
                    "pred_b":  rb["pred"],
                    "probs_a": ra["probs"],
                    "probs_b": rb["probs"],
                    "failure_type_a": failure_type,
                    "failure_type_b": classify_failure_case(rb["gt_name"], rb["pred_name"]),
                })
                break

    # 통계 출력
    print(f"\n  케이스 분포:")
    for tag, items in case_buckets.items():
        print(f"    {tag}: {len(items)}개")

    # Grad-CAM 생성 및 저장
    base_dir = Path(OUTPUT_ROOT) / port / pair_name
    summary_rows = []

    for tag, items in case_buckets.items():
        if not items:
            continue

        # 각 failure_type별 서브디렉토리
        ft_groups = {}
        for item in items:
            ft = item["failure_type_a"] or f"GT_{item['gt_name']}__PRED_correct"
            ft_groups.setdefault(ft, []).append(item)

        print(f"\n  [{tag}] 케이스별 CAM 생성:")
        for ft, ft_items in ft_groups.items():
            out_dir = base_dir / tag / ft
            out_dir.mkdir(parents=True, exist_ok=True)

            # args.max_per_case개까지만 저장
            selected = ft_items[:args.max_per_case]
            print(f"    {ft}: {len(ft_items)}개 중 {len(selected)}개 저장")

            for idx, item in enumerate(selected):
                img_path = item["path"]
                gt       = item["gt"]
                pred_a   = item["pred_a"]
                pred_b   = item["pred_b"]
                probs_a  = item["probs_a"]
                probs_b  = item["probs_b"]

                img_pil    = Image.open(img_path).convert("RGB").resize((512, 512))
                img_np     = np.array(img_pil)
                img_tensor = get_transform()(img_pil).unsqueeze(0).to(device)

                # Model A: pred-class CAM + gt-class CAM
                cam_a_pred, _, _ = gcam_a.generate(img_tensor.clone(), pred_a)
                cam_a_gt,   _, _ = gcam_a.generate(img_tensor.clone(), gt)

                # Model B: pred-class CAM + gt-class CAM
                cam_b_pred, _, _ = gcam_b.generate(img_tensor.clone(), pred_b)
                cam_b_gt,   _, _ = gcam_b.generate(img_tensor.clone(), gt)

                fig = make_comparison_figure(
                    img_np,
                    cam_a_pred, cam_a_gt,
                    cam_b_pred, cam_b_gt,
                    probs_a, probs_b,
                    pred_a, pred_b, gt,
                    cfg_a["label"], cfg_b["label"],
                    img_path.name, port, role,
                    f"{tag}/{ft}"
                )

                fname = (f"{idx:03d}_GT{CLASS_NAMES[gt]}_"
                         f"A{CLASS_NAMES[pred_a]}_B{CLASS_NAMES[pred_b]}_"
                         f"{img_path.stem}.png")
                fig.savefig(out_dir / fname, bbox_inches="tight", dpi=150)
                plt.close(fig)

                summary_rows.append({
                    "port": port, "pair": pair_name, "tag": tag,
                    "failure_type_a": item["failure_type_a"],
                    "failure_type_b": item["failure_type_b"],
                    "img": img_path.name,
                    "gt": CLASS_NAMES[gt],
                    "pred_a": CLASS_NAMES[pred_a],
                    "pred_b": CLASS_NAMES[pred_b],
                    "seafog_prob_a": round(float(probs_a[2]), 4),
                    "seafog_prob_b": round(float(probs_b[2]), 4),
                    "a_correct": int(pred_a == gt),
                    "b_correct": int(pred_b == gt),
                    "saved_path": str(out_dir / fname),
                })

    # 요약 저장
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(base_dir / "summary.csv", index=False, encoding="utf-8-sig")

        # 논문용 우선순위 케이스 자동 추천
        print(f"\n  ★ 논문 추천 케이스 (A_fail_B_correct, A→seafog 역전):")
        paper_cases = df[
            (df["tag"] == "A_fail_B_correct") &
            (df["gt"] == "seafog")
        ].head(5)
        if not paper_cases.empty:
            print(paper_cases[["img","gt","pred_a","pred_b",
                                "seafog_prob_a","seafog_prob_b"]].to_string(index=False))

    print(f"\n  저장 완료: {base_dir}")

    del model_a, model_b
    torch.cuda.empty_cache()


# ── 인자 파싱 ─────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", type=str, default="xception",
                   choices=["convnext_shrink", "xception", "mobilenet", "convnext", "efficientnet", "all"])
    p.add_argument("--port", type=str, default="haeundae")
    p.add_argument("--max_per_case", type=int, default=10,
                   help="failure type별 최대 저장 이미지 수")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform()
    random.seed(args.seed)
    np.random.seed(args.seed)

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    order = {"primary": 0, "negative_control": 1, "supplementary": 2}
    pairs = (list(MODEL_PAIRS.keys()) if args.pair == "all"
             else [args.pair])
    pairs = sorted(pairs, key=lambda x: order.get(MODEL_PAIRS[x]["role"], 9))

    for pair_name in pairs:
        run_pair(pair_name, args.port, device, transform, args)

    print("\n전체 완료!")
    print(f"결과: {OUTPUT_ROOT}")
