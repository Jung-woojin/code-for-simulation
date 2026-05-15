# -*- coding: utf-8 -*-
# models_erf.py — ERF 실험용 backbone 정의
# 위치: /home/wj/seafog/src/models_erf.py

import torch
import torch.nn as nn
import timm

# ── 실험 설정 ─────────────────────────────────────────────────
BACKBONE_TIMM_MAP = {
    "convnext":    "convnext_base",
    "efficientnet":"tf_efficientnetv2_m",
    "xception":    "xception",
    "mobilenet":   "mobilenetv3_large_100",
}

# backbone별 기본 depthwise 커널 크기
BASE_KERNEL_MAP = {
    "convnext":    7,
    "efficientnet":3,
    "xception":    3,
    "mobilenet":   3,
}

# 전체 실험 목록
# mode: base | typeA_{k} | typeB_{k}
EXPERIMENTS = []
for b in ["convnext", "efficientnet", "xception", "mobilenet"]:
    EXPERIMENTS.append({"backbone": b, "mode": "base"})
for b in ["convnext", "efficientnet", "xception", "mobilenet"]:
    for k in [3, 7, 11, 15]:
        EXPERIMENTS.append({"backbone": b, "mode": f"typeA_{k}"})
for b in ["convnext", "efficientnet", "xception", "mobilenet"]:
    for k in [7, 15]:
        EXPERIMENTS.append({"backbone": b, "mode": f"typeB_{k}"})
# 총 28개


# ── Depthwise 교체 유틸 ───────────────────────────────────────
def _replace_depthwise(model: nn.Module, kernel_size: int) -> nn.Module:
    """모델 내 모든 depthwise conv를 kernel_size로 교체 (random init)"""
    for name, module in list(model.named_modules()):
        if (
            isinstance(module, nn.Conv2d)
            and module.groups == module.in_channels
            and module.in_channels > 1
        ):
            in_ch  = module.in_channels
            pad    = kernel_size // 2
            new_conv = nn.Conv2d(
                in_ch, in_ch,
                kernel_size=kernel_size,
                stride=module.stride,
                padding=pad,
                groups=in_ch,
                bias=module.bias is not None,
            )
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out")
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)

            parts  = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)

    return model


# ── 분기형 모듈 ───────────────────────────────────────────────
class BranchDWConv(nn.Module):
    """
    RepLK 스타일 분기형 depthwise conv.
    base_kernel (원래 크기) + branch_kernel (확장 크기) 병렬 합산.
    """
    def __init__(self, channels: int, base_kernel: int, branch_kernel: int):
        super().__init__()
        self.base   = nn.Conv2d(
            channels, channels,
            kernel_size=base_kernel,
            padding=base_kernel // 2,
            groups=channels, bias=False,
        )
        self.branch = nn.Conv2d(
            channels, channels,
            kernel_size=branch_kernel,
            padding=branch_kernel // 2,
            groups=channels, bias=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.act  = nn.GELU()

        nn.init.kaiming_normal_(self.base.weight,   mode="fan_out")
        nn.init.kaiming_normal_(self.branch.weight, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        return self.act(self.norm(self.base(x) + self.branch(x)))


def _replace_depthwise_with_branch(
    model: nn.Module,
    base_kernel: int,
    branch_kernel: int,
) -> nn.Module:
    """모델 내 depthwise conv → BranchDWConv 교체"""
    for name, module in list(model.named_modules()):
        if (
            isinstance(module, nn.Conv2d)
            and module.groups == module.in_channels
            and module.in_channels > 1
        ):
            in_ch      = module.in_channels
            new_module = BranchDWConv(in_ch, base_kernel, branch_kernel)

            parts  = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_module)

    return model


# ── 메인 빌더 ─────────────────────────────────────────────────
def build_erf_model(
    backbone: str,
    mode: str,
    num_classes: int = 3,
    pretrained: bool = False,
) -> nn.Module:
    """
    Args:
        backbone   : convnext / efficientnet / xception / mobilenet
        mode       : base | typeA_{k} | typeB_{k}
        num_classes: 출력 클래스 수 (pretrain=100, finetune=3)
        pretrained : ImageNet pretrained weight 사용 여부
    """
    timm_name = BACKBONE_TIMM_MAP[backbone]
    model     = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)

    if mode == "base":
        return model

    if mode.startswith("typeA_"):
        k = int(mode.split("_")[-1])
        return _replace_depthwise(model, k)

    if mode.startswith("typeB_"):
        k        = int(mode.split("_")[-1])
        base_k   = BASE_KERNEL_MAP[backbone]
        return _replace_depthwise_with_branch(model, base_k, k)

    raise ValueError(f"Unknown mode: {mode}")


def load_pretrained_for_finetune(
    backbone: str,
    mode: str,
    pretrain_ckpt: str,
    num_classes: int = 3,
) -> nn.Module:
    """
    pretrain checkpoint에서 weight를 불러온 뒤
    head를 num_classes로 교체하여 반환.
    """
    # pretrain과 동일 구조로 빌드 (num_classes=100)
    model = build_erf_model(backbone, mode, num_classes=100, pretrained=False)

    ckpt = torch.load(pretrain_ckpt, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    print(f"  Loaded pretrain ckpt: {pretrain_ckpt}")

    # head 교체
    if hasattr(model, "head") and hasattr(model.head, "fc"):
        in_features = model.head.fc.in_features
        model.head.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "head"):
        if isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
            model.head  = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            in_features     = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc    = nn.Linear(in_features, num_classes)

    print(f"  Head replaced → num_classes={num_classes}")
    return model


# ── 동작 확인 ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"총 실험 수: {len(EXPERIMENTS)}\n")
    for i, exp in enumerate(EXPERIMENTS):
        m = build_erf_model(exp["backbone"], exp["mode"], num_classes=100)
        total = sum(p.numel() for p in m.parameters()) / 1e6
        dw    = sum(
            p.numel() for n, p in m.named_parameters()
            if "depthwise" in n or "dw" in n.lower()
        ) / 1e6
        print(f"[{i+1:02d}] {exp['backbone']:12s} {exp['mode']:15s} "
              f"total={total:.1f}M")
    print("\nOK")