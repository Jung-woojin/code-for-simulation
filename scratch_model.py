"""
scratch_model.py
Scratch CNN: small kernel vs large kernel ERF 직접 비교

strategy:
  - scratch_small : 3×3 depthwise (ERF 좁음, 기준점)
  - scratch_large : 7×7 + 9×9 depthwise (ERF 넓음, 제안)

두 모델 구조를 동일하게 유지하고 kernel size만 다르게 해서
ERF 확장 효과를 직접 비교.

사용:
  python train_scratch.py \
      --strategy scratch_large \
      --backbone dummy \
      --port daesan \
      --data_csv /data1/wj/seafog/data/splits.csv \
      --output /data1/wj/seafog/results/track_c/scratch_large/daesan/seed42 \
      --seed 42
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
#  공통 블록                                                           #
# ------------------------------------------------------------------ #
class ConvBNReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int,
                 stride: int = 1, padding: int = 0,
                 groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel,
                      stride=stride, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ------------------------------------------------------------------ #
#  Scratch CNN (small kernel 기준점)                                   #
# ------------------------------------------------------------------ #
class ScratchSmallKernel(nn.Module):
    """
    Small kernel (3×3) scratch CNN.
    ERF가 좁은 기준 모델.

    구조:
      stem : 3×3 stride=4 → C  (빠른 다운샘플)
             3×3 stride=2 → C
      dw1  : 3×3 depthwise     (좁은 ERF)
      dw2  : 3×3 depthwise     (좁은 ERF)
      pw   : 1×1 → feat_dim
      GAP  → head
    """

    def __init__(self, num_classes: int = 3,
                 mid_channels: int = 128,
                 feat_dim: int = 512):
        super().__init__()
        C = mid_channels

        self.stem = nn.Sequential(
            ConvBNReLU(3, C, kernel=3, stride=4, padding=1),
            ConvBNReLU(C, C, kernel=3, stride=2, padding=1),
        )
        self.dw1 = ConvBNReLU(C, C, kernel=3, stride=1,
                               padding=1, groups=C)
        self.dw2 = ConvBNReLU(C, C, kernel=3, stride=1,
                               padding=1, groups=C)
        self.pw  = ConvBNReLU(C, feat_dim, kernel=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(feat_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.pw(x)
        x = self.gap(x).flatten(1)
        return self.head(x)

    def get_alphas(self) -> List[float]:
        return []


# ------------------------------------------------------------------ #
#  Scratch CNN (large kernel 제안)                                     #
# ------------------------------------------------------------------ #
class ScratchLargeKernel(nn.Module):
    """
    Large kernel (7×7 + 9×9) scratch CNN.
    ERF가 넓은 제안 모델.

    구조:
      stem : 3×3 stride=4 → C  (빠른 다운샘플, small과 동일)
             3×3 stride=2 → C
      dw1  : 7×7 depthwise     (넓은 ERF)
      dw2  : 9×9 depthwise     (더 넓은 ERF)
      pw   : 1×1 → feat_dim    (small과 동일)
      GAP  → head
    """

    def __init__(self, num_classes: int = 3,
                 mid_channels: int = 128,
                 feat_dim: int = 512):
        super().__init__()
        C = mid_channels

        self.stem = nn.Sequential(
            ConvBNReLU(3, C, kernel=3, stride=4, padding=1),
            ConvBNReLU(C, C, kernel=3, stride=2, padding=1),
        )
        self.dw1 = ConvBNReLU(C, C, kernel=7, stride=1,
                               padding=3, groups=C)
        self.dw2 = ConvBNReLU(C, C, kernel=9, stride=1,
                               padding=4, groups=C)
        self.pw  = ConvBNReLU(C, feat_dim, kernel=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(feat_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.pw(x)
        x = self.gap(x).flatten(1)
        return self.head(x)

    def get_alphas(self) -> List[float]:
        return []


# ------------------------------------------------------------------ #
#  팩토리                                                              #
# ------------------------------------------------------------------ #
def build_scratch_model(
    strategy: str,
    num_classes: int = 3,
    mid_channels: int = 128,
    feat_dim: int = 512,
) -> nn.Module:
    if strategy == "scratch_small":
        return ScratchSmallKernel(num_classes=num_classes,
                                   mid_channels=mid_channels,
                                   feat_dim=feat_dim)
    elif strategy == "scratch_large":
        return ScratchLargeKernel(num_classes=num_classes,
                                   mid_channels=mid_channels,
                                   feat_dim=feat_dim)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ------------------------------------------------------------------ #
#  smoke test                                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    for strategy in ["scratch_small", "scratch_large"]:
        model = build_scratch_model(strategy)
        x = torch.randn(2, 3, 512, 512)
        out = model(x)
        total = sum(p.numel() for p in model.parameters())
        print(f"{strategy:20s} output={out.shape}  params={total:,}")
    print("OK")