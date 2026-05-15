"""
Experiment-only context branch models for Track C.

This file intentionally does not modify or import the original
`context_branch.py`. The original file accepts several strategy names, but the
current implementation behaves like one parallel side branch regardless of the
label. In this experiment copy, `stageaware` is implemented explicitly with a
real 4-stage kernel schedule so the strategy name matches the behavior.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_PARALLEL_KERNELS = [7, 7, 9, 9]
DEFAULT_STAGEAWARE_KERNELS = [3, 5, 7, 9]


def _kernel_padding(kernel_size: int) -> int:
    return kernel_size // 2


class BranchStage(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=_kernel_padding(kernel_size),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiStageContextBranch(nn.Module):
    """
    Independent image-side branch.

    `parallel` uses a fixed kernel schedule.
    `stageaware` uses a caller-provided 4-stage schedule.
    """

    def __init__(
        self,
        out_dim: int,
        stage_kernel_sizes: Sequence[int],
        mid_channels: int = 64,
    ) -> None:
        super().__init__()

        if len(stage_kernel_sizes) != 4:
            raise ValueError(
                f"stage_kernel_sizes must have length 4, got {list(stage_kernel_sizes)}"
            )

        self.stage_kernel_sizes = [int(k) for k in stage_kernel_sizes]
        self.stem = nn.Sequential(
            nn.Conv2d(3, mid_channels, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList(
            [BranchStage(mid_channels, kernel_size=k) for k in self.stage_kernel_sizes]
        )
        self.stage_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(mid_channels, out_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in self.stage_kernel_sizes
            ]
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.alpha_logits = nn.Parameter(torch.zeros(len(self.stage_kernel_sizes)))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_alphas(self) -> List[float]:
        weights = torch.softmax(self.alpha_logits.detach().cpu(), dim=0)
        return [float(v) for v in weights]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        stage_features = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            projected = self.stage_projections[idx](x)
            pooled = self.gap(projected).flatten(1)
            stage_features.append(pooled)
            if idx < len(self.stages) - 1:
                x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=False)

        alpha = torch.softmax(self.alpha_logits, dim=0)
        stacked = torch.stack(stage_features, dim=1)
        return torch.sum(stacked * alpha.view(1, -1, 1), dim=1)


class GateFusion(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.gate_fc = nn.Linear(feat_dim * 2, 1)
        nn.init.zeros_(self.gate_fc.weight)
        nn.init.zeros_(self.gate_fc.bias)

    def forward(
        self,
        trunk_feat: torch.Tensor,
        branch_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([trunk_feat, branch_feat], dim=1)
        gate = torch.sigmoid(self.gate_fc(concat))
        fused = gate * trunk_feat + (1.0 - gate) * branch_feat
        return fused, gate


class SeafogExperimentModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        strategy: str,
        num_classes: int = 3,
        img_size: int = 512,
        stage_kernel_sizes: Optional[Sequence[int]] = None,
        mid_channels: int = 64,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.strategy = strategy

        self.trunk = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        trunk_dim = self.trunk.num_features

        self.branch: Optional[MultiStageContextBranch]
        self.fusion: Optional[GateFusion]
        if strategy == "baseline":
            self.branch = None
            self.fusion = None
            self.stage_kernel_sizes = None
        else:
            if strategy == "parallel":
                kernels = list(DEFAULT_PARALLEL_KERNELS)
            elif strategy == "stageaware":
                kernels = list(stage_kernel_sizes or DEFAULT_STAGEAWARE_KERNELS)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            self.branch = MultiStageContextBranch(
                out_dim=trunk_dim,
                stage_kernel_sizes=kernels,
                mid_channels=mid_channels,
            )
            self.fusion = GateFusion(feat_dim=trunk_dim)
            self.stage_kernel_sizes = kernels

        self.head = nn.Linear(trunk_dim, num_classes)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk_feat = self.trunk(x)
        if self.branch is None or self.fusion is None:
            fused = trunk_feat
        else:
            branch_feat = self.branch(x)
            fused, _ = self.fusion(trunk_feat, branch_feat)
        return self.head(fused)

    def get_alphas(self) -> List[float]:
        if self.branch is None:
            return []
        return self.branch.get_alphas()

    def get_gate_value(self, x: torch.Tensor) -> float:
        if self.branch is None or self.fusion is None:
            return 1.0
        with torch.no_grad():
            trunk_feat = self.trunk(x)
            branch_feat = self.branch(x)
            _, gate = self.fusion(trunk_feat, branch_feat)
        return float(gate.mean().item())

    def get_param_groups_stage1(
        self,
        branch_lr: float = 1e-3,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> List[dict]:
        for param in self.trunk.parameters():
            param.requires_grad = False

        groups = [
            {
                "params": list(self.head.parameters()),
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            }
        ]

        if self.branch is not None and self.fusion is not None:
            groups.insert(
                0,
                {
                    "params": list(self.branch.parameters()),
                    "lr": branch_lr,
                    "weight_decay": weight_decay,
                    "name": "branch",
                },
            )
            groups.insert(
                1,
                {
                    "params": list(self.fusion.parameters()),
                    "lr": branch_lr,
                    "weight_decay": weight_decay,
                    "name": "fusion",
                },
            )

        return groups

    def get_param_groups_stage2(
        self,
        trunk_last_lr: float = 5e-5,
        branch_lr: float = 2e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> List[dict]:
        for param in self.trunk.parameters():
            param.requires_grad = False

        last_stage = self._get_last_stage()
        if last_stage is not None:
            for param in last_stage.parameters():
                param.requires_grad = True

        groups = [
            {
                "params": list(self.head.parameters()),
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            }
        ]

        if self.branch is not None and self.fusion is not None:
            groups.insert(
                0,
                {
                    "params": list(self.branch.parameters()),
                    "lr": branch_lr,
                    "weight_decay": weight_decay,
                    "name": "branch",
                },
            )
            groups.insert(
                1,
                {
                    "params": list(self.fusion.parameters()),
                    "lr": branch_lr,
                    "weight_decay": weight_decay,
                    "name": "fusion",
                },
            )

        if last_stage is not None:
            params = [p for p in last_stage.parameters() if p.requires_grad]
            if params:
                groups.append(
                    {
                        "params": params,
                        "lr": trunk_last_lr,
                        "weight_decay": weight_decay,
                        "name": "trunk_last",
                    }
                )

        return groups

    def get_param_groups_stage3(
        self,
        lr_head: float = 1e-3,
        lr_last: float = 2e-5,
        lr_mid: float = 1e-5,
        lr_first: float = 1e-6,
        branch_lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> List[dict]:
        for param in self.trunk.parameters():
            param.requires_grad = True

        groups = [
            {
                "params": list(self.head.parameters()),
                "lr": lr_head,
                "weight_decay": weight_decay,
                "name": "head",
            }
        ]

        if self.branch is not None and self.fusion is not None:
            groups.insert(
                0,
                {
                    "params": list(self.branch.parameters()),
                    "lr": branch_lr,
                    "weight_decay": weight_decay,
                    "name": "branch",
                },
            )
            groups.insert(
                1,
                {
                    "params": list(self.fusion.parameters()),
                    "lr": branch_lr,
                    "weight_decay": weight_decay,
                    "name": "fusion",
                },
            )

        groups.extend(self._get_llrd_groups(lr_last, lr_mid, lr_first, weight_decay))
        return groups

    def _get_last_stage(self) -> Optional[nn.Module]:
        trunk = self.trunk
        if hasattr(trunk, "layer4"):
            return trunk.layer4
        if hasattr(trunk, "stages"):
            stages = list(trunk.stages)
            return stages[-1] if stages else None
        if hasattr(trunk, "blocks"):
            blocks = list(trunk.blocks)
            return blocks[-1] if blocks else None
        return None

    def _get_llrd_groups(
        self,
        lr_last: float,
        lr_mid: float,
        lr_first: float,
        weight_decay: float,
    ) -> List[dict]:
        trunk = self.trunk
        groups = []

        if hasattr(trunk, "layer4"):
            stage_map = [
                ("last", [trunk.layer4], lr_last),
                ("mid", [trunk.layer3, trunk.layer2], lr_mid),
                ("first", [trunk.layer1], lr_first),
            ]
            for name, modules, lr in stage_map:
                params = [p for module in modules for p in module.parameters() if p.requires_grad]
                if params:
                    groups.append(
                        {
                            "params": params,
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "name": f"trunk_{name}",
                        }
                    )

            stem_params = []
            for attr in ("conv1", "bn1"):
                if hasattr(trunk, attr):
                    stem_params.extend(
                        [p for p in getattr(trunk, attr).parameters() if p.requires_grad]
                    )
            if stem_params:
                groups.append(
                    {
                        "params": stem_params,
                        "lr": lr_first,
                        "weight_decay": weight_decay,
                        "name": "trunk_stem",
                    }
                )
            return groups

        if hasattr(trunk, "stages"):
            stages = list(trunk.stages)
            if stages:
                split = max(1, len(stages) // 3)
                stage_slices = [
                    ("last", stages[-split:], lr_last),
                    ("mid", stages[max(0, len(stages) - 2 * split) : len(stages) - split], lr_mid),
                    ("first", stages[: max(1, len(stages) - 2 * split)], lr_first),
                ]
                seen_ids = set()
                for name, modules, lr in stage_slices:
                    params = []
                    for module in modules:
                        for param in module.parameters():
                            if param.requires_grad and id(param) not in seen_ids:
                                params.append(param)
                                seen_ids.add(id(param))
                    if params:
                        groups.append(
                            {
                                "params": params,
                                "lr": lr,
                                "weight_decay": weight_decay,
                                "name": f"trunk_{name}",
                            }
                        )
                return groups

        if hasattr(trunk, "blocks"):
            blocks = list(trunk.blocks)
            if blocks:
                chunk = max(1, len(blocks) // 3)
                stage_map = [
                    ("last", blocks[-chunk:], lr_last),
                    ("mid", blocks[max(0, len(blocks) - 2 * chunk) : len(blocks) - chunk], lr_mid),
                    ("first", blocks[: max(1, len(blocks) - 2 * chunk)], lr_first),
                ]
                seen_ids = set()
                for name, modules, lr in stage_map:
                    params = []
                    for module in modules:
                        for param in module.parameters():
                            if param.requires_grad and id(param) not in seen_ids:
                                params.append(param)
                                seen_ids.add(id(param))
                    if params:
                        groups.append(
                            {
                                "params": params,
                                "lr": lr,
                                "weight_decay": weight_decay,
                                "name": f"trunk_{name}",
                            }
                        )
                return groups

        params = [p for p in trunk.parameters() if p.requires_grad]
        if params:
            groups.append(
                {
                    "params": params,
                    "lr": lr_mid,
                    "weight_decay": weight_decay,
                    "name": "trunk_all",
                }
            )
        return groups


def build_context_model(
    backbone: str,
    strategy: str,
    num_classes: int = 3,
    img_size: int = 512,
    stage_kernel_sizes: Optional[List[int]] = None,
    mid_channels: int = 64,
) -> SeafogExperimentModel:
    if strategy not in ("baseline", "parallel", "stageaware"):
        raise ValueError(
            f"Unknown strategy: {strategy}. Use baseline, parallel, or stageaware."
        )

    return SeafogExperimentModel(
        backbone=backbone,
        strategy=strategy,
        num_classes=num_classes,
        img_size=img_size,
        stage_kernel_sizes=stage_kernel_sizes,
        mid_channels=mid_channels,
    )


if __name__ == "__main__":
    import sys

    backbone = sys.argv[1] if len(sys.argv) > 1 else "resnet101"
    strategy = sys.argv[2] if len(sys.argv) > 2 else "stageaware"
    model = build_context_model(
        backbone=backbone,
        strategy=strategy,
        stage_kernel_sizes=[3, 5, 7, 9],
    )

    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print("output_shape", tuple(y.shape))
    print("alphas", model.get_alphas())
