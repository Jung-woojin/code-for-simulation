from __future__ import annotations

from typing import List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Large Kernel Block                                                #
# ------------------------------------------------------------------ #
class LargeKernelBlock(nn.Module):
    """
    trunk 마지막 feature map에서 전역 문맥을 확장하는 large kernel branch.
    normal/lowvis/seafog 전체 시정 단계에 영향을 주는 context feature를 추출.
    """

    def __init__(self, in_channels: int, mid_channels: int = 256):
        super().__init__()

        self.dw7 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.dw9 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=9,
                padding=4,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.dw7(feat)
        x = self.dw9(x)
        x = self.pw(x)
        return self.gap(x).flatten(1)  # (B, mid_channels)


# ------------------------------------------------------------------ #
#  Ordinal Head                                                      #
# ------------------------------------------------------------------ #
class OrdinalHead(nn.Module):
    """
    trunk feature + context feature로 ordinal threshold 2개를 예측:
      ge1_logit: y >= 1   (normal vs degraded)
      ge2_logit: y >= 2   ((normal or lowvis) vs seafog)

    확률 조립 시 monotonicity(q2 <= q1)를 반영한다.
    """

    def __init__(
        self,
        trunk_dim: int,
        branch_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.trunk_proj = nn.Linear(trunk_dim, hidden_dim)
        self.branch_proj = nn.Linear(branch_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim * 2)

        self.ge1_head = nn.Linear(hidden_dim * 2, 1)
        self.ge2_head = nn.Linear(hidden_dim * 2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        trunk_feat: torch.Tensor,
        branch_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = F.gelu(self.trunk_proj(trunk_feat))
        b = F.gelu(self.branch_proj(branch_feat))
        h = self.norm(torch.cat([t, b], dim=1))

        ge1_logit = self.ge1_head(h).squeeze(1)  # (B,)
        ge2_logit = self.ge2_head(h).squeeze(1)  # (B,)

        return h, ge1_logit, ge2_logit


# ------------------------------------------------------------------ #
#  Ordinal Context Model                                             #
# ------------------------------------------------------------------ #
class OrdinalContextModel(nn.Module):
    """
    pretrained trunk + large kernel context branch + ordinal head

    핵심:
      - trunk는 pretrained semantics 보존
      - branch는 global degradation context 제공
      - final_prob = (1-gamma) * p_main + gamma * p_ord
      - gamma는 작은 범위로 제한하여 trunk를 절대 덮어쓰지 못하게 함
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int = 3,
        img_size: int = 512,
        branch_mid_channels: int = 256,
        ordinal_hidden_dim: int = 256,
        gamma_max: float = 0.35,
        main_ce_lambda: float = 0.3,
        ordinal_lambda: float = 0.5,
        monotonic_lambda: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.gamma_max = gamma_max
        self.main_ce_lambda = main_ce_lambda
        self.ordinal_lambda = ordinal_lambda
        self.monotonic_lambda = monotonic_lambda

        self.trunk = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        trunk_dim = self.trunk.num_features

        self.trunk_pool = nn.AdaptiveAvgPool2d(1)

        self.main_head = nn.Linear(trunk_dim, num_classes)
        nn.init.normal_(self.main_head.weight, std=0.02)
        nn.init.zeros_(self.main_head.bias)

        self.branch = LargeKernelBlock(
            in_channels=trunk_dim,
            mid_channels=branch_mid_channels,
        )

        self.ordinal_head = OrdinalHead(
            trunk_dim=trunk_dim,
            branch_dim=branch_mid_channels,
            hidden_dim=ordinal_hidden_dim,
        )

        # gamma in (0, gamma_max), 초기에는 매우 작게
        self.fusion_logit = nn.Parameter(torch.tensor(-2.0))

    def _build_ordinal_prob(
        self,
        ge1_logit: torch.Tensor,
        ge2_logit: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        q1 = sigmoid(ge1_logit)
        q2_raw = sigmoid(ge2_logit)

        monotonicity(q2 <= q1)를 강제로 만족시키기 위해
        q2 = q1 * q2_raw 로 조립
        """
        q1 = torch.sigmoid(ge1_logit)
        q2_raw = torch.sigmoid(ge2_logit)
        q2 = q1 * q2_raw

        p0 = 1.0 - q1
        p1 = q1 - q2
        p2 = q2

        ord_prob = torch.stack([p0, p1, p2], dim=1)
        ord_prob = ord_prob.clamp_min(1e-8)
        ord_prob = ord_prob / ord_prob.sum(dim=1, keepdim=True)
        return ord_prob, q1, q2

    def forward(self, x: torch.Tensor) -> dict:
        feat_map = self.trunk.forward_features(x)

        trunk_feat = self.trunk_pool(feat_map).flatten(1)
        main_logits = self.main_head(trunk_feat)
        main_prob = torch.softmax(main_logits, dim=1)

        branch_feat = self.branch(feat_map)
        _, ge1_logit, ge2_logit = self.ordinal_head(trunk_feat, branch_feat)

        ord_prob, q1, q2 = self._build_ordinal_prob(ge1_logit, ge2_logit)

        gamma = self.gamma_max * torch.sigmoid(self.fusion_logit)
        final_prob = (1.0 - gamma) * main_prob + gamma * ord_prob
        final_prob = final_prob.clamp_min(1e-8)
        final_prob = final_prob / final_prob.sum(dim=1, keepdim=True)
        final_logits = torch.log(final_prob)

        return {
            "final_logits": final_logits,
            "main_logits": main_logits,
            "main_prob": main_prob,
            "ord_prob": ord_prob,
            "ge1_logit": ge1_logit,
            "ge2_logit": ge2_logit,
            "q1": q1,
            "q2": q2,
            "gamma": gamma,
        }

    def compute_train_loss(
        self,
        output: dict,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        final_logits = output["final_logits"]
        main_logits = output["main_logits"]
        ge1_logit = output["ge1_logit"]
        ge2_logit = output["ge2_logit"]

        y_ge1 = (labels >= 1).float()
        y_ge2 = (labels >= 2).float()

        loss_final = criterion(final_logits, labels)
        loss_main = criterion(main_logits, labels)

        # autocast-safe ordinal supervision
        loss_ge1 = F.binary_cross_entropy_with_logits(ge1_logit, y_ge1)
        loss_ge2 = F.binary_cross_entropy_with_logits(ge2_logit, y_ge2)
        loss_ord = loss_ge1 + loss_ge2

        # soft monotonic penalty: q2_raw should not exceed q1 too much in logit sense
        # practical regularizer; small weight
        mono_penalty = F.relu(torch.sigmoid(ge2_logit) - torch.sigmoid(ge1_logit)).mean()

        return (
            loss_final
            + self.main_ce_lambda * loss_main
            + self.ordinal_lambda * loss_ord
            + self.monotonic_lambda * mono_penalty
        )

    def forward_main_only(self, x: torch.Tensor) -> torch.Tensor:
        output = self.forward(x)
        return output["final_logits"]

    def get_alphas(self) -> List[float]:
        gamma = self.gamma_max * torch.sigmoid(self.fusion_logit)
        return [float(gamma.detach().cpu().item())]

    # ---------------------------------------------------------------- #
    #  Stage별 parameter groups                                         #
    # ---------------------------------------------------------------- #
    def get_param_groups_stage1(
        self,
        branch_lr: float = 1e-3,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> List[dict]:
        for p in self.trunk.parameters():
            p.requires_grad = False
        for p in self.branch.parameters():
            p.requires_grad = True
        for p in self.main_head.parameters():
            p.requires_grad = True
        for p in self.ordinal_head.parameters():
            p.requires_grad = True
        self.fusion_logit.requires_grad = True

        return [
            {
                "params": list(self.branch.parameters()),
                "lr": branch_lr,
                "weight_decay": weight_decay,
                "name": "branch",
            },
            {
                "params": list(self.main_head.parameters()),
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "main_head",
            },
            {
                "params": list(self.ordinal_head.parameters()) + [self.fusion_logit],
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "ordinal_head",
            },
        ]

    def get_param_groups_stage2(
        self,
        trunk_last_lr: float = 5e-5,
        branch_lr: float = 2e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> List[dict]:
        for p in self.trunk.parameters():
            p.requires_grad = False

        last_stage = self._get_last_stage()
        if last_stage is not None:
            for p in last_stage.parameters():
                p.requires_grad = True

        groups = [
            {
                "params": list(self.branch.parameters()),
                "lr": branch_lr,
                "weight_decay": weight_decay,
                "name": "branch",
            },
            {
                "params": list(self.main_head.parameters()),
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "main_head",
            },
            {
                "params": list(self.ordinal_head.parameters()) + [self.fusion_logit],
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "ordinal_head",
            },
        ]
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
        for p in self.trunk.parameters():
            p.requires_grad = True

        groups = [
            {
                "params": list(self.branch.parameters()),
                "lr": branch_lr,
                "weight_decay": weight_decay,
                "name": "branch",
            },
            {
                "params": list(self.main_head.parameters()),
                "lr": lr_head,
                "weight_decay": weight_decay,
                "name": "main_head",
            },
            {
                "params": list(self.ordinal_head.parameters()) + [self.fusion_logit],
                "lr": lr_head,
                "weight_decay": weight_decay,
                "name": "ordinal_head",
            },
        ]
        groups.extend(self._get_llrd_groups(lr_last, lr_mid, lr_first, weight_decay))
        return groups

    # ---------------------------------------------------------------- #
    #  내부 헬퍼                                                        #
    # ---------------------------------------------------------------- #
    def _get_last_stage(self) -> Optional[nn.Module]:
        t = self.trunk
        if hasattr(t, "layer4"):
            return t.layer4
        if hasattr(t, "stages"):
            return list(t.stages)[-1]
        if hasattr(t, "blocks"):
            return list(t.blocks)[-1]
        return None

    def _get_llrd_groups(
        self,
        lr_last: float,
        lr_mid: float,
        lr_first: float,
        weight_decay: float,
    ) -> List[dict]:
        t = self.trunk
        groups = []

        if hasattr(t, "layer4"):  # ResNet
            stage_map = [
                ("last", [t.layer4], lr_last),
                ("mid", [t.layer3, t.layer2], lr_mid),
                ("first", [t.layer1], lr_first),
            ]
            for name, mods, lr in stage_map:
                params = [p for m in mods for p in m.parameters() if p.requires_grad]
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
                if hasattr(t, attr):
                    stem_params += [p for p in getattr(t, attr).parameters() if p.requires_grad]
            if stem_params:
                groups.append(
                    {
                        "params": stem_params,
                        "lr": lr_first,
                        "weight_decay": weight_decay,
                        "name": "trunk_stem",
                    }
                )

        elif hasattr(t, "blocks"):  # EfficientNet / Xception
            all_blocks = list(t.blocks)
            n = len(all_blocks)
            chunk = max(1, n // 3)
            for i, lr in enumerate([lr_last, lr_mid, lr_first]):
                sub = all_blocks[i * chunk: (i + 1) * chunk]
                params = [p for m in sub for p in m.parameters() if p.requires_grad]
                if params:
                    groups.append(
                        {
                            "params": params,
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "name": f"trunk_block{i}",
                        }
                    )
        else:
            params = [p for p in t.parameters() if p.requires_grad]
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


# ------------------------------------------------------------------ #
#  Factory                                                           #
# ------------------------------------------------------------------ #
def build_context_model(
    backbone: str,
    strategy: str,
    num_classes: int = 3,
    img_size: int = 512,
    stage_kernel_sizes: Optional[List[int]] = None,
    branch_mid_channels: int = 256,
):
    if strategy != "ordinal_context":
        raise ValueError(f"Unsupported strategy: {strategy}")

    return OrdinalContextModel(
        backbone=backbone,
        num_classes=num_classes,
        img_size=img_size,
        branch_mid_channels=branch_mid_channels,
        ordinal_hidden_dim=256,
        gamma_max=0.35,
        main_ce_lambda=0.3,
        ordinal_lambda=0.5,
        monotonic_lambda=0.1,
    )


# ------------------------------------------------------------------ #
#  smoke test                                                        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    backbone = sys.argv[1] if len(sys.argv) > 1 else "resnet101"
    strategy = sys.argv[2] if len(sys.argv) > 2 else "ordinal_context"
    print(f"Smoke test: backbone={backbone}  strategy={strategy}")

    model = build_context_model(backbone, strategy)
    x = torch.randn(2, 3, 512, 512)

    output = model(x)
    print(f"  final_logits : {output['final_logits'].shape}")
    print(f"  main_logits  : {output['main_logits'].shape}")
    print(f"  ord_prob     : {output['ord_prob'].shape}")
    print(f"  gamma        : {model.get_alphas()}")

    total = sum(p.numel() for p in model.parameters())
    trunk = sum(p.numel() for p in model.trunk.parameters())
    branch = sum(p.numel() for p in model.branch.parameters())
    print(f"  total params    : {total:,}")
    print(f"  trunk params    : {trunk:,}")
    print(f"  branch params   : {branch:,}")

    print("\n  [Stage 1]")
    for g in model.get_param_groups_stage1():
        cnt = sum(p.numel() for p in g["params"] if p.requires_grad)
        print(f"    {g['name']:20s} lr={g['lr']:.0e}  params={cnt:,}")

    print("\n  [Stage 3]")
    for g in model.get_param_groups_stage3():
        cnt = sum(p.numel() for p in g["params"] if p.requires_grad)
        print(f"    {g['name']:20s} lr={g['lr']:.0e}  params={cnt:,}")

    print("\nOK")