"""Mask-Guided Attention Pooling (MGAP) Network with 11-Class Multi-Task Auxiliary Tube Supervision.

Replaces standard Global Average Pooling with 4 orthogonal pooled spatial zones:
1. Global Context Pool: mean over all spatial cells.
2. Catheter-Ribbon Weighted Pool: isolates features strictly along the catheter shaft (S_consensus).
3. Normal CAJ Zone Pool: isolates features where Channel 2 has the cavoatrial junction plateau.
4. Distal Tip Focal Pool: isolates features at the distal tip peak neighborhood.

Joint 11-Class Supervision:
- 3 CVC placement classes (CVC - Normal, CVC - Borderline, CVC - Abnormal)
- 8 Auxiliary tube classes (ETT Normal/Borderline/Abnormal, NGT Normal/Borderline/Abnormal/Incompletely Imaged, Swan-Ganz)
Eliminates false-positive CVC triggers caused by confounding ETTs and NGTs in the mediastinum.
"""

from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

AUX_CLASSES = [
    "ETT - Normal",
    "ETT - Borderline",
    "ETT - Abnormal",
    "NGT - Normal",
    "NGT - Borderline",
    "NGT - Abnormal",
    "NGT - Incompletely Imaged",
    "Swan Ganz Catheter Present",
]


class MaskGuidedAttentionPool2d(nn.Module):
    """Extracts 4 pooled representations (Global + Catheter + CAJ Zone + Tip Peak)."""

    def __init__(self, in_features: int, num_classes: int = 3, num_aux: int = 8):
        super().__init__()
        self.in_features = in_features
        self.norm = nn.LayerNorm(in_features * 4)
        self.drop = nn.Dropout(0.30)
        self.fc_cvc = nn.Linear(in_features * 4, num_classes)
        self.fc_aux = nn.Linear(in_features * 4, num_aux)

    def forward(
        self, feat: torch.Tensor, ch1_seg: torch.Tensor, ch2_coord: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: [B, C, H_f, W_f], ch1_seg: [B, 1, H, W], ch2_coord: [B, 1, H, W]
        H_f, W_f = feat.shape[-2], feat.shape[-1]
        seg_ds = F.adaptive_avg_pool2d(ch1_seg, (H_f, W_f)).clamp(min=0.0, max=1.0)
        ch2_ds = F.adaptive_avg_pool2d(ch2_coord, (H_f, W_f))

        # 1. Global context pool
        v_global = feat.mean(dim=(-2, -1))

        # 2. Catheter-Ribbon weighted pool
        w_cath = seg_ds + 1e-4
        v_cath = (feat * w_cath).sum(dim=(-2, -1)) / w_cath.sum(dim=(-2, -1))

        # 3. Normal CAJ Zone pool (where ch2 has the CAJ plateau: ch2 in [0.68, 0.90])
        w_caj = ((ch2_ds >= 0.68) & (ch2_ds <= 0.90)).float() * (seg_ds + 0.05) + 1e-4
        v_caj = (feat * w_caj).sum(dim=(-2, -1)) / w_caj.sum(dim=(-2, -1))

        # 4. Distal Tip Focal pool (where ch2 has the tip peak >= 0.88 * seg_ds)
        w_tip = (ch2_ds >= 0.88).float() * (seg_ds + 0.10) + 1e-4
        v_tip = (feat * w_tip).sum(dim=(-2, -1)) / w_tip.sum(dim=(-2, -1))

        v_concat = torch.cat([v_global, v_cath, v_caj, v_tip], dim=1)
        v_norm = self.drop(self.norm(v_concat))
        logits_cvc = self.fc_cvc(v_norm)
        logits_aux = self.fc_aux(v_norm)
        return logits_cvc, logits_aux


class MaskGuidedSpatialNet(nn.Module):
    """Full Spatial Net with Mask-Guided Attention Pooling."""

    def __init__(
        self,
        backbone_name: str = "densenet121",
        pretrained: bool = False,
        num_classes: int = 3,
        num_aux: int = 8,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
        )
        in_features = self.backbone.num_features
        self.pool_head = MaskGuidedAttentionPool2d(in_features, num_classes=num_classes, num_aux=num_aux)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 3, H, W] -> Ch0: native image, Ch1: seg ribbon, Ch2: coord field
        ch1 = x[:, 1:2, :, :]
        ch2 = x[:, 2:3, :, :]
        feat = self.backbone.forward_features(x)
        logits_cvc, logits_aux = self.pool_head(feat, ch1, ch2)
        return logits_cvc, logits_aux

    def predict_cvc_probs(self, x: torch.Tensor) -> torch.Tensor:
        logits_cvc, _ = self.forward(x)
        return torch.sigmoid(logits_cvc)

