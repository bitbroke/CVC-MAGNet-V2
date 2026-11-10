"""Dual-Branch Fusion Architecture for Mediastinal CVC Placement.

Implements Step 4 of the CVC Placement Architecture:
- Visual Branch: ConvNeXt-Tiny (or EfficientNet-B0) accepting 384x384 3-channel input
  (Ch0: CLAHE crop, Ch1: seg_prob, Ch2: Carina/Tip spatial map) -> 128-D vector
- Geometric Branch: 2-layer MLP processing 11-D vector -> 32-D vector
- Modality Dropout: During training, p=0.25 probability of zeroing geometric branch
  to prevent visual dependency collapse
- Fusion Head: Concat(128 + 32 = 160) -> 64 -> 3 classes (Normal, Borderline, Abnormal)
- Supports multi-instance evaluation and ablation modes (visual-only, geom-only, fusion)
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeometricMLP(nn.Module):
    """2-layer MLP for 11-D geometric placement coordinates."""

    def __init__(self, in_features: int = 11, hidden_dim: int = 64, out_features: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)


class MediastinalFusionNet(nn.Module):
    """Dual-branch fusion network combining visual mediastinal RoI and geometric coordinates."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        geom_in_dim: int = 25,
        visual_dim: int = 128,
        geom_dim: int = 32,
        num_classes: int = 3,
        modality_dropout_p: float = 0.25,
        dropout_p: float = 0.20,
    ):
        super().__init__()
        self.modality_dropout_p = modality_dropout_p
        self.visual_dim = visual_dim
        self.geom_dim = geom_dim

        # 1. Visual Backbone
        self.backbone = timm.create_model(
            backbone_name,
            in_chans=3,
            pretrained=pretrained,
            num_classes=0,  # Pooling only, returns feature vector
        )
        backbone_out_features = self.backbone.num_features

        self.visual_proj = nn.Sequential(
            nn.Linear(backbone_out_features, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.GELU(),
        )

        # 2. Geometric Branch
        self.geom_mlp = GeometricMLP(
            in_features=geom_in_dim,
            hidden_dim=64,
            out_features=geom_dim,
        )

        # 3. Fusion Head
        fusion_dim = visual_dim + geom_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        roi_img: torch.Tensor,
        geom_vec: torch.Tensor,
        force_drop_geom: bool = False,
        force_drop_visual: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through dual branches with modality dropout.

        Args:
            roi_img: [B, 3, 384, 384] tensor
            geom_vec: [B, 11] tensor
            force_drop_geom: If True, zero out geometric branch (visual-only ablation)
            force_drop_visual: If True, zero out visual branch (geom-only ablation)

        Returns:
            Dict containing 'logits', 'probs', 'visual_feat', 'geom_feat'
        """
        batch_size = roi_img.shape[0]

        # 1. Visual Feature Extraction
        if force_drop_visual:
            v_feat = torch.zeros(batch_size, self.visual_dim, device=roi_img.device, dtype=roi_img.dtype)
        else:
            v_raw = self.backbone(roi_img)
            v_feat = self.visual_proj(v_raw)

        # 2. Geometric Feature Extraction
        if force_drop_geom:
            g_feat = torch.zeros(batch_size, self.geom_dim, device=geom_vec.device, dtype=geom_vec.dtype)
        else:
            g_feat = self.geom_mlp(geom_vec)
            
            # Apply modality dropout during training
            if self.training and self.modality_dropout_p > 0.0:
                mask = (torch.rand(batch_size, 1, device=g_feat.device) >= self.modality_dropout_p).float()
                g_feat = g_feat * mask

        # 3. Fusion & Multi-Label Classification
        fused = torch.cat([v_feat, g_feat], dim=-1)
        logits = self.fusion_head(fused)
        probs = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probs": probs,
            "visual_feat": v_feat,
            "geom_feat": g_feat,
        }

