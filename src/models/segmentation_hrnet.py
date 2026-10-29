"""CVC-Segmentation-HRNet Model for high-resolution thin catheter representation.

Architecture:
- Backbone: HRNet-W18 (timm features_only=True)
- High-Resolution Representation: Preserves high-resolution features (stride 2 to 32) throughout.
- Representation Head: Fuses multi-resolution stages with bilinear upsampling to stride 2.
- Segmentation Head: 1x1 Conv producing binary catheter logits.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import timm


class CVCSegmentationHRNet(nn.Module):
    """High-Resolution Net for CVC Catheter Segmentation."""

    def __init__(
        self,
        backbone_name: str = "hrnet_w18",
        pretrained: bool = True,
        freeze_encoder_bn: bool = True,
        feature_channels: int = 128,
    ):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        feature_info = self.encoder.feature_info.channels()

        # Projection layers to harmonize multi-stage channels
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, feature_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=feature_channels),
                nn.ReLU(inplace=True),
            )
            for in_ch in feature_info
        ])

        # Fused multi-scale feature aggregation
        total_channels = feature_channels * len(feature_info)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=feature_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Output segmentation head
        self.seg_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feature_channels // 2, 1, kernel_size=1),
        )

        if freeze_encoder_bn:
            for m in self.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[2:]
        features = self.encoder(x)

        # Target size is stage 0 (stride 2)
        target_size = features[0].shape[2:]
        projected = []
        for i, f in enumerate(features):
            proj = self.projections[i](f)
            if proj.shape[2:] != target_size:
                proj = nn.functional.interpolate(proj, size=target_size, mode="bilinear", align_corners=False)
            projected.append(proj)

        fused = self.fusion(torch.cat(projected, dim=1))
        logits = self.seg_head(fused)

        # Ensure exact match with input spatial dimensions
        if logits.shape[2:] != input_size:
            logits = nn.functional.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        probs = torch.sigmoid(logits)
        return {
            "seg_logits": logits,
            "seg_probs": probs,
        }
