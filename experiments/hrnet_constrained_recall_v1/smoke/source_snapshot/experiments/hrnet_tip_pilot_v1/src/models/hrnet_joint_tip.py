"""HRNet-W18 Joint Segmentation and Distal-Tip Heatmap Model (T1).

Architecture:
- Shared High-Resolution Encoder: timm 'hrnet_w18'
- Multi-Resolution Representations: Stages 1, 2, 3, 4 projected and fused at stride 4 (64 channels)
- Segmentation Head: 4x upsampling with GroupNorm refinement -> [B, 1, 768, 768] seg_logits
- Direct Multi-Tip Heatmap Head: Stride-4 spatial convolutional head -> [B, 1, 192, 192] tip_logits
- State Restoration: Exact weights loaded from frozen HRNet seed-42 checkpoint.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class HRNetJointTipNet(nn.Module):
    """Joint CVC Catheter Segmentation and Distal-Tip Heatmap Localization Model."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.encoder = timm.create_model("hrnet_w18", features_only=True, pretrained=pretrained)

        # Projections to common 64-channel space at stride 4
        self.proj1 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(512, 64, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(1024, 64, kernel_size=1, bias=False)

        # Multi-scale fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 4, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )

        # Stride 4 -> Stride 1 Segmentation Head (768x768 output)
        self.seg_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # Stride 4 Distal Tip Heatmap Head (192x192 output)
        self.tip_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Initialize tip_head final layer bias to -4.0 (initial sigmoid ~ 0.018 for sparse positive background)
        nn.init.kaiming_normal_(self.tip_head[0].weight, nonlinearity="relu")
        nn.init.xavier_normal_(self.tip_head[3].weight)
        nn.init.constant_(self.tip_head[3].bias, -4.0)

    def load_from_frozen_hrnet(self, checkpoint_path: Union[str, Path]):
        """Load encoder, projections, fusion and segmentation head weights from frozen HRNet."""
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Map keys from HRNetW18Net to HRNetJointTipNet
        # In HRNetW18Net, the segmentation head is named 'head'
        joint_sd = self.state_dict()
        loaded_keys = 0
        for k, v in sd.items():
            if k.startswith("head."):
                mapped_k = "seg_head." + k[len("head."):]
            else:
                mapped_k = k
            if mapped_k in joint_sd and joint_sd[mapped_k].shape == v.shape:
                joint_sd[mapped_k] = v
                loaded_keys += 1

        self.load_state_dict(joint_sd)
        print(f"[HRNetJointTipNet] Successfully loaded {loaded_keys} shared parameter tensors from {Path(checkpoint_path).name}.")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feats = self.encoder(x)
        f1, f2, f3, f4 = feats[1], feats[2], feats[3], feats[4]
        target_size = f1.shape[2:]

        p1 = self.proj1(f1)
        p2 = F.interpolate(self.proj2(f2), size=target_size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.proj3(f3), size=target_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.proj4(f4), size=target_size, mode="bilinear", align_corners=False)

        fused = self.fusion(torch.cat([p1, p2, p3, p4], dim=1))

        seg_logits = self.seg_head(fused)
        tip_logits = self.tip_head(fused)

        return {
            "seg_logits": seg_logits,
            "seg_probs": torch.sigmoid(seg_logits),
            "tip_logits": tip_logits,
            "tip_probs": torch.sigmoid(tip_logits),
            "fused_features": fused,
        }
