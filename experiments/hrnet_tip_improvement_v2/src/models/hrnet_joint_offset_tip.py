"""HRNet-W18 Joint Segmentation, Heatmap, and Continuous Offset Localization Model.

Extends HRNetJointTipNet with a continuous local offset regression head (2 channels: dx, dy)
at stride 4 (192x192 at 768px input) to resolve subpixel quantization errors without changing
encoder features, input resolution, or segmentation supervision.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class HRNetJointOffsetTipNet(nn.Module):
    """Joint CVC Catheter Segmentation, Tip Heatmap, and Continuous Offset Model."""

    def __init__(self, pretrained: bool = False, enable_offset: bool = True):
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

        self.enable_offset = enable_offset
        if enable_offset:
            # Stride 4 Continuous Offset Regression Head (192x192x2 output: dx, dy)
            self.offset_head = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, kernel_size=1),
            )
            # Initialize offset head so initial predictions are ~0.0 (preserves base behavior at step 0)
            nn.init.kaiming_normal_(self.offset_head[0].weight, nonlinearity="relu")
            nn.init.zeros_(self.offset_head[3].weight)
            nn.init.zeros_(self.offset_head[3].bias)
        else:
            self.offset_head = None

    def load_from_t1_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load encoder, proj, fusion, seg_head, and tip_head from frozen T1 checkpoint."""
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        model_sd = self.state_dict()
        loaded_keys = 0
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded_keys += 1

        self.load_state_dict(model_sd)
        print(f"[HRNetJointOffsetTipNet] Loaded {loaded_keys} parameter tensors from {Path(checkpoint_path).name}.")

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

        result = {
            "seg_logits": seg_logits,
            "seg_probs": torch.sigmoid(seg_logits),
            "tip_logits": tip_logits,
            "tip_probs": torch.sigmoid(tip_logits),
            "fused_features": fused,
        }

        if self.enable_offset and self.offset_head is not None:
            # Predict offsets [B, 2, 192, 192] clamped to [-0.5, 0.5]
            offset_raw = self.offset_head(fused)
            # Use tanh * 0.5 to strictly bound offsets within [-0.5, 0.5] of cell boundary
            offset_pred = 0.5 * torch.tanh(offset_raw)
            result["offset_pred"] = offset_pred

        return result
