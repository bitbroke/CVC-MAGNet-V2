"""Model registry for Architecture Comparison V2.

Three matched models:
1. EfficientNet-B0 baseline: ImageNet-pretrained encoder + freshly initialized top-down decoder with lateral fusion.
2. SegFormer-B0: ImageNet-pretrained MiT-B0 encoder + freshly initialized MLP segmentation head.
3. HRNet-W18: ImageNet-pretrained HRNet-W18 encoder + freshly initialized multi-resolution representation fusion head.

All models take [B, 1, H, W] or [B, 3, H, W] and return:
{
    'seg_logits': [B, 1, H, W],
    'seg_probs':  [B, 1, H, W],
}
"""

from typing import Dict, List, Optional, Tuple
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.segmentation_v3 import CVCSegmentationV3Net

FROZEN_S4_CKPT = "C:/Ankshit/CVC-MAGNet-V2/experiments/cvc_mamba_hybrid_20261017/frozen_reference/runs/s4_experiment_seed43/checkpoints/final_segmentation_v3.pt"


class EfficientNetBaselineNet(nn.Module):
    """EfficientNet-B0 baseline with documented ImageNet-pretrained encoder and fresh decoder head."""

    def __init__(self, pretrained: bool = True, freeze_encoder_bn: bool = True):
        super().__init__()
        self.net = CVCSegmentationV3Net(pretrained=pretrained, freeze_encoder_bn=freeze_encoder_bn)

    def train(self, mode: bool = True):
        self.net.train(mode)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net(x)


class SegFormerB0Net(nn.Module):
    """SegFormer-B0 with documented ImageNet-pretrained MiT-B0 encoder and fresh segmentation head."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "imagenet" if pretrained else None
        self.net = smp.Segformer(
            encoder_name="mit_b0",
            encoder_weights=weights,
            in_channels=3,
            classes=1,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.net(x)
        return {
            "seg_logits": logits,
            "seg_probs": torch.sigmoid(logits),
        }


class HRNetW18Net(nn.Module):
    """HRNet-W18 with documented ImageNet-pretrained encoder and fresh multi-resolution fusion head."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model("hrnet_w18", features_only=True, pretrained=pretrained)
        # HRNet-W18 stages:
        # stage 1: stride 4 (128 channels)
        # stage 2: stride 8 (256 channels)
        # stage 3: stride 16 (512 channels)
        # stage 4: stride 32 (1024 channels)
        self.proj1 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(512, 64, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(1024, 64, kernel_size=1, bias=False)

        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 4, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )

        # Stride 4 -> Stride 1 (4x upsampling with GroupNorm refinement)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # stride 2
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # stride 1
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

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
        logits = self.head(fused)
        return {
            "seg_logits": logits,
            "seg_probs": torch.sigmoid(logits),
        }


MODEL_REGISTRY = {
    "EfficientNet_B0": EfficientNetBaselineNet,
    "SegFormer_B0": SegFormerB0Net,
    "HRNet_W18": HRNetW18Net,
}


def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](pretrained=pretrained)
