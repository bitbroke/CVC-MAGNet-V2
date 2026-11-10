"""Three Comparable Models for CVC-Mamba-Hybrid-01.

All models share the exact same weights from the frozen S4 Seed-43 checkpoint.
New context modules in Arm B and Arm C use zero-initialized residual projections,
so their step-0 predictions are mathematically IDENTICAL to Arm A.

Arms:
- Arm A: Current model (EfficientNet-B0 + lateral multiscale decoder), continuation control.
- Arm B: Current model + FourWayMamba2DBlock at stride 32.
- Arm C: Current model + ConvContextBlock at stride 32 (capacity-matched control).
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import timm

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.segmentation_v3 import CVCSegmentationV3Net, DecoderBlock
from experiments.cvc_mamba_hybrid_20261017.src.models.mamba_context import (
    FourWayMamba2DBlock,
    ConvContextBlock,
)

FROZEN_S4_CKPT = (
    PROJECT_ROOT
    / "experiments"
    / "cvc_mamba_hybrid_20261017"
    / "frozen_reference"
    / "runs"
    / "s4_experiment_seed43"
    / "checkpoints"
    / "final_segmentation_v3.pt"
)


class ArmAContinuationNet(nn.Module):
    """Arm A: Matched continuation of current model from S4 reference checkpoint."""

    def __init__(self, load_checkpoint: bool = True, freeze_encoder_bn: bool = True):
        super().__init__()
        self.net = CVCSegmentationV3Net(
            pretrained=False,
            freeze_encoder_bn=freeze_encoder_bn,
            decoder_widths=(128, 64, 32, 16, 16),
        )
        if load_checkpoint:
            state = torch.load(FROZEN_S4_CKPT, map_location="cpu", weights_only=False)
            model_weights = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
            self.net.load_state_dict(model_weights)

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.train(mode)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net(x)


class ArmBMambaHybridNet(nn.Module):
    """Arm B: Current model + residual 4-way Mamba S6 block at Stride 32.
    
    Zero-initialized residual projection guarantees that at step 0,
    the output is mathematically identical to Arm A.
    """

    def __init__(self, load_checkpoint: bool = True, freeze_encoder_bn: bool = True):
        super().__init__()
        self.freeze_encoder_bn = freeze_encoder_bn
        self.encoder = timm.create_model("efficientnet_b0", features_only=True, pretrained=False)
        enc_channels = [16, 24, 40, 112, 320]
        w_d32, w_d16, w_d8, w_d4, w_d2 = (128, 64, 32, 16, 16)

        # Bottleneck conv
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[4], w_d32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, w_d32),
            nn.ReLU(inplace=True),
        )

        # Arm B: 4-way Mamba S6 context module
        self.mamba_context = FourWayMamba2DBlock(dim=w_d32, d_state=16, dt_rank=8)

        # Decoder blocks identical to CVCSegmentationV3Net
        self.block16 = DecoderBlock(w_d32, enc_channels[3], w_d16, num_groups=8)
        self.block8 = DecoderBlock(w_d16, enc_channels[2], w_d8, num_groups=8)
        self.block4 = DecoderBlock(w_d8, enc_channels[1], w_d4, num_groups=8)
        self.block2 = DecoderBlock(w_d4, enc_channels[0], w_d2, num_groups=8)

        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(w_d2, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        if load_checkpoint:
            state = torch.load(FROZEN_S4_CKPT, map_location="cpu", weights_only=False)
            model_weights = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
            incompatible = self.load_state_dict(model_weights, strict=False)
            assert len(incompatible.unexpected_keys) == 0, f"Unexpected keys: {incompatible.unexpected_keys}"
            missing_prefixes = set(k.split(".")[0] for k in incompatible.missing_keys)
            assert missing_prefixes == {"mamba_context"}, f"Missing keys outside mamba_context: {missing_prefixes}"

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_encoder_bn:
            for m in self.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats[0], feats[1], feats[2], feats[3], feats[4]

        b32 = self.bottleneck(f4)
        b32 = self.mamba_context(b32)  # Zero-initialized residual => b32 + 0 == b32 initially

        d16 = self.block16(b32, f3)
        d8 = self.block8(d16, f2)
        d4 = self.block4(d8, f1)
        d2 = self.block2(d4, f0)
        seg_logits = self.block1(d2)

        return {
            "seg_logits": seg_logits,
            "seg_probs": torch.sigmoid(seg_logits),
        }


class ArmCConvControlNet(nn.Module):
    """Arm C: Current model + residual conventional Conv block at Stride 32.
    
    Capacity-matched control for Arm B. Zero-initialized residual projection guarantees
    that at step 0, the output is mathematically identical to Arm A.
    """

    def __init__(self, load_checkpoint: bool = True, freeze_encoder_bn: bool = True):
        super().__init__()
        self.freeze_encoder_bn = freeze_encoder_bn
        self.encoder = timm.create_model("efficientnet_b0", features_only=True, pretrained=False)
        enc_channels = [16, 24, 40, 112, 320]
        w_d32, w_d16, w_d8, w_d4, w_d2 = (128, 64, 32, 16, 16)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[4], w_d32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, w_d32),
            nn.ReLU(inplace=True),
        )

        # Arm C: Conventional Conv context module
        self.conv_context = ConvContextBlock(dim=w_d32)

        self.block16 = DecoderBlock(w_d32, enc_channels[3], w_d16, num_groups=8)
        self.block8 = DecoderBlock(w_d16, enc_channels[2], w_d8, num_groups=8)
        self.block4 = DecoderBlock(w_d8, enc_channels[1], w_d4, num_groups=8)
        self.block2 = DecoderBlock(w_d4, enc_channels[0], w_d2, num_groups=8)

        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(w_d2, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        if load_checkpoint:
            state = torch.load(FROZEN_S4_CKPT, map_location="cpu", weights_only=False)
            model_weights = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
            incompatible = self.load_state_dict(model_weights, strict=False)
            assert len(incompatible.unexpected_keys) == 0, f"Unexpected keys: {incompatible.unexpected_keys}"
            missing_prefixes = set(k.split(".")[0] for k in incompatible.missing_keys)
            assert missing_prefixes == {"conv_context"}, f"Missing keys outside conv_context: {missing_prefixes}"

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_encoder_bn:
            for m in self.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats[0], feats[1], feats[2], feats[3], feats[4]

        b32 = self.bottleneck(f4)
        b32 = self.conv_context(b32)

        d16 = self.block16(b32, f3)
        d8 = self.block8(d16, f2)
        d4 = self.block4(d8, f1)
        d2 = self.block2(d4, f0)
        seg_logits = self.block1(d2)

        return {
            "seg_logits": seg_logits,
            "seg_probs": torch.sigmoid(seg_logits),
        }
