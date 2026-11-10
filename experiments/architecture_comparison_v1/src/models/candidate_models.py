"""The Four Architecture Candidates for CVC Union Segmentation.

Track A Candidates:
1. EfficientNetBaselineNet: EfficientNet-B0 + lateral multiscale decoder
2. SegFormerB0Net: SegFormer-B0 (nvidia/mit-b0) + lightweight MLP segmentation head
3. SmallHRNet: HRNet-W18-Small-v2 with high-resolution multiscale fusion head
4. EfficientNetMambaHybridNet: EfficientNet-B0 + 128-channel Mamba S6 bottleneck block + multiscale decoder
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.segmentation_v3 import CVCSegmentationV3Net, DecoderBlock
from experiments.architecture_comparison_v1.src.models.mamba_block import Mamba2DBottleneckBlock


class EfficientNetBaselineNet(nn.Module):
    """Candidate 1: EfficientNet-B0 Baseline."""

    def __init__(self, pretrained: bool = True, freeze_encoder_bn: bool = True):
        super().__init__()
        self.net = CVCSegmentationV3Net(
            pretrained=pretrained,
            freeze_encoder_bn=freeze_encoder_bn,
            decoder_widths=(128, 64, 32, 16, 16),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.train(mode)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net(x)


class SegFormerB0Net(nn.Module):
    """Candidate 2: SegFormer-B0 (nvidia/mit-b0) Transformer."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        config = SegformerConfig.from_pretrained(
            "nvidia/mit-b0",
            num_labels=1,
            reshape_last_stage=True,
        )
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0",
                config=config,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        b, c, h, w = x.shape
        out = self.model(pixel_values=x)
        # Segformer outputs logits at stride 4: [B, 1, H/4, W/4]
        logits_stride4 = out.logits
        # Bilinearly interpolate to original image input resolution
        logits = F.interpolate(logits_stride4, size=(h, w), mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)

        return {
            "seg_logits": logits,
            "seg_probs": probs,
        }


class SmallHRNet(nn.Module):
    """Candidate 3: HRNet-W18-Small-v2 with Multiscale Fusion Head."""

    def __init__(self, pretrained: bool = True, freeze_encoder_bn: bool = True):
        super().__init__()
        self.freeze_encoder_bn = freeze_encoder_bn
        # HRNet-W18-Small-v2 features: output features at strides [2, 4, 8, 16, 32]
        self.encoder = timm.create_model(
            "hrnet_w18_small_v2",
            features_only=True,
            pretrained=pretrained,
        )
        feature_info = self.encoder.feature_info.channels()
        c0, c1, c2, c3 = feature_info[0], feature_info[1], feature_info[2], feature_info[3]

        # 1x1 convs to unify feature dimensions to 64
        self.proj0 = nn.Sequential(nn.Conv2d(c0, 64, 1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.proj1 = nn.Sequential(nn.Conv2d(c1, 64, 1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.proj2 = nn.Sequential(nn.Conv2d(c2, 64, 1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.proj3 = nn.Sequential(nn.Conv2d(c3, 64, 1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(inplace=True))

        # Multiscale fusion conv at stride 2 (f0 size)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(64 * 4, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )

        # Final head: stride 2 -> stride 1 (native model input size)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_encoder_bn:
            for m in self.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        b, c, h, w = x.shape
        feats = self.encoder(x)
        f0, f1, f2, f3 = feats[0], feats[1], feats[2], feats[3]
        target_size = (f0.shape[2], f0.shape[3])

        p0 = self.proj0(f0)
        p1 = F.interpolate(self.proj1(f1), size=target_size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.proj2(f2), size=target_size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.proj3(f3), size=target_size, mode="bilinear", align_corners=False)

        fused = self.fuse_conv(torch.cat([p0, p1, p2, p3], dim=1))
        logits = self.head(fused)
        probs = torch.sigmoid(logits)

        return {
            "seg_logits": logits,
            "seg_probs": probs,
        }


class EfficientNetMambaHybridNet(nn.Module):
    """Candidate 4: EfficientNet-B0 with residual Mamba S6 block at Stride 32.
    
    Shares the exact same encoder and lateral decoder architecture as EfficientNetBaselineNet,
    with a 128-channel residual Mamba S6 block inserted at the stride-32 bottleneck.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_encoder_bn: bool = True,
        decoder_widths: Tuple[int, int, int, int, int] = (128, 64, 32, 16, 16),
    ):
        super().__init__()
        self.freeze_encoder_bn = freeze_encoder_bn
        self.encoder = timm.create_model("efficientnet_b0", features_only=True, pretrained=pretrained)
        enc_channels = [16, 24, 40, 112, 320]

        w_d32, w_d16, w_d8, w_d4, w_d2 = decoder_widths

        # Bottleneck at stride 32: Conv -> Mamba2DBottleneckBlock (128 channels)
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(enc_channels[4], w_d32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, w_d32),
            nn.ReLU(inplace=True),
        )
        self.mamba_bottleneck = Mamba2DBottleneckBlock(dim=w_d32)

        # Multiscale lateral decoders (identical to CVCSegmentationV3Net)
        self.block16 = DecoderBlock(w_d32, enc_channels[3], w_d16, num_groups=8)
        self.block8 = DecoderBlock(w_d16, enc_channels[2], w_d8, num_groups=8)
        self.block4 = DecoderBlock(w_d8, enc_channels[1], w_d4, num_groups=8)
        self.block2 = DecoderBlock(w_d4, enc_channels[0], w_d2, num_groups=8)

        # Final detail stage: stride 2 -> stride 1
        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(w_d2, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

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

        # Stride 32 with residual Mamba S6 long-range context
        b32 = self.bottleneck_conv(f4)
        b32 = self.mamba_bottleneck(b32)

        d16 = self.block16(b32, f3)
        d8 = self.block8(d16, f2)
        d4 = self.block4(d8, f1)
        d2 = self.block2(d4, f0)
        seg_logits = self.block1(d2)

        return {
            "seg_logits": seg_logits,
            "seg_probs": torch.sigmoid(seg_logits),
        }
