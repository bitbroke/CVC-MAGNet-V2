"""CVC-Segmentation-V3 Architecture.

Design according to CVC Implementation Contract:
- Backbone: ImageNet-pretrained EfficientNet-B0 encoder (via timm features_only)
- Freezes pretrained encoder BatchNorm running statistics for small batch stability
- Lightweight top-down decoder with lateral fusion across strides 32, 16, 8, 4, 2
- GroupNorm in all decoder blocks (immune to batch size 1-2 fluctuations)
- Final detail stage producing full-resolution catheter probability mask [B, 1, H, W]
- Output: Single-task union catheter segmentation logits (no tip or classification heads)
"""

from typing import Dict, List, Optional, Tuple
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Decoder block with lateral fusion, GroupNorm, and upsampling."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        self.lateral = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        ) if skip_channels > 0 else nn.Identity()

        total_in = in_channels + (out_channels if skip_channels > 0 else 0)
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 2x Bilinear upsample
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            skip_lat = self.lateral(skip)
            # Ensure spatial dimension match if odd dimensions occur
            if x_up.shape[2:] != skip_lat.shape[2:]:
                x_up = F.interpolate(x_up, size=skip_lat.shape[2:], mode="bilinear", align_corners=False)
            x_cat = torch.cat([x_up, skip_lat], dim=1)
        else:
            x_cat = x_up
        return self.conv(x_cat)


class CVCSegmentationV3Net(nn.Module):
    """CVC-Segmentation-V3: Pretrained EfficientNet-B0 + Multi-Scale FPN/U-Net Decoder."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_encoder_bn: bool = True,
        decoder_widths: Tuple[int, int, int, int, int] = (128, 64, 32, 16, 16),
    ):
        super().__init__()
        self.freeze_encoder_bn = freeze_encoder_bn

        # Encoder: EfficientNet-B0 feature extractor
        # Feature strides: [2, 4, 8, 16, 32]
        # Feature channels: [16, 24, 40, 112, 320]
        self.encoder = timm.create_model("efficientnet_b0", features_only=True, pretrained=pretrained)
        enc_channels = [16, 24, 40, 112, 320]

        w_d32, w_d16, w_d8, w_d4, w_d2 = decoder_widths

        # Bottleneck at stride 32 (320 -> 128)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[4], w_d32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, w_d32),
            nn.ReLU(inplace=True),
        )

        # Decoder stages with lateral skip connections
        self.block16 = DecoderBlock(w_d32, enc_channels[3], w_d16, num_groups=8)  # stride 32 -> 16
        self.block8 = DecoderBlock(w_d16, enc_channels[2], w_d8, num_groups=8)    # stride 16 -> 8
        self.block4 = DecoderBlock(w_d8, enc_channels[1], w_d4, num_groups=8)     # stride 8 -> 4
        self.block2 = DecoderBlock(w_d4, enc_channels[0], w_d2, num_groups=8)     # stride 4 -> 2

        # Final detail stage: stride 2 -> stride 1 (native model input size, e.g. 512x512)
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
            # Freeze BatchNorm statistics in the pretrained encoder for tiny batches (batch 1-2)
            for m in self.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, 1, H, W] or [B, 3, H, W]. If single-channel, replicated to 3 channels.

        Returns:
            Dict containing 'seg_logits' [B, 1, H, W] and 'seg_probs' [B, 1, H, W].
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats[0], feats[1], feats[2], feats[3], feats[4]

        b32 = self.bottleneck(f4)
        d16 = self.block16(b32, f3)
        d8 = self.block8(d16, f2)
        d4 = self.block4(d8, f1)
        d2 = self.block2(d4, f0)
        seg_logits = self.block1(d2)

        return {
            "seg_logits": seg_logits,
            "seg_probs": torch.sigmoid(seg_logits),
        }
