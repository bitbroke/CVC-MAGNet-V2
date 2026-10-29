"""CVC-Segmentation-V4 Architecture.

Key Innovations for High-Accuracy Catheter Segmentation:
1. Multi-Backbone Support via timm (EfficientNet-B0/B2/B4, ResNet-34D, ConvNeXt-Tiny).
2. Attention Gating on lateral skip connections:
   - Uses coarser decoder features as gating signal to filter out rib, clavicle, and spine clutter.
   - Preserves high-frequency catheter boundaries while suppressing irrelevant bone edges.
3. Deep Supervision:
   - Auxiliary prediction heads at stride 8, stride 4, and stride 2.
   - Ensures strong gradient flow to all decoder depths, preventing vanishing gradients on thin tubular structures.
4. GroupNorm in all decoder and attention blocks:
   - Immune to small batch size fluctuations (crucial for 4GB VRAM training).
"""

from typing import Dict, List, Optional, Tuple, Union
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Additive Attention Gate for suppressing irrelevant background features (ribs, spine)."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, inter_channels), num_channels=inter_channels),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, inter_channels), num_channels=inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Coarser decoder feature tensor (gating signal) [B, C_g, H_g, W_g]
            x: Skip connection feature tensor [B, C_x, H_x, W_x]
        """
        # Upsample g to match x spatial dimensions if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        phi = self.relu(self.w_g(g) + self.w_x(x))
        alpha = self.psi(phi)
        return x * alpha


class AttentionDecoderBlock(nn.Module):
    """Decoder block with Additive Attention Gating, lateral fusion, and GroupNorm."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_groups: int = 8,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention and (skip_channels > 0)
        
        if self.use_attention:
            inter_ch = max(16, out_channels // 2)
            self.attn = AttentionGate(
                gate_channels=in_channels,
                skip_channels=skip_channels,
                inter_channels=inter_ch,
            )
            self.lateral = nn.Sequential(
                nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(num_groups, out_channels), out_channels),
                nn.ReLU(inplace=True),
            )
        elif skip_channels > 0:
            self.attn = None
            self.lateral = nn.Sequential(
                nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(num_groups, out_channels), out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.attn = None
            self.lateral = nn.Identity()

        total_in = in_channels + (out_channels if skip_channels > 0 else 0)
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Upsample coarser features by 2x
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            if x_up.shape[2:] != skip.shape[2:]:
                x_up = F.interpolate(x_up, size=skip.shape[2:], mode="bilinear", align_corners=False)

            if self.use_attention and self.attn is not None:
                skip_attn = self.attn(g=x, x=skip)
                skip_lat = self.lateral(skip_attn)
            else:
                skip_lat = self.lateral(skip)

            x_cat = torch.cat([x_up, skip_lat], dim=1)
        else:
            x_cat = x_up

        return self.conv(x_cat)


class CVCSegmentationV4Net(nn.Module):
    """CVC-Segmentation-V4: Pretrained Encoder + Attention Gates + Deep Supervision."""

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_encoder_bn: bool = True,
        deep_supervision: bool = True,
        use_attention: bool = True,
        decoder_widths: Tuple[int, int, int, int, int] = (128, 64, 32, 16, 16),
    ):
        super().__init__()
        self.backbone_name = backbone
        self.freeze_encoder_bn = freeze_encoder_bn
        self.deep_supervision = deep_supervision
        self.use_attention = use_attention

        # Feature extractor from timm
        self.encoder = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        enc_channels = self.encoder.feature_info.channels()
        # Assumes 5 feature levels [stride 2, 4, 8, 16, 32]
        w_d32, w_d16, w_d8, w_d4, w_d2 = decoder_widths

        # Bottleneck at stride 32
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[4], w_d32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, w_d32), w_d32),
            nn.ReLU(inplace=True),
        )

        # Attention Decoder blocks
        self.block16 = AttentionDecoderBlock(w_d32, enc_channels[3], w_d16, use_attention=use_attention)
        self.block8 = AttentionDecoderBlock(w_d16, enc_channels[2], w_d8, use_attention=use_attention)
        self.block4 = AttentionDecoderBlock(w_d8, enc_channels[1], w_d4, use_attention=use_attention)
        self.block2 = AttentionDecoderBlock(w_d4, enc_channels[0], w_d2, use_attention=use_attention)

        # Final detail head: stride 2 -> full native resolution
        self.final_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(w_d2, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, 16), 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # Deep supervision auxiliary heads
        if deep_supervision:
            self.aux_head8 = nn.Conv2d(w_d8, 1, kernel_size=1)
            self.aux_head4 = nn.Conv2d(w_d4, 1, kernel_size=1)
            self.aux_head2 = nn.Conv2d(w_d2, 1, kernel_size=1)
        else:
            self.aux_head8 = None
            self.aux_head4 = None
            self.aux_head2 = None

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_encoder_bn:
            for m in self.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        input_size = x.shape[2:]
        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats[0], feats[1], feats[2], feats[3], feats[4]

        b32 = self.bottleneck(f4)
        d16 = self.block16(b32, f3)
        d8 = self.block8(d16, f2)
        d4 = self.block4(d8, f1)
        d2 = self.block2(d4, f0)
        seg_logits = self.final_head(d2)

        out = {
            "seg_logits": seg_logits,
            "seg_probs": torch.sigmoid(seg_logits),
        }

        if self.deep_supervision and self.training:
            aux_logits = [
                F.interpolate(self.aux_head8(d8), size=input_size, mode="bilinear", align_corners=False),
                F.interpolate(self.aux_head4(d4), size=input_size, mode="bilinear", align_corners=False),
                F.interpolate(self.aux_head2(d2), size=input_size, mode="bilinear", align_corners=False),
            ]
            out["aux_logits"] = aux_logits

        return out
