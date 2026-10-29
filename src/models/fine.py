"""Adapted Fine Segmentation Model (FineSegNet).

Based on MAG-Net's UNet_SW1_PAM_D architecture (Wang et al., 2022):
- 2-channel input: [Image Crop, Coarse Mask Crop].
- Learned Mask Attention (Attention Spatial Weight - ASW) to focus on catheter region.
- Position Attention Module (PAM) at bottleneck (stride 8) to capture long-range catheter continuity.
- Dense Upsampling Convolution (DUC with PixelShuffle) for artifact-free fine feature reconstruction.
- Dual outputs: (seg_logits, attention_mask) for multi-task supervision.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DoubleUPConv(nn.Module):
    """Decoder block: two 3x3 convs preserving channel count after skip concatenation."""

    def __init__(self, in_ch: int):
        super(DoubleUPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PositionAttentionModule(nn.Module):
    """Self-attention across spatial positions to preserve catheter continuity."""

    def __init__(self, in_dim: int):
        super(PositionAttentionModule, self).__init__()
        self.chanel_in = in_dim
        inter_dim = max(1, in_dim // 8)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)  # (B, N, C')
        proj_key = self.key_conv(x).view(b, -1, h * w)                        # (B, C', N)
        energy = torch.bmm(proj_query, proj_key)                               # (B, N, N)
        attention = self.softmax(energy)                                       # (B, N, N)
        proj_value = self.value_conv(x).view(b, -1, h * w)                     # (B, C, N)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                # (B, C, N)
        out = out.view(b, c, h, w)
        out = self.gamma * out + x
        return out


class DenseUpsamplingConv(nn.Module):
    """Dense Upsampling Convolution (DUC) using PixelShuffle."""

    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int = 2):
        super(DenseUpsamplingConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * (upscale_factor ** 2))
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class FineSegNet(nn.Module):
    """Fine ROI catheter refinement network with Mask Attention and Position Attention."""

    def __init__(
        self,
        in_channels: int = 2,  # [image_crop, coarse_mask_crop]
        out_channels: int = 1,
        base_channels: int = 32,
    ):
        super(FineSegNet, self).__init__()
        b = base_channels

        # Attention Spatial Weight (ASW) module
        self.conv_s1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
        )

        # Encoder
        self.conv1 = DoubleConv(1, b)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(b, b * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(b * 2, b * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(b * 4, b * 8)

        # Bottleneck Position Attention
        self.pam = PositionAttentionModule(b * 8)

        # Decoder using DUC PixelShuffle
        self.up4 = DenseUpsamplingConv(b * 8, b * 4, upscale_factor=2)
        self.decoder4 = DoubleUPConv(b * 8)  # concat up4 (b*4) + c3 (b*4)

        self.up3 = DenseUpsamplingConv(b * 8, b * 2, upscale_factor=2)
        self.decoder3 = DoubleUPConv(b * 4)  # concat up3 (b*2) + c2 (b*2)

        self.up2 = DenseUpsamplingConv(b * 4, b, upscale_factor=2)
        self.decoder2 = DoubleUPConv(b * 2)  # concat up2 (b) + c1 (b)

        # Segmentation Head (Logits)
        self.head = nn.Conv2d(b * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: (B, 2, H, W) tensor where x[:, 0] is image and x[:, 1] is coarse mask.
            
        Returns:
            seg_logits: (B, 1, H, W) raw segmentation logits.
            mask_att: (B, 1, H, W) learned spatial attention mask in [0, 1].
        """
        img, coarse_mask = x.split([1, 1], dim=1)

        # Learned Mask Attention (ASW)
        att_logits = self.conv_s1(coarse_mask)
        mask_att = torch.sigmoid(att_logits)

        # Modulate radiograph crop with learned catheter attention
        attended_img = img * mask_att

        # Encoder
        c1 = self.conv1(attended_img)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)

        # Bottleneck Position Attention
        c5 = self.pam(c4)

        # Decoder
        u4 = self.up4(c5)
        m4 = torch.cat([u4, c3], dim=1)
        d4 = self.decoder4(m4)

        u3 = self.up3(d4)
        m3 = torch.cat([u3, c2], dim=1)
        d3 = self.decoder3(m3)

        u2 = self.up2(d3)
        m2 = torch.cat([u2, c1], dim=1)
        d2 = self.decoder2(m2)

        seg_logits = self.head(d2)
        return seg_logits, mask_att

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return binary mask, probabilities, and learned attention mask."""
        logits, mask_att = self.forward(x)
        probs = torch.sigmoid(logits)
        binary = (probs > threshold).float()
        return binary, probs, mask_att
