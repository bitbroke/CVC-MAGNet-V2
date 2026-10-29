"""Adapted Coarse Segmentation Model (CoarseSegNet).

Based on MAG-Net's UNet_UP_Global architecture (Wang et al., 2022):
- Base-32 U-Net encoder-decoder backbone.
- Pyramid Pooling Module (PPM) at the bottleneck to capture multi-scale catheter context.
- Modernized with F.interpolate (replacing deprecated F.upsample).
- Direct logit outputs with numerical stability for BCEWithLogitsLoss.
"""
from typing import Optional, Tuple
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


class PyramidPoolingModule(nn.Module):
    """Multi-scale Pyramid Pooling Module (PPM) from PSPNet."""

    def __init__(self, in_channels: int, out_channels: int, pool_sizes: Tuple[int, ...] = (1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
            )
            for size in pool_sizes
        ])
        total_ch = in_channels + (in_channels // len(pool_sizes)) * len(pool_sizes)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(total_ch, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        priors = [
            F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False)
            for stage in self.stages
        ]
        priors.append(x)
        cat = torch.cat(priors, dim=1)
        return self.bottleneck(cat)


class CoarseSegNet(nn.Module):
    """Coarse global catheter segmentation network with Pyramid Pooling Module."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        target_size: Optional[Tuple[int, int]] = (512, 512),
    ):
        super(CoarseSegNet, self).__init__()
        self.target_size = target_size
        b = base_channels

        # Encoder
        self.conv1 = DoubleConv(in_channels, b)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(b, b * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(b * 2, b * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(b * 4, b * 8)

        # Bottleneck Pyramid Pooling
        self.ppm = PyramidPoolingModule(b * 8, b * 8, pool_sizes=(1, 2, 3, 6))

        # Decoder
        self.up7 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(b * 8, b * 4)

        self.up8 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(b * 4, b * 2)

        self.up9 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(b * 2, b)

        # Segmentation Head (Logits)
        self.head = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)

        # Bottleneck
        c4 = self.ppm(c4)

        # Decoder
        u7 = self.up7(c4)
        m7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(m7)

        u8 = self.up8(c7)
        m8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(m8)

        u9 = self.up9(c8)
        m9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(m9)

        logits = self.head(c9)
        return logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary mask and probability map."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        binary = (probs > threshold).float()
        return binary, probs
