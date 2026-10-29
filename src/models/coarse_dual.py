"""Dual-Head Coarse Model (Segmentation + Distal Tip Heatmap).

Maintains exact architectural parity with CoarseSegNet:
- Shared Base-32 U-Net encoder-decoder backbone.
- Pyramid Pooling Module (PPM) at bottleneck.
- Segmentation Head: 1x1 Conv predicting catheter binary segmentation logits.
- Distal Tip Head: 3x3 Conv + BN + ReLU + 1x1 Conv predicting distal tip landmark heatmap.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.coarse import DoubleConv, PyramidPoolingModule


class CoarseDualNet(nn.Module):
    """Dual-head coarse network for simultaneous catheter segmentation and distal tip heatmap localization."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        target_size: Optional[Tuple[int, int]] = (512, 512),
    ):
        super(CoarseDualNet, self).__init__()
        self.target_size = target_size
        b = base_channels

        # Shared Encoder
        self.conv1 = DoubleConv(in_channels, b)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(b, b * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(b * 2, b * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(b * 4, b * 8)

        # Bottleneck Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(b * 8, b * 8, pool_sizes=(1, 2, 3, 6))

        # Shared Decoder
        self.up7 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(b * 8, b * 4)

        self.up8 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(b * 4, b * 2)

        self.up9 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(b * 2, b)

        # Head 1: Catheter Segmentation (Logits)
        self.seg_head = nn.Conv2d(b, 1, kernel_size=1)

        # Head 2: Distal Tip Heatmap (Logits)
        self.tip_head = nn.Sequential(
            nn.Conv2d(b, b, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor (B, 1, H, W).
            
        Returns:
            seg_logits: Segmentation logits (B, 1, H, W).
            tip_logits: Distal tip heatmap logits (B, 1, H, W).
        """
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

        # Dual Heads
        seg_logits = self.seg_head(c9)
        tip_logits = self.tip_head(c9)
        return seg_logits, tip_logits

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        seg_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference prediction helper.
        
        Returns:
            seg_binary: Binary segmentation mask (B, 1, H, W).
            seg_probs: Segmentation probabilities (B, 1, H, W).
            tip_probs: Distal tip heatmaps in [0, 1] (B, 1, H, W).
        """
        seg_logits, tip_logits = self.forward(x)
        seg_probs = torch.sigmoid(seg_logits)
        seg_binary = (seg_probs > seg_threshold).float()
        tip_probs = torch.sigmoid(tip_logits)
        return seg_binary, seg_probs, tip_probs


def generate_target_heatmap(
    coords: List[Tuple[float, float]],
    shape: Tuple[int, int] = (512, 512),
    sigma: float = 2.0,
) -> np.ndarray:
    """Generate continuous multi-peak Gaussian target heatmap on output grid."""
    h, w = shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    if len(coords) == 0:
        return heatmap

    y_grid, x_grid = np.ogrid[:h, :w]
    for cx, cy in coords:
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        g = np.exp(-dist_sq / (2.0 * sigma * sigma))
        heatmap = np.maximum(heatmap, g)
    return heatmap


def decode_tip_peaks(
    heatmap: np.ndarray,
    threshold: float = 0.30,
    min_distance: int = 10,
    max_peaks: Optional[int] = 10,
) -> List[Tuple[float, float, float]]:
    """Extract local peak coordinates and confidences from a 2D heatmap.
    
    Canonical Reference Implementation:
    - 3x3 local maximum filter via morphological dilation in float32.
    - Candidate thresholding at >= threshold.
    - Greedy Euclidean distance suppression (radius min_distance in model pixels).
    - Top-k ranking by confidence (if max_peaks is None, returns all uncapped survivors).
    
    Returns:
        List of (x, y, confidence) tuples sorted by confidence descending.
    """
    import cv2
    h, w = heatmap.shape
    hm_f32 = heatmap.astype(np.float32)
    if float(hm_f32.max()) < threshold:
        return []

    # Local maximum filter via 3x3 dilation in float32
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(hm_f32, kernel)
    local_maxima = (hm_f32 == dilated) & (hm_f32 >= threshold)

    peak_ys, peak_xs = np.where(local_maxima)
    candidates = []
    for x, y in zip(peak_xs, peak_ys):
        conf = float(hm_f32[y, x])
        candidates.append((float(x), float(y), conf))

    # Sort descending by peak confidence
    candidates.sort(key=lambda c: c[2], reverse=True)

    # Greedy Euclidean distance non-maximum suppression
    kept = []
    for c in candidates:
        if max_peaks is not None and len(kept) >= max_peaks:
            break
        too_close = False
        for k in kept:
            dist = np.hypot(c[0] - k[0], c[1] - k[1])
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            kept.append(c)

    return kept
