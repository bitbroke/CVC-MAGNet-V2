"""CVC Tip Localization Models (T1 & T2) with Deep Spatial Context.

Implements Section 5 of the Experiment Specification:
- Shared Pretrained Backbone: ImageNet-pretrained EfficientNet-B0 with frozen BN running stats
- Spatial Lateral Decoder: Lateral skips at strides 32, 16, 8, 4 with GroupNorm (retaining deep context)
- Stride-4 Spatial Tip Head: Outputs multi-tip probability heatmap [B, 1, H/4, W/4] (e.g. 192x192 at 768px input)
- T1 (CVCTipHeatmapNet): Image -> Multi-tip heatmap
- T2 (CVCJointSegTipNet): Image -> Multi-tip heatmap + Full-resolution segmentation mask [B, 1, H, W]
- Exact supervision masking: loss applied only when supervision is complete (completely labelled positives or confirmed negatives)
- Gaussian heatmap rendering with native coordinate round-trip
"""

from typing import Dict, List, Optional, Tuple
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.segmentation_v3 import DecoderBlock


def render_gaussian_heatmap(
    coords: List[Tuple[float, float]],
    out_shape: Tuple[int, int] = (192, 192),
    sigma: float = 2.0,
    stride: int = 4,
    img_size: int = 768,
    native_w: int = 3000,
    native_h: int = 3000,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Render 2D Gaussian ground truth target heatmap [1, H_map, W_map].

    Args:
        coords: List of (native_x, native_y) tuples.
        out_shape: (H_map, W_map), typically (192, 192).
        sigma: Gaussian standard deviation in heatmap cell units.
        stride: Model stride (e.g. 4).
        img_size: Model input size (e.g. 768).
        native_w: Image width.
        native_h: Image height.
    """
    H_map, W_map = out_shape
    heatmap = torch.zeros((1, H_map, W_map), dtype=torch.float32, device=device)

    if not coords:
        return heatmap

    # Grid of coordinates
    y_range = torch.arange(H_map, dtype=torch.float32, device=device)
    x_range = torch.arange(W_map, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")

    radius = int(3 * sigma)

    for nx, ny in coords:
        # Transform native -> model -> heatmap coordinates
        mx = (nx / native_w) * img_size
        my = (ny / native_h) * img_size
        hx = (mx / stride) - 0.5
        hy = (my / stride) - 0.5

        if hx < 0 or hx >= W_map or hy < 0 or hy >= H_map:
            continue

        ix = int(round(hx))
        iy = int(round(hy))

        x_min = max(0, ix - radius)
        x_max = min(W_map, ix + radius + 1)
        y_min = max(0, iy - radius)
        y_max = min(H_map, iy + radius + 1)

        patch_x = grid_x[y_min:y_max, x_min:x_max]
        patch_y = grid_y[y_min:y_max, x_min:x_max]

        gaussian = torch.exp(-((patch_x - hx) ** 2 + (patch_y - hy) ** 2) / (2.0 * sigma ** 2))
        # Ensure peak maximum reaches 1.0 at center cell
        if gaussian.numel() > 0:
            gaussian = gaussian / torch.max(gaussian)
        heatmap[0, y_min:y_max, x_min:x_max] = torch.maximum(
            heatmap[0, y_min:y_max, x_min:x_max], gaussian
        )

    return heatmap


class TipFocalLoss(nn.Module):
    """Modified CenterNet Focal Loss for Gaussian tip heatmaps."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        pred_probs: torch.Tensor,
        target_heatmaps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pred_probs: [B, 1, H, W] predicted probabilities in (0, 1).
            target_heatmaps: [B, 1, H, W] ground truth Gaussian targets in [0, 1].
            mask: Optional [B, 1, 1, 1] or [B, 1, H, W] supervision mask.
        """
        pred = torch.clamp(pred_probs, self.eps, 1.0 - self.eps)
        target = target_heatmaps

        # Peak cells (>= 0.95) are positive targets; surrounding skirt (< 0.95) are penalty-reduced
        pos_mask = target.ge(0.95)
        neg_mask = target.lt(0.95)

        pos_loss = -torch.pow(1.0 - pred, self.alpha) * torch.log(pred) * pos_mask.float()
        neg_loss = (
            -torch.pow(1.0 - target, self.beta)
            * torch.pow(pred, self.alpha)
            * torch.log(1.0 - pred)
            * neg_mask.float()
        )

        loss = pos_loss + neg_loss

        if mask is not None:
            loss = loss * mask

        num_pos = pos_mask.float().sum()
        if num_pos > 0:
            return loss.sum() / num_pos
        else:
            return loss.sum() / max(target.numel() * 1e-4, 1.0)


class CVCTipHeatmapNet(nn.Module):
    """T1: Image -> Multi-Tip Heatmap (Stride 4 with deep context)."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_encoder_bn: bool = True,
        decoder_widths: Tuple[int, int, int, int] = (128, 64, 32, 16),
    ):
        super().__init__()
        self.freeze_encoder_bn = freeze_encoder_bn

        # Encoder: EfficientNet-B0 (strides 2, 4, 8, 16, 32)
        self.encoder = timm.create_model("efficientnet_b0", features_only=True, pretrained=pretrained)
        enc_channels = [16, 24, 40, 112, 320]

        w_d32, w_d16, w_d8, w_d4 = decoder_widths

        # Bottleneck at stride 32
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[4], w_d32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, w_d32),
            nn.ReLU(inplace=True),
        )

        # Lateral decoder stages down to stride 4 (deep context retained)
        self.block16 = DecoderBlock(w_d32, enc_channels[3], w_d16, num_groups=8)  # stride 32 -> 16
        self.block8 = DecoderBlock(w_d16, enc_channels[2], w_d8, num_groups=8)    # stride 16 -> 8
        self.block4 = DecoderBlock(w_d8, enc_channels[1], w_d4, num_groups=8)     # stride 8 -> 4

        # Stride-4 Spatial Tip Head
        self.tip_head = nn.Sequential(
            nn.Conv2d(w_d4, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        # CenterNet/RetinaNet focal loss prior initialization (p_init ≈ 0.018)
        nn.init.constant_(self.tip_head[-1].bias, -4.0)

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
        f1, f2, f3, f4 = feats[1], feats[2], feats[3], feats[4]

        b32 = self.bottleneck(f4)
        d16 = self.block16(b32, f3)
        d8 = self.block8(d16, f2)
        d4 = self.block4(d8, f1)

        tip_logits = self.tip_head(d4)
        tip_probs = torch.sigmoid(tip_logits)

        return {
            "tip_logits": tip_logits,
            "tip_probs": tip_probs,
        }


class CVCJointSegTipNet(nn.Module):
    """T2: Image -> Segmentation Mask (Stride 1) + Multi-Tip Heatmap (Stride 4)."""

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

        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[4], w_d32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, w_d32),
            nn.ReLU(inplace=True),
        )

        self.block16 = DecoderBlock(w_d32, enc_channels[3], w_d16, num_groups=8)
        self.block8 = DecoderBlock(w_d16, enc_channels[2], w_d8, num_groups=8)
        self.block4 = DecoderBlock(w_d8, enc_channels[1], w_d4, num_groups=8)

        # Stride-4 Spatial Tip Head
        self.tip_head = nn.Sequential(
            nn.Conv2d(w_d4, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        # CenterNet/RetinaNet focal loss prior initialization (p_init ≈ 0.018)
        nn.init.constant_(self.tip_head[-1].bias, -4.0)

        # Segmentation branch continuing down to native resolution: stride 4 -> 2 -> 1
        self.block2 = DecoderBlock(w_d4, enc_channels[0], w_d2, num_groups=8)
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

        b32 = self.bottleneck(f4)
        d16 = self.block16(b32, f3)
        d8 = self.block8(d16, f2)
        d4 = self.block4(d8, f1)

        tip_logits = self.tip_head(d4)
        tip_probs = torch.sigmoid(tip_logits)

        d2 = self.block2(d4, f0)
        seg_logits = self.block1(d2)
        seg_probs = torch.sigmoid(seg_logits)

        return {
            "tip_logits": tip_logits,
            "tip_probs": tip_probs,
            "seg_logits": seg_logits,
            "seg_probs": seg_probs,
        }
