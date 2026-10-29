"""Loss functions for coarse and fine catheter segmentation.

Implements the published MAG-Net loss formulation:
- Coarse Stage: BCE + Dice segmentation loss.
- Fine Stage: BCE + Dice segmentation loss + 0.5 * L1 Attention Loss against Gaussian-blurred mask.
- Safe empty-mask handling: Produces finite, non-NaN losses on negative cases.
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss with smooth term and safe empty-target handling."""

    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Strictly perform all reductions in float32 to prevent FP16 overflow (max 65,504)
        # on large 1,048,576+ pixel batches under mixed precision.
        probs = probs.float().contiguous().view(-1)
        targets = targets.float().contiguous().view(-1)

        intersection = (probs * targets).sum()
        denominator = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice


class CoarseLoss(nn.Module):
    """Composite loss for coarse catheter segmentation."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super(CoarseLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(torch.sigmoid(logits.float()), targets.float())
        total_loss = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        return total_loss, {
            "loss_total": float(total_loss.item()),
            "loss_bce": float(loss_bce.item()),
            "loss_dice": float(loss_dice.item()),
        }


class CoarseDualLoss(nn.Module):
    """Joint multi-task loss for coarse segmentation and distal tip heatmap localization.
    
    L_total = L_seg + tip_weight * L_tip
    where:
      L_seg = bce_weight * BCEWithLogits + dice_weight * DiceLoss (in float32)
      L_tip = BCEWithLogitsLoss(pos_weight=tip_pos_weight) on soft Gaussian heatmaps,
              masked for unknown/unreviewed tips (tip_mask=0 produces 0 gradient to tip head).
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 2.0,
        tip_weight: float = 1.0,
        tip_pos_weight: float = 500.0,
        shaft_weight: float = 1.0,
        shaft_tip_exclusion_thresh: float = 0.01,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super(CoarseDualLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tip_weight = tip_weight
        self.shaft_weight = shaft_weight
        self.shaft_tip_exclusion_thresh = shaft_tip_exclusion_thresh
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        
        # Heatmap BCE with positive weighting to balance ~50 peak pixels against ~262,000 background pixels
        if tip_pos_weight is not None:
            self.register_buffer("tip_pos_weight", torch.tensor([float(tip_pos_weight)]))
        else:
            self.tip_pos_weight = None

    def forward(
        self,
        seg_logits: torch.Tensor,
        tip_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        tip_targets: torch.Tensor,
        tip_mask: Optional[torch.Tensor] = None,
        tip_spatial_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_bce = self.bce(seg_logits, seg_targets)
        loss_dice = self.dice(torch.sigmoid(seg_logits.float()), seg_targets.float())
        loss_seg = self.bce_weight * loss_bce + self.dice_weight * loss_dice

        # Heatmap BCEWithLogits per pixel, unreduced per sample
        if self.tip_pos_weight is not None:
            pw = self.tip_pos_weight.to(tip_logits.device)
            per_pixel_tip = F.binary_cross_entropy_with_logits(
                tip_logits.float(), tip_targets.float(), pos_weight=pw, reduction="none"
            )
        else:
            per_pixel_tip = F.binary_cross_entropy_with_logits(
                tip_logits.float(), tip_targets.float(), reduction="none"
            )

        # Apply modest negative-only shaft penalty if shaft_weight > 1.0 (excluding tip neighborhoods)
        if self.shaft_weight > 1.0:
            tip_exclusion = (tip_targets > self.shaft_tip_exclusion_thresh)
            shaft_mask = ((seg_targets > 0.5) & (~tip_exclusion)).float()
            # Negative-only penalty: (w - 1) * shaft_mask * (1 - target) * softplus(logits)
            # Strictly downward gradient: d(extra)/dz = (w - 1) * shaft_mask * (1 - target) * sigmoid(z) >= 0
            extra_shaft_penalty = (
                (self.shaft_weight - 1.0)
                * shaft_mask
                * (1.0 - tip_targets.float())
                * F.softplus(tip_logits.float())
            )
            per_pixel_tip = per_pixel_tip + extra_shaft_penalty

        # Apply pixel-level spatial ignore mask (1=valid, 0=ignore/unreviewed catheter)
        if tip_spatial_mask is not None:
            mask_f = tip_spatial_mask.float()
            per_pixel_tip = per_pixel_tip * mask_f
            valid_pixels = mask_f.sum(dim=(1, 2, 3)).clamp(min=1.0)
            per_sample_tip = per_pixel_tip.sum(dim=(1, 2, 3)) / valid_pixels
        else:
            per_sample_tip = per_pixel_tip.mean(dim=(1, 2, 3))

        if tip_mask is not None:
            tip_mask_f = tip_mask.view(-1).float()
            valid_count = tip_mask_f.sum()
            if valid_count > 0:
                loss_tip = (per_sample_tip * tip_mask_f).sum() / valid_count
            else:
                loss_tip = 0.0 * tip_logits.sum()
        else:
            loss_tip = per_sample_tip.mean()

        total_loss = loss_seg + self.tip_weight * loss_tip
        return total_loss, {
            "loss_total": float(total_loss.item()),
            "loss_seg": float(loss_seg.item()),
            "loss_bce": float(loss_bce.item()),
            "loss_dice": float(loss_dice.item()),
            "loss_tip": float(loss_tip.item()),
        }


class FineLoss(nn.Module):
    """Composite multi-task loss for fine catheter refinement.
    
    Published formula:
    L_fine = L_seg(BCE + Dice) + 0.5 * L1(mask_attention, gaussian_blurred_mask)
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        attention_l1_weight: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super(FineLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.att_weight = attention_l1_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(
        self,
        seg_logits: torch.Tensor,
        mask_att: torch.Tensor,
        targets: torch.Tensor,
        gaussian_blurred_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_bce = self.bce(seg_logits, targets)
        loss_dice = self.dice(torch.sigmoid(seg_logits.float()), targets.float())
        loss_att = self.l1(mask_att.float(), gaussian_blurred_targets.float())

        total_loss = (
            self.bce_weight * loss_bce
            + self.dice_weight * loss_dice
            + self.att_weight * loss_att
        )
        return total_loss, {
            "loss_total": float(total_loss.item()),
            "loss_bce": float(loss_bce.item()),
            "loss_dice": float(loss_dice.item()),
            "loss_att_l1": float(loss_att.item()),
        }
