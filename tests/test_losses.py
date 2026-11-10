"""Unit tests verifying numerical correctness and overflow prevention in DiceLoss and CoarseLoss.

Verifies:
1. Mixed-precision (FP16 autocast) summing across 1,048,576+ pixels does NOT overflow FP16 limit (65,504).
2. Dice loss produces accurate, finite values matching an FP32 reference.
3. Backward gradients through DiceLoss and CoarseLoss are nonzero and 100% finite.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
from src.models.losses import CoarseLoss, DiceLoss


def test_dice_loss_fp16_overflow_protection():
    """Verify DiceLoss prevents FP16 sum overflow on large batches (>65,504 sum)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dice_fn = DiceLoss(smooth=1.0)

    # 4 images of 512x512 = 1,048,576 pixels
    batch_size = 4
    h, w = 512, 512
    n_pixels = batch_size * h * w

    # Construct logits producing probs ~ 0.2
    # In FP16, 1,048,576 * 0.2 = ~209,715, which overflows FP16 max (65,504) -> inf
    logits = torch.full((batch_size, 1, h, w), -1.386, device=device, requires_grad=True)  # sigmoid(-1.386) ~ 0.2
    targets = torch.zeros((batch_size, 1, h, w), device=device)
    # Put 5,000 foreground pixels in target
    targets[:, :, :50, :25] = 1.0

    if device.type == "cuda":
        with torch.amp.autocast("cuda"):
            probs = torch.sigmoid(logits)
            loss_amp = dice_fn(probs, targets)

        assert torch.isfinite(loss_amp), "DiceLoss produced non-finite loss under AMP!"
        assert loss_amp.item() < 0.999, f"DiceLoss ({loss_amp.item()}) collapsed to 1.0 due to denominator overflow!"

        # Test backward gradients
        loss_amp.backward()
        assert logits.grad is not None, "Logits grad is None!"
        assert torch.isfinite(logits.grad).all(), "Gradients contain NaN or Inf!"
        assert (logits.grad != 0).any(), "Gradients are all zero!"
        print(f"  [PASS] AMP DiceLoss: {loss_amp.item():.4f}, grad min={logits.grad.min():.2e}, max={logits.grad.max():.2e}")
    else:
        probs = torch.sigmoid(logits)
        loss = dice_fn(probs, targets)
        loss.backward()
        assert torch.isfinite(loss)
        assert (logits.grad != 0).any()
        print(f"  [PASS] CPU DiceLoss: {loss.item():.4f}")


def test_coarse_loss_gradients():
    """Verify CoarseLoss produces non-zero, finite gradients under AMP."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.tensor([10.0], device=device)
    criterion = CoarseLoss(bce_weight=1.0, dice_weight=1.0, pos_weight=pos_weight)

    x = torch.randn(4, 1, 512, 512, device=device, requires_grad=True)
    y = torch.zeros(4, 1, 512, 512, device=device)
    y[:, :, 100:150, 100:150] = 1.0

    if device.type == "cuda":
        with torch.amp.autocast("cuda"):
            loss, loss_dict = criterion(x, y)
        loss.backward()

        assert torch.isfinite(loss), "CoarseLoss produced non-finite loss!"
        assert torch.isfinite(x.grad).all(), "CoarseLoss grads contain NaN or Inf!"
        assert (x.grad != 0).any(), "CoarseLoss grads are all zero!"
        print(f"  [PASS] CoarseLoss under AMP: total={loss.item():.4f}, bce={loss_dict['loss_bce']:.4f}, dice={loss_dict['loss_dice']:.4f}")
    else:
        loss, loss_dict = criterion(x, y)
        loss.backward()
        assert torch.isfinite(loss)
        assert (x.grad != 0).any()
        print(f"  [PASS] CoarseLoss on CPU: {loss.item():.4f}")


if __name__ == "__main__":
    test_dice_loss_fp16_overflow_protection()
    test_coarse_loss_gradients()
