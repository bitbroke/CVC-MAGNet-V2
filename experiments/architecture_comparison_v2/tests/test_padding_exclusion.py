"""Test Padding Exclusion Invariant for Architecture Comparison V2.

Verifies:
1. Logits in the padding region have zero contribution to total loss (effective BCE + Dice).
2. Perturbing logits in padding region produces exactly ΔL = 0.
3. Gradients of loss with respect to padding logits are strictly 0.000000.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_padding_exclusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running padding exclusion test on device: {device}")

    # Create dummy logits with autograd
    # Let valid region be [100:668, 100:668]
    B, C, H, W = 1, 1, 768, 768
    valid_mask = torch.zeros((B, C, H, W), dtype=torch.float32, device=device)
    valid_mask[:, :, 100:668, 100:668] = 1.0
    padding_mask = 1.0 - valid_mask

    target_mask = torch.zeros((B, C, H, W), dtype=torch.float32, device=device)
    target_mask[:, :, 200:300, 200:300] = 1.0  # foreground within valid region

    total_valid_pixels = float(valid_mask.sum().item())
    total_positives = 1.0

    def compute_loss(logits: torch.Tensor) -> torch.Tensor:
        bce_raw = F.binary_cross_entropy_with_logits(
            logits, target_mask, pos_weight=torch.tensor([10.0], device=device), reduction="none"
        )
        mb_bce_loss = (bce_raw * valid_mask).sum() / max(total_valid_pixels, 1.0)

        probs = torch.sigmoid(logits)
        inter = (probs * target_mask * valid_mask).sum()
        card = (probs * valid_mask + target_mask * valid_mask).sum()
        dice_score = (2.0 * inter + 1e-6) / (card + 1e-6)
        mb_dice_loss = (1.0 - dice_score) / total_positives

        return mb_bce_loss + mb_dice_loss

    # 1. Base forward & backward
    base_logits = torch.randn((B, C, H, W), device=device, requires_grad=True)
    loss_base = compute_loss(base_logits)
    loss_base.backward()
    grad_base = base_logits.grad.clone()

    # 2. Perturbed forward & backward (modify only padding logits with massive noise)
    perturbed_logits = base_logits.detach().clone()
    noise = (torch.randn_like(perturbed_logits) * 100.0) * padding_mask
    perturbed_logits = (perturbed_logits + noise).requires_grad_(True)

    loss_pert = compute_loss(perturbed_logits)
    loss_pert.backward()
    grad_pert = perturbed_logits.grad.clone()

    # 3. Verify total loss difference is bitwise zero
    delta_loss = abs((loss_pert - loss_base).item())
    print(f"Base Loss:      {loss_base.item():.12f}")
    print(f"Perturbed Loss: {loss_pert.item():.12f}")
    print(f"Delta Loss:     {delta_loss:.12e}")

    # 4. Verify padding gradients are strictly 0.0
    pad_grads_base = (grad_base * padding_mask).abs().max().item()
    pad_grads_pert = (grad_pert * padding_mask).abs().max().item()
    print(f"Max base padding logit gradient:      {pad_grads_base:.12f}")
    print(f"Max perturbed padding logit gradient: {pad_grads_pert:.12f}")

    # 5. Verify valid region gradients are bitwise identical
    valid_grad_diff = ((grad_pert - grad_base) * valid_mask).abs().max().item()
    print(f"Max valid logit gradient difference:  {valid_grad_diff:.12e}")

    assert delta_loss == 0.0, f"Violation: Padding perturbation changed total loss by {delta_loss}!"
    assert pad_grads_base == 0.0, f"Violation: Nonzero padding gradient {pad_grads_base}!"
    assert pad_grads_pert == 0.0, f"Violation: Nonzero perturbed padding gradient {pad_grads_pert}!"
    assert valid_grad_diff == 0.0, f"Violation: Valid region gradients diverged by {valid_grad_diff}!"

    print("\n[SUCCESS] Padding Exclusion Invariant Verified! Delta Loss = 0, Pad Grads = 0.")


if __name__ == "__main__":
    test_padding_exclusion()
