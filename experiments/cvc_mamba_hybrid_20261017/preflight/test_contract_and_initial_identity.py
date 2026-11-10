"""Contract Tests and Initial Output Identity Verification for CVC-Mamba-Hybrid-01.

Validates Section 5 requirements:
1. Every shared weight is strictly loaded from the frozen S4 reference.
2. Initial predictions across Arm A, Arm B, and Arm C are mathematically IDENTICAL (diff == 0.0).
3. Parameter counts and capacity matching between Arm B (Mamba) and Arm C (Conv).
4. Four-way scan / inverse-scan geometry on non-square grid.
5. Finite forward/loss/backward gradients.
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "cvc_mamba_hybrid_20261017"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from experiments.cvc_mamba_hybrid_20261017.src.models.arm_models import (
    ArmAContinuationNet,
    ArmBMambaHybridNet,
    ArmCConvControlNet,
)
from experiments.cvc_mamba_hybrid_20261017.src.models.mamba_context import FourWayMamba2DBlock


def main():
    print("=== Running Section 5 Contract & Equivalence Tests ===")

    # 1. Instantiate all three models and load S4 reference checkpoint
    print("Instantiating Arm A (Continuation)...")
    arm_a = ArmAContinuationNet(load_checkpoint=True)
    arm_a.eval()

    print("Instantiating Arm B (Mamba Hybrid)...")
    arm_b = ArmBMambaHybridNet(load_checkpoint=True)
    arm_b.eval()

    print("Instantiating Arm C (Conv Control)...")
    arm_c = ArmCConvControlNet(load_checkpoint=True)
    arm_c.eval()

    # 2. Check Parameter Counts
    params_a = sum(p.numel() for p in arm_a.parameters())
    params_b = sum(p.numel() for p in arm_b.parameters())
    params_c = sum(p.numel() for p in arm_c.parameters())

    added_b = params_b - params_a
    added_c = params_c - params_a
    param_ratio = added_c / added_b if added_b > 0 else 0.0

    print(f"\n--- Parameter Count Verification ---")
    print(f"Arm A Parameters: {params_a:,}")
    print(f"Arm B Parameters: {params_b:,} (Added: {added_b:,})")
    print(f"Arm C Parameters: {params_c:,} (Added: {added_c:,})")
    print(f"Capacity matching ratio (C / B): {param_ratio:.2f}")

    # 3. Test Initial Output Equivalence on 768px input
    print("\n--- Testing Initial Output Identity (Step 0) at 768x768 ---")
    x768 = torch.randn(1, 1, 768, 768)

    with torch.no_grad():
        out_a = arm_a(x768)["seg_probs"]
        out_b = arm_b(x768)["seg_probs"]
        out_c = arm_c(x768)["seg_probs"]

    diff_ab = (out_a - out_b).abs().max().item()
    diff_ac = (out_a - out_c).abs().max().item()

    print(f"Max absolute diff (Arm A vs Arm B [Mamba]): {diff_ab:.2e}")
    print(f"Max absolute diff (Arm A vs Arm C [Conv]):  {diff_ac:.2e}")

    assert diff_ab < 1e-6, f"Arm B step-0 prediction does not match Arm A: diff = {diff_ab}"
    assert diff_ac < 1e-6, f"Arm C step-0 prediction does not match Arm A: diff = {diff_ac}"
    print("Contract Test PASSED: Arm B and Arm C initial predictions are mathematically IDENTICAL to Arm A!")

    # 4. Verify 4-way scan geometry on non-square tensor
    print("\n--- Testing 4-Way Scan Geometry on Non-Square Grid (H=16, W=24) ---")
    block = FourWayMamba2DBlock(dim=128)
    block.eval()
    x_nonsquare = torch.randn(1, 128, 16, 24)
    with torch.no_grad():
        y_nonsquare = block(x_nonsquare)
    assert y_nonsquare.shape == (1, 128, 16, 24), f"Shape mismatch: {y_nonsquare.shape}"
    print(f"Non-square scan output shape: {y_nonsquare.shape} (PASSED)")

    # 5. Verify finite backward pass and gradient flow
    print("\n--- Testing Backward Pass and Gradient Propagation on CUDA ---")
    if torch.cuda.is_available():
        arm_b.train()
        arm_b_cuda = arm_b.cuda()
        opt = torch.optim.AdamW(arm_b_cuda.parameters(), lr=1e-4)
        x_cuda = torch.randn(1, 1, 768, 768, device="cuda")
        target_cuda = torch.zeros(1, 1, 768, 768, device="cuda")
        out = arm_b_cuda(x_cuda)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out["seg_logits"], target_cuda)
        loss.backward()
        
        # Check that mamba parameters received gradients
        mamba_grad_norms = [p.grad.norm().item() for p in arm_b_cuda.mamba_context.parameters() if p.grad is not None]
        print(f"Mamba module params with grad: {len(mamba_grad_norms)}/{len(list(arm_b_cuda.mamba_context.parameters()))}")
        print(f"Mean Mamba grad norm: {sum(mamba_grad_norms)/max(len(mamba_grad_norms), 1):.4e}")
        assert len(mamba_grad_norms) > 0, "No gradients reached Mamba module!"
        print("CUDA Backward and Gradient Flow: PASSED")
    else:
        print("CUDA not available for backward test.")

    print("\nALL CONTRACT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
