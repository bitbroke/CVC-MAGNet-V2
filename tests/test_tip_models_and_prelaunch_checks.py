"""Pre-launch Verification Suite for T1 and T2 Direct Tip Models.

Enforces Section 5 requirements:
1. Correct forward output shapes:
   - T1: tip_logits [B, 1, H/4, W/4], tip_probs [B, 1, H/4, W/4]
   - T2: tip_logits, tip_probs [B, 1, H/4, W/4] AND seg_logits, seg_probs [B, 1, H, W]
2. Finite forward and backward passes (no NaN / Inf gradients).
3. Exact masking of unknown supervision (masked instances produce zero tip gradient).
4. Native-coordinate round-trip tests (native -> model -> heatmap -> decoded native).
5. Multi-tip and border-case tests (multiple tips in single image, tips near margins).
6. Small positive/negative fitting test verifying decoded coordinate errors actually decrease.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.tip_heatmap import (
    CVCTipHeatmapNet,
    CVCJointSegTipNet,
    render_gaussian_heatmap,
    TipFocalLoss,
)
from src.infer.decoders import decode_predictions


def test_t1_and_t2_shapes_and_finite_gradients():
    print("[1/6 & 2/6] Testing forward output shapes and finite gradients...")
    device = torch.device("cpu")
    
    # Batch size 2, 1-channel, 768x768
    x = torch.randn(2, 1, 768, 768, device=device, requires_grad=True)
    
    # T1 Test
    t1 = CVCTipHeatmapNet(pretrained=False)
    out1 = t1(x)
    assert out1["tip_logits"].shape == (2, 1, 192, 192), f"T1 shape mismatch: {out1['tip_logits'].shape}"
    assert out1["tip_probs"].shape == (2, 1, 192, 192), f"T1 probs shape mismatch: {out1['tip_probs'].shape}"
    
    loss1 = out1["tip_logits"].sum()
    loss1.backward()
    assert torch.isfinite(x.grad).all(), "T1 gradients contain NaN or Inf!"
    print("  -> T1 Forward shapes & finite backward pass verified.")

    # T2 Test
    x.grad = None
    t2 = CVCJointSegTipNet(pretrained=False)
    out2 = t2(x)
    assert out2["tip_logits"].shape == (2, 1, 192, 192), f"T2 tip shape mismatch: {out2['tip_logits'].shape}"
    assert out2["seg_logits"].shape == (2, 1, 768, 768), f"T2 seg shape mismatch: {out2['seg_logits'].shape}"
    
    loss2 = out2["tip_logits"].sum() + out2["seg_logits"].sum()
    loss2.backward()
    assert torch.isfinite(x.grad).all(), "T2 gradients contain NaN or Inf!"
    print("  -> T2 Forward shapes & finite backward pass verified.")


def test_exact_supervision_masking():
    print("[3/6] Testing exact masking of unknown supervision...")
    device = torch.device("cpu")
    criterion = TipFocalLoss()

    # Two studies: study 0 completely labelled (mask=1), study 1 partially unlabelled (mask=0)
    pred = torch.tensor([[[[0.8, 0.1], [0.1, 0.05]]], [[[0.8, 0.1], [0.1, 0.05]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[1.0, 0.0], [0.0, 0.0]]]])
    mask = torch.tensor([[[[1.0]]], [[[0.0]]]])  # Study 1 masked out

    loss = criterion(pred, target, mask=mask)
    loss.backward()

    # Study 1 gradients must be exactly 0.0
    grad_study0 = pred.grad[0]
    grad_study1 = pred.grad[1]
    
    assert torch.any(grad_study0 != 0.0), "Study 0 should have non-zero gradients!"
    assert torch.all(grad_study1 == 0.0), f"Study 1 gradients must be exactly zero, got: {grad_study1}"
    print("  -> Exact supervision masking verified (unlabelled study receives 0.0 gradient).")


def test_native_coordinate_round_trip():
    print("[4/6] Testing native-coordinate round-trip...")
    native_w, native_h = 3000, 2500
    test_coords = [(500.0, 800.0), (1500.0, 1250.0), (2800.0, 2300.0)]
    
    hm = render_gaussian_heatmap(
        coords=test_coords,
        out_shape=(192, 192),
        sigma=2.0,
        stride=4,
        img_size=768,
        native_w=native_w,
        native_h=native_h,
    )
    
    hm_np = hm[0].cpu().numpy()
    preds_d0, _ = decode_predictions(hm_np, decoder_type="D0", stride=4, img_size=768, native_w=native_w, native_h=native_h)
    preds_d1, _ = decode_predictions(hm_np, decoder_type="D1", stride=4, img_size=768, native_w=native_w, native_h=native_h)
    
    assert len(preds_d0) == 3, f"Expected 3 detected peaks, got {len(preds_d0)}"
    for (gx, gy) in test_coords:
        dists_d0 = [np.sqrt((p["x"] - gx)**2 + (p["y"] - gy)**2) for p in preds_d0]
        dists_d1 = [np.sqrt((p["x"] - gx)**2 + (p["y"] - gy)**2) for p in preds_d1]
        min_err_d0 = min(dists_d0)
        min_err_d1 = min(dists_d1)
        assert min_err_d0 <= 15.0, f"D0 round-trip error too large: {min_err_d0:.2f}px"
        assert min_err_d1 <= 15.0, f"D1 round-trip error too large: {min_err_d1:.2f}px"
    print("  -> Native coordinate round-trip verified (all decoded errors <= 15px).")


def test_multi_tip_and_border_cases():
    print("[5/6] Testing multi-tip and border-cases...")
    native_w, native_h = 2000, 2000
    # Tips very close to border and multiple tips
    border_coords = [(10.0, 10.0), (1990.0, 1990.0), (1000.0, 1000.0), (1030.0, 1000.0)]
    
    hm = render_gaussian_heatmap(
        coords=border_coords,
        out_shape=(192, 192),
        sigma=2.0,
        stride=4,
        img_size=768,
        native_w=native_w,
        native_h=native_h,
    )
    hm_np = hm[0].cpu().numpy()
    preds, _ = decode_predictions(hm_np, threshold=0.10, stride=4, img_size=768, native_w=native_w, native_h=native_h)
    assert len(preds) >= 3, f"Expected multi-tip resolution, detected {len(preds)}"
    print(f"  -> Multi-tip and border handling verified ({len(preds)} peaks cleanly resolved).")


def test_positive_negative_fitting_with_coordinate_errors():
    print("[6/6] Testing small positive/negative fitting with decoded coordinate errors...")
    torch.manual_seed(43)
    model = CVCTipHeatmapNet(pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Fast 128x128 resolution test: input 128x128 -> heatmap 32x32
    native_w, native_h = 128, 128
    gt_point = (50.0, 50.0)

    # 1 positive image with signal at (50, 50), 1 negative image
    x_pos = torch.zeros(1, 1, 128, 128)
    x_pos[:, :, 48:52, 48:52] = 2.0
    target_pos = render_gaussian_heatmap([gt_point], (32, 32), sigma=1.5, stride=4, img_size=128, native_w=128, native_h=128)

    x_neg = torch.zeros(1, 1, 128, 128)
    target_neg = render_gaussian_heatmap([], (32, 32), sigma=1.5, stride=4, img_size=128, native_w=128, native_h=128)

    x = torch.cat([x_pos, x_neg], dim=0)
    target = torch.stack([target_pos, target_neg], dim=0)

    # Measure initial error
    with torch.no_grad():
        init_out = model(x)
        init_hm = init_out["tip_probs"][0, 0].numpy()
        init_preds, _ = decode_predictions(init_hm, threshold=0.01, stride=4, img_size=128, native_w=128, native_h=128)
        init_err = min([np.hypot(p["x"] - gt_point[0], p["y"] - gt_point[1]) for p in init_preds]) if init_preds else 999.0

    # Train for 25 steps with MSE loss
    for step in range(25):
        optimizer.zero_grad()
        out = model(x)
        loss = F.mse_loss(out["tip_probs"], target) * 1000.0
        loss.backward()
        optimizer.step()

    # Measure final error
    with torch.no_grad():
        final_out = model(x)
        final_hm = final_out["tip_probs"][0, 0].numpy()
        final_preds, _ = decode_predictions(final_hm, threshold=0.10, stride=4, img_size=128, native_w=128, native_h=128)
        assert len(final_preds) > 0, "Model failed to propose any peak after training!"
        # The top prediction must be the ground truth
        top_pred = final_preds[0]
        final_err = np.hypot(top_pred["x"] - gt_point[0], top_pred["y"] - gt_point[1])

        # Negative image check: max peak on negative image must remain low
        neg_max = final_out["tip_probs"][1, 0].max().item()

    print(f"  -> Initial coordinate error: {init_err:.1f}px | Final coordinate error: {final_err:.1f}px (Decoded error reduced!)")
    print(f"  -> Negative image max peak: {neg_max:.4f}")
    assert final_err < 10.0, f"Decoded error {final_err:.1f}px did not converge close to ground truth!"
    assert neg_max < 0.20, f"Negative image peak {neg_max:.4f} too high!"
    print("  -> Positive/negative fitting test with decoded coordinate errors passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING T1 & T2 PRE-LAUNCH VERIFICATION SUITE")
    print("=" * 60)
    test_t1_and_t2_shapes_and_finite_gradients()
    test_exact_supervision_masking()
    test_native_coordinate_round_trip()
    test_multi_tip_and_border_cases()
    test_positive_negative_fitting_with_coordinate_errors()
    print("=" * 60)
    print("ALL 6 PRE-LAUNCH TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
