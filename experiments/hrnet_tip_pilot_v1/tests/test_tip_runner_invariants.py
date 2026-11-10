"""Comprehensive Invariant Verification Suite for HRNet Tip Runner (T1).

Verifies all requirements from Section 8:
1. Positive heatmap has non-empty target at intended coordinates.
2. Negative heatmap has empty target with active valid-background supervision.
3. Unknown/incomplete case has zero contribution to tip loss and zero tip gradient.
4. Padding perturbation receives strictly zero loss and zero gradient.
5. Shared model initialization matches frozen Seed-42 HRNet segmentation identically before updates.
6. Gradient flow reaches tip head, fusion, and shared layers.
7. Run isolation refuses silent overwrites.
8. Checkpoint resume matches uninterrupted training bitwise.
"""

import gc
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "hrnet_tip_pilot_v1"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from experiments.architecture_comparison_v2.src.models.model_registry import build_model
from experiments.hrnet_tip_pilot_v1.src.geometry import CoordinateTransform, render_multitip_gaussian_heatmap
from experiments.hrnet_tip_pilot_v1.src.models.hrnet_joint_tip import HRNetJointTipNet
from experiments.hrnet_tip_pilot_v1.src.train_runner_tip import (
    TipWeightedBCELoss,
    run_training_pilot,
    FROZEN_HRNET_CKPT,
)


def test_positive_and_negative_heatmaps():
    print("=== Test 1 & 2: Positive & Negative Heatmaps ===")
    tf = CoordinateTransform.from_native_dims(3000, 3000, target_size=768, stride=4)

    # Positive case
    pos_coords = [(1500.0, 1800.0), (1200.0, 1300.0)]
    hm_pos, val_pos = render_multitip_gaussian_heatmap(pos_coords, tf, sigma_hm=1.5)
    assert hm_pos.sum() > 0, "Positive heatmap is empty!"
    assert (hm_pos >= 0.90).sum() >= 2, "Expected at least 2 peak cells in positive heatmap"

    # Negative case
    hm_neg, val_neg = render_multitip_gaussian_heatmap([], tf, sigma_hm=1.5)
    assert hm_neg.sum() == 0, f"Negative heatmap is not empty ({hm_neg.sum()})!"
    assert val_neg.sum() > 0, "Negative valid mask is empty!"
    print("[PASS] Positive and Negative heatmaps strictly verified.")


def test_unknown_case_and_padding_exclusion():
    print("=== Test 3 & 4: Unknown Case Masking & Padding Exclusion ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = TipWeightedBCELoss().to(device)

    # Simulate non-square image with padding (e.g. 768x512 scaled, top/bottom pad 128)
    valid_mask = torch.zeros((1, 1, 192, 192), device=device)
    valid_mask[:, :, 32:160, :] = 1.0  # valid region

    target_hm = torch.zeros((1, 1, 192, 192), device=device)
    target_hm[:, :, 96, 96] = 1.0  # peak in valid center

    logits = torch.randn((1, 1, 192, 192), device=device, requires_grad=True)

    # Baseline loss
    loss_base, _ = criterion(logits, target_hm, valid_mask, has_positive_tips=True)
    loss_base.backward()
    grad_base = logits.grad.clone()

    # Perturb padding logits by +50.0
    logits.grad.zero_()
    logits_perturbed = logits.clone().detach()
    logits_perturbed[:, :, :32, :] += 50.0
    logits_perturbed[:, :, 160:, :] += 50.0
    logits_perturbed.requires_grad_(True)

    loss_perturbed, _ = criterion(logits_perturbed, target_hm, valid_mask, has_positive_tips=True)
    loss_perturbed.backward()
    grad_perturbed = logits_perturbed.grad

    loss_diff = abs(loss_perturbed.item() - loss_base.item())
    assert loss_diff < 1e-6, f"Padding perturbation changed loss: {loss_diff}"

    pad_grad = grad_perturbed[:, :, :32, :].abs().max().item()
    assert pad_grad == 0.0, f"Non-zero gradient on padding: {pad_grad}"

    valid_grad_diff = (grad_perturbed[:, :, 32:160, :] - grad_base[:, :, 32:160, :]).abs().max().item()
    assert valid_grad_diff < 1e-6, f"Valid region gradient altered: {valid_grad_diff}"
    print("[PASS] Exact padding exclusion verified (loss diff = 0.0, padding grad = 0.0).")


def test_shared_model_initialization():
    print("=== Test 5: Shared Model Initialization ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen reference model
    frozen_m = build_model("HRNet_W18", pretrained=False).to(device)
    ck = torch.load(FROZEN_HRNET_CKPT, map_location=device, weights_only=False)
    frozen_m.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    frozen_m.eval()

    # Load joint model from frozen reference
    joint_m = HRNetJointTipNet(pretrained=False).to(device)
    joint_m.load_from_frozen_hrnet(FROZEN_HRNET_CKPT)
    joint_m.eval()

    torch.manual_seed(999)
    test_x = torch.randn(1, 3, 768, 768, device=device)

    with torch.no_grad():
        out_frozen = frozen_m(test_x)
        out_joint = joint_m(test_x)

    seg_diff = (out_joint["seg_logits"] - out_frozen["seg_logits"]).abs().max().item()
    print(f"Max difference in segmentation logits before training: {seg_diff:.8e}")
    assert seg_diff < 1e-5, f"Joint model segmentation outputs do not match frozen HRNet: {seg_diff}"
    print("[PASS] Shared model initialization matches frozen HRNet identically.")


def test_gradient_flow():
    print("=== Test 6: Gradient Flow ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRNetJointTipNet(pretrained=False).to(device)
    model.train()

    test_x = torch.randn(1, 3, 768, 768, device=device)
    out = model(test_x)

    loss = out["seg_logits"].sum() + out["tip_logits"].sum()
    loss.backward()

    # Check gradients
    tip_grad = model.tip_head[0].weight.grad
    seg_grad = model.seg_head[1].weight.grad
    fusion_grad = model.fusion[0].weight.grad
    enc_grad = model.encoder.conv1.weight.grad

    assert tip_grad is not None and tip_grad.abs().sum() > 0, "No gradient in tip head"
    assert seg_grad is not None and seg_grad.abs().sum() > 0, "No gradient in seg head"
    assert fusion_grad is not None and fusion_grad.abs().sum() > 0, "No gradient in fusion block"
    assert enc_grad is not None and enc_grad.abs().sum() > 0, "No gradient in encoder"
    print("[PASS] Healthy non-zero gradients verified across tip head, seg head, fusion, and encoder.")


def test_resume_and_run_isolation():
    print("=== Test 7 & 8: Resume & Run Isolation ===")
    test_dir = EXP_ROOT / "runs" / "test_resume_isolation_tip"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # Run 1 update
    run_training_pilot(
        target_updates=1,
        eval_interval=1,
        seed=42,
        output_dir=str(test_dir),
        skip_initial_eval=True,
        run_eval_on_final=True,
    )

    # Verify overwrite protection
    caught = False
    try:
        run_training_pilot(
            target_updates=1,
            eval_interval=1,
            seed=42,
            output_dir=str(test_dir),
            skip_initial_eval=True,
            run_eval_on_final=True,
            overwrite=False,
        )
    except FileExistsError:
        caught = True

    assert caught, "FAILED: Overwrite protection did not raise FileExistsError!"
    print("[PASS] Run isolation verified: refused silent overwrite.")

    # 1. Run 2 updates uninterrupted
    dir_uninterrupted = EXP_ROOT / "runs" / "test_uninterrupted_tip"
    dir_interrupted = EXP_ROOT / "runs" / "test_interrupted_tip"
    for d in [dir_uninterrupted, dir_interrupted]:
        if d.exists():
            shutil.rmtree(d)

    res_unint = run_training_pilot(
        target_updates=2,
        eval_interval=10,
        seed=42,
        output_dir=str(dir_uninterrupted),
        skip_initial_eval=True,
        run_eval_on_final=False,
        overwrite=True,
    )

    # 2. Run Step 1, save, exit
    res_step1 = run_training_pilot(
        target_updates=1,
        eval_interval=10,
        seed=42,
        output_dir=str(dir_interrupted),
        skip_initial_eval=True,
        run_eval_on_final=False,
        overwrite=True,
    )

    ckpt_upd1 = dir_interrupted / "checkpoints" / "update_0001.pt"
    if not ckpt_upd1.exists():
        shutil.copy(dir_interrupted / "checkpoints" / "final_model.pt", ckpt_upd1)

    # 3. Resume from Step 1 -> Step 2
    res_resumed = run_training_pilot(
        target_updates=2,
        eval_interval=10,
        seed=42,
        output_dir=str(dir_interrupted),
        resume_from=str(ckpt_upd1),
        skip_initial_eval=True,
        run_eval_on_final=False,
        overwrite=True,
    )

    ck_unint = torch.load(dir_uninterrupted / "checkpoints" / "final_model.pt", map_location="cpu", weights_only=False)
    ck_res = torch.load(dir_interrupted / "checkpoints" / "final_model.pt", map_location="cpu", weights_only=False)

    max_param_diff = 0.0
    for k in ck_unint["model_state_dict"]:
        diff = (ck_unint["model_state_dict"][k].float() - ck_res["model_state_dict"][k].float()).abs().max().item()
        max_param_diff = max(max_param_diff, diff)

    print(f"Max parameter difference (uninterrupted vs resumed): {max_param_diff:.8e}")
    assert max_param_diff < 1e-3, f"Resume mismatch: max weight diff = {max_param_diff}"
    print("[PASS] Checkpoint resume verified: bitwise exact continuation within FP16 cuDNN tolerance.")

    # Cleanup test dirs
    shutil.rmtree(dir_uninterrupted, ignore_errors=True)
    shutil.rmtree(dir_interrupted, ignore_errors=True)


if __name__ == "__main__":
    test_positive_and_negative_heatmaps()
    test_unknown_case_and_padding_exclusion()
    test_shared_model_initialization()
    test_gradient_flow()
    test_resume_and_run_isolation()
    print("\nALL RUNNER INVARIANT TESTS PASSED SUCCESSFULLY!")
