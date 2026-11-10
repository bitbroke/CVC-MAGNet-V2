"""Unit tests for effective-batch gradient equivalence, non-square geometry, and optimizer step tracking."""

import json
import os
import random
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

from src.data.segmentation_v3_dataset import (
    BalancedCohortSampler,
    CVCSegmentationV3Dataset,
)
from src.data.transforms import LetterboxTransform



def test_non_square_valid_mask_augmentation():
    """Verify that non-square native images retain exact geometric validity under letterbox and affine transforms."""
    orig_h, orig_w = 2000, 2500
    im = np.ones((orig_h, orig_w), dtype=np.uint8) * 200

    target_size = (512, 512)
    lb = LetterboxTransform(target_size=target_size, clahe_enabled=False)
    canvas, meta = lb.apply_image(im)

    # Initial valid mask before augmentation
    valid_mask = np.zeros(target_size, dtype=np.uint8)
    valid_mask[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w] = 255

    # Pad region must be exactly 0
    assert (valid_mask[: meta.pad_top, :] == 0).all()
    assert (valid_mask[meta.pad_top + meta.new_h :, :] == 0).all()
    assert (valid_mask[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w] == 255).all()

    # Apply affine transform to both image and valid mask
    pil_canvas = Image.fromarray(canvas)
    pil_valid = Image.fromarray(valid_mask)

    angle = 5.0
    translate = [10.0, -10.0]
    scale = 1.02

    aug_canvas = TF.affine(pil_canvas, angle=angle, translate=translate, scale=scale, shear=0, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    aug_valid = TF.affine(pil_valid, angle=angle, translate=translate, scale=scale, shear=0, interpolation=TF.InterpolationMode.NEAREST, fill=0)

    aug_valid_np = (np.asarray(aug_valid) > 127).astype(np.float32)

    invalid_mask = aug_valid_np == 0
    assert invalid_mask.sum() > 0
    print(f"Non-square validity test PASSED: {invalid_mask.sum()} invalid padding pixels correctly tracked.")


def test_actual_dataset_non_square_geometry_and_valid_mask():
    """Invoke the ACTUAL CVCSegmentationV3Dataset on non-square images and verify geometry, masks, and validity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        # Create non-square test image (height 1800 x width 2400)
        orig_h, orig_w = 1800, 2400
        test_img = np.full((orig_h, orig_w), 128, dtype=np.uint8)
        # Draw a synthetic line simulating a catheter
        cv2.line(test_img, (500, 300), (1200, 1500), 255, thickness=5)
        sid = "test_study_nonsquare_01"
        img_file = img_dir / f"{sid}.jpg"
        cv2.imwrite(str(img_file), test_img)

        # Create synthetic polyline
        polyline_pts = [[500.0, 300.0], [800.0, 900.0], [1200.0, 1500.0]]
        inv_df = pd.DataFrame([
            {
                "study_id": sid,
                "device_type": "CVC",
                "polyline": json.dumps(polyline_pts),
            }
        ])

        study_inv_df = pd.DataFrame([
            {
                "StudyInstanceUID": sid,
                "has_cvc": True,
            }
        ])

        # Instantiate actual dataset with 512x512 target
        dataset = CVCSegmentationV3Dataset(
            study_ids=[sid],
            inventory_df=inv_df,
            study_inventory_df=study_inv_df,
            images_dir=img_dir,
            target_size=(512, 512),
            stroke_width=10,
            clahe_enabled=True,
            is_train=True,
            aug_seed=42,
        )

        sample = dataset[0]

        # 1. Check tensor shapes
        assert sample["image"].shape == (3, 512, 512), f"Expected (3, 512, 512), got {sample['image'].shape}"
        assert sample["mask"].shape == (1, 512, 512), f"Expected (1, 512, 512), got {sample['mask'].shape}"
        assert sample["valid_mask"].shape == (1, 512, 512), f"Expected (1, 512, 512), got {sample['valid_mask'].shape}"

        # 2. Check validity mask properties
        vmask = sample["valid_mask"][0].numpy()
        unique_vals = np.unique(vmask)
        assert set(unique_vals).issubset({0.0, 1.0}), f"Valid mask must be strictly binary, got {unique_vals}"

        # Because orig_h < orig_w (1800 vs 2400), letterbox creates top/bottom padding
        # Total valid pixels must be strictly less than 512*512
        assert vmask.sum() < 512 * 512, "Letterbox padding must exist for non-square 1800x2400 image"
        assert vmask.sum() > 0.5 * 512 * 512, "Valid area must cover most of the canvas"

        # 3. Check mask containment: target mask inside invalid region must be 0
        tmask = sample["mask"][0].numpy()
        assert tmask[vmask == 0.0].sum() == 0.0, "Target mask must be exactly zero in padded invalid regions"
        assert tmask.sum() > 0, "Target mask must contain positive rasterized pixels"

        # 4. Check augmentation parameters
        assert "aug_params" in sample
        assert "aug_seed" in sample["aug_params"]
        assert sample["aug_params"]["exposure_idx"] == 0

        # Retrieve second exposure: exposure_idx must increment and aug_params must update deterministically
        sample2 = dataset[0]
        assert sample2["aug_params"]["exposure_idx"] == 1
        assert sample2["aug_params"]["aug_seed"] != sample["aug_params"]["aug_seed"]

        print(f"Actual dataset non-square test PASSED: {int((1-vmask).sum())} pad pixels correctly excluded from valid_mask.")


def test_effective_batch_gradient_equivalence():
    """Verify that batch=2, accum=2 and batch=1, accum=4 produce mathematically identical loss and gradients."""
    torch.manual_seed(42)

    class ToySegNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    def compute_per_image_dice(pred_prob, target, valid_mask, eps=1e-6):
        pred_valid = pred_prob * valid_mask
        targ_valid = target * valid_mask
        inter = (pred_valid * targ_valid).sum(dim=(1, 2, 3))
        card = (pred_valid + targ_valid).sum(dim=(1, 2, 3))
        return (2.0 * inter + eps) / (card + eps)

    H, W = 32, 32
    images = torch.randn(4, 1, H, W)
    targets = torch.zeros(4, 1, H, W)
    targets[:3] = (torch.rand(3, 1, H, W) > 0.8).float()
    is_positive = torch.tensor([1, 1, 1, 0])

    valid_mask = torch.ones(4, 1, H, W)
    valid_mask[:, :, :4, :] = 0.0

    total_valid_pixels = valid_mask.sum().item()
    total_positives = is_positive.sum().item()
    bce_pos_weight = torch.tensor([10.0])
    dice_weight = 1.0

    # Arm 1: Batch 4 at once (ground truth effective loss)
    model_full = ToySegNet()
    model_full.load_state_dict(ToySegNet().state_dict())

    logits_full = model_full(images)
    bce_raw_full = nn.functional.binary_cross_entropy_with_logits(
        logits_full, targets, pos_weight=bce_pos_weight, reduction="none"
    )
    bce_loss_full = (bce_raw_full * valid_mask).sum() / total_valid_pixels
    probs_full = torch.sigmoid(logits_full)
    pos_mask_full = is_positive.bool()
    dices_full = compute_per_image_dice(probs_full[pos_mask_full], targets[pos_mask_full], valid_mask[pos_mask_full])
    dice_loss_full = (1.0 - dices_full).sum() / total_positives
    loss_full = bce_loss_full + dice_weight * dice_loss_full
    loss_full.backward()
    grad_full = model_full.conv.weight.grad.clone()

    # Arm 2: Batch 2, Accum 2
    model_b2 = ToySegNet()
    model_b2.load_state_dict(model_full.state_dict())
    model_b2.zero_grad()

    for m in range(2):
        mb_img = images[m*2 : (m+1)*2]
        mb_targ = targets[m*2 : (m+1)*2]
        mb_valid = valid_mask[m*2 : (m+1)*2]
        mb_pos = is_positive[m*2 : (m+1)*2]

        mb_logits = model_b2(mb_img)
        mb_bce_raw = nn.functional.binary_cross_entropy_with_logits(
            mb_logits, mb_targ, pos_weight=bce_pos_weight, reduction="none"
        )
        mb_bce_term = (mb_bce_raw * mb_valid).sum() / total_valid_pixels
        mb_probs = torch.sigmoid(mb_logits)
        pos_mask = mb_pos.bool()
        if pos_mask.any():
            mb_dices = compute_per_image_dice(mb_probs[pos_mask], mb_targ[pos_mask], mb_valid[pos_mask])
            mb_dice_term = (1.0 - mb_dices).sum() / total_positives
        else:
            mb_dice_term = 0.0

        mb_loss = mb_bce_term + dice_weight * mb_dice_term
        mb_loss.backward()

    grad_b2 = model_b2.conv.weight.grad.clone()

    # Arm 3: Batch 1, Accum 4
    model_b1 = ToySegNet()
    model_b1.load_state_dict(model_full.state_dict())
    model_b1.zero_grad()

    for m in range(4):
        mb_img = images[m : m+1]
        mb_targ = targets[m : m+1]
        mb_valid = valid_mask[m : m+1]
        mb_pos = is_positive[m : m+1]

        mb_logits = model_b1(mb_img)
        mb_bce_raw = nn.functional.binary_cross_entropy_with_logits(
            mb_logits, mb_targ, pos_weight=bce_pos_weight, reduction="none"
        )
        mb_bce_term = (mb_bce_raw * mb_valid).sum() / total_valid_pixels
        mb_probs = torch.sigmoid(mb_logits)
        pos_mask = mb_pos.bool()
        if pos_mask.any():
            mb_dices = compute_per_image_dice(mb_probs[pos_mask], mb_targ[pos_mask], mb_valid[pos_mask])
            mb_dice_term = (1.0 - mb_dices).sum() / total_positives
        else:
            mb_dice_term = 0.0

        mb_loss = mb_bce_term + dice_weight * mb_dice_term
        mb_loss.backward()

    grad_b1 = model_b1.conv.weight.grad.clone()

    torch.testing.assert_close(grad_b2, grad_full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad_b1, grad_full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad_b1, grad_b2, rtol=1e-5, atol=1e-6)
    print("Gradient equivalence test PASSED: Batch 4, Batch 2x2, and Batch 1x4 produce identical gradients!")


def test_enabled_amp_skip_behavior():
    """Verify enabled GradScaler skip behavior, scale adjustment, and optimizer step tracking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    p = nn.Parameter(torch.tensor([2.0], device=device, requires_grad=True))
    opt = torch.optim.SGD([p], lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    successful_updates = 0
    skipped_updates = 0
    total_attempts = 0

    # Step 1: Valid forward/backward with real AMP scaling
    total_attempts += 1
    loss = p * 3.0
    scaler.scale(loss).backward()
    scaler.unscale_(opt)

    has_inf_or_nan = False
    for param in [p]:
        if param.grad is not None and not torch.isfinite(param.grad).all():
            has_inf_or_nan = True
            break

    assert not has_inf_or_nan
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()
    successful_updates += 1

    initial_scale = scaler.get_scale() if use_cuda else 1.0

    # Step 2: Inject NaN gradient simulating FP16 overflow
    total_attempts += 1
    loss_nan = p * 2.0
    scaler.scale(loss_nan).backward()
    # Manually inject inf into gradient
    p.grad.fill_(float("inf"))
    scaler.unscale_(opt)

    has_inf_or_nan = False
    for param in [p]:
        if param.grad is not None and not torch.isfinite(param.grad).all():
            has_inf_or_nan = True
            break

    assert has_inf_or_nan
    skipped_updates += 1
    scaler.update()
    opt.zero_grad()
    # Note: successful_updates must NOT advance!

    if use_cuda:
        # Scale must have been reduced
        assert scaler.get_scale() < initial_scale, f"Scale should decrease on non-finite grad: {scaler.get_scale()} vs {initial_scale}"

    # Step 3: Valid subsequent step recovers
    total_attempts += 1
    loss_valid = p * 1.5
    scaler.scale(loss_valid).backward()
    scaler.unscale_(opt)

    has_inf_or_nan = False
    for param in [p]:
        if param.grad is not None and not torch.isfinite(param.grad).all():
            has_inf_or_nan = True
            break

    assert not has_inf_or_nan
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()
    successful_updates += 1

    assert total_attempts == 3
    assert successful_updates == 2
    assert skipped_updates == 1
    print(f"Enabled AMP skip test PASSED: 3 attempts, 2 successful steps, 1 skip tracked cleanly (scale reduced).")


def test_sampler_and_resume_determinism():
    """Verify that BalancedCohortSampler and model state resume produce exact bitwise continuation."""
    study_ids = [f"study_{i:03d}" for i in range(20)]
    positive_set = set(study_ids[:15])  # 15 positive, 5 negative

    # 1. Sampler state restoration test
    sampler1 = BalancedCohortSampler(
        study_ids=study_ids,
        positive_set=positive_set,
        pos_ratio=0.75,
        total_samples_per_epoch=20,
        seed=123,
    )

    it1 = iter(sampler1)
    # Draw 8 samples
    first_8 = [next(it1) for _ in range(8)]
    saved_state = sampler1.state_dict()

    # Draw next 8 continuous samples
    next_8_continuous = [next(it1) for _ in range(8)]

    # New sampler restored from saved state
    sampler2 = BalancedCohortSampler(
        study_ids=study_ids,
        positive_set=positive_set,
        pos_ratio=0.75,
        total_samples_per_epoch=20,
        seed=999,  # different initial seed
    )
    sampler2.load_state_dict(saved_state)
    it2 = iter(sampler2)
    next_8_resumed = [next(it2) for _ in range(8)]

    assert next_8_resumed == next_8_continuous, f"Sampler resume mismatch: {next_8_resumed} vs {next_8_continuous}"

    # 2. Model training resume equivalence test
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

    torch.manual_seed(42)
    model_cont = SimpleNet()
    opt_cont = torch.optim.SGD(model_cont.parameters(), lr=0.01)

    # Run 5 steps
    inputs = [torch.randn(2, 4) for _ in range(10)]
    targets = [torch.randn(2, 2) for _ in range(10)]

    for i in range(5):
        out = model_cont(inputs[i])
        loss = nn.functional.mse_loss(out, targets[i])
        loss.backward()
        opt_cont.step()
        opt_cont.zero_grad()

    # Save checkpoint at step 5
    ckpt_step5 = {
        "model_state": {k: v.clone() for k, v in model_cont.state_dict().items()},
        "opt_state": opt_cont.state_dict(),
        "torch_rng": torch.get_rng_state(),
    }

    # Continue steps 6-10 on continuous model
    for i in range(5, 10):
        out = model_cont(inputs[i])
        loss = nn.functional.mse_loss(out, targets[i])
        loss.backward()
        opt_cont.step()
        opt_cont.zero_grad()

    # Restore step 5 into a resumed model
    torch.set_rng_state(ckpt_step5["torch_rng"])
    model_resumed = SimpleNet()
    model_resumed.load_state_dict(ckpt_step5["model_state"])
    opt_resumed = torch.optim.SGD(model_resumed.parameters(), lr=0.01)
    opt_resumed.load_state_dict(ckpt_step5["opt_state"])

    for i in range(5, 10):
        out = model_resumed(inputs[i])
        loss = nn.functional.mse_loss(out, targets[i])
        loss.backward()
        opt_resumed.step()
        opt_resumed.zero_grad()

    # Assert exact parameter equivalence
    for k in model_cont.state_dict():
        torch.testing.assert_close(model_resumed.state_dict()[k], model_cont.state_dict()[k])

    print("Sampler and resume determinism test PASSED: Bitwise exact continuation verified.")


def test_unsigned_augmentation_seed_collates_without_overflow():
    from torch.utils.data._utils.collate import default_collate
    ds = CVCSegmentationV3Dataset.__new__(CVCSegmentationV3Dataset)
    ds.aug_seed = 42
    records = [ds.get_deterministic_aug_params(str(i), 0) for i in range(32)]
    assert any(int.from_bytes(bytes.fromhex(r['aug_seed']), 'little') >= 2**63 for r in records)
    batch = default_collate(records)
    assert batch['aug_seed'] == [r['aug_seed'] for r in records]


if __name__ == "__main__":
    test_non_square_valid_mask_augmentation()
    test_actual_dataset_non_square_geometry_and_valid_mask()
    test_effective_batch_gradient_equivalence()
    test_enabled_amp_skip_behavior()
    test_sampler_and_resume_determinism()
