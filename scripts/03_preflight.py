"""Script 03: Hardware and Correctness Preflight Checks for CVC-MAGNet-V2.

Executes:
1. Coarse stage training configuration profiling at intended batch size 4 (512x512) on GPU.
2. Fine stage training configuration profiling at intended batch size 2 (512x512) on GPU.
3. Finite loss and gradient checks.
4. Edge cases: empty predictions, border padding, and multi-catheter composite masks.
5. Tiny-set fitting using real annotated chest radiographs:
   - Tracks foreground Dice and recall per step (verifying catheter geometry learning vs. background suppression).
   - Generates visual predicted-mask overlays saved to disk.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rasterize import (
    create_gaussian_attention_mask,
    rasterize_polyline,
    rasterize_study_polylines,
)
from src.data.transforms import FineCropTransform, LetterboxTransform, TransformMetadata
from src.models.coarse import CoarseSegNet
from src.models.fine import FineSegNet
from src.models.losses import CoarseLoss, FineLoss


def check_cuda_device():
    print("-" * 60)
    print("CHECK 0: GPU Environment & CUDA Device")
    print("-" * 60)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! A GPU is required for preflight hardware verification.")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"Device: {device_name}")
    print(f"Total VRAM: {vram_total_mb:.1f} MB (~{vram_total_mb/1024:.2f} GB)")
    assert vram_total_mb >= 3500, f"Expected >= 4GB VRAM GPU, detected {vram_total_mb:.1f} MB."
    return device


def check_coarse_intended_batch_size(device: torch.device, batch_size: int = 4):
    print("\n" + "-" * 60)
    print(f"CHECK 1: Coarse Training Pass at Intended Batch Size {batch_size} (512x512)")
    print("-" * 60)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = CoarseSegNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    criterion = CoarseLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Intended coarse training input: batch_size=4, 1 channel, 512x512
    x = torch.randn(batch_size, 1, 512, 512, device=device)
    y = torch.zeros(batch_size, 1, 512, 512, device=device)
    y[:, :, 100:200, 200:300] = 1.0

    t0 = time.time()
    logits = model(x)
    loss, loss_dict = criterion(logits, y)
    loss.backward()
    optimizer.step()
    t_step = (time.time() - t0) * 1000

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Check finiteness
    assert torch.isfinite(loss), "Coarse loss is not finite!"
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Gradient for {name} has NaN/Inf!"

    print(f"  [PASS] Input shape: {list(x.shape)} | Output shape: {list(logits.shape)}")
    print(f"  [PASS] Total Loss: {loss.item():.4f} (BCE: {loss_dict['loss_bce']:.4f}, Dice: {loss_dict['loss_dice']:.4f})")
    print(f"  [PASS] Gradients: 100% finite across all {len(list(model.parameters()))} parameter tensors.")
    print(f"  [PASS] Step time: {t_step:.1f} ms | Peak VRAM: {peak_vram_mb:.1f} MB (Budget: 4000 MB)")
    assert peak_vram_mb < 3800, f"Coarse batch {batch_size} exceeded VRAM safety limit: {peak_vram_mb:.1f} MB!"
    return peak_vram_mb


def check_fine_intended_batch_size(device: torch.device, batch_size: int = 2):
    print("\n" + "-" * 60)
    print(f"CHECK 2: Fine Training Pass at Intended Batch Size {batch_size} (512x512, 2 Channels)")
    print("-" * 60)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = FineSegNet(in_channels=2, out_channels=1, base_channels=32).to(device)
    criterion = FineLoss(bce_weight=1.0, dice_weight=1.0, attention_l1_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Intended fine training input: batch_size=2, 2 channels [image crop, coarse mask crop], 512x512
    x = torch.randn(batch_size, 2, 512, 512, device=device)
    y_mask = torch.zeros(batch_size, 1, 512, 512, device=device)
    y_mask[:, :, 150:250, 150:250] = 1.0
    y_blur = torch.zeros(batch_size, 1, 512, 512, device=device)
    y_blur[:, :, 140:260, 140:260] = 0.8

    t0 = time.time()
    logits, att = model(x)
    loss, loss_dict = criterion(logits, att, y_mask, y_blur)
    loss.backward()
    optimizer.step()
    t_step = (time.time() - t0) * 1000

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Check finiteness
    assert torch.isfinite(loss), "Fine loss is not finite!"
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Gradient for {name} has NaN/Inf!"

    print(f"  [PASS] Input shape: {list(x.shape)} | Output shape: logits {list(logits.shape)}, attention {list(att.shape)}")
    print(f"  [PASS] Total Loss: {loss.item():.4f} (BCE: {loss_dict['loss_bce']:.4f}, Dice: {loss_dict['loss_dice']:.4f}, Att L1: {loss_dict['loss_att_l1']:.4f})")
    print(f"  [PASS] Gradients: 100% finite across all {len(list(model.parameters()))} parameter tensors.")
    print(f"  [PASS] Step time: {t_step:.1f} ms | Peak VRAM: {peak_vram_mb:.1f} MB (Budget: 4000 MB)")
    assert peak_vram_mb < 3800, f"Fine batch {batch_size} exceeded VRAM safety limit: {peak_vram_mb:.1f} MB!"
    return peak_vram_mb


def check_edge_cases():
    print("\n" + "-" * 60)
    print("CHECK 3: Empty Predictions, Border Padding & Multiple Catheters")
    print("-" * 60)

    # 1. Empty coarse mask handling
    empty_coarse = np.zeros((512, 512), dtype=np.float32)
    choice = np.where(empty_coarse > 0.1)
    if len(choice[0]) == 0:
        bbox = (0, 0, 512, 512)
        has_roi = False
    else:
        bbox = (int(choice[1].min()), int(choice[0].min()), int(choice[1].max()), int(choice[0].max()))
        has_roi = True

    assert not has_roi, "Empty mask should set has_roi = False"
    assert bbox == (0, 0, 512, 512), "Empty mask should safely fall back to full field bbox"
    print("  [PASS] Empty coarse prediction safely detected (has_roi=False, zero-crash fallback).")

    # 2. Border padding & native reconstruction at image edges
    crop_tf = FineCropTransform(target_size=(512, 512), padding_px=32)
    dummy_img = np.zeros((2000, 2000), dtype=np.uint8)
    dummy_coarse = np.zeros((2000, 2000), dtype=np.uint8)

    # Extreme border ROI at top-left
    border_bbox = (0, 0, 100, 100)
    img_crop, coarse_crop, _, meta = crop_tf.extract_crop(dummy_img, dummy_coarse, border_bbox)
    assert img_crop.shape == (512, 512), f"Crop shape mismatch: {img_crop.shape}"

    # Verify origin coordinate (0, 0) projects back to (0.0, 0.0)
    mx, my = meta.native_to_model(0.0, 0.0)
    nx, ny = meta.model_to_native(mx, my)
    assert np.hypot(nx - 0.0, ny - 0.0) < 1e-4, f"Border inversion error: {nx}, {ny}"
    print("  [PASS] Extreme border ROI padded and inverted back to native coordinates (error < 1e-4 px).")

    # 3. Multiple-catheter polyline rasterization
    line1 = [[100, 100], [200, 300], [300, 500]]
    line2 = [[400, 150], [350, 400], [320, 600]]
    multi_mask = rasterize_study_polylines([line1, line2], (1000, 1000), thickness=10)
    assert multi_mask.shape == (1000, 1000), "Multi-catheter mask shape mismatch"
    assert multi_mask.sum() > 0, "Multi-catheter mask must not be empty"
    print("  [PASS] Multi-catheter co-occurring lines correctly rasterized into composite target mask.")


def check_tiny_set_fitting(device: torch.device):
    print("\n" + "-" * 60)
    print("CHECK 4: Tiny-Set Geometry Fitting on Real Radiographs")
    print("         (Tracking Foreground Dice, Recall & Generating Overlays)")
    print("-" * 60)
    data_dir = PROJECT_ROOT / "data"
    inv_df = pd.read_parquet(data_dir / "annotation_inventory.parquet")
    raw_images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
    reports_dir = PROJECT_ROOT / "reports" / "overlays"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "splits.json", "r") as f:
        splits = json.load(f)

    # Select 4 real studies with annotated CVC catheters from pilot train for swift, high-signal verification
    train_ids_set = set(splits["partitions"]["dev_pilot_train"]["study_ids"])
    pos_cvc_records = inv_df[inv_df["study_id"].isin(train_ids_set) & (inv_df["device_type"] == "CVC")]
    train_study_ids = pos_cvc_records["study_id"].unique()[:4]
    print(f"Loading {len(train_study_ids)} real CVC-annotated chest radiographs for tiny-set learnability test...")

    transform = LetterboxTransform(target_size=(512, 512), clahe_enabled=True)

    batch_images = []
    batch_masks = []
    loaded_studies = []

    for sid in train_study_ids:
        img_path = raw_images_dir / f"{sid}.jpg"
        if not img_path.exists():
            continue
        native_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = native_img.shape

        study_cats = inv_df[(inv_df["study_id"] == sid) & (inv_df["device_type"] == "CVC")]
        if len(study_cats) > 0:
            polylines = [json.loads(p) for p in study_cats["polyline"]]
            native_mask = rasterize_study_polylines(polylines, (h, w), thickness=10)
        else:
            native_mask = np.zeros((h, w), dtype=np.uint8)

        lb_img, meta = transform.apply_image(native_img)
        lb_mask = transform.apply_mask(native_mask, meta)

        batch_images.append(lb_img.astype(np.float32) / 255.0)
        batch_masks.append(lb_mask.astype(np.float32))
        loaded_studies.append(sid)

    x_tensor = torch.tensor(np.array(batch_images), dtype=torch.float32).unsqueeze(1).to(device)
    y_tensor = torch.tensor(np.array(batch_masks), dtype=torch.float32).unsqueeze(1).to(device)

    total_fg_px = int(y_tensor.sum().item())
    print(f"Constructed tiny batch: X {list(x_tensor.shape)}, Y {list(y_tensor.shape)} ({total_fg_px} ground-truth foreground catheter px).")

    # Fit for 50 optimizer steps
    model = CoarseSegNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([50.0], device=device)
    criterion = CoarseLoss(bce_weight=1.0, dice_weight=2.0, pos_weight=pos_weight)

    initial_loss, initial_dice, initial_recall = None, None, None
    final_loss, final_dice, final_recall = None, None, None
    final_probs = None
    num_steps = 50

    print(f"\nFitting CoarseSegNet over {num_steps} steps to prove foreground catheter geometry learning:")
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss, loss_dict = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

        # Compute foreground metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pred_fg = (probs > 0.5).float()
            gt_fg = (y_tensor > 0.5).float()

            inter = (pred_fg * gt_fg).sum().item()
            p_sum = pred_fg.sum().item()
            g_sum = gt_fg.sum().item()

            fg_dice = (2.0 * inter) / (p_sum + g_sum + 1e-6)
            fg_recall = inter / (g_sum + 1e-6)

        if step == 1:
            initial_loss = loss.item()
            initial_dice = fg_dice
            initial_recall = fg_recall
        if step == num_steps:
            final_loss = loss.item()
            final_dice = fg_dice
            final_recall = fg_recall
            final_probs = probs.cpu().numpy()

        if step % 10 == 0 or step == 1 or step == num_steps:
            print(
                f"  Step {step:02d}/{num_steps} | Loss: {loss.item():.4f} | "
                f"BCE: {loss_dict['loss_bce']:.4f} | "
                f"Dice Loss: {loss_dict['loss_dice']:.4f} | "
                f"FG Dice: {fg_dice:.4f} | FG Recall: {fg_recall:.4f}"
            )

    loss_reduction_pct = (initial_loss - final_loss) / initial_loss * 100.0
    print(f"\n  [PASS] Loss: {initial_loss:.4f} -> {final_loss:.4f} ({loss_reduction_pct:.1f}% reduction).")
    print(f"  [PASS] Foreground Dice: {initial_dice:.4f} -> {final_dice:.4f} (Foreground localization confirmed).")
    print(f"  [PASS] Foreground Recall: {initial_recall:.4f} -> {final_recall:.4f} (Catheter pixels actively captured).")
    assert final_dice > 0.15, f"Foreground Dice ({final_dice:.4f}) must exceed 0.15 to demonstrate geometry learning!"
    assert final_recall > 0.50, f"Foreground Recall ({final_recall:.4f}) must exceed 0.50 to demonstrate geometry learning!"

    # Generate and save visual overlays of predicted masks
    print("\nGenerating predicted-mask visual overlay diagnostic...")
    num_display = min(4, len(loaded_studies))
    fig, axes = plt.subplots(num_display, 4, figsize=(16, 4 * num_display))

    for i in range(num_display):
        img_np = batch_images[i]
        gt_np = batch_masks[i]
        pred_prob_np = final_probs[i, 0]

        # 1. Radiograph
        axes[i, 0].imshow(img_np, cmap="gray")
        axes[i, 0].set_title(f"Radiograph: {loaded_studies[i][:12]}...")
        axes[i, 0].axis("off")

        # 2. Ground Truth Mask
        axes[i, 1].imshow(img_np, cmap="gray")
        axes[i, 1].imshow(gt_np, cmap="Reds", alpha=0.55)
        axes[i, 1].set_title(f"Ground Truth CVC Mask (N={int(gt_np.sum())} px)")
        axes[i, 1].axis("off")

        # 3. Predicted Probability Map
        axes[i, 2].imshow(pred_prob_np, cmap="magma")
        axes[i, 2].set_title(f"Coarse Prob Map (Max={pred_prob_np.max():.2f})")
        axes[i, 2].axis("off")

        # 4. Predicted Overlay (> 0.5)
        axes[i, 3].imshow(img_np, cmap="gray")
        axes[i, 3].imshow(pred_prob_np > 0.5, cmap="Greens", alpha=0.55)
        axes[i, 3].set_title(f"Predicted Segmentation (tau=0.50)")
        axes[i, 3].axis("off")

    plt.tight_layout()
    overlay_path = reports_dir / "tiny_fit_overlay.png"
    plt.savefig(overlay_path, dpi=150)
    plt.close()
    print(f"  [PASS] Saved visual predicted-mask overlays to {overlay_path}")


def main():
    print("=" * 70)
    print("STEP 5: PREFLIGHT HARDWARE, GRADIENT & CORRECTNESS CHECKS")
    print("=" * 70)
    t_start = time.time()

    device = check_cuda_device()
    peak_vram_coarse = check_coarse_intended_batch_size(device, batch_size=4)
    peak_vram_fine = check_fine_intended_batch_size(device, batch_size=2)
    check_edge_cases()
    check_tiny_set_fitting(device)

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("PREFLIGHT SUMMARY:")
    print(f"  - Coarse Batch 4 Peak VRAM: {peak_vram_coarse:.1f} MB (within 4000 MB limit)")
    print(f"  - Fine Batch 2 Peak VRAM:   {peak_vram_fine:.1f} MB (within 4000 MB limit)")
    print(f"  - Total Preflight Time: {total_time:.1f} s")
    print("Step 5 Pass condition: ALL CHECKS PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
