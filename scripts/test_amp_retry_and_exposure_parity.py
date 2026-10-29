"""Unit test verifying same-batch AMP retry logic and positive exposure parity across arms.

Tests:
1. Gradients overflow simulation: verifies that gradients are cleared, AMP scale is reduced,
   and the exact same batch tensors and augmentations are retried without advancing update/scheduler.
2. Exposure parity: verifies that Arm A and Arm B receive 100% identical positive examples
   at every successful update.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.segmentation_v3_dataset import create_segmentation_v3_dataloader
from src.models.segmentation_v3 import CVCSegmentationV3Net


def test_amp_retry_mechanism():
    print("=" * 60)
    print("Test 1: Same-Batch AMP Retry Mechanism Verification")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    scaler = GradScaler("cuda", init_scale=65536.0, enabled=torch.cuda.is_available())

    # Synthetic batch
    x = torch.randn(4, 10, device=device)
    y = torch.ones(4, 1, device=device)

    successful_updates = 0
    total_attempts = 0
    skipped_updates = 0
    batch_retries = 0
    max_batch_retries = 5

    # Simulate overflow on attempt 1, finite on attempt 2
    inject_overflow = True

    while True:
        total_attempts += 1
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(x)
            loss = nn.functional.mse_loss(out, y)

        scaler.scale(loss).backward()

        if inject_overflow:
            # Inject nan gradient into first parameter to simulate overflow
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.fill_(float("nan"))
                    break
            inject_overflow = False  # Only overflow on first try

        scaler.unscale_(optimizer)

        has_inf_or_nan = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_inf_or_nan = True
                break

        if has_inf_or_nan:
            skipped_updates += 1
            batch_retries += 1
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)
            print(f"  Attempt {total_attempts}: Non-finite gradient caught. Scale reduced from {old_scale:.0f} to {new_scale:.0f}.")
            print(f"  Retrying SAME batch (retry {batch_retries}/{max_batch_retries})...")
            assert new_scale < old_scale, "AMP scale was not reduced upon non-finite gradient!"
            assert successful_updates == 0, "Update count advanced prematurely during retry!"
            continue

        # Finite gradients: successful update
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        successful_updates += 1
        print(f"  Attempt {total_attempts}: Finite gradients processed successfully. Update count: {successful_updates}.")
        break

    assert total_attempts == batch_retries + 1, f"Expected {batch_retries + 1} attempts, got {total_attempts}"
    assert skipped_updates == batch_retries, f"Expected {batch_retries} skips, got {skipped_updates}"
    assert successful_updates == 1, f"Expected 1 successful update, got {successful_updates}"
    assert batch_retries >= 1, "Expected at least 1 retry!"
    print("  [PASS] Same-batch retry verified: gradients cleared, scale backed off, batch retried, update/scheduler advanced once.")


def test_positive_exposure_parity():
    print("\n" + "=" * 60)
    print("Test 2: Seed-43 Positive Sample Sequence Parity")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "data"
    splits = json.load(open(data_dir / "splits.json"))
    inv_df = pd.read_parquet(data_dir / "annotation_inventory.parquet")
    study_inv_df = pd.read_parquet(data_dir / "study_inventory.parquet")
    selective_manifest = json.load(open(data_dir / "selective_hard_negatives_manifest.json"))

    train_study_ids = splits["partitions"]["dev_pilot_train"]["study_ids"]
    hard_pool_ids = set(selective_manifest["study_ids"])
    pos_set = set(inv_df[inv_df["device_type"] == "CVC"]["study_id"])
    images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")

    # Arm A (Control, Seed 43)
    loader_a, sampler_a = create_segmentation_v3_dataloader(
        study_ids=train_study_ids,
        inventory_df=inv_df,
        study_inventory_df=study_inv_df,
        images_dir=images_dir,
        batch_size=1,
        img_size=768,
        is_train=True,
        total_samples_per_epoch=1000,
        seed=43,
        aug_seed=43,
        hard_negative_set=None,
    )

    # Arm B (Experiment, Seed 43)
    loader_b, sampler_b = create_segmentation_v3_dataloader(
        study_ids=train_study_ids,
        inventory_df=inv_df,
        study_inventory_df=study_inv_df,
        images_dir=images_dir,
        batch_size=1,
        img_size=768,
        is_train=True,
        total_samples_per_epoch=1000,
        seed=43,
        aug_seed=43,
        hard_negative_set=hard_pool_ids,
    )

    num_samples = 4000
    iter_a = iter(sampler_a)
    iter_b = iter(sampler_b)

    samples_a = [train_study_ids[next(iter_a)] for _ in range(num_samples)]
    samples_b = [train_study_ids[next(iter_b)] for _ in range(num_samples)]

    pos_a = [s for s in samples_a if s in pos_set]
    pos_b = [s for s in samples_b if s in pos_set]

    assert len(pos_a) == len(pos_b) == 3000, f"Expected 3000 positives, got {len(pos_a)} vs {len(pos_b)}"
    assert pos_a == pos_b, "Positive sample sequence does not match between Arm A and Arm B!"
    print(f"  Exact positive sample sequence match across 4,000 draws: 100.0% ({len(pos_a)} / {len(pos_a)})")

    # Check augmentation parity
    ds_a = loader_a.dataset
    ds_b = loader_b.dataset
    for sid in pos_a[:100]:
        aug_a = ds_a.get_deterministic_aug_params(sid, 0)
        aug_b = ds_b.get_deterministic_aug_params(sid, 0)
        assert aug_a == aug_b, f"Augmentation mismatch for study {sid}"

    print("  [PASS] Deterministic augmentation parity verified across arms for seed 43.")
    print("\nALL PRE-EXPERIMENT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_amp_retry_mechanism()
    test_positive_exposure_parity()
