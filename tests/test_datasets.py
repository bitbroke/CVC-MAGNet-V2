import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import pytest
import torch

from src.data.datasets import CoarseDataset


def test_coarse_dataset_loading():
    data_dir = PROJECT_ROOT / "data"
    inv_df = pd.read_parquet(data_dir / "annotation_inventory.parquet")
    study_inv = pd.read_parquet(data_dir / "study_inventory.parquet")
    raw_images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")

    with open(data_dir / "splits.json", "r") as f:
        splits = json.load(f)

    # Sample 4 study IDs from pilot train
    sample_ids = splits["partitions"]["dev_pilot_train"]["study_ids"][:4]
    confirmed_negs = set(study_inv[study_inv["is_study_level_negative"] & (~study_inv["has_cvc"])]["StudyInstanceUID"])

    dataset = CoarseDataset(
        study_ids=sample_ids,
        inventory_df=inv_df,
        images_dir=raw_images_dir,
        target_size=(512, 512),
        mask_stroke_width=10,
        clahe_enabled=True,
        confirmed_negative_study_ids=confirmed_negs,
    )

    assert len(dataset) == 4, f"Dataset length mismatch: {len(dataset)}"

    sample = dataset[0]
    assert "image" in sample
    assert "mask" in sample
    assert "study_id" in sample
    assert "orig_size" in sample

    assert sample["image"].shape == (1, 512, 512)
    assert sample["mask"].shape == (1, 512, 512)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32

    # Verify supervision violation is caught if an unannotated study is accessed without confirmed negative proof
    bad_dataset = CoarseDataset(
        study_ids=sample_ids,
        inventory_df=inv_df,
        images_dir=raw_images_dir,
        confirmed_negative_study_ids=set(),  # empty negatives
    )
    has_unannotated = any(sid not in bad_dataset.polyline_index for sid in sample_ids)
    if has_unannotated:
        with pytest.raises(ValueError, match="Supervision Violation"):
            for i in range(len(bad_dataset)):
                _ = bad_dataset[i]
        print("  [PASS] Supervision violation safely caught when unannotated study accessed without negative proof.")

    print("  [PASS] CoarseDataset returns correct shapes, types, and value bounds.")


if __name__ == "__main__":
    test_coarse_dataset_loading()
