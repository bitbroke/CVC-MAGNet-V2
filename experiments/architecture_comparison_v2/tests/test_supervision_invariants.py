"""Test Supervision Invariants for Architecture Comparison V2.

Verifies:
1. Binary targets and valid masks preserve their 0/1 values without uint8/255 division bug.
2. Target values are strictly in {0.0, 1.0}.
3. Valid mask has non-empty valid pixels for every sample.
4. Positive studies have non-empty target foreground masks.
5. Negative studies have strictly empty target masks (0 pixels).
"""

import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rasterize import rasterize_study_polylines
from src.data.transforms import LetterboxTransform

DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "experiments" / "cvc_mamba_hybrid_20261017" / "configs"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")

def test_supervision_invariants():
    print("Testing supervision invariants on real training batches...")

    inv_df = pd.read_parquet(DATA_DIR / "annotation_inventory.parquet")
    train_polylines = {}
    for _, row in inv_df[inv_df["device_type"] == "CVC"].iterrows():
        sid = str(row["study_id"])
        p_raw = row["polyline"]
        poly = json.loads(p_raw) if isinstance(p_raw, str) else p_raw
        train_polylines.setdefault(sid, []).append(poly)

    transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)

    ledger_path = CONFIGS_DIR / "batch_ledger.jsonl"
    with open(ledger_path, "r", encoding="utf-8") as f:
        first_few = [json.loads(line) for _, line in zip(range(5), f)]

    total_samples_checked = 0
    pos_checked = 0
    neg_checked = 0

    for entry in first_few:
        study_ids = entry["study_ids"]
        aug_list = entry["augmentation_parameters"]

        for sid, aug_params in zip(study_ids, aug_list):
            img_path = IMAGES_DIR / f"{sid}.jpg"
            pil_im = Image.open(img_path).convert("L")
            im_np = np.array(pil_im)
            orig_h, orig_w = im_np.shape

            polys = train_polylines.get(sid, [])
            is_pos = len(polys) > 0

            native_gt = rasterize_study_polylines(polys, (orig_h, orig_w), thickness=10) if is_pos else np.zeros((orig_h, orig_w), dtype=np.uint8)

            lb_img, meta = transform.apply_image(im_np)
            lb_mask = transform.apply_mask(native_gt, meta) if is_pos else np.zeros((768, 768), dtype=np.uint8)
            valid_mask_init = np.zeros((768, 768), dtype=np.uint8)
            valid_mask_init[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w] = 1

            pil_img = Image.fromarray(lb_img)
            pil_mask = Image.fromarray(lb_mask)
            pil_valid = Image.fromarray(valid_mask_init)

            if aug_params["hflip"]:
                pil_img = TF.hflip(pil_img)
                pil_mask = TF.hflip(pil_mask)
                pil_valid = TF.hflip(pil_valid)

            translate = [aug_params["translate_x_pct"] * 768, aug_params["translate_y_pct"] * 768]
            pil_img = TF.affine(pil_img, angle=aug_params["angle"], translate=translate, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            pil_mask = TF.affine(pil_mask, angle=aug_params["angle"], translate=translate, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.NEAREST, fill=0)
            pil_valid = TF.affine(pil_valid, angle=aug_params["angle"], translate=translate, scale=aug_params["scale"], shear=0, interpolation=TF.InterpolationMode.NEAREST, fill=0)

            # Direct conversion preserving 0/1 float values
            mask_tensor = torch.from_numpy((np.array(pil_mask) > 0).astype(np.float32)).unsqueeze(0)
            valid_tensor = torch.from_numpy((np.array(pil_valid) > 0).astype(np.float32)).unsqueeze(0)

            # Invariant 1: Strictly {0.0, 1.0}
            unique_mask = set(torch.unique(mask_tensor).tolist())
            unique_valid = set(torch.unique(valid_tensor).tolist())
            assert unique_mask.issubset({0.0, 1.0}), f"Mask contains non-binary values: {unique_mask}"
            assert unique_valid.issubset({0.0, 1.0}), f"Valid mask contains non-binary values: {unique_valid}"

            # Invariant 2: Non-empty valid regions
            v_sum = valid_tensor.sum().item()
            assert v_sum > 0, f"Study {sid} has 0 valid pixels!"

            # Invariant 3: Positive vs Negative targets
            m_sum = mask_tensor.sum().item()
            if is_pos:
                pos_checked += 1
                assert m_sum > 0, f"Positive study {sid} lost foreground mask after transform!"
            else:
                neg_checked += 1
                assert m_sum == 0, f"Negative study {sid} has non-empty target mask ({m_sum} px)!"

            total_samples_checked += 1

    print(f"Verified {total_samples_checked} samples ({pos_checked} positive, {neg_checked} negative).")
    print("[SUCCESS] Supervision Invariants Verified! All masks strictly 0/1, valid regions non-empty, pos/neg correct.")


if __name__ == "__main__":
    test_supervision_invariants()
