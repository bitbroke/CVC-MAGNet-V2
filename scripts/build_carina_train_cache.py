"""Batch CarinaNet Inference and Caching on Traced Training Studies (N=6,600).

Runs CarinaNet landmark detection across all unique training studies with CVC polyline traces.
Applies the audit-grade fallback:
- Mediastinal bounding check: X in [0.35, 0.65], Y in [0.25, 0.60]
- Confidence threshold >= 0.15
- Out-of-bounds or low-confidence cases get cohort centroid (X=0.4805, Y=0.4000) with is_carina_fallback=1.
Saves incrementally to data/curated_cvc_v1/carina_train_cache.parquet.
"""

import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carinanet import CarinaNetModel

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
INV_PATH = DATA_DIR / "annotation_inventory.parquet"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
CACHE_DIR = DATA_DIR / "curated_cvc_v1"
CACHE_PATH = CACHE_DIR / "carina_train_cache.parquet"

FALLBACK_X = 0.4805
FALLBACK_Y = 0.4000


def main():
    print("=== Starting CarinaNet Inference on Traced Training Cohort ===")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(MANIFEST_PATH)
    inv = pd.read_parquet(INV_PATH)

    train_sids = set(manifest[manifest["split"] == "train"]["StudyInstanceUID"])
    traced_train_cvc = inv[(inv["study_id"].isin(train_sids)) & (inv["device_type"] == "CVC")]
    unique_sids = sorted(traced_train_cvc["study_id"].unique())
    print(f"Total traced training studies with CVCs: {len(unique_sids):,}")

    cached_results = {}
    if CACHE_PATH.exists():
        df_existing = pd.read_parquet(CACHE_PATH)
        for _, r in df_existing.iterrows():
            cached_results[r["study_id"]] = r.to_dict()
        print(f"Loaded existing cache with {len(cached_results):,} studies.")

    remaining_sids = [sid for sid in unique_sids if sid not in cached_results]
    print(f"Remaining studies to process: {len(remaining_sids):,}")

    if not remaining_sids:
        print("All training studies already cached!")
        return

    model = CarinaNetModel()
    results = list(cached_results.values())
    t_start = time.time()
    batch_count = 0

    for i, sid in enumerate(remaining_sids):
        img_path = IMAGES_DIR / f"{sid}.jpg"
        if not img_path.exists():
            img_path = IMAGES_DIR / f"{sid}.png"
        if not img_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        native_h, native_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)

        res = model.predict(img_resized)
        cx_640, cy_640 = res.get("carina", (0.0, 0.0))
        c_conf = float(res.get("carina_confidence", 0.0))

        ett_x_640, ett_y_640 = res.get("ett", (0.0, 0.0))
        ett_conf = float(res.get("ett_confidence", 0.0))

        raw_cx_norm = cx_640 / 640.0
        raw_cy_norm = cy_640 / 640.0

        # Plausibility check
        is_oob = not (0.35 <= raw_cx_norm <= 0.65 and 0.25 <= raw_cy_norm <= 0.60)
        is_low_conf = c_conf < 0.15
        needs_fallback = is_oob or is_low_conf

        if needs_fallback:
            cx_norm = FALLBACK_X
            cy_norm = FALLBACK_Y
            is_fallback = 1
        else:
            cx_norm = raw_cx_norm
            cy_norm = raw_cy_norm
            is_fallback = 0

        native_cx = cx_norm * native_w
        native_cy = cy_norm * native_h

        results.append({
            "study_id": sid,
            "native_w": native_w,
            "native_h": native_h,
            "carina_x_640": cx_640,
            "carina_y_640": cy_640,
            "carina_conf": c_conf,
            "raw_carina_x_norm": raw_cx_norm,
            "raw_carina_y_norm": raw_cy_norm,
            "carina_x_native": native_cx,
            "carina_y_native": native_cy,
            "carina_x_norm": cx_norm,
            "carina_y_norm": cy_norm,
            "is_carina_fallback": is_fallback,
            "ett_x_640": ett_x_640,
            "ett_y_640": ett_y_640,
            "ett_conf": ett_conf,
        })
        batch_count += 1

        if batch_count % 500 == 0 or (i + 1) == len(remaining_sids):
            elapsed = time.time() - t_start
            speed = batch_count / max(elapsed, 1.0)
            df_out = pd.DataFrame(results)
            df_out.to_parquet(CACHE_PATH, index=False)
            print(f"  [CarinaNet Train] Processed {len(results):,}/{len(unique_sids):,} ({speed:.1f} img/s) | Saved to cache", flush=True)

    print(f"\nCarinaNet inference complete for all {len(results):,} training studies.")
    print(f"Saved cache to: {CACHE_PATH}")


if __name__ == "__main__":
    main()

