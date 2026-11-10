"""Build Untraced Training Cache: T2 Continuous Segmentation, CarinaNet Landmarks, and 54-D Bilateral Features.

Applies the high-confidence ribbon filter:
- ribbon_pixels (seg_prob >= 0.25) >= 200
- max(seg_prob) >= 0.35
Saves:
1. data/curated_cvc_v1/seg_cache_512/{sid}.png
2. data/curated_cvc_v1/carina_untraced_train.parquet
3. data/curated_cvc_v1/bilateral_untraced_train_features.parquet
4. data/splits/untraced_highconf_manifest.parquet
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tip_heatmap import CVCJointSegTipNet
from carinanet import CarinaNetModel
from scripts.train_and_eval_bilateral_gbdt import compute_bilateral_features_for_study

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
SEG_CACHE_DIR = DATA_DIR / "curated_cvc_v1" / "seg_cache_512"
CARINA_UNTRACED_PATH = DATA_DIR / "curated_cvc_v1" / "carina_untraced_train.parquet"
BILATERAL_UNTRACED_PATH = DATA_DIR / "curated_cvc_v1" / "bilateral_untraced_train_features.parquet"
HIGHCONF_MANIFEST_PATH = DATA_DIR / "splits" / "untraced_highconf_manifest.parquet"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
T2_CKPT_PATH = PROJECT_ROOT / "runs" / "t2_repaired_finetune_400" / "checkpoints" / "best.pt"

FALLBACK_X = 0.4805
FALLBACK_Y = 0.4000
CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


class UntracedT2Dataset(Dataset):
    """Fast dataset for screening untraced studies with T2."""

    def __init__(self, study_ids: List[str], images_dir: Path):
        self.study_ids = study_ids
        self.images_dir = images_dir
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, bool]:
        sid = self.study_ids[idx]
        img_path = self.images_dir / f"{sid}.jpg"
        if not img_path.exists():
            return torch.zeros((3, 512, 512), dtype=torch.float32), sid, False

        img = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_COLOR_4)
        if img is None:
            return torch.zeros((3, 512, 512), dtype=torch.float32), sid, False

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

        t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        t = (t - self.mean) / self.std
        return t, sid, True


def main():
    print("=" * 80)
    print("STAGE 5: UNTRACED TRAINING COHORT SCREENING, CACHING & FEATURE EXTRACTION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    SEG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Identify Candidate Studies
    manifest = pd.read_parquet(MANIFEST_PATH)
    untraced_df = manifest[
        (manifest["split"] == "train")
        & (manifest["annotated_cvc_count"] == 0)
        & (manifest[CLASSES].sum(axis=1) > 0)
    ].reset_index(drop=True)

    print(f"Total candidate CVC-positive untraced training studies: {len(untraced_df):,}")

    # Check if already processed
    existing_highconf = None
    if HIGHCONF_MANIFEST_PATH.exists() and BILATERAL_UNTRACED_PATH.exists() and CARINA_UNTRACED_PATH.exists():
        print("Existing high-confidence cache files detected!")
        existing_highconf = pd.read_parquet(HIGHCONF_MANIFEST_PATH)
        print(f"Loaded {len(existing_highconf)} high-confidence studies from manifest.")
        if len(existing_highconf) >= 8000:
            print("Untraced cohort already fully cached and filtered! Exiting early.")
            return

    # 2. Pass 1: T2 Screening & Mask Caching
    print("\n--- Pass 1: Screening Untraced Studies with T2 ---")
    print(f"Loading T2 model from {T2_CKPT_PATH.name}...")
    t2_model = CVCJointSegTipNet(pretrained=False)
    ckpt = torch.load(T2_CKPT_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    t2_model.load_state_dict(state)
    t2_model.to(device)
    t2_model.eval()

    dataset = UntracedT2Dataset(untraced_df["StudyInstanceUID"].tolist(), IMAGES_DIR)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    highconf_records = []
    t0 = time.time()
    total_screened = 0

    with torch.no_grad():
        for b_idx, (imgs, sids, valids) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = t2_model(imgs)
                seg_probs = out["seg_probs"].squeeze(1).cpu().numpy()  # [B, 512, 512]
                tip_probs = out["tip_probs"].squeeze(1).cpu().numpy()  # [B, 128, 128]

            for i in range(len(sids)):
                total_screened += 1
                if not valids[i]:
                    continue
                sid = sids[i]
                mask_f32 = seg_probs[i]
                tip_f32 = tip_probs[i]

                # Ribbon filter rule
                ribbon_px = int((mask_f32 >= 0.25).sum())
                max_seg = float(mask_f32.max())

                if ribbon_px >= 200 and max_seg >= 0.35:
                    # Save compact uint8 PNG
                    mask_uint8 = (mask_f32 * 255.0).clip(0, 255).astype(np.uint8)
                    png_path = SEG_CACHE_DIR / f"{sid}.png"
                    if not png_path.exists():
                        cv2.imwrite(str(png_path), mask_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 3])

                    # Extract peak tip
                    tip_max = float(tip_f32.max())
                    if tip_max >= 0.15:
                        ty, tx = np.unravel_index(np.argmax(tip_f32), tip_f32.shape)
                        xtip_512 = float(tx * 4.0)
                        ytip_512 = float(ty * 4.0)
                        conf_tip = tip_max
                    else:
                        xtip_512, ytip_512, conf_tip = 0.0, 0.0, 0.0

                    highconf_records.append({
                        "study_id": sid,
                        "ribbon_pixels": ribbon_px,
                        "max_seg": max_seg,
                        "conf_tip": conf_tip,
                        "xtip_512": xtip_512,
                        "ytip_512": ytip_512,
                    })

            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(loader):
                elapsed = time.time() - t0
                speed = total_screened / max(elapsed, 1e-4)
                print(f"  Screened {total_screened}/{len(untraced_df)} ({speed:.1f} img/s) | Passed Filter: {len(highconf_records)}", flush=True)

    del t2_model
    torch.cuda.empty_cache()

    df_highconf = pd.DataFrame(highconf_records)
    print(f"\nPass 1 Complete! Total High-Confidence Untraced Studies: {len(df_highconf):,} / {len(untraced_df):,} ({len(df_highconf)/len(untraced_df)*100:.1f}%)")

    # Merge labels from manifest
    df_highconf = df_highconf.merge(
        untraced_df[["StudyInstanceUID"] + CLASSES].rename(columns={"StudyInstanceUID": "study_id"}),
        on="study_id",
        how="left"
    )
    df_highconf.to_parquet(HIGHCONF_MANIFEST_PATH, index=False)
    print(f"Saved high-confidence manifest to {HIGHCONF_MANIFEST_PATH.name}")

    # 3. Pass 2: CarinaNet Landmark Detection on High-Confidence Studies
    print("\n--- Pass 2: CarinaNet Landmark Detection on High-Confidence Studies ---")
    c_model = CarinaNetModel()
    carina_records = []
    t_c0 = time.time()
    batch_size_c = 16

    all_h_sids = df_highconf["study_id"].tolist()
    for start_idx in range(0, len(all_h_sids), batch_size_c):
        batch_sids = all_h_sids[start_idx : start_idx + batch_size_c]
        batch_pairs = []
        for sid in batch_sids:
            img_p = IMAGES_DIR / f"{sid}.jpg"
            img = cv2.imread(str(img_p), cv2.IMREAD_REDUCED_COLOR_4)
            if img is None:
                img = np.zeros((640, 640, 3), dtype=np.uint8)
            else:
                img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 640))
            t_img = c_model.preprocess_image(img)
            batch_pairs.append((t_img, sid))

        preds = c_model.model.predict(batch_pairs)
        for sid in batch_sids:
            pred = preds.get(sid, {})
            # Carina (class 0)
            if 0 in pred and "pred" in pred[0]:
                cx_640, cy_640 = pred[0]["pred"]
                c_conf = float(pred[0].get("confidence", 0.0))
            else:
                cx_640, cy_640, c_conf = 0.0, 0.0, 0.0

            raw_cx = cx_640 / 640.0
            raw_cy = cy_640 / 640.0
            is_oob = not (0.35 <= raw_cx <= 0.65 and 0.25 <= raw_cy <= 0.60)
            is_low_conf = c_conf < 0.15
            needs_fallback = is_oob or is_low_conf

            carina_records.append({
                "study_id": sid,
                "carina_x": (FALLBACK_X * 512.0) if needs_fallback else (raw_cx * 512.0),
                "carina_y": (FALLBACK_Y * 512.0) if needs_fallback else (raw_cy * 512.0),
                "carina_conf": c_conf,
                "is_carina_fallback": int(needs_fallback),
                "native_width": 512,
                "native_height": 512,
            })

        if (start_idx + batch_size_c) % 500 < batch_size_c or (start_idx + batch_size_c) >= len(all_h_sids):
            elapsed = time.time() - t_c0
            done = min(start_idx + batch_size_c, len(all_h_sids))
            print(f"  Processed {done}/{len(all_h_sids)} Carina landmarks ({done/max(elapsed, 1e-4):.1f} img/s)...", flush=True)

    del c_model
    torch.cuda.empty_cache()

    df_carina_untraced = pd.DataFrame(carina_records)
    df_carina_untraced.to_parquet(CARINA_UNTRACED_PATH, index=False)
    print(f"Pass 2 Complete! Saved {len(df_carina_untraced)} Carina coordinates to {CARINA_UNTRACED_PATH.name}")

    # 4. Pass 3: 54-D Bilateral Feature Extraction
    print("\n--- Pass 3: Extracting 54-D Bilateral Features for High-Confidence Untraced Studies ---")
    carina_dict = df_carina_untraced.set_index("study_id").to_dict("index")
    tip_dict = df_highconf.set_index("study_id").to_dict("index")

    t_f0 = time.time()
    bi_rows = []
    H, W = 512, 512
    y_grid, x_grid = np.indices((H, W))

    for idx, sid in enumerate(all_h_sids):
        png_p = SEG_CACHE_DIR / f"{sid}.png"
        mask = cv2.imread(str(png_p), cv2.IMREAD_GRAYSCALE) if png_p.exists() else np.zeros((H, W), dtype=np.uint8)
        if mask is None:
            mask = np.zeros((H, W), dtype=np.uint8)

        crow = carina_dict.get(sid, {})
        cx_norm = crow.get("carina_x", 256.0) / 512.0
        cy_norm = crow.get("carina_y", 230.0) / 512.0

        # Bilateral features
        bf = compute_bilateral_features_for_study(mask, cx_norm, cy_norm)
        bf["study_id"] = sid

        # Base zonal features
        seg_f32 = mask.astype(np.float32) / 255.0
        c_x = cx_norm * W
        c_y = cy_norm * H
        dy_grid = (y_grid - c_y) / float(H)
        dx_grid = (x_grid - c_x) / float(W)

        is_central = np.abs(dx_grid) <= 0.12
        m_neck = float(seg_f32[(dy_grid < -0.06) & is_central].sum())
        m_upper_svc = float(seg_f32[(dy_grid >= -0.06) & (dy_grid < -0.02) & is_central].sum())
        m_borderline = float(seg_f32[(dy_grid >= -0.02) & (dy_grid < 0.0) & is_central].sum())
        m_caj_normal = float(seg_f32[(dy_grid >= 0.0) & (dy_grid <= 0.05) & is_central].sum())
        m_atrium_deep = float(seg_f32[(dy_grid > 0.05) & (dy_grid <= 0.12) & is_central].sum())
        m_lateral = float(seg_f32[np.abs(dx_grid) > 0.12].sum())

        m_total_med = m_neck + m_upper_svc + m_borderline + m_caj_normal + m_atrium_deep
        denom = max(m_total_med, 1e-5)

        bf["m_neck"] = m_neck
        bf["m_upper_svc"] = m_upper_svc
        bf["m_borderline"] = m_borderline
        bf["m_caj_normal"] = m_caj_normal
        bf["m_atrium_deep"] = m_atrium_deep
        bf["m_lateral"] = m_lateral
        bf["m_total_med"] = m_total_med
        bf["frac_neck"] = m_neck / denom
        bf["frac_upper_svc"] = m_upper_svc / denom
        bf["frac_borderline"] = m_borderline / denom
        bf["frac_caj_normal"] = m_caj_normal / denom
        bf["frac_atrium_deep"] = m_atrium_deep / denom

        # Tip features
        trow = tip_dict.get(sid, {})
        ribbon_px = trow.get("ribbon_pixels", 0)
        conf_tip = trow.get("conf_tip", 0.0)
        has_valid_tip = 1.0 if conf_tip >= 0.20 else 0.0

        if has_valid_tip:
            dx_norm = float((trow.get("xtip_512", c_x) - c_x) / float(W))
            dy_norm = float((trow.get("ytip_512", c_y) - c_y) / float(H))
            r_norm = float(np.sqrt(dx_norm ** 2 + dy_norm ** 2))
            cos_theta = float(dx_norm / (r_norm + 1e-6))
            sin_theta = float(dy_norm / (r_norm + 1e-6))
        else:
            dx_norm, dy_norm, r_norm, cos_theta, sin_theta = 0.0, 0.0, 0.0, 0.0, 0.0

        bf["dx_norm"] = dx_norm
        bf["dy_norm"] = dy_norm
        bf["r_norm"] = r_norm
        bf["cos_theta"] = cos_theta
        bf["sin_theta"] = sin_theta
        bf["has_valid_tip"] = has_valid_tip
        bf["conf_tip"] = conf_tip
        bf["conf_carina"] = crow.get("carina_conf", 0.0)
        bf["is_carina_fallback"] = crow.get("is_carina_fallback", 0)
        bf["ribbon_pixels"] = ribbon_px
        bf["deepest_y_norm"] = dy_norm
        bf["p95_y_norm"] = dy_norm
        bf["highest_y_norm"] = dy_norm

        bi_rows.append(bf)

        if (idx + 1) % 1500 == 0 or (idx + 1) == len(all_h_sids):
            elapsed = time.time() - t_f0
            print(f"  Extracted {idx+1}/{len(all_h_sids)} features ({elapsed:.1f}s)...", flush=True)

    df_bi_untraced = pd.DataFrame(bi_rows)
    # Merge label columns
    df_bi_untraced = df_bi_untraced.merge(
        df_highconf[["study_id"] + CLASSES],
        on="study_id",
        how="left"
    )
    df_bi_untraced.to_parquet(BILATERAL_UNTRACED_PATH, index=False)
    print(f"Pass 3 Complete! Saved {len(df_bi_untraced)} bilateral features to {BILATERAL_UNTRACED_PATH.name}")

    print("\n" + "=" * 80)
    print("STAGE 5 UNTRACED DATA PIPELINE COMPLETE AND FULLY CACHED!")
    print("=" * 80)


if __name__ == "__main__":
    main()

