"""Step 2: Dual-Segmenter Consensus Mask Fusion & High-Res D1 Tip Extraction.

Implements the Dual-Segmenter Consensus protocol:
1. S_consensus(x, y) = 0.60 * S_T3(x, y) + 0.40 * S_T2(x, y)
   - Eliminates rib-crossing dropouts where either model dips.
   - Preserves probability distribution compatibility with E56, E62, and E61b.
2. Extracts T3 tips at 768px via D1 DARK subpixel refinement.
3. Updates tip lookup whenever conf_tip_T3 >= conf_tip_T2.
4. Saves consensus masks to data/curated_cvc_v1/consensus_seg_cache_512/
   (preserving original seg_cache_512 atomically).
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer.decoders import extract_peaks_from_heatmap, decode_d1_subpixel_dark
from src.models.tip_heatmap import CVCJointSegTipNet

DATA_DIR = PROJECT_ROOT / "data"
T2_CACHE_DIR = DATA_DIR / "curated_cvc_v1" / "seg_cache_512"
CONSENSUS_CACHE_DIR = DATA_DIR / "curated_cvc_v1" / "consensus_seg_cache_512"
T3_CKPT_PATH = PROJECT_ROOT / "runs" / "T3_anatomical_unet" / "checkpoints" / "best.pt"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
ZONAL_TR_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"
CARINA_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"
CARINA_TR_PATH = DATA_DIR / "curated_cvc_v1" / "carina_train_cache.parquet"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")


def process_cohort_fusion(
    model: torch.nn.Module,
    sids: List[str],
    zonal_df: pd.DataFrame,
    carina_df: pd.DataFrame,
    cohort_name: str,
    device: torch.device,
) -> pd.DataFrame:
    print(f"\nProcessing Dual-Segmenter Fusion for {cohort_name} (N={len(sids):,})...")
    CONSENSUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    zonal_map = zonal_df.set_index("study_id").to_dict(orient="index") if "study_id" in zonal_df.columns else {}
    carina_map = carina_df.set_index("study_id").to_dict(orient="index") if "study_id" in carina_df.columns else {}

    tip_records = []
    t0 = time.time()
    n_upgraded = 0

    mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_rgb = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    with torch.no_grad():
        for idx, sid in enumerate(sids):
            # 1. Read Image
            img_path = IMAGES_DIR / f"{sid}.jpg"
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                orig_h, orig_w = 3000, 3000
                img_tensor = torch.zeros(1, 3, 768, 768, device=device)
            else:
                orig_h, orig_w = img_bgr.shape[:2]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (768, 768), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                img_norm = (img_resized - mean_rgb) / std_rgb
                img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # 2. Forward T3
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(img_tensor)
                s3_768 = out["seg_probs"][0, 0]
                t3_map = out["tip_probs"][0, 0].cpu().float().numpy()

            # Downsample T3 mask to 512x512
            s3_512 = F.interpolate(s3_768.unsqueeze(0).unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False)[0, 0].cpu().float().numpy()

            # 3. Read T2 Cached Mask
            t2_path = T2_CACHE_DIR / f"{sid}.png"
            if t2_path.exists():
                s2_512 = cv2.imread(str(t2_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            else:
                s2_512 = s3_512

            # 4. Dual-Segmenter Consensus Ribbon: 0.60 * T3 + 0.40 * T2
            s_consensus = 0.60 * s3_512 + 0.40 * s2_512
            s_consensus_uint8 = np.clip(s_consensus * 255.0, 0, 255).astype(np.uint8)

            out_mask_path = CONSENSUS_CACHE_DIR / f"{sid}.png"
            cv2.imwrite(str(out_mask_path), s_consensus_uint8)

            # 5. Extract T3 Tip via D1 DARK Subpixel Decoding
            _, nms_peaks = extract_peaks_from_heatmap(t3_map, threshold=0.15, nms_kernel_size=3, suppression_radius_px=3.0)
            
            t2_info = zonal_map.get(sid, {})
            c_info = carina_map.get(sid, {})
            
            # Carina anchor
            if "raw_carina_x_norm" in c_info:
                cx_512 = float(c_info["raw_carina_x_norm"]) * 512.0
                cy_512 = float(c_info["raw_carina_y_norm"]) * 512.0
            else:
                cx_512 = float(c_info.get("carina_x_640", 307.5)) * (512.0 / 640.0)
                cy_512 = float(c_info.get("carina_y_640", 256.0)) * (512.0 / 640.0)

            t2_conf = float(t2_info.get("conf_tip", 0.0))
            t2_xtip = cx_512 + float(t2_info.get("dx_norm", 0.0)) * 512.0
            t2_ytip = cy_512 + float(t2_info.get("dy_norm", 0.0)) * 512.0

            if nms_peaks:
                best_peak = nms_peaks[0]
                nx, ny, _ = decode_d1_subpixel_dark(t3_map, best_peak, stride=4, img_size=768, native_w=orig_w, native_h=orig_h)
                t3_xtip = (nx / orig_w) * 512.0
                t3_ytip = (ny / orig_h) * 512.0
                t3_conf = float(best_peak["conf"])
            else:
                t3_xtip = t2_xtip
                t3_ytip = t2_ytip
                t3_conf = 0.0

            # Consensus Tip Selection: Upgrade to T3 whenever conf_T3 >= conf_T2
            if t3_conf >= t2_conf and t3_conf > 0.15:
                xtip = t3_xtip
                ytip = t3_ytip
                conf_tip = t3_conf
                source = "T3_highres"
                n_upgraded += 1
            else:
                xtip = t2_xtip
                ytip = t2_ytip
                conf_tip = t2_conf
                source = "T2_prior"

            tip_records.append({
                "study_id": sid,
                "xtip_512": float(xtip),
                "ytip_512": float(ytip),
                "conf_tip": float(conf_tip),
                "tip_source": source,
                "dx_norm": float((xtip - cx_512) / 512.0),
                "dy_norm": float((ytip - cy_512) / 512.0),
            })

            if (idx + 1) % 500 == 0 or (idx + 1) == len(sids):
                rate = (idx + 1) / max(time.time() - t0, 1e-4)
                print(f"  Processed {idx+1}/{len(sids)} studies ({rate:.1f} img/s) | T3 Upgrades: {n_upgraded}/{idx+1} ({n_upgraded/(idx+1)*100:.1f}%)")

    df_tips = pd.DataFrame(tip_records)
    out_tips_path = DATA_DIR / "curated_cvc_v1" / f"t3_consensus_{cohort_name}_tips.parquet"
    df_tips.to_parquet(out_tips_path, index=False)
    print(f"Completed {cohort_name} in {time.time()-t0:.1f}s -> {out_tips_path.name}")
    return df_tips


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"STEP 2: DUAL-SEGMENTER CONSENSUS MASK FUSION & T3 D1 TIP UPGRADE")
    print("=" * 80)

    # 1. Load Trained T3 Model
    print(f"Loading T3 Model from {T3_CKPT_PATH}...")
    model = CVCJointSegTipNet(pretrained=False, freeze_encoder_bn=True)
    ckpt = torch.load(T3_CKPT_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state_f32 = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    model.load_state_dict(state_f32)
    model.to(device)
    model.eval()
    print("T3 weights successfully loaded into model!")

    # 2. Load Manifests
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    tr_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)

    val_sids = list(val_manifest["StudyInstanceUID"].values)
    tr_sids = list(tr_manifest[tr_manifest["annotated_cvc_count"] > 0]["StudyInstanceUID"].values)

    df_zonal_val = pd.read_parquet(ZONAL_VAL_PATH)
    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)
    df_zonal_tr = pd.read_parquet(ZONAL_TR_PATH)
    df_carina_tr = pd.read_parquet(CARINA_TR_PATH)

    # 3. Process Validation Split (N=2,558)
    process_cohort_fusion(model, val_sids, df_zonal_val, df_carina_val, "val", device)

    # 4. Process Traced Train Split (N=6,600)
    process_cohort_fusion(model, tr_sids, df_zonal_tr, df_carina_tr, "train", device)

    print("\n" + "=" * 80)
    print("STEP 2 COMPLETE: Consensus Masks and High-Res Tips Successfully Generated!")
    print("=" * 80)


if __name__ == "__main__":
    main()

