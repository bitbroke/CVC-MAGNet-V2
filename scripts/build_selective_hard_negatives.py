"""Build and audit genuinely selective top-99 hard negative pool.

Protocol:
1. Rank all 396 confirmed training negatives by false foreground area % on valid image area.
2. Select top 99 (top 25%).
3. Verify strict containment in dev_pilot_train with ZERO patient or study leakage into val.
4. Export data/selective_hard_negatives_manifest.json and reports/hard_negatives_top99_manifest.csv.
5. Render visual overlays for top 5 and boundary ranks 95-99 for human audit.
"""

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import LetterboxTransform
from src.models.segmentation_v3 import CVCSegmentationV3Net


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 60)
    print("Building Selective Hard-Negative Pool (Top 99 / 25%)")
    print("=" * 60)

    # 1. Load splits and verify
    data_dir = PROJECT_ROOT / "data"
    splits = json.load(open(data_dir / "splits.json"))
    study_inv_df = pd.read_parquet(data_dir / "study_inventory.parquet")
    study_to_pid = dict(zip(study_inv_df["StudyInstanceUID"], study_inv_df["PatientID"]))

    train_study_ids = set(splits["partitions"]["dev_pilot_train"]["study_ids"])
    train_patient_ids = {study_to_pid[sid] for sid in train_study_ids if sid in study_to_pid}
    val_study_ids = set(splits["partitions"]["dev_pilot_val"]["study_ids"])
    val_patient_ids = {study_to_pid[sid] for sid in val_study_ids if sid in study_to_pid}

    ranked_csv_path = PROJECT_ROOT / "reports" / "hard_negatives_ranked.csv"
    if not ranked_csv_path.exists():
        raise FileNotFoundError(f"Missing {ranked_csv_path}")

    ranked_df = pd.read_csv(ranked_csv_path)
    print(f"Loaded {len(ranked_df)} ranked training negatives.")

    # Select top 99
    top99_df = ranked_df.iloc[:99].copy()
    top99_studies = list(top99_df["study_id"])
    top99_patients = set(top99_df["patient_id"])

    # Strict isolation assertions
    assert len(top99_studies) == 99, f"Expected 99 studies, got {len(top99_studies)}"
    assert len(set(top99_studies)) == 99, "Duplicate study IDs in top 99!"
    assert set(top99_studies).issubset(train_study_ids), "LEAKAGE: Study IDs outside dev_pilot_train!"
    assert len(set(top99_studies).intersection(val_study_ids)) == 0, "LEAKAGE: Study IDs in dev_pilot_val!"
    assert len(top99_patients.intersection(val_patient_ids)) == 0, "LEAKAGE: Patient IDs in dev_pilot_val!"

    print("Strict subset and zero validation leakage assertions PASSED.")
    print(f"Rank 1 false area: {top99_df.iloc[0]['false_area_pct_tau50']} ({top99_df.iloc[0]['false_pixels_tau50']} px)")
    print(f"Rank 99 false area: {top99_df.iloc[98]['false_area_pct_tau50']} ({top99_df.iloc[98]['false_pixels_tau50']} px)")
    print(f"Rank 100 false area (cutoff): {ranked_df.iloc[99]['false_area_pct_tau50']} ({ranked_df.iloc[99]['false_pixels_tau50']} px)")

    # Scoring checkpoint provenance
    scoring_ckpt = PROJECT_ROOT / "runs" / "s2_repaired_768" / "checkpoints" / "final_segmentation_v3.pt"
    scoring_sha = compute_sha256(str(scoring_ckpt))

    manifest = {
        "description": "Genuinely selective hard-negative pool consisting of top 99 (top 25%) confirmed training negatives ranked by false foreground area % on valid image area.",
        "pool_size": 99,
        "total_training_negatives": 396,
        "pool_fraction": 0.25,
        "scoring_checkpoint": str(scoring_ckpt.relative_to(PROJECT_ROOT)),
        "scoring_checkpoint_sha256": scoring_sha,
        "decision_threshold": 0.50,
        "resolution": 768,
        "rank_1_false_area_pct": float(top99_df.iloc[0]["false_area_pct_tau50"].rstrip("%")),
        "rank_99_false_area_pct": float(top99_df.iloc[98]["false_area_pct_tau50"].rstrip("%")),
        "rank_100_cutoff_pct": float(ranked_df.iloc[99]["false_area_pct_tau50"].rstrip("%")),
        "study_ids": top99_studies,
        "studies": top99_df.to_dict(orient="records"),
    }

    manifest_path = data_dir / "selective_hard_negatives_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved selective hard negative manifest to: {manifest_path}")

    # Generate boundary overlays for ranks 95-99
    overlays_dir = PROJECT_ROOT / "reports" / "overlays" / "selective_hard_negatives"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(scoring_ckpt), map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = CVCSegmentationV3Net(pretrained=False, freeze_encoder_bn=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)

    # Render panels for rank 1-5 and ranks 95-99
    inspect_indices = [0, 1, 2, 3, 4, 94, 95, 96, 97, 98]
    print(f"Generating visual audit panels for {len(inspect_indices)} selected hard negatives...")

    for idx in inspect_indices:
        row = top99_df.iloc[idx]
        sid = row["study_id"]
        rank = row["rank"]
        img_path = images_dir / f"{sid}.jpg"
        im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = im.shape[:2]

        lb_im, meta = transform.apply_image(im)
        tensor = torch.from_numpy(lb_im).float().unsqueeze(0).repeat(3, 1, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)
        norm_tensor = (tensor - mean) / std

        with torch.no_grad():
            outputs = model(norm_tensor)
            probs = outputs["seg_probs"][0, 0].cpu().numpy()

        # Unletterbox to native resolution
        valid_probs = probs[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w]
        native_probs = cv2.resize(valid_probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        native_mask = (native_probs >= 0.50).astype(np.uint8)

        # Build 3-panel visualization
        vis_im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        heat = cv2.applyColorMap((native_probs * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat_overlay = cv2.addWeighted(vis_im, 0.6, heat, 0.4, 0)

        mask_overlay = vis_im.copy()
        mask_overlay[native_mask > 0] = [0, 0, 255]  # Red false track
        mask_overlay = cv2.addWeighted(vis_im, 0.5, mask_overlay, 0.5, 0)

        # Annotations text
        header = f"Rank {rank} (Neg) | False Area: {row['false_area_pct_tau50']} | Max Prob: {row['max_prob']:.3f}"
        sub = f"Other lines: {row['has_other_lines']}, ETT: {row['has_ett']}, NGT: {row['has_ngt']}"

        # Resize for display
        disp_h, disp_w = 600, int(600 * (orig_w / orig_h))
        panel1 = cv2.resize(vis_im, (disp_w, disp_h))
        panel2 = cv2.resize(heat_overlay, (disp_w, disp_h))
        panel3 = cv2.resize(mask_overlay, (disp_w, disp_h))

        combined = np.hstack([panel1, panel2, panel3])
        bar = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, header, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(bar, sub, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        final_panel = np.vstack([bar, combined])

        out_name = f"rank_{rank:03d}_{sid[:16]}.png"
        cv2.imwrite(str(overlays_dir / out_name), final_panel)
        print(f"  Generated audit overlay: {out_name}")

    print("Selective hard-negative pool generation and inspection complete.")


if __name__ == "__main__":
    main()
