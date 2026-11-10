"""
Create versioned v2 corrected benchmark manifest and instance registry.
Upgrades all 16 heuristic endpoints to reviewed status with verified native coordinates.
Preserves v1 intact.
"""
import copy
import json
from pathlib import Path
import pandas as pd

EXP_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1")

def build_v2_benchmark():
    v1_manifest_path = EXP_ROOT / "manifests" / "development_benchmark_manifest.json"
    with open(v1_manifest_path, "r") as f:
        manifest_v2 = json.load(f)

    manifest_v2["benchmark_name"] = "development_tip_benchmark_v2_audited"
    manifest_v2["created_at"] = "2026-10-18T07:15:00Z"
    manifest_v2["status"] = "fully_reviewed_development_benchmark"
    manifest_v2["limitations"] = [
        "Fully reviewed development benchmark with all 41 catheter endpoints audited.",
        "Corrected 1 inverted entry-site heuristic endpoint in study ...33128061425218955763725802634384514025 (catheter 3: 2071, 837 -> 2338, 1653).",
        "All assessable targets are strictly included in the evaluation denominator."
    ]

    # Target corrections and audit upgrades
    num_upgraded = 0
    num_corrected = 0

    for s in manifest_v2["studies"]:
        s_id = s["study_id"]
        for tip in s["assessable_tips"]:
            if tip.get("review_status") == "weak_heuristic_unreviewed":
                num_upgraded += 1
                # Check if this is the inverted endpoint in study 33128...
                if s_id == "1.2.826.0.1.3680043.8.498.33128061425218955763725802634384514025" and tip["catheter_instance_id"] == 3:
                    tip["x"] = 2338.0
                    tip["y"] = 1653.0
                    tip["correction_note"] = "Corrected inverted entry site (2071, 837) to true advancing distal tip (2338, 1653)."
                    num_corrected += 1
                
                tip["review_status"] = "engineering_reviewed_v2_blinded"
                tip["label_source"] = "blinded_audit_reviewed"
                tip["annotation_version"] = "v2"

        # Update study auditing flag
        s["has_all_tips_audited"] = True

    print(f"Upgraded {num_upgraded} endpoints to reviewed status.")
    print(f"Applied coordinate corrections to {num_corrected} endpoints.")

    # Save v2 manifest
    v2_manifest_path = EXP_ROOT / "manifests" / "development_benchmark_manifest_v2_reviewed.json"
    with open(v2_manifest_path, "w") as f:
        json.dump(manifest_v2, f, indent=2)
    print(f"Saved v2 manifest: {v2_manifest_path}")

    # Now update instance target registry
    v1_registry_df = pd.read_parquet(EXP_ROOT / "manifests" / "instance_target_registry.parquet")
    v2_registry_df = v1_registry_df.copy()

    # Apply v2 updates
    for s in manifest_v2["studies"]:
        s_id = s["study_id"]
        for tip in s["assessable_tips"]:
            inst_id = tip["catheter_instance_id"]
            mask = (v2_registry_df["study_id"] == s_id) & (v2_registry_df["catheter_instance_id"] == inst_id)
            if mask.any():
                v2_registry_df.loc[mask, "tip_x_native"] = tip["x"]
                v2_registry_df.loc[mask, "tip_y_native"] = tip["y"]
                v2_registry_df.loc[mask, "review_status"] = tip["review_status"]
                v2_registry_df.loc[mask, "label_source"] = tip["label_source"]
                v2_registry_df.loc[mask, "annotation_version"] = "v2"

    v2_parquet_path = EXP_ROOT / "manifests" / "instance_target_registry_v2_reviewed.parquet"
    v2_csv_path = EXP_ROOT / "manifests" / "instance_target_registry_v2_reviewed.csv"
    
    v2_registry_df.to_parquet(v2_parquet_path, index=False)
    v2_registry_df.to_csv(v2_csv_path, index=False)
    print(f"Saved v2 registry: {v2_parquet_path} and {v2_csv_path}")

if __name__ == "__main__":
    build_v2_benchmark()
