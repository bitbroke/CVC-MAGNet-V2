"""Script to build training-only label manifest using the exact frozen endpoint selector.

Implementation details:
1. Frozen Selector: Uses select_endpoint from outputs.endpoint_selector_pilot.endpoint_selector (VERSION='conservative_endpoint_v1').
2. Reviewed-Label Reconciliation:
   - Uses data/reviewed_distal_tips_v2.parquet (train split).
   - Reviewed corrections (such as Case 18, which corrected the heuristic to Point 0) are legitimately valid and retained.
   - Provenance explicitly recorded as: "provisional_engineering_review_awaiting_clinical_attestation".
3. Split Isolation:
   - Strictly dev_pilot_train (400 positive CVC studies, 396 confirmed negative studies).
   - Zero validation studies exposed.
4. Multi-CVC Completeness Rule:
   - In multi-CVC studies, tip supervision is active ONLY if EVERY CVC instance in that study is resolved (reviewed or approved by frozen selector).
   - If any line in a multi-CVC study abstains, the entire study is masked from tip supervision (tip_mask = 0).
5. Output format:
   - data/training_expansion_manifest.parquet
   - data/training_expansion_manifest.csv
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
sys.path.insert(0, str(PROJECT_ROOT))

from outputs.endpoint_selector_pilot.endpoint_selector import select_endpoint, VERSION

def build_manifest():
    print("=" * 70)
    print(f"BUILD TRAINING EXPANSION MANIFEST (SELECTOR: {VERSION})")
    print("=" * 70)

    # 1. Load Splits
    with open(PROJECT_ROOT / "data" / "splits.json") as f:
        splits = json.load(f)
    train_sids = set(splits["partitions"]["dev_pilot_train"]["study_ids"])
    print(f"Loaded dev_pilot_train: {len(train_sids)} studies")

    # 2. Load Inventories & Reviewed V2
    inv_df = pd.read_parquet(PROJECT_ROOT / "data" / "annotation_inventory.parquet")
    study_df = pd.read_parquet(PROJECT_ROOT / "data" / "study_inventory.parquet")
    reviewed_v2 = pd.read_parquet(PROJECT_ROOT / "data" / "reviewed_distal_tips_v2.parquet")

    # Filter to CVC in dev_pilot_train
    train_cvc = inv_df[(inv_df["device_type"] == "CVC") & (inv_df["study_id"].isin(train_sids))].copy()
    print(f"Train CVC instances: {len(train_cvc)} across {train_cvc['study_id'].nunique()} positive studies")

    # Index reviewed training cases
    reviewed_train = reviewed_v2[reviewed_v2["split"] == "train"].copy()
    reviewed_lookup = {}
    for _, r in reviewed_train.iterrows():
        key = (r["study_id"], int(r["catheter_instance_id"]))
        reviewed_lookup[key] = r
    print(f"Indexed {len(reviewed_lookup)} reviewed training instances from v2")

    # First pass: evaluate each instance with reviewed priority or frozen selector
    study_instances: Dict[str, List[Dict]] = {}
    selector_proposed_count = 0
    selector_abstained_count = 0
    reviewed_count = 0

    for _, row in train_cvc.iterrows():
        sid = row["study_id"]
        cid = int(row["catheter_instance_id"])
        pts = json.loads(row["polyline"])
        w = int(row["image_width"])
        h = int(row["image_height"])

        p0 = [float(pts[0][0]), float(pts[0][1])]
        plast = [float(pts[-1][0]), float(pts[-1][1])]

        is_reviewed = (sid, cid) in reviewed_lookup
        
        if is_reviewed:
            # Reviewed instance takes precedence
            rev_r = reviewed_lookup[(sid, cid)]
            tip_x = float(rev_r["reviewed_tip_x"])
            tip_y = float(rev_r["reviewed_tip_y"])
            ep_class = str(rev_r["reviewed_endpoint_class"])  # e.g. TIP_ENDPOINT_0 or TIP_ENDPOINT_LAST
            source = "provisional_engineering_review_awaiting_clinical_attestation"
            uncertainty = False
            abstain_reason = None
            reviewed_count += 1
        else:
            # Run exact frozen conservative selector
            decision, reason = select_endpoint(pts, w, h)
            if decision == 'A':
                tip_x = p0[0]
                tip_y = p0[1]
                ep_class = "A"
                source = "provisional_automatic_selector_v1"
                uncertainty = False
                abstain_reason = None
                selector_proposed_count += 1
            elif decision == 'B':
                tip_x = plast[0]
                tip_y = plast[1]
                ep_class = "B"
                source = "provisional_automatic_selector_v1"
                uncertainty = False
                abstain_reason = None
                selector_proposed_count += 1
            else:
                tip_x = None
                tip_y = None
                ep_class = None
                source = f"abstained_{reason}"
                uncertainty = True
                abstain_reason = reason
                selector_abstained_count += 1

        inst_data = {
            "study_id": sid,
            "catheter_instance_id": cid,
            "instance_key": f"{sid}_{cid}",
            "split": "dev_pilot_train",
            "device_label": str(row["label"]),
            "image_width": w,
            "image_height": h,
            "num_points": len(pts),
            "polyline": row["polyline"],
            "endpoint_a": p0,
            "endpoint_b": plast,
            "is_reviewed": is_reviewed,
            "candidate_endpoint_class": ep_class,
            "candidate_tip_x": tip_x,
            "candidate_tip_y": tip_y,
            "candidate_source": source,
            "candidate_uncertain": uncertainty,
            "abstain_reason": abstain_reason,
        }
        study_instances.setdefault(sid, []).append(inst_data)

    print(f"Initial Instance Evaluation:")
    print(f"  Reviewed cases (retained as valid provisional): {reviewed_count}")
    print(f"  Frozen selector proposed (unreviewed):          {selector_proposed_count}")
    print(f"  Frozen selector abstained (unreviewed):         {selector_abstained_count}")

    # Second pass: Apply Multi-CVC Completeness Rule
    # For a study to have tip_supervision_active = True, EVERY CVC in that study must have candidate_uncertain == False
    manifest_records = []
    active_instances = 0
    masked_instances = 0
    active_studies = 0
    masked_studies = 0

    for sid, instances in study_instances.items():
        study_cvc_count = len(instances)
        # Check if EVERY instance has a valid, unambiguous candidate
        study_all_valid = all(not inst["candidate_uncertain"] for inst in instances)

        for inst in instances:
            rec = dict(inst)
            rec["total_cvc_in_study"] = study_cvc_count
            rec["study_all_cvc_valid"] = study_all_valid
            rec["seg_supervision_active"] = True  # Catheter path segmentation is 100% active

            if study_all_valid:
                rec["tip_supervision_active"] = True
                rec["tip_supervision_status"] = "ACTIVE"
                rec["final_tip_x"] = inst["candidate_tip_x"]
                rec["final_tip_y"] = inst["candidate_tip_y"]
                rec["reviewed_tip_x"] = inst["candidate_tip_x"]
                rec["reviewed_tip_y"] = inst["candidate_tip_y"]
                active_instances += 1
            else:
                rec["tip_supervision_active"] = False
                rec["final_tip_x"] = None
                rec["final_tip_y"] = None
                rec["reviewed_tip_x"] = None
                rec["reviewed_tip_y"] = None
                masked_instances += 1
                if study_cvc_count > 1 and not inst["candidate_uncertain"]:
                    rec["tip_supervision_status"] = "MASKED_INCOMPLETE_MULTI_CVC"
                else:
                    rec["tip_supervision_status"] = f"MASKED_{inst['candidate_source'].upper()}"

            manifest_records.append(rec)

        if study_all_valid:
            active_studies += 1
        else:
            masked_studies += 1

    manifest_df = pd.DataFrame(manifest_records)

    # Reconcile exact counts
    print("\n" + "=" * 70)
    print("RECONCILED EXPANSION MANIFEST SUMMARY")
    print("=" * 70)
    print(f"Total Positive Studies:       {len(study_instances)}")
    print(f"  Studies with ACTIVE Tips:   {active_studies} ({active_studies/len(study_instances)*100:.1f}%)")
    print(f"  Studies with MASKED Tips:   {masked_studies} ({masked_studies/len(study_instances)*100:.1f}%)")
    print(f"Total CVC Training Instances: {len(manifest_df)}")
    print(f"  Tip Supervision ACTIVE:     {active_instances} ({active_instances/len(manifest_df)*100:.1f}%)")
    print(f"  Tip Supervision MASKED:     {masked_instances} ({masked_instances/len(manifest_df)*100:.1f}%)")

    print("\nBreakdown by Final Tip Supervision Status:")
    print(manifest_df["tip_supervision_status"].value_counts())

    single_cvc = manifest_df[manifest_df["total_cvc_in_study"] == 1]
    multi_cvc = manifest_df[manifest_df["total_cvc_in_study"] > 1]
    print(f"\nSingle-CVC Studies (N={len(single_cvc)}): Active={single_cvc['tip_supervision_active'].sum()}, Masked={(~single_cvc['tip_supervision_active']).sum()}")
    print(f"Multi-CVC Studies  (N={len(multi_cvc)}): Active={multi_cvc['tip_supervision_active'].sum()}, Masked={(~multi_cvc['tip_supervision_active']).sum()}")

    # Save to data directory
    out_parquet = PROJECT_ROOT / "data" / "training_expansion_manifest.parquet"
    out_csv = PROJECT_ROOT / "data" / "training_expansion_manifest.csv"
    out_json = PROJECT_ROOT / "data" / "training_expansion_manifest.json"
    manifest_df.to_parquet(out_parquet, index=False)
    manifest_df.to_csv(out_csv, index=False)
    manifest_df.to_json(out_json, orient="records", indent=2)
    print(f"\nSaved reconciled manifest to:")
    print(f"  {out_parquet}")
    print(f"  {out_csv}")
    print(f"  {out_json}")

    return manifest_df

if __name__ == "__main__":
    build_manifest()
