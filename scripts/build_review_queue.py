"""Generate targeted review queue manifest for the Endpoint A vs B visual review interface.

Triage priority:
1. Multi-CVC studies in dev_pilot_train (13 studies) and dev_pilot_val (14 studies) that have
   partially reviewed lines. Reviewing their unreviewed CVC instances immediately unlocks all 27 studies!
2. Border distance ambiguous cases (|dist_A - dist_B| < 60 px).
3. Representative sample of single-CVC cases from dev_pilot_train and dev_pilot_val.
"""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd

def build_review_queue():
    root = Path(r"C:\Ankshit\CVC-MAGNet-V2")
    study_df = pd.read_parquet(root / "data" / "study_inventory.parquet")
    inv_df = pd.read_parquet(root / "data" / "annotation_inventory.parquet")
    reviewed_v2 = pd.read_parquet(root / "data" / "reviewed_distal_tips_v2.parquet")
    with open(root / "data" / "splits.json") as f:
        splits = json.load(f)

    pilot_train_sids = set(splits["partitions"]["dev_pilot_train"]["study_ids"])
    pilot_val_sids = set(splits["partitions"]["dev_pilot_val"]["study_ids"])
    pilot_sids = pilot_train_sids.union(pilot_val_sids)

    # Filter to CVC instances in pilot cohort
    cvc_inv = inv_df[(inv_df["device_type"] == "CVC") & (inv_df["study_id"].isin(pilot_sids))].copy()
    
    # Already reviewed keys in v2
    already_reviewed = set(zip(reviewed_v2["study_id"], reviewed_v2["catheter_instance_id"]))

    # Triage Group 1: Partially reviewed multi-CVC studies
    study_cvc_counts = cvc_inv.groupby("study_id")["catheter_instance_id"].nunique().to_dict()
    study_rev_counts = reviewed_v2.groupby("study_id")["catheter_instance_id"].nunique().to_dict()

    group1_sids = set()
    for sid, tot in study_cvc_counts.items():
        rev = study_rev_counts.get(sid, 0)
        if 0 < rev < tot:
            group1_sids.add(sid)

    # Triage Group 2: Border distance ambiguous cases (|dist_A - dist_B| < 60 px)
    queue_cases = []

    for _, row in cvc_inv.iterrows():
        sid = row["study_id"]
        cid = int(row["catheter_instance_id"])
        pts = np.asarray(json.loads(row["polyline"]), dtype=np.float32)
        w, h = float(row["image_width"]), float(row["image_height"])

        pt_a = pts[0].tolist()  # (x, y) first point
        pt_b = pts[-1].tolist() # (x, y) last point

        def bdist(p):
            return min(p[0], w - p[0], p[1], h - p[1])

        bdist_a = float(bdist(pt_a))
        bdist_b = float(bdist(pt_b))
        diff_bdist = abs(bdist_a - bdist_b)

        # Check existing review if any
        is_reviewed = (sid, cid) in already_reviewed
        rev_row = reviewed_v2[(reviewed_v2["study_id"] == sid) & (reviewed_v2["catheter_instance_id"] == cid)]
        current_decision = None
        if len(rev_row) > 0:
            current_decision = str(rev_row.iloc[0]["reviewed_endpoint_class"])

        split = "dev_pilot_train" if sid in pilot_train_sids else "dev_pilot_val"
        is_partial_multi = sid in group1_sids
        is_ambiguous = diff_bdist < 60.0

        # Heuristic tip index
        heur_tip_idx = int(row["tip_endpoint_index"])
        heur_choice = "A" if heur_tip_idx == 0 else "B"

        priority = 99
        priority_reason = "general_pilot"
        if is_partial_multi and not is_reviewed:
            priority = 1
            priority_reason = "unlock_multi_cvc_study"
        elif is_ambiguous and not is_reviewed:
            priority = 2
            priority_reason = "border_distance_ambiguous (<60px)"
        elif not is_reviewed:
            priority = 3
            priority_reason = "unreviewed_pilot"
        else:
            priority = 10
            priority_reason = "already_reviewed_in_v2"

        queue_cases.append({
            "study_id": sid,
            "catheter_instance_id": cid,
            "instance_key": f"{sid}_{cid}",
            "split": split,
            "device_label": str(row["label"]),
            "image_width": int(w),
            "image_height": int(h),
            "num_points": len(pts),
            "polyline": pts.tolist(),
            "endpoint_a": [round(pt_a[0], 1), round(pt_a[1], 1)],
            "endpoint_b": [round(pt_b[0], 1), round(pt_b[1], 1)],
            "bdist_a": round(bdist_a, 1),
            "bdist_b": round(bdist_b, 1),
            "diff_bdist": round(diff_bdist, 1),
            "heuristic_choice": heur_choice,
            "priority": priority,
            "priority_reason": priority_reason,
            "is_reviewed": is_reviewed,
            "current_decision": current_decision,
            "total_cvc_in_study": study_cvc_counts.get(sid, 1),
        })

    # Sort queue: Priority 1 first (unlock multi-CVC), then Priority 2 (ambiguous), then Priority 3 (unreviewed)
    queue_cases.sort(key=lambda x: (x["priority"], x["split"], x["diff_bdist"]))

    # Tag 1-based index
    for i, c in enumerate(queue_cases, 1):
        c["queue_index"] = i

    out_file = root / "data" / "review_queue.json"
    with open(out_file, "w") as f:
        json.dump(queue_cases, f, indent=2)

    p1 = sum(1 for c in queue_cases if c["priority"] == 1)
    p2 = sum(1 for c in queue_cases if c["priority"] == 2)
    p3 = sum(1 for c in queue_cases if c["priority"] == 3)
    p_rev = sum(1 for c in queue_cases if c["is_reviewed"])

    print(f"Generated review queue: {len(queue_cases)} cases -> {out_file}")
    print(f"  Priority 1 (Unlock Multi-CVC Studies): {p1} cases")
    print(f"  Priority 2 (Ambiguous Border Distance < 60px): {p2} cases")
    print(f"  Priority 3 (Other Unreviewed Pilot CVCs): {p3} cases")
    print(f"  Already Reviewed in V2: {p_rev} cases (included for reference/audit)")

if __name__ == "__main__":
    build_review_queue()
