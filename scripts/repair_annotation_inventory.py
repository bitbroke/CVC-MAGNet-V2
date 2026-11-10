"""Repair Catheter Annotation Inventory and Contracts.

Replaces the naive `distance_to_image_border` CVC endpoint resolution with the
anatomically grounded vascular corridor / mediality resolution rule.
Updates both `annotation_inventory.parquet` and `cvc_catheter_contracts.parquet`.
"""

import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tip_rules import resolve_catheter_tip

DATA_DIR = PROJECT_ROOT / "data"
INV_PATH = DATA_DIR / "annotation_inventory.parquet"
CONTRACTS_PATH = DATA_DIR / "cvc_catheter_contracts.parquet"
CARINA_CACHE_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"


def main():
    print("=== Repairing Annotation Inventory & Catheter Contracts ===")

    # 1. Backups
    inv_backup = DATA_DIR / "annotation_inventory_v1_backup.parquet"
    contracts_backup = DATA_DIR / "cvc_catheter_contracts_v1_backup.parquet"

    if not inv_backup.exists():
        shutil.copy2(INV_PATH, inv_backup)
        print(f"Created backup: {inv_backup}")
    if not contracts_backup.exists():
        shutil.copy2(CONTRACTS_PATH, contracts_backup)
        print(f"Created backup: {contracts_backup}")

    df_inv = pd.read_parquet(INV_PATH)
    df_contracts = pd.read_parquet(CONTRACTS_PATH)

    carina_map = {}
    if CARINA_CACHE_PATH.exists():
        df_carina = pd.read_parquet(CARINA_CACHE_PATH)
        for _, r in df_carina.iterrows():
            carina_map[r["study_id"]] = (float(r["carina_x_native"]), float(r["carina_y_native"]))

    print(f"Loaded {len(df_inv)} inventory instances and {len(df_contracts)} contract instances.")
    print(f"Carina cache available for {len(carina_map)} studies.")

    # 2. Update Inventory
    inv_flips = 0
    flips_by_label = {}

    new_tip_x = []
    new_tip_y = []
    new_entry_x = []
    new_entry_y = []
    new_tip_idx = []
    new_rule = []
    new_ann_source = []

    for _, row in df_inv.iterrows():
        dev_type = row["device_type"]
        label = str(row.get("label", ""))
        source = str(row.get("annotation_source", ""))
        sid = row["study_id"]
        w = int(row["image_width"])
        h = int(row["image_height"])

        # Leave expert-verified and unassessable alone
        if "expert" in source or source == "clinical_expert_verified":
            new_tip_x.append(row["tip_x"])
            new_tip_y.append(row["tip_y"])
            new_entry_x.append(row.get("entry_x", np.nan))
            new_entry_y.append(row.get("entry_y", np.nan))
            new_tip_idx.append(row.get("tip_endpoint_index", -1))
            new_rule.append("expert_verified")
            new_ann_source.append(source)
            continue

        raw_pts = json.loads(row["polyline"]) if isinstance(row["polyline"], str) else row["polyline"]
        pts_np = np.asarray(raw_pts, dtype=np.float32)

        carina_pt = carina_map.get(sid, (0.4805 * w, 0.4000 * h))
        res = resolve_catheter_tip(pts_np, dev_type, (h, w), carina_coord=carina_pt)

        old_idx = row.get("tip_endpoint_index")
        tip_idx = res["tip_idx"]

        if old_idx is not None and tip_idx != old_idx:
            inv_flips += 1
            flips_by_label[label] = flips_by_label.get(label, 0) + 1

        if res["is_assessable"]:
            new_tip_x.append(float(res["tip"][0]))
            new_tip_y.append(float(res["tip"][1]))
            new_entry_x.append(float(res["entry"][0]))
            new_entry_y.append(float(res["entry"][1]))
            new_tip_idx.append(tip_idx)
            new_rule.append(res["rule"])
            new_ann_source.append("heuristic_anatomical_corridor")
        else:
            new_tip_x.append(np.nan)
            new_tip_y.append(np.nan)
            new_entry_x.append(float(res["entry"][0]))
            new_entry_y.append(float(res["entry"][1]))
            new_tip_idx.append(tip_idx)
            new_rule.append(res["rule"])
            new_ann_source.append("unassessable_crop")

    df_inv["tip_x"] = new_tip_x
    df_inv["tip_y"] = new_tip_y
    df_inv["entry_x"] = new_entry_x
    df_inv["entry_y"] = new_entry_y
    df_inv["tip_endpoint_index"] = new_tip_idx
    df_inv["rule"] = new_rule
    df_inv["annotation_source"] = new_ann_source

    print(f"\nInventory repair complete: {inv_flips} total catheter endpoints corrected.")
    print("Corrections by label:")
    for lbl, count in sorted(flips_by_label.items()):
        print(f"  {lbl}: {count}")

    # 3. Update Contracts
    # Map from (study_id, instance_id)
    inv_map = df_inv.set_index(["study_id", "catheter_instance_id"]).to_dict("index")

    c_tip_x = []
    c_tip_y = []
    c_tip_source = []
    c_tip_state = []

    for _, crow in df_contracts.iterrows():
        key = (crow["study_id"], crow["instance_id"])
        inv_item = inv_map.get(key)
        if inv_item is not None:
            c_tip_x.append(inv_item["tip_x"])
            c_tip_y.append(inv_item["tip_y"])
            c_tip_source.append(inv_item["annotation_source"])
            c_tip_state.append("assessable" if pd.notna(inv_item["tip_x"]) else "unassessable")
        else:
            c_tip_x.append(crow["tip_x"])
            c_tip_y.append(crow["tip_y"])
            c_tip_source.append(crow.get("tip_label_source", "unknown"))
            c_tip_state.append(crow.get("tip_state", "assessable"))

    df_contracts["tip_x"] = c_tip_x
    df_contracts["tip_y"] = c_tip_y
    df_contracts["tip_label_source"] = c_tip_source
    df_contracts["tip_state"] = c_tip_state

    # 4. Save
    v2_inv_path = DATA_DIR / "annotation_inventory_v2.parquet"
    df_inv.to_parquet(v2_inv_path, index=False)
    df_inv.to_parquet(INV_PATH, index=False)
    print(f"Saved repaired inventory to {v2_inv_path} and {INV_PATH}")

    v2_contracts_path = DATA_DIR / "cvc_catheter_contracts_v2.parquet"
    df_contracts.to_parquet(v2_contracts_path, index=False)
    df_contracts.to_parquet(CONTRACTS_PATH, index=False)
    print(f"Saved repaired contracts to {v2_contracts_path} and {CONTRACTS_PATH}")


if __name__ == "__main__":
    main()

