"""Build Fixed Effective-Batch Ledger and Manifests for Architecture Comparison v1.

Locks exactly 1,000 updates (4,000 exposures) with identical image selection,
sampling category classification, augmentation parameters, and source image SHA-256 hashes.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
DATA_DIR = PROJECT_ROOT / "data"
EXP_DIR = PROJECT_ROOT / "experiments" / "architecture_comparison_v1"
MANIFEST_DIR = EXP_DIR / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=== Building Fixed Effective-Batch Ledger for Architecture Comparison v1 ===")

    # 1. Load S4 Reference Exposure Ledger
    s4_ledger_path = PROJECT_ROOT / "runs" / "s4_experiment_selective_hard_neg" / "exposure_ledger.jsonl"
    with open(s4_ledger_path, "r", encoding="utf-8") as f:
        raw_lines = [json.loads(line) for line in f if line.strip()]

    # Filter to updates 1 through 1000
    updates_1000 = [entry for entry in raw_lines if 1 <= entry.get("update", 0) <= 1000]
    print(f"Loaded {len(updates_1000)} updates from S4 ledger.")

    # Load top-99 hard negative IDs
    hard_neg_path = DATA_DIR / "selective_hard_negatives_manifest.json"
    with open(hard_neg_path, "r", encoding="utf-8") as f:
        hard_neg_data = json.load(f)
    hard_neg_set = set(hard_neg_data["study_ids"])
    print(f"Loaded {len(hard_neg_set)} selective hard-negative study IDs.")

    # Cache image hashes
    image_hash_cache: Dict[str, str] = {}
    
    unique_study_ids = set()
    for entry in updates_1000:
        for sid in entry["study_ids"]:
            unique_study_ids.add(sid)
    print(f"Total unique study IDs across 1,000 updates: {len(unique_study_ids)}")

    print("Computing SHA-256 hashes for source images...")
    missing_images = []
    for sid in unique_study_ids:
        img_path = IMAGES_DIR / f"{sid}.jpg"
        if img_path.exists():
            image_hash_cache[sid] = hash_file(img_path)
        else:
            missing_images.append(sid)
    
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} source images, e.g. {missing_images[:3]}")
    print(f"Hashed {len(image_hash_cache)} source images successfully.")

    # Build ledger
    fixed_ledger_entries: List[Dict[str, Any]] = []
    for entry in updates_1000:
        upd = entry["update"]
        sids = entry["study_ids"]
        is_pos_list = entry.get("is_positive", [])
        aug_params = entry.get("aug_params", [])

        sampling_cats = []
        for i, sid in enumerate(sids):
            is_pos = is_pos_list[i] if i < len(is_pos_list) else None
            if is_pos == 1 or (is_pos is None and sid not in hard_neg_set):
                sampling_cats.append("positive")
            elif sid in hard_neg_set:
                sampling_cats.append("selective_hard_negative")
            else:
                sampling_cats.append("uniform_negative")

        ledger_item = {
            "update_id": upd,
            "study_ids": sids,
            "sampling_category": sampling_cats,
            "augmentation_parameters": aug_params,
            "source_image_hashes": [image_hash_cache[sid] for sid in sids],
        }
        fixed_ledger_entries.append(ledger_item)

    # Save to manifests/batch_ledger.jsonl
    out_ledger_path = MANIFEST_DIR / "batch_ledger.jsonl"
    with open(out_ledger_path, "w", encoding="utf-8") as f:
        for item in fixed_ledger_entries:
            f.write(json.dumps(item) + "\n")
    print(f"Saved fixed ledger: {out_ledger_path} ({len(fixed_ledger_entries)} updates)")

    # Compute ledger hash
    ledger_sha256 = hash_file(out_ledger_path)

    manifest_hashes = {
        "batch_ledger_path": str(out_ledger_path),
        "batch_ledger_sha256": ledger_sha256,
        "total_updates": len(fixed_ledger_entries),
        "total_exposures": len(fixed_ledger_entries) * 4,
        "hard_negatives_manifest_sha256": hash_file(hard_neg_path),
        "splits_sha256": hash_file(DATA_DIR / "splits.json"),
    }
    with open(MANIFEST_DIR / "manifest_hashes.json", "w", encoding="utf-8") as f:
        json.dump(manifest_hashes, f, indent=2)
    print(f"Saved manifest hashes to {MANIFEST_DIR / 'manifest_hashes.json'}")
    print(f"Batch Ledger SHA-256: {ledger_sha256}")


if __name__ == "__main__":
    main()
