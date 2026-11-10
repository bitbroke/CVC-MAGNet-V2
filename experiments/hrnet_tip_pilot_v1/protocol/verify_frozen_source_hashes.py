"""Verify that all frozen reference files have remained bitwise-identical throughout the experiment."""

import hashlib
import json
import sys
from pathlib import Path

EXP_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1")
SOURCE_HASHES_FILE = EXP_ROOT / "protocol" / "source_hashes.json"


def hash_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    with open(SOURCE_HASHES_FILE, "r", encoding="utf-8") as f:
        stored_hashes = json.load(f)

    print(f"Verifying {len(stored_hashes)} frozen reference files...")
    all_matched = True
    results = {}

    for name, info in stored_hashes.items():
        src_path = Path(info["source_path"])
        expected_sha = info["sha256"]
        expected_size = info["size_bytes"]

        if not src_path.exists():
            print(f"FAILED: {name} not found at {src_path}")
            all_matched = False
            results[name] = {"status": "MISSING", "path": str(src_path)}
            continue

        actual_sha = hash_file(src_path)
        actual_size = src_path.stat().st_size

        if actual_sha != expected_sha or actual_size != expected_size:
            print(f"FAILED: {name} MUTATED! Expected {expected_sha}, got {actual_sha}")
            all_matched = False
            results[name] = {
                "status": "MUTATED",
                "expected_sha": expected_sha,
                "actual_sha": actual_sha,
            }
        else:
            print(f"PASSED: {name} (size: {actual_size:,} B, sha: {actual_sha[:12]}...)")
            results[name] = {"status": "BITWISE_IDENTICAL", "sha256": actual_sha}

    summary = {
        "all_frozen_references_intact": all_matched,
        "verified_count": len(stored_hashes),
        "results": results,
    }

    out_file = EXP_ROOT / "protocol" / "source_hashes_post_verification.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    if all_matched:
        print("ALL FROZEN REFERENCES VERIFIED: ZERO MUTATION DETECTED.")
    else:
        print("WARNING: ONE OR MORE FROZEN REFERENCES MUTATED!")
    print("=" * 60)

    if not all_matched:
        sys.exit(1)


if __name__ == "__main__":
    main()
