import sys
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Ankshit\CVC-MAGNet-V2")
manifest_path = ROOT / "data" / "splits" / "placement_locked_manifest.parquet"
df = pd.read_parquet(manifest_path)
print(f"Total manifest studies: {len(df)}")
print("Split counts:")
print(df["split"].value_counts())

seg_cache = ROOT / "data" / "curated_cvc_v1" / "seg_cache_512"
consensus_cache = ROOT / "data" / "curated_cvc_v1" / "consensus_seg_cache_512"

train_df = df[df["split"] == "train"]
sids = train_df["StudyInstanceUID"].tolist()

in_consensus = sum(1 for sid in sids if (consensus_cache / f"{sid}.png").exists())
in_seg = sum(1 for sid in sids if (seg_cache / f"{sid}.png").exists())
in_either = sum(1 for sid in sids if (consensus_cache / f"{sid}.png").exists() or (seg_cache / f"{sid}.png").exists())

print(f"\nTrain set studies: {len(sids)}")
print(f"  In consensus_seg_cache_512: {in_consensus}")
print(f"  In seg_cache_512:           {in_seg}")
print(f"  In either cache:            {in_either} ({in_either/len(sids)*100:.1f}%)")

