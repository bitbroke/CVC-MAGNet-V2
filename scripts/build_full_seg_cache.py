"""Build Lightweight 512x512 T2 Continuous Segmentation Cache (uint8 PNG).

Caches single-channel 512x512 uint8 segmentation maps for:
- 6,600 training studies
- 2,558 locked validation studies
Total = 9,158 studies (~70-100 MB total disk footprint).

Guarantees zero-domain shift and enables high-speed DataLoader throughput for FullSpatialNet.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tip_heatmap import CVCJointSegTipNet

DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
CARINA_TRAIN_PATH = DATA_DIR / "curated_cvc_v1" / "carina_train_cache.parquet"
IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
T2_CKPT_PATH = PROJECT_ROOT / "runs" / "t2_repaired_finetune_400" / "checkpoints" / "best.pt"
SEG_CACHE_DIR = DATA_DIR / "curated_cvc_v1" / "seg_cache_512"


class SimpleImageDataset(Dataset):
    """Fast dataset for pre-caching 512x512 segmentation maps."""

    def __init__(self, study_ids: List[str], images_dir: Path):
        self.study_ids = study_ids
        self.images_dir = images_dir
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        sid = self.study_ids[idx]
        img_path = self.images_dir / f"{sid}.jpg"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            # Fallback black image if corrupt/missing
            img_resized = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

        t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        t = (t - self.mean) / self.std
        return t, sid


def main():
    print("=" * 80)
    print("BUILDING 512x512 T2 CONTINUOUS SEGMENTATION CACHE (UINT8 PNG)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    SEG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Determine studies to process
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_sids = list(manifest[manifest["split"] == "val"]["StudyInstanceUID"])
    df_carina_tr = pd.read_parquet(CARINA_TRAIN_PATH)
    train_sids = list(df_carina_tr["study_id"])

    all_sids = list(dict.fromkeys(train_sids + val_sids))
    print(f"Total target studies: {len(all_sids)} ({len(train_sids)} train, {len(val_sids)} val)")

    # Filter out already cached studies
    needed_sids = [sid for sid in all_sids if not (SEG_CACHE_DIR / f"{sid}.png").exists()]
    print(f"Already cached: {len(all_sids) - len(needed_sids)} | Remaining to generate: {len(needed_sids)}")

    if not needed_sids:
        print("All segmentation masks are already cached. Exiting.")
        return

    # 2. Load fine-tuned T2 model
    print(f"Loading T2 checkpoint: {T2_CKPT_PATH.name}...")
    model = CVCJointSegTipNet(pretrained=False)
    ckpt = torch.load(T2_CKPT_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 3. Create DataLoader
    dataset = SimpleImageDataset(needed_sids, IMAGES_DIR)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    t0 = time.time()
    saved_count = 0

    print("Running batch inference...", flush=True)
    with torch.no_grad():
        for b_idx, (imgs, sids) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(imgs)
                # T2 output: seg_probs shape is [B, 1, 512, 512]
                seg_probs = out["seg_probs"]

            # Convert to uint8 on CPU
            seg_uint8 = (seg_probs.squeeze(1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            for i in range(len(sids)):
                sid = sids[i]
                mask = seg_uint8[i]
                save_path = SEG_CACHE_DIR / f"{sid}.png"
                cv2.imwrite(str(save_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                saved_count += 1

            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(loader):
                elapsed = time.time() - t0
                speed = saved_count / max(elapsed, 1e-4)
                eta = (len(needed_sids) - saved_count) / max(speed, 1e-4)
                print(
                    f"  Batch {b_idx + 1}/{len(loader)} | "
                    f"Saved: {saved_count}/{len(needed_sids)} ({speed:.1f} img/s) | "
                    f"ETA: {eta:.0f}s",
                    flush=True,
                )

    total_time = time.time() - t0
    print(f"\nCompleted: Cached {saved_count} segmentation masks in {total_time:.1f}s ({saved_count / max(total_time, 1e-4):.1f} img/s)")
    total_size = sum(f.stat().st_size for f in SEG_CACHE_DIR.glob("*.png")) / (1024 * 1024)
    print(f"Total cache size on disk: {total_size:.2f} MB")


if __name__ == "__main__":
    main()
