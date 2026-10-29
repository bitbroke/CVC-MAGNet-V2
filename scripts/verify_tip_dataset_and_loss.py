import json
from pathlib import Path
import sys
import pandas as pd
import torch

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tip_dataset import CVCTipDataset
from src.models.tip_heatmap import CVCTipHeatmapNet, TipFocalLoss

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
catheter_df = pd.read_parquet(PROJECT_ROOT / "data" / "cvc_catheter_contracts.parquet")
study_sup_df = pd.read_parquet(PROJECT_ROOT / "data" / "cvc_study_supervision.parquet")

with open(PROJECT_ROOT / "data" / "expanded_training_cohort_manifest.json") as f:
    pos_ids = json.load(f)["positive_study_ids"][:10]

images_dir = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
ds = CVCTipDataset(
    study_ids=pos_ids,
    catheter_contracts_df=catheter_df,
    study_supervision_df=study_sup_df,
    images_dir=images_dir,
    img_size=768,
    is_train=True,
)

sample = ds[0]
print(f"Study: {sample['study_id']}")
print(f"Tip mask: {sample['tip_mask'].item()}")
print(f"Heatmap max: {sample['tip_heatmap'].max().item():.4f}")
print(f"Num tips: {sample['num_tips']}")

model = CVCTipHeatmapNet(pretrained=False)
loss_fn = TipFocalLoss()
out = model(sample["image"].unsqueeze(0))
loss = loss_fn(
    out["tip_probs"].float(),
    sample["tip_heatmap"].unsqueeze(0).float(),
    mask=sample["tip_mask"].view(1, 1, 1, 1),
)
print(f"Computed loss: {loss.item():.4f}")
print("SUCCESS: Tip supervision and loss verified!")
