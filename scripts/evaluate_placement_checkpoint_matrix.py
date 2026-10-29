"""Evaluate multi-label placement checkpoints on identical locked validation strata.

Never reads the locked test split. Reports rank and calibration metrics, avoiding
validation-fitted thresholds in the primary comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.placement_dataset import CVCPlacementDataset
from src.models.placement_baseline import CVCPlacementBaselineNet

LABELS = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def metric_block(y: np.ndarray, p: np.ndarray) -> dict:
    per_class = {}
    aucs, aps, briers = [], [], []
    for i, label in enumerate(LABELS):
        yi, pi = y[:, i], p[:, i]
        auc = float(roc_auc_score(yi, pi)) if np.unique(yi).size == 2 else None
        ap = float(average_precision_score(yi, pi)) if yi.sum() > 0 and yi.sum() < len(yi) else None
        brier = float(brier_score_loss(yi, pi))
        per_class[label] = {"support_positive": int(yi.sum()), "auroc": auc, "auprc": ap, "brier": brier}
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)
        briers.append(brier)
    return {
        "n": int(len(y)),
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "macro_auprc": float(np.mean(aps)) if aps else None,
        "macro_brier": float(np.mean(briers)) if briers else None,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/splits/placement_locked_manifest.parquet")
    parser.add_argument("--images-dir", default=r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
    parser.add_argument("--checkpoint", action="append", required=True, help="Repeat as name|backbone|path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    df = pd.read_parquet(manifest_path)
    val = df[df.split == "val"].copy()
    if val.empty or val.is_exposed.astype(bool).any():
        raise ValueError("Validation partition is empty or includes exposed studies")
    if val.StudyInstanceUID.duplicated().any():
        raise ValueError("Duplicate validation study IDs")
    dataset = CVCPlacementDataset(val, images_dir=args.images_dir, is_train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strata = {
        "all_locked_validation": np.ones(len(val), dtype=bool),
        "annotated_trace_any": val.annotated_cvc_count.to_numpy() >= 1,
        "no_annotated_trace": val.annotated_cvc_count.to_numpy() == 0,
        "one_annotated_trace": val.annotated_cvc_count.to_numpy() == 1,
        "multiple_annotated_traces": val.annotated_cvc_count.to_numpy() > 1,
        "multiple_image_labels": val[LABELS].sum(axis=1).to_numpy() > 1,
    }
    results = {"validation_count": len(val), "strata_counts": {k: int(v.sum()) for k, v in strata.items()}, "models": {}}

    for specification in args.checkpoint:
        try:
            name, backbone, checkpoint_arg = specification.split("|", 2)
        except ValueError as exc:
            raise ValueError("--checkpoint must be name|backbone|path") from exc
        checkpoint = Path(checkpoint_arg)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = CVCPlacementBaselineNet(backbone_name=backbone, pretrained=False, num_classes=3, dropout_rate=0.25)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device).eval()
        probabilities = []
        targets = []
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device, non_blocking=True)
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
                with amp:
                    logits = model(x)
                probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
                targets.append(batch["labels"].numpy())
        p, y = np.concatenate(probabilities), np.concatenate(targets)
        model_result = {"checkpoint": str(checkpoint), "checkpoint_update": payload.get("update"), "metrics": {}}
        for group, mask in strata.items():
            if mask.any():
                model_result["metrics"][group] = metric_block(y[mask], p[mask])
        results["models"][name] = model_result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(out), "strata_counts": results["strata_counts"], "models": list(results["models"])}, indent=2))


if __name__ == "__main__":
    main()
