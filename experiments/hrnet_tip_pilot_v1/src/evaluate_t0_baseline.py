"""Evaluate Arm T0: Frozen HRNet Segmentation -> Existing Endpoint Extractor.

Establishes the current distal-tip localization baseline without any tip training.
- Model: Frozen Seed-42 HRNet (unmodified weights from architecture comparison)
- Inference: Native segmentation probability grid via Letterbox unpadding
- Endpoint Extractor: src.infer.endpoints.extract_distal_tips (morphological skeletonization)
- Benchmark: Locked development benchmark (25 assessable tips + 8 confirmed negative studies)
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "hrnet_tip_pilot_v1"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from src.data.transforms import LetterboxTransform
from src.infer.endpoints import extract_distal_tips
from experiments.architecture_comparison_v2.src.models.model_registry import build_model
from experiments.hrnet_tip_pilot_v1.src.evaluation import evaluate_cohort_tips

IMAGES_DIR = Path(r"C:\Users\M S I\Downloads\ranzcr-clip-catheter-line-classification\ranzcr-clip-catheter-line-classification\train")
DATA_DIR = PROJECT_ROOT / "data"
FROZEN_HRNET_CKPT = EXP_ROOT / "frozen_reference" / "checkpoints" / "hrnet_seed42_best.pt"


def run_t0_evaluation(benchmark_manifest_path: Path) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("=== RUNNING ARM T0: FROZEN HRNET + TOPOLOGICAL ENDPOINT EXTRACTOR ===")
    print("=" * 75 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen HRNet model
    print(f"Loading frozen HRNet checkpoint from {FROZEN_HRNET_CKPT}...")
    model = build_model("HRNet_W18", pretrained=False).to(device)
    ck = torch.load(FROZEN_HRNET_CKPT, map_location=device, weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model.load_state_dict(sd)
    model.eval()

    # Load benchmark manifest
    with open(benchmark_manifest_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    studies = benchmark["studies"]
    print(f"Locked Benchmark contains {len(studies)} studies ({benchmark['summary']['positive_studies']} positive, {benchmark['summary']['negative_studies']} negative, {benchmark['summary']['total_assessable_targets']} assessable tips).")

    transform = LetterboxTransform(target_size=(768, 768), clahe_enabled=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    cohort_results = []
    all_predictions_by_study = {}
    t_start = time.time()

    with torch.no_grad():
        for idx, s in enumerate(studies, 1):
            sid = s["study_id"]
            img_path = IMAGES_DIR / f"{sid}.jpg"
            if not img_path.exists():
                print(f"Warning: Image missing for study {sid}")
                continue

            pil_im = Image.open(img_path).convert("L")
            im_np = np.array(pil_im)
            orig_h, orig_w = im_np.shape

            lb_img, meta = transform.apply_image(im_np)
            raw_t = torch.from_numpy(lb_img).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0
            inp = (raw_t.repeat(1, 3, 1, 1) - mean) / std

            out = model(inp)
            logits = out["seg_logits"] if isinstance(out, dict) else out
            prob_map = torch.sigmoid(logits.float()).squeeze().cpu().numpy().astype(np.float32)

            valid_crop = prob_map[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w]
            prob_native = cv2.resize(valid_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # Apply topological skeletonization endpoint extractor
            candidates, diag = extract_distal_tips(prob_native, threshold=0.50, min_component_area=50)

            study_preds = [
                {
                    "native_x": float(c.x),
                    "native_y": float(c.y),
                    "conf": float(c.confidence),
                    "method": c.source_method,
                }
                for c in candidates
            ]
            all_predictions_by_study[sid] = study_preds

            targets = [(float(t["x"]), float(t["y"])) for t in s.get("assessable_tips", [])]

            cohort_results.append({
                "study_id": sid,
                "is_negative": s.get("is_negative", False),
                "num_cvcs": s.get("num_cvcs", len(targets)),
                "targets": targets,
                "predictions": study_preds,
            })

            if idx % 10 == 0 or idx == len(studies):
                print(f"Evaluated [{idx}/{len(studies)}] studies...", flush=True)

    elapsed = time.time() - t_start
    print(f"\nInference completed in {elapsed:.1f}s ({elapsed/len(studies):.2f}s/study).")

    # Compute comprehensive evaluation metrics
    metrics = evaluate_cohort_tips(cohort_results)
    metrics["arm"] = "T0_frozen_hrnet_endpoint_extractor"
    metrics["model_initialization"] = "frozen_hrnet_w18_seed42"
    metrics["checkpoint_sha256"] = ck.get("checkpoint_final_sha256", "")
    metrics["elapsed_time_sec"] = elapsed

    print("\n" + "=" * 65)
    print("=== T0 BASELINE EVALUATION OUTCOMES ===")
    print(f"Primary One-to-One Success@15px: {metrics['primary_success_at_15px']*100:.2f}%")
    print(f"Success@5px:  {metrics['success_at_5px']*100:.2f}%")
    print(f"Success@10px: {metrics['success_at_10px']*100:.2f}%")
    print(f"Success@25px: {metrics['success_at_25px']*100:.2f}%")
    print(f"Total Targets Evaluated: {metrics['total_targets_evaluated']}")
    print(f"Mean False Tips / Image: {metrics['mean_false_tips_per_image']:.3f}")
    print(f"Mean False Tips / Negative Image: {metrics['mean_false_tips_per_negative_image']:.3f}")
    print(f"Median Localization Error: {metrics['median_localization_error_px']:.1f} px (p75: {metrics['p75_localization_error_px']:.1f} px)")
    print(f"Single-CVC Success@15: {metrics['subgroup_single_cvc_success_15px']*100:.2f}% ({metrics['single_cvc_targets']} targets)")
    print(f"Multi-CVC Success@15:  {metrics['subgroup_multi_cvc_success_15px']*100:.2f}% ({metrics['multi_cvc_targets']} targets)")
    print("=" * 65 + "\n")

    # Save outputs
    rep_dir = EXP_ROOT / "reports"
    pred_dir = EXP_ROOT / "predictions"
    rep_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = rep_dir / "t0_baseline_metrics.json"
    preds_path = pred_dir / "t0_baseline_predictions.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions_by_study, f, indent=2)

    print(f"Saved T0 metrics to {metrics_path}")
    print(f"Saved T0 predictions to {preds_path}")

    return metrics


if __name__ == "__main__":
    benchmark_path = EXP_ROOT / "manifests" / "development_benchmark_manifest.json"
    run_t0_evaluation(benchmark_path)
