"""Master Orchestration and Comparative Synthesis for Three-Model Pilot.

Models evaluated:
1. EfficientNet_B0 (Baseline: ImageNet pretrained encoder + fresh segmentation head)
2. SegFormer_B0 (MiT-B0 ImageNet pretrained encoder + fresh segmentation head)
3. HRNet_W18 (HRNet-W18 ImageNet pretrained encoder + fresh multi-resolution fusion head)
Reference:
- Frozen S4 (Canonical trained checkpoint evaluated under identical protocol)

Protocol:
- Resolution: 768x768 letterbox
- Effective batch: 4
- Budget: 1,000 updates
- Seed: 42
- Dataset: 400 positives + 396 negatives (from immutable batch_ledger.jsonl)
- Evaluation: Canonical parity validator on dev_pilot_val (193 studies) at steps 0, 250, 500, 750, 1000.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "architecture_comparison_v2"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from experiments.architecture_comparison_v2.scripts.train_runner import (
    run_training_pilot,
    CanonicalParityValidator,
    seed_everything,
    IMAGES_DIR,
    DATA_DIR,
)
from experiments.architecture_comparison_v2.src.models.model_registry import (
    build_model,
    FROZEN_S4_CKPT,
)

MODELS = ["EfficientNet_B0", "SegFormer_B0", "HRNet_W18"]


def run_sequential_pilots(
    models: List[str] = MODELS,
    target_updates: int = 1000,
    eval_interval: int = 250,
    seed: int = 42,
    overwrite: bool = False,
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print(f"=== COMMENCING THREE-MODEL PILOT EXECUTION (Seed {seed}, {target_updates} Updates) ===")
    print("=" * 75 + "\n")

    results: Dict[str, Any] = {}
    total_start = time.time()

    for idx, model_name in enumerate(models, 1):
        print(f"\n{'#' * 75}")
        print(f"### [{idx}/{len(models)}] Starting Pilot for {model_name} ###")
        print(f"{'#' * 75}\n")

        run_dir = EXP_ROOT / "runs" / f"seed_{seed}" / model_name.lower()
        final_ckpt = run_dir / "checkpoints" / "final_model.pt"

        # Check if already completed
        if final_ckpt.exists() and not overwrite:
            print(f"[{model_name}] Found existing completed final checkpoint at {final_ckpt}.")
            print(f"[{model_name}] Loading historical results to verify integrity...")
            history_file = run_dir / "history.jsonl"
            history = []
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    history = [json.loads(line) for line in f if line.strip()]

            # Load checkpoint to get metrics
            ck = torch.load(final_ckpt, map_location="cpu", weights_only=False)
            results[model_name] = {
                "model_name": model_name,
                "seed": seed,
                "total_updates": ck.get("update", target_updates),
                "peak_vram_mb": ck.get("peak_vram_mb", 0.0),
                "best_update": ck.get("best_step", -1),
                "best_macro_dice": ck.get("best_dice", -1.0),
                "final_metrics": history[-1].get("val_metrics") if history else None,
                "history": history,
            }
            print(f"[{model_name}] Skipped re-training (already complete). Best Dice: {ck.get('best_dice', 0):.4f}")
            continue

        # Clean VRAM before running
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t_m_start = time.time()
        res = run_training_pilot(
            model_name=model_name,
            target_updates=target_updates,
            eval_interval=eval_interval,
            seed=seed,
            output_dir=str(run_dir),
            overwrite=overwrite,
        )
        res["execution_wall_time_sec"] = time.time() - t_m_start
        results[model_name] = res

        print(f"\n[Completed {model_name}] Wall time: {res['execution_wall_time_sec']/60:.1f} min\n")

        # Cleanup memory after run
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_wall = time.time() - total_start
    print("\n" + "=" * 75)
    print(f"=== ALL THREE PILOTS COMPLETED IN {total_wall/60:.1f} MINUTES ===")
    print("=" * 75 + "\n")

    return results


def evaluate_and_compare(seed: int = 42) -> Dict[str, Any]:
    print("\n=== Compiling Comprehensive Architecture Comparison Report ===")
    from src.models.segmentation_v3 import CVCSegmentationV3Net

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validator = CanonicalParityValidator(target_size=768, device=device)

    # 1. Evaluate Frozen Reference Checkpoint
    print("\n--- Evaluating Frozen S4 Reference Checkpoint ---")
    frozen_model = CVCSegmentationV3Net(pretrained=False).to(device)
    ck_frozen = torch.load(FROZEN_S4_CKPT, map_location=device, weights_only=False)
    frozen_sd = ck_frozen["model_state_dict"] if "model_state_dict" in ck_frozen else ck_frozen
    frozen_model.load_state_dict(frozen_sd)
    frozen_model.eval()

    eval_frozen = validator.evaluate_model(frozen_model, threshold=0.50)
    frozen_metrics = eval_frozen["summary"]

    # 2. Evaluate Final Checkpoints of All 3 Models
    model_evals: Dict[str, Dict[str, Any]] = {"Frozen_S4_Ref": frozen_metrics}
    model_instances: Dict[str, torch.nn.Module] = {"Frozen_S4_Ref": frozen_model}

    for model_name in MODELS:
        run_dir = EXP_ROOT / "runs" / f"seed_{seed}" / model_name.lower()
        final_ckpt = run_dir / "checkpoints" / "final_model.pt"
        best_ckpt = run_dir / "checkpoints" / "best_model.pt"

        eval_ckpt = best_ckpt if best_ckpt.exists() else final_ckpt
        print(f"\n--- Evaluating Best/Final Checkpoint for {model_name} ({eval_ckpt.name}) ---")

        m = build_model(model_name, pretrained=False).to(device)
        ck = torch.load(eval_ckpt, map_location=device, weights_only=False)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        m.load_state_dict(sd)
        m.eval()

        res = validator.evaluate_model(m, threshold=0.50)
        sum_m = res["summary"]
        sum_m["eval_checkpoint"] = eval_ckpt.name
        sum_m["best_step"] = ck.get("best_step", ck.get("update", 1000))
        sum_m["peak_vram_mb"] = ck.get("peak_vram_mb", 0.0)

        # Retrieve training time and history
        hist_file = run_dir / "history.jsonl"
        if hist_file.exists():
            with open(hist_file, "r") as f:
                hist = [json.loads(l) for l in f if l.strip()]
            sum_m["history"] = hist

        model_evals[model_name] = sum_m
        model_instances[model_name] = m

    # 3. Generate Comparative Metrics Table
    print("\n" + "=" * 105)
    print(f"{'Model':<20} | {'Macro Dice':<10} | {'Cov@10px':<10} | {'Miss Rate':<10} | {'False Area%':<11} | {'Prec':<7} | {'Recall':<7} | {'VRAM(MB)':<8}")
    print("-" * 105)
    for name, m in model_evals.items():
        dice = m["macro_dice"]
        cov = m["centerline_coverage_10px"] * 100
        miss = m["missed_catheter_rate"]
        fa = m["negative_false_area_pct"]
        prec = m["precision"]
        rec = m["recall"]
        vram = m.get("peak_vram_mb", 0.0)
        print(f"{name:<20} | {dice:<10.4f} | {cov:<9.2f}% | {miss:<9.2f}% | {fa:<10.4f}% | {prec:<7.4f} | {rec:<7.4f} | {vram:<8.0f}")
    print("=" * 105 + "\n")

    # 4. Generate Visual Overlays for Qualitative Inspection
    generate_comparative_overlays(model_instances, validator)

    # 5. Save Summary Report
    rep_dir = EXP_ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    summary_path = rep_dir / f"pilot_comparison_seed{seed}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        # Don't save raw history inside json summary to keep clean
        clean_evals = {k: {sk: sv for sk, sv in v.items() if sk != "history"} for k, v in model_evals.items()}
        json.dump(clean_evals, f, indent=2)

    return model_evals


def generate_comparative_overlays(
    model_instances: Dict[str, torch.nn.Module],
    validator: CanonicalParityValidator,
    num_pos: int = 3,
    num_neg: int = 2,
):
    print("\n--- Generating Qualitative Comparative Image Overlays ---")
    vis_dir = EXP_ROOT / "reports" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Select representative studies
    pos_studies = [s for s in validator.val_study_ids if s in validator.polylines_by_study][:num_pos]
    neg_studies = [s for s in validator.val_study_ids if s not in validator.polylines_by_study][:num_neg]
    selected_studies = [(s, True) for s in pos_studies] + [(s, False) for s in neg_studies]

    font = ImageFont.load_default()

    for idx, (sid, is_pos) in enumerate(selected_studies, 1):
        img_path = IMAGES_DIR / f"{sid}.jpg"
        pil_orig = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_orig.size
        im_np = np.array(Image.open(img_path).convert("L"))

        # Create GT Overlay
        gt_img = pil_orig.copy()
        draw_gt = ImageDraw.Draw(gt_img)
        if is_pos and sid in validator.polylines_by_study:
            for poly in validator.polylines_by_study[sid]:
                pts = [(int(p[0]), int(p[1])) for p in poly]
                if len(pts) >= 2:
                    draw_gt.line(pts, fill=(0, 255, 0), width=6)
        draw_gt.text((20, 20), f"Study: {sid[:12]}..\nGT {'POS' if is_pos else 'NEG'}", fill=(255, 255, 255), font=font)

        # Get Predictions from all models
        pred_panels = [gt_img]
        panel_titles = ["Ground Truth"]

        # Run model inference
        for m_name, model in model_instances.items():
            lb_img, meta = validator.transform.apply_image(im_np)
            raw_t = torch.from_numpy(lb_img).unsqueeze(0).unsqueeze(0).to(validator.device).float() / 255.0
            inp = (raw_t.repeat(1, 3, 1, 1) - validator.mean) / validator.std

            with torch.no_grad():
                out = model(inp)
                logits = out["seg_logits"] if isinstance(out, dict) else out
                prob_map = torch.sigmoid(logits.float()).squeeze().cpu().numpy().astype(np.float32)

            valid_crop = prob_map[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w]
            native_prob = cv2.resize(valid_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask_bin = (native_prob >= 0.50).astype(np.uint8)

            # Create prediction overlay
            overlay = pil_orig.copy()
            overlay_np = np.array(overlay)
            # Magenta highlight for prediction
            overlay_np[mask_bin > 0, 0] = np.clip(overlay_np[mask_bin > 0, 0].astype(int) + 180, 0, 255)
            overlay_np[mask_bin > 0, 2] = np.clip(overlay_np[mask_bin > 0, 2].astype(int) + 220, 0, 255)

            panel_img = Image.fromarray(overlay_np)
            draw_p = ImageDraw.Draw(panel_img)
            pred_area = int(mask_bin.sum())
            draw_p.text((20, 20), f"{m_name}\nPred px: {pred_area:,}", fill=(255, 255, 0), font=font)

            pred_panels.append(panel_img)
            panel_titles.append(m_name)

        # Concatenate horizontally
        thumb_h = 512
        resized_panels = []
        for p in pred_panels:
            aspect = p.width / p.height
            thumb_w = int(thumb_h * aspect)
            resized_panels.append(p.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR))

        total_w = sum(p.width for p in resized_panels)
        composite = Image.new("RGB", (total_w, thumb_h + 40), (20, 20, 20))
        draw_comp = ImageDraw.Draw(composite)

        curr_x = 0
        for p, title in zip(resized_panels, panel_titles):
            composite.paste(p, (curr_x, 40))
            draw_comp.text((curr_x + 10, 10), title, fill=(255, 255, 255), font=font)
            curr_x += p.width

        out_fn = vis_dir / f"case_{idx:02d}_{'pos' if is_pos else 'neg'}_{sid[-12:]}.png"
        composite.save(out_fn)
        print(f"Saved overlay to {out_fn}")


def main():
    parser = argparse.ArgumentParser(description="Three-Model Pilot Runner and Evaluator")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    if not args.eval_only:
        run_sequential_pilots(
            models=MODELS,
            target_updates=args.updates,
            eval_interval=args.eval_interval,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    evaluate_and_compare(seed=args.seed)


if __name__ == "__main__":
    main()
