"""
Re-evaluate saved T0 and T1 predictions against the corrected v2 benchmark manifest.
Evaluates:
- T0 Baseline
- T1 Joint HRNet (Seed 42)
- T1 Joint HRNet (Seed 43)

Reports:
- Primary Success@15, @5, @10, @25
- Single-CVC and Multi-CVC subgroups
- False tips per image (all studies and confirmed negatives)
- Matched 1-to-1 localization errors
- Nearest-prediction diagnostics (finite percentiles and distance bins)
Saves output to reports/t0_t1_v2_corrected_benchmark_metrics.json
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

EXP_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1")
BENCHMARK_V2_PATH = EXP_ROOT / "manifests" / "development_benchmark_manifest_v2_reviewed.json"
OUT_PATH = EXP_ROOT / "reports" / "t0_t1_v2_corrected_benchmark_metrics.json"

def match_study_tips_constrained(
    targets: List[Tuple[float, float]],
    preds: List[Dict[str, Any]],
    tolerance_px: float = 15.0,
) -> Dict[str, Any]:
    num_targets = len(targets)
    num_preds = len(preds)
    if num_targets == 0 or num_preds == 0:
        return {
            "num_targets": num_targets,
            "num_preds": num_preds,
            "matches": [],
            "matched_distances": [],
            "unmatched_target_indices": list(range(num_targets)),
            "unmatched_pred_indices": list(range(num_preds)),
            "target_nearest_distances": [float("inf")] * num_targets,
        }

    cost_matrix = np.full((num_targets, num_preds), 1e6, dtype=np.float64)
    dist_matrix = np.zeros((num_targets, num_preds), dtype=np.float64)

    for i, t in enumerate(targets):
        for j, p in enumerate(preds):
            d = math.hypot(t[0] - p["native_x"], t[1] - p["native_y"])
            dist_matrix[i, j] = d
            if d <= tolerance_px:
                cost_matrix[i, j] = d

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    matched_dists = []
    matched_targets = set()
    matched_preds = set()

    for r, c in zip(row_ind, col_ind):
        d = dist_matrix[r, c]
        if d <= tolerance_px:
            matches.append((int(r), int(c), float(d)))
            matched_dists.append(float(d))
            matched_targets.add(r)
            matched_preds.add(c)

    # Nearest prediction distance for each target
    target_nearest = np.min(dist_matrix, axis=1).tolist()

    return {
        "num_targets": num_targets,
        "num_preds": num_preds,
        "matches": matches,
        "matched_distances": matched_dists,
        "unmatched_target_indices": [i for i in range(num_targets) if i not in matched_targets],
        "unmatched_pred_indices": [j for j in range(num_preds) if j not in matched_preds],
        "target_nearest_distances": target_nearest,
    }


def evaluate_model_on_v2(
    manifest_v2: Dict[str, Any],
    preds_by_study: Dict[str, List[Dict[str, Any]]],
    model_name: str,
) -> Dict[str, Any]:
    studies = manifest_v2["studies"]
    total_studies = len(studies)
    neg_studies = [s for s in studies if s.get("is_negative", False)]
    pos_studies = [s for s in studies if not s.get("is_negative", False)]

    total_targets = 0
    total_matches_5 = 0
    total_matches_10 = 0
    total_matches_15 = 0
    total_matches_25 = 0
    total_fp_15 = 0
    neg_fp_15 = 0

    single_targets = 0
    single_matches_15 = 0
    multi_targets = 0
    multi_matches_15 = 0

    all_matched_dists_15 = []
    all_target_nearest_dists = []
    
    per_study_records = []

    for s in studies:
        sid = s["study_id"]
        is_neg = s.get("is_negative", False)
        preds = preds_by_study.get(sid, [])
        tips = s.get("assessable_tips", [])
        targets = [(float(t["x"]), float(t["y"])) for t in tips]
        num_t = len(targets)
        total_targets += num_t

        is_single = (num_t == 1)
        if num_t > 0:
            if is_single:
                single_targets += num_t
            else:
                multi_targets += num_t

        # Match at multiple tolerances
        m5 = match_study_tips_constrained(targets, preds, tolerance_px=5.0)
        m10 = match_study_tips_constrained(targets, preds, tolerance_px=10.0)
        m15 = match_study_tips_constrained(targets, preds, tolerance_px=15.0)
        m25 = match_study_tips_constrained(targets, preds, tolerance_px=25.0)

        total_matches_5 += len(m5["matches"])
        total_matches_10 += len(m10["matches"])
        total_matches_15 += len(m15["matches"])
        total_matches_25 += len(m25["matches"])

        if num_t > 0:
            if is_single:
                single_matches_15 += len(m15["matches"])
            else:
                multi_matches_15 += len(m15["matches"])

        all_matched_dists_15.extend(m15["matched_distances"])
        all_target_nearest_dists.extend(m15["target_nearest_distances"])

        fp = len(m15["unmatched_pred_indices"])
        total_fp_15 += fp
        if is_neg:
            neg_fp_15 += fp

        per_study_records.append({
            "study_id": sid,
            "is_negative": is_neg,
            "num_targets": num_t,
            "num_preds": len(preds),
            "matches_15": len(m15["matches"]),
            "false_positives_15": fp,
        })

    # Error quantiles for matched targets (1-to-1)
    median_matched = float(np.median(all_matched_dists_15)) if all_matched_dists_15 else None
    p75_matched = float(np.percentile(all_matched_dists_15, 75)) if all_matched_dists_15 else None
    p90_matched = float(np.percentile(all_matched_dists_15, 90)) if all_matched_dists_15 else None

    # Error quantiles for nearest prediction diagnostic across all targets (finite only)
    finite_nearest = [d for d in all_target_nearest_dists if not math.isinf(d)]
    median_nearest = float(np.median(finite_nearest)) if finite_nearest else None
    p75_nearest = float(np.percentile(finite_nearest, 75)) if finite_nearest else None
    p90_nearest = float(np.percentile(finite_nearest, 90)) if finite_nearest else None

    dist_bins = {
        "within_15px": sum(1 for d in all_target_nearest_dists if d <= 15.0),
        "15_to_20px": sum(1 for d in all_target_nearest_dists if 15.0 < d <= 20.0),
        "20_to_25px": sum(1 for d in all_target_nearest_dists if 20.0 < d <= 25.0),
        "25_to_70px": sum(1 for d in all_target_nearest_dists if 25.0 < d <= 70.0),
        "over_70px_finite": sum(1 for d in all_target_nearest_dists if 70.0 < d < float("inf")),
        "infinite_no_preds": sum(1 for d in all_target_nearest_dists if math.isinf(d)),
    }

    results = {
        "model_name": model_name,
        "total_targets": total_targets,
        "success_at_5px": total_matches_5 / total_targets,
        "success_at_10px": total_matches_10 / total_targets,
        "success_at_15px": total_matches_15 / total_targets,
        "success_at_25px": total_matches_25 / total_targets,
        "matches_count_15px": total_matches_15,
        "mean_false_tips_per_image": total_fp_15 / total_studies,
        "mean_false_tips_per_negative_image": neg_fp_15 / len(neg_studies),
        "total_false_tips_15px": total_fp_15,
        "negative_false_tips_15px": neg_fp_15,
        "single_cvc": {
            "targets": single_targets,
            "matches_15px": single_matches_15,
            "success_15px": single_matches_15 / single_targets if single_targets else 0.0,
        },
        "multi_cvc": {
            "targets": multi_targets,
            "matches_15px": multi_matches_15,
            "success_15px": multi_matches_15 / multi_targets if multi_targets else 0.0,
        },
        "matched_error_15px": {
            "median": median_matched,
            "p75": p75_matched,
            "p90": p90_matched,
            "count": len(all_matched_dists_15),
        },
        "nearest_pred_diagnostic": {
            "median_finite": median_nearest,
            "p75_finite": p75_nearest,
            "p90_finite": p90_nearest,
            "finite_count": len(finite_nearest),
            "infinite_count": sum(1 for d in all_target_nearest_dists if math.isinf(d)),
            "distance_bins": dist_bins,
        },
    }
    return results


def run_v2_evaluation():
    with open(BENCHMARK_V2_PATH, "r") as f:
        manifest_v2 = json.load(f)

    # 1. Load T0 Predictions
    with open(EXP_ROOT / "predictions" / "t0_baseline_predictions.json", "r") as f:
        raw_t0 = json.load(f)
    t0_preds_by_study = {}
    for sid, preds in raw_t0.items():
        t0_preds_by_study[sid] = [
            {"native_x": float(p["native_x"]), "native_y": float(p["native_y"]), "conf": float(p.get("conf", 1.0))}
            for p in preds
        ]

    # 2. Load T1 Seed 42 Predictions
    df_s42 = pd.read_csv(EXP_ROOT / "runs" / "seed_42" / "t1_joint_hrnet" / "decoded_candidates.csv")
    s42_preds_by_study = {}
    for sid, group in df_s42.groupby("study_id"):
        s42_preds_by_study[sid] = [
            {"native_x": float(r["native_x"]), "native_y": float(r["native_y"]), "conf": float(r["conf"])}
            for _, r in group.iterrows()
        ]

    # 3. Load T1 Seed 43 Predictions
    df_s43 = pd.read_csv(EXP_ROOT / "runs" / "seed_43" / "t1_joint_hrnet" / "decoded_candidates.csv")
    s43_preds_by_study = {}
    for sid, group in df_s43.groupby("study_id"):
        s43_preds_by_study[sid] = [
            {"native_x": float(r["native_x"]), "native_y": float(r["native_y"]), "conf": float(r["conf"])}
            for _, r in group.iterrows()
        ]

    # Evaluate all models
    res_t0 = evaluate_model_on_v2(manifest_v2, t0_preds_by_study, "T0_Baseline")
    res_s42 = evaluate_model_on_v2(manifest_v2, s42_preds_by_study, "T1_HRNet_Seed_42")
    res_s43 = evaluate_model_on_v2(manifest_v2, s43_preds_by_study, "T1_HRNet_Seed_43")

    report = {
        "benchmark_version": "v2_audited",
        "total_studies": manifest_v2["summary"]["total_studies"],
        "total_targets": manifest_v2["summary"]["total_assessable_targets"],
        "models": {
            "T0_Baseline": res_t0,
            "T1_Seed_42": res_s42,
            "T1_Seed_43": res_s43,
        }
    }

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("=== Re-evaluation against Corrected v2 Benchmark ===")
    for mname, mres in [("T0 Baseline", res_t0), ("T1 Seed 42", res_s42), ("T1 Seed 43", res_s43)]:
        print(f"\nModel: {mname}")
        print(f"  Success@15: {mres['success_at_15px']*100:.2f}% ({mres['matches_count_15px']}/{mres['total_targets']})")
        print(f"  Success@5:  {mres['success_at_5px']*100:.2f}% | Success@10: {mres['success_at_10px']*100:.2f}% | Success@25: {mres['success_at_25px']*100:.2f}%")
        print(f"  Single-CVC: {mres['single_cvc']['success_15px']*100:.2f}% ({mres['single_cvc']['matches_15px']}/{mres['single_cvc']['targets']})")
        print(f"  Multi-CVC:  {mres['multi_cvc']['success_15px']*100:.2f}% ({mres['multi_cvc']['matches_15px']}/{mres['multi_cvc']['targets']})")
        print(f"  False Tips/Image: {mres['mean_false_tips_per_image']:.2f} (All) | {mres['mean_false_tips_per_negative_image']:.2f} (Negatives)")
        print(f"  1-to-1 Matched Error (median, p75): {mres['matched_error_15px']['median']:.2f} px, {mres['matched_error_15px']['p75']:.2f} px")
        print(f"  Nearest Pred Diagnostic (median, p75): {mres['nearest_pred_diagnostic']['median_finite']:.2f} px, {mres['nearest_pred_diagnostic']['p75_finite']:.2f} px")
        print(f"  Distance Bins: {mres['nearest_pred_diagnostic']['distance_bins']}")

    print(f"\nSaved v2 metrics to: {OUT_PATH}")

if __name__ == "__main__":
    run_v2_evaluation()
