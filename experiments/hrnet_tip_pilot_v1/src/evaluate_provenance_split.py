"""Evaluate saved predictions partitioned by target provenance.

Evaluates:
1. Reviewed Targets Only (N = 25, provisional_engineering_review_v3)
2. Weak Heuristic Targets Only (N = 16, weak_heuristic_unreviewed)
3. Mixed Exploratory Cohort (N = 41, baseline development benchmark)

Meets Task 1 requirements:
- Distinguishes audited endpoints from weak heuristic endpoints
- Evaluates T0 baseline, T1 Seed 42, and T1 Seed 43
- Reports Success@5, @10, @15, @25, error distributions, and nearest-prediction diagnostics
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

EXP_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1")
BENCHMARK_PATH = EXP_ROOT / "manifests" / "development_benchmark_manifest.json"


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
            matches.append((r, c))
            matched_dists.append(d)
            matched_targets.add(r)
            matched_preds.add(c)

    return {
        "num_targets": num_targets,
        "num_preds": num_preds,
        "matches": matches,
        "matched_distances": matched_dists,
        "unmatched_target_indices": [i for i in range(num_targets) if i not in matched_targets],
        "unmatched_pred_indices": [j for j in range(num_preds) if j not in matched_preds],
    }


def evaluate_cohort(
    studies: List[Dict[str, Any]],
    predictions_by_study: Dict[str, List[Dict[str, Any]]],
    target_filter_status: Optional[str] = None,
) -> Dict[str, Any]:
    all_matched_dists_15 = []
    nearest_pred_dists = []
    total_targets = 0
    total_matches_5 = 0
    total_matches_10 = 0
    total_matches_15 = 0
    total_matches_25 = 0
    single_cvc_targets = 0
    single_cvc_matches_15 = 0
    multi_cvc_targets = 0
    multi_cvc_matches_15 = 0

    study_results = []

    for s in studies:
        sid = s["study_id"]
        is_neg = s.get("is_negative", False)
        preds = predictions_by_study.get(sid, [])
        all_tips = s.get("assessable_tips", [])

        if target_filter_status is not None:
            filtered_tips = [t for t in all_tips if t.get("review_status") == target_filter_status]
        else:
            filtered_tips = all_tips

        targets = [(float(t["x"]), float(t["y"])) for t in filtered_tips]
        num_t = len(targets)
        total_targets += num_t

        is_single = (len(all_tips) == 1)
        if is_single:
            single_cvc_targets += num_t
        else:
            multi_cvc_targets += num_t

        # Match at tolerances
        m5 = match_study_tips_constrained(targets, preds, tolerance_px=5.0)
        m10 = match_study_tips_constrained(targets, preds, tolerance_px=10.0)
        m15 = match_study_tips_constrained(targets, preds, tolerance_px=15.0)
        m25 = match_study_tips_constrained(targets, preds, tolerance_px=25.0)

        total_matches_5 += len(m5["matches"])
        total_matches_10 += len(m10["matches"])
        total_matches_15 += len(m15["matches"])
        total_matches_25 += len(m25["matches"])

        if is_single:
            single_cvc_matches_15 += len(m15["matches"])
        else:
            multi_cvc_matches_15 += len(m15["matches"])

        all_matched_dists_15.extend(m15["matched_distances"])

        # Nearest prediction diagnostic for each target
        for t in targets:
            if len(preds) > 0:
                min_d = min(math.hypot(t[0] - p["native_x"], t[1] - p["native_y"]) for p in preds)
            else:
                min_d = float("inf")
            nearest_pred_dists.append(min_d)

    # Error distribution among 1-to-1 matches at 15px
    matched_errors = all_matched_dists_15
    median_matched = float(np.median(matched_errors)) if matched_errors else None
    p75_matched = float(np.percentile(matched_errors, 75)) if matched_errors else None
    p90_matched = float(np.percentile(matched_errors, 90)) if matched_errors else None

    # Error distribution among nearest predictions for all targets (finite only)
    finite_nearest = [d for d in nearest_pred_dists if not math.isinf(d)]
    median_nearest = float(np.median(finite_nearest)) if finite_nearest else None
    p75_nearest = float(np.percentile(finite_nearest, 75)) if finite_nearest else None
    p90_nearest = float(np.percentile(finite_nearest, 90)) if finite_nearest else None

    # Distance bins for targets
    dist_bins = {
        "within_15px": sum(1 for d in nearest_pred_dists if d <= 15.0),
        "15_to_20px": sum(1 for d in nearest_pred_dists if 15.0 < d <= 20.0),
        "20_to_25px": sum(1 for d in nearest_pred_dists if 20.0 < d <= 25.0),
        "25_to_70px": sum(1 for d in nearest_pred_dists if 25.0 < d <= 70.0),
        "over_70px_finite": sum(1 for d in nearest_pred_dists if 70.0 < d < float("inf")),
        "infinite_no_preds": sum(1 for d in nearest_pred_dists if math.isinf(d)),
    }

    return {
        "total_targets": total_targets,
        "success_at_5px": float(total_matches_5 / total_targets) if total_targets else 0.0,
        "success_at_10px": float(total_matches_10 / total_targets) if total_targets else 0.0,
        "success_at_15px": float(total_matches_15 / total_targets) if total_targets else 0.0,
        "success_at_25px": float(total_matches_25 / total_targets) if total_targets else 0.0,
        "matches_count_15px": total_matches_15,
        "single_cvc_targets": single_cvc_targets,
        "single_cvc_matches_15px": single_cvc_matches_15,
        "single_cvc_success_15px": float(single_cvc_matches_15 / single_cvc_targets) if single_cvc_targets else 0.0,
        "multi_cvc_targets": multi_cvc_targets,
        "multi_cvc_matches_15px": multi_cvc_matches_15,
        "multi_cvc_success_15px": float(multi_cvc_matches_15 / multi_cvc_targets) if multi_cvc_targets else 0.0,
        "matched_error_15px": {
            "median": median_matched,
            "p75": p75_matched,
            "p90": p90_matched,
            "count": len(matched_errors),
        },
        "nearest_pred_diagnostic": {
            "median_finite": median_nearest,
            "p75_finite": p75_nearest,
            "p90_finite": p90_nearest,
            "finite_count": len(finite_nearest),
            "infinite_count": sum(1 for d in nearest_pred_dists if math.isinf(d)),
            "distance_bins": dist_bins,
        },
    }


def main():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bm = json.load(f)

    # Ingest T0 predictions
    with open(EXP_ROOT / "predictions" / "t0_baseline_predictions.json", "r", encoding="utf-8") as f:
        t0_data = json.load(f)
    if isinstance(t0_data, dict) and "studies" in t0_data:
        t0_by_study = {s["study_id"]: s["predictions"] for s in t0_data["studies"]}
    else:
        t0_by_study = t0_data

    # Ingest T1 Seed 42 candidates
    t1_s42_df = pd.read_csv(EXP_ROOT / "runs" / "seed_42" / "t1_joint_hrnet" / "decoded_candidates.csv")
    t1_s42_by_study = {}
    for sid, group in t1_s42_df.groupby("study_id"):
        t1_s42_by_study[sid] = [
            {"native_x": float(r["native_x"]), "native_y": float(r["native_y"]), "conf": float(r["conf"])}
            for _, r in group.iterrows()
        ]

    # Ingest T1 Seed 43 candidates
    t1_s43_df = pd.read_csv(EXP_ROOT / "runs" / "seed_43" / "t1_joint_hrnet" / "decoded_candidates.csv")
    t1_s43_by_study = {}
    for sid, group in t1_s43_df.groupby("study_id"):
        t1_s43_by_study[sid] = [
            {"native_x": float(r["native_x"]), "native_y": float(r["native_y"]), "conf": float(r["conf"])}
            for _, r in group.iterrows()
        ]

    models = {
        "T0_baseline": t0_by_study,
        "T1_seed_42": t1_s42_by_study,
        "T1_seed_43": t1_s43_by_study,
    }

    results = {}

    for m_name, preds_dict in models.items():
        res_reviewed = evaluate_cohort(
            bm["studies"], preds_dict, target_filter_status="provisional_engineering_review_v3"
        )
        res_weak = evaluate_cohort(
            bm["studies"], preds_dict, target_filter_status="weak_heuristic_unreviewed"
        )
        res_mixed = evaluate_cohort(
            bm["studies"], preds_dict, target_filter_status=None
        )

        results[m_name] = {
            "reviewed_targets_only": res_reviewed,
            "weak_heuristic_targets_only": res_weak,
            "mixed_exploratory_cohort": res_mixed,
        }

    out_path = EXP_ROOT / "reports" / "provenance_split_evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== TARGET PROVENANCE SPLIT EVALUATION COMPLETE ===")
    print(f"Results saved to {out_path}\n")

    for m_name in models:
        print(f"--- Model: {m_name} ---")
        rev = results[m_name]["reviewed_targets_only"]
        weak = results[m_name]["weak_heuristic_targets_only"]
        mix = results[m_name]["mixed_exploratory_cohort"]

        print(f"Reviewed Targets (N={rev['total_targets']}): Success@15 = {rev['success_at_15px']*100:.2f}% ({rev['matches_count_15px']}/{rev['total_targets']}) | S@10 = {rev['success_at_10px']*100:.2f}% | S@5 = {rev['success_at_5px']*100:.2f}% | S@25 = {rev['success_at_25px']*100:.2f}%")
        print(f"  Single-CVC (N={rev['single_cvc_targets']}): {rev['single_cvc_success_15px']*100:.2f}% ({rev['single_cvc_matches_15px']}/{rev['single_cvc_targets']}) | Multi-CVC (N={rev['multi_cvc_targets']}): {rev['multi_cvc_success_15px']*100:.2f}% ({rev['multi_cvc_matches_15px']}/{rev['multi_cvc_targets']})")
        print(f"Weak Heuristic (N={weak['total_targets']}):  Success@15 = {weak['success_at_15px']*100:.2f}% ({weak['matches_count_15px']}/{weak['total_targets']}) | S@10 = {weak['success_at_10px']*100:.2f}% | S@5 = {weak['success_at_5px']*100:.2f}% | S@25 = {weak['success_at_25px']*100:.2f}%")
        print(f"  Single-CVC (N={weak['single_cvc_targets']}): {weak['single_cvc_success_15px']*100:.2f}% ({weak['single_cvc_matches_15px']}/{weak['single_cvc_targets']}) | Multi-CVC (N={weak['multi_cvc_targets']}): {weak['multi_cvc_success_15px']*100:.2f}% ({weak['multi_cvc_matches_15px']}/{weak['multi_cvc_targets']})")
        print(f"Mixed Cohort   (N={mix['total_targets']}):  Success@15 = {mix['success_at_15px']*100:.2f}% ({mix['matches_count_15px']}/{mix['total_targets']}) | S@10 = {mix['success_at_10px']*100:.2f}% | S@5 = {mix['success_at_5px']*100:.2f}% | S@25 = {mix['success_at_25px']*100:.2f}%\n")


if __name__ == "__main__":
    main()
