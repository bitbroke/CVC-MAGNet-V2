"""Distal-Tip Evaluation Metrics and Bipartite Matching.

Implements Section 12 of the Experiment Specification:
- Primary outcome: One-to-one Success@15 (assessable targets matched within 15 native pixels ÷ all assessable targets)
- Threshold-constrained bipartite matching prioritizing maximum valid matches, then distance
- Precision tolerances: Success@5, @10, @15, @25
- False tips per image (all images and confirmed negative images)
- Targets without nearby prediction (gross misses)
- Full error distribution across all targets (not just successful matches)
- Multi-CVC subgroup breakdown
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def match_study_tips(
    targets_native: List[Tuple[float, float]],
    preds_native: List[Dict[str, Any]],
    tolerance_px: float = 15.0,
) -> Dict[str, Any]:
    """Perform threshold-constrained bipartite matching for a single study.

    Args:
        targets_native: List of ground-truth (x, y) coordinates in native pixels.
        preds_native: List of prediction dicts containing 'native_x', 'native_y', 'conf'.
        tolerance_px: Maximum distance in native pixels to consider a match valid.

    Returns:
        Dictionary with matching results:
            - 'matches': List of (target_idx, pred_idx, distance_px)
            - 'num_targets': int
            - 'num_preds': int
            - 'num_matches': int
            - 'num_misses': int
            - 'num_false_positives': int
            - 'target_errors': List of distance to closest prediction for each target
    """
    num_t = len(targets_native)
    num_p = len(preds_native)

    if num_t == 0:
        return {
            "matches": [],
            "num_targets": 0,
            "num_preds": num_p,
            "num_matches": 0,
            "num_misses": 0,
            "num_false_positives": num_p,
            "target_errors": [],
        }

    if num_p == 0:
        return {
            "matches": [],
            "num_targets": num_t,
            "num_preds": 0,
            "num_matches": 0,
            "num_misses": num_t,
            "num_false_positives": 0,
            "target_errors": [float("inf")] * num_t,
        }

    # Build cost matrix
    T = np.array(targets_native, dtype=np.float64)  # [N, 2]
    P = np.array([[p["native_x"], p["native_y"]] for p in preds_native], dtype=np.float64)  # [M, 2]

    # Pairwise Euclidean distances
    diff = T[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))  # [N, M]

    # Closest prediction distance for each target (independent of matching)
    target_errors = np.min(dist_mat, axis=1).tolist()

    # Threshold-constrained cost: pairs > tolerance penalized heavily
    cost_mat = np.copy(dist_mat)
    ineligible = dist_mat > tolerance_px
    cost_mat[ineligible] = 1e6 + dist_mat[ineligible]

    row_ind, col_ind = linear_sum_assignment(cost_mat)

    valid_matches = []
    matched_preds = set()

    for r, c in zip(row_ind, col_ind):
        d = dist_mat[r, c]
        if d <= tolerance_px:
            valid_matches.append((int(r), int(c), float(d)))
            matched_preds.add(int(c))

    num_m = len(valid_matches)
    num_fp = num_p - num_m
    num_miss = num_t - num_m

    return {
        "matches": valid_matches,
        "num_targets": num_t,
        "num_preds": num_p,
        "num_matches": num_m,
        "num_misses": num_miss,
        "num_false_positives": num_fp,
        "target_errors": target_errors,
    }


def evaluate_cohort_tips(
    cohort_results: List[Dict[str, Any]],
    tolerances: Tuple[float, ...] = (5.0, 10.0, 15.0, 25.0),
) -> Dict[str, Any]:
    """Compute cohort-level tip localization metrics across tolerances.

    Args:
        cohort_results: List of dicts per study with:
            - 'study_id': str
            - 'is_negative': bool
            - 'num_cvcs': int
            - 'targets': List of (x, y)
            - 'predictions': List of prediction dicts

    Returns:
        Comprehensive metric dictionary.
    """
    total_studies = len(cohort_results)
    neg_studies = [r for r in cohort_results if r.get("is_negative", False)]
    pos_studies = [r for r in cohort_results if not r.get("is_negative", False)]

    total_targets = sum(len(r.get("targets", [])) for r in pos_studies)

    # Metrics per tolerance
    tolerance_metrics = {}
    for tau in tolerances:
        total_matches = 0
        total_fp = 0
        neg_fp = 0

        for r in cohort_results:
            targets = r.get("targets", [])
            preds = r.get("predictions", [])
            m_res = match_study_tips(targets, preds, tolerance_px=tau)

            total_matches += m_res["num_matches"]
            total_fp += m_res["num_false_positives"]
            if r.get("is_negative", False):
                neg_fp += m_res["num_false_positives"]

        succ_rate = total_matches / max(total_targets, 1)
        mean_fp_per_image = total_fp / max(total_studies, 1)
        mean_fp_neg_image = neg_fp / max(len(neg_studies), 1)

        tolerance_metrics[f"success_at_{int(tau)}px"] = succ_rate
        tolerance_metrics[f"total_matches_at_{int(tau)}px"] = total_matches
        tolerance_metrics[f"mean_false_tips_per_image_{int(tau)}px"] = mean_fp_per_image

    # Global distance distribution across all targets
    all_target_distances = []
    for r in pos_studies:
        targets = r.get("targets", [])
        preds = r.get("predictions", [])
        m_res = match_study_tips(targets, preds, tolerance_px=1000.0)  # unconstrained
        all_target_distances.extend(m_res["target_errors"])

    # Multi-CVC subgroup
    single_cvc_targets = 0
    single_cvc_matches_15 = 0
    multi_cvc_targets = 0
    multi_cvc_matches_15 = 0

    for r in pos_studies:
        num_c = len(r.get("targets", []))
        preds = r.get("predictions", [])
        m15 = match_study_tips(r.get("targets", []), preds, tolerance_px=15.0)

        if num_c == 1:
            single_cvc_targets += num_c
            single_cvc_matches_15 += m15["num_matches"]
        elif num_c > 1:
            multi_cvc_targets += num_c
            multi_cvc_matches_15 += m15["num_matches"]

    valid_dists = [d for d in all_target_distances if math.isfinite(d)]
    median_err = float(np.median(valid_dists)) if valid_dists else float("nan")
    p75_err = float(np.percentile(valid_dists, 75)) if valid_dists else float("nan")
    p90_err = float(np.percentile(valid_dists, 90)) if valid_dists else float("nan")

    return {
        "primary_success_at_15px": tolerance_metrics["success_at_15px"],
        "success_at_5px": tolerance_metrics["success_at_5px"],
        "success_at_10px": tolerance_metrics["success_at_10px"],
        "success_at_25px": tolerance_metrics["success_at_25px"],
        "total_targets_evaluated": total_targets,
        "total_studies_evaluated": total_studies,
        "positive_studies": len(pos_studies),
        "negative_studies": len(neg_studies),
        "mean_false_tips_per_image": tolerance_metrics["mean_false_tips_per_image_15px"],
        "mean_false_tips_per_negative_image": neg_fp / max(len(neg_studies), 1),
        "median_localization_error_px": median_err,
        "p75_localization_error_px": p75_err,
        "p90_localization_error_px": p90_err,
        "subgroup_single_cvc_success_15px": single_cvc_matches_15 / max(single_cvc_targets, 1),
        "subgroup_multi_cvc_success_15px": multi_cvc_matches_15 / max(multi_cvc_targets, 1),
        "single_cvc_targets": single_cvc_targets,
        "multi_cvc_targets": multi_cvc_targets,
        "tolerances_detail": tolerance_metrics,
    }
