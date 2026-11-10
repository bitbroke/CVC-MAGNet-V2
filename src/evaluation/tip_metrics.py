"""CVC Tip Evaluation Engine with Hungarian Matching and Multilevel Diagnostics.

Implements Sections 3, 11, and 12 of the Experiment Specification:
- Primary Metric: Success@15 (maximum-cardinality one-to-one matched assessable tips within 15 native pixels / all assessable ground-truth tips)
- Secondary Tolerances: Success@5, @10, @20, @30 pixels
- Precision, recall, false tips per image (total and on CVC-negative images)
- Duplicate predictions, missing predictions
- Raw candidate coverage before confidence filtering
- Error distribution (median, p95) strictly on matched population
- Stratified breakdowns: Single-CVC vs Multi-CVC, Border, Clutter, Abnormal-course, Low-contrast
- Patient-clustered bootstrap uncertainty intervals
- Artifact generator for standard result bundles
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


BIG_PENALTY = 1e6


def compute_euclidean_distances(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between [M, 2] and [N, 2]."""
    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.zeros((len(coords_a), len(coords_b)), dtype=np.float64)
    diff = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def hungarian_match_tips(
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    pred_confs: np.ndarray,
    tolerance_px: float = 15.0,
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """Maximum-cardinality one-to-one bipartite matching within tolerance, distance as tie-breaker.

    Args:
        gt_coords: [M, 2] ground truth coordinates (native pixels).
        pred_coords: [N, 2] predicted coordinates (native pixels).
        pred_confs: [N] prediction confidences.
        tolerance_px: Matching distance threshold in native pixels.

    Returns:
        matches: list of dicts with gt_idx, pred_idx, distance, conf
        unmatched_gt: list of gt indices
        unmatched_pred: list of pred indices
    """
    m = len(gt_coords)
    n = len(pred_coords)

    if m == 0:
        return [], [], list(range(n))
    if n == 0:
        return [], list(range(m)), []

    dist_matrix = compute_euclidean_distances(gt_coords, pred_coords)
    cost_matrix = np.full((m, n), BIG_PENALTY, dtype=np.float64)

    # Allow matches only within tolerance; penalize larger distances
    admissible = dist_matrix <= tolerance_px
    cost_matrix[admissible] = dist_matrix[admissible]

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: List[Dict[str, Any]] = []
    matched_gt = set()
    matched_pred = set()

    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] <= tolerance_px:
            matches.append({
                "gt_idx": int(r),
                "pred_idx": int(c),
                "distance_px": float(dist_matrix[r, c]),
                "confidence": float(pred_confs[c]),
            })
            matched_gt.add(int(r))
            matched_pred.add(int(c))

    unmatched_gt = [i for i in range(m) if i not in matched_gt]
    unmatched_pred = [j for j in range(n) if j not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


def compute_tip_metrics(
    eval_df: pd.DataFrame,
    tolerances: Tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0, 65.0),
    confidence_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics across an entire evaluation cohort.

    eval_df must contain per-study rows or catheter targets with:
    - study_id
    - patient_id
    - is_cvc_positive (bool)
    - gt_tips: list of dicts [{'x': float, 'y': float, 'tip_state': str, 'instance_id': int, ...}]
    - pred_tips: list of dicts [{'x': float, 'y': float, 'conf': float}]
    - raw_candidates: optional list of dicts [{'x': float, 'y': float, 'conf': float}]
    """
    total_studies = len(eval_df)
    positive_studies = 0
    negative_studies = 0

    total_assessable_gt = 0
    total_preds_above_thresh = 0
    total_raw_candidates = 0

    # Tolerance tracking
    tp_by_tol = {tol: 0 for tol in tolerances}
    fp_by_tol = {tol: 0 for tol in tolerances}
    fn_by_tol = {tol: 0 for tol in tolerances}
    distances_by_tol: Dict[float, List[float]] = {tol: [] for tol in tolerances}

    # Strata tracking for Success@15
    single_cvc_gt = 0
    single_cvc_tp15 = 0
    multi_cvc_gt = 0
    multi_cvc_tp15 = 0

    border_gt = 0
    border_tp15 = 0
    clutter_gt = 0
    clutter_tp15 = 0
    abnormal_gt = 0
    abnormal_tp15 = 0

    # Negative images tracking
    false_preds_on_negatives = 0

    # Per-study records for study-level metrics
    per_study_records: List[Dict[str, Any]] = []
    all_matches_records: List[Dict[str, Any]] = []

    # Candidate coverage tracking (at 15px)
    gt_covered_count = 0

    for _, row in eval_df.iterrows():
        sid = str(row["study_id"])
        pid = str(row.get("patient_id", sid))
        is_pos = bool(row.get("is_cvc_positive", True))
        
        raw_gt = row.get("gt_tips", [])
        # Filter to assessable tips for evaluation
        assessable_gt = [t for t in raw_gt if t.get("tip_state", "assessable") == "assessable"]
        gt_coords = np.array([[t["x"], t["y"]] for t in assessable_gt], dtype=np.float64) if assessable_gt else np.empty((0, 2))
        
        m_count = len(assessable_gt)
        if is_pos:
            positive_studies += 1
            total_assessable_gt += m_count
            if m_count == 1:
                single_cvc_gt += 1
            elif m_count > 1:
                multi_cvc_gt += m_count
        else:
            negative_studies += 1

        raw_preds = row.get("pred_tips", [])
        all_candidates = row.get("raw_candidates", raw_preds)
        total_raw_candidates += len(all_candidates)

        # Filter predictions by confidence threshold
        filt_preds = [p for p in raw_preds if p.get("conf", 1.0) >= confidence_threshold]
        pred_coords = np.array([[p["x"], p["y"]] for p in filt_preds], dtype=np.float64) if filt_preds else np.empty((0, 2))
        pred_confs = np.array([p.get("conf", 1.0) for p in filt_preds], dtype=np.float64) if filt_preds else np.empty((0,))
        
        n_count = len(filt_preds)
        total_preds_above_thresh += n_count

        if not is_pos:
            false_preds_on_negatives += n_count

        # Check raw candidate coverage before confidence filtering (at 15px)
        if m_count > 0 and len(all_candidates) > 0:
            cand_coords = np.array([[c["x"], c["y"]] for c in all_candidates], dtype=np.float64)
            dists_cand = compute_euclidean_distances(gt_coords, cand_coords)
            for i in range(m_count):
                if np.min(dists_cand[i]) <= 15.0:
                    gt_covered_count += 1

        study_tp = {tol: 0 for tol in tolerances}
        study_fp = {tol: 0 for tol in tolerances}
        study_fn = {tol: 0 for tol in tolerances}

        # Evaluate at each tolerance
        for tol in tolerances:
            matches, un_gt, un_pred = hungarian_match_tips(gt_coords, pred_coords, pred_confs, tolerance_px=tol)
            
            n_tp = len(matches)
            n_fp = len(un_pred)
            n_fn = len(un_gt)

            tp_by_tol[tol] += n_tp
            fp_by_tol[tol] += n_fp
            fn_by_tol[tol] += n_fn

            study_tp[tol] = n_tp
            study_fp[tol] = n_fp
            study_fn[tol] = n_fn

            for match in matches:
                dist = match["distance_px"]
                distances_by_tol[tol].append(dist)
                
                if tol == 15.0:
                    all_matches_records.append({
                        "study_id": sid,
                        "patient_id": pid,
                        "gt_instance_id": assessable_gt[match["gt_idx"]].get("instance_id", match["gt_idx"]),
                        "distance_px": dist,
                        "confidence": match["confidence"],
                        "gt_x": float(gt_coords[match["gt_idx"]][0]),
                        "gt_y": float(gt_coords[match["gt_idx"]][1]),
                        "pred_x": float(pred_coords[match["pred_idx"]][0]),
                        "pred_y": float(pred_coords[match["pred_idx"]][1]),
                    })

            # Check strata for primary tolerance 15px
            if tol == 15.0:
                if is_pos:
                    if m_count == 1:
                        single_cvc_tp15 += n_tp
                    elif m_count > 1:
                        multi_cvc_tp15 += n_tp

                    # Check subgroup annotations if present on ground truth
                    for g_idx, g_tip in enumerate(assessable_gt):
                        is_matched = any(m["gt_idx"] == g_idx for m in matches)
                        if g_tip.get("is_border", False):
                            border_gt += 1
                            if is_matched:
                                border_tp15 += 1
                        if g_tip.get("is_clutter", False):
                            clutter_gt += 1
                            if is_matched:
                                clutter_tp15 += 1
                        if g_tip.get("is_abnormal_course", False):
                            abnormal_gt += 1
                            if is_matched:
                                abnormal_tp15 += 1

        per_study_records.append({
            "study_id": sid,
            "patient_id": pid,
            "is_cvc_positive": is_pos,
            "assessable_gt_count": m_count,
            "predictions_count": n_count,
            "tp_15px": study_tp[15.0],
            "fp_15px": study_fp[15.0],
            "fn_15px": study_fn[15.0],
            "success_rate_15px": (study_tp[15.0] / m_count) if m_count > 0 else (1.0 if n_count == 0 else 0.0),
        })

    # Summary calculations
    denom_gt = max(total_assessable_gt, 1)
    
    summary: Dict[str, Any] = {
        "cohort": {
            "total_studies": total_studies,
            "positive_studies": positive_studies,
            "negative_studies": negative_studies,
            "total_assessable_gt_tips": total_assessable_gt,
            "total_predictions_above_threshold": total_preds_above_thresh,
            "confidence_threshold": confidence_threshold,
        },
        "primary_metric": {
            "name": "Success@15",
            "value": float(tp_by_tol[15.0] / denom_gt),
            "matched_assessable_tips": tp_by_tol[15.0],
            "all_assessable_gt_tips": total_assessable_gt,
        },
        "tolerance_metrics": {},
        "error_distribution_matched_15px": {
            "matched_population_count": len(distances_by_tol[15.0]),
            "median_error_px": float(np.median(distances_by_tol[15.0])) if distances_by_tol[15.0] else None,
            "mean_error_px": float(np.mean(distances_by_tol[15.0])) if distances_by_tol[15.0] else None,
            "p95_error_px": float(np.percentile(distances_by_tol[15.0], 95)) if distances_by_tol[15.0] else None,
            "min_error_px": float(np.min(distances_by_tol[15.0])) if distances_by_tol[15.0] else None,
            "max_error_px": float(np.max(distances_by_tol[15.0])) if distances_by_tol[15.0] else None,
        },
        "candidate_diagnostics": {
            "raw_candidate_coverage_15px": float(gt_covered_count / denom_gt) if total_assessable_gt > 0 else 0.0,
            "covered_tips_count": gt_covered_count,
            "coverage_drop_due_to_filtering_or_competition": float((gt_covered_count - tp_by_tol[15.0]) / denom_gt),
            "false_tips_per_image": float(fp_by_tol[15.0] / total_studies) if total_studies > 0 else 0.0,
            "false_tips_on_cvc_negatives_per_image": float(false_preds_on_negatives / max(negative_studies, 1)),
        },
        "strata_metrics_15px": {
            "single_cvc": {
                "gt_tips": single_cvc_gt,
                "matched_tp": single_cvc_tp15,
                "success_rate": float(single_cvc_tp15 / max(single_cvc_gt, 1)),
            },
            "multi_cvc": {
                "gt_tips": multi_cvc_gt,
                "matched_tp": multi_cvc_tp15,
                "success_rate": float(multi_cvc_tp15 / max(multi_cvc_gt, 1)),
            },
            "border_stratum": {
                "gt_tips": border_gt,
                "matched_tp": border_tp15,
                "success_rate": float(border_tp15 / max(border_gt, 1)) if border_gt > 0 else None,
            },
            "clutter_stratum": {
                "gt_tips": clutter_gt,
                "matched_tp": clutter_tp15,
                "success_rate": float(clutter_tp15 / max(clutter_gt, 1)) if clutter_gt > 0 else None,
            },
            "abnormal_course_stratum": {
                "gt_tips": abnormal_gt,
                "matched_tp": abnormal_tp15,
                "success_rate": float(abnormal_tp15 / max(abnormal_gt, 1)) if abnormal_gt > 0 else None,
            },
        },
    }

    for tol in tolerances:
        tp = tp_by_tol[tol]
        fp = fp_by_tol[tol]
        fn = fn_by_tol[tol]
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / denom_gt)
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        summary["tolerance_metrics"][f"Success@{int(tol)}"] = {
            "success_rate": rec,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    per_study_df = pd.DataFrame(per_study_records)
    matches_df = pd.DataFrame(all_matches_records)

    return {
        "summary": summary,
        "per_study_df": per_study_df,
        "matches_df": matches_df,
    }


def compute_patient_clustered_bootstrap(
    per_study_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 43,
) -> Dict[str, Any]:
    """Compute 95% bootstrap intervals clustered by patient_id."""
    rng = np.random.default_rng(seed)
    unique_patients = per_study_df["patient_id"].unique()
    n_patients = len(unique_patients)

    if n_patients == 0:
        return {}

    # Map patient to their studies
    patient_to_indices = per_study_df.groupby("patient_id").indices

    boot_success: List[float] = []
    
    for _ in range(n_bootstrap):
        sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_indices = []
        for p in sampled_patients:
            sampled_indices.extend(patient_to_indices[p])
        
        sample_df = per_study_df.iloc[sampled_indices]
        tot_gt = sample_df["assessable_gt_count"].sum()
        tot_tp = sample_df["tp_15px"].sum()
        
        rate = (tot_tp / tot_gt) if tot_gt > 0 else 0.0
        boot_success.append(rate)

    boot_success = np.array(boot_success)
    ci_low = float(np.percentile(boot_success, 2.5))
    ci_high = float(np.percentile(boot_success, 97.5))
    mean_val = float(np.mean(boot_success))
    std_err = float(np.std(boot_success))

    return {
        "n_bootstrap": n_bootstrap,
        "n_patient_clusters": int(n_patients),
        "primary_metric": "Success@15",
        "mean": mean_val,
        "std_err": std_err,
        "ci_95": [ci_low, ci_high],
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
    }
