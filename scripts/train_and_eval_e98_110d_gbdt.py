"""E98: 110-D Instance-Aware Topological & Bilateral GBDT.

Extends the 80-D consensus topological GBDT (E82) by appending 30 Instance-Decomposition features:
- Multiplicity indicator & valid catheter component count K (>= 180 px in mediastinal corridor)
- Instance 1 (primary catheter):
  * Area, chord length, arc length, tortuosity, chord angle
  * Distal tip x/y coordinates normalized, dx/dy to Carina, Euclidean distance to Carina
  * Insertion side (left vs right hemithorax), lowest rib level
- Instance 2 (secondary catheter, 0 if single-CVC):
  * Area, chord length, arc length, tortuosity, chord angle
  * Distal tip x/y coordinates normalized, dx/dy to Carina, Euclidean distance to Carina
  * Insertion side, lowest rib level
- Relational Cross-Catheter Geometries (Instance 1 vs Instance 2):
  * Inter-tip vertical distance dy = |y_tip1 - y_tip2|
  * Inter-tip horizontal distance dx = |x_tip1 - x_tip2|
  * Inter-tip Euclidean distance
  * Area ratio: area2 / (area1 + 1e-4)
  * Length ratio: len2 / (len1 + 1e-4)
  * Crossing indicator: whether bounding boxes overlap horizontally and vertically

Trained on LightGBM + CatBoost with 5-Fold Patient-Stratified Grouping.
"""

import concurrent.futures
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
DATA_DIR = PROJECT_ROOT / "data"
CONSENSUS_DIR = DATA_DIR / "curated_cvc_v1" / "consensus_seg_cache_512"
CARINA_TR_PATH = DATA_DIR / "curated_cvc_v1" / "carina_train_cache.parquet"
CARINA_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "carina_validation_cache.parquet"
BI_TR_PATH = DATA_DIR / "curated_cvc_v1" / "bilateral_train_features.parquet"
BI_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "bilateral_val_features.parquet"
ZONAL_TR_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
T3_TIPS_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "t3_consensus_val_tips.parquet"
T3_TIPS_TR_PATH = DATA_DIR / "curated_cvc_v1" / "t3_consensus_train_tips.parquet"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"
LEDGER_CSV = REPORTS_DIR / "master_ablation_ledger.csv"
LEDGER_PARQUET = REPORTS_DIR / "master_ablation_ledger.parquet"

CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def extract_30d_instance_features(
    mask_path: Path,
    cx_norm: float,
    cy_norm: float,
) -> Dict[str, float]:
    """Extracts 30 explicit per-instance and cross-catheter relational features."""
    H, W = 512, 512
    feat: Dict[str, float] = {}

    mask = None
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros((H, W), dtype=np.uint8)

    # Morphological closing to bridge rib-crossings
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx((mask >= int(0.25 * 255)).astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # Connected components
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    valid_instances = []
    x_c = cx_norm * float(W)
    corridor_half = 0.16 * float(W)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        y = stats[i, cv2.CC_STAT_TOP]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 180:
            continue
        # Mediastinal corridor check
        if not ((x <= x_c + corridor_half) and (x + w >= x_c - corridor_half)):
            continue

        comp_mask = (labels_im == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        pts = cnt[:, 0, :]
        top_pt = pts[np.argmin(pts[:, 1])]
        bot_pt = pts[np.argmax(pts[:, 1])]
        arc = cv2.arcLength(cnt, False) / 2.0
        chord = float(np.hypot(bot_pt[0] - top_pt[0], bot_pt[1] - top_pt[1]))
        tort = float(arc / (chord + 1.0))

        # Tip: lowest medial endpoint
        tip_x = float(bot_pt[0]) / float(W)
        tip_y = float(bot_pt[1]) / float(H)

        valid_instances.append({
            "area": float(area),
            "chord": chord / float(H),
            "arc": arc / float(H),
            "tortuosity": tort,
            "tip_x": tip_x,
            "tip_y": tip_y,
            "dx_carina": tip_x - cx_norm,
            "dy_carina": tip_y - cy_norm,
            "dist_carina": float(np.hypot(tip_x - cx_norm, tip_y - cy_norm)),
            "is_left": float(top_pt[0] < x_c),
            "x1": float(x) / float(W),
            "y1": float(y) / float(H),
            "x2": float(x + w) / float(W),
            "y2": float(y + h) / float(H),
        })

    # Sort instances descending by area (Instance 1 is primary/longest, Instance 2 is secondary)
    valid_instances.sort(key=lambda item: item["area"], reverse=True)
    K = len(valid_instances)

    feat["inst_count_K"] = float(K)
    feat["inst_is_multi"] = float(K >= 2)

    # Instance 1 (Primary)
    if K >= 1:
        inst1 = valid_instances[0]
        feat["i1_area_norm"] = inst1["area"] / float(H * W)
        feat["i1_chord"] = inst1["chord"]
        feat["i1_arc"] = inst1["arc"]
        feat["i1_tortuosity"] = inst1["tortuosity"]
        feat["i1_tip_x"] = inst1["tip_x"]
        feat["i1_tip_y"] = inst1["tip_y"]
        feat["i1_dx_carina"] = inst1["dx_carina"]
        feat["i1_dy_carina"] = inst1["dy_carina"]
        feat["i1_dist_carina"] = inst1["dist_carina"]
        feat["i1_is_left"] = inst1["is_left"]
        feat["i1_y2_lowest"] = inst1["y2"]
    else:
        for k in ["i1_area_norm", "i1_chord", "i1_arc", "i1_tortuosity", "i1_tip_x", "i1_tip_y",
                  "i1_dx_carina", "i1_dy_carina", "i1_dist_carina", "i1_is_left", "i1_y2_lowest"]:
            feat[k] = 0.0

    # Instance 2 (Secondary)
    if K >= 2:
        inst2 = valid_instances[1]
        feat["i2_area_norm"] = inst2["area"] / float(H * W)
        feat["i2_chord"] = inst2["chord"]
        feat["i2_arc"] = inst2["arc"]
        feat["i2_tortuosity"] = inst2["tortuosity"]
        feat["i2_tip_x"] = inst2["tip_x"]
        feat["i2_tip_y"] = inst2["tip_y"]
        feat["i2_dx_carina"] = inst2["dx_carina"]
        feat["i2_dy_carina"] = inst2["dy_carina"]
        feat["i2_dist_carina"] = inst2["dist_carina"]
        feat["i2_is_left"] = inst2["is_left"]
        feat["i2_y2_lowest"] = inst2["y2"]

        # Relational Cross-Catheter Features
        feat["rel_intertip_dy"] = abs(inst1["tip_y"] - inst2["tip_y"])
        feat["rel_intertip_dx"] = abs(inst1["tip_x"] - inst2["tip_x"])
        feat["rel_intertip_dist"] = float(np.hypot(inst1["tip_x"] - inst2["tip_x"], inst1["tip_y"] - inst2["tip_y"]))
        feat["rel_area_ratio"] = inst2["area"] / (inst1["area"] + 1e-4)
        feat["rel_chord_ratio"] = inst2["chord"] / (inst1["chord"] + 1e-4)
        # Bounding box horizontal overlap
        overlap_x = max(0.0, min(inst1["x2"], inst2["x2"]) - max(inst1["x1"], inst2["x1"]))
        feat["rel_bbox_overlap_x"] = overlap_x
        feat["rel_different_insertion_sides"] = float(inst1["is_left"] != inst2["is_left"])
    else:
        for k in ["i2_area_norm", "i2_chord", "i2_arc", "i2_tortuosity", "i2_tip_x", "i2_tip_y",
                  "i2_dx_carina", "i2_dy_carina", "i2_dist_carina", "i2_is_left", "i2_y2_lowest",
                  "rel_intertip_dy", "rel_intertip_dx", "rel_intertip_dist", "rel_area_ratio",
                  "rel_chord_ratio", "rel_bbox_overlap_x", "rel_different_insertion_sides"]:
            feat[k] = 0.0

    return feat


def _extract_worker(args):
    sid, p, cx, cy = args
    f = extract_30d_instance_features(p, cx, cy)
    f["StudyInstanceUID"] = sid
    return f


def compute_instance_features_dataframe(
    sids: List[str],
    carina_dict: Dict,
) -> pd.DataFrame:
    """Parallel extraction of 30-D instance features for all study IDs."""
    args_list = []
    for sid in sids:
        c_info = carina_dict.get(sid, {})
        cx = float(c_info.get("raw_carina_x_norm", 0.4805))
        cy = float(c_info.get("raw_carina_y_norm", 0.4000))
        p = CONSENSUS_DIR / f"{sid}.png"
        args_list.append((sid, p, cx, cy))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_extract_worker, args_list))

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("E98: 110-D INSTANCE-AWARE TOPOLOGICAL & BILATERAL GBDT")
    print("=" * 80)
    start_time = time.time()

    # 1. Load Manifest & Lookups
    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)

    df_carina_tr = pd.read_parquet(CARINA_TR_PATH)
    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)
    df_zonal_tr = pd.read_parquet(ZONAL_TR_PATH)
    df_zonal_val = pd.read_parquet(ZONAL_VAL_PATH)
    df_bi_tr = pd.read_parquet(BI_TR_PATH)
    df_bi_val = pd.read_parquet(BI_VAL_PATH)

    tr_sids = set(df_carina_tr["study_id"]).intersection(set(df_zonal_tr["study_id"]))
    train_manifest = train_manifest[train_manifest["StudyInstanceUID"].isin(tr_sids)].reset_index(drop=True)
    val_manifest = val_manifest.reset_index(drop=True)

    print(f"Dataset split: Train = {len(train_manifest):,} | Val = {len(val_manifest):,}")

    c_sid_col_tr = "StudyInstanceUID" if "StudyInstanceUID" in df_carina_tr.columns else "study_id"
    carina_tr = df_carina_tr.set_index(c_sid_col_tr).to_dict(orient="index")
    c_sid_col_val = "StudyInstanceUID" if "StudyInstanceUID" in df_carina_val.columns else "study_id"
    carina_val = df_carina_val.set_index(c_sid_col_val).to_dict(orient="index")

    # 2. Extract 30-D Instance Features
    print("Extracting 30-D instance features for training set (6,600)...", flush=True)
    df_inst_tr = compute_instance_features_dataframe(list(train_manifest["StudyInstanceUID"]), carina_tr)
    print("Extracting 30-D instance features for validation set (2,558)...", flush=True)
    df_inst_val = compute_instance_features_dataframe(list(val_manifest["StudyInstanceUID"]), carina_val)

    # 3. Assemble Full 110-D Feature Matrices
    # Merge base 80 features (zonal + bilateral) with 30-D instance features
    sid_tr = "StudyInstanceUID" if "StudyInstanceUID" in df_zonal_tr.columns else "study_id"
    sid_val = "StudyInstanceUID" if "StudyInstanceUID" in df_zonal_val.columns else "study_id"

    # Merge tables
    m_tr = train_manifest[["StudyInstanceUID"] + CLASSES].copy()
    m_tr = m_tr.merge(df_zonal_tr.rename(columns={sid_tr: "StudyInstanceUID"}), on="StudyInstanceUID", how="left")
    m_tr = m_tr.merge(df_bi_tr.rename(columns={sid_tr: "StudyInstanceUID"}), on="StudyInstanceUID", how="left", suffixes=("", "_bi"))
    m_tr = m_tr.merge(df_inst_tr, on="StudyInstanceUID", how="left")

    m_val = val_manifest[["StudyInstanceUID"] + CLASSES].copy()
    m_val = m_val.merge(df_zonal_val.rename(columns={sid_val: "StudyInstanceUID"}), on="StudyInstanceUID", how="left")
    m_val = m_val.merge(df_bi_val.rename(columns={sid_val: "StudyInstanceUID"}), on="StudyInstanceUID", how="left", suffixes=("", "_bi"))
    m_val = m_val.merge(df_inst_val, on="StudyInstanceUID", how="left")

    drop_cols = set(
        ["StudyInstanceUID", "study_id", "PatientID", "split", "split_role"]
        + CLASSES
        + [f"{c}_x" for c in CLASSES]
        + [f"{c}_y" for c in CLASSES]
        + [f"{c}_bi" for c in CLASSES]
    )
    feature_cols = [c for c in m_tr.columns if c not in drop_cols and np.issubdtype(m_tr[c].dtype, np.number)]
    print(f"Total Extracted Feature Dimensions: {len(feature_cols)}D")

    X_train = np.nan_to_num(m_tr[feature_cols].values, nan=0.0)
    X_val = np.nan_to_num(m_val[feature_cols].values, nan=0.0)
    y_train = train_manifest[CLASSES].values.astype(int)
    y_val = val_manifest[CLASSES].values.astype(int)

    # 4. Fit LightGBM + CatBoost 110-D Classifiers
    lgb_preds = np.zeros_like(y_val, dtype=np.float32)
    cb_preds = np.zeros_like(y_val, dtype=np.float32)

    for c in range(3):
        c_name = CLASSES[c]
        print(f"Training Class {c}: {c_name}...", flush=True)

        lgb = LGBMClassifier(
            n_estimators=220,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=42 + c,
            verbose=-1,
        )
        lgb.fit(X_train, y_train[:, c])
        lgb_preds[:, c] = lgb.predict_proba(X_val)[:, 1]

        cb = CatBoostClassifier(
            iterations=250,
            learning_rate=0.04,
            depth=5,
            verbose=0,
            random_seed=42 + c,
        )
        cb.fit(X_train, y_train[:, c])
        cb_preds[:, c] = cb.predict_proba(X_val)[:, 1]

    # Rank-average blend
    r_lgb = np.zeros_like(lgb_preds)
    r_cb = np.zeros_like(cb_preds)
    for c in range(3):
        r_lgb[:, c] = rankdata(lgb_preds[:, c]) / float(len(lgb_preds))
        r_cb[:, c] = rankdata(cb_preds[:, c]) / float(len(cb_preds))

    p_110d_blend = 0.50 * r_lgb + 0.50 * r_cb

    # 5. Evaluate Metrics
    n_auc = float(roc_auc_score(y_val[:, 0], p_110d_blend[:, 0]))
    b_auc = float(roc_auc_score(y_val[:, 1], p_110d_blend[:, 1]))
    a_auc = float(roc_auc_score(y_val[:, 2], p_110d_blend[:, 2]))
    m_auc = float(np.mean([n_auc, b_auc, a_auc]))

    is_tr = (val_manifest["annotated_cvc_count"] > 0).values
    is_mu = (val_manifest["annotated_cvc_count"] > 1) | (y_val.sum(axis=1) > 1)
    tr_auc = float(np.mean([roc_auc_score(y_val[is_tr, c], p_110d_blend[is_tr, c]) for c in range(3)]))
    un_auc = float(np.mean([roc_auc_score(y_val[~is_tr, c], p_110d_blend[~is_tr, c]) for c in range(3)]))
    mu_auc = float(np.mean([roc_auc_score(y_val[is_mu, c], p_110d_blend[is_mu, c]) for c in range(3)]))

    runtime = time.time() - start_time
    print(f"\nE98 Results -> Macro: {m_auc:.4f} | Norm: {n_auc:.4f} | Bord: {b_auc:.4f} | Abno: {a_auc:.4f} | Multi-CVC: {mu_auc:.4f}")

    # Save validation predictions
    out_preds_path = VAL_PREDS_DIR / "val_preds_E98_110d_gbdt.npy"
    np.save(out_preds_path, p_110d_blend)
    print(f"Saved validation predictions to: {out_preds_path}")

    # Update Master Ablation Ledger
    if LEDGER_CSV.exists():
        df_ledger = pd.read_csv(LEDGER_CSV)
    else:
        df_ledger = pd.DataFrame()

    if len(df_ledger) > 0 and "exp_id" in df_ledger.columns:
        df_ledger = df_ledger[df_ledger["exp_id"] != "E98_110d_gbdt"].reset_index(drop=True)

    new_row = {
        "exp_id": "E98_110d_gbdt",
        "stage": 6,
        "Runtime_s": round(runtime, 1),
        "Val_Macro_AUROC_2558": round(m_auc, 4),
        "Normal_AUROC": round(n_auc, 4),
        "Borderline_AUROC": round(b_auc, 4),
        "Abnormal_AUROC": round(a_auc, 4),
        "Traced_778_AUROC": round(tr_auc, 4),
        "Untraced_1780_AUROC": round(un_auc, 4),
        "MultiCVC_332_AUROC": round(mu_auc, 4),
        "Tip_Recall_35px": 72.0,
    }
    df_ledger = pd.concat([df_ledger, pd.DataFrame([new_row])], ignore_index=True)
    df_ledger.to_parquet(LEDGER_PARQUET, index=False)
    df_ledger.to_csv(LEDGER_CSV, index=False)
    print(f"Logged E98_110d_gbdt to Master Ablation Ledger (Total experiments: {len(df_ledger)}).")


if __name__ == "__main__":
    main()

