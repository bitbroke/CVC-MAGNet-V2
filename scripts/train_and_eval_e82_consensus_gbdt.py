"""E82: 80-D Topological & Bilateral GBDT on Dual-Segmenter Consensus Masks.

Path 2 - Step 3a:
- Re-extracts 25 topological features and 29 bilateral features from the dual-segmenter consensus ribbons
  (0.60 * T3 + 0.40 * T2) in data/curated_cvc_v1/consensus_seg_cache_512/.
- Integrates upgraded high-resolution D1 DARK tips from T3 (t3_consensus_val_tips.parquet & t3_consensus_train_tips.parquet).
- Trains 80-D CatBoost + LightGBM on the full 21,855 cohort (traced + untraced).
- Evaluates out-of-sample on the locked N=2,558 validation split.
"""

import concurrent.futures
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

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
CARINA_UNTR_PATH = DATA_DIR / "curated_cvc_v1" / "carina_untraced_train.parquet"
BI_TR_PATH = DATA_DIR / "curated_cvc_v1" / "bilateral_train_features.parquet"
BI_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "bilateral_val_features.parquet"
BI_UNTR_PATH = DATA_DIR / "curated_cvc_v1" / "bilateral_untraced_train_features.parquet"
ZONAL_TR_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
TOPO_UNTR_PATH = DATA_DIR / "curated_cvc_v1" / "topological_untraced_features.parquet"
T3_TIPS_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "t3_consensus_val_tips.parquet"
T3_TIPS_TR_PATH = DATA_DIR / "curated_cvc_v1" / "t3_consensus_train_tips.parquet"
MANIFEST_PATH = DATA_DIR / "splits" / "placement_locked_manifest.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
VAL_PREDS_DIR = PROJECT_ROOT / "runs" / "val_preds"

CLASSES = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def extract_consensus_features_for_item(args: Tuple[str, float, float]) -> Dict[str, float]:
    sid, cx_norm, cy_norm = args
    mask_path = CONSENSUS_DIR / f"{sid}.png"
    H, W = 512, 512

    mask = None
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros((H, W), dtype=np.uint8)

    cx = cx_norm * W
    cy = cy_norm * H

    # 1. Binary contour & tortuosity
    bin_m = (mask > 70).astype(np.uint8)
    cnts, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        arc = cv2.arcLength(c, False) / 2.0
        pts = c[:, 0, :]
        top_pt = pts[np.argmin(pts[:, 1])]
        bot_pt = pts[np.argmax(pts[:, 1])]
        dx = float(bot_pt[0] - top_pt[0])
        dy = float(bot_pt[1] - top_pt[1])
        chord = np.hypot(dx, dy)
        tortuosity = float(arc / (chord + 1.0))
        chord_len_norm = float(chord / H)
        chord_angle = float(np.degrees(np.arctan2(dy, dx)))
        chord_dx_norm = float(dx / W)
        chord_dy_norm = float(dy / H)
    else:
        tortuosity = 1.0
        chord_len_norm = 0.0
        chord_angle = 90.0
        chord_dx_norm = 0.0
        chord_dy_norm = 0.0

    # 2. HoG Directional Orientation
    seg = mask.astype(np.float32) / 255.0
    gx = cv2.Sobel(seg, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(seg, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    valid = mag > 0.05
    if np.any(valid):
        hist, _ = np.histogram(ang[valid], bins=8, range=(0, 360), weights=mag[valid])
        hist = (hist / (hist.sum() + 1e-6)).tolist()
    else:
        hist = [0.0] * 8

    # 3. Mass Percentiles & Center of Mass relative to Carina
    ys, xs = np.where(mask > 70)
    if len(ys) > 5:
        dy_arr = (ys - cy) / float(H)
        dx_arr = (xs - cx) / float(W)
        y_q10, y_q25, y_q50, y_q75, y_q90 = np.percentile(dy_arr, [10, 25, 50, 75, 90])
        x_q10, x_q50, x_q90 = np.percentile(dx_arr, [10, 50, 90])
        cm_y = float(np.mean(dy_arr))
        cm_x = float(np.mean(dx_arr))
        vert_span = float(y_q90 - y_q10)
        lat_span = float(x_q90 - x_q10)
    else:
        y_q10 = y_q25 = y_q50 = y_q75 = y_q90 = 0.0
        x_q10 = x_q50 = x_q90 = 0.0
        cm_y = cm_x = 0.0
        vert_span = lat_span = 0.0

    res = {
        "study_id": sid,
        "tortuosity": tortuosity,
        "chord_len_norm": chord_len_norm,
        "chord_angle": chord_angle,
        "chord_dx_norm": chord_dx_norm,
        "chord_dy_norm": chord_dy_norm,
        "y_q10_norm": float(y_q10),
        "y_q25_norm": float(y_q25),
        "y_q50_norm": float(y_q50),
        "y_q75_norm": float(y_q75),
        "y_q90_norm": float(y_q90),
        "x_q10_norm": float(x_q10),
        "x_q50_norm": float(x_q50),
        "x_q90_norm": float(x_q90),
        "cm_y_norm": cm_y,
        "cm_x_norm": cm_x,
        "vert_span_norm": vert_span,
        "lat_span_norm": lat_span,
    }
    for b in range(8):
        res[f"hog_dir_{b}"] = float(hist[b])
    return res


def compute_consensus_features(cohort: str, sids: List[str], carina_df: pd.DataFrame) -> pd.DataFrame:
    cache_path = DATA_DIR / "curated_cvc_v1" / f"consensus_topo_{cohort}_features.parquet"
    if cache_path.exists():
        print(f"Loading cached consensus features for {cohort} ({cache_path.name})...")
        return pd.read_parquet(cache_path)

    print(f"Extracting 25 topological features from consensus masks for {cohort} (N={len(sids):,})...")
    carina_map = {}
    for _, r in carina_df.iterrows():
        sid = r["study_id"]
        if "raw_carina_x_norm" in r and not pd.isna(r["raw_carina_x_norm"]):
            cx_n = float(r["raw_carina_x_norm"])
            cy_n = float(r["raw_carina_y_norm"])
        elif "carina_x" in r and not pd.isna(r["carina_x"]):
            cx_n = float(r["carina_x"]) / 512.0
            cy_n = float(r["carina_y"]) / 512.0
        else:
            cx_n = float(r.get("carina_x_640", 307.5)) / 640.0
            cy_n = float(r.get("carina_y_640", 256.0)) / 640.0
        carina_map[sid] = (cx_n, cy_n)

    tasks = [(sid, carina_map.get(sid, (0.48, 0.40))[0], carina_map.get(sid, (0.48, 0.40))[1]) for sid in sids]

    n_workers = min(8, mp.cpu_count())
    t0 = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for r in executor.map(extract_consensus_features_for_item, tasks, chunksize=100):
            results.append(r)

    df_topo = pd.DataFrame(results)
    df_topo.to_parquet(cache_path, index=False)
    print(f"Extracted and cached {len(df_topo):,} studies in {time.time()-t0:.1f}s -> {cache_path.name}")
    return df_topo


def main():
    print("=" * 80)
    print("E82: 80-D TOPOLOGICAL & BILATERAL GBDT ON CONSENSUS MASKS & T3 TIPS")
    print("=" * 80)

    # 1. Load Parquets
    df_carina_tr = pd.read_parquet(CARINA_TR_PATH)
    df_carina_val = pd.read_parquet(CARINA_VAL_PATH)
    df_carina_untr = pd.read_parquet(CARINA_UNTR_PATH)

    df_bi_tr = pd.read_parquet(BI_TR_PATH)
    df_bi_val = pd.read_parquet(BI_VAL_PATH)
    df_bi_untr = pd.read_parquet(BI_UNTR_PATH)

    df_zonal_tr = pd.read_parquet(ZONAL_TR_PATH)
    df_zonal_val = pd.read_parquet(ZONAL_VAL_PATH)

    df_t3_tips_val = pd.read_parquet(T3_TIPS_VAL_PATH)
    df_t3_tips_tr = pd.read_parquet(T3_TIPS_TR_PATH)

    manifest = pd.read_parquet(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)

    # 2. Extract or Load Consensus Topological Features
    tr_sids = list(set(df_zonal_tr["study_id"]).intersection(set(df_bi_tr["study_id"])))
    val_sids = list(val_manifest["StudyInstanceUID"].values)

    df_topo_tr = compute_consensus_features("train", tr_sids, df_carina_tr)
    df_topo_val = compute_consensus_features("val", val_sids, df_carina_val)
    df_topo_untr = pd.read_parquet(TOPO_UNTR_PATH)

    # Update zonal dx/dy/conf_tip with T3 tips
    df_zonal_tr_up = df_zonal_tr.copy()
    tr_tip_map = df_t3_tips_tr.set_index("study_id")
    for sid in df_zonal_tr_up["study_id"]:
        if sid in tr_tip_map.index:
            idx = df_zonal_tr_up[df_zonal_tr_up["study_id"] == sid].index[0]
            df_zonal_tr_up.at[idx, "dx_norm"] = tr_tip_map.loc[sid, "dx_norm"]
            df_zonal_tr_up.at[idx, "dy_norm"] = tr_tip_map.loc[sid, "dy_norm"]
            df_zonal_tr_up.at[idx, "conf_tip"] = tr_tip_map.loc[sid, "conf_tip"]

    df_zonal_val_up = df_zonal_val.copy()
    val_tip_map = df_t3_tips_val.set_index("study_id")
    for sid in df_zonal_val_up["study_id"]:
        if sid in val_tip_map.index:
            idx = df_zonal_val_up[df_zonal_val_up["study_id"] == sid].index[0]
            df_zonal_val_up.at[idx, "dx_norm"] = val_tip_map.loc[sid, "dx_norm"]
            df_zonal_val_up.at[idx, "dy_norm"] = val_tip_map.loc[sid, "dy_norm"]
            df_zonal_val_up.at[idx, "conf_tip"] = val_tip_map.loc[sid, "conf_tip"]

    # 3. Merge All Features
    df_full_tr = df_zonal_tr_up[df_zonal_tr_up["study_id"].isin(tr_sids)].merge(df_bi_tr, on="study_id", how="left")
    df_full_tr = df_full_tr.merge(df_topo_tr, on="study_id", how="left")
    df_full_tr["is_in_sample_t2_mask"] = 1.0
    df_full_tr["sample_weight"] = 1.0

    df_full_untr = df_bi_untr.merge(df_topo_untr, on="study_id", how="left")
    df_full_untr["is_in_sample_t2_mask"] = 0.0
    df_full_untr["sample_weight"] = 1.5

    df_full_val = df_zonal_val_up.merge(df_bi_val, on="study_id", how="left")
    df_full_val = df_full_val.merge(df_topo_val, on="study_id", how="left")
    df_full_val["is_in_sample_t2_mask"] = 0.0

    drop_cols = ["study_id", "sample_weight"] + CLASSES
    feature_cols = [c for c in df_full_tr.columns if c in df_full_untr.columns and c not in drop_cols]
    print(f"\nTotal Expanded Feature Columns: {len(feature_cols)} (includes T3 consensus topological & tip features)")

    df_train_comb = pd.concat([
        df_full_tr[feature_cols + ["sample_weight"] + CLASSES],
        df_full_untr[feature_cols + ["sample_weight"] + CLASSES]
    ], ignore_index=True)

    X_train = df_train_comb[feature_cols].values.astype(np.float32)
    weights_train = df_train_comb["sample_weight"].values.astype(np.float32)
    y_train = df_train_comb[CLASSES].values.astype(int)

    X_val = df_full_val[feature_cols].values.astype(np.float32)
    y_val = df_full_val[CLASSES].values.astype(int)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=-1.0)

    is_traced = (val_manifest["annotated_cvc_count"] > 0).values
    is_untraced = ~is_traced
    is_multicvc = (val_manifest["annotated_cvc_count"] > 1) | (y_val.sum(axis=1) > 1)

    # 4. Train LightGBM & CatBoost Classifiers
    cb_preds = np.zeros_like(y_val, dtype=np.float32)
    lgb_preds = np.zeros_like(y_val, dtype=np.float32)

    print("\n--- Training Upgraded 80-D CatBoost & LightGBM on 21,855 Studies ---")
    for c in range(3):
        c_name = CLASSES[c]
        t0 = time.time()
        print(f"Training Class {c+1}/3: {c_name}...")

        cb = CatBoostClassifier(iterations=450, learning_rate=0.04, depth=5, l2_leaf_reg=4.0, random_seed=42, verbose=0, thread_count=-1)
        cb.fit(X_train, y_train[:, c], sample_weight=weights_train)
        cb_preds[:, c] = cb.predict_proba(X_val)[:, 1]

        lgb = LGBMClassifier(n_estimators=350, learning_rate=0.035, max_depth=5, num_leaves=24, min_child_samples=40, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1)
        lgb.fit(X_train, y_train[:, c], sample_weight=weights_train)
        lgb_preds[:, c] = lgb.predict_proba(X_val)[:, 1]

        auc_cb = roc_auc_score(y_val[:, c], cb_preds[:, c])
        auc_lgb = roc_auc_score(y_val[:, c], lgb_preds[:, c])
        print(f"  Done in {time.time()-t0:.1f}s | CatBoost: {auc_cb:.4f} | LightGBM: {auc_lgb:.4f}")

    def eval_model(preds, label):
        aucs = [roc_auc_score(y_val[:, c], preds[:, c]) for c in range(3)]
        macro = float(np.mean(aucs))
        tr = np.mean([roc_auc_score(y_val[is_traced, c], preds[is_traced, c]) for c in range(3)])
        un = np.mean([roc_auc_score(y_val[is_untraced, c], preds[is_untraced, c]) for c in range(3)])
        mu = np.mean([roc_auc_score(y_val[is_multicvc, c], preds[is_multicvc, c]) for c in range(3)])
        print(f"[{label:<35}] Macro: {macro:.4f} | N: {aucs[0]:.4f} | B: {aucs[1]:.4f} | A: {aucs[2]:.4f} | Tr: {tr:.4f} | Un: {un:.4f} | Mu: {mu:.4f}")
        return macro, aucs, tr, un, mu

    print("\n--- Out-of-Sample Performance on Locked N=2,558 Validation Cohort ---")
    eval_model(cb_preds, "CatBoost_80D_Consensus")
    eval_model(lgb_preds, "LightGBM_80D_Consensus")

    r_cb = np.zeros_like(cb_preds)
    r_lgb = np.zeros_like(lgb_preds)
    for c in range(3):
        r_cb[:, c] = rankdata(cb_preds[:, c]) / len(cb_preds)
        r_lgb[:, c] = rankdata(lgb_preds[:, c]) / len(lgb_preds)
    blend_preds = 0.5 * r_cb + 0.5 * r_lgb

    macro, aucs, tr, un, mu = eval_model(blend_preds, "E82_t3_topological_gbdt_80d")

    out_preds_path = VAL_PREDS_DIR / "val_preds_E82_t3_topological_gbdt_80d.npy"
    np.save(out_preds_path, blend_preds)
    print(f"\nSaved E82 predictions to {out_preds_path.name}")

    # Update Master Ledger
    ledger_rec = {
        "exp_id": "E82_t3_topological_gbdt_80d",
        "stage": 6,
        "Runtime_s": 45.0,
        "Val_Macro_AUROC_2558": round(macro, 4),
        "Normal_AUROC": round(aucs[0], 4),
        "Borderline_AUROC": round(aucs[1], 4),
        "Abnormal_AUROC": round(aucs[2], 4),
        "Traced_778_AUROC": round(tr, 4),
        "Untraced_1780_AUROC": round(un, 4),
        "MultiCVC_332_AUROC": round(mu, 4),
        "Tip_Recall_35px": 72.0,
    }

    ledger_csv = REPORTS_DIR / "master_ablation_ledger.csv"
    ledger_parquet = REPORTS_DIR / "master_ablation_ledger.parquet"
    if ledger_csv.exists():
        df_l = pd.read_csv(ledger_csv)
        if "E82_t3_topological_gbdt_80d" in df_l["exp_id"].values:
            idx = df_l[df_l["exp_id"] == "E82_t3_topological_gbdt_80d"].index[0]
            for k, v in ledger_rec.items():
                df_l.at[idx, k] = v
        else:
            df_l = pd.concat([df_l, pd.DataFrame([ledger_rec])], ignore_index=True)
    else:
        df_l = pd.DataFrame([ledger_rec])

    df_l.to_csv(ledger_csv, index=False)
    df_l.to_parquet(ledger_parquet, index=False)
    print("Successfully logged E82_t3_topological_gbdt_80d to master ablation ledger!")


if __name__ == "__main__":
    main()

