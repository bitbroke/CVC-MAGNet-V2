"""Train and Evaluate Multi-Label Zonal Mass + Gated Geometric GBDT.

Trains 3 independent binary Gradient Boosting models (Normal, Borderline, Abnormal)
on the N=6,600 training cohort using:
1. 6 Zonal Segmentation Mass Integrals (Neck, Upper SVC, Borderline, Normal CAJ, Deep RA, Lateral)
2. 5 Normalized Mass Fractions (frac_caj_normal, frac_borderline, frac_upper_svc, frac_neck, frac_atrium_deep)
3. 3 Ribbon Vertical Extent Features (deepest_y_norm, p95_y_norm, highest_y_norm)
4. 7 Confidence-Gated Tip Coordinates (dx, dy, r, cos_theta, sin_theta, has_valid_tip, conf_tip)
5. 2 Carina Landmark Metadata Features (conf_carina, is_carina_fallback)

Evaluates out-of-sample on all N=2,558 locked validation studies.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
DATA_DIR = PROJECT_ROOT / "data"
ZONAL_VAL_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_val_features.parquet"
ZONAL_TRAIN_PATH = DATA_DIR / "curated_cvc_v1" / "zonal_train_features.parquet"

FEATURE_COLS = [
    # Zonal masses
    "m_neck",
    "m_upper_svc",
    "m_borderline",
    "m_caj_normal",
    "m_atrium_deep",
    "m_lateral",
    "m_total_med",
    # Normalized fractions
    "frac_neck",
    "frac_upper_svc",
    "frac_borderline",
    "frac_caj_normal",
    "frac_atrium_deep",
    # Ribbon extent
    "deepest_y_norm",
    "p95_y_norm",
    "highest_y_norm",
    "ribbon_pixels",
    # Gated tip coordinates
    "dx_norm",
    "dy_norm",
    "r_norm",
    "cos_theta",
    "sin_theta",
    "has_valid_tip",
    "conf_tip",
    # CarinaNet metadata
    "conf_carina",
    "is_carina_fallback",
]

TARGET_COLS = ["CVC - Normal", "CVC - Borderline", "CVC - Abnormal"]


def main():
    print("=" * 75)
    print("STEP 2.5: MULTI-LABEL ZONAL MASS + GATED GEOMETRIC GBDT BENCHMARK")
    print("=" * 75)

    if not ZONAL_TRAIN_PATH.exists() or not ZONAL_VAL_PATH.exists():
        print("ERROR: Zonal feature caches not found. Run scripts/build_zonal_feature_cache.py first.")
        return

    df_train = pd.read_parquet(ZONAL_TRAIN_PATH)
    df_val = pd.read_parquet(ZONAL_VAL_PATH)

    print(f"Training instances: {len(df_train):,} | Validation instances: {len(df_val):,}")
    print(f"Total features: {len(FEATURE_COLS)}")

    X_train = df_train[FEATURE_COLS].values.astype(np.float32)
    X_val = df_val[FEATURE_COLS].values.astype(np.float32)

    # Check for NaNs
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)

    results = {}
    aurocs = []
    models = {}

    for target in TARGET_COLS:
        y_train = df_train[target].values.astype(np.int64)
        y_val = df_val[target].values.astype(np.int64)

        pos_train = int(y_train.sum())
        pos_val = int(y_val.sum())

        print(f"\n--- Training Binary GBDT for {target} (Train Pos: {pos_train}/{len(y_train)}) ---")
        
        # Calculate scale_pos_weight approximation via sample weights
        neg_train = len(y_train) - pos_train
        pos_weight = neg_train / max(pos_train, 1)
        sample_weights = np.where(y_train == 1, math.sqrt(pos_weight), 1.0)

        clf = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        models[target] = clf

        # Predict probability of positive class
        y_val_pred = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred)
        aurocs.append(auc)

        # Feature importances
        top_feat_idx = np.argsort(clf.feature_importances_)[::-1][:5]
        top_feats = {FEATURE_COLS[i]: round(float(clf.feature_importances_[i]), 4) for i in top_feat_idx}

        results[target] = {
            "auroc": round(float(auc), 4),
            "pos_support": pos_val,
            "top_5_features": top_feats,
        }
        print(f"  --> {target} AUROC: {auc:.4f} (pos={pos_val}/{len(y_val)})")
        print(f"      Top Features: {top_feats}")

    macro_auc = float(np.mean(aurocs))
    results["macro_auroc"] = round(macro_auc, 4)
    results["total_validation_studies"] = len(df_val)

    print("\n" + "=" * 75)
    print("ZONAL MASS + GATED GBDT BENCHMARK SUMMARY (N=2,558)")
    print("=" * 75)
    print(f"{'Target Class':<20} | {'AUROC':<10} | {'Pos Support':<12}")
    print("-" * 50)
    for target in TARGET_COLS:
        print(f"{target:<20} | {results[target]['auroc']:<10.4f} | {results[target]['pos_support']:<12}")
    print("-" * 50)
    print(f"{'Macro AUROC':<20} | {macro_auc:<10.4f} | {len(df_val):<12}")
    print("=" * 75)

    # Save results
    out_path = PROJECT_ROOT / "runs" / "zonal_gbdt_evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
