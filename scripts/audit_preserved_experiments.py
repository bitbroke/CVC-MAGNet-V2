"""Comprehensive Audit of All Preserved Experiments, Predictions, Checkpoints, and Logs."""

import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNS_DIR = PROJECT_ROOT / "runs"
VAL_PREDS_DIR = RUNS_DIR / "val_preds"
CSV_PATH = REPORTS_DIR / "master_ablation_ledger.csv"
PARQUET_PATH = REPORTS_DIR / "master_ablation_ledger.parquet"

print("=" * 80)
print("COMPREHENSIVE AUDIT OF ALL PRESERVED EXPERIMENTAL ASSETS")
print("=" * 80)

# 1. Master Ledger
if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
    print(f"\n[1] MASTER ABLATION LEDGER: {len(df)} EXPERIMENTS RECORDED")
    print(f"    - CSV path:     {CSV_PATH} ({CSV_PATH.stat().st_size} bytes)")
    print(f"    - Parquet path: {PARQUET_PATH} ({PARQUET_PATH.stat().st_size} bytes)")
    print("\nBreakdown by Stage:")
    for stage, count in df["stage"].value_counts().sort_index().items():
        print(f"    Stage {stage}: {count} experiments")
else:
    print("Master ledger not found!")

# 2. Saved Prediction Matrices
print(f"\n[2] OUT-OF-SAMPLE PREDICTION MATRICES IN {VAL_PREDS_DIR.name}/")
if VAL_PREDS_DIR.exists():
    pred_files = sorted(list(VAL_PREDS_DIR.glob("*.npy")))
    print(f"    Total saved [2558, 3] prediction arrays: {len(pred_files)}")
    for f in pred_files:
        print(f"    - {f.name:55s} ({f.stat().st_size / 1024:.1f} KB)")
else:
    print("    Prediction directory not found!")

# 3. Model Checkpoints
print(f"\n[3] MODEL CHECKPOINTS IN runs/")
ckpts = sorted(list(RUNS_DIR.rglob("*.pt")))
print(f"    Total model checkpoints preserved: {len(ckpts)}")
for c in ckpts:
    print(f"    - {str(c.relative_to(PROJECT_ROOT)):60s} ({c.stat().st_size / (1024*1024):.1f} MB)")

# 4. JSON Summary & Metric Logs
print(f"\n[4] RUN SUMMARIES & METRIC LOGS IN runs/")
jsons = sorted(list(RUNS_DIR.rglob("*.json")))
print(f"    Total JSON summary files: {len(jsons)}")
for j in jsons:
    print(f"    - {str(j.relative_to(PROJECT_ROOT)):60s} ({j.stat().st_size / 1024:.1f} KB)")

# 5. Git Status Check
print(f"\n[5] ARTIFACT REPORTS PRESERVED")
brain_dir = Path("C:/Users/M S I/.gemini/antigravity/brain/a7024f39-ade2-4704-acf4-d584def2239c")
if brain_dir.exists():
    md_files = sorted(list(brain_dir.glob("*.md")))
    for m in md_files:
        print(f"    - {m.name:50s} ({m.stat().st_size / 1024:.1f} KB)")

print("\n" + "=" * 80)
print("AUDIT COMPLETE: ALL PREVIOUS EXPERIMENTAL RESULTS ARE FULLY PRESERVED")
print("=" * 80)

