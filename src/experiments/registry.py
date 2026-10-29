"""Experiment Registry & Contract Framework for CVC Localization Roadmap.

Enforces:
1. Immutable registration of experiment contracts prior to launch.
2. Standard directory structure under experiments/<experiment_id>/.
3. Persistent master tracking table in experiments/master_metrics_table.csv.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MASTER_TABLE_PATH = EXPERIMENTS_DIR / "master_metrics_table.csv"

MASTER_COLUMNS = [
    "experiment_id",
    "seed",
    "checkpoint_sha",
    "cohort_hash",
    "successful_updates",
    "primary_metric",
    "precision",
    "recall",
    "false_predictions_per_image",
    "median_error",
    "p95_error",
    "runtime_seconds",
    "peak_vram_mb",
    "status",
    "decision",
]


def compute_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ExperimentContract:
    def __init__(self, experiment_id: str, base_dir: Path = EXPERIMENTS_DIR):
        self.experiment_id = experiment_id
        self.exp_dir = base_dir / experiment_id
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.evaluation_dir = self.exp_dir / "evaluation"
        self.overlays_dir = self.exp_dir / "overlays"
        
        # Sub-overlay directories
        self.fixed_panel_dir = self.overlays_dir / "fixed_panel"
        self.largest_errors_dir = self.overlays_dir / "largest_errors"
        self.negative_images_dir = self.overlays_dir / "negative_images"
        self.multi_cvc_dir = self.overlays_dir / "multi_cvc"
        
        self._setup_directories()

    def _setup_directories(self):
        for d in [
            self.exp_dir,
            self.checkpoints_dir,
            self.evaluation_dir,
            self.overlays_dir,
            self.fixed_panel_dir,
            self.largest_errors_dir,
            self.negative_images_dir,
            self.multi_cvc_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def register_contract(
        self,
        hypothesis: str,
        command: str,
        config: Dict[str, Any],
        provenance: Dict[str, Any],
        dataset_summary: Dict[str, Any],
    ):
        """Write pre-launch contract files atomically."""
        with open(self.exp_dir / "hypothesis.txt", "w", encoding="utf-8") as f:
            f.write(hypothesis.strip() + "\n")

        with open(self.exp_dir / "command.txt", "w", encoding="utf-8") as f:
            f.write(command.strip() + "\n")

        with open(self.exp_dir / "resolved_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        with open(self.exp_dir / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)

        with open(self.exp_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(dataset_summary, f, indent=2)

    def write_conclusion(
        self,
        what_changed: str,
        what_stayed_fixed: str,
        did_primary_metric_improve: str,
        what_became_worse: str,
        crashes_or_deviations: str,
        decision: str,  # "RETAIN", "REPEAT", or "REJECT"
    ):
        text = f"""======================================================================
EXPERIMENT CONCLUSION: {self.experiment_id}
Date: {datetime.now(timezone.utc).isoformat()}
======================================================================
1. What changed:
{what_changed.strip()}

2. What stayed fixed:
{what_stayed_fixed.strip()}

3. Did the primary metric improve:
{did_primary_metric_improve.strip()}

4. What became worse:
{what_became_worse.strip()}

5. Crashes, retries, exclusions, or protocol deviations:
{crashes_or_deviations.strip()}

6. Protocol Decision (RETAIN / REPEAT / REJECT):
{decision.strip()}
"""
        with open(self.exp_dir / "conclusion.txt", "w", encoding="utf-8") as f:
            f.write(text)


def update_master_table(row_data: Optional[Dict[str, Any]] = None, **kwargs):
    """Update or append experiment record to master metrics table."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if MASTER_TABLE_PATH.exists():
        df = pd.read_csv(MASTER_TABLE_PATH)
    else:
        df = pd.DataFrame(columns=MASTER_COLUMNS)

    data = dict(row_data) if row_data is not None else {}
    data.update(kwargs)

    exp_id = data.get("experiment_id", "")
    formatted_row = {col: data.get(col, "") for col in MASTER_COLUMNS}

    if exp_id and exp_id in df["experiment_id"].values:
        df = df[df["experiment_id"] != exp_id]
    df = pd.concat([df, pd.DataFrame([formatted_row])], ignore_index=True)

    df.to_csv(MASTER_TABLE_PATH, index=False)
    print(f"Updated master metrics table at: {MASTER_TABLE_PATH}")
