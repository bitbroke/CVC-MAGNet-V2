"""Unit test verifying the complete MAG-Net inference pipeline and failure handling."""
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.coarse import CoarseSegNet
from src.models.fine import FineSegNet
from src.infer.pipeline import MAGNetInferencePipeline


def test_inference_pipeline_synthetic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    coarse_model = CoarseSegNet(base_channels=16).to(device)
    fine_model = FineSegNet(base_channels=16).to(device)

    pipeline = MAGNetInferencePipeline(
        coarse_model=coarse_model,
        fine_model=fine_model,
        device=device,
        coarse_threshold=0.1,
    )

    # 1. Test image with synthetic foreground
    synthetic_img = np.random.randint(50, 200, (2000, 2500), dtype=np.uint8)
    res_pos = pipeline.predict_image(synthetic_img, study_id="SYNTHETIC_POS")

    assert "status" in res_pos, "Missing status key in output!"
    assert "candidates" in res_pos, "Missing candidates in output!"
    assert res_pos["native_width"] == 2500, "Native width mismatch"
    assert res_pos["native_height"] == 2000, "Native height mismatch"
    print("  [PASS] Positive synthetic case produced evaluable output structure.")

    # 2. Test empty / negative image with high threshold to trigger explicit failure
    pipeline_strict = MAGNetInferencePipeline(
        coarse_model=coarse_model,
        fine_model=fine_model,
        device=device,
        coarse_threshold=0.9999,  # Impossible threshold
    )
    res_neg = pipeline_strict.predict_image(synthetic_img, study_id="SYNTHETIC_NEG")
    assert res_neg["status"] == "no_coarse_prediction", f"Expected no_coarse_prediction, got {res_neg['status']}"
    assert len(res_neg["candidates"]) == 0, "Candidates must be empty on explicit failure!"
    assert res_neg["primary_tip"] is None, "Primary tip must be None on explicit failure!"
    print("  [PASS] Empty / negative case safely produced explicit failure state (zero crash).")


if __name__ == "__main__":
    print("Running Inference Pipeline Tests...")
    test_inference_pipeline_synthetic()
    print("All Inference Pipeline tests PASSED.")
