"""Test Canonical Evaluator Parity for Architecture Comparison V2.

Verifies:
NativeValidator on dev_pilot_val reproduces canonical reference metrics for FROZEN_S4_CKPT:
- Centerline Coverage @ 10px: ~80.53% (canonical 80.5253%)
- Missed Catheter Rate @ 10px: ~4.44% (canonical 4.4355%)
- Negative False Area %: ~0.0777% (canonical 0.077652%)
- Macro Dice: ~0.4419 (canonical 0.441878)
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "architecture_comparison_v2"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from src.models.segmentation_v3 import CVCSegmentationV3Net
from experiments.architecture_comparison_v2.scripts.train_runner import NativeValidator, FROZEN_S4_CKPT

def test_canonical_eval_parity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Canonical Evaluator Parity against {FROZEN_S4_CKPT} on {device}...")

    ckpt = torch.load(FROZEN_S4_CKPT, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = CVCSegmentationV3Net(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    validator = NativeValidator(target_size=768, device=device)
    eval_res = validator.evaluate_model(model, threshold=0.50)
    s = eval_res["summary"]

    print("\n--- Evaluator Parity Results ---")
    print(f"Centerline Coverage @ 10px: {s['centerline_coverage_10px']*100:.4f}% (Canonical: 80.5253%)")
    print(f"Missed Catheter Rate:       {s['missed_catheter_rate']:.4f}% (Canonical: 4.4355%)")
    print(f"Negative False Area %:      {s['negative_false_area_pct']:.6f}% (Canonical: 0.077652%)")
    print(f"Negative Total False Pixels:{s['negative_total_false_pixels']} px (Canonical: 45714 px)")
    print(f"Macro Dice:                 {s['macro_dice']:.6f} (Canonical: 0.441878)")

    cov_diff = abs(s['centerline_coverage_10px']*100.0 - 80.525278)
    miss_diff = abs(s['missed_catheter_rate'] - 4.435484)
    neg_diff = abs(s['negative_false_area_pct'] - 0.077652)
    dice_diff = abs(s['macro_dice'] - 0.441878)

    assert cov_diff < 0.01, f"Centerline coverage diverged: {cov_diff}"
    assert miss_diff < 0.01, f"Missed catheter rate diverged: {miss_diff}"
    assert neg_diff < 0.001, f"Negative false area diverged: {neg_diff}"
    assert dice_diff < 0.001, f"Macro dice diverged: {dice_diff}"

    print(f"\n[SUCCESS] Canonical Evaluator Parity Verified! Coverage diff: {cov_diff:.6f}%, False Area diff: {neg_diff:.6f}%, Dice diff: {dice_diff:.6f}.")


if __name__ == "__main__":
    test_canonical_eval_parity()
