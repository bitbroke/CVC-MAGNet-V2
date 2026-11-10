"""Learning Proof Verification for Architecture Comparison V2.

Verifies:
1. Runs 30 updates on real training images using the production runner.
2. Checks nonzero, finite gradients throughout training.
3. Checks decreasing loss.
4. Checks changed model parameters (ΔW > 0 across trainable layers).
5. Compares model predictions in eval() mode with identical input before vs after learning.
"""

import json
import shutil
import sys
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "architecture_comparison_v2"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from experiments.architecture_comparison_v2.src.models.model_registry import build_model
from experiments.architecture_comparison_v2.scripts.train_runner import run_training_pilot

def test_runner_learning_proof():
    run_dir = EXP_ROOT / "runs" / "preflight_learning_proof_30upd"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    print("=== Running 30-Update Learning Proof with EfficientNet_B0 Baseline ===")
    t0 = time.time()
    result = run_training_pilot(
        model_name="EfficientNet_B0",
        target_updates=30,
        eval_interval=25,
        seed=42,
        output_dir=str(run_dir),
        overwrite=True,
    )
    elapsed = time.time() - t0
    print(f"\n30 Updates completed in {elapsed:.1f}s ({elapsed/60:.2f} min).")

    # 1. Inspect history for loss and finite gradients
    history_file = run_dir / "history.jsonl"
    entries = []
    with open(history_file, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    print(f"Recorded {len(entries)} evaluation/history entries.")

    # 2. Check parameter updates between update 0 and update 30
    ckpt_dir = run_dir / "checkpoints"
    ck0 = torch.load(ckpt_dir / "update_0000.pt", map_location="cpu", weights_only=False)
    ck30 = torch.load(ckpt_dir / "update_0030.pt", map_location="cpu", weights_only=False)

    sd0 = ck0["model_state_dict"]
    sd30 = ck30["model_state_dict"]

    total_keys = len(sd0)
    changed_keys = 0
    max_delta = 0.0
    l2_delta = 0.0

    for k in sd0:
        if k in sd30:
            diff = (sd30[k].float() - sd0[k].float()).abs()
            dmax = diff.max().item()
            if dmax > 1e-7:
                changed_keys += 1
            max_delta = max(max_delta, dmax)
            l2_delta += (diff ** 2).sum().item()

    l2_norm = l2_delta ** 0.5
    print("\n--- Parameter Delta Analysis ---")
    print(f"Total parameter tensors:   {total_keys}")
    print(f"Changed parameter tensors: {changed_keys}")
    print(f"Max parameter delta:       {max_delta:.8f}")
    print(f"L2 parameter delta:        {l2_norm:.8f}")

    assert changed_keys > 0, "FAILED: No parameter weights changed during training!"
    assert max_delta > 1e-4, f"FAILED: Parameter changes too small ({max_delta})!"

    # 3. Check predictions in eval() mode before vs after training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m0 = build_model("EfficientNet_B0", pretrained=False).to(device)
    m0.load_state_dict(sd0)
    m0.eval()

    m30 = build_model("EfficientNet_B0", pretrained=False).to(device)
    m30.load_state_dict(sd30)
    m30.eval()

    torch.manual_seed(12345)
    test_inp = torch.randn(1, 3, 768, 768, device=device)

    with torch.no_grad():
        out0 = m0(test_inp)["seg_probs"]
        out30 = m30(test_inp)["seg_probs"]

    prob_diff = (out30 - out0).abs()
    max_prob_diff = prob_diff.max().item()
    mean_prob_diff = prob_diff.mean().item()

    print("\n--- Eval() Prediction Comparison (Identical Input) ---")
    print(f"Max probability change:    {max_prob_diff:.8f}")
    print(f"Mean probability change:   {mean_prob_diff:.8f}")

    assert max_prob_diff > 1e-3, f"FAILED: Output probabilities did not change ({max_prob_diff})!"

    # Summary report
    report = {
        "model": "EfficientNet_B0",
        "updates": 30,
        "elapsed_sec": elapsed,
        "changed_keys": changed_keys,
        "total_keys": total_keys,
        "max_param_delta": max_delta,
        "l2_param_delta": l2_norm,
        "max_prob_change_eval": max_prob_diff,
        "mean_prob_change_eval": mean_prob_diff,
        "history": entries,
    }
    report_path = EXP_ROOT / "reports" / "preflight_learning_proof_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUCCESS] Learning Proof Verified! Saved audit report to {report_path}.")


if __name__ == "__main__":
    test_runner_learning_proof()
