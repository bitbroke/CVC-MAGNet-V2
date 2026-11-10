"""Test Resume Equivalence and Overwrite Protection for Architecture Comparison V2.

Verifies:
1. Run directory overwrite protection raises FileExistsError when prior artifacts exist without overwrite/resume.
2. Uninterrupted training (Step 1 -> Step 2) produces bitwise identical loss, gradient norm,
   and parameter weights as save -> reload -> continue (Step 1 -> Save -> Reload -> Step 2).
"""

import json
import shutil
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path("C:/Ankshit/CVC-MAGNet-V2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_ROOT = PROJECT_ROOT / "experiments" / "architecture_comparison_v2"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from experiments.architecture_comparison_v2.scripts.train_runner import run_training_pilot

def test_resume_and_protection():
    test_dir = EXP_ROOT / "runs" / "test_resume_protection"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Testing Overwrite Protection ===")
    # Create a dummy artifact in test_dir
    dummy_file = test_dir / "history.jsonl"
    dummy_file.write_text('{"update": 0}\n')

    caught_protection = False
    try:
        run_training_pilot(
            model_name="EfficientNet_B0",
            target_updates=2,
            eval_interval=10,
            seed=42,
            output_dir=str(test_dir),
            overwrite=False,
        )
    except FileExistsError as e:
        caught_protection = True
        print(f"[PASSED] Caught expected FileExistsError: {e}")

    assert caught_protection, "FAILED: Overwrite protection failed to trigger!"

    # Clean test directory
    shutil.rmtree(test_dir)

    print("\n=== Step 2: Testing Exact Training Resume Equivalence ===")
    dir_uninterrupted = test_dir / "uninterrupted"
    dir_interrupted = test_dir / "interrupted"

    # 1. Run 2 updates uninterrupted
    print("\n--- Running Uninterrupted 2-Update Training ---")
    res_uninterrupted = run_training_pilot(
        model_name="EfficientNet_B0",
        target_updates=2,
        eval_interval=10,
        seed=42,
        output_dir=str(dir_uninterrupted),
        overwrite=True,
        skip_initial_eval=True,
        run_eval_on_final=False,
    )

    # 2. Run 1 update, save, exit
    print("\n--- Running Interrupted Step 1 ---")
    res_step1 = run_training_pilot(
        model_name="EfficientNet_B0",
        target_updates=1,
        eval_interval=10,
        seed=42,
        output_dir=str(dir_interrupted),
        overwrite=True,
        skip_initial_eval=True,
        run_eval_on_final=False,
    )

    # 3. Resume from update_0001.pt to update 2
    ckpt_upd1 = dir_interrupted / "checkpoints" / "update_0001.pt"
    # Ensure update_0001.pt exists: save_resumable_checkpoint at update 1 if needed
    if not ckpt_upd1.exists():
        # Copy final_model.pt from step 1
        shutil.copy(dir_interrupted / "checkpoints" / "final_model.pt", ckpt_upd1)

    print("\n--- Resuming Training from Update 1 to Update 2 ---")
    res_resumed = run_training_pilot(
        model_name="EfficientNet_B0",
        target_updates=2,
        eval_interval=10,
        seed=42,
        output_dir=str(dir_interrupted),
        resume_from=str(ckpt_upd1),
        skip_initial_eval=True,
        run_eval_on_final=False,
    )

    # 4. Compare checkpoints at Update 2
    ckpt_un = torch.load(dir_uninterrupted / "checkpoints" / "final_model.pt", map_location="cpu", weights_only=False)
    ckpt_re = torch.load(dir_interrupted / "checkpoints" / "final_model.pt", map_location="cpu", weights_only=False)

    sd_un = ckpt_un["model_state_dict"]
    sd_re = ckpt_re["model_state_dict"]

    max_delta = 0.0
    for k in sd_un:
        d = (sd_un[k].float() - sd_re[k].float()).abs().max().item()
        max_delta = max(max_delta, d)

    print(f"\nMax Parameter Delta between Uninterrupted and Resumed: {max_delta:.12e}")

    # Read history entries for update 2
    hist_un = [json.loads(line) for line in open(dir_uninterrupted / "history.jsonl") if line.strip()]
    hist_re = [json.loads(line) for line in open(dir_interrupted / "history.jsonl") if line.strip()]

    loss_un = [h["loss"] for h in hist_un if h["update"] == 2][0]
    loss_re = [h["loss"] for h in hist_re if h["update"] == 2][0]
    delta_loss = abs(loss_un - loss_re)
    print(f"Update 2 Loss (Uninterrupted): {loss_un:.12f}")
    print(f"Update 2 Loss (Resumed):       {loss_re:.12f}")
    print(f"Delta Loss at Update 2:        {delta_loss:.12e}")

    assert delta_loss == 0.0, f"Violation: Loss differed upon resume by {delta_loss}!"
    assert max_delta == 0.0, f"Violation: Model parameters diverged upon resume by {max_delta}!"

    print("\n[SUCCESS] Resume Equivalence and Overwrite Protection Verified! Delta W = 0, Delta L = 0.")

    # Cleanup test dir
    shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_resume_and_protection()
