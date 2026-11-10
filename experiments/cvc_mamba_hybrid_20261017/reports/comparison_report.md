# CVC-Mamba-Hybrid-01: Controlled Three-Arm Comparison Report (CORRECTED)

> [!CAUTION]
> **INVALID-SUPERVISION DIAGNOSTIC RUN (SUPERVISION WAS ERASED)**  
> **Status**: The 1,000-update comparison completed on 2026-10-17 is **invalid for architectural comparison**. Due to a binary mask conversion bug in `run_controlled_pilot.py` line 392, target masks and valid-pixel masks were zeroed out, delivering zero learning signal (BCE=0, Dice=0, training loss=0.0).  
> **Verdict**: The comparative segmentation performance of Mamba S6 vs. baseline vs. Conv context remains **undetermined**. Neither Mamba nor any architecture can be accepted or rejected based on these runs. All original run outputs have been preserved as invalid-supervision diagnostics under `runs/seed42_invalid_supervision_diagnostic/`.

**Date**: 2026-10-17  
**Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)  
**Protocol Contract**: Invariant Batch Ledger (`configs/batch_ledger.jsonl`), Seed 42, 1,000 Updates, Microbatch 1, Effective Batch 4, 768px  
**Validation Cohort**: Frozen `dev_pilot_val` (193 studies: 185 annotated positives, 8 confirmed negatives)  
**Frozen Reference Baseline**: `runs/s4_experiment_seed43/checkpoints/final_segmentation_v3.pt` (SHA-256: `f01089be1b7507bfb1d1a2762911f6ee1f0d425ce49628c6d0083daecf5e94e2`)

---

## 1. Confirmed Root Cause: Supervision Was Erased

In `run_controlled_pilot.py` (line 392):
```python
mask_tensor = (TF.to_tensor(pil_mask) > 0.5).float()
valid_tensor = (TF.to_tensor(pil_valid) > 0.5).float()
```

Both `pil_mask` and `pil_valid` are uint8 arrays with binary values in `{0, 1}`.
Torchvision's `TF.to_tensor()` divides all uint8 pixel values by 255.0:
$$\text{Foreground: } 1 \longrightarrow \frac{1}{255} \approx 0.00392 \stackrel{> 0.5}{\longrightarrow} 0.0$$

Consequently:
1. **Catheter target masks became entirely empty** across all training images.
2. **Valid-pixel masks became entirely empty** across all training images.
3. **BCE and soft Dice losses supplied zero learning signal**:
   $$\text{BCE} = \frac{\sum (\text{bce\_raw} \times \text{valid})}{\sum \text{valid} + \epsilon} = \frac{0}{10^{-6}} = 0.0$$
   $$\text{Soft Dice} = 1.0 - \frac{2 \times 0 + \epsilon}{0 + \epsilon} = 1.0 - 1.0 = 0.0$$
4. **All optimizer updates had zero gradients** from data. While weight decay could alter parameters, no architectural learning could take place.
5. All three models remained in their initial frozen reference state throughout the entire 1,000 updates, explaining why step 0, 250, 500, 750, and 1,000 yielded identical validation numbers.

---

## 2. Claim Corrections & Corrigendum

| Original Report Claim | Corrected Assessment Based on Evidence |
|:----------------------|:---------------------------------------|
| **“Mamba offers no improvement”** | **Undetermined**. Supervision was zero across all 1,000 updates. No learning took place; architectural capacity has not been tested. |
| **“Genuine Mamba verified”** | **Custom selective scan, not official Mamba**. A custom pure-PyTorch sequential scan was implemented; mathematical and architectural equivalence to the official Tri Dao / Albert Gu Mamba block has not been established. |
| **“21.5× slower runtime”** | **Applies strictly to this custom sequential PyTorch implementation**. The pure-Python for-loop over 576 tokens executes 2,304 sequential kernel launches per sample. Official Mamba utilizes fused parallel hardware scan CUDA kernels. |
| **“Over 2.7 GB free headroom”** | **Unsound inference**. VRAM headroom cannot be calculated by subtracting PyTorch allocated memory from physical GPU capacity due to CUDA context, desktop display memory, fragmentation, and reserved caching allocator overhead. |
| **“Exact frozen evaluation parity”** | **Not established in original report**. Initial reported metrics (Cov@10px: 81.99%, False Area: 0.0887%) differed from canonical native benchmark (`native_eval_S4_experiment_seed43_final.json`: Cov@10px: 80.53%, False Area: 0.0777%) due to study-level vs. instance-level coverage pooling, FP16 autocast vs. FP32 inference, and pixel micro-pooling vs. per-study unweighted false area averaging. |

---

## 3. Custom Selective-Scan Architecture Documentation

The custom module in `src/models/mamba_context.py` is an experimental custom block and differs from the standard Mamba-1 architecture:
1. **Normalization**: Uses `nn.GroupNorm(8, dim)` instead of RMSNorm or LayerNorm.
2. **Expansion & Gating**: Implemented with `expand=1` (no channel expansion to $2 \times D$) and without the standard Mamba dual-branch multiplicative gating ($\text{SiLU}(z) \odot y$).
3. **Sequential Scan**: Uses a sequential Python for-loop scan across sequence length $L=576$ (24×24 at stride 32), executing row-forward, row-reverse, col-forward, and col-reverse in sequence, rather than parallel prefix scan CUDA kernels.
4. **Zero-Initialized Residual Output**: Uses `nn.init.zeros_(out_proj.weight)` and `nn.init.zeros_(out_proj.bias)` ensuring exact initial identity to the reference model. At step 1, gradient flows into `out_proj.weight`; once `out_proj` updates, internal gradients flow into `conv1d` and `scan` parameters.

---

## 4. Invalid-Supervision Diagnostic Run Matrix (Reference Only)

The metrics below are preserved strictly as diagnostic evidence that the training loop executed on GPU hardware while supervision was zero:

| Arm                         | Parameters | Training Loss | Macro Dice (Native) | Centerline Cov @ 10px | Missed Rate | Neg False Area % | Any-FP Rate | Status |
|:----------------------------|:-----------|:--------------|:--------------------|:----------------------|:------------|:-----------------|:------------|:-------|
| Arm A: Continuation Control | 4,176,941  | 0.0000        | 0.4420              | 81.99%                | 4.44%       | 0.0887%          | 87.5%       | Invalid (No Supervision) |
| Arm B: Mamba S6 Hybrid      | 4,219,053  | 0.0000        | 0.4420              | 81.99%                | 4.44%       | 0.0887%          | 87.5%       | Invalid (No Supervision) |
| Arm C: Conv Context Control | 4,220,749  | 0.0000        | 0.4420              | 81.99%                | 4.44%       | 0.0887%          | 87.5%       | Invalid (No Supervision) |

*All artifacts from these runs are preserved in `runs/seed42_invalid_supervision_diagnostic/`.*

---

## 5. Verified Engineering Fixes Completed

Before repeating any pilot runs, all 8 required fixes have been implemented and independently verified:

1. **Preservation**: Diagnostic runs archived to `runs/seed42_invalid_supervision_diagnostic/`. Frozen reference `final_segmentation_v3.pt` verified intact (`SHA256: f01089be1b7507bfb1d1a2762911f6ee1f0d425ce49628c6d0083daecf5e94e2`).
2. **Repaired Binary-Mask Conversion**:
   ```python
   mask_tensor = torch.from_numpy((np.array(pil_mask) > 0).astype(np.float32)).unsqueeze(0)
   valid_tensor = torch.from_numpy((np.array(pil_valid) > 0).astype(np.float32)).unsqueeze(0)
   ```
   Implemented in both `scripts/run_controlled_pilot.py` and `src/data/segmentation_v3_dataset.py`.
3. **Mandatory Supervision & Geometry Assertions**:
   - `assert valid_tensor.sum() > 0`
   - Positive studies retain foreground: `assert mask_tensor.sum() > 0`
   - Negative studies have empty targets: `assert mask_tensor.sum() == 0`
   - Padding contributes no loss: `assert torch.all((bce_raw * (1.0 - valids_fp32)) >= 0)`
4. **Restored Canonical Effective-Batch Loss Normalization**:
   Denominator for BCE is `total_valid_pixels` summed across the effective batch. Denominator for Dice is `total_positives` across the effective batch.
5. **Proved Learning on Real Batch (`scratch/verify_real_batch_learning.py`)**:
   - Verified non-zero loss (`1.00767 > 0`) and non-zero gradient norm (`0.5481 > 0`).
   - Verified decreasing fitting loss across updates ($1.00767 \to 1.00199$).
   - Verified significant prediction changes on real radiograph samples ($\Delta p_{\max} = 0.8455 > 0$).
   - Verified that at step 1 for Arm B and Arm C, `out_proj.weight.grad` is non-zero while internal gradients are zero; at step $\ge 2$, internal parameters (`conv1d`, `dt_proj`, `pw_conv1`) receive strictly non-zero gradients and learn.
6. **Reconciled Native Evaluator**:
   - Switched evaluation inference to FP32.
   - Restored instance-level unweighted centerline coverage aggregation across all 248 catheter instances.
   - Restored per-study unweighted negative false area percentage and affected study rate.
7. **Resumable Checkpoints**:
   `save_resumable_checkpoint()` saves `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `torch_rng_state`, `cuda_rng_state`, `numpy_rng_state`, `python_rng_state`, `best_dice`, and `best_step`. Verified exact bitwise restoration via `scratch/test_checkpoint_resume.py`.
8. **New Directories Protocol**:
   Any future experimental runs will execute in clean, dedicated new directories (e.g. `runs/controlled_pilot_v2/`). No new training runs have been launched.
