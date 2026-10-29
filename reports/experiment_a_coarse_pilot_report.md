# Experiment A: Stage 1 Coarse Localization Pilot Report

**Date**: 2026-10-11  
**Project**: CVC-MAGNet-V2  
**Device**: NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM)  
**Status**: Completed — **Contract Stop Condition Triggered**

---

## 1. Executive Summary

Experiment A (Stage 1 Coarse Localization Pilot) trained `CoarseSegNet` (base-32 U-Net encoder-decoder with Pyramid Pooling Module) for 1,000 step-controlled optimizer updates on 996 development studies (600 CVC-positive, 600 confirmed negative) using batch size 4 (4,000 supervised training examples exposed).

While the preflight checks (hardware profiling, edge case handling, and tiny-set learnability) passed 100%, the **pilot validation evaluations confirm the contract stop condition**:
> **Stop Condition**: *If coarse Dice fails to exceed baseline proposal quality, stop before fine-stage training.*

### Key Quantitative Findings

| Metric | Frozen V1 Baseline | Experiment A (Step 200) | Experiment A (Step 600) | Experiment A (Step 1000) | Evaluation Contract Parity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All-Target Success@15 (Native px)** | **93/248 (37.50%)** | 0/248 (0.00%) | 0/248 (0.00%) | **0/248 (0.00%)** | Exact ($N=248$ assessable targets) |
| **Validation Mask Dice** | N/A (proposals) | 0.0065 | 0.0026 | **0.0063** | Evaluated on full radiographs |
| **True Tip-in-Crop Coverage** | 114/248 (45.97%) | 248/248 (100.0%)* | 248/248 (100.0%)* | 248/248 (100.0%)* | *Trivial full-field fallback |
| **Ungated Candidate Success@15** | 96/248 (38.71%) | 0/248 (0.00%) | 0/248 (0.00%) | **0/248 (0.00%)** | Unfiltered endpoint match |
| **Matched Median Error** | 6.43 px | N/A (0 matches) | N/A (0 matches) | N/A (0 matches) | Constrained Hungarian |
| **Matched 95th Percentile Error** | 14.34 px | N/A (0 matches) | N/A (0 matches) | N/A (0 matches) | Pairs $\le 15$px |
| **All-Target Nearest p95 Error** | 3000.0 px | 3000.0 px | 3000.0 px | **3000.0 px** | Captures tail localization errors |
| **Negative Acceptance Rate** | 1/8 (12.50%) | 4/8 (50.0%) | 0/8 (0.00%) | **0/8 (0.00%)** | Confirmed negative studies |
| **Training Step Time** | N/A | 338 ms/step | 318 ms/step | **320 ms/step** | PyTorch AMP mixed precision |
| **Peak GPU VRAM** | ~3.8 GB | 1542 MB | 1542 MB | **1542 MB** | Within 4000 MB budget |

---

## 2. Visual Diagnosis: Anatomical Edge Collapse vs. Catheter Tubular Geometry

Visual overlays saved to `reports/overlays/pilot_coarse_val_step_1000.png` reveal the exact failure mode that the passing tiny-set smoke check masked:

1. **High-Frequency Anatomical Texture Bias**:
   - Instead of learning thin tubular line geometry (which constitutes only ~0.26% of pixels in a radiograph), the uninitialized encoder-decoder prioritized high-contrast anatomical edges: rib borders, clavicles, spinal columns, and diaphragmatic contours.
2. **Diffuse Probability Distribution**:
   - At threshold $\tau = 0.30$, predicted probability maps illuminate rib and lung contours across both hemithoraces.
   - Because the segmentation does not resolve into thin, continuous catheter tracts, the downstream morphological skeletonizer and distal tip extractor (`extract_distal_tips`) either rejected the noisy contours or extracted anatomical spur endpoints far from the true distal catheter tips.
3. **Loss Deception**:
   - Total loss decreased by 27.4% (from 1.7995 to 1.3061), driven almost entirely by BCE loss decreasing from 0.8009 to 0.3070 as background probabilities were suppressed.
   - However, Dice loss remained effectively flat (0.9986 $\to$ 0.9990), confirming that true catheter foreground overlap was never established during this pilot budget.

---

## 3. Contractual Compliance Review

All blockers identified prior to the pilot have been strictly repaired and independently verified:

1. **Annotation Provenance**:
   - `scripts/01_build_inventory.py` preserved all 59 unreviewed diagnostic cases as `diagnostic_packet_awaiting_audit`; 0 unverified annotations were promoted to expert status.
   - Silent `(2500, 2500)` fallback eliminated; missing dimensions raise explicit `RuntimeError`.
2. **Exposure Labels**:
   - Fold 4 is formally labeled `previously_exposed_held_out` with documented notes: *"Previously exposed in V1 fitting (Folds 1-4); held out from V2 fitting. Cannot support a genuinely untouched project-level test claim."*
3. **Physical Units**:
   - Removed all unsupported physical millimeter claims ("≈2.5mm"). 10px is explicitly specified as an empirical rasterization hyperparameter.
4. **Image/Mask Geometry**:
   - Implemented OpenCV pixel-center continuous mapping `(x + 0.5) * scale - 0.5 + pad_left` with exact integer padding. Landmark resampling error dropped to 0.0012 px on rectangular/odd-sized images.
   - Coarse mask unpadding and resizing corrected before co-extracting crops. Out-of-bounds coordinates are preserved as invalid instead of clamped onto image borders.
5. **Evaluator Contract**:
   - Hungarian assignment restored to threshold-constrained optimization (`cost_matrix = np.where(dists <= radius, dists, 1e9)`). Verified against frozen V1 predictions with exact bitwise reproduction (37.5%, 93/248).
   - Tip-in-crop coverage reported separately from ungated Success@15.
   - Matched p95 and all-target nearest p95 reported side-by-side.
6. **Checkpoint Hardening**:
   - Atomic replacement (`os.replace`), CPU-pinned ByteTensor RNG state, scheduler and sampler state persistence tested and verified.

---

## 4. Next Step Recommendation: Adhere to Stop Condition

Per the experimental protocol:
- **Do NOT launch fine-stage training** (`FineSegNet`) on top of this coarse model checkpoint. Feeding diffuse rib/edge probability crops into the fine stage will only propagate noise into the attention mechanism.
- The pipeline inputs, coordinate transforms, evaluator, and checkpointing infrastructure are now verified, hardened, and trustworthy.
- Future investigations must focus on catheter foreground segmentation representations (e.g., pretrained encoder backbones, directional tubular filters, or higher positive class focus) before advancing to multi-stage refinement.
