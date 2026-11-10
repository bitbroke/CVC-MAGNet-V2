# Architecture Comparison V2: Comprehensive Pilot Report
**EfficientNet-B0 Baseline vs. SegFormer-B0 vs. HRNet-W18**  
**Date:** October 18, 2026  
**Directory:** `experiments/architecture_comparison_v2/`

---

## 1. Executive Summary

Following the full repair of the shared training runner, a rigorous 5-step evaluation protocol was executed on the local GPU across two independent random seeds (42 and 43), with identical dataset exposures (400 positives + 396 negatives from `batch_ledger.jsonl`), effective batch size of 4, and letterbox resolution of 768×768.

**Core Findings:**
1. **Runner Learning Proof Succeeded**: All pre-flight invariants passed (strictly preserved 0/1 binary masks, non-empty targets, exact padding exclusion with $\Delta L = 0.0$ and $\nabla_{pad} L = 0.0$, bitwise exact resume, decreasing fixed-batch loss, and exact canonical parity against the frozen S4 reference).
2. **HRNet-W18 is the Decisive Architectural Winner**:
   - **Macro Dice**: **0.4771** (Mean across seeds 42 & 43) vs. **0.3287** for the EfficientNet baseline (**+0.1484** absolute improvement, a **+45.1% relative gain**).
   - **Centerline Coverage @ 10px**: **87.24%** vs. **77.82%** for baseline (**+9.42%** improvement).
   - **Negative False Area %**: **0.0430%** vs. **0.5345%** for baseline (**over 12× reduction in background false positives**).
   - **Outperformed Frozen S4 Reference**: HRNet-W18 trained for only 1,000 updates from ImageNet weights surpassed the historical trained S4 reference across all metrics (Dice 0.4771 vs. 0.4419; Coverage 87.24% vs. 80.53%; False Area 0.0430% vs. 0.0777%).
3. **SegFormer-B0 Underperformed on Fine Tubular Catheters**:
   - Although SegFormer achieved lower false-positive area on negatives (0.1590%) than EfficientNet, its attention-downsampled representations struggled on thin continuous lines: **Missed Catheter Rate reached 10.89%** and **Recall dropped to 0.5309** (vs. 0.8361 for HRNet).
4. **VRAM and Compute Efficiency**:
   - All three models easily fit within the 4 GB VRAM limit at 768×768 in FP16 (Peak VRAM: EfficientNet 863 MB, SegFormer 1,234 MB, HRNet 1,303 MB; runtime ~40–55 min per 1,000 updates).

---

## 2. Shared Runner Verification & Pre-Flight Invariants (Steps 1 & 2)

Prior to running full-budget pilots, the shared runner was validated across five strict unit tests:

| Test Script | Tested Condition | Result | Evidence |
| :--- | :--- | :---: | :--- |
| `test_padding_exclusion.py` | Perturb padding logits; measure $\Delta L$ and $\nabla_{pad} L$ | **PASSED** | $\Delta L = 0.000000000000$, $\max |\nabla_{pad} L| = 0.000000000000$ |
| `test_supervision_invariants.py` | Verify 0/1 float targets, non-empty valid regions, pos/neg invariants | **PASSED** | All 20 audit samples strictly $\{0.0, 1.0\}$; no float/255 conversion artifacts |
| `test_resume_and_protection.py` | Directory overwrite refusal & bitwise exact resume continuation | **PASSED** | Raised `FileExistsError` on existing dir; $\Delta L_{upd2} = 0.000000$, $\max |\Delta W| = 0.000000$ |
| `test_canonical_eval_parity.py` | Parity of `NativeValidator` against historical frozen S4 on `dev_pilot_val` | **PASSED** | Cov@10px: 80.5253% vs 80.5253% ($\Delta = 0.000000\%$); Miss: 4.4355% vs 4.4355% |
| `test_runner_learning_proof.py` | 30 real training updates: loss reduction, finite GN, $\Delta W > 0$, eval() output change | **PASSED** | Loss decreased $1.8724 \to 1.6157$; 252/396 tensors updated; eval() $\Delta P_{max} = 0.414$ |

---

## 3. Comprehensive Metric Comparison (Steps 3, 4, and 5)

All evaluations conducted on `dev_pilot_val` (185 positive studies with 248 ground-truth catheters + 8 negative studies) at native radiograph resolution using a fixed threshold $\tau = 0.50$.

### Multi-Model Pilot Results Table

| Model | Seed | Updates | Macro Dice | Centerline Cov @ 10px | Missed Catheter Rate | False Area on Negatives | Precision | Recall | Peak VRAM | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** *(Baseline)* | 42 | 1,000 | 0.3251 | 77.28% | 3.23% | 0.4469% | 0.2117 | 0.7334 | 863 MB | 39.4 min |
| **SegFormer-B0** | 42 | 1,000 | 0.3693 | 61.32% | 10.89% | 0.1590% | 0.3021 | 0.5309 | 1,234 MB | 40.9 min |
| **HRNet-W18** | 42 | 1,000 | **0.4732** | **87.18%** | **3.63%** | **0.0420%** | **0.3438** | **0.8361** | 1,303 MB | 42.3 min |
| *Frozen S4 Reference* | *frozen* | — | *0.4419* | *80.53%* | *4.44%* | *0.0777%* | *0.3191* | *0.7621* | — | — |
| **EfficientNet-B0** *(Confirmation)* | 43 | 1,000 | 0.3322 | 78.35% | 3.63% | 0.6220% | 0.2139 | 0.7380 | 863 MB | 44.8 min |
| **HRNet-W18** *(Confirmation)* | 43 | 1,000 | **0.4809** | **87.30%** | **4.03%** | **0.0439%** | **0.3567** | **0.8331** | 1,303 MB | 57.4 min |

### Mean Performance Summary (Across Seeds 42 and 43)

| Model Architecture | Mean Macro Dice | Mean Cov @ 10px | Mean False Area % | Mean Recall | Relative Dice Gain vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0 Baseline** | 0.3287 | 77.82% | 0.5345% | 0.7357 | Baseline |
| **SegFormer-B0** *(Seed 42 only)* | 0.3693 | 61.32% | 0.1590% | 0.5309 | +12.4% |
| **HRNet-W18** | **0.4771** | **87.24%** | **0.0430%** | **0.8346** | **+45.1%** |
| *Frozen S4 Reference (Benchmark)* | *0.4419* | *80.53%* | *0.0777%* | *0.7621* | *+34.4%* |

---

## 4. In-Depth Architectural Analysis

### Why HRNet-W18 Dominates
1. **High-Resolution Representation Maintained Throughout**:
   Unlike standard UNet/FPN architectures that aggressively downsample feature maps to $1/32$ and attempt to reconstruct fine boundaries through upsampling, HRNet maintains parallel high-resolution representations ($1/4$ resolution) across all stages with continuous multi-scale fusion.
2. **Continuous Catheter Topology**:
   CVC catheters in thoracic radiographs are 1- to 3-pixel wide tubular structures traversing high-contrast bone and soft tissue. In HRNet, fine feature maps never lose structural spatial precision, leading to **87.24% centerline coverage** and **0.8346 recall**.
3. **Drastic False Positive Suppression**:
   HRNet achieved a negative false area of only **0.0430%** (an order of magnitude lower than EfficientNet's 0.5345%). By maintaining fine anatomical context, it effectively rejects clavicle edges, rib borders, and ECG leads that easily fool standard CNNs.

### Why SegFormer-B0 Underperformed
- SegFormer relies on spatial reduction attention (SRA) in its MiT-B0 encoder. While transformer self-attention excels at global context (producing relatively clean backgrounds on negative images), the spatial reduction blurs precise linear catheter traces.
- This resulted in an unacceptably high **missed catheter rate of 10.89%** and low recall (0.5309).

---

## 5. Qualitative Visual Overlay Gallery

Representative comparison panels are rendered at full resolution under `reports/visualizations/`:
- `case_01_pos_903644095396.png`: High-density mediastinum with subtle CVC catheter. HRNet traces the entire path seamlessly; EfficientNet produces patchy fragments.
- `case_02_pos_785055284143.png`: Dual-catheter case showing distinct separation and distal tip continuity.
- `case_03_pos_938968452713.png`: Right internal jugular CVC with heavy clavicular bone overlay.
- `case_04_neg_922892522748.png`: True negative radiograph with prominent ECG leads and rib contours. EfficientNet predicts false positive strokes; HRNet outputs an entirely empty mask (0 false pixels).
- `case_05_neg_923653562395.png`: True negative case confirming zero spurious activations for HRNet.

---

## 6. Recommendations & Next Steps

1. **Select HRNet as the Standard Backbone**:
   - HRNet-W18 should be selected as the definitive backbone for CVC segmentation and downstream tip localization.
2. **Precise Tip Localization**:
   - Leverage the preserved high-resolution feature maps ($1/4$ resolution) from HRNet to train the distal tip detection head (`T1`/`T2` joint heatmap head). The continuous spatial alignment in HRNet eliminates the interpolation error inherent in standard decoders.
3. **Custom Mamba Rerun**:
   - With the baseline established and HRNet setting a high-water mark of **0.4771 Dice / 87.24% Coverage**, any future exploration of custom State-Space/Mamba blocks should be evaluated specifically against HRNet-W18 under this exact verified runner.
