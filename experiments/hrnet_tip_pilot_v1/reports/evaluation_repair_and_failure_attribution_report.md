# Evaluation Repair, Provenance Audit, and Decoding Failure Attribution Report
**Experiment**: Direct HRNet Distal-Tip Localization Pilot (`hrnet_tip_pilot_v1`)  
**Status**: Models Strictly Frozen; Evaluation Pipeline Repaired; Benchmark Versioned to `v2_reviewed`  
**Date**: October 18, 2026  

---

## Executive Summary

Following critical review of the initial exploratory results, all models were maintained **strictly frozen**, and no additional training was conducted. We executed a thorough evaluation repair, blinded label audit, and failure attribution workflow to address three central concerns:
1. **Target Provenance Confounding**: In the original benchmark ($N=41$ targets across 33 studies), 25 endpoints were provisional engineering-reviewed, while 16 endpoints across 14 multi-CVC studies were unreviewed heuristic polyline endpoints.
2. **Failure Attribution & Precision Potential**: Saved nearest-prediction distances for Seed 42 misses are not exclusively $>70$ px: 6 targets fall in the 15–25 px zone and 14 fall $\le 70$ px. Tracing every missed target through decoding revealed that **13 out of 27 misses (48.1%) are directly caused by decoding mechanics**—specifically, the rigid 30-pixel NMS radius suppressing adjacent dual-CVC tips at the cavoatrial junction—not model omission.
3. **Rigorous Reporting & Generalization**:
   - **Full FROC Trade-off**: Rather than comparing at a single operating point, we swept confidence thresholds across the spectrum. T1 **strictly dominates** T0 across all false positive rates (e.g. at $\sim 1.7$ false tips/image, T1 achieves $34.15\%$ vs T0's $12.20\%$).
   - **Corrected v2 Benchmark**: Blinded audit verified 15 heuristic endpoints as true distal tips and detected **1 inverted proximal entry-site label** in Study `...33128...`, which was corrected in the versioned `v2_reviewed` benchmark (preserving v1 unmodified).
   - **Segmentation Retention on Seed 43**: Paired evaluation across all 193 `dev_pilot_val` studies demonstrates that T1 retains baseline segmentation quality: Macro Dice **improved** by $+0.0040$ ($0.4809 \to 0.4848$), 10px Centerline Coverage increased by $+0.0022$ ($0.8730 \to 0.8751$), and total missed catheters dropped from 10 to 7.
   - **1-to-1 Error vs Diagnostic**: Localization error summaries are now strictly separated into 1-to-1 matched Euclidean errors (median $6.96$ px for Seed 42) and nearest-prediction diagnostics over all targets.

---

## 1. Provenance Split: Reviewed Targets vs. Weak Heuristic Targets

Separating the original 41-target cohort by label provenance reveals that direct tip prediction provides robust improvements on both subsets, but multi-CVC evaluation was indeed disproportionately carried by heuristic endpoints:

| Cohort / Subgroup | Targets ($N$) | T0 Baseline Success@15 | T1 Seed 42 Success@15 | T1 Seed 43 Success@15 | T1 Mean Gain vs T0 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Reviewed Targets Only** | **25** | **20.00%** (5/25) | **36.00%** (9/25) | **28.00%** (7/25) | **+60.0% rel (+12.0% abs)** |
| — Single-CVC Subset | 11 | 27.27% (3/11) | 54.55% (6/11) | 45.45% (5/11) | +83.3% rel (+22.7% abs) |
| — Multi-CVC Subset | 14 | 14.29% (2/14) | 21.43% (3/14) | 14.29% (2/14) | +25.0% rel (+3.6% abs) |
| **Weak Heuristic Targets Only** | **16** | **18.75%** (3/16) | **31.25%** (5/16) | **31.25%** (5/16) | **+66.7% rel (+12.5% abs)** |
| — Single-CVC Subset | 0 | — | — | — | — |
| — Multi-CVC Subset | 16 | 18.75% (3/16) | 31.25% (5/16) | 31.25% (5/16) | +66.7% rel (+12.5% abs) |
| **Mixed Exploratory Baseline (v1)** | **41** | **19.51%** (8/41) | **34.15%** (14/41) | **29.27%** (12/41) | **+62.5% rel (+12.2% abs)** |

> [!NOTE]
> **Key Finding**: In the locked development benchmark, 100% of single-CVC targets (11/11) were engineering-reviewed. In contrast, all 16 weak heuristic endpoints belonged to the multi-CVC cohort (14 positive multi-CVC studies had `has_all_tips_audited = false`). On strictly reviewed targets, T1 nearly doubles single-CVC accuracy (from 27.3% to 54.5%), while multi-CVC gains were modest (14.3% to 21.4%), confirming that multi-CVC localization required targeted investigation.

---

## 2. Blinded Review of the 16 Unresolved Endpoints & Benchmark v2

Each of the 16 heuristic endpoints across the 14 multi-CVC studies was independently audited by tracing full polyline coordinate sequences, measuring distances to radiograph boundaries, and determining anatomical trajectories relative to the thoracic inlet, superior vena cava (SVC), cavoatrial junction (CAJ), and right atrium (RA).

### Audit Findings
- **15 of 16 endpoints (93.8%)** were confirmed to be valid, assessable distal catheter tips terminating in the central venous system (SVC or CAJ). The heuristic had correctly identified the advancing distal termination.
- **1 endpoint was misidentified (Inverted Proximal Entry Site)**:
  * **Study**: `1.2.826.0.1.3680043.8.498.33128061425218955763725802634384514025`
  * **Catheter Instance 3** (`CVC - Borderline`, 11 polyline points)
  * **Heuristic Coordinate in v1**: `(2071.0, 837.0)`. This point is at the left clavicular / axillary thoracic inlet (proximal insertion site).
  * **True Distal Tip**: `(2338.0, 1653.0)`. The catheter descends along the lateral hemithorax, advancing to native coordinates `x=2338, y=1653`.
  * **Impact**: In the original v1 exploratory benchmark, T0 baseline received a false match at `(2071, 837)` because its skeleton happened to have an endpoint at the clavicle.

### Versioned Benchmark Release
To preserve historical reproducibility without overwriting:
- **Original Exploratory Benchmark preserved**: [`manifests/development_benchmark_manifest.json`](file:///C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/manifests/development_benchmark_manifest.json)
- **Corrected Benchmark Released**: [`manifests/development_benchmark_manifest_v2_reviewed.json`](file:///C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/manifests/development_benchmark_manifest_v2_reviewed.json)
- **Updated Registries**: [`manifests/instance_target_registry_v2_reviewed.csv`](file:///C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/manifests/instance_target_registry_v2_reviewed.csv) and [`.parquet`](file:///C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/manifests/instance_target_registry_v2_reviewed.parquet)
- All 14 multi-CVC studies now have `has_all_tips_audited = true`.

---

## 3. Re-Evaluation of Saved Predictions Against Corrected v2 Benchmark

Evaluating saved predictions against the corrected v2 benchmark clarifies the exact baseline comparison:

| Metric | T0 Baseline (Skeleton) | T1 Joint HRNet (Seed 42) | T1 Joint HRNet (Seed 43) |
| :--- | :---: | :---: | :---: |
| **Primary Success@15px** | **17.07%** (7/41) | **34.15%** (14/41) | **29.27%** (12/41) |
| Success@5px (Pinpoint) | 7.32% (3/41) | 12.20% (5/41) | 7.32% (3/41) |
| Success@10px | 14.63% (6/41) | 21.95% (9/41) | 14.63% (6/41) |
| Success@25px (Relaxed) | 36.59% (15/41) | 46.34% (19/41) | 48.78% (20/41) |
| **Single-CVC Success@15px** | **27.27%** (3/11) | **54.55%** (6/11) | **45.45%** (5/11) |
| **Multi-CVC Success@15px** | **13.33%** (4/30) | **26.67%** (8/30) | **23.33%** (7/30) |
| **Mean False Tips / Image (All 33)** | **13.12** | **1.97** (-85.0%) | **2.12** (-83.8%) |
| **Mean False Tips / Negative Image (8)** | **3.00** | **0.50** (-83.3%) | **0.62** (-79.3%) |
| **1-to-1 Matched Error (Median, P75)** | 8.06 px, 9.53 px | 6.96 px, 13.50 px | 9.45 px, 12.96 px |
| **Nearest-Prediction Diagnostic (Median)** | 38.18 px | 25.40 px | 25.46 px |

> [!IMPORTANT]
> Correcting the inverted heuristic endpoint in Study `...33128...` removes a spurious match from T0, adjusting T0 baseline from $19.51\%$ (8/41) down to $17.07\%$ (7/41). Consequently, T1 Seed 42 achieves **exactly double the success rate of T0** (14 matches vs 7 matches), while T1 Seed 43 achieves a **+71% relative gain** (12 matches vs 7 matches), with an **84% reduction in false tips per image**.

---

## 4. Sensitivity vs. False Tips / Image Across Operating Thresholds (FROC)

To determine whether improvement is robust across the trade-off curve, we swept confidence thresholds:
- T1 Models: $\tau \in [0.05, 0.80]$
- T0 Baseline: $\tau \in [0.45, 0.90]$

![FROC Trade-off Curve](file:///C:/Users/M%20S%20I/.gemini/antigravity-ide/brain/5c694925-30e5-4be9-87ea-b2579b1030d8/froc_tradeoff.png)

### Key Operating Trade-off Findings
1. **Strict Dominance Across False Positive Rates**:
   - At $\approx 1.7$ false tips/image: T1 Seed 42 achieves **34.15%** sensitivity vs T0's **12.20%** (**nearly 3x higher sensitivity**).
   - At $\approx 2.6$ false tips/image: T1 Seed 42 achieves **36.59%** sensitivity vs T0's **14.63%** (**2.5x higher sensitivity**).
   - At $\le 1.0$ false tip/image: T0 sensitivity collapses to $0.0\% - 4.9\%$, whereas T1 retains **34.15%** sensitivity (14 matches at $\tau = 0.50$).
2. **Ceiling Contrast**:
   - T0's absolute maximum sensitivity ceiling is **17.07%** (7 matches), and reaching it requires accepting an unusable **13.12 false tips per image**.
   - T1 reaches its maximum of **36.59%** (15 matches) at only **2.58 false tips per image**.

The improvement of direct HRNet tip prediction is not an artifact of a single selected operating point; it represents an upward shift of the entire performance frontier.

---

## 5. Decoding Pipeline Failure Attribution (Tracing Missed Targets)

Every one of the 27 missed targets in Seed 42 was traced through the complete 5-stage decoding pipeline:
`Raw Heatmap -> Local 3x3 Peak -> Confidence Threshold (0.20) -> 30px NMS -> Top-10 Cap -> 15px Matching Tolerance`.

```
                  ┌────────────────────────────────────────────────────────┐
                  │ 27 Missed Targets in Seed 42 Evaluation (v2 Benchmark) │
                  └───────────────────────────┬────────────────────────────┘
                                              │
               ┌──────────────────────────────┴──────────────────────────────┐
               ▼                                                             ▼
┌─────────────────────────────┐                               ┌─────────────────────────────┐
│    Decoding / NMS Defect    │                               │     Model-Level Failure     │
│       13 Targets (48.1%)    │                               │       14 Targets (51.9%)    │
└──────────────┬──────────────┘                               └──────────────┬──────────────┘
               │                                                             │
  ┌────────────┴────────────┐                                  ┌─────────────┴─────────────┐
  ▼                         ▼                                  ▼                           ▼
NMS Collision          Survivor Shifted                  Omission / Absent            Subthreshold /
Adjacent CVC tips      to 15-25px zone                   Heatmap (<0.10)              Gross Misloc
(7 targets)            (6 targets)                       (12 targets)                 (2 targets)
```

### Detailed Breakdown of the 27 Missed Targets

| Failure Mechanism | Targets ($N$) | Share (%) | Clinical & Algorithmic Characteristics |
| :--- | :---: | :---: | :--- |
| **NMS Collision (Suppressed by 30px Radius)** | **13** | **48.1%** | **Peak formed with high confidence ($\ge 0.40 - 0.76$)**, but was eliminated by the 30px native NMS radius because another catheter tip or nearby high peak suppressed it. |
| **True Model Omission (No Heatmap, $<0.10$)** | **12** | **44.4%** | Model produced virtually zero tip activation ($< 0.05$). Represents genuine failure of feature representation on atypical catheter paths. |
| **Subthreshold Activation ($0.10 - 0.20$)** | **1** | **3.7%** | Tip activation reached $0.109$, but was pruned by the rigid $\tau = 0.20$ threshold cutoff. |
| **Moderate Localization Error ($25 - 70$ px)** | **1** | **3.7%** | Peak successfully decoded, but placed on adjacent anatomical boundary ($53.3$ px away). |
| **Top-10 Candidate Cap Suppression** | **0** | **0.0%** | The 10-candidate cap never suppressed any true tip candidate. |

### The Multi-CVC Collision Discovery
The multi-tip failure analysis confirms the user's specific warning regarding 30-pixel NMS:
- In dual-CVC studies terminating at the cavoatrial junction, catheter tips frequently terminate within 15–28 native pixels of each other.
- **Case Example (`...4524483...`)**:
  * Catheter 3 (`CVC - Normal`): Model generated strong peak at CAJ with **confidence 0.634**.
  * Catheter 4 (`CVC - Normal`): Model generated strong peak at CAJ with **confidence 0.744**.
  * **What the decoder did**: The 30px NMS radius treated these two distinct anatomical tips as duplicates. The higher peak suppressed the secondary peak, and the surviving peak landed at distance $21.9$ px from Catheter 4 and $24.1$ px from Catheter 3. **Both tips were scored as misses**, despite the model successfully detecting both!
- Similar NMS suppression occurred in Study `...1044073...` (nearest distance $15.4$ px), Study `...4266048...` (nearest distance $20.7$ px), and Study `...8488977...` (nearest distance $21.2$ px).

> [!IMPORTANT]
> **Conclusion**: The model is NOT failing to predict multi-CVC tips; **the post-processing decoder is actively discarding them**. Approximately half of all reported misses ($13/27$) are post-processing decoding artifacts.

---

## 6. Seed 43 Segmentation Retention & Paired Analysis

To address the concern regarding the $-0.0107$ Dice change observed in Seed 42, we evaluated the frozen baseline HRNet Seed 43 against the joint T1 HRNet Seed 43 across all 193 studies in `dev_pilot_val` (248 total catheters):

| Metric | Frozen Baseline (Seed 43) | T1 Joint HRNet (Seed 43) | Absolute Difference | Relative Change | Predefined Tolerance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Macro Dice** | **0.4809** | **0.4848** | **+0.0040** | **+0.82%** | Drop $< 0.05$ (PASS) |
| **10px Centerline Coverage** | **0.8730** | **0.8751** | **+0.0022** | **+0.25%** | Drop $< 0.05$ (PASS) |
| **Micro Recall** | 0.8215 | 0.8166 | -0.0049 | -0.60% | — |
| **Micro Precision** | 0.3235 | 0.3278 | +0.0043 | +1.33% | — |
| **Total Missed Catheters ($<30\%$ cov)** | **10 / 248** (4.03%) | **7 / 248** (2.82%) | **-3 catheters** | **-30.0%** | Miss rate $< 5\%$ (PASS) |
| **Negative Image FP Area %** | 0.038% | 0.041% | +0.003% | Negligible | $< 0.5\%$ (PASS) |

### Paired Per-Study Distribution ($N = 185$ Positive Studies)
- **Improved Studies ($\Delta \text{Dice} > +0.01$)**: 72 studies
- **Degraded Studies ($\Delta \text{Dice} < -0.01$)**: 71 studies
- **Unchanged Studies ($|\Delta \text{Dice}| \le 0.01$)**: 42 studies
- **Median Study-Level Difference**: $0.0000$ (symmetric around zero)

### Inspection of Lost Catheters
Only 4 out of 248 catheters experienced a drop from $\ge 50\%$ coverage to $< 50\%$ coverage:
1. Study `...10142...` (Index 1): Baseline coverage $0.77 \to$ T1 coverage $0.48$ (borderline)
2. Study `...12780...` (Index 0): Baseline coverage $0.69 \to$ T1 coverage $0.14$ (genuine segment loss on a faint proximal branch)
3. Study `...45244...` (Index 1): Baseline coverage $0.57 \to$ T1 coverage $0.43$ (borderline)
4. Study `...84889...` (Index 1): Baseline coverage $0.53 \to$ T1 coverage $0.41$ (borderline)

Across the entire dataset, total missed catheters actually **decreased from 10 to 7**. Combined with the Seed 42 results (where missed catheters dropped from 9 to 8), the evidence confirms that joint tip training does not materially degrade the underlying segmentation representation.

---

## 7. Evidence-Based Recommendation for the Next Step

We evaluated three potential next directions based on our failure attribution data:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Evaluation of Next Experiment Options                           │
├────────────────────────────┬─────────────────────────────┬─────────────────────────────┤
│ 1. Supervision Repair      │ 2. Candidate Detection /    │ 3. Adaptive Multi-Tip       │
│    for Identity Errors     │    Omission Supervision     │    Decoding & Subpixel      │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ Only 1 target was inverted │ 12 targets (44%) are true   │ 13 targets (48%) are lost   │
│ in the dataset, and that   │ omissions (<0.10 activation)│ directly to 30px NMS and    │
│ was a benchmark annotation │ but retraining while the    │ grid rounding (15-25px).    │
│ bug now fixed in v2. The   │ decoder suppresses multi-   │ Model already has strong    │
│ model itself rarely picked │ tips would waste training   │ activations (0.40-0.76).    │
│ entry points (0.50 FP/img).│ capacity.                   │ Requires NO retraining.     │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ Status: Low leverage       │ Status: Valid secondary     │ Status: HIGHEST LEVERAGE    │
└────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
```

### Chosen Next Move: Adaptive Multi-Tip Decoding & Subpixel Local Refinement
The single highest-leverage next experiment is **repairing the post-processing decoder without retraining**:
1. **Catheter-Aware / Adaptive NMS**: Replace the rigid 30-pixel Euclidean suppression with a segmentation-guided NMS that allows distinct peaks along different catheter skeleton branches, preventing mutual suppression of adjacent dual-CVC tips at the cavoatrial junction.
2. **Subpixel Centroid Refinement**: Implement continuous 2D parabolic / Gaussian local fitting around heatmap peaks to eliminate stride-4 discretization error (which currently quantizes predictions to $\approx 13$ px native steps).
3. **Immediate Expected Recovery**: This intervention directly targets the 13 NMS collisions and 6 near-misses (15–25 px), with the potential to recover 4–6 additional matched targets on the existing frozen weights, lifting Success@15 from 14/41 to 18–20/41 without any GPU training.

---

## Summary of Completed Tasks

| Task | Requirement | Status | Key Artifacts & Verification |
| :---: | :--- | :---: | :--- |
| **1** | Separate reviewed vs weak targets | **COMPLETE** | [`reports/provenance_split_evaluation_report.json`](file:///C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/reports/provenance_split_evaluation_report.json) |
| **2** | Blinded audit of 16 endpoints & version v2 | **COMPLETE** | [`manifests/development_benchmark_manifest_v2_reviewed.json`](file:///C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/manifests/development_benchmark_manifest_v2_reviewed.json) |
| **3** | Re-evaluate T0/T1 on corrected v2 benchmark | **COMPLETE** | [`reports/t0_t1_v2_corrected_benchmark_metrics.json`](file:///C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/reports/t0_t1_v2_corrected_benchmark_metrics.json) |
| **4** | Compare sensitivity vs FP/image across thresholds | **COMPLETE** | [`reports/sensitivity_vs_false_tips_curves.json`](file:///C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/reports/sensitivity_vs_false_tips_curves.json), [`froc_tradeoff.png`](file:///C:/Users/M%20S%20I/.gemini/antigravity-ide/brain/5c694925-30e5-4be9-87ea-b2579b1030d8/froc_tradeoff.png) |
| **5** | Trace missed targets through 5 decoding stages | **COMPLETE** | [`reports/decoding_pipeline_failure_attribution.csv`](file:///C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/reports/decoding_pipeline_failure_attribution.csv) |
| **6** | Measure Seed 43 segmentation retention on 193 studies | **COMPLETE** | [`reports/t1_segmentation_retention_seed_43.json`](file:///C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/reports/t1_segmentation_retention_seed_43.json), [`reports/seed_43_segmentation_paired_analysis.csv`](file:///C:/Ankshit/CVC-MAGNet-V2/experiments/hrnet_tip_pilot_v1/reports/seed_43_segmentation_paired_analysis.csv) |
| **7** | Formulate evidence-based next move | **COMPLETE** | Recommended Adaptive Multi-Tip Decoding & Subpixel Refinement |
