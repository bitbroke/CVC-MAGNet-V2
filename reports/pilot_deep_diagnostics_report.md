# Deep Diagnostic Report: Heatmap Tracing, Uncapped FROC, and Spatial Failure Analysis

**Date**: 2026-10-15  
**Cohort**: `dev_pilot_val` (193 studies: 185 positive, 8 confirmed negative)  
**Evaluated Artifacts**:
- Arm A (Control): [final_coarse_dual_head.pt](file:///C:/Ankshit/CVC-MAGNet-V2/runs/pilot_control_reviewed_only/checkpoints/final_coarse_dual_head.pt) (`64a9fb3da89ec81c96a4ce9e487f55b6823c34fa5f61d564ce8354c0e64c3c3a`)
- Arm B (Experiment): [final_coarse_dual_head.pt](file:///C:/Ankshit/CVC-MAGNet-V2/runs/pilot_experiment_expanded/checkpoints/final_coarse_dual_head.pt) (`ca4c59a35e69aa4b7cb6bbd4ea9b9776f8272fdfbbdf51e70e9a5933d311d9f8`)
- Provenance Registry: [reports/frozen_pilot_checkpoints_provenance.json](file:///C:/Ankshit/CVC-MAGNet-V2/reports/frozen_pilot_checkpoints_provenance.json)
- Full Raw Diagnostics Data: [reports/deep_heatmap_diagnostics.json](file:///C:/Ankshit/CVC-MAGNet-V2/reports/deep_heatmap_diagnostics.json)

---

## 1. Frozen Checkpoint & Data Registry

All artifacts, manifest versions, and checkpoints have been computed and frozen:

| File | Path | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| **Arm A Final Checkpoint** | `runs/pilot_control_reviewed_only/checkpoints/final_coarse_dual_head.pt` | `64a9fb3da89ec81c96a4ce9e487f55b6823c34fa5f61d564ce8354c0e64c3c3a` | Frozen |
| **Arm A Best Checkpoint** | `runs/pilot_control_reviewed_only/checkpoints/best_coarse_dual_head.pt` | `102146e278693c0429a1491cf1f5b11910aa1a5568177dfbceca84c6e9e4f1a2` | Frozen |
| **Arm B Final Checkpoint** | `runs/pilot_experiment_expanded/checkpoints/final_coarse_dual_head.pt` | `ca4c59a35e69aa4b7cb6bbd4ea9b9776f8272fdfbbdf51e70e9a5933d311d9f8` | Frozen |
| **Arm B Best Checkpoint** | `runs/pilot_experiment_expanded/checkpoints/best_coarse_dual_head.pt` | `90236a999cb502c46fa01cba6393b5ae248a3d463b2f518e9882200dc89025c8` | Frozen |
| **Training Expansion Manifest** | `data/training_expansion_manifest.parquet` | `a5fce443658514194c5e31da29b47e24a87ad18659178128ea330a1ee0869da7` | Frozen |
| **Validation Targets Registry** | `data/splits/dev_pilot_val_targets.csv` | `7ad60c1d68a2bf60835f8d0a89d71a81dc4113e6179b50db1f47dc79cf5aa730` | Frozen |
| **Reviewed Distal Tips V2** | `data/reviewed_distal_tips_v2.parquet` | `a3a8309fa926a8d7950c920f6970894080183b1c676d1a9dc91c0e815e985b83` | Frozen |

---

## 2. Heatmap Tracing: Why Are Ground-Truth Peaks Not Matching?

To determine whether correct peaks are absent, displaced, suppressed by NMS, or cut by the top-10 cap, we traced every single ground-truth target tip through the continuous 2D heatmap output.

```mermaid
pie title Arm B Distal Tip Fate on Full 248 Registry
    "Matched at 15px": 18
    "Suppressed by NMS (min_distance=10)": 121
    "Removed by Top-10 Cap": 5
    "Displaced Nearby (15-50px)": 15
    "Displaced Distant (>50px)": 89
    "Displaced to Entry Point": 0
```

### Exact Fate Breakdown on Assessable Validation Targets ($N=248$)

| Fate Category | Arm A (Control) | Arm B (Experiment) | Single-CVC (Arm B, $N=126$) | Multi-CVC (Arm B, $N=122$) | Diagnostic Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Matched within 15px** | 1 (0.4%) | **18 (7.3%)** | 12 (9.5%) | 6 (4.9%) | Clean, successful localizations |
| **Suppressed by NMS** | 134 (54.0%) | **121 (48.8%)** | 61 (48.4%) | 60 (49.2%) | **PRIMARY BOTTLENECK**: Local peak $\ge 0.30$ within 15px was wiped out by a higher pixel within `min_distance=10` |
| **Removed by Top-10 Cap** | 22 (8.9%) | **5 (2.0%)** | 3 (2.4%) | 2 (1.6%) | Survived NMS but outside top-10 confidence |
| **Displaced to Entry Point** | 0 (0.0%) | **0 (0.0%)** | 0 (0.0%) | 0 (0.0%) | **ZERO orientation confusion**: Model never predicts the proximal entry site |
| **Displaced Nearby (15–50px)** | 3 (1.2%) | **15 (6.0%)** | 5 (4.0%) | 10 (8.2%) | Peak exists along catheter path, but off by 16–50 native pixels |
| **Displaced Distant (>50px)** | 88 (35.5%) | **89 (35.9%)** | 45 (35.7%) | 44 (36.1%) | Highest peak located in mediastinum or on adjacent anatomy |
| **No Peaks in Image** | 0 (0.0%) | **0 (0.0%)** | 0 (0.0%) | 0 (0.0%) | Model always produces candidate signal |

### Heatmap Value Intensity at the Ground-Truth Tip Pixel

| Metric | Arm A (Control) | Arm B (Experiment) | Direct Finding |
| :--- | :--- | :--- | :--- |
| **Mean Heatmap Value at True Tip** | 0.2952 | **0.4021** | +36% higher activation |
| **Median Heatmap Value at True Tip** | 0.3131 | **0.4329** | +38% higher activation |
| **Mean Max Value within 15px** | 0.3235 | **0.4414** | Strong localized activation |
| **Median Max Value within 15px** | 0.3401 | **0.4797** | Median peak near 0.50 |
| **Targets with Max Value $\ge 0.30$ within 15px** | 197 / 248 (79.4%) | **211 / 248 (85.1%)** | **85.1% of all tips have true signal** |

> [!IMPORTANT]
> **Key Insight**: Peaks are **NOT absent**. In Arm B, **85.1% of all ground-truth tips** have heatmap activation $\ge 0.30$ within 15 native pixels. The failure to match is almost entirely driven by **NMS suppression (48.8%)**: `min_distance = 10` on a 512x512 grid represents a $\approx 35-45$ native pixel exclusion radius, causing broad or adjacent peaks in the mediastinum to extinguish the true tip.

---

## 3. Uncapped False Peaks & Free-Response ROC (FROC) Analysis

When the arbitrary `max_peaks=10` cap is completely removed (`max_peaks = None`), we measure uncapped false peaks on the 8 confirmed negative studies across threshold sweeps $\tau \in [0.05, 0.80]$:

| Threshold ($\tau$) | Arm B Tip Head FP / Negative Study | Arm B Tip Head Success@15 (248) | Arm B Seg Skeleton FP / Negative Study | Arm B Seg Skeleton Success@15 (248) |
| :--- | :--- | :--- | :--- | :--- |
| **$\tau = 0.50$** | **0.0 FP / study** | **18 / 248 (7.26%)** | 0.4 FP / study ($\tau_s=0.50$) | 34 / 248 (13.71%) |
| **$\tau = 0.40$** | **0.2 FP / study** | **20 / 248 (8.06%)** | 0.5 FP / study ($\tau_s=0.40$) | 27 / 248 (10.89%) |
| **$\tau = 0.35$** | **0.4 FP / study** | **21 / 248 (8.47%)** | 0.7 FP / study ($\tau_s=0.30$) | 24 / 248 (9.68%) |
| **$\tau = 0.30$** | **1.5 FP / study** | **23 / 248 (9.27%)** | 2.2 FP / study ($\tau_s=0.20$) | 14 / 248 (5.65%) |
| **$\tau = 0.25$** | **3.3 FP / study** | **24 / 248 (9.68%)** | - | - |
| **$\tau = 0.20$** | **8.7 FP / study** | **24 / 248 (9.68%)** | 2.2 FP / study ($\tau_s=0.20$) | 14 / 248 (5.65%) |
| **$\tau \le 0.15$** | **8.7 FP / study** (plateau) | **24 / 248 (9.68%)** | - | - |

### Matched False-Positive Comparison: Tip Head vs. Seg Skeleton

1. **At Low False-Positive Tolerance ($\le 0.4$ FP / study)**:
   - **Tip Heatmap Head ($\tau = 0.35$)**: Achieves **8.47% Success@15** with only **0.4 false peaks per negative study**.
   - **Seg Skeleton Endpoints ($\tau_s = 0.50$)**: Achieves **13.71% Success@15** with **0.4 false candidates per negative study**.
2. **At Zero False-Positive Tolerance ($0.0$ FP / study)**:
   - **Tip Heatmap Head ($\tau = 0.50$)**: Retains **18 / 248 (7.26%)** with **strictly 0.0 false positives across all negative studies**.
   - **Seg Skeleton Endpoints**: Drops to **0.00%** (cannot operate at zero FP without wiping out all predictions).

---

## 4. Inspection of the 25 Audited Cases Specifically

Tracing the 25 provisional engineering-audited cases in Arm B reveals:
- **21 / 25 cases (84.0%)** have $\max_{B_{15}} H \ge 0.30$ within 15px.
- **15 / 25 cases (60.0%)** have $\max_{B_{15}} H \ge 0.40$.
- **9 / 25 cases (36.0%)** have $\max_{B_{15}} H \ge 0.50$.
- **Why did only 1 match at 15px?**
  - **13 / 25 (52.0%) were directly suppressed by NMS**. For example, in Case 32, the model predicted an intense heatmap value of **0.542** directly on the true tip, but NMS suppressed the peak in favor of an adjacent higher mediastinal pixel.
  - **3 / 25 (12.0%) were displaced by 15–50px**. In Case 27, the nearest peak was at **23.55 native pixels** (confidence 0.52)—missing the 15px cutoff by just 8 pixels.
  - **9 / 25 (36.0%) were displaced >50px**.

---

## 5. Evidence-Based Next Step: Background Suppression vs. Spatial Decoding Resolution

The user posed the choice:
> *Choose one targeted experiment based on those findings: background suppression if false peaks dominate, or spatial decoding/resolution if peaks consistently miss the tip.*

### The Empirical Evidence:
1. **False peaks do NOT dominate**: At $\tau = 0.35 - 0.50$, the tip head averages only **0.0 to 0.4 false peaks per negative study**. Background suppression on negatives is already remarkably effective.
2. **True signal IS present at the tip**: **85.1% of true tips** already exhibit heatmap activation $\ge 0.30$ within 15px (median max value: **0.48**).
3. **The true bottleneck is spatial decoding resolution and NMS suppression**:
   - 48.8% of all true peaks are suppressed by `min_distance=10` NMS on the coarse 512x512 grid.
   - Another 6.0% miss by just 15–50 native pixels because downsampling a 3000x3000 image to 512x512 makes 1 model pixel equal to $\approx 4-6$ native pixels, so a 2-pixel quantization error pushes the prediction past the 15px Hungarian matching threshold.

### Targeted Recommendation:
**Spatial Decoding & Resolution Refinement (without retraining)**:
1. **Sub-Pixel Local Center-of-Mass / Soft-Argmax**: Instead of picking integer pixel coordinates from `np.where(local_maxima)`, calculate the continuous centroid $\mathbf{x}_{\text{sub}} = \frac{\sum_{(x,y) \in \mathcal{N}} (x,y) \cdot H(x,y)^\gamma}{\sum H(x,y)^\gamma}$ around high-confidence candidate patches.
2. **Adaptive NMS Radius**: Reduce `min_distance` from 10 model pixels ($\approx 40$ native px) to 3–4 model pixels ($\approx 12-16$ native px) or use catheter-constrained endpoint masking, preventing adjacent mediastinal activations from wiping out valid distal tip peaks.
3. **Stage 2 ROI Crop**: The coarse head already identifies the catheter tip neighborhood in >85% of cases; a high-resolution $256 \times 256$ native crop around the coarse candidate is the architecturally intended mechanism to resolve sub-15px distal tip accuracy.
