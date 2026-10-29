# Canonical Reference Freeze & Hard-Negative False-Track Suppression Protocol

**Date**: 2026-10-16  
**Status**: Frozen Reference Established; Two-Arm Sampling Pilot Scheduled  
**Objective**: Suppress false-positive CVC tracks without degrading real catheter recall or centerline coverage.

---

## 1. Frozen Reference Baseline

The repaired 768px segmentation model (`CVC-Segmentation-V3`) is formally frozen as the reference baseline:

### Component Freeze Record:
- **Architecture**: `CVCSegmentationV3Net` (timm EfficientNet-B0 backbone + FPN decoder, 768×768 resolution, frozen encoder BatchNorm, single binary segmentation head).
- **Frozen Reference Checkpoint**:
  - File: `runs/s2_repaired_768/checkpoints/final_segmentation_v3.pt`
  - SHA256: `6d126d3cdd9390fd1c9f4d76f3f019f5dfb8f362a22b7bf591bfb4bf5a92cf99`
- **Certified Initial Weights**:
  - File: `runs/initial_weights.pt`
  - SHA256: `bb605c6fee0e7b279cdaaa8d8c8c0b954f64915ecf74f1fa5bc9707af9d1e17f`
- **Evaluation Dataset**: `dev_pilot_val` (193 studies: 185 positive, 8 confirmed negative; 132 unique patients; zero padding contamination).
- **Ground Truth Target**: Polylines rasterized at 10 native pixels on identical native image grids.
- **Decision Threshold**: $\tau = 0.50$.

### Reference Operating Point at $\tau=0.50$:
- **Macro Binary Dice**: **0.326** (32.6%)
- **Pooled Pixel Recall**: **69.5%**
- **Negative False-Mask Area**: **0.390%** of total radiograph area

---

## 2. Training-Set Hard-Negative Mining Protocol

### Source Cohort:
- **Partition**: `dev_pilot_train` confirmed-negative studies (`has_cvc == False`).
- **Total Images**: **396 confirmed negatives**.
- **Data Leakage Prohibition**: Zero validation studies are included in mining; only training-set negatives are evaluated.

### Scoring & Ranking:
The frozen reference model scores each training negative:
$$\text{False Pixels}_{\tau=0.50} = \sum_{p \in \text{valid}} \mathbb{I}[P(y_p=1 \mid x) \ge 0.50]$$
$$\text{False Area \%} = \frac{\text{False Pixels}_{\tau=0.50}}{\text{Total Valid Pixels}} \times 100\%$$
- **Primary Sort Key**: `false_pixels_tau50` descending.
- **Secondary Sort Key**: `max_prob` descending.

### Hardness Tiers:
- **`severe_hard`**: $\text{False Pixels}_{\tau=0.50} \ge 1,000$ px.
- **`moderate_hard`**: $200 \le \text{False Pixels}_{\tau=0.50} < 1,000$ px.
- **`mild_hard`**: $\text{False Pixels}_{\tau=0.50} > 0$ or $\max(P) \ge 0.30$.
- **`clean_negative`**: Zero false pixels at $\tau=0.50$ and low peak background probability.

---

## 3. Human Audit of Highest-Scoring Negatives

Before active supervision:
1. Inspect the top 15 worst false-positive studies using generated visual overlays (`reports/overlays/hard_negatives/`).
2. Verify label credibility: confirm whether false activations correspond to non-CVC anatomical/external structures (e.g., clavicle border, ribs, ECG wires, endotracheal tubes, feeding tubes, skin folds) rather than an unlabelled true CVC.
3. If any radiograph is ambiguous or suspected to contain an unlabelled catheter, flag `excluded_from_hard_negatives = True` in `data/hard_negatives_manifest.json` so it is safely omitted from training.

---

## 4. Two-Arm Matched Sampling Experiment

Both arms train sequentially on the local GPU under exact parity:

| Setting | Arm A: Control (Uniform Negatives) | Arm B: Experiment (Hard Negative Mining) |
| :--- | :---: | :---: |
| **Model Architecture** | Existing 768px `CVCSegmentationV3Net` | Identical |
| **Initialization** | Frozen reference checkpoint (`s2_repaired_768`) | Same frozen reference checkpoint |
| **Positive Sampling** | 75% positive studies (unchanged) | 75% positive studies (unchanged) |
| **Negative Sampling Pool** | Uniform across all 396 training negatives | **50% uniform, 50% hard negatives** |
| **Total Pos / Neg Ratio** | Strictly 3 : 1 (unchanged) | Strictly 3 : 1 (unchanged) |
| **Training Budget** | Exactly 1,000 successful updates | Exactly 1,000 successful updates |
| **Optimizer & Scheduler** | AdamW ($\text{lr}=10^{-4}$ cosine decay to $10^{-6}$) | Identical |
| **Loss & Denominators** | Effective-batch BCE ($w=10$) + Soft Dice ($w=1$) | Identical |
| **Augmentation** | Dedicated per-exposure RNG | Identical |

---

## 5. Multi-Metric Decision Gate (At $\tau=0.50$)

Success requires simultaneous satisfaction of the following criteria:
1. **False Track Reduction**: Statistically significant drop in negative false-mask area ($< 0.390\%$).
2. **Precision Improvement**: Pooled pixel precision improves over reference.
3. **Catheter Preservation**: Real catheter recall preserved ($\ge 68.0\%$) and centerline coverage @ 10px maintained.
4. **Missed Catheters**: Missed-path rate does not increase meaningfully ($< 8\%$).
5. **Visual Verification**: Visibly fewer false branches on the fixed 50-study inspection panel. Empty or universally suppressed masks do NOT count as progress.

---

## 6. Placement Classification Isolation

CVC placement classification (`CVC-Placement-V1`) remains strictly isolated:
- No segmentation masks from this stage will be fed into the placement classifier.
- Classification benchmarks continue independently on native radiographs.
