# Canonical Native-Resolution CVC Segmentation Benchmark Report

**Evaluation Methodology**:
- **Grid**: All predictions un-letterboxed via bilinear interpolation to exact native radiograph pixel coordinates.
- **Target**: Ground truth polylines rasterized at 10 native pixels on identical native image grids.
- **Cohort**: `dev_pilot_val` (193 studies: 185 positive, 8 verified negative; zero padding contamination).
- **Metrics**: Macro Binary Dice, Macro Soft Dice, Pooled Pixel Precision/Recall, True Native Centerline Coverage at 10px, Native False Area (px and % of radiograph area).

## 1. Primary Fixed-Budget Comparison (1,000 Updates, Final Checkpoints)

| Model | Resolution | $\tau$ | Binary Dice | Soft Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0 Frozen Reference (Final)** | 512×512 | **0.30** | 0.1425 | 0.0301 | 7.60% | 77.36% | 84.4% | 2.0% | 199,476 px | 3.074% |
| **S0 Frozen Reference (Final)** | 512×512 | **0.50** | 0.2037 | 0.0301 | 12.00% | 67.49% | 75.8% | 4.4% | 98,222 px | 1.515% |
| **S0 Frozen Reference (Final)** | 512×512 | **0.60** | 0.2302 | 0.0301 | 14.36% | 62.88% | 71.9% | 5.6% | 71,713 px | 1.105% |
| **S1 Pretrained Baseline (Final)** | 512×512 | **0.30** | 0.1880 | 0.0260 | 10.41% | 77.62% | 81.4% | 3.6% | 126,485 px | 1.904% |
| **S1 Pretrained Baseline (Final)** | 512×512 | **0.50** | 0.2973 | 0.0260 | 19.78% | 66.04% | 71.7% | 9.3% | 42,305 px | 0.649% |
| **S1 Pretrained Baseline (Final)** | 512×512 | **0.60** | 0.3249 | 0.0260 | 23.41% | 60.05% | 67.1% | 10.1% | 30,627 px | 0.469% |
| **S2 Higher-Res 768px (Final)** | 768×768 | **0.30** | 0.1875 | 0.0242 | 10.28% | 79.19% | 83.3% | 1.6% | 124,697 px | 1.846% |
| **S2 Higher-Res 768px (Final)** | 768×768 | **0.50** | 0.3078 | 0.0242 | 19.97% | 68.34% | 74.2% | 6.5% | 37,388 px | 0.565% |
| **S2 Higher-Res 768px (Final)** | 768×768 | **0.60** | 0.3626 | 0.0242 | 27.74% | 59.75% | 68.0% | 8.9% | 21,181 px | 0.323% |

## 2. Development Best-Checkpoint Comparison (Selected by Validation Soft Dice)

| Model | Resolution | $\tau$ | Binary Dice | Soft Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1 Pretrained Baseline (Best)** | 512×512 | **0.30** | 0.1880 | 0.0260 | 10.41% | 77.62% | 81.4% | 3.6% | 126,485 px | 1.904% |
| **S1 Pretrained Baseline (Best)** | 512×512 | **0.50** | 0.2973 | 0.0260 | 19.78% | 66.04% | 71.7% | 9.3% | 42,305 px | 0.649% |
| **S1 Pretrained Baseline (Best)** | 512×512 | **0.60** | 0.3249 | 0.0260 | 23.41% | 60.05% | 67.1% | 10.1% | 30,627 px | 0.469% |
| **S2 Higher-Res 768px (Best)** | 768×768 | **0.30** | 0.1875 | 0.0242 | 10.28% | 79.19% | 83.3% | 1.6% | 124,697 px | 1.846% |
| **S2 Higher-Res 768px (Best)** | 768×768 | **0.50** | 0.3078 | 0.0242 | 19.97% | 68.34% | 74.2% | 6.5% | 37,388 px | 0.565% |
| **S2 Higher-Res 768px (Best)** | 768×768 | **0.60** | 0.3626 | 0.0242 | 27.74% | 59.75% | 68.0% | 8.9% | 21,181 px | 0.323% |

## 3. Matched-Recall Operating Point Analysis

To prevent selective threshold bias, comparing models at matched recall regimes:

### High-Recall Regime (Recall ~ 77–80%):
- **S0 Frozen Reference (Final)** ($\tau=0.30$): Recall = **77.36%**, Precision = **7.60%**, Binary Dice = **0.1425**, Centerline Cov @ 10px = **84.4%**, Neg False Area = **3.074%** (199,476 px)
- **S1 Pretrained Baseline (Final)** ($\tau=0.30$): Recall = **77.62%**, Precision = **10.41%**, Binary Dice = **0.1880**, Centerline Cov @ 10px = **81.4%**, Neg False Area = **1.904%** (126,485 px)
- **S2 Higher-Res 768px (Final)** ($\tau=0.30$): Recall = **79.19%**, Precision = **10.28%**, Binary Dice = **0.1875**, Centerline Cov @ 10px = **83.3%**, Neg False Area = **1.846%** (124,697 px)

### Balanced-Recall Regime (Recall ~ 65–68%):
- **S0 Frozen Reference (Final)** ($\tau=0.50$): Recall = **67.49%**, Precision = **12.00%**, Binary Dice = **0.2037**, Centerline Cov @ 10px = **75.8%**, Neg False Area = **1.515%** (98,222 px)
- **S1 Pretrained Baseline (Final)** ($\tau=0.50$): Recall = **66.04%**, Precision = **19.78%**, Binary Dice = **0.2973**, Centerline Cov @ 10px = **71.7%**, Neg False Area = **0.649%** (42,305 px)
- **S2 Higher-Res 768px (Final)** ($\tau=0.50$): Recall = **68.34%**, Precision = **19.97%**, Binary Dice = **0.3078**, Centerline Cov @ 10px = **74.2%**, Neg False Area = **0.565%** (37,388 px)

## 4. Paired Patient-Cluster Bootstrap Confidence Intervals (95% CI, $\tau=0.50$)

- **S1 Final vs S0 Frozen** (N=132 patients, 185 positive studies):
  - $\Delta$ Macro Binary Dice: **[+0.0817, +0.1051]**
  - $\Delta$ Pooled Precision: **[+6.95%, +8.56%]**
  - $\Delta$ Pooled Recall: **[-3.71%, +0.84%]**

- **S2 Final vs S1 Final** (N=132 patients, 185 positive studies):
  - $\Delta$ Macro Binary Dice: **[+0.0016, +0.0183]**
  - $\Delta$ Pooled Precision: **[-0.93%, +1.11%]**
  - $\Delta$ Pooled Recall: **[+0.39%, +3.96%]**

## 5. Audit Panel Integrity
- Frozen Audit Panel established at `reports/frozen_50_study_audit_panel_ids.json` (50 studies total: 10 multi-CVC, 10 abnormal, 10 cluttered, 10 single-CVC, 8 negatives, 2 other).
- Separate per-model measurements saved: `audit_panel_measurements_S0_frozen_final.csv`, `audit_panel_measurements_S1_pretrained_final.csv`, `audit_panel_measurements_S2_higher_res_final.csv`.
