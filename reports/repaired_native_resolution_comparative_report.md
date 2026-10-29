# Repaired Native-Resolution CVC Segmentation Benchmark Report

**Evaluation Methodology**:
- **Rigorous Experimental Parity**: Exactly 1,000 updates, identical initial weights (`runs/initial_weights.pt`), identical effective batch size (4), identical exposure sequence and per-exposure augmentation parameters.
- **Native Resolution Grid**: All predictions unletterboxed to original radiograph pixels and evaluated against 10px ground truth rasterized masks.
- **Evaluation Cohort**: `dev_pilot_val` (193 studies: 185 positive, 8 verified negative; 132 unique patients; zero padding contamination).
- **Metrics**: Macro Binary Dice, Native Soft Dice, Pooled Pixel Precision/Recall, Centerline Coverage @ 10px, Missed-path rate, False foreground on negative radiographs.

## 1. Primary Repaired Comparison (1,000 Updates, Final Checkpoints)

| Model | Resolution | $\tau$ | Binary Dice | Soft Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0 Frozen Reference** | 512×512 | **0.30** | 0.1425 | 0.0301 | 7.60% | 77.36% | 84.4% | 2.0% | 199,476 px | 3.074% |
| **S0 Frozen Reference** | 512×512 | **0.40** | 0.1756 | 0.0301 | 9.85% | 72.03% | 79.7% | 4.0% | 135,103 px | 2.085% |
| **S0 Frozen Reference** | 512×512 | **0.50** | 0.2037 | 0.0301 | 12.00% | 67.49% | 75.8% | 4.4% | 98,222 px | 1.515% |
| **S0 Frozen Reference** | 512×512 | **0.60** | 0.2302 | 0.0301 | 14.36% | 62.88% | 71.9% | 5.6% | 71,712 px | 1.105% |
| **S0 Frozen Reference** | 512×512 | **0.70** | 0.2585 | 0.0301 | 17.40% | 57.28% | 67.4% | 8.9% | 49,062 px | 0.755% |
| **S1 Repaired 512px (Final)** | 512×512 | **0.30** | 0.1682 | 0.0280 | 8.49% | 76.81% | 81.1% | 4.4% | 151,084 px | 2.281% |
| **S1 Repaired 512px (Final)** | 512×512 | **0.40** | 0.2328 | 0.0280 | 12.66% | 70.55% | 75.6% | 7.3% | 79,161 px | 1.175% |
| **S1 Repaired 512px (Final)** | 512×512 | **0.50** | 0.2736 | 0.0280 | 16.90% | 65.49% | 71.6% | 8.5% | 42,044 px | 0.621% |
| **S1 Repaired 512px (Final)** | 512×512 | **0.60** | 0.3077 | 0.0280 | 21.32% | 59.84% | 67.2% | 10.9% | 29,644 px | 0.439% |
| **S1 Repaired 512px (Final)** | 512×512 | **0.70** | 0.3340 | 0.0280 | 25.96% | 52.27% | 61.1% | 14.9% | 19,466 px | 0.289% |
| **S2 Repaired 768px (Final)** | 768×768 | **0.30** | 0.2010 | 0.0299 | 10.33% | 78.31% | 82.9% | 3.6% | 76,112 px | 1.187% |
| **S2 Repaired 768px (Final)** | 768×768 | **0.40** | 0.2737 | 0.0299 | 15.22% | 73.18% | 78.4% | 4.4% | 33,132 px | 0.515% |
| **S2 Repaired 768px (Final)** | 768×768 | **0.50** | 0.3185 | 0.0299 | 19.74% | 69.52% | 75.7% | 5.6% | 22,920 px | 0.357% |
| **S2 Repaired 768px (Final)** | 768×768 | **0.60** | 0.3532 | 0.0299 | 24.30% | 65.45% | 72.8% | 6.5% | 17,369 px | 0.270% |
| **S2 Repaired 768px (Final)** | 768×768 | **0.70** | 0.3870 | 0.0299 | 29.82% | 59.80% | 69.2% | 9.3% | 12,278 px | 0.191% |

## 2. Repaired Best-Checkpoint Comparison (Selected by Val Soft Dice)

| Model | Resolution | $\tau$ | Binary Dice | Soft Dice | Precision | Recall | Cov @ 10px | Missed @ 10px | Neg False Px | Neg False % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1 Repaired 512px (Best)** | 512×512 | **0.30** | 0.1682 | 0.0280 | 8.49% | 76.81% | 81.1% | 4.4% | 151,084 px | 2.281% |
| **S1 Repaired 512px (Best)** | 512×512 | **0.40** | 0.2328 | 0.0280 | 12.66% | 70.55% | 75.6% | 7.3% | 79,161 px | 1.175% |
| **S1 Repaired 512px (Best)** | 512×512 | **0.50** | 0.2736 | 0.0280 | 16.90% | 65.49% | 71.6% | 8.5% | 42,044 px | 0.621% |
| **S1 Repaired 512px (Best)** | 512×512 | **0.60** | 0.3077 | 0.0280 | 21.32% | 59.84% | 67.2% | 10.9% | 29,644 px | 0.439% |
| **S1 Repaired 512px (Best)** | 512×512 | **0.70** | 0.3340 | 0.0280 | 25.96% | 52.27% | 61.1% | 14.9% | 19,466 px | 0.289% |
| **S2 Repaired 768px (Best)** | 768×768 | **0.30** | 0.1873 | 0.0301 | 9.46% | 79.93% | 84.3% | 1.2% | 100,594 px | 1.564% |
| **S2 Repaired 768px (Best)** | 768×768 | **0.40** | 0.2544 | 0.0301 | 13.79% | 74.91% | 80.1% | 4.4% | 50,657 px | 0.786% |
| **S2 Repaired 768px (Best)** | 768×768 | **0.50** | 0.2981 | 0.0301 | 17.80% | 71.25% | 77.5% | 5.2% | 36,967 px | 0.575% |
| **S2 Repaired 768px (Best)** | 768×768 | **0.60** | 0.3331 | 0.0301 | 21.90% | 67.33% | 75.0% | 5.6% | 28,410 px | 0.442% |
| **S2 Repaired 768px (Best)** | 768×768 | **0.70** | 0.3681 | 0.0301 | 26.81% | 62.08% | 71.6% | 7.7% | 20,598 px | 0.320% |

## 3. Matched-Recall Operating Point Analysis

Comparing models at matched recall regimes to eliminate operating point bias:

### High-Recall Operating Regime (Recall ~ 76–80%):
- **S0 Frozen Reference** ($\tau=0.30$): Recall = **77.36%**, Precision = **7.60%**, Binary Dice = **0.1425**, Centerline Cov @ 10px = **84.4%**, Neg False Area = **3.074%** (199,476 px)
- **S1 Repaired Final** ($\tau=0.30$): Recall = **76.81%**, Precision = **8.49%**, Binary Dice = **0.1682**, Centerline Cov @ 10px = **81.1%**, Neg False Area = **2.281%** (151,084 px)
- **S2 Repaired Final** ($\tau=0.30$): Recall = **78.31%**, Precision = **10.33%**, Binary Dice = **0.2010**, Centerline Cov @ 10px = **82.9%**, Neg False Area = **1.187%** (76,112 px)

### Balanced Operating Regime (Recall ~ 65–68%):
- **S0 Frozen Reference** ($\tau=0.50$): Recall = **67.49%**, Precision = **12.00%**, Binary Dice = **0.2037**, Centerline Cov @ 10px = **75.8%**, Neg False Area = **1.515%** (98,222 px)
- **S1 Repaired Final** ($\tau=0.50$): Recall = **65.49%**, Precision = **16.90%**, Binary Dice = **0.2736**, Centerline Cov @ 10px = **71.6%**, Neg False Area = **0.621%** (42,044 px)
- **S2 Repaired Final** ($\tau=0.50$): Recall = **69.52%**, Precision = **19.74%**, Binary Dice = **0.3185**, Centerline Cov @ 10px = **75.7%**, Neg False Area = **0.357%** (22,920 px)

## 4. Paired Patient-Cluster Bootstrap Confidence Intervals (95% CI, $\tau=0.50$)

- **S1 Repaired Final vs S0 Frozen Reference** (N=132 patients, 185 positive studies):
  - $\Delta$ Macro Binary Dice: **[+0.0568, +0.0825]**
  - $\Delta$ Pooled Precision: **[+3.62%, +6.11%]**
  - $\Delta$ Pooled Recall: **[-4.03%, +0.08%]**

- **S2 Repaired Final vs S1 Repaired Final** (N=132 patients, 185 positive studies):
  - $\Delta$ Macro Binary Dice: **[+0.0359, +0.0545]**
  - $\Delta$ Pooled Precision: **[+1.88%, +3.82%]**
  - $\Delta$ Pooled Recall: **[+2.15%, +5.93%]**

- **S2 Repaired Best vs S1 Repaired Best** (N=132 patients, 185 positive studies):
  - $\Delta$ Macro Binary Dice: **[+0.0156, +0.0336]**
  - $\Delta$ Pooled Precision: **[+0.14%, +1.71%]**
  - $\Delta$ Pooled Recall: **[+3.86%, +7.67%]**

## 5. Audit Panel & Reproducibility Provenance
- Full per-study measurements saved for each model in `reports/native_eval_<tag>_per_study.json` and `reports/native_eval_<tag>_all_studies_tau50.csv`.
- Frozen Audit Panel evaluated and persisted: `audit_panel_measurements_S1_repaired_final.csv`, `audit_panel_measurements_S2_repaired_final.csv`.
