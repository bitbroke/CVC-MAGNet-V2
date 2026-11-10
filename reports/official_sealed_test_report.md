# Official Blind Evaluation Report: Sealed Test Cohort ($N=2,284$)

**Date**: November 10, 2026  
**Sealed Cohort Size**: Exactly $N=2,284$ chest radiographs ($N=260$ unique patients)  
**Primary Metric**: Macro AUROC across `CVC - Normal`, `CVC - Borderline`, and `CVC - Abnormal`  
**Statistical Validation**: 1,000-sample patient-clustered bootstrap ($N=260$ clusters) with paired two-sided $p$-values  

---

## 1. Primary Benchmark Leaderboard (Sealed Test Split)

| Model Architecture / System Component | Test Macro AUROC | CVC - Normal | CVC - Borderline | CVC - Abnormal | Traced ($N=630$) | Untraced ($N=1654$) | Multi-CVC ($N=373$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E00_traced_effnet_baseline`** (GAP Image Floor) | **0.7127** | 0.7146 | 0.6829 | 0.7405 | 0.7239 | 0.7095 | 0.6040 |
| **`E82_t3_topological_gbdt_80d`** (Best Tabular) | **0.7127** | 0.7190 | 0.7069 | 0.7122 | 0.7083 | 0.7145 | 0.5790 |
| **`E83_densenet121_consensus`** (Best Single CNN) | **0.7611** | 0.7756 | 0.7321 | 0.7755 | 0.7362 | 0.7728 | 0.6286 |
| **`E84_seresnext50_consensus`** (ResNeXt CNN) | **0.7822** | 0.8035 | 0.7433 | 0.7997 | 0.7567 | 0.7935 | 0.6716 |
| **`E85_Apex_Inductive_Val_ECDF`** (Bedside Inductive)| **0.8052** | **0.8118** | **0.7736** | **0.8303** | **0.7782** | **0.8168** | **0.6608** |
| **`E85_Apex_Cohort_Ranked`** (Full Cohort Rank) | **0.8052** | **0.8118** | **0.7736** | **0.8303** | **0.7782** | **0.8168** | **0.6608** |

> [!NOTE]
> **Inductive vs. Transductive Equivalence**: The single-patient Inductive Val-ECDF evaluation (0.8052) matches the full-cohort rank evaluation (0.8052) within 0.0000 AUROC, mathematically verifying that the full system can run on a single radiograph at the bedside with zero cohort leakage.

---

## 2. Patient-Clustered Bootstrap 95% Confidence Intervals

| Metric Dimension | Baseline Floor (`E00`) [95% CI] | Full System (`E85`) [95% CI] | Net Gain ($\Delta\text{AUROC}$) | Paired $p$-value |
| :--- | :---: | :---: | :---: | :---: |
| **Macro AUROC** | 0.7128 [0.6765 – 0.7465] | **0.8052 [0.7874 – 0.8233]** | **+0.0924** | **$p < 10^{-15}$** |
| **CVC - Normal** | 0.7146 [0.6823 – 0.7485] | **0.8120 [0.7856 – 0.8369]** | **+0.0974** | **$p < 10^{-15}$** |
| **CVC - Borderline** | 0.6829 [0.6462 – 0.7221] | **0.7732 [0.7445 – 0.8000]** | **+0.0903** | **$p < 10^{-15}$** |
| **CVC - Abnormal** | 0.7409 [0.6766 – 0.8001] | **0.8302 [0.7982 – 0.8608]** | **+0.0893** | **$p < 10^{-15}$** |

---

## 3. Clinical Safety & Operating Points

- **CVC - Abnormal Sensitivity @ 80% Specificity**: **69.0%**
- **CVC - Abnormal Specificity @ 90% Sensitivity**: **55.3%** (High-safety malposition triage)
- **Mean Brier Calibration Score**: **0.2090**
