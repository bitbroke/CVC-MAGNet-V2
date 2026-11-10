# Coordinate Decoder Comparison: D0 (Cell Center) vs D1 (DARK CVPR 2020)

**Date**: 2026-10-17  
**Model Checkpoint**: `runs/t2_repaired_finetune_400/checkpoints/best.pt`  
**Threshold**: 0.2 | **NMS Suppression Radius**: 3.0 px  
**Diagnostics**:
- Total raw peaks above threshold: 5190
- Post-NMS peaks retained: 530 (mean 2.75 / image)

---

## Benchmark Comparison (Hungarian Matching on Ground Truth Tips)

| Metric                    | D0 (Cell Center)   | D1 (DARK Taylor)   | Delta (D1 - D0)   |
|:--------------------------|:-------------------|:-------------------|:------------------|
| Success@5                 | 5.97%              | 9.20%              | +3.23%            |
| Success@10                | 19.65%             | 21.14%             | +1.49%            |
| Success@15                | 33.08%             | 33.08%             | +0.00%            |
| Success@20                | 40.30%             | 42.04%             | +1.74%            |
| Success@25                | 46.27%             | 46.27%             | +0.00%            |
| Success@35                | 53.48%             | 53.48%             | +0.00%            |
| Success@50                | 60.70%             | 61.19%             | +0.50%            |
| Success@65                | 62.94%             | 63.18%             | +0.25%            |
| Matched Error Median (px) | 8.64 px            | 8.42 px            | -0.22 px          |
| Matched Error P95 (px)    | 14.21 px           | 14.20 px           | -0.01 px          |

---

## Conclusion
- **D0 (Cell Center)**: Direct scaling of integer grid coordinates $(x + 0.5) 	imes 4$.
- **D1 (DARK Subpixel Refinement)**: Taylor expansion around local Gaussian peak surface using gradient and Hessian.
- Benefit of subpixel refinement is strictly measured on native radiograph pixel tolerances.
