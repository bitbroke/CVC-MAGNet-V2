# Coordinate Decoder Comparison: D0 (Cell Center) vs D1 (DARK CVPR 2020)

**Date**: 2026-10-17  
**Model Checkpoint**: `runs/t2_joint_seg_tip_1000/checkpoints/final.pt`  
**Threshold**: 0.2 | **NMS Suppression Radius**: 3.0 px  
**Diagnostics**:
- Total raw peaks above threshold: 6312
- Post-NMS peaks retained: 574 (mean 2.97 / image)

---

## Benchmark Comparison (Hungarian Matching on Ground Truth Tips)

| Metric                    | D0 (Cell Center)   | D1 (DARK Taylor)   | Delta (D1 - D0)   |
|:--------------------------|:-------------------|:-------------------|:------------------|
| Success@5                 | 5.46%              | 8.44%              | +2.98%            |
| Success@10                | 17.87%             | 21.84%             | +3.97%            |
| Success@15                | 30.52%             | 33.75%             | +3.23%            |
| Success@20                | 39.45%             | 39.21%             | -0.25%            |
| Success@30                | 50.62%             | 51.61%             | +0.99%            |
| Matched Error Median (px) | 9.16 px            | 8.69 px            | -0.48 px          |
| Matched Error P95 (px)    | 14.23 px           | 14.16 px           | -0.07 px          |

---

## Conclusion
- **D0 (Cell Center)**: Direct scaling of integer grid coordinates $(x + 0.5) 	imes 4$.
- **D1 (DARK Subpixel Refinement)**: Taylor expansion around local Gaussian peak surface using gradient and Hessian.
- Benefit of subpixel refinement is strictly measured on native radiograph pixel tolerances.
