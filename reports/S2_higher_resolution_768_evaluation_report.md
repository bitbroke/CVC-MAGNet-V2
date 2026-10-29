# S2_higher_resolution_768_evaluation_report: CVC Segmentation Mask Evaluation

- **Checkpoint**: `C:\Ankshit\CVC-MAGNet-V2\runs\s2_higher_resolution_768\checkpoints\best_segmentation_v3.pt`
- **SHA256**: `95179360e1c546473100a3354019418d9be0e36cf138b09e37ad9f57a75884ee`
- **Evaluation Partition**: `dev_pilot_val` (193 studies: 185 positive, 8 negative)
- **Evaluation Principle**: Direct segmentation mask assessment excluding letterbox padding; tip Success@15 omitted.

## Performance Across Operating Thresholds

| Threshold ($\tau$) | Macro Dice | Macro IoU | Precision | Recall | Centerline Cov (10px) | Missed Catheter Rate (10px) | Negatives False Area (px) | Negatives Affected Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 0.0083 | 0.0042 | 0.0042 | 0.9999 | 100.0% | 0.0% | 486851.9 px | 100.0% |
| **0.20** | 0.0844 | 0.0447 | 0.0267 | 0.8780 | 92.0% | 0.0% | 33890.6 px | 100.0% |
| **0.30** | 0.1872 | 0.1049 | 0.1068 | 0.7921 | 83.0% | 1.6% | 9084.9 px | 100.0% |
| **0.40** | 0.2493 | 0.1451 | 0.1532 | 0.7407 | 78.7% | 3.6% | 4936.8 px | 100.0% |
| **0.50** | 0.3061 | 0.1847 | 0.2033 | 0.6854 | 73.9% | 6.5% | 2798.8 px | 100.0% |
| **0.60** | 0.3620 | 0.2268 | 0.2821 | 0.6020 | 67.6% | 8.9% | 1607.4 px | 100.0% |
| **0.70** | 0.3528 | 0.2251 | 0.4055 | 0.3596 | 49.9% | 22.6% | 650.6 px | 87.5% |

## Baseline Operating Point Analysis ($\tau = 0.30$, Viewer Default)
- **Macro Dice (Positives)**: 0.1872
- **Pooled Pixel Precision**: 0.1068 (Severe precision penalty due to diffuse foreground)
- **Pooled Pixel Recall**: 0.7921
- **Centerline Coverage @ 10px**: 83.0%
- **Missed Catheters (<30% coverage)**: 1.6%
- **Negative Studies Affected**: 100.0% with mean false area 9084.9 px

### Stratified Performance by Clinical Cohort ($\tau = 0.30$)
- **Single CVC (N=126)**: Macro Dice = 0.1671
- **Multiple CVC (N=59)**: Macro Dice = 0.2302
- **Abnormal Placement (N=23)**: Macro Dice = 0.2012
- **Cluttered / Other Lines (N=94)**: Macro Dice = 0.1727
