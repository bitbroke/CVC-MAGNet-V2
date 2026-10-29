# S1_pretrained_segmentation_evaluation_report: CVC Segmentation Mask Evaluation

- **Checkpoint**: `C:\Ankshit\CVC-MAGNet-V2\runs\s1_pretrained_segmentation\checkpoints\best_segmentation_v3.pt`
- **SHA256**: `213f2ffe4a8d9039a77ba56a84ca8f1531f933b911b1fd9bb7e86510c2361b9d`
- **Evaluation Partition**: `dev_pilot_val` (193 studies: 185 positive, 8 negative)
- **Evaluation Principle**: Direct segmentation mask assessment excluding letterbox padding; tip Success@15 omitted.

## Performance Across Operating Thresholds

| Threshold ($\tau$) | Macro Dice | Macro IoU | Precision | Recall | Centerline Cov (10px) | Missed Catheter Rate (10px) | Negatives False Area (px) | Negatives Affected Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 0.0083 | 0.0042 | 0.0042 | 1.0000 | 100.0% | 0.0% | 216319.5 px | 100.0% |
| **0.20** | 0.0627 | 0.0327 | 0.0221 | 0.9218 | 95.1% | 0.0% | 22545.2 px | 100.0% |
| **0.30** | 0.1855 | 0.1038 | 0.1068 | 0.7723 | 80.9% | 3.6% | 4247.2 px | 100.0% |
| **0.40** | 0.2604 | 0.1522 | 0.1648 | 0.7095 | 75.2% | 7.3% | 2060.0 px | 100.0% |
| **0.50** | 0.2979 | 0.1784 | 0.2025 | 0.6589 | 70.9% | 9.7% | 1416.8 px | 100.0% |
| **0.60** | 0.3252 | 0.1987 | 0.2382 | 0.6021 | 66.2% | 10.1% | 1030.8 px | 100.0% |
| **0.70** | 0.3466 | 0.2157 | 0.2850 | 0.5229 | 59.4% | 16.5% | 696.9 px | 100.0% |

## Baseline Operating Point Analysis ($\tau = 0.30$, Viewer Default)
- **Macro Dice (Positives)**: 0.1855
- **Pooled Pixel Precision**: 0.1068 (Severe precision penalty due to diffuse foreground)
- **Pooled Pixel Recall**: 0.7723
- **Centerline Coverage @ 10px**: 80.9%
- **Missed Catheters (<30% coverage)**: 3.6%
- **Negative Studies Affected**: 100.0% with mean false area 4247.2 px

### Stratified Performance by Clinical Cohort ($\tau = 0.30$)
- **Single CVC (N=126)**: Macro Dice = 0.1702
- **Multiple CVC (N=59)**: Macro Dice = 0.2182
- **Abnormal Placement (N=23)**: Macro Dice = 0.1993
- **Cluttered / Other Lines (N=94)**: Macro Dice = 0.1702
