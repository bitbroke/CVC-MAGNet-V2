# S0: Frozen Model CVC Segmentation Mask Evaluation

- **Checkpoint**: `C:\Ankshit\CVC-MAGNet-V2\runs\pilot_experiment_expanded\checkpoints\final_coarse_dual_head.pt`
- **SHA256**: `ed3e2acb2e0229f0452b00a41ed527a1ebfa65e0f9ef920d3dbe05ea626b5b55`
- **Evaluation Partition**: `dev_pilot_val` (193 studies: 185 positive, 8 negative)
- **Evaluation Principle**: Direct segmentation mask assessment excluding letterbox padding; tip Success@15 omitted.

## Performance Across Operating Thresholds

| Threshold ($\tau$) | Macro Dice | Macro IoU | Precision | Recall | Centerline Cov (10px) | Missed Catheter Rate (10px) | Negatives False Area (px) | Negatives Affected Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 0.0083 | 0.0042 | 0.0042 | 1.0000 | 100.0% | 0.0% | 216296.5 px | 100.0% |
| **0.20** | 0.0772 | 0.0405 | 0.0396 | 0.8673 | 91.7% | 0.4% | 17399.0 px | 100.0% |
| **0.30** | 0.1449 | 0.0792 | 0.0781 | 0.7798 | 83.5% | 3.2% | 6614.4 px | 100.0% |
| **0.40** | 0.1777 | 0.0991 | 0.1007 | 0.7290 | 78.7% | 4.0% | 4520.1 px | 100.0% |
| **0.50** | 0.2053 | 0.1165 | 0.1219 | 0.6867 | 74.8% | 4.4% | 3336.1 px | 100.0% |
| **0.60** | 0.2311 | 0.1333 | 0.1447 | 0.6456 | 70.9% | 6.0% | 2491.6 px | 100.0% |
| **0.70** | 0.2596 | 0.1527 | 0.1744 | 0.5957 | 66.5% | 9.3% | 1736.0 px | 100.0% |

## Baseline Operating Point Analysis ($\tau = 0.30$, Viewer Default)
- **Macro Dice (Positives)**: 0.1449
- **Pooled Pixel Precision**: 0.0781 (Severe precision penalty due to diffuse foreground)
- **Pooled Pixel Recall**: 0.7798
- **Centerline Coverage @ 10px**: 83.5%
- **Missed Catheters (<30% coverage)**: 3.2%
- **Negative Studies Affected**: 100.0% with mean false area 6614.4 px

### Stratified Performance by Clinical Cohort ($\tau = 0.30$)
- **Single CVC (N=126)**: Macro Dice = 0.1232
- **Multiple CVC (N=59)**: Macro Dice = 0.1913
- **Abnormal Placement (N=23)**: Macro Dice = 0.1620
- **Cluttered / Other Lines (N=94)**: Macro Dice = 0.1339
