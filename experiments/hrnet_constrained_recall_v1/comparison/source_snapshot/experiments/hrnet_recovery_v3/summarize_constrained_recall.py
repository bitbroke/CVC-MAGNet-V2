"""Report the locked constrained-recall decision without promoting checkpoints."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/hrnet_constrained_recall_v1/comparison'
def main():
    assert json.loads((OUT/'status.json').read_text())['status']=='complete'
    r=json.loads((OUT/'comparison.json').read_text());rows=[]
    for name,x in r.items():
        n=x['native'];t=x['mixed_provisional_tips']
        assert (n['num_positive_studies'],n['num_negative_studies'],n['total_catheters_evaluated'])==(185,8,248)
        rows.append(dict(arm=name,dice=n['macro_dice'],precision=n['precision'],coverage=n['centerline_coverage_10px'],missed=round(n['missed_catheter_rate']*248/100),negative_false_area_pct=n['negative_false_area_pct'],tips15=t['matches_15px'],false_tips=t['mean_false_tips_per_image']))
    table=pd.DataFrame(rows).set_index('arm');table.to_csv(OUT/'comparison_table.csv')
    b,c,e=[table.loc[x] for x in ['baseline','control','constrained']]
    tests=dict(fewer_misses_than_baseline=bool(e.missed<b.missed),fewer_misses_than_control=bool(e.missed<c.missed),dice_preserved=bool(e.dice>=b.dice),precision_preserved=bool(e.precision>=b.precision),coverage_preserved=bool(e.coverage>=b.coverage),negative_false_area_preserved=bool(e.negative_false_area_pct<=b.negative_false_area_pct),tip_metrics_identical=r['constrained']['mixed_provisional_tips']==r['baseline']['mixed_provisional_tips'])
    histories={name:json.loads((OUT/name/'history.json').read_text()) for name in ['control','constrained']}
    for name,h in histories.items():
        assert [v['update'] for v in h]==list(range(1,251))
        assert json.loads((OUT/name/'freeze_verification.json').read_text())['fp32_tip_logits_bitwise_equal']
    np.testing.assert_allclose(histories['control'][0]['loss'][1:],histories['constrained'][0]['loss'][1:],atol=1e-7,rtol=0)
    inv=pd.read_parquet(ROOT/'data/study_inventory.parquet').set_index('StudyInstanceUID')
    a=pd.DataFrame(json.loads((OUT/'control_native.json').read_text())['per_study']).set_index('study_id')
    z=pd.DataFrame(json.loads((OUT/'constrained_native.json').read_text())['per_study']).set_index('study_id')
    assert set(a.index)==set(z.index);a=a[a.is_positive];z=z.loc[a.index]
    d=pd.DataFrame({'delta':z.centerline_coverage_10px-a.centerline_coverage_10px,'patient':inv.loc[a.index,'PatientID']})
    groups=[g.delta.to_numpy() for _,g in d.groupby('patient')];rng=np.random.default_rng(42)
    draws=[np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]).mean() for _ in range(2000)]
    ci=np.quantile(draws,[.025,.975]).tolist()
    decision=dict(criteria=tests,all_criteria_passed=all(tests.values()),promoted=False,paired_study_mean_coverage_difference=float(d.delta.mean()),patient_cluster_ci95=ci,patient_count=len(groups),note='Single-seed repeatedly used development benchmark; no external validation. Study-weighted interval differs from catheter-weighted headline coverage.')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2))
    lines=['# Constrained recall pilot — 2026-10-18','',
      'Both arms completed 250 successful updates on an identical image/augmentation ledger, initialized from the frozen HRNet T1 baseline. Only segmentation-head parameters changed. Shared encoder, fusion and tip-head tensors remained bitwise unchanged. Both arms used BCE + Dice + 0.25 instance recall loss. The constrained arm additionally used 0.25 hard-background penalty and 0.25 reference-foreground retention penalty. No labels were promoted or thresholds tuned.',
      '', '| Model | Missed paths /248 | Coverage @10px | Dice | Mean-study precision | Negative false area | Tips @15px /41 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name,x in table.iterrows():lines.append(f'| {name} | {int(x.missed)} | {x.coverage:.2%} | {x.dice:.4f} | {x.precision:.2%} | {x.negative_false_area_pct:.5f}% | {int(x.tips15)} |')
    lines+=['',f'**Locked all-criteria decision: {"PASS" if all(tests.values()) else "FAIL"}. No checkpoint promoted.**','']
    lines += [f'- {k}: {"PASS" if v else "FAIL"}' for k,v in tests.items()]
    lines+=['',f'Paired study-mean coverage difference versus control: {d.delta.mean()*100:.2f} percentage points; patient-cluster bootstrap 95% interval [{ci[0]*100:.2f}, {ci[1]*100:.2f}].',
      '', 'Dice is positive-study macro Dice (185 studies). Coverage is catheter-weighted (248 paths). Missed means under 30% path coverage at 10 native pixels, not a tip-localization failure. Precision uses the inherited mean-study metric including negative-study conventions. Negative false area is the mean across only eight negatives.',
      '', 'The 41 tip targets remain provisional, including 16 unverified endpoints. Exact preservation verifies no tip regression on this model path; it does not establish clinical correctness. Freezing inherited weights does not remove their historical label or development-selection exposure. No claim of absent overfitting, clinical safety, millimeter accuracy or perfect-model performance is supported.',
      '', 'Next: retain the baseline if any locked criterion fails. If all pass, repeat the fixed experiment on a second seed and a larger independently reviewed evaluation cohort before promotion. Do not extend this configuration unchanged merely to seek a better development score.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    print(table.to_string());print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
