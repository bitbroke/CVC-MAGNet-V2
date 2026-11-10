"""Read-only analysis of completed controlled pilot, with paired patient bootstrap."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/hrnet_missed_catheter_pilot_v1/comparison'
def main():
    assert json.loads((OUT/'status.json').read_text())['status']=='complete'
    results=json.loads((OUT/'comparison.json').read_text())
    histories={arm:json.loads((OUT/arm/'history.json').read_text()) for arm in ['control','instance_recall']}
    for history in histories.values():
        assert [r['update'] for r in history]==list(range(1,251)), 'Incomplete successful-update budget'
    np.testing.assert_allclose(histories['control'][0]['loss'][1:3],histories['instance_recall'][0]['loss'][1:3],rtol=0,atol=1e-7)
    rows=[]
    for name,r in results.items():
        s=r['native'];t=r['mixed_provisional_tips']
        assert (s['num_positive_studies'],s['num_negative_studies'],s['total_catheters_evaluated'])==(185,8,248)
        rows.append(dict(arm=name,dice=s['macro_dice'],coverage_10=s['centerline_coverage_10px'],missed_rate_pct=s['missed_catheter_rate'],missed_count=round(s['missed_catheter_rate']*s['total_catheters_evaluated']/100),precision=s['precision'],recall=s['recall'],negative_false_area_pct=s['negative_false_area_pct'],tip_matches_15=t['matches_15px'],tip_targets=41,false_tips_per_image=t['mean_false_tips_per_image']))
    table=pd.DataFrame(rows).set_index('arm');table.to_csv(OUT/'comparison_table.csv')
    b,c,e=[table.loc[x] for x in ['baseline','control','instance_recall']]
    passes=bool(e.missed_count<c.missed_count and e.coverage_10>=b.coverage_10-.01)
    inventory=pd.read_parquet(ROOT/'data/study_inventory.parquet').set_index('StudyInstanceUID')
    a=pd.DataFrame(json.loads((OUT/'control_native.json').read_text())['per_study']).set_index('study_id')
    z=pd.DataFrame(json.loads((OUT/'instance_recall_native.json').read_text())['per_study']).set_index('study_id')
    assert set(a.index)==set(z.index)
    a=a[a.is_positive];z=z.loc[a.index]
    delta=z.centerline_coverage_10px-a.centerline_coverage_10px
    paired=pd.DataFrame({'delta':delta,'patient':inventory.loc[a.index,'PatientID']})
    groups=[g.delta.to_numpy() for _,g in paired.groupby('patient')]
    rng=np.random.default_rng(42)
    draws=[np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]).mean() for _ in range(2000)]
    ci=np.quantile(draws,[.025,.975]).tolist()
    decision=dict(primary_rule_passed=passes,patient_cluster_bootstrap_study_mean_coverage_delta=float(delta.mean()),ci95=ci,patient_count=len(groups),note='Study-weighted paired diagnostic; canonical headline coverage may use a different aggregation. Repeated development selection and single seed limit generalization claims.')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2))
    lines=['# Missed-catheter recall pilot — completed','',
      'All checkpoints were preserved. Both arms started from the same frozen T1 weights and received the identical 250-update image/augmentation ledger. The only arm difference was a 0.25 equal-instance soft recall loss. BatchNorm running statistics were frozen in both arms. Unverified tips were excluded from tip supervision; segmentation used existing CVC polylines.','',
      '| Arm | Dice | Coverage @10 | Missed /248 | Precision | Negative false area | Tips @15 /41 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in table.iterrows():lines.append(f'| {name} | {r.dice:.4f} | {r.coverage_10:.2%} | {int(r.missed_count)} | {r.precision:.2%} | {r.negative_false_area_pct:.4f}% | {int(r.tip_matches_15)} |')
    lines+=['','Metric definitions: Dice is the mean over 185 positive studies. Centerline coverage is the mean over 248 catheter paths; a path with less than 30% of sampled points within 10 native pixels of foreground is counted as missed. Precision is the legacy evaluator mean across all 193 studies (including negative-study conventions), not pooled pixel precision. Negative false area is the mean across eight negative studies.',f'Predeclared progression rule: **{"PASS" if passes else "FAIL"}**. Requires fewer missed paths than control and baseline coverage retained within one percentage point.',
      f'Paired study-mean coverage difference (experiment minus control): {delta.mean()*100:.2f} percentage points; patient-cluster 95% bootstrap interval [{ci[0]*100:.2f}, {ci[1]*100:.2f}].',
      '', 'A progression pass alone does not justify promotion: inspect precision, negative false area, tip performance and paired overlays. The tip benchmark remains mixed provisional (including 16 unverified endpoints); it is a development diagnostic, not clinical accuracy. Complete-provisional-subset results are retained in each metrics JSON. No external generalization, overfitting exclusion or millimeter accuracy claim is justified. Masking weak labels during continuation does not undo their historical influence in the inherited checkpoint. This is a controlled development continuation, not a clean-label retraining.',
      '', 'Tip-supervision exposure: the common ledger contains 17 provisional-positive exposures, 250 confirmed-negative exposures and 733 masked exposures. This imbalance is a plausible contributor to tip suppression, not an isolated causal finding. Do not treat this experiment as a sufficient test of the achievable tip-localization performance. A subsequent segmentation-focused experiment should explicitly protect the existing tip behavior rather than assume sparse joint supervision preserves it.',
      '', 'Next: retain the baseline unless the complete trade-off is acceptable. Replicate a promising improvement on another seed before scaling. If the primary rule fails, do not extend this loss configuration or overwrite the baseline.']
    if e.dice<b.dice and e.tip_matches_15<b.tip_matches_15:
        lines+=['','## Final disposition: do not promote',
          'The experiment passes the narrow coverage progression rule but has lower Dice and fewer correctly localized provisional tips than the frozen baseline. Its missed-path count equals, rather than improves upon, the baseline. Keep the frozen baseline as the main reference; retain this checkpoint only as evidence of a recall-versus-mask-quality trade-off. The reduced false area on eight negatives does not compensate for poorer positive-study masks or establish reliable specificity.',
          'A next experiment should constrain positive-study false foreground and protect tip behavior while retaining the instance-recall signal. Do not expand training duration for this configuration unchanged. Do not launch a second-seed confirmation merely because the narrow coverage rule passed.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    print(table.to_string());print(decision)
if __name__=='__main__':main()
