"""Report fixed-step outcomes and matched per-patient subgroup uncertainty."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'comparison';RUN=OUT/'run'
baseline=ROOT/'experiments/hrnet_segmentation_features_v1/comparison'
oldhash=json.loads((baseline/'source_hashes.json').read_text())
for filename in ['splits.json','annotation_inventory.parquet','study_inventory.parquet']:
    path=ROOT/'data'/filename
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==oldhash[str(path)]
inv=pd.read_parquet(ROOT/'data/study_inventory.parquet').rename(columns={'StudyInstanceUID':'study_id','PatientID':'patient_id'})
ann=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');ann=ann[ann.device_type.eq('CVC')]
inv['n_cvc']=inv.study_id.map(ann.groupby('study_id').size()).fillna(0).astype(int)
cols=[c for c in inv if c.startswith(('ETT -','NGT -')) or c=='Swan Ganz Catheter Present']
inv['other']=inv[cols].fillna(0).gt(0).any(axis=1)
rules={'all_positive':lambda d:d.is_positive,'simple':lambda d:d.n_cvc.eq(1)&~d.other,'single_with_other':lambda d:d.n_cvc.eq(1)&d.other,'two_cvcs':lambda d:d.n_cvc.eq(2),'three_plus':lambda d:d.n_cvc.ge(3),'multiple_with_other':lambda d:d.n_cvc.ge(2)&d.other}
rows=[];tables={};paths={};summaries={}
for name,folder,prefix in [('original',baseline,'baseline'),('mixed_data',RUN,'mixed_data'),('simple_first',RUN,'simple_first')]:
    r=json.loads((folder/f'{prefix}_native.json').read_text());summaries[name]=r['summary']
    d=inv.merge(pd.DataFrame(r['per_study']),on='study_id',validate='one_to_one');tables[name]=d
    p=pd.read_csv(folder/f'{prefix}_paths.csv');paths[name]=p
    for group,fn in rules.items():
        sub=d[fn(d)];pp=p[p.study_id.isin(sub.study_id)]
        if len(sub):rows.append(dict(model=name,group=group,studies=len(sub),patients=sub.patient_id.nunique(),paths=len(pp),dice=float(sub.dice.mean()),coverage10=float(pp.coverage10.mean()),misses=int(pp.missed.sum()),mean_gap_px=float(pp.longest_gap_native_px.mean())))
pd.DataFrame(rows).to_csv(RUN/'subgroup_metrics.csv',index=False)
pairs=tables['mixed_data'][['study_id','patient_id','is_positive','dice']].merge(tables['simple_first'][['study_id','dice']],on='study_id',suffixes=('_mixed','_simple'),validate='one_to_one')
pairs=pairs[pairs.is_positive];pairs['difference']=pairs.dice_simple-pairs.dice_mixed;pairs.to_csv(RUN/'paired_study_dice.csv',index=False)
g=pairs.groupby('patient_id').difference.agg(['sum','count']).to_numpy();rng=np.random.default_rng(42);idx=rng.integers(0,len(g),(5000,len(g)));boot=g[idx,0].sum(axis=1)/g[idx,1].sum(axis=1)
ci=np.quantile(boot,[.025,.975]);(RUN/'paired_dice_bootstrap.json').write_text(json.dumps(dict(mean_difference=float(pairs.difference.mean()),ci95=ci.tolist(),unit='patient',replicates=5000),indent=2))
keys=['study_id','catheter_instance_id'];pp=paths['mixed_data'].merge(paths['simple_first'],on=keys,suffixes=('_mixed','_simple'),validate='one_to_one');pp['coverage_change']=pp.coverage10_simple-pp.coverage10_mixed;pp.to_csv(RUN/'paired_paths.csv',index=False)
# Track the four previously recovered paths separately; no hidden aggregate substitution.
recover=pd.read_csv(ROOT/'experiments/hrnet_boundary_precision_v1/comparison/recovered_path_checks.csv')[keys]
recover.merge(pp,on=keys,validate='one_to_one').to_csv(RUN/'previously_recovered_paths.csv',index=False)
decision=json.loads((RUN/'decision.json').read_text());lines=['# Simple-first vs mixed-data pilot','', 'Same positive/negative image and augmentation multisets; only presentation order differs. Both start from original frozen HRNet. No checkpoints promoted.','', '| Model | Dice | Precision | Coverage10 | Misses | Negative FP pixels |','|---|---:|---:|---:|---:|---:|']
quality={name:dict(dice_preserved=s['macro_dice']>=summaries['original']['macro_dice'],precision_preserved=s['precision']>=summaries['original']['precision'],coverage_preserved=s['centerline_coverage_10px']>=summaries['original']['centerline_coverage_10px']) for name,s in summaries.items() if name!='original'}
(RUN/'original_quality_comparison.json').write_text(json.dumps(quality,indent=2))
for name,s in summaries.items():lines.append(f"| {name} | {s['macro_dice']:.4f} | {s['precision']:.2%} | {s['centerline_coverage_10px']:.2%} | {int(paths[name].missed.sum())}/248 | {s['negative_total_false_pixels']} |")
lines+=['',f"Paired mean study Dice difference (simple-first minus mixed): {pairs.difference.mean():+.4f}; patient-bootstrap 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}].",f"Predeclared safeguards: {'PASS' if decision['passed'] else 'FAIL'}.",'','| Model | Group | Studies | Dice | Coverage10 | Misses |','|---|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['model']} | {r['group']} | {r['studies']} | {r['dice']:.4f} | {r['coverage10']:.2%} | {r['misses']}/{r['paths']} |")
lines+=['','One seed, 250-update continuation pilot, repeatedly used development cohort; eight negatives. Simple means metadata-based low clutter, not clinical confirmation. A nonzero subgroup difference is not proof of generalization. No claim of improved tip accuracy: complete original tip branch remains separate and frozen.','', 'Labels remain rasterized polyline annotations. Visual sample review is provisional engineering inspection. Full dataset and all existing models are preserved.']
lines+=['','The visual review of 12 training images found loops, faint lines, external wires, poor contrast and short peripheral traces despite passing the low-clutter metadata rule. None was relabeled or called expert-verified.','', 'Timing caveat: the control history contains a long wall-clock gap. Its reported elapsed seconds must not be interpreted as uninterrupted GPU compute time.']
(RUN/'REPORT.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines[:12]))
