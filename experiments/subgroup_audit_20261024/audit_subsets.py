"""Post-hoc development subgroup audit; no training or original data edits."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=False)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x): p.write_text(json.dumps(x, indent=2, allow_nan=False), encoding='utf-8')
sources = [ROOT/'data/study_inventory.parquet', ROOT/'data/annotation_inventory.parquet', ROOT/'data/splits.json']
inv = pd.read_parquet(sources[0]).rename(columns={'StudyInstanceUID':'study_id','PatientID':'patient_id'})
assert inv.study_id.is_unique
ann = pd.read_parquet(sources[1]); cvc = ann[ann.device_type.eq('CVC')]
counts = cvc.groupby('study_id').size()
inv['annotated_cvc_count'] = inv.study_id.map(counts).fillna(0).astype(int)
other_cols = [c for c in inv if c.startswith(('ETT -', 'NGT -')) or c == 'Swan Ganz Catheter Present']
inv['other_device_present'] = inv[other_cols].fillna(0).gt(0).any(axis=1)
inv['low_clutter_proxy'] = inv.annotated_cvc_count.eq(1) & ~inv.other_device_present
splits = json.loads(sources[2].read_text())['partitions']
val_ids = set(splits['dev_pilot_val']['study_ids'])
val = inv[inv.study_id.isin(val_ids)].copy()
assert len(val) == len(val_ids) == 193
groups = {
 'all_cvc_positive': lambda d: d.has_cvc.astype(bool),
 'single_cvc': lambda d: d.annotated_cvc_count.eq(1),
 'two_cvcs': lambda d: d.annotated_cvc_count.eq(2),
 'three_or_more_cvcs': lambda d: d.annotated_cvc_count.ge(3),
 'simple_single_no_other_device': lambda d: d.low_clutter_proxy,
 'single_with_other_devices': lambda d: d.annotated_cvc_count.eq(1) & d.other_device_present,
 'multiple_no_other_device': lambda d: d.annotated_cvc_count.ge(2) & ~d.other_device_present,
 'multiple_with_other_devices': lambda d: d.annotated_cvc_count.ge(2) & d.other_device_present,
}
base = ROOT/'experiments/hrnet_segmentation_features_v1/comparison'
boundary = ROOT/'experiments/hrnet_boundary_precision_v1/comparison'
models = {'original_hrnet':(base,'baseline'), 'recovery_parent':(base,'feature_adaptation'), 'boundary_candidate':(boundary,'boundary')}
rows=[]
for model,(folder,prefix) in models.items():
    native=folder/f'{prefix}_native.json'; paths=folder/f'{prefix}_paths.csv'
    sources.extend([native,paths])
    data=json.loads(native.read_text()); per=pd.DataFrame(data['per_study'])
    assert set(per.study_id)==val_ids and per.study_id.is_unique
    df=val.merge(per,on='study_id',validate='one_to_one'); p=pd.read_csv(paths)
    assert not p.duplicated(['study_id','catheter_instance_id']).any()
    for name,rule in groups.items():
        sub=df[rule(df)]; pp=p[p.study_id.isin(sub.study_id)]
        if not len(sub): continue
        assert len(pp)==sub.annotated_cvc_count.sum()
        # Patient-cluster bootstrap for mean study Dice, preserving within-patient images.
        means=sub.groupby('patient_id').dice.agg(['sum','count']); rng=np.random.default_rng(42)
        boot=[]
        for _ in range(1000):
            z=means.iloc[rng.integers(0,len(means),len(means))]
            boot.append(z['sum'].sum()/z['count'].sum())
        lo,hi=np.quantile(boot,[.025,.975])
        rows.append(dict(model=model,subset=name,studies=len(sub),patients=sub.patient_id.nunique(),paths=len(pp),
          mean_study_dice=sub.dice.mean(),dice_ci_low=lo,dice_ci_high=hi,
          path_coverage10=pp.coverage10.mean(),missed_paths=int(pp.missed.sum()),
          missed_rate=float(pp.missed.mean()),mean_longest_gap_px=pp.longest_gap_native_px.mean()))
    df.to_csv(OUT/f'{model}_per_study.csv',index=False)
pd.DataFrame(rows).to_csv(OUT/'subgroup_metrics.csv',index=False)
for name,rule in groups.items():
    val[rule(val)].to_csv(OUT/f'validation_{name}.csv',index=False)
# Training candidates are defined by metadata only, never by validation correctness.
train_ids=set(splits['dev_full_train']['study_ids']) | set(splits['dev_pilot_train']['study_ids'])
train=inv[inv.study_id.isin(train_ids)].copy()
for key in ['study_id','patient_id','duplicate_group_id']:
    assert not set(train[key]) & set(val[key]), f'{key} overlaps validation'
train[train.low_clutter_proxy].to_csv(OUT/'training_simple_candidates.csv',index=False)
train[train.is_study_level_negative.astype(bool)].to_csv(OUT/'training_negative_candidates.csv',index=False)
val.to_csv(OUT/'full_validation_registry_unchanged.csv',index=False)
save(OUT/'provenance.json', {'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
 'evaluation':'post-hoc development analysis; not independent generalization',
 'simple_definition':'Exactly one inventory-annotated CVC; no image-level ETT/NGT/Swan positive label. Not a clinical image-quality judgment.',
 'limitations':['Counts are annotation inventory counts, not proof of all visible CVCs.',
 'No tip-accuracy claim; metrics measure segmentation against rasterized polylines.',
 'Groups overlap; do not sum subgroup counts.', 'Negative cohort retained separately: only eight validation studies.',
 'Selecting the highest scoring subgroup does not improve population accuracy.'],
 'training_simple_candidates':int(train.low_clutter_proxy.sum()),'training_negative_candidates':int(train.is_study_level_negative.sum()),
 'originals_modified':False,'training_started':False})
lines=['# Frozen-model dataset subgroup audit','', 'Native-resolution segmentation, threshold 0.50; unchanged 193-study development cohort.',
 'Simple = one annotated CVC and no labeled ETT/NGT/Swan. Counts are inventory-based, not clinical confirmation.', '',
 '| Model | Subset | Studies / patients / paths | Dice | Path coverage @10px | Missed paths |',
 '|---|---|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['model']} | {r['subset']} | {r['studies']} / {r['patients']} / {r['paths']} | {r['mean_study_dice']:.3f} | {r['path_coverage10']:.1%} | {r['missed_paths']} |")
lines += ['', 'Patient-bootstrap Dice intervals are in subgroup_metrics.csv. Groups overlap. Do not interpret small-subgroup rankings as proven superiority.',
 'Training-only simple and negative manifests are candidates for a controlled curriculum, not a replacement dataset. No records removed; no training launched.',
 'Keep complex cases for later training and all cohorts for evaluation. Tip labels and full visible catheter counts still require verification.']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
print('TRAIN SIMPLE',int(train.low_clutter_proxy.sum()),'NEGATIVE',int(train.is_study_level_negative.sum()))
