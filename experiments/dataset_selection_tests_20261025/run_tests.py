"""Frozen-output dataset selection diagnostics; no training or source changes."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[ROOT/'data/study_inventory.parquet',ROOT/'data/annotation_inventory.parquet',ROOT/'data/splits.json']
before={str(p):sha(p) for p in sources}
inv=pd.read_parquet(sources[0]).rename(columns={'StudyInstanceUID':'study_id','PatientID':'patient_id'});ann=pd.read_parquet(sources[1]);cvc=ann[ann.device_type.eq('CVC')].copy();splits=json.loads(sources[2].read_text())['partitions']
assert inv.study_id.is_unique
def valid(x):
 try:
  p=np.asarray(json.loads(x) if isinstance(x,str) else x,float)
  return p.ndim==2 and p.shape[1]==2 and len(p)>=2 and np.isfinite(p).all() and len(np.unique(p,axis=0))>=2
 except (ValueError,TypeError):return False
cvc['geometry_valid']=cvc.polyline.map(valid)
stats=cvc.groupby('study_id').agg(annotated_count=('geometry_valid','size'),all_geometry_valid=('geometry_valid','all'),total_path_length=('polyline_length_px','sum'))
inv=inv.merge(stats,on='study_id',how='left',validate='one_to_one');inv['annotated_count']=inv.annotated_count.fillna(0).astype(int);inv['all_geometry_valid']=inv.all_geometry_valid.fillna(False).astype(bool)
other=[c for c in inv if c.startswith(('ETT -','NGT -')) or c=='Swan Ganz Catheter Present'];inv['other_devices']=inv[other].fillna(0).gt(0).any(axis=1)
inv['placement']=np.select([inv['CVC - Abnormal'].gt(0),inv['CVC - Borderline'].gt(0),inv['CVC - Normal'].gt(0)],['abnormal_present','borderline_without_abnormal','normal_only'],default='no_cvc_label')
inv['eligible_positive']=inv.has_cvc.astype(bool)&inv.annotated_count.gt(0)&inv.all_geometry_valid
inv['eligible_negative']=inv.is_study_level_negative.astype(bool)&~inv.has_cvc.astype(bool)&inv.annotated_count.eq(0)
inv['status']=np.select([inv.eligible_positive,inv.eligible_negative,inv.has_cvc.astype(bool)&inv.annotated_count.eq(0)],['source_supported_positive','label_supported_negative','positive_missing_path'],default='review_required')
val=inv[inv.study_id.isin(splits['dev_pilot_val']['study_ids'])].copy();train=inv[inv.study_id.isin(set(splits['dev_pilot_train']['study_ids'])|set(splits['dev_full_train']['study_ids']))].copy()
overlap={k:len(set(train[k])&set(val[k])) for k in ['study_id','patient_id','duplicate_group_id']};assert not any(overlap.values())
groups={'all_annotated_positive':lambda d:d.eligible_positive,'single_no_other_devices':lambda d:d.eligible_positive&d.annotated_count.eq(1)&~d.other_devices,'single_with_other_devices':lambda d:d.eligible_positive&d.annotated_count.eq(1)&d.other_devices,'two_catheters':lambda d:d.eligible_positive&d.annotated_count.eq(2),'three_plus_catheters':lambda d:d.eligible_positive&d.annotated_count.ge(3),'multiple_no_other_devices':lambda d:d.eligible_positive&d.annotated_count.ge(2)&~d.other_devices,'multiple_with_other_devices':lambda d:d.eligible_positive&d.annotated_count.ge(2)&d.other_devices,'normal_only':lambda d:d.eligible_positive&d.placement.eq('normal_only'),'borderline_without_abnormal':lambda d:d.eligible_positive&d.placement.eq('borderline_without_abnormal'),'abnormal_present':lambda d:d.eligible_positive&d.placement.eq('abnormal_present')}
protocol={'tests':['geometry and label availability','patient/study/duplicate separation','three frozen model subgroup comparison','patient-bootstrap uncertainty','training-only cohort manifests'],'threshold':.5,'bootstrap_replicates':1000,'seed':42,'selection':'metadata only; no individual model-score filtering','limitations':['Repeated development cohort; not independent test','Overlapping subgroups; descriptive intervals not multiple-comparison adjusted','Inventory counts do not establish all visible catheters','No review attestation or dense boundary truth','No training started; no models loaded or modified']}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
pixel_path=ROOT/'experiments/precision_annotation_audit_20261025/results/pixel_diagnostics.csv';pixels=pd.read_csv(pixel_path);before[str(pixel_path)]=sha(pixel_path)
base=ROOT/'experiments/hrnet_segmentation_features_v1/comparison';recent=ROOT/'experiments/simple_first_v1/comparison/run';models={'original':(base/'baseline_native.json',base/'baseline_paths.csv'),'mixed':(recent/'mixed_data_native.json',recent/'mixed_data_paths.csv'),'simple_first':(recent/'simple_first_native.json',recent/'simple_first_paths.csv')}
rows=[]
for model,(native,path) in models.items():
 before[str(native)]=sha(native);before[str(path)]=sha(path)
 per=pd.DataFrame(json.loads(native.read_text())['per_study']);assert set(per.study_id)==set(val.study_id)
 d=val.merge(per,on='study_id',validate='one_to_one').merge(pixels[pixels.arm.eq(model)],on='study_id',validate='one_to_one');paths=pd.read_csv(path)
 for name,rule in groups.items():
  g=d[rule(d)];p=paths[paths.study_id.isin(g.study_id)];assert len(p)==g.annotated_count.sum()
  if not len(g):continue
  agg=g.groupby('patient_id').dice.agg(['sum','count']);rng=np.random.default_rng(42);bs=[]
  for _ in range(1000):
   a=agg.iloc[rng.integers(0,len(agg),len(agg))];bs.append(a['sum'].sum()/a['count'].sum())
  lo,hi=np.quantile(bs,[.025,.975]);rows.append(dict(model=model,subset=name,studies=len(g),patients=g.patient_id.nunique(),paths=len(p),dice=g.dice.mean(),dice_ci_low=lo,dice_ci_high=hi,positive_macro_precision=g.precision.mean(),pooled_precision=g.tp.sum()/max(1,g.pred.sum()),coverage10=p.coverage10.mean(),misses=int(p.missed.sum()),mean_longest_gap=p.longest_gap_native_px.mean(),false_pixels=int(g.fp.sum()),remote_fp_fraction=g.fp_beyond15.sum()/max(1,g.fp.sum()),training_studies=int(rule(train).sum())))
metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'subgroup_comparison.csv',index=False)
# Versioned proposals: no source rows deleted, no split reassignments.
train.to_csv(OUT/'training_inventory.csv',index=False);val.to_csv(OUT/'validation_registry_unchanged.csv',index=False)
for name,rule in groups.items():train[rule(train)].to_csv(OUT/f'candidate_{name}.csv',index=False)
train[train.eligible_negative].to_csv(OUT/'candidate_negatives.csv',index=False);train[~(train.eligible_positive|train.eligible_negative)].to_csv(OUT/'not_eligible_for_dense_supervision.csv',index=False)
# A fixed random training-only review queue stratified over difficulty, not validation failures.
def stratum(d):return np.where(d.annotated_count.ge(2),'multiple',np.where(d.other_devices,'single_cluttered','single_low_clutter'))
eligible=train[train.eligible_positive].copy();eligible['review_stratum']=stratum(eligible);selected=[];used=set()
for name,g in eligible.groupby('review_stratum'):
 g=g.sort_values('study_id').sample(frac=1,random_state=42)
 n=0
 for _,r in g.iterrows():
  if r.patient_id in used:continue
  selected.append(r.to_dict());used.add(r.patient_id);n+=1
  if n==10:break
review=pd.DataFrame(selected);review['review_status']='pending_not_verified';review.to_csv(OUT/'training_review_queue_30.csv',index=False)
summary={'source_geometry_invalid_instances':int((~cvc.geometry_valid).sum()),'all_inventory_status_counts':inv.status.value_counts().to_dict(),'allowed_training_status_counts':train.status.value_counts().to_dict(),'validation_status_counts':val.status.value_counts().to_dict(),'train_validation_overlap':overlap,'review_studies':len(review),'training_started':False,'original_data_modified':False}
(OUT/'integrity_summary.json').write_text(json.dumps(summary,indent=2));assert all(sha(Path(p))==v for p,v in before.items());(OUT/'input_hashes.json').write_text(json.dumps(before,indent=2))
lines=['DATASET SELECTION TEST SERIES — 25 October 2026','Five diagnostic tests completed. No models loaded/trained and no labels modified.','',json.dumps(summary,indent=2),'','Frozen-model comparisons:',metrics.to_string(index=False),'','Interpretation: Rank groups descriptively only. Best observed performance does not identify the best training distribution. Compare uncertainty and sample size; do not discard abnormal/multiple/faint cases. Candidate manifests are source-supported, not clinically verified. The 30-study training-only queue must be reviewed before declaring a high-confidence curated core. All negatives and the complete validation registry are retained.']
(OUT/'REPORT.txt').write_text('\n'.join(lines));print(json.dumps(summary,indent=2));print(metrics[metrics.model.eq('original')].to_string(index=False));print('DONE')
