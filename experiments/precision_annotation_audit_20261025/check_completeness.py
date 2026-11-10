import json
from collections import Counter
from pathlib import Path
import pandas as pd,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)
cfg=yaml.safe_load((ROOT/'configs/pilot.yaml').read_text());raw=pd.read_csv(cfg['data']['annotations_csv']);inv=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');studies=pd.read_parquet(ROOT/'data/study_inventory.parquet')
def sig(x):return tuple(map(tuple,np.asarray(json.loads(x) if isinstance(x,str) else x,float)))
r=raw[raw.label.str.startswith('CVC')];a=inv[inv.device_type.eq('CVC')]
rg={k:Counter(map(sig,g.data)) for k,g in r.groupby('StudyInstanceUID')};ag={k:Counter(map(sig,g.polyline)) for k,g in a.groupby('study_id')}
train=pd.read_csv(ROOT/'experiments/simple_first_v1/comparison/training_cohort.csv');val=json.loads((ROOT/'data/splits.json').read_text())['partitions']['dev_pilot_val']['study_ids']
rows=[]
for cohort,ids in [('experiment_train',train.StudyInstanceUID.tolist()),('development_validation',val)]:
 for sid in ids:
  z=studies[studies.StudyInstanceUID.eq(sid)].iloc[0];p=a[a.study_id.eq(sid)]
  rows.append(dict(cohort=cohort,study_id=sid,label_has_cvc=bool(z.has_cvc),inventory_instances=len(p),source_instances=sum(rg.get(sid,Counter()).values()),exact_polylines_preserved=ag.get(sid,Counter())==rg.get(sid,Counter()),label_positive_without_polyline=bool(z.has_cvc) and len(p)==0,label_negative_with_polyline=not bool(z.has_cvc) and len(p)>0,short_paths_under300=int((p.polyline_length_px<300).sum()),border_contact_paths=int(p.is_border_contact.sum()),unassessable_tips=int((~p.is_assessable).sum())))
df=pd.DataFrame(rows);df.to_csv(OUT/'cohort_completeness.csv',index=False)
summary={}
for name,g in df.groupby('cohort'):
 summary[name]={'studies':len(g),'source_polyline_mismatch_studies':int((~g.exact_polylines_preserved).sum()),**{k:int(g[k].sum()) for k in ['inventory_instances','label_positive_without_polyline','label_negative_with_polyline','short_paths_under300','border_contact_paths','unassessable_tips']}}
summary['whole_inventory_label_positive_without_cvc_polyline']=int((studies.has_cvc&~studies.StudyInstanceUID.isin(a.study_id)).sum())
(OUT/'completeness_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
