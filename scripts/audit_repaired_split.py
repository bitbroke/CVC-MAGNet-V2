"""Read-only split identity and exact image-content audit for the active pilot."""
from pathlib import Path
import json, hashlib
from collections import defaultdict
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/paired_repaired_20261016'
splits=json.loads((ROOT/'data/splits.json').read_text())['partitions']
inventory=pd.read_parquet(ROOT/'data/study_inventory.parquet').set_index('StudyInstanceUID')
images=Path('C:/Users/M S I/Downloads/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification/train')
ids={k:splits[k]['study_ids'] for k in ['dev_pilot_train','dev_pilot_val']}
a,b=ids.values()
result={'study_overlap':len(set(a)&set(b)),'counts':{k:len(v) for k,v in ids.items()}}
for c in ['PatientID','duplicate_group_id']:
    assert inventory.loc[a+b,c].notna().all()
    result[c+'_overlap']=len(set(inventory.loc[a,c])&set(inventory.loc[b,c]))
hashes={}
for split,studies in ids.items():
    hashes[split]=defaultdict(list)
    for sid in studies:
        p=images/f'{sid}.jpg'
        assert p.stat().st_size>0
        hashes[split][hashlib.sha256(p.read_bytes()).hexdigest()].append(sid)
shared=set(hashes['dev_pilot_train'])&set(hashes['dev_pilot_val'])
result['cross_split_identical_image_hash_groups']=len(shared)
result['collisions']={h:{k:v[h] for k,v in hashes.items()} for h in shared}
result['limitations']='Exact-byte and recorded-group checks do not detect all visually similar or re-encoded duplicates, historical selection exposure, or external validity.'
(OUT/'split_integrity.json').write_text(json.dumps(result,indent=2))
(OUT/'pilot_image_hashes.json').write_text(json.dumps(hashes,indent=2))
assert all(result[k]==0 for k in ['study_overlap','PatientID_overlap','duplicate_group_id_overlap','cross_split_identical_image_hash_groups'])
print(json.dumps(result,indent=2))
