"""Matched exposure-order experiment; existing checkpoints and datasets are read-only."""
import sys,json,copy,gc,shutil,traceback,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,cv2,torch
from PIL import Image,ImageDraw
from collections import Counter, OrderedDict
from experiments.hrnet_recovery_v3.run_feature_pilot import train,native_eval,load,tensor,SplitInference
from experiments.hrnet_recovery_v3.run_recall_pilot import SOURCE,IMAGES_DIR,sha,save
from src.data.transforms import LetterboxTransform
from src.data.rasterize import rasterize_study_polylines
OUT=Path(__file__).resolve().parent/'comparison'

class DiskCache:
    """Bound decoded training RAM; preserve deterministic per-image cache contents."""
    def __init__(self,folder):
        self.folder=folder;self.ids=sorted(p.stem for p in folder.glob('*.npz'));self.recent=OrderedDict()
    def __getitem__(self,sid):
        if sid not in self.recent:
            with np.load(self.folder/f'{sid}.npz') as d:
                self.recent[sid]=(d['image'],list(d['instances']),d['valid'],torch.zeros(1,192,192),torch.ones(1,192,192),False)
            if len(self.recent)>16:self.recent.popitem(last=False)
        self.recent.move_to_end(sid);return self.recent[sid]
    def values(self):
        for sid in self.ids:yield self[sid]

def prepare():
    assert shutil.disk_usage(ROOT).free>3*2**30,'Need 3 GiB free'
    OUT.mkdir(exist_ok=False)
    save(OUT/'status.json',{'status':'preparing'})
    inv=pd.read_parquet(ROOT/'data/study_inventory.parquet').set_index('StudyInstanceUID')
    ann=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');ann=ann[ann.device_type.eq('CVC')]
    splits=json.loads((ROOT/'data/splits.json').read_text())['partitions']
    val=set(splits['dev_pilot_val']['study_ids'])
    allowed=set(splits['dev_full_train']['study_ids'])|set(splits['dev_pilot_train']['study_ids'])
    simple_df=pd.read_csv(ROOT/'experiments/subgroup_audit_20261024/results/training_simple_candidates.csv')
    simple=sorted(set(simple_df.study_id)&allowed)
    complex_ids=sorted((set(ann.study_id)&allowed)-set(simple))
    neg=sorted(inv.index[inv.is_study_level_negative.astype(bool)&~inv.has_cvc.astype(bool)].intersection(sorted(allowed)))
    rng=np.random.default_rng(42)
    spos=rng.choice(simple,375,replace=False).tolist();cpos=rng.choice(complex_ids,125,replace=False).tolist()
    latter=spos[250:]+cpos;rng.shuffle(latter);ordered=spos[:250]+latter
    mixed=ordered.copy();rng.shuffle(mixed)
    negatives=(rng.permutation(neg).tolist()*2)[:500]
    # Each positive exposure owns its augmentation in both arms.
    def aug():return dict(hflip=bool(rng.random()<.5),angle=float(rng.uniform(-5,5)),translate_x_pct=float(rng.uniform(-.02,.02)),translate_y_pct=float(rng.uniform(-.02,.02)),scale=float(rng.uniform(.95,1.05)),brightness=float(rng.uniform(.9,1.1)),contrast=float(rng.uniform(.9,1.1)))
    augpos={s:aug() for s in ordered};augneg=[aug() for _ in negatives]
    ids=sorted(set(ordered+negatives));assert set(ids)<=allowed and not set(ids)&val
    for col in ['PatientID','duplicate_group_id']:
        assert not inv.loc[ids,col].isna().any()
        assert not set(inv.loc[ids,col])&set(inv.loc[sorted(val),col])
    sources=[Path(__file__),ROOT/'data/splits.json',ROOT/'data/annotation_inventory.parquet',ROOT/'data/study_inventory.parquet',ROOT/'experiments/subgroup_audit_20261024/results/training_simple_candidates.csv',ROOT/'experiments/hrnet_recovery_v3/run_feature_pilot.py',ROOT/'experiments/hrnet_recovery_v3/run_recall_pilot.py']
    save(OUT/'source_hashes.json',{str(p):sha(p) for p in sources})
    frozen={str(p):sha(p) for p in ROOT.rglob('*.pt') if 'vendor' not in p.parts}
    save(OUT/'frozen_hashes.json',frozen)
    save(OUT/'protocol.json',dict(seed=42,updates_per_arm=250,positive_exposures=500,negative_exposures=500,positive_simple=375,positive_complex=125,
      initial=str(SOURCE),initial_sha256=sha(SOURCE),resolution=768,effective_batch=4,microbatch=1,
      optimizer='AdamW; head 5e-5, features 1e-5; cosine eta_min1e-6; weight_decay1e-4',
      loss='BCE posweight10 + foreground Dice + .25 equal-instance recall; identical arms',
      order='simple_first first125 updates: 2 simple +2negative; last125: mixed positive pool. mixed_data: shuffled identical positive exposures throughout. Negative slots identical.',
      scope='Tests exposure order under a fixed simple-enriched mixture, NOT full-population natural sampling or training from scratch.',
      primary='Full 193-study native macro Dice; path coverage/misses/precision/negative area safeguards vs matched control',
      pass_rule='Dice > mixed; precision >= mixed; path coverage >= mixed-.005; misses <= mixed; negative total FP and mean area <= mixed; original quality separately reported',
      limitations='Repeated development set, provisional polyline labels, eight negative validation studies; no clinical or independent accuracy claim',tip_policy='Complete separate original frozen model; copied tip outputs never deployed',promotion=False))
    for p in sources:
        dest=OUT/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    # Audit a varied deterministic training-only sample before running optimization.
    sample=simple_df[simple_df.study_id.isin(spos)].copy()
    lengths=ann.groupby('study_id').polyline_length_px.sum();sample['length']=sample.study_id.map(lengths)
    chosen=[]
    for label in ['CVC - Abnormal','CVC - Borderline','CVC - Normal']:
        sub=sample[sample[label].eq(1)].sort_values('length')
        for q in [0,.33,.66,1]:
            if len(sub):chosen.append(sub.iloc[round(q*(len(sub)-1))].study_id)
    chosen=list(dict.fromkeys(chosen));chosen+= [s for s in spos if s not in chosen][:12-len(chosen)]
    review=[];(OUT/'inspection').mkdir();font=None
    for page in range(3):
        sheet=Image.new('RGB',(1200,4*520),'#101722');draw=ImageDraw.Draw(sheet)
        for row,sid in enumerate(chosen[page*4:page*4+4]):
            with Image.open(IMAGES_DIR/f'{sid}.jpg') as f:im=f.convert('RGB')
            w,h=im.size;poly=ann[ann.study_id.eq(sid)].iloc[0].polyline
            poly=json.loads(poly) if isinstance(poly,str) else poly
            overlay=im.copy();d=ImageDraw.Draw(overlay);d.line([tuple(p) for p in poly],fill='#00e6ff',width=max(3,round(w/600)))
            for label,img,x in [('RAW',im,0),('ANNOTATION (not prediction)',overlay,600)]:
                img.thumbnail((590,475));sheet.paste(img,(x+(590-img.width)//2,row*520+40))
                draw.text((x+8,row*520+8),f'{page*4+row+1:02d} {label} ...{sid[-12:]}',fill='white')
            r=sample[sample.study_id.eq(sid)].iloc[0]
            review.append(dict(sample_id=page*4+row+1,study_id=sid,patient_id=r.patient_id,width=w,height=h,annotation_points=len(poly),length_px=r['length'],placement_labels=';'.join(c for c in ['CVC - Normal','CVC - Borderline','CVC - Abnormal'] if r[c]),review_status='pending_visual_engineering_inspection'))
        sheet.save(OUT/'inspection'/f'sample_{page+1}.jpg',quality=92)
    pd.DataFrame(review).to_csv(OUT/'inspection/sample_registry.csv',index=False)
    hashes={};cache=OUT/'cache';cache.mkdir();tf=LetterboxTransform(target_size=(768,768),clahe_enabled=True)
    for i,sid in enumerate(ids):
        path=IMAGES_DIR/f'{sid}.jpg';hashes[sid]=sha(path)
        with Image.open(path) as im:im=np.array(im.convert('L'))
        h,w=im.shape;im,meta=tf.apply_image(im);packed=[];rs=ann[ann.study_id.eq(sid)]
        assert len(rs) or (inv.loc[sid,'is_study_level_negative'] and not inv.loc[sid,'has_cvc'])
        for poly in rs.polyline:
            poly=json.loads(poly) if isinstance(poly,str) else poly
            assert len(poly)>=2 and np.isfinite(np.asarray(poly)).all()
            mask=tf.apply_mask(rasterize_study_polylines([poly],(h,w),thickness=10),meta)>0
            assert mask.any();packed.append(np.packbits(mask))
        v=np.zeros((768,768),np.uint8);v[meta.pad_top:meta.pad_top+meta.new_h,meta.pad_left:meta.pad_left+meta.new_w]=1
        np.savez_compressed(cache/f'{sid}.npz',image=im,instances=np.array(packed,dtype=np.uint8).reshape(-1,768*768//8),valid=v)
        if (i+1)%50==0:print('CACHE',i+1,len(ids),flush=True);save(OUT/'status.json',dict(status='preparing',cached=i+1,total=len(ids)))
    save(OUT/'image_hashes.json',hashes)
    ledgers={}
    for name,order in [('mixed_data',mixed),('simple_first',ordered)]:
        ledger=[]
        for i in range(250):
            ss=order[2*i:2*i+2]+negatives[2*i:2*i+2]
            ledger.append(dict(study_ids=ss,augmentation_parameters=[augpos[s] for s in ss[:2]]+augneg[2*i:2*i+2],source_image_hashes=[hashes[s] for s in ss]))
        save(OUT/f'{name}_ledger.json',ledger);ledgers[name]=ledger
    def events(ledger):return Counter((s,json.dumps(a,sort_keys=True)) for b in ledger for s,a in zip(b['study_ids'],b['augmentation_parameters']))
    assert events(ledgers['mixed_data'])==events(ledgers['simple_first'])
    assert all(a['study_ids'][2:]==b['study_ids'][2:] for a,b in zip(*ledgers.values()))
    assert all(set(b['study_ids'][:2])<=set(simple) for b in ledgers['simple_first'][:125])
    inv.loc[ids].assign(is_simple=lambda d:d.index.isin(simple)).to_csv(OUT/'training_cohort.csv')
    save(OUT/'parity.json',dict(identical_image_exposure_multiset=True,identical_image_augmentation_multiset=True,identical_negative_slots=True,study_patient_duplicate_validation_overlap=0))
    save(OUT/'status.json',dict(status='ready_for_inspection',unique_training_images=len(ids)))

def run(smoke=False):
    assert (OUT/'inspection/visual_review.json').exists(),'Inspect sample before training'
    hashes=json.loads((OUT/'source_hashes.json').read_text());assert all(sha(p)==h for p,h in hashes.items())
    assert all(sha(p)==h for p,h in json.loads((OUT/'frozen_hashes.json').read_text()).items())
    dest=OUT/('smoke' if smoke else 'run');dest.mkdir(exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(0);cache=DiskCache(OUT/'cache')
    ann=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');ann=ann[ann.device_type.eq('CVC')];results={}
    for name in ['mixed_data','simple_first']:
        ledger=json.loads((OUT/f'{name}_ledger.json').read_text());ledger=ledger[:2] if smoke else ledger
        m=train(name,True,dest,ledger,cache)
        if not smoke:
            save(dest/'status.json',dict(status='evaluating',arm=name));results[name]=native_eval(m,dest,name,ann)
            fixed=load().requires_grad_(False);x=tensor(next(iter(cache.values()))[0])
            with torch.inference_mode():assert torch.equal(SplitInference(m,fixed)(x)['tip_logits'],fixed(x)['tip_logits'])
            save(dest/name/'deployment.json',dict(status='not_promoted',tip_model=str(SOURCE),tip_sha256=sha(SOURCE),tip_output_bitwise_equal=True));del fixed
            save(dest/'comparison.json',results)
        del m;gc.collect();torch.cuda.empty_cache()
    if not smoke:
        b,e=results['mixed_data'],results['simple_first']
        crit=dict(dice_improves=e['macro_dice']>b['macro_dice'],precision_preserved=e['precision']>=b['precision'],coverage_preserved=e['centerline_coverage_10px']>=b['centerline_coverage_10px']-.005,misses_not_increased=e['missed_paths']<=b['missed_paths'],negative_area_preserved=e['negative_false_area_pct']<=b['negative_false_area_pct'],negative_pixels_preserved=e['negative_total_false_pixels']<=b['negative_total_false_pixels'])
        save(dest/'decision.json',dict(criteria=crit,passed=all(crit.values()),promoted=False))
    assert all(sha(p)==h for p,h in json.loads((OUT/'frozen_hashes.json').read_text()).items())
    assert all(sha(p)==h for p,h in hashes.items())
    save(dest/'status.json',dict(status='complete',smoke=smoke,updates_per_arm=2 if smoke else 250,original_checkpoints_unchanged=True,source_inputs_unchanged=True))
    if not smoke:save(OUT/'status.json',dict(status='complete',updates_per_arm=250))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','run']);a=p.parse_args()
    try:
        if a.mode=='prepare':prepare()
        else:run(a.mode=='smoke')
    except Exception:
        if OUT.exists():save(OUT/'failure.json',dict(mode=a.mode,traceback=traceback.format_exc()))
        raise
