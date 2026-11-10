"""Read-only saved-prediction diagnostic. No inference or training."""
import sys,json,hashlib
from pathlib import Path
from collections import Counter
import cv2,numpy as np,pandas as pd,yaml
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data.transforms import LetterboxTransform
from src.data.rasterize import rasterize_study_polylines
OUT=Path(__file__).resolve().parent;DEST=OUT/'results';DEST.mkdir(exist_ok=True)
cv2.setNumThreads(2)
cfg=yaml.safe_load((ROOT/'configs/pilot.yaml').read_text());raw=Path(cfg['data']['raw_dir'])
ann=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');source=pd.read_csv(raw/'train_annotations.csv')
ids=json.loads((ROOT/'data/splits.json').read_text())['partitions']['dev_pilot_val']['study_ids']
run=ROOT/'experiments/simple_first_v1/comparison/run'
folders={'original':ROOT/'experiments/hrnet_segmentation_features_v1/comparison/baseline_probabilities','mixed':run/'mixed_data_probabilities','simple_first':run/'simple_first_probabilities'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
models=list(ROOT.rglob('*.pt'));before={str(p.relative_to(ROOT)):sha(p) for p in models}
tf=LetterboxTransform((768,768),clahe_enabled=False)
def poly(x):return json.loads(x) if isinstance(x,str) else x
def sig(x):return tuple(map(tuple,np.asarray(poly(x),float)))
integrity=[];rows=[]
for i,sid in enumerate(ids):
 a=ann[ann.study_id.eq(sid)];c=a[a.device_type.eq('CVC')];s=source[source.StudyInstanceUID.eq(sid)&source.label.str.startswith('CVC')]
 integrity.append({'study_id':sid,'inventory_cvc':len(c),'source_cvc':len(s),'exact_polyline_multiset_equal':Counter(map(sig,c.polyline))==Counter(map(sig,s.data))})
 with Image.open(raw/'train'/f'{sid}.jpg') as im:w,h=im.size
 meta=tf.compute_params(h,w);gt=rasterize_study_polylines([poly(x) for x in c.polyline],(h,w),10).astype(bool)
 other=rasterize_study_polylines([poly(x) for x in a[~a.device_type.eq('CVC')].polyline],(h,w),10).astype(bool)
 dist=cv2.distanceTransform((~gt).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if gt.any() else None
 for arm,folder in folders.items():
  p=np.load(folder/f'{sid}.npz')['prob'];p=cv2.resize(p[meta.pad_top:meta.pad_top+meta.new_h,meta.pad_left:meta.pad_left+meta.new_w],(w,h),interpolation=cv2.INTER_LINEAR)
  pred=p>=.5;tp=int((pred&gt).sum());fp=pred&~gt
  rows.append(dict(study_id=sid,arm=arm,positive=bool(gt.any()),tp=tp,fp=int(fp.sum()),gt=int(gt.sum()),pred=int(pred.sum()),precision=tp/max(1,int(pred.sum())),recall=tp/max(1,int(gt.sum())),fp_within5=int((fp&(dist<=5)).sum()) if dist is not None else 0,fp_5to15=int((fp&(dist>5)&(dist<=15)).sum()) if dist is not None else 0,fp_beyond15=int((fp&(dist>15)).sum()) if dist is not None else int(fp.sum()),fp_on_other_annotation=int((fp&other).sum())))
 if i%20==0:print('Audited',i+1,len(ids),flush=True)
df=pd.DataFrame(rows);df.to_csv(DEST/'pixel_diagnostics.csv',index=False);pd.DataFrame(integrity).to_csv(DEST/'source_annotation_reconciliation.csv',index=False)
summary={}
for arm,g in df.groupby('arm'):
 pos=g[g.positive];neg=g[~g.positive];f=int(pos.fp.sum())
 summary[arm]={'positive_macro_precision':float(pos.precision.mean()),'positive_pooled_precision':float(pos.tp.sum()/pos.pred.sum()),'positive_macro_recall':float(pos.recall.mean()),'positive_fp_pixels':f,'positive_fp_within5_fraction':float(pos.fp_within5.sum()/f),'positive_fp_5to15_fraction':float(pos.fp_5to15.sum()/f),'positive_fp_beyond15_fraction':float(pos.fp_beyond15.sum()/f),'positive_fp_other_annotation_fraction':float(pos.fp_on_other_annotation.sum()/f),'negative_fp_pixels':int(neg.fp.sum()),'negative_other_annotation_overlap':int(neg.fp_on_other_annotation.sum())}
(DEST/'summary.json').write_text(json.dumps(summary,indent=2))
# Native-resolution crops selected by the largest false-mask concentration; no labels changed.
selected=pd.read_csv(run/'overlays/registry.csv').study_id.tolist()+pd.read_csv(run/'negative_overlays/registry.csv').study_id.tolist()
for n,sid in enumerate(selected):
 im=cv2.imread(str(raw/'train'/f'{sid}.jpg'),0);h,w=im.shape;meta=tf.compute_params(h,w)
 c=ann[ann.study_id.eq(sid)&ann.device_type.eq('CVC')];gt=rasterize_study_polylines([poly(x) for x in c.polyline],(h,w),10).astype(bool)
 probs={}
 for arm,folder in folders.items():
  p=np.load(folder/f'{sid}.npz')['prob'];probs[arm]=cv2.resize(p[meta.pad_top:meta.pad_top+meta.new_h,meta.pad_left:meta.pad_left+meta.new_w],(w,h))
 fp=((probs['simple_first']>=.5)&~gt).astype('float32');density=cv2.boxFilter(fp,-1,(512,512),normalize=False);cy,cx=np.unravel_index(density.argmax(),density.shape)
 x=max(0,min(w-640,cx-320));y=max(0,min(h-640,cy-320));page=Image.new('RGB',(2560,680));draw=ImageDraw.Draw(page)
 for j,arm in enumerate(['raw','original','mixed','simple_first']):
  pix=np.repeat(im[y:y+640,x:x+640,None],3,axis=2)
  if arm!='raw':
   mask=probs[arm][y:y+640,x:x+640]>=.5;pix[mask]=(.5*pix[mask]+.5*np.array([255,130,0])).astype('uint8')
   gg=gt[y:y+640,x:x+640];contours,_=cv2.findContours(gg.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(pix,contours,-1,(0,255,255),1)
  page.paste(Image.fromarray(pix),(640*j,40));draw.text((640*j+5,5),f'{arm} | native crop ({x},{y}) | ...{sid[-12:]}',fill='white')
 page.save(DEST/f'native_{n+1:02d}.jpg',quality=93)
assert all(sha(ROOT/p)==v for p,v in before.items())
(DEST/'preservation.json').write_text(json.dumps({'checkpoint_count':len(before),'all_checkpoint_hashes_unchanged':True,'checkpoint_hashes':before},indent=2))
print(json.dumps(summary,indent=2));print('DONE',flush=True)
