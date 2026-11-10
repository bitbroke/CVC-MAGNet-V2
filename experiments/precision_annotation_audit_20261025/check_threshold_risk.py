"""Diagnostic only: fixed threshold sweep on the four previously recovered paths."""
import sys,json
from pathlib import Path
import cv2,numpy as np,pandas as pd,yaml
from PIL import Image
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data.transforms import LetterboxTransform
OUT=Path(__file__).resolve().parent/'results';run=ROOT/'experiments/simple_first_v1/comparison/run'
ann=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');cfg=yaml.safe_load((ROOT/'configs/pilot.yaml').read_text());images=Path(cfg['data']['raw_dir'])/'train';tf=LetterboxTransform((768,768),clahe_enabled=False)
def sample(poly):
 pts=np.asarray(poly,dtype=np.float32);sampled=[]
 for i in range(len(pts)-1):
  p1,p2=pts[i:i+2];d=np.linalg.norm(p2-p1)
  if d<1e-4:continue
  for x in np.linspace(0,1,max(1,int(np.ceil(d/2))),endpoint=i==len(pts)-2):sampled.append(p1+x*(p2-p1))
 return np.asarray(sampled,np.float32)
rows=[]
for r in pd.read_csv(run/'previously_recovered_paths.csv').to_dict('records'):
 sid=r['study_id'];a=ann[ann.study_id.eq(sid)&ann.catheter_instance_id.eq(r['catheter_instance_id'])&ann.device_type.eq('CVC')];assert len(a)==1
 poly=a.iloc[0].polyline;pts=sample(json.loads(poly) if isinstance(poly,str) else poly)
 with Image.open(images/f'{sid}.jpg') as im:w,h=im.size
 pts=pts[(pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)];m=tf.compute_params(h,w)
 p=np.load(run/'simple_first_probabilities'/f'{sid}.npz')['prob'];p=cv2.resize(p[m.pad_top:m.pad_top+m.new_h,m.pad_left:m.pad_left+m.new_w],(w,h))
 for t in [.5,.6,.7]:
  yy,xx=np.nonzero(p>=t);d=cKDTree(np.column_stack([xx,yy])).query(pts)[0] if len(xx) else np.full(len(pts),np.inf)
  cov=float(np.mean(d<=10));rows.append(dict(study_id=sid,instance_id=r['catheter_instance_id'],threshold=t,coverage10=cov,missed=cov<.3))
df=pd.DataFrame(rows);df.to_csv(OUT/'recovered_path_threshold_risk.csv',index=False);print(df.to_string(index=False))
