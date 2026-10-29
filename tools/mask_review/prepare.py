"""Export every frozen validation prediction for manual inspection; inference only."""
import sys,json,hashlib
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch,yaml
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.coarse_dual import CoarseDualNet
from src.data.transforms import LetterboxTransform
OUT=ROOT/'outputs/predicted_mask_review_expanded'
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'masks').mkdir(exist_ok=True)
ckpt=ROOT/'runs/pilot_experiment_expanded/checkpoints/final_coarse_dual_head.pt'
digest=hashlib.sha256(ckpt.read_bytes()).hexdigest()
cfg=yaml.safe_load((ROOT/'configs/pilot.yaml').read_text())
raw=Path(cfg['data']['raw_dir'])/'train'
split_path=ROOT/'data/splits.json'
sids=json.loads(split_path.read_text())['partitions']['dev_pilot_val']['study_ids']
inv=pd.read_parquet(ROOT/'data/study_inventory.parquet').set_index('StudyInstanceUID')
transform=LetterboxTransform(target_size=(512,512),clahe_enabled=True)
model=CoarseDualNet(in_channels=1,base_channels=32).cuda().eval()
model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=False)['model_state_dict'])
records=[]
with torch.inference_mode():
 for i,sid in enumerate(sids):
  source=raw/f'{sid}.jpg'
  with Image.open(source) as im: image=np.asarray(im)
  assert image.ndim==2
  h,w=image.shape
  inp,meta=transform.apply_image(image)
  x=torch.from_numpy(inp.copy()).float()[None,None].cuda()/255.
  with torch.amp.autocast('cuda'): seg,_=model(x)
  probability=torch.sigmoid(seg).squeeze().cpu().numpy()
  # Match evaluator threshold first, remove letterbox padding, map categorical mask by nearest-neighbor.
  binary=(probability>.30).astype(np.uint8)
  crop=binary[meta.pad_top:meta.pad_top+meta.new_h,meta.pad_left:meta.pad_left+meta.new_w]
  mask=cv2.resize(crop,(w,h),interpolation=cv2.INTER_NEAREST)
  assert mask.shape==image.shape and set(np.unique(mask))<={0,1}
  rgba=np.zeros((h,w,4),np.uint8);rgba[:,:,0]=255;rgba[:,:,1]=180;rgba[:,:,3]=mask*255
  destination=OUT/'masks'/f'{sid}.png'
  assert cv2.imwrite(str(destination),rgba)
  check=cv2.imread(str(destination),cv2.IMREAD_UNCHANGED)
  assert check.shape==(h,w,4) and np.array_equal(check[:,:,3],mask*255)
  labels=[str(c).replace('CVC - ','') for c in ['CVC - Normal','CVC - Borderline','CVC - Abnormal'] if inv.loc[sid,c]==1]
  records.append(dict(study_id=sid,image_path=str(source),mask_path=str(destination),width=w,height=h,labels=labels,foreground_fraction=float(mask.mean()),geometry=meta.to_dict(),mask_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
  if (i+1)%20==0: print(f'Exported and verified {i+1}/{len(sids)}',flush=True)
manifest=dict(model='Expanded dual-head final checkpoint',checkpoint_path=str(ckpt),checkpoint_sha256=digest,split='dev_pilot_val',split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),threshold=.30,mask_type='Model-predicted CVC segmentation; no annotation overlays',mapping='Threshold >0.30 on model grid, remove letterbox padding, nearest-neighbor resize to native dimensions',studies=records)
assert len(records)==len(set(sids))==193
tmp=OUT/'manifest.tmp';tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(OUT/'manifest.json')
print(f'COMPLETE: {len(records)} masks; manifest {OUT / "manifest.json"}',flush=True)
