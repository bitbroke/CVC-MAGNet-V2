"""Fixed-rule model comparison overlays; does not alter labels or model outputs."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,cv2
from PIL import Image,ImageDraw
from src.data.transforms import LetterboxTransform
import yaml
OUT=Path(__file__).resolve().parent/'comparison/run';DEST=OUT/'negative_overlays';DEST.mkdir(exist_ok=False)
cfg=yaml.safe_load((ROOT/'configs/pilot.yaml').read_text());images=Path(cfg['data']['raw_dir'])/'train'
ann=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');ann=ann[ann.device_type.eq('CVC')]
d=pd.read_csv(OUT/'paired_study_dice.csv').sort_values('difference')
negative=pd.DataFrame(json.loads((OUT/'simple_first_native.json').read_text())['per_study']); ids=negative[~negative.is_positive].sort_values('false_positive_pixels',ascending=False).head(2).study_id.tolist()
tf=LetterboxTransform(target_size=(768,768),clahe_enabled=False)
for i,sid in enumerate(ids,1):
    with Image.open(images/f'{sid}.jpg') as f:im=np.array(f.convert('L'))
    im,meta=tf.apply_image(im);base=np.repeat(im[:,:,None],3,axis=2)
    page=Image.new('RGB',(768*4,810),'#101722');draw=ImageDraw.Draw(page)
    for k,arm in enumerate(['annotation','original','mixed_data','simple_first']):
        pix=base.copy()
        if arm!='annotation':
            folder=ROOT/'experiments/hrnet_segmentation_features_v1/comparison/baseline_probabilities' if arm=='original' else OUT/f'{arm}_probabilities'
            prob=np.load(folder/f'{sid}.npz')['prob'];mask=prob>=.5
            valid=np.zeros(mask.shape,bool);valid[meta.pad_top:meta.pad_top+meta.new_h,meta.pad_left:meta.pad_left+meta.new_w]=True;mask &= valid
            pix[mask]=(.55*pix[mask]+.45*np.array([255,140,0])).astype(np.uint8)
        for poly in ann[ann.study_id.eq(sid)].polyline:
            poly=json.loads(poly) if isinstance(poly,str) else poly
            # Use the same affine metadata as the production transform.
            xy=np.asarray(poly,float);xy[:,0]=xy[:,0]*meta.scale+meta.pad_left;xy[:,1]=xy[:,1]*meta.scale+meta.pad_top
            cv2.polylines(pix,[np.round(xy).astype(np.int32)],False,(0,220,255),1)
        page.paste(Image.fromarray(pix),(768*k,42));draw.text((768*k+8,10),f'{arm}: cyan annotation; orange prediction | ...{sid[-12:]}',fill='white')
    page.save(DEST/f'comparison_{i:02d}.jpg',quality=90)
pd.DataFrame({'study_id':ids,'selection':'two negative studies with largest simple-first false pixel counts'}).to_csv(DEST/'registry.csv',index=False)
print('Overlays',len(ids))

