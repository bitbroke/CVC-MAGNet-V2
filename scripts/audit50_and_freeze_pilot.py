"""Prepare faithful annotation/prediction review sheets and snapshot reference models."""
import sys,json,hashlib,shutil
from pathlib import Path
import cv2,numpy as np,pandas as pd
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.transforms import LetterboxTransform
from src.data.rasterize import rasterize_study_polylines
cv2.setNumThreads(1)
OUT=ROOT/'outputs/audit50_repaired_pilot_20261016';OUT.mkdir(exist_ok=True)
ARCHIVE=OUT/'frozen_references';ARCHIVE.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
refs=[]
for run,file in [('pilot_experiment_expanded','final_coarse_dual_head.pt'),('s1_pretrained_segmentation','final_segmentation_v3.pt'),('s2_higher_resolution_768','final_segmentation_v3.pt')]:
    p=ROOT/'runs'/run/'checkpoints'/file;dest=ARCHIVE/f'{run}_{file}'
    if not dest.exists():shutil.copy2(p,dest)
    assert sha(p)==sha(dest)
    refs.append({'source':str(p),'snapshot':str(dest),'sha256':sha(p)})
(OUT/'frozen_reference_hashes.json').write_text(json.dumps(refs,indent=2))
panel=json.loads((ROOT/'reports/frozen_50_study_audit_panel_ids.json').read_text())['study_ids'];assert len(panel)==len(set(panel))==50
report=json.loads((ROOT/'reports/native_eval_S2_higher_res_final.json').read_text());digest=report['checkpoint_sha256']
inv=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');groups={str(k):v for k,v in inv.groupby('study_id')}
studies=pd.read_parquet(ROOT/'data/study_inventory.parquet').set_index('StudyInstanceUID')
raw=Path('C:/Users/M S I/Downloads/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification/train')
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
rows=[]
for i,sid in enumerate(panel):
    source=raw/f'{sid}.jpg';im=cv2.imread(str(source),0);h,w=im.shape
    cachefiles=list((ROOT/'cache/mask_probabilities').glob(f'native_{digest[:12]}_v3_768_*{sid}.npz'));assert cachefiles
    cache=max(cachefiles,key=lambda p:p.stat().st_mtime)
    with np.load(cache) as z:
        assert str(z['ckpt_sha256'])==digest;pm=z['prob_map']
        if 'img_sha256' in z and str(z['img_sha256']):assert sha(source).startswith(str(z['img_sha256']))
    meta=LetterboxTransform((768,768)).compute_params(h,w)
    pm=cv2.resize(pm[meta.pad_top:meta.pad_top+meta.new_h,meta.pad_left:meta.pad_left+meta.new_w],(w,h))
    pred=pm>=.5
    rec=groups.get(sid,pd.DataFrame(columns=inv.columns))
    polys=[];other=[]
    for _,r in rec.iterrows():
        poly=json.loads(r.polyline) if isinstance(r.polyline,str) else r.polyline
        (polys if r.device_type=='CVC' else other).append(poly)
    gt=rasterize_study_polylines(polys,(h,w),10)>0
    other_gt=rasterize_study_polylines(other,(h,w),10)>0
    fp=pred&~gt;tp=int((pred&gt).sum());fpcount=int(fp.sum())
    if gt.any():
        dist=cv2.distanceTransform((~gt).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        near=int((fp&(dist<=15)).sum());far=int((fp&(dist>30)).sum())
    else:near=0;far=fpcount
    recstat={'index':i+1,'study_id':sid,'cvc_instances':len(polys),'other_annotated_lines':len(other),'abnormal':int(studies.loc[sid,'CVC - Abnormal']),
        'dice':2*tp/max(int(pred.sum()+gt.sum()),1),'precision':tp/max(int(pred.sum()),1),'recall':tp/max(int(gt.sum()),1),
        'predicted_pixels':int(pred.sum()),'target_pixels':int(gt.sum()),'false_pixels':fpcount,
        'false_within15px_of_target':near,'false_farther30px_from_target':far,
        'false_overlapping_other_annotated_paths':int((fp&other_gt).sum()),'image_path':str(source)}
    rows.append(recstat)
    rgb=np.repeat(im[:,:,None],3,axis=2);ann=rgb.copy();ann[gt]=(rgb[gt]*.25+np.array([40,255,70])*.75).astype('uint8')
    overlay=rgb.copy();overlay[pred]=(rgb[pred]*.4+np.array([255,70,180])*.6).astype('uint8')
    case=Image.new('RGB',(1800,665),(18,22,28));draw=ImageDraw.Draw(case)
    title=f"{i+1:02d} | CVC={len(polys)} other={len(other)} abnormal={recstat['abnormal']} | Dice={recstat['dice']:.3f} P={recstat['precision']:.2f} R={recstat['recall']:.2f}"
    draw.text((12,4),title,font=font,fill='white')
    for j,(arr,label) in enumerate([(rgb,'Original'),(ann,'CVC annotation (green)'),(overlay,'S2 prediction >=0.50 (pink)')]):
        draw.text((j*600+12,34),label,font=small,fill='white');image=Image.fromarray(arr);image.thumbnail((590,590));case.paste(image,(j*600+(600-image.width)//2,65))
    case.save(OUT/f'case_{i+1:02d}.jpg',quality=95)
    if i%5==0:sheet=Image.new('RGB',(1800,665*5),(18,22,28))
    sheet.paste(case,(0,(i%5)*665))
    if i%5==4:sheet.save(OUT/f'sheet_{i//5+1:02d}.jpg',quality=95)
    if (i+1)%5==0:print(f'Rendered {i+1}/50',flush=True)
pd.DataFrame(rows).to_csv(OUT/'quantitative_audit.csv',index=False)
(OUT/'quantitative_audit.json').write_text(json.dumps(rows,indent=2))
print('COMPLETE',OUT,flush=True)
