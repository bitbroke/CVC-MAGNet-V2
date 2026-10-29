"""Render cached S2 predictions faithfully; no training or generated imagery."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.transforms import LetterboxTransform
report=json.loads((ROOT/'reports/native_eval_S2_higher_res_final.json').read_text())
raw=json.loads((ROOT/'reports/native_eval_S2_higher_res_final_per_study.json').read_text())
records=raw['per_study_results'] if isinstance(raw,dict) and 'per_study_results' in raw else raw
if isinstance(records,dict):
    raise ValueError(list(records))
out=ROOT/'outputs/current_S2_mask_examples';out.mkdir(exist_ok=True)
sha=report['checkpoint_sha256']
assert hashlib.sha256(Path(report['checkpoint']).read_bytes()).hexdigest()==sha
images=Path('C:/Users/M S I/Downloads/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification/train')
chosen=[];used=set()
for label,predicate in [('Single CVC',lambda r:r['is_positive'] and r['num_cvc_instances']==1),('Multiple CVCs',lambda r:r['is_positive'] and r['num_cvc_instances']>1),('Abnormal-labeled study',lambda r:r['is_positive'] and r['cvc_abnormal']==1),('CVC-negative study',lambda r:not r['is_positive'])]:
    r=next(r for r in records if r['study_id'] not in used and predicate(r));chosen.append((label,r));used.add(r['study_id'])
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
sheet=Image.new('RGB',(1500,4*560+100),(18,22,29));d=ImageDraw.Draw(sheet)
d.text((20,10),'S2 final | 768 input | native threshold >= 0.50 | predicted masks only',font=font,fill='white')
for j,t in enumerate(['Original X-ray','Binary CVC prediction','Prediction overlay (cyan)']):d.text((j*500+20,52),t,font=font,fill='white')
manifest=[]
for i,(label,r) in enumerate(chosen):
    sid=r['study_id'];source=images/f'{sid}.jpg';im=cv2.imread(str(source),0);h,w=im.shape
    caches=list((ROOT/'cache/mask_probabilities').glob(f'native_{sha[:12]}_v3_768_*{sid}.npz'))
    assert caches,sid
    cache=max(caches,key=lambda p:p.stat().st_mtime)
    with np.load(cache) as z:
        assert str(z['ckpt_sha256'])==sha
        prob=z['prob_map'];assert prob.shape==(768,768)
    meta=LetterboxTransform((768,768)).compute_params(h,w)
    p=cv2.resize(prob[meta.pad_top:meta.pad_top+meta.new_h,meta.pad_left:meta.pad_left+meta.new_w],(w,h),interpolation=cv2.INTER_LINEAR)
    mask=p>=.5
    tm=r['threshold_metrics']['tau_50'];assert int(mask.sum())==tm['pred_area_native_px']
    assert cv2.imwrite(str(out/f'{i+1}_mask_native.png'),mask.astype('uint8')*255)
    rgb=np.repeat(im[:,:,None],3,axis=2);overlay=rgb.copy();overlay[mask]=(rgb[mask]*.55+np.array([0,210,255])*.45).astype('uint8')
    Image.fromarray(overlay).save(out/f'{i+1}_overlay_native.jpg',quality=95)
    y=100+i*560
    d.text((20,y),f'{i+1}. {label} | '+(f"Dice {tm['binary_dice']:.3f}" if r['is_positive'] else f"False mask area {100*mask.mean():.2f}%"),font=font,fill='white')
    for j,array in enumerate([rgb,np.repeat((mask*255).astype('uint8')[:,:,None],3,axis=2),overlay]):
        img=Image.fromarray(array);img.thumbnail((480,490));sheet.paste(img,(j*500+(500-img.width)//2,y+40))
    manifest.append({'study_id':sid,'category':label,'cache':str(cache),'threshold':.5,'checkpoint_sha256':sha,'predicted_pixels':int(mask.sum())})
sheet.save(out/'S2_current_masks.png')
(out/'manifest.json').write_text(json.dumps({'selection':'First distinct validation study in each category; not best-case selection','examples':manifest},indent=2))
print(json.dumps({'overview':str(out/'S2_current_masks.png'),'examples':len(manifest),'metrics':report['summary_by_threshold']['tau_50']}))
