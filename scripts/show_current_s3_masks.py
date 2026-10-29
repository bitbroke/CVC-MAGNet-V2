"""Render actual cached S3 predictions; no training or image synthesis."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,pandas as pd
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.transforms import LetterboxTransform
from src.data.rasterize import rasterize_study_polylines
parser=argparse.ArgumentParser()
parser.add_argument('--report-tag',default='S3_experiment_final')
parser.add_argument('--run-name',default='s3_experiment_hard_neg')
parser.add_argument('--output-name',default='current_s3_masks_20261017')
parser.add_argument('--display-name',default='S3 final')
args=parser.parse_args()
report=json.loads((ROOT/f'reports/native_eval_{args.report_tag}.json').read_text())
ckpt=ROOT/'runs'/args.run_name/'checkpoints/final_segmentation_v3.pt'
sha=hashlib.sha256(ckpt.read_bytes()).hexdigest()
assert sha==report['checkpoint_sha256']
records=json.loads((ROOT/f'reports/native_eval_{args.report_tag}_per_study.json').read_text())
lookup={r['study_id']:r for r in records}
panel=json.loads((ROOT/'reports/frozen_50_study_audit_panel_ids.json').read_text())['study_ids']
ann=pd.read_parquet(ROOT/'data/annotation_inventory.parquet')
raw=Path('C:/Users/M S I/Downloads/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification/train')
out=ROOT/'outputs'/args.output_name;out.mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
for index in [49,21,41]:
    sid=panel[index-1];src=raw/f'{sid}.jpg';im=cv2.imread(str(src),0);h,w=im.shape
    cache=list((ROOT/'cache/mask_probabilities').glob(f'native_{sha[:12]}_v3_768_*{sid}.npz'))
    assert cache,(index,sid)
    with np.load(cache[-1]) as z:
        assert str(z['ckpt_sha256'])==sha
        assert hashlib.sha256(src.read_bytes()).hexdigest().startswith(str(z['img_sha256']))
        pm=z['prob_map']
    m=LetterboxTransform((768,768)).compute_params(h,w)
    prob=cv2.resize(pm[m.pad_top:m.pad_top+m.new_h,m.pad_left:m.pad_left+m.new_w],(w,h))
    pred=prob>=.5
    rows=ann[(ann.study_id==sid)&(ann.device_type=='CVC')]
    polys=[json.loads(p) if isinstance(p,str) else p for p in rows.polyline]
    gt=rasterize_study_polylines(polys,(h,w),10)>0
    r=lookup[sid]['threshold_metrics']['tau_50']
    assert int(pred.sum())==r['pred_area_native_px']
    rgb=np.repeat(im[:,:,None],3,axis=2);g=rgb.copy();p=rgb.copy()
    g[gt]=(g[gt]*.25+np.array([40,255,70])*.75).astype('uint8')
    p[pred]=(p[pred]*.4+np.array([255,70,180])*.6).astype('uint8')
    canvas=Image.new('RGB',(1800,690),(18,22,28));d=ImageDraw.Draw(canvas)
    title=f'Case {index} | {args.display_name} | threshold 0.50 | CVC annotations: {len(polys)}'
    d.text((10,5),title,font=font,fill='white')
    for j,(arr,label) in enumerate([(rgb,'Original X-ray'),(g,'Annotation (green)'),(p,'Model prediction (pink)')]):
        d.text((j*600+10,40),label,font=font,fill='white')
        tile=Image.fromarray(arr);tile.thumbnail((590,600))
        canvas.paste(tile,(j*600+(600-tile.width)//2,80))
    canvas.save(out/f'case_{index}.jpg',quality=95)
    cv2.imwrite(str(out/f'case_{index}_native_binary_mask.png'),pred.astype('uint8')*255)
    print(index,'Dice',r['binary_dice'],'predicted pixels',int(pred.sum()),str(out/f'case_{index}.jpg'))
(out/'provenance.json').write_text(json.dumps({'checkpoint':str(ckpt),'sha256':sha,'threshold':.5,'cases':[49,21,41],'selection':'Same cases illustrated during preceding frozen-versus-repaired review; not a random performance sample.'},indent=2))
print(json.dumps(report['summary_by_threshold']['tau_50'],indent=2))
