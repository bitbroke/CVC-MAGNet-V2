"""Post-run panels selected using only baseline performance, never challenger gains."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import cv2,torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.hrnet_recovery_v3.run_recall_pilot import SOURCE,IMAGES_DIR,sha
from experiments.hrnet_tip_pilot_v1.src.models.hrnet_joint_tip import HRNetJointTipNet
from src.data.transforms import LetterboxTransform
OUT=ROOT/'experiments/hrnet_missed_catheter_pilot_v1/comparison'
def main():
    assert json.loads((OUT/'status.json').read_text())['status']=='complete'
    torch.set_num_threads(4)
    folder=OUT/'overlays';folder.mkdir(exist_ok=True)
    rows=json.loads((OUT/'baseline_native.json').read_text())['per_study']
    pos=sorted([r for r in rows if r['is_positive']],key=lambda r:r['centerline_coverage_10px'])
    inv=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');inv=inv[inv.device_type=='CVC']
    counts=inv.groupby('study_id').size()
    cases=[('low_coverage_1',pos[0]['study_id']),('low_coverage_2',pos[1]['study_id']),('multi_cvc',next(r['study_id'] for r in pos if counts.get(r['study_id'],0)>1)),('negative',next(r['study_id'] for r in rows if not r['is_positive']))]
    checkpoints={'baseline':SOURCE,'control':OUT/'control/last.pt','instance_recall':OUT/'instance_recall/last.pt'}
    hashes={k:sha(p) for k,p in checkpoints.items()};predictions={}
    tf=LetterboxTransform(target_size=(768,768),clahe_enabled=True)
    for name,path in checkpoints.items():
        model=HRNetJointTipNet(pretrained=False).eval()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
        for category,sid in cases:
            with Image.open(IMAGES_DIR/f'{sid}.jpg') as im:raw=np.array(im.convert('L'))
            lb,m=tf.apply_image(raw);x=torch.from_numpy(lb).float()[None,None].repeat(1,3,1,1)/255
            x=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
            with torch.inference_mode():p=model(x)['seg_logits'].float().sigmoid()[0,0].numpy()
            h,w=raw.shape;p=cv2.resize(p[m.pad_top:m.pad_top+m.new_h,m.pad_left:m.pad_left+m.new_w],(w,h),interpolation=cv2.INTER_LINEAR)
            predictions[(name,category)]=p>=.5
            assert cv2.imwrite(str(folder/f'{category}_{name}_mask.png'),(p>=.5).astype(np.uint8)*255)
    for category,sid in cases:
        with Image.open(IMAGES_DIR/f'{sid}.jpg') as im:raw=np.array(im.convert('L'))
        fig,axes=plt.subplots(1,4,figsize=(16,5))
        for ax in axes:ax.imshow(raw,cmap='gray',vmin=0,vmax=255);ax.axis('off')
        axes[0].set_title('Reference CVC polylines')
        for poly in inv[inv.study_id==sid].polyline:
            pts=np.array(json.loads(poly));axes[0].plot(pts[:,0],pts[:,1],color='lime',linewidth=1)
        for ax,name in zip(axes[1:],checkpoints):
            mask=predictions[(name,category)];rgba=np.zeros((*mask.shape,4),np.float32);rgba[mask]=[1,.1,.45,.65]
            ax.imshow(rgba);ax.set_title(name)
        fig.suptitle(category+' | selected by baseline only | native threshold 0.50');fig.tight_layout();fig.savefig(folder/f'{category}.png',dpi=140);plt.close(fig)
    assert all(sha(p)==hashes[k] for k,p in checkpoints.items())
    (folder/'provenance.json').write_text(json.dumps(dict(cases=cases,checkpoint_hashes=hashes,inference='CPU FP32',selection='two lowest baseline coverage studies, lowest baseline coverage multi-CVC, first listed negative; illustrative not representative'),indent=2))
    print('Saved four comparison panels and twelve binary masks:',folder)
if __name__=='__main__':main()
