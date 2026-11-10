"""Render every path missed by baseline or feature model, from saved outputs only."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.hrnet_recovery_v3.run_recall_pilot import IMAGES_DIR
from src.data.transforms import LetterboxTransform
out=ROOT/'experiments/hrnet_segmentation_features_v1/comparison'
df=pd.read_csv(out/'path_transitions.csv');ann=pd.read_parquet(ROOT/'data/annotation_inventory.parquet')
folder=out/'overlays';folder.mkdir(exist_ok=True);tf=LetterboxTransform(target_size=(768,768),clahe_enabled=False)
selected=df[df.baseline_missed|df.feature_adaptation_missed];records=[]
for i,r in enumerate(selected.to_dict('records'),1):
    sid=r['study_id'];iid=r['catheter_instance_id'];rows=ann[(ann.study_id==sid)&(ann.device_type=='CVC')]
    target=rows[rows.catheter_instance_id==iid];assert len(target)==1
    poly=np.array(json.loads(target.iloc[0].polyline))
    with Image.open(IMAGES_DIR/f'{sid}.jpg') as im:raw=np.array(im.convert('L'))
    h,w=raw.shape;meta=tf.compute_params(h,w)
    fig,axes=plt.subplots(1,4,figsize=(16,6))
    for ax in axes:ax.imshow(raw,cmap='gray',vmin=0,vmax=255)
    for rec in rows.to_dict('records'):
        p=np.array(json.loads(rec['polyline']));axes[0].plot(p[:,0],p[:,1],color='yellow' if rec['catheter_instance_id']==iid else 'lime',lw=1)
    axes[0].set_title('Reference: selected path in yellow')
    for j,name in enumerate(['baseline','head_control','feature_adaptation'],1):
        p=np.load(out/f'{name}_probabilities'/f'{sid}.npz')['prob']
        p=cv2.resize(p[meta.pad_top:meta.pad_top+meta.new_h,meta.pad_left:meta.pad_left+meta.new_w],(w,h))
        rgba=np.zeros((h,w,4),np.float32);rgba[p>=.5]=[1,.05,.45,.65];axes[j].imshow(rgba)
        axes[j].set_title(f"{name}\nCoverage {r[name+'_coverage10']:.1%}; longest gap {r[name+'_longest_gap_native_px']:.0f}px")
    for ax in axes:
        ax.set_xlim(max(0,poly[:,0].min()-150),min(w,poly[:,0].max()+150));ax.set_ylim(min(h,poly[:,1].max()+150),max(0,poly[:,1].min()-150));ax.axis('off')
    fig.suptitle(f"{r['transition']} | {sid} | instance {iid}",fontsize=9,y=.99);fig.tight_layout(rect=(0,0,1,.91))
    dest=folder/f'path_{i:02d}.png';fig.savefig(dest,dpi=110);plt.close(fig)
    records.append(dict(study_id=sid,catheter_instance_id=iid,transition=r['transition'],panel=str(dest)))
pd.DataFrame(records).to_csv(folder/'index.csv',index=False)
print('Saved',len(records),'panels',flush=True)
