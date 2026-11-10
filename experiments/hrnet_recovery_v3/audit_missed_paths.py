"""Inference-only, per-instance audit of all paths missed by any frozen arm."""
import json,sys,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import cv2,torch
from PIL import Image
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.hrnet_recovery_v3.run_recall_pilot import SOURCE,IMAGES_DIR,sha,save
from experiments.hrnet_tip_pilot_v1.src.models.hrnet_joint_tip import HRNetJointTipNet
from experiments.architecture_comparison_v2.scripts.train_runner import sample_polyline_points
from src.data.transforms import LetterboxTransform
RUN=ROOT/'experiments/hrnet_constrained_recall_v1/comparison'
OUT=ROOT/'experiments/hrnet_constrained_recall_v1/path_audit'

def distance_to_mask(prob,pts,threshold):
    h,w=prob.shape
    x0=max(0,int(np.floor(pts[:,0].min()))-15);x1=min(w,int(np.ceil(pts[:,0].max()))+15)
    y0=max(0,int(np.floor(pts[:,1].min()))-15);y1=min(h,int(np.ceil(pts[:,1].max()))+15)
    yy,xx=np.nonzero(prob[y0:y1,x0:x1]>=threshold)
    if not len(xx):return np.full(len(pts),np.inf)
    return cKDTree(np.column_stack([xx+x0,yy+y0])).query(pts,k=1)[0]

def main():
    OUT.mkdir(exist_ok=False);(OUT/'panels').mkdir();(OUT/'probabilities').mkdir();(OUT/'point_traces').mkdir()
    torch.set_num_threads(4);cv2.setNumThreads(0)
    checkpoints={'baseline':SOURCE,'control':RUN/'control/last.pt','constrained':RUN/'constrained/last.pt'}
    hashes={k:sha(p) for k,p in checkpoints.items()}
    ann=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');ann=ann[ann.device_type=='CVC']
    n=ann.groupby('study_id').size();candidates=set();previous={}
    for name in checkpoints:
        previous[name]={r['study_id']:r for r in json.loads((RUN/f'{name}_native.json').read_text())['per_study']}
        # If a study mean is >= (n-1+.3)/n, no individual path can have coverage <.3.
        # This conservatively selects EVERY potentially missed instance, including multi-CVC.
        candidates.update(sid for sid,r in previous[name].items() if r['is_positive'] and r['centerline_coverage_10px']<(n[sid]-1+.3)/n[sid]+1e-8)
    candidates=sorted(candidates)
    save(OUT/'provenance.json',dict(checkpoints={k:dict(path=str(p),sha256=hashes[k]) for k,p in checkpoints.items()},selected_studies=candidates,selection='Conservative mathematical bound from per-study mean and instance count; includes every potentially missed path. No training or threshold selection.',source_sha256=sha(__file__)))
    tf=LetterboxTransform(target_size=(768,768),clahe_enabled=True);device=torch.device('cuda')
    for name,path in checkpoints.items():
        model=HRNetJointTipNet(pretrained=False).to(device).eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
        with torch.inference_mode():
            for sid in candidates:
                with Image.open(IMAGES_DIR/f'{sid}.jpg') as im:raw=np.array(im.convert('L'))
                lb,meta=tf.apply_image(raw);x=torch.from_numpy(lb).float()[None,None].repeat(1,3,1,1).to(device)/255
                x=(x-torch.tensor([.485,.456,.406],device=device)[None,:,None,None])/torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
                prob=model(x)['seg_logits'].float().sigmoid()[0,0].cpu().numpy()
                np.savez_compressed(OUT/'probabilities'/f'{sid}_{name}.npz',prob=prob)
        print('INFERENCE COMPLETE',name,len(candidates),flush=True);del model;gc.collect();torch.cuda.empty_cache()
    records=[];panel_count=0
    for sid in candidates:
        with Image.open(IMAGES_DIR/f'{sid}.jpg') as im:raw=np.array(im.convert('L'))
        h,w=raw.shape;_,meta=tf.apply_image(raw);group=ann[ann.study_id==sid]
        probs={}
        for name in checkpoints:
            p=np.load(OUT/'probabilities'/f'{sid}_{name}.npz')['prob']
            probs[name]=cv2.resize(p[meta.pad_top:meta.pad_top+meta.new_h,meta.pad_left:meta.pad_left+meta.new_w],(w,h),interpolation=cv2.INTER_LINEAR)
        per_study={name:[] for name in checkpoints}
        for row in group.to_dict('records'):
            poly=np.asarray(json.loads(row['polyline']),np.float32);pts=sample_polyline_points(poly.tolist())
            pts=pts[(pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)]
            assert len(pts)>0
            record=dict(study_id=sid,catheter_instance_id=row['catheter_instance_id'],instance_key=row['instance_key'],cvc_count=len(group),native_w=w,native_h=h,model_scale=meta.new_w/w,polyline_length_px=row['polyline_length_px'],review_status=row['review_status'])
            traces={'x':pts[:,0],'y':pts[:,1]}
            for name,p in probs.items():
                vals=cv2.remap(p,pts[:,0].reshape(-1,1),pts[:,1].reshape(-1,1),cv2.INTER_LINEAR).ravel()
                traces[name+'_probability_on_polyline']=vals
                record[name+'_path_probability_median']=float(np.median(vals));record[name+'_path_probability_max']=float(vals.max())
                for threshold in [.2,.3,.4,.5,.6]:
                    d=distance_to_mask(p,pts,threshold);cov=float(np.mean(d<=10))
                    record[name+f'_coverage_t{threshold:.1f}']=cov
                    if threshold==.5:
                        per_study[name].append(cov);traces[name+'_distance_to_foreground_roi']=d
                        record[name+'_missed']=cov<.3
            b,c,e=[record[k+'_missed'] for k in checkpoints]
            record['transition_vs_baseline']='persistently_missed' if b and e else 'lost_by_constrained' if e else 'recovered_by_constrained' if b else 'retained'
            record['control_transition']='missed' if c else 'covered'
            records.append(record)
            if not(b or c or e):continue
            panel_count+=1;key=f"case_{panel_count:02d}_{row['catheter_instance_id']}";record['panel_key']=key
            pd.DataFrame(traces).to_csv(OUT/'point_traces'/f'{key}.csv',index=False)
            x0=max(0,int(poly[:,0].min())-120);x1=min(w,int(poly[:,0].max())+120)
            y0=max(0,int(poly[:,1].min())-120);y1=min(h,int(poly[:,1].max())+120)
            fig,axes=plt.subplots(2,4,figsize=(17,10))
            for ax in axes[0]:ax.imshow(raw,cmap='gray',vmin=0,vmax=255)
            for ax in axes[1]:ax.imshow(raw,cmap='gray',vmin=0,vmax=255);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
            for ax in axes[:,0]:
                for other in group.to_dict('records'):
                    q=np.asarray(json.loads(other['polyline']))
                    color='yellow' if other['catheter_instance_id']==row['catheter_instance_id'] else 'lime'
                    ax.plot(q[:,0],q[:,1],color=color,lw=1.5)
            axes[0,0].set_title('Reference paths: yellow = audited instance')
            axes[1,0].set_title('Target path crop (not a clinical tip review)')
            for j,(name,p) in enumerate(probs.items(),1):
                rgba=np.zeros((h,w,4),np.float32);rgba[p>=.5]=[1,.05,.45,.65]
                for ax in axes[:,j]:ax.imshow(rgba)
                axes[0,j].set_title(f"{name}: coverage {record[name+'_coverage_t0.5']:.1%}")
                axes[1,j].set_title(f"Mask at 0.50; coverage at 0.30: {record[name+'_coverage_t0.3']:.1%}")
            for ax in axes.flat:ax.axis('off')
            fig.suptitle(f"{key} | {record['transition_vs_baseline']} | CVC instance {row['catheter_instance_id']} | {sid}",fontsize=10)
            fig.tight_layout();fig.savefig(OUT/'panels'/f'{key}.png',dpi=125);plt.close(fig)
        for name in checkpoints:
            np.testing.assert_allclose(np.mean(per_study[name]),previous[name][sid]['centerline_coverage_10px'],atol=1e-7,rtol=0)
        print('STUDY AUDITED',sid,flush=True)
    df=pd.DataFrame(records);df.to_csv(OUT/'instance_audit.csv',index=False)
    failed=df[df[[k+'_missed' for k in checkpoints]].any(axis=1)]
    failed.to_csv(OUT/'missed_union.csv',index=False)
    counts={k:int(df[k+'_missed'].sum()) for k in checkpoints};assert counts=={'baseline':8,'control':7,'constrained':10},counts
    assert all(sha(p)==hashes[k] for k,p in checkpoints.items())
    save(OUT/'summary.json',dict(status='complete',training_started=False,checkpoints_unchanged=True,studies=len(candidates),instances=len(df),union_missed_instances=len(failed),missed_counts=counts,constrained_transitions=df.transition_vs_baseline.value_counts().to_dict(),panels=panel_count))
    print('COMPLETE',counts,flush=True)
if __name__=='__main__':main()
