"""Matched recall-loss pilot. Writes only to a fresh run directory; no label promotion."""
import argparse, hashlib, json, random, sys, time, traceback, gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.amp import GradScaler,autocast
import torchvision.transforms.functional as TF
from experiments.hrnet_recovery_v3.contracts import study_tip_supervision
from experiments.hrnet_tip_improvement_v2.src.run_continuation_pilot import TipWeightedBCELoss, evaluate_model_on_benchmark, seed_everything, IMAGES_DIR
from experiments.hrnet_tip_pilot_v1.src.models.hrnet_joint_tip import HRNetJointTipNet
from experiments.hrnet_tip_pilot_v1.src.geometry import CoordinateTransform,render_multitip_gaussian_heatmap
from experiments.architecture_comparison_v2.scripts.train_runner import NativeValidator
from src.data.transforms import LetterboxTransform
from src.data.rasterize import rasterize_study_polylines
V1=ROOT/'experiments/hrnet_tip_pilot_v1'
SOURCE=V1/'runs/seed_42/t1_joint_hrnet/checkpoints/final_model.pt'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8');t.replace(p)
def recall_loss(logits,instances):
    """Equal weight per visible annotated catheter, independent of line length."""
    probs=logits.float().sigmoid().reshape(-1)
    terms=[1-probs[indices].mean() for indices in instances if indices.numel()]
    return torch.stack(terms).mean() if terms else logits.float().sum()*0

def prepare(out,updates):
    ledger_path=V1/'frozen_reference/configs/batch_ledger.jsonl'
    ledger=[json.loads(x) for x in ledger_path.read_text().splitlines() if x.strip()][:updates]
    assert len(ledger)==updates and all(len(x['study_ids'])==4 for x in ledger)
    save(out/'batch_ledger.json',ledger)
    sids=sorted({s for x in ledger for s in x['study_ids']})
    splits=json.loads((ROOT/'data/splits.json').read_text())['partitions']
    val=set(splits['dev_pilot_val']['study_ids']);train=set(splits['dev_pilot_train']['study_ids'])
    assert set(sids)<=train and not(set(sids)&val)
    inv=pd.read_parquet(ROOT/'data/study_inventory.parquet').set_index('StudyInstanceUID')
    for col in ['PatientID','duplicate_group_id']:
        assert not inv.loc[sids,col].isna().any()
        assert not(set(inv.loc[sids,col])&set(inv.loc[list(val),col]))
    ann=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');ann=ann[ann.device_type=='CVC']
    reg=pd.read_parquet(V1/'manifests/instance_target_registry.parquet')
    tf=LetterboxTransform(target_size=(768,768),clahe_enabled=True)
    cache={};audit=[];image_hashes={}
    for i,sid in enumerate(sids):
        image_path=IMAGES_DIR/f'{sid}.jpg';image_hashes[sid]=sha(image_path)
        with Image.open(image_path) as im:image=np.array(im.convert('L'))
        h,w=image.shape; rows=ann[ann.study_id==sid]
        negative=bool(inv.loc[sid,'is_study_level_negative']) and not bool(inv.loc[sid,'has_cvc'])
        assert len(rows)>0 or negative, f'Unknown segmentation supervision: {sid}'
        records=[]
        for r in reg[reg.study_id==sid].to_dict('records'):
            r['x']=r['tip_x_native'];r['y']=r['tip_y_native'];records.append(r)
        sup=study_tip_supervision(records,rows.catheter_instance_id.tolist(),confirmed_negative=negative)
        image,meta=tf.apply_image(image)
        instances=[]
        for p in rows.polyline:
            p=json.loads(p) if isinstance(p,str) else p
            assert len(p)>=2 and np.isfinite(np.asarray(p)).all()
            native=rasterize_study_polylines([p],(h,w),thickness=10)
            mask=(tf.apply_mask(native,meta)>0).astype(np.uint8)
            assert mask.any(),f'Empty annotated path: {sid}'
            instances.append(np.packbits(mask))
        valid=np.zeros((768,768),np.uint8)
        valid[meta.pad_top:meta.pad_top+meta.new_h,meta.pad_left:meta.pad_left+meta.new_w]=1
        coord=CoordinateTransform.from_native_dims(w,h,target_size=768,stride=4)
        hm,hv=render_multitip_gaussian_heatmap(sup['targets'] if sup['enabled'] else [],coord,sigma_hm=1.5)
        cache[sid]=(image,instances,valid,hm,hv,sup['enabled'])
        audit.append(dict(study_id=sid,negative=negative,instances=len(instances),tip_enabled=sup['enabled'],source=sup['source']))
        if (i+1)%100==0:print('CACHE',i+1,len(sids),flush=True)
    # Verify source images against the already frozen batch ledger, not just filenames.
    for entry in ledger:
        assert all(image_hashes[s]==h for s,h in zip(entry['study_ids'],entry['source_image_hashes']))
    save(out/'supervision_audit.json',audit);save(out/'image_hashes.json',image_hashes)
    print('SUPERVISION',pd.DataFrame(audit).groupby(['source','tip_enabled']).size().to_dict(),flush=True)
    return ledger,cache

def augment(cached,a):
    image,packed,valid,hm,hv,supervised=cached
    masks=[np.unpackbits(m,count=768*768).reshape(768,768).astype(np.uint8) for m in packed]
    def affine(x,interpolation,size):
        if a['hflip']:x=TF.hflip(x)
        return TF.affine(x,angle=a['angle'],translate=[a['translate_x_pct']*size,a['translate_y_pct']*size],scale=a['scale'],shear=0,interpolation=interpolation,fill=0)
    image=affine(Image.fromarray(image),TF.InterpolationMode.BILINEAR,768)
    image=TF.adjust_contrast(TF.adjust_brightness(image,a['brightness']),a['contrast'])
    masks=[np.array(affine(Image.fromarray(m),TF.InterpolationMode.NEAREST,768))>0 for m in masks]
    valid=np.array(affine(Image.fromarray(valid),TF.InterpolationMode.NEAREST,768))>0
    union=np.logical_or.reduce(masks) if masks else np.zeros((768,768),bool)
    indices=[torch.from_numpy(np.flatnonzero(m&valid)) for m in masks]
    hm=affine(hm.clone(),TF.InterpolationMode.BILINEAR,192)
    hv=affine(hv.clone(),TF.InterpolationMode.NEAREST,192)
    mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
    return ((TF.to_tensor(image).repeat(3,1,1)-mean)/std,torch.tensor(union,dtype=torch.float32)[None],torch.tensor(valid,dtype=torch.float32)[None],indices,hm,hv,supervised)

def train(arm,weight,out,ledger,cache,device):
    seed_everything(42)
    model=HRNetJointTipNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(SOURCE,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    model.train()
    # Freeze running statistics for microbatch 1 in both arms; affine BN remains trainable.
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=len(ledger),eta_min=1e-6)
    scaler=GradScaler('cuda',init_scale=1024);criterion=TipWeightedBCELoss()
    history=[];start=time.time();attempts=0
    folder=out/arm;folder.mkdir()
    for step,entry in enumerate(ledger,1):
        batch=[augment(cache[s],a) for s,a in zip(entry['study_ids'],entry['augmentation_parameters'])]
        npix=sum(float(b[2].sum()) for b in batch);npos=sum(bool(b[3]) for b in batch);nsup=sum(b[-1] for b in batch)
        for retry in range(8):
            attempts+=1;optimizer.zero_grad(set_to_none=True);totals=np.zeros(4)
            # Reuse exact transformed tensors on AMP retry; BN buffers do not mutate.
            for x,y,v,indices,hm,hv,sup in batch:
                x,y,v,hm,hv=[t.unsqueeze(0).to(device) for t in [x,y,v,hm,hv]]
                indices=[t.to(device) for t in indices]
                with autocast('cuda'):
                    pred=model(x);logits=pred['seg_logits'].float();p=logits.sigmoid()
                    bce=nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor([10.],device=device),reduction='none')
                    seg=(bce*v).sum()/npix
                    if indices:seg=seg+(1-(2*(p*y*v).sum()+1e-6)/((p*v+y*v).sum()+1e-6))/max(npos,1)
                    tip=criterion(pred['tip_logits'],hm,hv,has_positive_tips=bool(indices))[0]/nsup if sup else logits.sum()*0
                    recall=recall_loss(logits,indices)/max(npos,1)
                    loss=seg+tip+weight*recall
                if not torch.isfinite(loss):raise FloatingPointError(f'{arm} step {step}: nonfinite loss')
                totals+=np.array([loss.item(),seg.item(),tip.item(),recall.item()])
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if finite:nn.utils.clip_grad_norm_(model.parameters(),10.)
            scaler.step(optimizer);scaler.update()
            if finite:break
        else:raise FloatingPointError('AMP retry limit reached')
        scheduler.step()
        row=dict(update=step,attempts=attempts,loss=totals.tolist(),seconds=time.time()-start,amp_scale=scaler.get_scale(),peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
        history.append(row)
        if step%25==0 or step==len(ledger):
            save(folder/'history.json',history);save(out/'status.json',dict(status='training',arm=arm,update=step,total_updates=len(ledger)))
            print(arm,row,flush=True)
        if step%50==0 or step==len(ledger):
            path=folder/'last.pt';temporary=folder/'last.tmp'
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),update=step,source_sha256=sha(SOURCE)),temporary);temporary.replace(path)
    return model

def evaluate(model,out,name,device):
    print('NATIVE EVALUATION',name,flush=True)
    native=NativeValidator(target_size=768,device=device).evaluate_model(model)
    save(out/f'{name}_native.json',native)
    manifest=json.loads((V1/'manifests/development_benchmark_manifest.json').read_text())['studies']
    restored=pd.read_csv(ROOT/'experiments/hrnet_recovery_v3/benchmark_provenance_restored.csv')
    tips=evaluate_model_on_benchmark(model,device,restored,manifest,use_offset_head=False);tips.pop('seg_dice',None)
    complete=[s for s in manifest if s['is_negative'] or s['has_all_tips_audited']]
    ids={s['study_id'] for s in complete};registry=restored[restored.study_id.isin(ids)]
    subset=evaluate_model_on_benchmark(model,device,registry,complete,use_offset_head=False);subset.pop('seg_dice',None)
    for k in ['single_cvc_success','multi_cvc_success']:subset.pop(k,None)
    subset['target_count']=len(registry)
    result=dict(native=native['summary'],mixed_provisional_tips=tips,complete_provisional_tips=subset)
    save(out/f'{name}_metrics.json',result);print(name,result,flush=True)
    return result

def main(args):
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=False)
    frozen={str(p):sha(p) for p in [SOURCE,ROOT/'experiments/hrnet_tip_improvement_v2/checkpoints/control_final_model.pt',ROOT/'experiments/hrnet_tip_improvement_v2/checkpoints/experiment_final_model.pt']}
    save(out/'frozen_hashes.json',frozen)
    save(out/'protocol.json',dict(updates=args.updates,seed=42,arms={'control':0.,'instance_recall':.25},effective_batch=4,microbatch=1,learning_rate=5e-5,bn_running_stats='frozen',tip_policy='complete provisional studies only; weak targets masked',primary='missed catheter rate and native centerline coverage at 10px',decision='challenger must reduce missed paths vs control and preserve baseline coverage within 1pp; report precision and false-area tradeoffs; no winner from Dice alone',tip_decoder='quadratic, confidence 0.4, native NMS 20px, cap 10; existing helper',source_code_sha256=sha(__file__)))
    torch.set_num_threads(4);assert torch.cuda.is_available();device=torch.device('cuda')
    try:
        save(out/'status.json',dict(status='preparing'))
        ledger,cache=prepare(out,args.updates)
        results={}
        if not args.smoke:
            base=HRNetJointTipNet(pretrained=False).to(device);base.load_state_dict(torch.load(SOURCE,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
            results['baseline']=evaluate(base,out,'baseline',device);del base;gc.collect();torch.cuda.empty_cache()
        for name,weight in [('control',0.),('instance_recall',.25)]:
            model=train(name,weight,out,ledger,cache,device)
            results[name]=evaluate(model,out,name,device) if not args.smoke else {'smoke_forward_backward_completed':True}
            save(out/'comparison.json',results);del model;gc.collect();torch.cuda.empty_cache()
        assert all(sha(p)==h for p,h in frozen.items())
        save(out/'status.json',dict(status='complete',source_checkpoints_unchanged=True,updates_per_arm=args.updates))
    except Exception:
        save(out/'status.json',dict(status='failed',traceback=traceback.format_exc(),source_checkpoints_unchanged=all(sha(p)==h for p,h in frozen.items())))
        raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=250);parser.add_argument('--out',required=True);parser.add_argument('--smoke',action='store_true')
    main(parser.parse_args())
