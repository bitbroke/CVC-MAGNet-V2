"""Segmentation-head-only matched constrained recall pilot. All earlier models immutable."""
import argparse,copy,gc,json,sys,time,traceback,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.amp import autocast,GradScaler
from experiments.hrnet_recovery_v3.run_recall_pilot import (
    SOURCE,V1,sha,save,prepare,augment,recall_loss,evaluate,seed_everything,HRNetJointTipNet)

def constraint_losses(logits,reference_logits,target,valid):
    """FP penalty outside a 7px model-grid corridor; retention only on GT foreground."""
    p=logits.float().sigmoid();q=reference_logits.detach().float().sigmoid()
    corridor=nn.functional.max_pool2d(target.float(),7,stride=1,padding=3)
    background=(corridor<.5)&(valid>.5)
    vals=p[background]
    fp=vals.topk(max(1,int(np.ceil(vals.numel()*.01)))).values.mean() if vals.numel() else p.sum()*0
    foreground=(target>.5)&(valid>.5)
    retention=torch.relu(q[foreground]-p[foreground]-.02).mean() if foreground.any() else p.sum()*0
    return fp,retention

def frozen_state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        if not k.startswith('seg_head.'):
            h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load_model(device):
    m=HRNetJointTipNet(pretrained=False).to(device)
    m.load_state_dict(torch.load(SOURCE,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    for name,param in m.named_parameters():param.requires_grad_(name.startswith('seg_head.'))
    m.eval() # Encoder and tip modules stay in evaluation mode throughout training.
    m.seg_head.train()
    return m

def train(arm,constrained,out,ledger,cache,device):
    seed_everything(42);model=load_model(device)
    initial_frozen_hash=frozen_state_hash(model)
    teacher_head=copy.deepcopy(model.seg_head).eval().requires_grad_(False)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=5e-5,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=len(ledger),eta_min=1e-6)
    scaler=GradScaler('cuda',init_scale=1024)
    folder=out/arm;folder.mkdir();history=[];start=time.time();attempts=0
    torch.cuda.reset_peak_memory_stats()
    for step,entry in enumerate(ledger,1):
        batch=[augment(cache[s],a) for s,a in zip(entry['study_ids'],entry['augmentation_parameters'])]
        npix=sum(float(b[2].sum()) for b in batch);npos=sum(any(t.numel()>0 for t in b[3]) for b in batch)
        for retry in range(8):
            attempts+=1;optimizer.zero_grad(set_to_none=True);totals=np.zeros(5)
            for x,y,v,indices,hm,hv,sup in batch:
                x,y,v=[t.unsqueeze(0).to(device) for t in [x,y,v]]
                indices=[t.to(device) for t in indices if t.numel()]
                with autocast('cuda'):
                    pred=model(x);logits=pred['seg_logits'].float();p=logits.sigmoid()
                    with torch.no_grad():reference=teacher_head(pred['fused_features'].detach())
                    bce=nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor([10.],device=device),reduction='none')
                    seg=(bce*v).sum()/npix
                    if indices:seg+=(1-(2*(p*y*v).sum()+1e-6)/((p*v+y*v).sum()+1e-6))/max(npos,1)
                    recall=recall_loss(logits,indices)/max(npos,1)
                    fp,retain=constraint_losses(logits,reference,y,v)
                    fp=fp/len(batch);retain=retain/max(npos,1)
                    loss=seg+.25*recall+(.25*fp+.25*retain if constrained else 0.)
                if not torch.isfinite(loss):raise FloatingPointError(f'{arm} {step} nonfinite loss')
                totals+=np.array([loss.item(),seg.item(),recall.item(),fp.item(),retain.item()])
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite=all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
            if finite:
                norm=nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
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
            tmp=folder/'last.tmp'
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),scaler_state_dict=scaler.state_dict(),update=step,source_sha256=sha(SOURCE)),tmp);tmp.replace(folder/'last.pt')
    assert initial_frozen_hash==frozen_state_hash(model),'Frozen feature/tip tensors changed'
    # FP32 equality, rather than a tolerance or rounded metric, protects the tip path.
    x=augment(cache[ledger[0]['study_ids'][0]],ledger[0]['augmentation_parameters'][0])[0][None].to(device)
    model.eval();reference_model=load_model(device).eval()
    with torch.no_grad():
        tip_a=model(x)['tip_logits'];tip_b=reference_model(x)['tip_logits']
    assert torch.equal(tip_a,tip_b),'Tip logits changed despite the frozen path'
    save(folder/'freeze_verification.json',dict(non_segmentation_state_sha256=initial_frozen_hash,state_unchanged=True,fp32_tip_logits_bitwise_equal=True,trainable_parameters=sum(p.numel() for p in parameters)))
    del teacher_head,reference_model;gc.collect();torch.cuda.empty_cache()
    return model

def main(args):
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    old=[SOURCE,V1/'runs/seed_43/t1_joint_hrnet/checkpoints/final_model.pt',ROOT/'experiments/hrnet_tip_improvement_v2/checkpoints/control_final_model.pt',ROOT/'experiments/hrnet_tip_improvement_v2/checkpoints/experiment_final_model.pt']
    old += [ROOT/f'experiments/hrnet_missed_catheter_pilot_v1/comparison/{a}/last.pt' for a in ['control','instance_recall']]
    frozen={str(p):sha(p) for p in old};save(out/'frozen_hashes.json',frozen)
    save(out/'protocol.json',dict(seed=42,updates=args.updates,model='HRNetJointTipNet',trainable='seg_head only',source=str(SOURCE),learning_rate=5e-5,effective_batch=4,microbatch=1,
        arms={'control':{'recall':.25,'background':0,'retention':0},'constrained':{'recall':.25,'background':.25,'retention':.25}},
        background='top 1% predicted probabilities outside 7x7 dilated GT; valid image only',retention='relu(reference probability - prediction - .02) on GT foreground',
        tip_policy='no tip training; exact shared-feature and tip-head freeze; inherited weak-label limitations remain',
        progression='strict: fewer misses than baseline and control; Dice and mean-study precision >= baseline; negative false area <= baseline; coverage >= baseline; identical tip metrics. No automatic promotion.',
        benchmark='193 studies /248 paths; mixed provisional 41 tips plus complete-provisional subset',code_sha256=sha(__file__)))
    sources=[Path(__file__),ROOT/'experiments/hrnet_recovery_v3/run_recall_pilot.py',ROOT/'experiments/hrnet_recovery_v3/contracts.py',ROOT/'data/splits.json',ROOT/'data/annotation_inventory.parquet',ROOT/'data/study_inventory.parquet',V1/'manifests/instance_target_registry.parquet',ROOT/'experiments/hrnet_recovery_v3/benchmark_provenance_restored.csv',V1/'src/models/hrnet_joint_tip.py',ROOT/'src/data/transforms.py',V1/'src/geometry.py',ROOT/'experiments/architecture_comparison_v2/scripts/train_runner.py',ROOT/'experiments/hrnet_tip_improvement_v2/src/run_continuation_pilot.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources};save(out/'source_hashes.json',hashes)
    for p in sources:
        dest=out/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    torch.set_num_threads(4);assert torch.cuda.is_available();device=torch.device('cuda')
    try:
        save(out/'status.json',dict(status='preparing'))
        ledger,cache=prepare(out,args.updates);results={}
        if not args.smoke:
            base=load_model(device).eval();results['baseline']=evaluate(base,out,'baseline',device);del base;gc.collect();torch.cuda.empty_cache()
        for name,flag in [('control',False),('constrained',True)]:
            model=train(name,flag,out,ledger,cache,device)
            if args.smoke:results[name]={'smoke_passed':True}
            else:
                save(out/'status.json',dict(status='evaluating',arm=name))
                results[name]=evaluate(model,out,name,device)
                assert results[name]['mixed_provisional_tips']==results['baseline']['mixed_provisional_tips']
                assert results[name]['complete_provisional_tips']==results['baseline']['complete_provisional_tips']
            save(out/'comparison.json',results);del model;gc.collect();torch.cuda.empty_cache()
        assert all(sha(p)==h for p,h in frozen.items())
        assert all(sha(ROOT/p)==h for p,h in hashes.items())
        save(out/'status.json',dict(status='complete',source_checkpoints_unchanged=True,updates_per_arm=args.updates,tip_path_preserved=True))
    except Exception:
        save(out/'status.json',dict(status='failed',traceback=traceback.format_exc(),source_checkpoints_unchanged=all(sha(p)==h for p,h in frozen.items())))
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=250);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
