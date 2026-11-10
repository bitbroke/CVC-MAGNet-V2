"""Matched boundary-precision experiment. Existing checkpoints are read-only."""
import sys,json,gc,time,traceback,shutil,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,cv2,torch
from torch import nn
from torch.amp import autocast,GradScaler
from experiments.hrnet_recovery_v3.run_recall_pilot import SOURCE,sha,save,prepare,augment,recall_loss,seed_everything,HRNetJointTipNet
from experiments.hrnet_recovery_v3.run_feature_pilot import configure,immutable_hash,native_eval,tensor,SplitInference,load as load_tip
PARENT=ROOT/'experiments/hrnet_segmentation_features_v1/comparison'
INITIAL=PARENT/'feature_adaptation/last.pt'

def load():
    m=HRNetJointTipNet(pretrained=False).cuda().eval()
    m.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    return m

def ring_mask(target,valid):
    """One model pixel of uncertainty outside GT; supervise only the 1..8px ring."""
    y=target.detach().cpu().numpy().squeeze()>.5
    v=valid.detach().cpu().numpy().squeeze()>.5
    if not y.any():return torch.zeros_like(target)
    distance=cv2.distanceTransform((~y).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return torch.from_numpy(((distance>1)&(distance<=8)&v).astype(np.float32)).reshape(target.shape)

def precision_loss(logits,ring):
    logits=logits.float();ring=ring.to(logits.device).float()
    return (nn.functional.softplus(logits)*ring).sum(dtype=torch.float32)/ring.sum(dtype=torch.float32).clamp_min(1)

def retention_loss(logits,teacher,target,valid):
    p=logits.float().sigmoid();q=teacher.detach().float().sigmoid()
    mask=(target>.5)&(valid>.5)&(q>=.5)
    return torch.relu(q[mask]-p[mask]-.02).mean() if mask.any() else logits.float().sum()*0

def training_diagnostic(out,cache):
    m=load().requires_grad_(False);rows=[]
    with torch.inference_mode():
        for i,(sid,b) in enumerate(cache.items()):
            if not b[1]:continue
            y=np.logical_or.reduce([np.unpackbits(v,count=768*768).reshape(768,768)>0 for v in b[1]])
            prob=m(tensor(b[0]))['seg_logits'].float().sigmoid()[0,0].cpu().numpy()
            pred=(prob>=.5)&(b[2]>.5);fp=pred&~y
            dist=cv2.distanceTransform((~y).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            rows.append(dict(study_id=sid,predicted_pixels=int(pred.sum()),gt_pixels=int(y.sum()),false_pixels=int(fp.sum()),within_1px=int((fp&(dist<=1)).sum()),boundary_1_to_8px=int((fp&(dist>1)&(dist<=8)).sum()),distant_over_8px=int((fp&(dist>8)).sum())))
            if i%100==0:print('TRAIN BOUNDARY AUDIT',i,len(cache),flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'training_boundary_audit.csv',index=False)
    sums=df.drop(columns='study_id').sum().to_dict();sums={k:int(v) for k,v in sums.items()}
    sums['studies']=len(df);sums['coordinate_space']='768 model grid; reference is annotation ribbon, not clinical catheter boundary'
    save(out/'training_boundary_summary.json',sums)
    del m;gc.collect();torch.cuda.empty_cache()

def train(arm,weight,out,ledger,cache):
    seed_everything(42);m=load();params=configure(m,True);frozen=immutable_hash(m,True)
    teacher=load().requires_grad_(False)
    opt=torch.optim.AdamW([dict(params=[p for n,p in m.named_parameters() if p.requires_grad and n.startswith('seg_head.')],lr=1e-5),dict(params=[p for n,p in m.named_parameters() if p.requires_grad and not n.startswith('seg_head.')],lr=2e-6)],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,len(ledger),eta_min=1e-7)
    scaler=GradScaler('cuda',init_scale=1024);folder=out/arm;folder.mkdir();history=[];attempts=0;start=time.time()
    torch.cuda.reset_peak_memory_stats()
    for step,entry in enumerate(ledger,1):
        batch=[augment(cache[s],a) for s,a in zip(entry['study_ids'],entry['augmentation_parameters'])]
        rings=[ring_mask(b[1],b[2]) for b in batch]
        # Frozen teacher uses exactly the same augmented image. Cache for same-batch retries.
        with torch.no_grad(),autocast('cuda'):
            refs=[teacher(b[0][None].cuda())['seg_logits'].detach().float() for b in batch]
        npix=sum(float(b[2].sum()) for b in batch);npos=sum(any(t.numel() for t in b[3]) for b in batch)
        for retry in range(8):
            attempts+=1;opt.zero_grad(set_to_none=True);total=np.zeros(5)
            for b,ring,reference in zip(batch,rings,refs):
                x,y,v,indices,*_=b;x,y,v=[t[None].cuda() for t in (x,y,v)];indices=[t.cuda() for t in indices if t.numel()]
                with autocast('cuda'):
                    logits=m(x)['seg_logits'].float();p=logits.sigmoid()
                    bce=nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor([10.],device='cuda'),reduction='none')
                    seg=(bce*v).sum()/npix
                    if indices:seg+=(1-(2*(p*y*v).sum()+1e-6)/((p*v+y*v).sum()+1e-6))/max(npos,1)
                    rec=recall_loss(logits,indices)/max(npos,1)
                    retain=retention_loss(logits,reference,y,v)/max(npos,1)
                    boundary=precision_loss(logits,ring[None])/max(npos,1)
                    loss=seg+.25*rec+.5*retain+weight*boundary
                if not torch.isfinite(loss):raise FloatingPointError(f'{arm}/{step}: nonfinite loss')
                total+=np.array([loss.item(),seg.item(),rec.item(),retain.item(),boundary.item()]);scaler.scale(loss).backward()
            scaler.unscale_(opt);finite=all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
            if finite:nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
            scaler.step(opt);scaler.update()
            if finite:break
        else:raise FloatingPointError('AMP retry budget exhausted')
        scheduler.step();row=dict(update=step,attempts=attempts,loss=total.tolist(),seconds=time.time()-start,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,amp_scale=scaler.get_scale());history.append(row)
        if step%25==0 or step==len(ledger):
            save(folder/'history.json',history);save(out/'status.json',dict(status='training',arm=arm,**row));print(arm,row,flush=True)
        if step%50==0 or step==len(ledger):
            tmp=folder/'last.tmp';torch.save(dict(model_state_dict=m.state_dict(),optimizer_state_dict=opt.state_dict(),scheduler_state_dict=scheduler.state_dict(),scaler_state_dict=scaler.state_dict(),update=step,source_sha256=sha(INITIAL),deployment='Segmentation only; use complete frozen original tip model separately'),tmp);tmp.replace(folder/'last.pt')
    assert immutable_hash(m,True)==frozen
    save(folder/'freeze_verification.json',dict(tip_head_and_bn_buffers_unchanged=True,frozen_tensor_hash=frozen))
    del teacher;gc.collect();torch.cuda.empty_cache();return m.eval()

def report(out,results):
    histories=[json.loads((out/n/'history.json').read_text()) for n in ['control','boundary']]
    np.testing.assert_allclose(histories[0][0]['loss'][1:],histories[1][0]['loss'][1:],atol=1e-6,rtol=0)
    keys=['study_id','catheter_instance_id'];frames={n:pd.read_csv(out/f'{n}_paths.csv').set_index(keys) for n in results}
    paired=pd.concat({n:f[['coverage10','longest_gap_native_px','missed']] for n,f in frames.items()},axis=1)
    assert len(paired)==248 and not paired.isna().any().any()
    recovered=frames['baseline'].missed&~frames['parent'].missed
    assert int(recovered.sum())==4
    parent,c,e=results['parent'],results['control'],results['boundary']
    preserved=(frames['boundary'].loc[recovered,'coverage10']>=frames['parent'].loc[recovered,'coverage10']-.01)&(frames['boundary'].loc[recovered,'longest_gap_native_px']<=frames['parent'].loc[recovered,'longest_gap_native_px']+10)
    criteria=dict(dice_improves_parent_and_control=e['macro_dice']>max(parent['macro_dice'],c['macro_dice']),precision_improves_parent_and_control=e['precision']>max(parent['precision'],c['precision']),all_four_recoveries_preserved=bool(preserved.all()),no_new_misses_vs_parent=bool((~frames['boundary'].missed|frames['parent'].missed).all()),coverage_within_half_pp=e['centerline_coverage_10px']>=parent['centerline_coverage_10px']-.005,mean_gap_no_more_than_10px_worse=e['mean_longest_gap_px']<=parent['mean_longest_gap_px']+10,negative_mean_area_not_worse=e['negative_false_area_pct']<=parent['negative_false_area_pct'],negative_total_pixels_not_worse=e['negative_total_false_pixels']<=parent['negative_total_false_pixels'])
    paired.columns=['_'.join(c) for c in paired.columns];paired['previously_recovered']=recovered;paired.to_csv(out/'path_transitions.csv')
    paired[recovered].to_csv(out/'recovered_path_checks.csv')
    production=dict(dice_at_least_original=e['macro_dice']>=results['baseline']['macro_dice'],precision_at_least_original=e['precision']>=results['baseline']['precision'])
    save(out/'decision.json',dict(criteria=criteria,experiment_passed=all(criteria.values()),original_quality_checks=production,all_requirements_passed=all(criteria.values()) and all(production.values()),promoted=False,qualification='Repeatedly used development set; annotations provisional, eight negatives. No clinical or independent-test claim.'))
    lines=['# Boundary precision experiment — 2026-10-22','','| Arm | Misses /248 | Coverage10 | Dice | Mean-study precision | Mean negative area % | Total negative FP pixels | Mean longest gap px |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,s in results.items():lines.append(f"| {n} | {s['missed_paths']} | {s['centerline_coverage_10px']:.2%} | {s['macro_dice']:.4f} | {s['precision']:.2%} | {s['negative_false_area_pct']:.5f} | {s['negative_total_false_pixels']} | {s['mean_longest_gap_px']:.1f} |")
    lines+=['',f"Controlled experiment decision: {'PASS' if all(criteria.values()) else 'FAIL'}. No model promoted.",'','Both arms start from the frozen feature-adaptation checkpoint and use identical batches, augmentations, learning rates, successful-update budgets and foreground-retention loss. Only the 0.25 boundary term differs. The boundary target is distance 1 < d <= 8 model pixels outside the annotated mask; one pixel is ignored for alignment uncertainty. This is supervision against rasterized polylines, not independently traced anatomical boundaries.','', 'Recovered-path preservation requires each of the four paths to retain coverage within 1 percentage point and longest gap within 10 native pixels of the parent. Overall coverage may fall at most 0.5 percentage points. These tolerances are predeclared, not exact non-regression guarantees. Tips use the original separate frozen model.']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')

def main(args):
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    frozen={};old=list((ROOT/'experiments').rglob('*.pt'))
    for i,p in enumerate(old,1):
        frozen[str(p)]=sha(p)
        if i%25==0:print('FREEZE HASH',i,len(old),flush=True)
    save(out/'frozen_hashes.json',frozen)
    sourcepaths=[Path(__file__),ROOT/'experiments/hrnet_recovery_v3/run_feature_pilot.py',ROOT/'experiments/hrnet_recovery_v3/run_recall_pilot.py',ROOT/'data/splits.json',ROOT/'data/annotation_inventory.parquet',ROOT/'data/study_inventory.parquet',PARENT/'matched_batch_ledger.json']
    hashes={str(p):sha(p) for p in sourcepaths};save(out/'source_hashes.json',hashes)
    for p in sourcepaths:
        dest=out/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'protocol.json',dict(seed=42,updates=args.updates,initial=str(INITIAL),initial_sha256=sha(INITIAL),effective_batch=4,microbatch=1,resolution=768,head_lr=1e-5,feature_lr=2e-6,bn='frozen',base_loss='BCE posweight10 + Dice + .25 recall + .5 teacher retention on GT where teacher>=.5; retention margin .02',arms={'control':0,'boundary':.25},boundary='softplus(logit) mean outside GT at distance >1 and <=8 model pixels; valid-image only; positive studies only',threshold=.5,tip_model=str(SOURCE),preservation='four old recoveries within 1pp coverage and +10 native px gap individually; no new misses; total coverage within .5pp; mean gap within10px; negative mean area AND total FP not worse',promotion='no automatic promotion; original Dice and precision additionally required'))
    torch.set_num_threads(4);cv2.setNumThreads(0)
    try:
        save(out/'status.json',dict(status='preparing'))
        ledger,cache=prepare(out,args.updates if args.smoke else 1000)
        if not args.smoke:
            ledger=json.loads((PARENT/'matched_batch_ledger.json').read_text())[:args.updates]
            assert len(ledger)==args.updates
            imagehash=json.loads((out/'image_hashes.json').read_text())
            assert all(imagehash[s]==h for b in ledger for s,h in zip(b['study_ids'],b['source_image_hashes']))
        save(out/'matched_batch_ledger.json',ledger)
        ann=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');ann=ann[ann.device_type=='CVC'];results={}
        if not args.smoke:
            training_diagnostic(out,cache)
            prior=json.loads((PARENT/'comparison.json').read_text())
            assert sha(SOURCE)==json.loads((PARENT/'frozen_hashes.json').read_text())[str(SOURCE)]
            for oldname,name in [('baseline','baseline'),('feature_adaptation','parent')]:
                results[name]=prior[oldname];shutil.copy2(PARENT/f'{oldname}_paths.csv',out/f'{name}_paths.csv')
            assert sha(INITIAL)==json.loads((PARENT/'postrun_verification.json').read_text())['feature_adaptation']['checkpoint_sha256']
            # Verify benchmark identity before reusing frozen source reference scores.
            priorhash=json.loads((PARENT/'source_hashes.json').read_text())
            for name in ['splits.json','annotation_inventory.parquet','study_inventory.parquet']:assert sha(ROOT/'data'/name)==priorhash[str(ROOT/'data'/name)]
        for name,weight in [('control',0.),('boundary',.25)]:
            m=train(name,weight,out,ledger,cache)
            if not args.smoke:
                save(out/'status.json',dict(status='evaluating',arm=name));results[name]=native_eval(m,out,name,ann)
                fixed=load_tip().requires_grad_(False);x=tensor(next(iter(cache.values()))[0])
                with torch.inference_mode():assert torch.equal(SplitInference(m,fixed)(x)['tip_logits'],fixed(x)['tip_logits'])
                save(out/name/'deployment.json',dict(status='experimental_not_promoted',segmentation=str(out/name/'last.pt'),tip_model=str(SOURCE),tip_sha256=sha(SOURCE),separate_tip_model_bitwise_check=True))
                del fixed;save(out/'comparison.json',results)
            del m;gc.collect();torch.cuda.empty_cache()
        if not args.smoke:report(out,results)
        assert all(sha(p)==h for p,h in frozen.items());assert all(sha(p)==h for p,h in hashes.items())
        save(out/'status.json',dict(status='complete',updates_per_arm=args.updates,original_checkpoints_unchanged=True,source_inputs_unchanged=True,smoke=args.smoke))
    except Exception:
        save(out/'status.json',dict(status='failed',traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--updates',type=int,default=250);p.add_argument('--smoke',action='store_true');main(p.parse_args())
