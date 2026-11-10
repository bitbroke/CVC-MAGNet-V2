"""Isolated segmentation feature adaptation; production tip inference stays separate."""
import argparse, copy, gc, hashlib, json, sys, time, traceback, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np,pandas as pd,torch
from PIL import Image
from torch import nn
from torch.amp import autocast,GradScaler
from experiments.hrnet_recovery_v3.run_recall_pilot import SOURCE,V1,IMAGES_DIR,sha,save,prepare,augment,recall_loss,seed_everything,HRNetJointTipNet,evaluate
from experiments.hrnet_recovery_v3.audit_missed_paths import distance_to_mask
from experiments.architecture_comparison_v2.scripts.train_runner import NativeValidator,sample_polyline_points
from src.data.transforms import LetterboxTransform

def load():
    m=HRNetJointTipNet(pretrained=False).cuda().eval()
    m.load_state_dict(torch.load(SOURCE,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    return m

def tensor(image):
    x=torch.from_numpy(image).float()[None,None].repeat(1,3,1,1).cuda()/255
    return (x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]

def path_stats(prob,poly):
    pts=sample_polyline_points(poly);h,w=prob.shape
    inside=(pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)
    assert inside.any(),'No visible annotated path'
    valid=pts[inside];covered=distance_to_mask(prob,valid,.5)<=10
    # Integrate actual sampled arc lengths, not the approximate 2px sampling interval.
    edges=np.linalg.norm(np.diff(pts,axis=0),axis=1)
    flags=np.zeros(len(pts),bool);flags[inside]=covered
    observable=inside[:-1]&inside[1:]
    absent=observable&~flags[:-1]&~flags[1:]
    longest=run=0.;gaps=0;previous=False
    for length,missing in zip(edges,absent):
        if missing:
            run+=float(length);longest=max(longest,run)
            if not previous:gaps+=1
        else:run=0.
        previous=bool(missing)
    coverage=float(covered.mean())
    return dict(coverage10=coverage,missed=coverage<.3,longest_gap_native_px=longest,
        gap_runs=gaps,visible_length_native_px=float(edges[observable].sum()),
        coverage_at_least_90pct=coverage>=.9,coverage_complete=bool(covered.all()))

def score_study(prob,sid,ann):
    with Image.open(IMAGES_DIR/f'{sid}.jpg') as im:w,h=im.size
    # Use the canonical transform metadata rather than duplicating rounding conventions.
    tf=LetterboxTransform(target_size=(768,768),clahe_enabled=False)
    meta=tf.compute_params(h,w)
    p=cv2.resize(prob[meta.pad_top:meta.pad_top+meta.new_h,meta.pad_left:meta.pad_left+meta.new_w],(w,h),interpolation=cv2.INTER_LINEAR)
    rows=[];group=ann[ann.study_id==sid]
    for r in group.to_dict('records'):
        poly=json.loads(r['polyline']) if isinstance(r['polyline'],str) else r['polyline']
        rows.append(dict(study_id=sid,catheter_instance_id=r['catheter_instance_id'],cvc_count=len(group),**path_stats(p,poly)))
    return rows

def mine(out,cache,ledger,ann):
    model=load();rows=[]
    with torch.inference_mode():
        for i,(sid,cached) in enumerate(cache.items()):
            if not cached[1]:continue
            prob=model(tensor(cached[0]))['seg_logits'].sigmoid()[0,0].cpu().numpy()
            rows.extend(score_study(prob,sid,ann))
            if i%50==0:print('TRAIN MINING',i,len(cache),flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'training_path_inventory.csv',index=False)
    # Include severe gaps and the lowest-coverage quartile; fixed entirely on training data.
    cutoff=float(df.coverage10.quantile(.25))
    hard=df[(df.coverage10<=cutoff)|(df.longest_gap_native_px>=100)]
    hard.to_csv(out/'training_hard_paths.csv',index=False)
    hard_ids=sorted(hard.study_id.unique());assert hard_ids
    partitions=json.loads((ROOT/'data/splits.json').read_text())['partitions']
    assert set(hard_ids)<=set(partitions['dev_pilot_train']['study_ids'])
    assert not set(hard_ids)&set(partitions['dev_pilot_val']['study_ids'])
    rng=np.random.default_rng(42);new=copy.deepcopy(ledger);replacements=0
    for entry in new:
        for i,sid in enumerate(entry['study_ids']):
            if cache[sid][1] and rng.random()<.5:
                chosen=str(rng.choice(hard_ids));entry['study_ids'][i]=chosen
                entry['source_image_hashes'][i]=sha(IMAGES_DIR/f'{chosen}.jpg')
                entry['sampling_category'][i]='training_hard_positive';replacements+=1
    save(out/'matched_batch_ledger.json',new)
    save(out/'mining_summary.json',dict(instances=len(df),hard_instances=len(hard),hard_studies=len(hard_ids),coverage_quartile=cutoff,replacements=replacements,validation_used=False,policy='50% of positive slots replaced from training-only low coverage or >=100px longest-gap studies; negative slots unchanged'))
    del model;gc.collect();torch.cuda.empty_cache();return new

def configure(m,features):
    for name,p in m.named_parameters():p.requires_grad_(not name.startswith('tip_head.') if features else name.startswith('seg_head.'))
    m.eval() # Fixed BN buffers and deterministic module behavior in both arms.
    return [p for p in m.parameters() if p.requires_grad]

def immutable_hash(m,features):
    names={n for n,p in m.named_parameters() if p.requires_grad}
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):
        if k not in names:h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def train(name,features,out,ledger,cache):
    seed_everything(42);m=load();params=configure(m,features);frozen_hash=immutable_hash(m,features)
    groups=[dict(params=[p for n,p in m.named_parameters() if p.requires_grad and n.startswith('seg_head.')],lr=5e-5)]
    if features:groups.append(dict(params=[p for n,p in m.named_parameters() if p.requires_grad and not n.startswith('seg_head.')],lr=1e-5))
    opt=torch.optim.AdamW(groups,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,len(ledger),eta_min=1e-6)
    scaler=GradScaler('cuda',init_scale=1024);folder=out/name;folder.mkdir();history=[];attempts=0;start=time.time()
    save(folder/'trainable_parameters.json',{n:p.numel() for n,p in m.named_parameters() if p.requires_grad})
    torch.cuda.reset_peak_memory_stats()
    for step,entry in enumerate(ledger,1):
        batch=[augment(cache[s],a) for s,a in zip(entry['study_ids'],entry['augmentation_parameters'])]
        npix=sum(float(b[2].sum()) for b in batch);npos=sum(any(t.numel()>0 for t in b[3]) for b in batch)
        for retry in range(8):
            attempts+=1;opt.zero_grad(set_to_none=True);total=np.zeros(3)
            for x,y,v,indices,*_ in batch:
                x,y,v=[t[None].cuda() for t in (x,y,v)];indices=[t.cuda() for t in indices if t.numel()]
                with autocast('cuda'):
                    logits=m(x)['seg_logits'].float();p=logits.sigmoid()
                    bce=nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor([10.],device='cuda'),reduction='none')
                    seg=(bce*v).sum()/npix
                    if indices:seg+=(1-(2*(p*y*v).sum()+1e-6)/((p*v+y*v).sum()+1e-6))/max(npos,1)
                    rec=recall_loss(logits,indices)/max(npos,1);loss=seg+.25*rec
                if not torch.isfinite(loss):raise FloatingPointError(f'{name} update {step}: nonfinite loss')
                total+=np.array([loss.item(),seg.item(),rec.item()]);scaler.scale(loss).backward()
            scaler.unscale_(opt);finite=all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
            if finite:norm=nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
            scaler.step(opt);scaler.update()
            if finite:break
        else:raise FloatingPointError('AMP same-batch retry limit')
        sched.step();row=dict(update=step,attempts=attempts,loss=total.tolist(),seconds=time.time()-start,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,amp_scale=scaler.get_scale())
        history.append(row)
        if step%25==0 or step==len(ledger):
            save(folder/'history.json',history);save(out/'status.json',dict(status='training',arm=name,**row));print(name,row,flush=True)
        if step%50==0 or step==len(ledger):
            tmp=folder/'last.tmp';torch.save(dict(model_state_dict=m.state_dict(),optimizer_state_dict=opt.state_dict(),scheduler_state_dict=sched.state_dict(),scaler_state_dict=scaler.state_dict(),update=step,source_sha256=sha(SOURCE),deployment='Segmentation only; never use this copy for tip predictions'),tmp);tmp.replace(folder/'last.pt')
    assert frozen_hash==immutable_hash(m,features),'Frozen tensors changed'
    save(folder/'freeze_verification.json',dict(frozen_tensor_hash=frozen_hash,unchanged=True,tip_head_weights_frozen=True,tip_output_policy='Separate complete frozen reference; adapted copy tip outputs are NOT used'))
    return m.eval()

class SplitInference(nn.Module):
    def __init__(self,seg,tip):super().__init__();self.seg=seg;self.tip=tip
    def forward(self,x):
        out=self.seg(x);fixed=self.tip(x)
        return dict(seg_logits=out['seg_logits'],tip_logits=fixed['tip_logits'],tip_probs=fixed['tip_probs'])

def native_eval(m,out,name,ann):
    folder=out/(name+'_probabilities');folder.mkdir();validator=NativeValidator(768,torch.device('cuda'));counter=[0]
    assert all((IMAGES_DIR/f'{sid}.jpg').exists() for sid in validator.val_study_ids)
    def capture(module,inputs,outputs):
        sid=validator.val_study_ids[counter[0]];counter[0]+=1
        np.savez_compressed(folder/f'{sid}.npz',prob=outputs['seg_logits'].float().sigmoid()[0,0].detach().cpu().numpy())
    handle=m.register_forward_hook(capture)
    try:result=validator.evaluate_model(m)
    finally:handle.remove();m.eval()
    assert counter[0]==193
    rows=[]
    for sid in validator.val_study_ids:rows.extend(score_study(np.load(folder/f'{sid}.npz')['prob'],sid,ann))
    df=pd.DataFrame(rows);assert len(df)==248
    np.testing.assert_allclose(df.coverage10.mean(),result['summary']['centerline_coverage_10px'],atol=1e-8)
    df.to_csv(out/f'{name}_paths.csv',index=False);save(out/f'{name}_native.json',result)
    s=result['summary'];s.update(missed_paths=int(df.missed.sum()),mean_longest_gap_px=float(df.longest_gap_native_px.mean()),paths_90pct=int(df.coverage_at_least_90pct.sum()),paths_complete=int(df.coverage_complete.sum()))
    return s

def report(out,results):
    histories=[json.loads((out/n/'history.json').read_text()) for n in ['head_control','feature_adaptation']]
    assert len(histories[0])==len(histories[1])
    np.testing.assert_allclose(histories[0][0]['loss'],histories[1][0]['loss'],atol=1e-6,rtol=0)
    frames={n:pd.read_csv(out/f'{n}_paths.csv').set_index(['study_id','catheter_instance_id']) for n in results}
    paired=frames['baseline'][['coverage10','longest_gap_native_px','missed','cvc_count']].add_prefix('baseline_')
    for n in ['head_control','feature_adaptation']:
        paired=paired.join(frames[n][['coverage10','longest_gap_native_px','missed']].add_prefix(n+'_'),validate='one_to_one')
    assert len(paired)==248 and not paired.isna().any().any()
    paired['transition']=np.where(paired.baseline_missed,np.where(paired.feature_adaptation_missed,'persistent_miss','crossed_above_30pct'),np.where(paired.feature_adaptation_missed,'new_miss','retained'))
    paired.to_csv(out/'path_transitions.csv')
    b,c,e=[results[n] for n in ['baseline','head_control','feature_adaptation']]
    criteria=dict(fewer_misses_than_baseline=e['missed_paths']<b['missed_paths'],fewer_misses_than_control=e['missed_paths']<c['missed_paths'],dice_preserved=e['macro_dice']>=b['macro_dice'],precision_preserved=e['precision']>=b['precision'],coverage_preserved=e['centerline_coverage_10px']>=b['centerline_coverage_10px'],negative_false_area_preserved=e['negative_false_area_pct']<=b['negative_false_area_pct'],longest_gap_not_worse=e['mean_longest_gap_px']<=b['mean_longest_gap_px'])
    save(out/'decision.json',dict(criteria=criteria,passed=all(criteria.values()),promoted=False,tip_branch='Original frozen model',qualification='Single-seed repeatedly used development cohort; provisional tip truth; not independent clinical validation'))
    lines=['# Segmentation feature adaptation pilot','', '| Arm | Misses /248 | Coverage10 | Dice | Precision | Negative false area % | Mean longest gap px | Paths >=90% |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,s in results.items():lines.append(f"| {n} | {s['missed_paths']} | {s['centerline_coverage_10px']:.2%} | {s['macro_dice']:.4f} | {s['precision']:.2%} | {s['negative_false_area_pct']:.5f} | {s['mean_longest_gap_px']:.1f} | {s['paths_90pct']} |")
    lines+=['',f"Locked progression decision: {'PASS' if all(criteria.values()) else 'FAIL'}. No model promoted.",'','Only trainable parameter scope differs between arms. Same training-mined hard-positive ledger, losses, resolution, seed, successful updates and segmentation-head learning rate. Newly trainable features use LR 1e-5. BN buffers stay fixed. Tip outputs must use the complete original frozen model, never the adapted copy.','', 'Coverage is proximity to a union mask, not instance association. Gap metrics integrate consecutive uncovered samples along the visible annotated path, in native pixels. Source annotations may be incomplete. Eight negatives cannot establish reliable clinical specificity. No millimeter calibration or independent test claim.']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')

def main(args):
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    old=list((ROOT/'experiments').rglob('*.pt'));frozen={str(p):sha(p) for p in old};save(out/'frozen_hashes.json',frozen)
    sources=[Path(__file__),ROOT/'experiments/hrnet_recovery_v3/run_recall_pilot.py',ROOT/'experiments/hrnet_recovery_v3/audit_missed_paths.py',V1/'src/models/hrnet_joint_tip.py',ROOT/'data/splits.json',ROOT/'data/annotation_inventory.parquet',ROOT/'data/study_inventory.parquet']
    for p in sources:
        dest=out/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'source_hashes.json',{str(p):sha(p) for p in sources})
    save(out/'protocol.json',dict(updates=args.updates,seed=42,input=768,effective_batch=4,microbatch=1,arms=['head_control','feature_adaptation'],head_lr=5e-5,feature_lr=1e-5,loss='BCE pos_weight10 + Dice + 0.25 equal-instance recall; identical arms',threshold=.5,mining='training only; same hard-positive ledger in both arms',tip_policy='Separate entire frozen original model',promotion='None automatic; fewer misses than baseline and control; baseline Dice, precision, coverage, false area and longest-gap preservation',limitations='Development experiment, provisional labels, not clinically attested'))
    torch.set_num_threads(4);cv2.setNumThreads(0)
    try:
        save(out/'status.json',dict(status='preparing'))
        ledger,cache=prepare(out,1000 if not args.smoke else args.updates)
        ann=pd.read_parquet(ROOT/'data/annotation_inventory.parquet');ann=ann[ann.device_type=='CVC']
        ledger=mine(out,cache,ledger[:args.updates],ann) if not args.smoke else ledger
        if args.smoke:
            for name,feature in [('head_control',False),('feature_adaptation',True)]:
                m=train(name,feature,out,ledger,cache);del m;gc.collect();torch.cuda.empty_cache()
        else:
            results={};m=load();results['baseline']=native_eval(m,out,'baseline',ann);del m;gc.collect();torch.cuda.empty_cache()
            for name,feature in [('head_control',False),('feature_adaptation',True)]:
                m=train(name,feature,out,ledger,cache)
                save(out/'status.json',dict(status='evaluating',arm=name));results[name]=native_eval(m,out,name,ann)
                # Verify the actual deployment adapter, not unused adapted tip logits.
                fixed=load().requires_grad_(False);x=tensor(next(iter(cache.values()))[0])
                with torch.inference_mode():
                    expected=fixed(x)['tip_logits'];actual=SplitInference(m,fixed).eval()(x)['tip_logits']
                assert torch.equal(expected,actual)
                save(out/name/'deployment_tip_check.json',dict(fp32_bitwise_equal=True,source_sha256=sha(SOURCE),separate_frozen_model=True))
                save(out/'comparison.json',results);del m,fixed;gc.collect();torch.cuda.empty_cache()
            report(out,results)
        assert all(sha(p)==h for p,h in frozen.items()),'Original checkpoint changed'
        save(out/'status.json',dict(status='complete',updates_per_arm=args.updates,original_checkpoints_unchanged=True,smoke=args.smoke))
    except Exception:
        save(out/'status.json',dict(status='failed',traceback=traceback.format_exc(),original_checkpoints_unchanged=all(sha(p)==h for p,h in frozen.items())));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--updates',type=int,default=250);p.add_argument('--smoke',action='store_true');main(p.parse_args())
