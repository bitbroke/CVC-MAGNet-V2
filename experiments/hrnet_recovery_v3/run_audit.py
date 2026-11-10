"""Non-destructive recovery audit; no training and no automatic clinical attestation."""
import sys,json,hashlib,datetime,copy
from pathlib import Path
ROOT=Path('C:/Ankshit/CVC-MAGNet-V2');sys.path.insert(0,str(ROOT))
import torch
import pandas as pd
from experiments.architecture_comparison_v2.scripts.train_runner import NativeValidator
from experiments.hrnet_tip_pilot_v1.src.models.hrnet_joint_tip import HRNetJointTipNet
from experiments.hrnet_tip_improvement_v2.src.models.hrnet_joint_offset_tip import HRNetJointOffsetTipNet
from experiments.hrnet_tip_improvement_v2.src.run_continuation_pilot import evaluate_model_on_benchmark

OUT=Path(__file__).resolve().parent
V1=ROOT/'experiments/hrnet_tip_pilot_v1'
V2=ROOT/'experiments/hrnet_tip_improvement_v2'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,data):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(p)

def main():
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths={'baseline':V1/'runs/seed_42/t1_joint_hrnet/checkpoints/final_model.pt',
           'control':V2/'checkpoints/control_final_model.pt',
           'offset':V2/'checkpoints/experiment_final_model.pt'}
    hashes={k:sha(p) for k,p in paths.items()}
    save('provenance.json',{'checkpoints':{k:{'path':str(p),'sha256':hashes[k]} for k,p in paths.items()},'purpose':'re-evaluation only; original files untouched'})
    manifest=json.loads((V1/'manifests/development_benchmark_manifest.json').read_text())
    proposed=pd.read_csv(V2/'benchmark_v2.csv')
    records=[]
    for s in manifest['studies']:
        for t in s['assessable_tips']:
            candidate=proposed[(proposed.study_id==s['study_id'])&(proposed.catheter_instance_id==t['catheter_instance_id'])]
            assert len(candidate)==1
            records.append({'study_id':s['study_id'],'patient_id':s['patient_id'],
                'catheter_instance_id':t['catheter_instance_id'],'x':t['x'],'y':t['y'],
                'review_status':t['review_status'],'reviewer_attested':False,
                'proposed_v2_x':float(candidate.iloc[0].x),'proposed_v2_y':float(candidate.iloc[0].y),
                'requires_review':t['review_status']=='weak_heuristic_unreviewed'})
    registry=pd.DataFrame(records);registry.to_csv(OUT/'benchmark_provenance_restored.csv',index=False)
    registry[registry.requires_review].to_csv(OUT/'review_required_16.csv',index=False)
    failures=pd.read_csv(V2/'failure_analysis.csv')
    failures['previous_heuristic_category']=failures.failure_category
    failures['failure_category']=failures.apply(lambda r:'matched_under_provisional_reference' if r.status=='MATCHED' else 'unmatched_cause_unverified',axis=1)
    failures['causal_diagnosis_verified']=False
    failures.to_csv(OUT/'failure_analysis_corrected.csv',index=False)
    reviewed=registry[~registry.requires_review]
    complete_ids={s['study_id'] for s in manifest['studies'] if s['is_negative'] or s['has_all_tips_audited']}
    complete_studies=[s for s in manifest['studies'] if s['study_id'] in complete_ids]
    results={}
    for name,path in paths.items():
        print('START',name,flush=True)
        m=HRNetJointOffsetTipNet(pretrained=False,enable_offset=True) if name=='offset' else HRNetJointTipNet(pretrained=False)
        state=torch.load(path,map_location='cpu',weights_only=False)
        m.load_state_dict(state.get('model_state_dict',state),strict=True);m.to(device).eval()
        validator=NativeValidator(target_size=768,device=device)
        native=validator.evaluate_model(m)
        assert native['summary']['num_positive_studies']==185 and native['summary']['num_negative_studies']==8
        save(f'{name}_native_segmentation.json',native)
        print(name,'native',native['summary'],flush=True)
        results[name]={'native_segmentation':native['summary'],'tip':{}}
        for offsets in ([False,True] if name=='offset' else [False]):
            tag='learned_offsets' if offsets else 'quadratic_decoder'
            r=evaluate_model_on_benchmark(m,device,proposed,manifest['studies'],use_offset_head=offsets)
            r.pop('seg_dice',None)  # Incompatible pooled model-grid metric intentionally excluded.
            results[name]['tip'][tag]={'mixed_provisional_v2':r}
            # Same fully reviewed provisional studies for all arms, with genuine negatives.
            r2=evaluate_model_on_benchmark(m,device,registry[registry.study_id.isin(complete_ids)],complete_studies,use_offset_head=offsets)
            r2.pop('seg_dice',None)
            # Legacy helper hard-codes subgroup denominators: do not retain those rates.
            for key in ['single_cvc_success','multi_cvc_success']:r2.pop(key,None)
            r2['target_count']=int(registry.study_id.isin(complete_ids).sum())
            r2['study_count']=len(complete_studies)
            results[name]['tip'][tag]['complete_provisional_studies']=r2
            print(name,tag,r,flush=True)
        save('audit_results.json',results)
        del m;torch.cuda.empty_cache()
    assert all(sha(p)==hashes[k] for k,p in paths.items())
    save('status.json',{'status':'audit_complete','training_started':False,'source_checkpoint_hashes_unchanged':True,'unresolved_review_targets':16})

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        save('status.json',{'status':'failed','traceback':traceback.format_exc()})
        raise
