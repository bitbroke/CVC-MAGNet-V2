"""Post-run audit; CPU only, never changes checkpoints."""
import json,hashlib
from pathlib import Path
import torch
torch.set_num_threads(4)
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'experiments/hrnet_segmentation_features_v1/comparison'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=ROOT/'experiments/hrnet_tip_pilot_v1/runs/seed_42/t1_joint_hrnet/checkpoints/final_model.pt'
assert json.loads((out/'status.json').read_text())['original_checkpoints_unchanged']
inputs=json.loads((out/'source_hashes.json').read_text());assert all(sha(p)==h for p,h in inputs.items())
base=torch.load(source,map_location='cpu',weights_only=False)['model_state_dict'];report={}
for arm in ['head_control','feature_adaptation']:
    checkpoint=out/arm/'last.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
    changed=[k for k in base if not torch.equal(base[k],state[k])]
    assert changed and not any(k.startswith('tip_head.') for k in changed)
    if arm=='head_control':assert all(k.startswith('seg_head.') for k in changed)
    else:assert any(k.startswith('encoder.') for k in changed) and any(k.startswith('fusion.') for k in changed)
    assert not any(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in changed)
    report[arm]=dict(checkpoint_sha256=sha(checkpoint),changed_tensor_count=len(changed),changed_modules=sorted({k.split('.')[0] for k in changed}),tip_head_weights_unchanged=True,bn_buffers_unchanged=True)
    deployment=dict(status='experimental_not_promoted',segmentation_checkpoint=str(checkpoint),segmentation_sha256=sha(checkpoint),tip_checkpoint=str(source),tip_sha256=sha(source),requirement='Use separate complete frozen tip model. Do not decode adapted model internal tip logits.')
    (out/arm/'deployment_manifest.json').write_text(json.dumps(deployment,indent=2))
report['source_inputs_unchanged']=True
(out/'postrun_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
