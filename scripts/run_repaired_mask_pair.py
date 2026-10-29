"""Isolated, sequential pilot and native evaluation; never overwrite frozen models."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/paired_repaired_20261016'
FROZEN = ROOT / 'outputs/audit50_repaired_pilot_20261016/frozen_reference_hashes.json'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def status(state, **kwargs):
    tmp = OUT / 'status.tmp'
    tmp.write_text(json.dumps(dict(status=state, timestamp=time.time(), **kwargs), indent=2))
    os.replace(tmp, OUT / 'status.json')

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    refs = json.loads(FROZEN.read_text())
    for r in refs:
        assert digest(r['source']) == r['sha256'] == digest(r['snapshot'])
    files = ['scripts/train_segmentation_v3.py', 'src/data/segmentation_v3_dataset.py',
             'src/models/segmentation_v3.py', 'scripts/evaluate_cvc_masks_native.py',
             'runs/initial_weights.pt', 'data/splits.json', 'data/annotation_inventory.parquet']
    (OUT/'input_hashes.json').write_text(json.dumps({f:digest(ROOT/f) for f in files}, indent=2))
    try:
        for name, size, micro in [('s1_512',512,2), ('s2_768',768,1)]:
            status('running', arm=name, phase='training')
            cmd = [sys.executable,'-u','scripts/train_segmentation_v3.py',
                   '--output_dir',str(OUT/name),'--target_updates','1000',
                   '--img_size',str(size),'--micro_batch_size',str(micro),
                   '--effective_batch_size','4','--load_initial_weights','runs/initial_weights.pt',
                   '--seed','42','--aug_seed','42','--bce_pos_weight','10','--dice_weight','1']
            with open(OUT/f'{name}_training.log','w') as log:
                subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            status('running', arm=name, phase='native_evaluation')
            cmd = [sys.executable,'-u','scripts/evaluate_cvc_masks_native.py',
                   '--checkpoint',str(OUT/name/'checkpoints/final_segmentation_v3.pt'),
                   '--img_size',str(size),'--report_tag',name,'--output_dir',str(OUT/'evaluation')]
            with open(OUT/f'{name}_evaluation.log','w') as log:
                subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for r in refs: assert digest(r['source']) == r['sha256'] == digest(r['snapshot'])
        status('complete', frozen_hashes_unchanged=True)
    except BaseException as e:
        status('failed',error=repr(e))
        raise

if __name__ == '__main__': main()
