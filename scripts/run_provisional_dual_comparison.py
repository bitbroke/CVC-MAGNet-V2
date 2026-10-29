"""Sequential, logged execution of the user-authorized provisional-label comparison."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs' / 'provisional_comparison_20261012'
RUN.mkdir(exist_ok=False)
env = os.environ.copy()
env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')

def write(name, data):
    dest = RUN / name
    temp = dest.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(dest)

status = {'status': 'running', 'scope': 'Provisional engineering-label feasibility; not clinical accuracy',
          'started_utc': datetime.now(timezone.utc).isoformat(), 'arms': {}}
write('status.json', status)
results = {}
try:
    for mode in ('seg_only', 'dual_head'):
        output = RUN / mode
        command = [sys.executable, '-u', str(ROOT/'scripts/05_train_coarse_dual.py'),
                   '--mode', mode, '--output_dir', str(output), '--max_steps', '1000',
                   '--batch_size', '4', '--seed', '42', '--seg_pos_weight', '50',
                   '--seg_dice_weight', '2', '--seg_bce_weight', '1',
                   '--tip_pos_weight', '100', '--tip_weight', '2',
                   '--partial_review_policy', 'exclude', '--execute_training']
        status['current_arm'] = mode
        status['arms'][mode] = {'status': 'running', 'command': command}
        write('status.json', status)
        with (RUN/f'{mode}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{mode} exited {result.returncode}; see its persistent log')
        history = json.loads((output/'history.json').read_text())
        if history['final_step'] != 1000:
            raise RuntimeError(f'{mode} did not complete the fixed-step budget')
        results[mode] = {'final': history['val_evaluations'][-1],
                         'best_step': history['best_step'],
                         'best_metric': history['best_metric'],
                         'seconds': history['total_time_seconds'],
                         'initialization': history['resolved_config']['initialization']}
        status['arms'][mode]['status'] = 'complete'
        write('comparison.json', results)
        write('status.json', status)
    if results['seg_only']['initialization']['backbone_hash'] != results['dual_head']['initialization']['backbone_hash']:
        raise RuntimeError('Initial backbone hash mismatch')
    status['status'] = 'complete'
except Exception as exc:
    status['status'] = 'failed'
    status['error'] = str(exc)
    raise
finally:
    status['updated_utc'] = datetime.now(timezone.utc).isoformat()
    write('status.json', status)
