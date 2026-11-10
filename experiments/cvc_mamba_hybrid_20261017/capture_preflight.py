"""Read-only host inventory and copied-reference integrity; does not run CUDA kernels."""
import datetime
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def command(args):
    result = subprocess.run(args, capture_output=True, timeout=30)
    return {'returncode': result.returncode,
            'stdout': result.stdout.decode('utf-8', errors='replace'),
            'stderr': result.stderr.decode('utf-8', errors='replace')}

def main():
    for name in ['frozen_reference', 'preflight', 'configs', 'src', 'scripts', 'runs', 'reports']:
        (OUT / name).mkdir(exist_ok=True)
    manifest_path = OUT / 'frozen_reference' / 'manifest.json'
    if manifest_path.exists():
        raise RuntimeError('Reference already frozen; refusing to overwrite this capture.')
    sources = [
        'runs/s4_experiment_seed43/checkpoints/final_segmentation_v3.pt',
        'runs/s4_experiment_seed43/train_summary.json',
        'src/models/segmentation_v3.py', 'src/data/segmentation_v3_dataset.py',
        'scripts/train_segmentation_v3.py', 'scripts/evaluate_s4_experiment.py',
        'data/splits.json', 'data/selective_hard_negatives_manifest.json',
        'data/study_inventory.parquet', 'data/annotation_inventory.parquet',
        'reports/native_eval_S4_experiment_seed43_final.json',
        'reports/native_eval_S4_experiment_seed43_final_per_study.json',
    ]
    records = []
    for rel in sources:
        source = ROOT / rel
        if not source.is_file():
            raise FileNotFoundError(source)
        before = sha(source)
        target = OUT / 'frozen_reference' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(source) != before or sha(target) != before:
            raise RuntimeError(f'Hash changed during copy: {rel}')
        records.append({'original': str(source), 'copy': str(target),
                        'sha256': before, 'bytes': source.stat().st_size})
    manifest_path.write_text(json.dumps({'created': datetime.datetime.now().astimezone().isoformat(),
                                        'files': records}, indent=2))
    import torch
    report = {
        'timestamp': datetime.datetime.now().astimezone().isoformat(),
        'status': 'environment_checked_gpu_feasibility_not_tested',
        'platform': platform.platform(), 'python': sys.version,
        'python_executable': sys.executable, 'torch': torch.__version__,
        'torch_cuda_build': torch.version.cuda,
        'packages_present': {x: importlib.util.find_spec(x) is not None
                             for x in ['mamba_ssm', 'causal_conv1d', 'triton']},
        'nvidia_smi': command(['nvidia-smi']),
        'processes': command(['powershell', '-NoProfile', '-Command',
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' } | Select-Object ProcessId,CommandLine | ConvertTo-Json"]),
        'blockers': ['Existing S5 training occupies GPU; no concurrent benchmark attempted.',
                     'Official Mamba dependencies absent from active environment.',
                     'Official Linux route requires separate setup; WSL not installed.'],
        'gpu_fit': None, 'training_launched': False,
        'original_reference_hashes_unchanged': all(sha(Path(r['original'])) == r['sha256'] for r in records),
    }
    (OUT / 'preflight' / 'environment.json').write_text(json.dumps(report, indent=2))
    (OUT / 'preflight' / 'packages.txt').write_text(command([sys.executable, '-m', 'pip', 'freeze'])['stdout'])
    print(json.dumps({'status': report['status'], 'files_frozen': len(records),
                      'original_hashes_unchanged': report['original_reference_hashes_unchanged'],
                      'output': str(OUT)}, indent=2))

if __name__ == '__main__':
    main()
