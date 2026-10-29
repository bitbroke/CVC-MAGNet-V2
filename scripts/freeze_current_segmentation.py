"""Snapshot the current research baseline without loading or training a model."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'archives' / 'CVC-SegTip-V2-Frozen-20261016'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    if DEST.exists():
        raise SystemExit('Archive already exists; refusing to overwrite.')
    files = []
    for folder in ['src', 'scripts', 'configs', 'tests',
                   'runs/pilot_experiment_expanded', 'reports/independent_heatmap_audit']:
        files.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    files.extend((ROOT / 'data').glob('*.*'))
    records = []
    for source in sorted(set(files)):
        relative = source.relative_to(ROOT)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        shutil.copy2(source, target)
        if before != sha(target) or before != sha(source):
            raise RuntimeError(f'Snapshot integrity failure: {relative}')
        records.append({'path': relative.as_posix(), 'sha256': before,
                        'bytes': source.stat().st_size})
    manifest = {'name': DEST.name, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'policy': 'Reference-only. Never resume or overwrite this archive. New experiments use new directories.',
                'scope': 'Expanded run checkpoints/config/history, current code snapshot, top-level data registries, independent heatmap audit. Code snapshot is current, not asserted identical to historical training code.',
                'raw_images_included': False, 'files': records}
    (DEST / 'FREEZE_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'archive': str(DEST), 'verified_files': len(records),
                      'bytes': sum(r['bytes'] for r in records)}))

if __name__ == '__main__':
    main()
