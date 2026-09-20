#!/usr/bin/env python3
"""Append-only, self-contained snapshots of physical embedding visualization runs.

Independent copies (never hard links) preserve data even when working caches
are overwritten. A manifest is committed only after every copy verifies.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'outputs/embedding_history'
SOURCES = [
    'outputs/fig9_original_recovery',
    'outputs/pusht_physical_atlas_v1',
    'outputs/cube_gear_physical_atlas_v1',
    'outputs/focused_physical_atlas_v1',
    'outputs/physical_probe_ranking_v1',
    'outputs/figs',
    'outputs/PHYSICAL_PROBE_RANKING.md',
    'tests/test_embedding_archive.py',
    'tests/test_focused_physical_atlas.py',
    'scripts/archive_embedding_round.py',
    'scripts/pusht_physical_atlas.py',
    'scripts/cube_gear_physical_atlas.py',
    'scripts/recover_fig9_atlas.py',
    'scripts/build_focused_physical_atlas.py',
    'scripts/rank_physical_probes.py',
    'scripts/check_physical_probe_stability.py',
    'trim/training/utilities.py',
]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def snapshot(label, *, root=ROOT, archive=ARCHIVE, sources=None, metadata=None):
    root,archive=Path(root),Path(archive)
    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
        raise ValueError('Snapshot label must be a safe identifier')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    dest=archive/f'{stamp}_{label}_{uuid.uuid4().hex[:8]}'
    dest.mkdir(parents=True,exist_ok=False)
    records=[]; missing=[]; seen=set()
    for name in SOURCES if sources is None else sources:
        source=root/name
        if not source.exists():missing.append(name);continue
        files=sorted(source.rglob('*')) if source.is_dir() else [source]
        for path in files:
            if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts:continue
            rel=path.relative_to(root)
            if rel in seen:continue
            seen.add(rel)
            if dest.is_relative_to(path) or path.is_relative_to(archive):
                raise ValueError('Archive cannot include itself')
            target=dest/'files'/rel;target.parent.mkdir(parents=True,exist_ok=True)
            before=path.stat(); expected=digest(path)
            shutil.copy2(path,target)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or digest(target)!=expected:
                raise RuntimeError(f'Source changed during snapshot: {path}; incomplete archive has no manifest')
            records.append(dict(path=str(rel),bytes=target.stat().st_size,sha256=expected))
    packages={}
    for name in ['numpy','scipy','scikit-learn','matplotlib','torch']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=root,text=True,capture_output=True)
    manifest=dict(version=1,label=label,created_utc=stamp,files=records,missing_optional_sources=missing,
        argv=sys.argv,python=sys.version,packages=packages,git_head=git.stdout.strip() if git.returncode==0 else None,
        metadata=metadata or {},storage='Independent copies; timestamp and UUID; no snapshot reused.',
        limitations='Local archive, not off-machine backup. Raw datasets and checkpoint weights are referenced by source provenance, not copied. Paper PDF may predate figure regeneration; see phase metadata.')
    with (dest/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    verify(dest)
    print(f'[embedding archive] {dest} ({len(records)} files)',flush=True)
    return dest

def verify(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    for entry in manifest['files']:
        path=directory/'files'/entry['path']
        if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
            raise ValueError(f'Archive integrity failure: {path}')
    return len(manifest['files'])

@contextmanager
def archive_round(label):
    # If archiving fails, stop before overwriting the working artifacts.
    previous=snapshot(label+'_before',metadata={'phase':'before'})
    try:
        yield
    except BaseException:
        snapshot(label+'_failed',metadata={'phase':'failed-partial','previous_snapshot':str(previous)})
        raise
    else:
        snapshot(label+'_after',metadata={'phase':'completed','previous_snapshot':str(previous)})

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',default='manual')
    parser.add_argument('--verify',type=Path)
    args=parser.parse_args()
    if args.verify:print(f'Verified {verify(args.verify)} files')
    else:snapshot(args.label,metadata={'phase':'manual-current-state'})

if __name__=='__main__':main()
