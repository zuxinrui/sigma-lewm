"""Repository root and namespaced artifact paths; no downloads or implicit writes."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(uri):
    scheme,tail=uri.split(':',1)
    if scheme=='isaac_python':
        return os.environ.get('ISAAC_PYTHON',str(REPO_ROOT/'artifacts/isaac/bin/python'))
    roots={
        'repo':REPO_ROOT,
        'stable':Path(os.environ.get('STABLEWM_HOME',str(REPO_ROOT/'artifacts/stablewm'))),
        'data':Path(os.environ.get('SLEWM_DATA_ROOT',str(REPO_ROOT/'artifacts/data'))),
        'checkpoints':Path(os.environ.get('SLEWM_CHECKPOINT_ROOT',str(REPO_ROOT/'artifacts/checkpoints'))),
        'legacy':Path(os.environ.get('SLEWM_LEGACY_ARTIFACTS',str(REPO_ROOT/'artifacts/legacy'))),
        'runs':Path(os.environ.get('SLEWM_RUN_ROOT',str(REPO_ROOT/'runs'))),
    }
    if scheme not in roots:raise ValueError(f'Unknown artifact namespace: {scheme}')
    relative=Path(tail)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Artifact path escapes its root')
    return str(roots[scheme]/relative)


def resolve_recorded(path):
    """Resolve a namespaced path stored in a manifest; plain paths pass through."""
    return resolve_path(path) if path.split(':',1)[0] in ('repo','stable','data','checkpoints','legacy','runs') else path
