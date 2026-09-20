"""Record observed package versions, not authenticated package-index URLs."""
from importlib import metadata
import json
from pathlib import Path
import platform

R=Path(__file__).resolve().parents[1]
packages=['torch','torchvision','numpy','scipy','pandas','h5py','einops','hydra-core',
 'omegaconf','lightning','stable-pretraining','stable-worldmodel','mujoco',
 'gymnasium','scikit-learn','matplotlib','Pillow','transformers','wandb','pytest']
found={};missing=[]
for p in packages:
    try:found[p]=metadata.version(p)
    except metadata.PackageNotFoundError:missing.append(p)
(R/'requirements-observed.txt').write_text('# Observed environment; not a validated portable lockfile.\n'+
    '\n'.join(p+'=='+v for p,v in sorted(found.items()))+'\n')
(R/'repro/environment.json').write_text(json.dumps(dict(python=platform.python_version(),
    system=platform.system(),packages=found,missing=missing),indent=2))
