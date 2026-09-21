#!/usr/bin/env python
"""Download OGBench `visual-scene-play-v0` and write $STABLEWM_HOME/datasets/visual_scene_play.h5.

    python scripts/prep_ogb_scene.py            # ~1.9 GB download, ~12.6 GB HDF5, a few minutes
    python scripts/prep_ogb_scene.py --force    # rebuild an existing file

The conversion is AC-MTM's (slewm/data/ogb_scene.py), so training data and evaluation cases
match the OGBench-Scene numbers AC-MTM reports.
"""
import argparse, os
from pathlib import Path

import h5py
import numpy as np
import ogbench

from slewm.data.ogb_scene import SCENE_DATASET_NAME, SCENE_H5_NAME, _write_scene_hdf5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite an existing HDF5")
    ap.add_argument("--keep-npz", action="store_true", help="keep the downloaded OGBench archive")
    a = ap.parse_args()
    home = os.environ.get("STABLEWM_HOME")
    if not home:
        ap.error("set STABLEWM_HOME")
    out = Path(home).expanduser() / "datasets" / f"{SCENE_H5_NAME}.h5"
    if out.exists() and not a.force:
        print(f"exists, not touching: {out}")
        return
    raw = Path(home).expanduser() / "ogbench_raw"
    raw.mkdir(parents=True, exist_ok=True)
    ogbench.download_datasets([SCENE_DATASET_NAME], str(raw))
    src = raw / f"{SCENE_DATASET_NAME}.npz"
    part = out.with_suffix(".h5.partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    part.unlink(missing_ok=True)
    print(f"converting {src} -> {out}", flush=True)
    with np.load(src) as z:
        _write_scene_hdf5(source_npz=z, output_h5=part, max_episodes=None)
    os.replace(part, out)
    with h5py.File(out, "r") as h5:
        n = h5["pixels"].shape[0]
        print(f"{out}: {n} transitions, {len(h5['ep_len'])} episodes, pixels {h5['pixels'].shape[1:]}")
    if not a.keep_npz:
        for f in raw.glob(f"{SCENE_DATASET_NAME}*.npz"):
            f.unlink()


if __name__ == "__main__":
    main()
