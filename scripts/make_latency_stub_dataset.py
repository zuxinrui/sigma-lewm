#!/usr/bin/env python3
"""Shrink a benchmark dataset to its first few episodes, for latency runs.

eval.py fits its normalisers on the non-pixel columns, and the planner gets its images from
the simulator rather than from the file, so a latency run needs the schema and a plausible
set of statistics, not the whole dataset. Episodes are kept whole, because the row indices
are derived from ep_offset and truncating rows alone breaks them. Timing depends on tensor
shapes and iteration counts, not on the values of the normalisation constants, so the
measurement is unaffected; nothing else should use these files.

    python scripts/make_latency_stub_dataset.py SOURCE.h5 DEST.h5 [--episodes 20]
"""
import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the codecs the sources use)
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    a.dest.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.source, "r") as src, h5py.File(a.dest, "w") as dst:
        lengths, offsets = np.asarray(src["ep_len"]), np.asarray(src["ep_offset"])
        keep = min(a.episodes, len(lengths))
        rows = int(offsets[keep - 1] + lengths[keep - 1])        # episodes are contiguous
        print(f"  keeping {keep}/{len(lengths)} episodes = {rows}/{offsets[-1] + lengths[-1]} rows")
        for key in src:
            data = src[key]
            if key == "ep_len" or key == "ep_offset":
                dst.create_dataset(key, data=np.asarray(data[:keep]))
            elif data.shape and data.shape[0] == lengths.shape[0] and key != "pixels":
                dst.create_dataset(key, data=np.asarray(data[:keep]))       # per-episode column
            elif data.shape and data.shape[0] >= rows:
                dst.create_dataset(key, data=np.asarray(data[:rows]),
                                   compression="gzip" if key == "pixels" else None)
            else:
                dst.create_dataset(key, data=np.asarray(data))
        for k, v in src.attrs.items():
            dst.attrs[k] = v
    mb = a.dest.stat().st_size / 1e6
    print(f"wrote {a.dest} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
