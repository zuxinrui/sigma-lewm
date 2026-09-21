"""OGBench visual Scene -> the HDF5 layout the Scene configs read.

`_episode_bounds` and `_write_scene_hdf5` are copied unchanged from AC-MTM
(https://github.com/jackboyla/action-contrastive-jepa, scene_prep.py at 43c96f66bb12),
MIT licensed; see third_party/ac_mtm/LICENSE. Only the Modal app around them is dropped,
so the Scene data and evaluation cases are exactly the ones AC-MTM reports on.
"""

from __future__ import annotations

SCENE_DATASET_NAME = "visual-scene-play-v0"
# Flat file under $STABLEWM_HOME/datasets/: a name containing "/" would be taken for a
# Hugging Face repo id by stable-worldmodel's loader.
SCENE_H5_NAME = "visual_scene_play"


def _episode_bounds(terminals):
    import numpy as np

    terminal_idx = np.flatnonzero(np.asarray(terminals, dtype=bool))
    if terminal_idx.size == 0:
        raise ValueError("Scene dataset has no terminal markers")
    starts = np.concatenate([[0], terminal_idx[:-1] + 1])
    lengths = terminal_idx - starts + 1
    if np.any(lengths <= 1):
        raise ValueError(f"Scene episodes must contain transitions; got lengths={lengths}")
    return starts.astype(np.int64), lengths.astype(np.int64)


def _write_scene_hdf5(
    *,
    source_npz,
    output_h5,
    max_episodes: int | None = None,
) -> None:
    import h5py
    import numpy as np

    required = {"observations", "actions", "terminals", "qpos", "qvel", "button_states"}
    missing = required.difference(source_npz.files)
    if missing:
        raise KeyError(f"Scene npz missing required arrays: {sorted(missing)}")

    observations = source_npz["observations"]
    actions = source_npz["actions"]
    terminals = source_npz["terminals"]
    qpos = source_npz["qpos"]
    qvel = source_npz["qvel"]
    button_states = source_npz["button_states"]

    if observations.ndim != 4 or observations.shape[-1] not in (1, 3):
        raise ValueError(
            "Expected visual observations with shape (T, H, W, C); "
            f"got {observations.shape}"
        )
    if qpos.shape[1] < 25:
        raise ValueError(f"Expected Scene qpos dim >=25; got {qpos.shape}")
    if button_states.shape[1] < 2:
        raise ValueError(f"Expected two Scene button states; got {button_states.shape}")

    starts, raw_lengths = _episode_bounds(terminals)
    transition_lengths = raw_lengths - 1
    if max_episodes is not None:
        starts = starts[:max_episodes]
        transition_lengths = transition_lengths[:max_episodes]
    total = int(transition_lengths.sum())
    offsets = np.concatenate([[0], np.cumsum(transition_lengths[:-1])]).astype(np.int64)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    pixel_chunks = (min(256, total), *observations.shape[1:])
    vector_chunks = (min(4096, total),)

    with h5py.File(output_h5, "w") as h5:
        h5.create_dataset("ep_len", data=transition_lengths.astype(np.int64))
        h5.create_dataset("ep_offset", data=offsets)
        h5.attrs["source_dataset"] = SCENE_DATASET_NAME
        h5.attrs["notes"] = "OGBench visual Scene with terminal observations dropped."

        h5.create_dataset(
            "pixels",
            shape=(total, *observations.shape[1:]),
            dtype=observations.dtype,
            chunks=pixel_chunks,
        )
        h5.create_dataset(
            "action",
            shape=(total, actions.shape[1]),
            dtype="float32",
            chunks=(min(4096, total), actions.shape[1]),
        )
        h5.create_dataset(
            "qpos",
            shape=(total, qpos.shape[1]),
            dtype="float32",
            chunks=(min(4096, total), qpos.shape[1]),
        )
        h5.create_dataset(
            "qvel",
            shape=(total, qvel.shape[1]),
            dtype="float32",
            chunks=(min(4096, total), qvel.shape[1]),
        )
        h5.create_dataset(
            "button_states",
            shape=(total, button_states.shape[1]),
            dtype=button_states.dtype,
            chunks=(min(4096, total), button_states.shape[1]),
        )
        for name, dtype in (
            ("button_state_0", "int64"),
            ("button_state_1", "int64"),
            ("privileged_drawer_pos", "float32"),
            ("privileged_window_pos", "float32"),
            ("episode_idx", "int64"),
            ("step_idx", "int64"),
            ("terminal", "bool"),
        ):
            h5.create_dataset(name, shape=(total,), dtype=dtype, chunks=vector_chunks)
        h5.create_dataset(
            "privileged_block_0_pos",
            shape=(total, 3),
            dtype="float32",
            chunks=(min(4096, total), 3),
        )
        h5.create_dataset(
            "privileged_block_0_quat",
            shape=(total, 4),
            dtype="float32",
            chunks=(min(4096, total), 4),
        )

        cursor = 0
        for ep_idx, (start, length) in enumerate(zip(starts, transition_lengths)):
            src = slice(int(start), int(start + length))
            dst = slice(cursor, cursor + int(length))

            h5["pixels"][dst] = observations[src]
            h5["action"][dst] = actions[src].astype("float32", copy=False)
            h5["qpos"][dst] = qpos[src].astype("float32", copy=False)
            h5["qvel"][dst] = qvel[src].astype("float32", copy=False)
            h5["button_states"][dst] = button_states[src]
            h5["button_state_0"][dst] = button_states[src, 0]
            h5["button_state_1"][dst] = button_states[src, 1]
            h5["privileged_block_0_pos"][dst] = qpos[src, 14:17].astype(
                "float32", copy=False
            )
            h5["privileged_block_0_quat"][dst] = qpos[src, 17:21].astype(
                "float32", copy=False
            )
            h5["privileged_drawer_pos"][dst] = qpos[src, 23].astype(
                "float32", copy=False
            )
            h5["privileged_window_pos"][dst] = qpos[src, 24].astype(
                "float32", copy=False
            )
            h5["episode_idx"][dst] = ep_idx
            h5["step_idx"][dst] = np.arange(length, dtype=np.int64)
            terminal = np.zeros(int(length), dtype=bool)
            terminal[-1] = True
            h5["terminal"][dst] = terminal

            cursor += int(length)
            if (ep_idx + 1) % 25 == 0 or ep_idx + 1 == len(starts):
                print(
                    f"wrote {ep_idx + 1}/{len(starts)} episodes "
                    f"({cursor}/{total} rows)",
                    flush=True,
                )
