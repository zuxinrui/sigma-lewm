"""Exact multi-step HDF5 reader for fixed-rate visual control trajectories.

One stored row represents a physical observation ``s_t`` and the action
``a_t`` that starts the next raw-control transition.  A sample with
``transitions=5`` and ``frameskip=5`` therefore contains six images
``s_t,s_(t+5),...,s_(t+25)`` and five macro actions, each formed from the five
distinct raw actions between adjacent images.

Unlike the legacy reader, this class never invents an action for the final
observation and makes the state/action cardinality explicit.  Raw-tick clip
starts remain allowed, matching the shipped Cube data protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch

import stable_worldmodel as swm


class AlignedControlHDF5Dataset(torch.utils.data.Dataset):
    """Read ``T+1`` observations and exactly ``T`` action blocks."""

    def __init__(
        self,
        *,
        name: str,
        cache_dir: str | Path | None = None,
        transitions: int = 5,
        frameskip: int = 5,
        clip_stride: int = 1,
        goal_window_repeat: int = 0,
        keys_to_load: list[str] | None = None,
        transform: Callable[[dict], dict] | None = None,
        required_action_alignment: str | None = None,
    ) -> None:
        super().__init__()
        if transitions < 1:
            raise ValueError("transitions must be positive")
        if frameskip < 1:
            raise ValueError("frameskip must be positive")
        if clip_stride < 1:
            raise ValueError("clip_stride must be positive")
        if goal_window_repeat < 0:
            raise ValueError("goal_window_repeat must be non-negative")

        raw_path = Path(name).expanduser()
        if raw_path.is_absolute() or raw_path.exists():
            path = raw_path
        else:
            path = Path(
                swm.data.utils.get_cache_dir(
                    cache_dir, sub_folder="datasets"
                ),
                name,
            )
        if not path.is_file():
            raise FileNotFoundError(path)

        self.h5_path = path.resolve()
        self.transitions = int(transitions)
        self.frameskip = int(frameskip)
        self.clip_stride = int(clip_stride)
        self.goal_window_repeat = int(goal_window_repeat)
        self.transform = transform
        self.h5_file: h5py.File | None = None
        self._keys = list(dict.fromkeys(keys_to_load or ["pixels", "action"]))
        if "action" not in self._keys:
            self._keys.append("action")

        with h5py.File(self.h5_path, "r") as source:
            missing = ({"ep_len", "ep_offset"} | set(self._keys)) - set(
                source.keys()
            )
            if missing:
                raise KeyError(f"HDF5 columns missing: {sorted(missing)}")
            if required_action_alignment is not None:
                alignment = source.attrs.get("action_alignment", "")
                if isinstance(alignment, bytes):
                    alignment = alignment.decode("utf-8")
                if alignment != required_action_alignment:
                    raise ValueError(
                        f"action alignment {alignment!r} does not match required "
                        f"{required_action_alignment!r}"
                    )
            self.lengths = source["ep_len"][:].astype(np.int64)
            self.offsets = source["ep_offset"][:].astype(np.int64)
            self.goal_indices = (
                source["goal_idx"][:].astype(np.int64)
                if "goal_idx" in source
                else None
            )
            self.episode_has_success = (
                source["ep_has_success"][:].astype(bool)
                if "ep_has_success" in source
                else None
            )
            if len(self.lengths) != len(self.offsets):
                raise ValueError("ep_len and ep_offset have different lengths")
            if len(self.lengths) == 0:
                raise ValueError("dataset contains no episodes")
            total_rows = int(source["action"].shape[0])
            for episode, (offset, length) in enumerate(
                zip(self.offsets, self.lengths)
            ):
                if length < 1 or offset < 0 or offset + length > total_rows:
                    raise ValueError(
                        f"episode {episode} has invalid offset/length "
                        f"({offset}, {length}) for {total_rows} rows"
                    )

        required_ticks = self.transitions * self.frameskip
        self.clip_indices = [
            (episode, start)
            for episode, length in enumerate(self.lengths)
            for start in range(
                0,
                max(0, int(length) - required_ticks),
                self.clip_stride,
            )
        ]
        self.goal_clip_indices: list[tuple[int, int]] = []
        if self.goal_window_repeat:
            if self.goal_indices is None or self.episode_has_success is None:
                raise ValueError(
                    "goal_window_repeat requires goal_idx and ep_has_success metadata"
                )
            if (
                len(self.goal_indices) != len(self.lengths)
                or len(self.episode_has_success) != len(self.lengths)
            ):
                raise ValueError("goal metadata has a different episode count")
            for episode, (offset, length, goal, has_success) in enumerate(
                zip(
                    self.offsets,
                    self.lengths,
                    self.goal_indices,
                    self.episode_has_success,
                )
            ):
                if not has_success:
                    continue
                local_start = int(goal) - int(offset) - required_ticks
                if not 0 <= local_start < int(length) - required_ticks:
                    raise ValueError(
                        f"episode {episode} has invalid goal-aligned clip start {local_start}"
                    )
                target = (episode, local_start)
                self.goal_clip_indices.append(target)
                existing = self.clip_indices.count(target)
                self.clip_indices.extend(
                    [target] * max(0, self.goal_window_repeat - existing)
                )
        if not self.clip_indices:
            raise ValueError(
                "dataset has no clip with T action blocks and T+1 observations"
            )

    @property
    def column_names(self) -> list[str]:
        return list(self._keys)

    def __len__(self) -> int:
        return len(self.clip_indices)

    def _open(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.h5_path,
                "r",
                swmr=True,
                rdcc_nbytes=256 * 1024 * 1024,
            )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self._open()
        episode, local_start = self.clip_indices[index]
        start = int(self.offsets[episode]) + int(local_start)
        observation_rows = start + np.arange(
            self.transitions + 1, dtype=np.int64
        ) * self.frameskip

        sample: dict[str, torch.Tensor] = {}
        for key in self._keys:
            if key == "action":
                value = np.asarray(
                    self.h5_file[key][
                        start : start + self.transitions * self.frameskip
                    ]
                )
            else:
                value = np.asarray(self.h5_file[key][observation_rows])
            if value.dtype.kind in ("S", "U", "O"):
                raise TypeError(f"string column {key!r} is metadata, not model input")
            tensor = torch.from_numpy(value)
            if (
                key.startswith("pixels")
                and tensor.ndim == 4
                and tensor.shape[-1] in (1, 3)
            ):
                tensor = tensor.permute(0, 3, 1, 2)
            sample[key] = tensor

        # Training transforms fit raw-action statistics, so apply them before
        # concatenating each block.  This exactly matches the legacy HDF5 reader.
        if self.transform:
            sample = self.transform(sample)
        raw_action = sample["action"]
        if raw_action.shape[0] != self.transitions * self.frameskip:
            raise AssertionError("raw action sequence length changed")
        if not bool(torch.isfinite(raw_action).all()):
            raise AssertionError("non-finite action leaked into a real transition")
        sample["action"] = raw_action.reshape(self.transitions, -1)

        if sample["pixels"].shape[0] != self.transitions + 1:
            raise AssertionError("state sequence length changed")
        # These audit-only identities let action-discriminative objectives
        # guarantee that an in-batch negative comes from another trajectory.
        sample["_episode_index"] = torch.tensor(episode, dtype=torch.long)
        sample["_clip_start"] = torch.tensor(local_start, dtype=torch.long)
        return sample

    def get_col_data(self, column: str) -> np.ndarray:
        self._open()
        if column not in self.h5_file:
            raise KeyError(column)
        return self.h5_file[column][:]

    def get_column_stats(
        self, column: str, *, episode_indices: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return statistics over rows that encode real transitions.

        The last row of every episode is an observation with no stored
        successor.  Converters keep an action-shaped placeholder on that row
        for the rectangular HDF5 schema, but it must not affect action
        normalisation.  Keeping this rule in the aligned reader also makes
        episode-restricted fitting exact.
        """

        self._open()
        if column not in self.h5_file:
            raise KeyError(column)
        episodes = (
            range(len(self.lengths))
            if episode_indices is None
            else [int(episode) for episode in episode_indices]
        )
        chunks: list[np.ndarray] = []
        for episode in episodes:
            if not 0 <= episode < len(self.lengths):
                raise IndexError(f"episode index {episode} is out of range")
            start = int(self.offsets[episode])
            length = int(self.lengths[episode])
            end = start + length - (1 if column == "action" else 0)
            if end > start:
                chunks.append(np.asarray(self.h5_file[column][start:end]))
        if not chunks:
            raise ValueError(f"no real rows found for column {column!r}")
        values = np.concatenate(chunks, axis=0)
        feature_shape = values.shape[1:]
        feature_dim = int(np.prod(feature_shape)) if feature_shape else 1
        values = values.reshape(-1, feature_dim)
        values = values[np.isfinite(values).all(axis=1)]
        if len(values) == 0:
            raise ValueError(f"no finite rows found for column {column!r}")
        tensor = torch.from_numpy(values)
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0) if len(values) > 1 else torch.zeros_like(mean)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return mean.numpy(), std.numpy()

    def get_dim(self, column: str) -> int:
        data = self.get_col_data(column)
        return int(np.prod(data.shape[1:])) if data.ndim > 1 else 1

    def get_row_data(self, row_idx: int | list[int]) -> dict[str, np.ndarray]:
        self._open()
        return {key: self.h5_file[key][row_idx] for key in self._keys}


__all__ = ["AlignedControlHDF5Dataset"]
