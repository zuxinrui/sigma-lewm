"""Strict macro-aligned HDF5 reader for PlanarMorphPush-v2.

The generic StableWorldModel HDF5 reader assumes one action row for every
observation row and enumerates every raw-tick phase as a valid clip start.
That contract is convenient for dense control logs, but it is wrong for the
v2 audit dataset:

* an episode contains all 26 physical states ``s_0, ..., s_25``;
* only rows 0--24 have actions, each mapping ``s_t -> s_{t+1}``;
* row 25 is a real terminal state and has no outgoing action;
* LeWM transitions are five controller ticks, so starts must be macro aligned.

This reader constructs samples directly from that contract.  It never feeds
the terminal NaN action to a model and never invents a dummy transition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch

import stable_worldmodel as swm


V2_SCHEMA_VERSION = "lewm-planar-morph-push-rgb-v2.1"
V2_TASK_ID = "PlanarMorphPush-v2-VisualRoller"
V2_SUCCESS_TOLERANCE_M = 0.015
FORBIDDEN_MODEL_INPUT_COLUMNS = frozenset(
    {
        "behavior_mode",
        "episode_contact",
        "episode_legal",
        "episode_success",
        "goal",
        "morphology_id",
        "morphology_name",
        "puck",
        "scene_id",
    }
)


class AlignedMacroHDF5Dataset(torch.utils.data.Dataset):
    """Read exact macro transitions from a PlanarMorphPush-v2 HDF5 file.

    Args:
        name: Dataset filename or absolute path.
        cache_dir: StableWorldModel cache root.  ``name`` is resolved below
            ``<cache_dir>/datasets`` when it is not an existing/absolute path.
        transitions: Number of consecutive five-tick transitions per sample.
            ``5`` gives the full h=5 sequence: six states and five actions.
        frameskip: Raw controller ticks per macro transition. Frozen to five.
        keys_to_load: Observation/context columns to return. ``action`` is
            always returned because it defines the supervised transition.
        transform: Optional dict-in/dict-out sample transform.
    """

    def __init__(
        self,
        *,
        name: str,
        cache_dir: str | Path | None = None,
        transitions: int = 5,
        frameskip: int = 5,
        keys_to_load: list[str] | None = None,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        super().__init__()
        if frameskip != 5:
            raise ValueError("PlanarMorphPush-v2 is frozen to frameskip=5")
        if not 1 <= transitions <= 5:
            raise ValueError("transitions must lie in [1,5]")
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
        self.frameskip = int(frameskip)
        self.transitions = int(transitions)
        self.transform = transform
        self.h5_file: h5py.File | None = None
        self._cache: dict[str, np.ndarray] = {}
        requested = keys_to_load or [
            "pixels",
            "action",
            "morph_nodes",
            "joint_mask",
            "q",
            "dt",
        ]
        self._keys = list(dict.fromkeys(requested))
        if "action" not in self._keys:
            self._keys.append("action")
        leaked = FORBIDDEN_MODEL_INPUT_COLUMNS.intersection(self._keys)
        if leaked:
            raise ValueError(
                "forbidden VisualRoller model-input columns requested: "
                + ", ".join(sorted(leaked))
            )

        with h5py.File(self.h5_path, "r") as source:
            schema = source.attrs.get("schema_version", "")
            if isinstance(schema, bytes):
                schema = schema.decode("utf-8")
            if schema != V2_SCHEMA_VERSION:
                raise ValueError(
                    f"{self.h5_path} has schema {schema!r}, expected "
                    f"{V2_SCHEMA_VERSION!r}"
                )
            task = source.attrs.get("task", "")
            dynamics = source.attrs.get("dynamics", "")
            if isinstance(task, bytes):
                task = task.decode("utf-8")
            if isinstance(dynamics, bytes):
                dynamics = dynamics.decode("utf-8")
            if task != V2_TASK_ID or dynamics != "VisualAffordancePush":
                raise ValueError(
                    "HDF5 is not calibrated VisualRoller v2 data: "
                    f"task={task!r}, dynamics={dynamics!r}"
                )
            tolerance = float(
                source.attrs.get("success_tolerance_m", np.nan)
            )
            if (
                not np.isfinite(tolerance)
                or abs(tolerance - V2_SUCCESS_TOLERANCE_M) > 1e-12
            ):
                raise ValueError(
                    "HDF5 success tolerance is not preregistered 0.015 m"
                )
            if int(source.attrs.get("frameskip", -1)) != self.frameskip:
                raise ValueError("HDF5 frameskip disagrees with v2 reader")
            self.lengths = source["ep_len"][:].astype(np.int64)
            self.offsets = source["ep_offset"][:].astype(np.int64)
            missing = set(self._keys) - set(source.keys())
            if missing:
                raise KeyError(f"HDF5 columns missing: {sorted(missing)}")
            if "action_valid" not in source:
                raise KeyError("v2 HDF5 requires action_valid")
            self._validate_file_contract(source)
            self.morphology_names = tuple(
                sorted(
                    value.decode("utf-8")
                    if isinstance(value, bytes)
                    else str(value)
                    for value in np.unique(source["morphology_name"][:])
                )
            )

        # ``clip_indices`` remains episode/start based for compatibility with
        # audit utilities, but every start is an exact macro boundary.
        required_ticks = self.transitions * self.frameskip
        self.clip_indices = [
            (episode, start)
            for episode, length in enumerate(self.lengths)
            for start in range(
                0,
                int(length) - required_ticks,
                self.frameskip,
            )
            if start + required_ticks <= int(length) - 1
        ]
        if not self.clip_indices:
            raise ValueError("dataset has no valid aligned macro clips")
        if any(start % self.frameskip for _, start in self.clip_indices):
            raise AssertionError("non-aligned clip escaped construction")

    def _validate_file_contract(self, source: h5py.File) -> None:
        for episode, (offset, length) in enumerate(
            zip(self.offsets, self.lengths)
        ):
            if int(length) != 26:
                raise ValueError(
                    f"episode {episode} has {length} rows; v2 requires 26 "
                    "physical states s_0..s_25"
                )
            action_valid = source["action_valid"][
                int(offset) : int(offset + length)
            ].astype(bool)
            if not bool(action_valid[:25].all()) or bool(action_valid[25]):
                raise ValueError(
                    f"episode {episode} action_valid must be true on rows "
                    "0..24 and false on terminal row 25"
                )
            terminal_action = source["action"][int(offset + 25)]
            if not bool(np.isnan(terminal_action).all()):
                raise ValueError(
                    f"episode {episode} terminal action must be explicit NaN"
                )
            if not bool(
                np.isfinite(
                    source["action"][int(offset) : int(offset + 25)]
                ).all()
            ):
                raise ValueError(
                    f"episode {episode} has non-finite real transition action"
                )
            if "step_index" not in source:
                raise KeyError("v2 HDF5 requires step_index")
            step_index = source["step_index"][
                int(offset) : int(offset + length)
            ]
            if not np.array_equal(
                step_index,
                np.arange(26, dtype=step_index.dtype),
            ):
                raise ValueError(
                    f"episode {episode} step_index must be exactly 0..25"
                )

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

    def __len__(self) -> int:
        return len(self.clip_indices)

    @property
    def column_names(self) -> list[str]:
        return list(self._keys)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self._open()
        episode, local_start = self.clip_indices[index]
        offset = int(self.offsets[episode])
        start = offset + local_start
        observation_rows = start + np.arange(
            self.transitions + 1, dtype=np.int64
        ) * self.frameskip

        sample: dict[str, torch.Tensor] = {}
        for key in self._keys:
            if key == "action":
                blocks = [
                    self.h5_file["action"][
                        start + step * self.frameskip :
                        start + (step + 1) * self.frameskip
                    ]
                    for step in range(self.transitions)
                ]
                value = np.stack(blocks).reshape(self.transitions, -1)
            else:
                value = self.h5_file[key][observation_rows]
            if value.dtype.kind in ("S", "U", "O"):
                raise TypeError(
                    f"string column {key!r} is metadata, not model input"
                )
            tensor = torch.from_numpy(np.asarray(value))
            if (
                key.startswith("pixels")
                and tensor.ndim == 4
                and tensor.shape[-1] in (1, 3)
            ):
                tensor = tensor.permute(0, 3, 1, 2)
            sample[key] = tensor

        if sample["pixels"].shape[0] != self.transitions + 1:
            raise AssertionError("state sequence length changed")
        if sample["action"].shape[0] != self.transitions:
            raise AssertionError("transition action sequence length changed")
        if not bool(torch.isfinite(sample["action"]).all()):
            raise AssertionError("terminal/non-finite action leaked into sample")
        return self.transform(sample) if self.transform else sample

    def get_col_data(self, column: str) -> np.ndarray:
        self._open()
        if column not in self.h5_file:
            raise KeyError(column)
        return self.h5_file[column][:]

    def get_dim(self, column: str) -> int:
        data = self.get_col_data(column)
        return int(np.prod(data.shape[1:])) if data.ndim > 1 else 1


def assert_disjoint_morphologies(
    train: AlignedMacroHDF5Dataset,
    validation: AlignedMacroHDF5Dataset,
) -> None:
    """Fail closed if validation robot identities entered the training file."""

    overlap = set(train.morphology_names) & set(validation.morphology_names)
    if overlap:
        raise ValueError(
            "training/validation morphology identities overlap: "
            + ", ".join(sorted(overlap))
        )
    logging.info(
        "Verified morphology-disjoint v2 split: %d train, %d validation",
        len(train.morphology_names),
        len(validation.morphology_names),
    )


__all__ = [
    "AlignedMacroHDF5Dataset",
    "FORBIDDEN_MODEL_INPUT_COLUMNS",
    "V2_SCHEMA_VERSION",
    "V2_SUCCESS_TOLERANCE_M",
    "V2_TASK_ID",
    "assert_disjoint_morphologies",
]
