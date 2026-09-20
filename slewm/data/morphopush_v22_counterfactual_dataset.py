"""Isolated aligned reader for PlanarMorphPush-v2.2 counterfactual pairs.

This file deliberately does not modify or wrap the frozen v2.1 reader:
checkpoint audit sidecars bind ``morphopush_v2_dataset.py`` by SHA-256.  V2.2
therefore owns a separate schema, reader, and source hash.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Sampler

import stable_worldmodel as swm


V22_COUNTERFACTUAL_SCHEMA_VERSION = (
    "lewm-planar-morph-push-rgb-v2.2-counterfactual"
)
V2_TASK_ID = "PlanarMorphPush-v2-VisualRoller"
V2_SUCCESS_TOLERANCE_M = 0.015
FORBIDDEN_MODEL_INPUT_COLUMNS = frozenset(
    {
        "anchor_id",
        "behavior_mode",
        "branch_id",
        "branch_role",
        "candidate_name",
        "candidate_seed",
        "episode_contact",
        "episode_legal",
        "episode_success",
        "goal",
        "morphology_id",
        "morphology_name",
        "pair_action_rms",
        "pair_endpoint_separation_m",
        "planner_support",
        "puck",
        "scene_id",
    }
)


class AlignedCounterfactualMacroHDF5Dataset(
    torch.utils.data.Dataset
):
    """Read v2.2 pairs while returning only ordinary LeWM tensors."""

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
            raise ValueError("PlanarMorphPush-v2.2 is frozen to frameskip=5")
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
                "forbidden VisualRoller-v2.2 model-input columns requested: "
                + ", ".join(sorted(leaked))
            )

        with h5py.File(self.h5_path, "r") as source:
            schema = self._decode(
                source.attrs.get("schema_version", "")
            )
            if schema != V22_COUNTERFACTUAL_SCHEMA_VERSION:
                raise ValueError(
                    f"{self.h5_path} has schema {schema!r}, expected "
                    f"{V22_COUNTERFACTUAL_SCHEMA_VERSION!r}"
                )
            task = self._decode(source.attrs.get("task", ""))
            dynamics = self._decode(
                source.attrs.get("dynamics", "")
            )
            if task != V2_TASK_ID or dynamics != "VisualAffordancePush":
                raise ValueError(
                    "HDF5 is not calibrated VisualRoller v2.2 data"
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
                raise ValueError(
                    "HDF5 frameskip disagrees with v2.2 reader"
                )
            self.lengths = source["ep_len"][:].astype(np.int64)
            self.offsets = source["ep_offset"][:].astype(np.int64)
            missing = set(self._keys) - set(source.keys())
            if missing:
                raise KeyError(
                    f"HDF5 columns missing: {sorted(missing)}"
                )
            self._validate_transition_contract(source)
            self._validate_pair_contract(source)
            self.morphology_names = tuple(
                sorted(
                    self._decode(value)
                    for value in np.unique(
                        source["morphology_name"][:]
                    )
                )
            )

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
        if self.transitions == 5:
            clip_by_episode = {
                episode: index
                for index, (episode, local_start) in enumerate(
                    self.clip_indices
                )
                if local_start == 0
            }
            if len(clip_by_episode) != len(self.offsets):
                raise ValueError(
                    "v2.2 paired training requires one h=5 clip per episode"
                )
            self.anchor_pairs = tuple(
                (
                    clip_by_episode[positive_episode],
                    clip_by_episode[counterfactual_episode],
                )
                for positive_episode, counterfactual_episode in (
                    self.episode_pairs
                )
            )
        else:
            self.anchor_pairs = ()

    @staticmethod
    def _decode(value: object) -> str:
        return (
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
        )

    def _validate_transition_contract(self, source: h5py.File) -> None:
        if "action_valid" not in source or "step_index" not in source:
            raise KeyError(
                "v2.2 requires action_valid and step_index"
            )
        for episode, (offset, length) in enumerate(
            zip(self.offsets, self.lengths)
        ):
            start = int(offset)
            length = int(length)
            if length != 26:
                raise ValueError(
                    f"episode {episode} has {length} rows; expected 26"
                )
            action_valid = source["action_valid"][
                start : start + length
            ].astype(bool)
            if not bool(action_valid[:25].all()) or bool(
                action_valid[25]
            ):
                raise ValueError(
                    f"episode {episode} action_valid contract changed"
                )
            if not bool(
                np.isfinite(source["action"][start : start + 25]).all()
            ) or not bool(
                np.isnan(source["action"][start + 25]).all()
            ):
                raise ValueError(
                    f"episode {episode} terminal action contract changed"
                )
            step_index = source["step_index"][
                start : start + length
            ]
            if not np.array_equal(
                step_index,
                np.arange(26, dtype=step_index.dtype),
            ):
                raise ValueError(
                    f"episode {episode} step_index must be 0..25"
                )

    def _validate_pair_contract(self, source: h5py.File) -> None:
        required = {
            "anchor_id",
            "branch_id",
            "branch_role",
            "candidate_name",
            "candidate_seed",
            "episode_contact",
            "episode_legal",
            "goal",
            "joint_mask",
            "morph_nodes",
            "morphology_id",
            "morphology_name",
            "pair_action_rms",
            "pair_endpoint_separation_m",
            "planner_support",
            "puck",
        }
        missing = required - set(source)
        if missing:
            raise KeyError(
                "v2.2 counterfactual metadata missing: "
                + ", ".join(sorted(missing))
            )
        if (
            int(
                source.attrs.get(
                    "counterfactual_branches_per_anchor", -1
                )
            )
            != 2
        ):
            raise ValueError(
                "v2.2 requires exactly two branches per anchor"
            )

        groups: dict[int, list[int]] = {}
        for episode, offset in enumerate(self.offsets):
            start = int(offset)
            length = int(self.lengths[episode])
            for key in (
                "anchor_id",
                "branch_id",
                "branch_role",
                "candidate_name",
                "candidate_seed",
                "pair_action_rms",
                "pair_endpoint_separation_m",
                "planner_support",
            ):
                rows = source[key][start : start + length]
                if not np.all(rows == rows[0]):
                    raise ValueError(
                        f"episode {episode} metadata {key} is not constant"
                    )
            groups.setdefault(
                int(source["anchor_id"][start]), []
            ).append(episode)
        if len(groups) * 2 != len(self.offsets):
            raise ValueError(
                "v2.2 anchor/episode cardinality is not exactly two"
            )

        initial_keys = (
            "pixels",
            "morph_nodes",
            "joint_mask",
            "q",
            "puck",
            "goal",
            "dt",
            "morphology_id",
            "morphology_name",
        )
        episode_pairs: dict[int, tuple[int, int]] = {}
        roles_by_morphology: dict[str, set[str]] = {}
        for anchor_id, episodes in groups.items():
            if len(episodes) != 2:
                raise ValueError(
                    f"anchor {anchor_id} does not have two branches"
                )
            branch_to_episode: dict[int, int] = {}
            for episode in episodes:
                start = int(self.offsets[episode])
                branch_id = int(source["branch_id"][start])
                if branch_id in branch_to_episode:
                    raise ValueError(
                        f"anchor {anchor_id} repeats branch {branch_id}"
                    )
                branch_to_episode[branch_id] = episode
            if set(branch_to_episode) != {0, 1}:
                raise ValueError(
                    f"anchor {anchor_id} branch IDs must be 0 and 1"
                )
            positive_start = int(
                self.offsets[branch_to_episode[0]]
            )
            counterfactual_start = int(
                self.offsets[branch_to_episode[1]]
            )
            episode_pairs[anchor_id] = (
                branch_to_episode[0],
                branch_to_episode[1],
            )
            morphology_name = self._decode(
                source["morphology_name"][positive_start]
            )
            if self._decode(
                source["branch_role"][positive_start]
            ) != "positive":
                raise ValueError(
                    f"anchor {anchor_id} branch 0 is not positive"
                )
            counterfactual_role = self._decode(
                source["branch_role"][counterfactual_start]
            )
            if counterfactual_role not in {
                "contact_divergent",
                "no_contact",
            }:
                raise ValueError(
                    f"anchor {anchor_id} has invalid counterfactual role"
                )
            roles_by_morphology.setdefault(
                morphology_name, set()
            ).add(counterfactual_role)
            for key in initial_keys:
                if not np.array_equal(
                    source[key][positive_start],
                    source[key][counterfactual_start],
                ):
                    raise ValueError(
                        f"anchor {anchor_id} row-0 {key} differs"
                    )
            if not (
                bool(source["episode_legal"][positive_start])
                and bool(
                    source["episode_legal"][counterfactual_start]
                )
            ):
                raise ValueError(
                    f"anchor {anchor_id} contains an illegal branch"
                )
            if not bool(source["episode_contact"][positive_start]):
                raise ValueError(
                    f"anchor {anchor_id} positive branch lacks contact"
                )
            expected_contact = (
                counterfactual_role == "contact_divergent"
            )
            if (
                bool(
                    source["episode_contact"][
                        counterfactual_start
                    ]
                )
                != expected_contact
            ):
                raise ValueError(
                    f"anchor {anchor_id} role/contact mismatch"
                )

            active = source["joint_mask"][positive_start].astype(bool)
            if active.ndim != 1 or not bool(active.any()):
                raise ValueError(
                    f"anchor {anchor_id} has invalid joint mask"
                )
            positive_actions = source["action"][
                positive_start : positive_start + 25, active
            ].astype(np.float64)
            counterfactual_actions = source["action"][
                counterfactual_start :
                counterfactual_start + 25,
                active,
            ].astype(np.float64)
            measured_action_rms = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            positive_actions
                            - counterfactual_actions
                        )
                    )
                )
            )
            declared_action_rms = float(
                source["pair_action_rms"][positive_start]
            )
            minimum_action_rms = (
                0.150
                if counterfactual_role == "no_contact"
                else 0.030
            )
            if (
                not np.isfinite(declared_action_rms)
                or abs(measured_action_rms - declared_action_rms)
                > 2e-6
                or declared_action_rms + 1e-12 < minimum_action_rms
            ):
                raise ValueError(
                    f"anchor {anchor_id} action-RMS contract changed"
                )

            measured_separation = float(
                np.linalg.norm(
                    source["puck"][positive_start + 25].astype(
                        np.float64
                    )
                    - source["puck"][
                        counterfactual_start + 25
                    ].astype(np.float64)
                )
            )
            declared_separation = float(
                source["pair_endpoint_separation_m"][
                    positive_start
                ]
            )
            minimum_separation = (
                0.008
                if counterfactual_role == "no_contact"
                else 0.003
            )
            if (
                not np.isfinite(declared_separation)
                or abs(measured_separation - declared_separation)
                > 2e-6
                or declared_separation + 1e-12 < minimum_separation
            ):
                raise ValueError(
                    f"anchor {anchor_id} endpoint-separation contract changed"
                )
            positive_displacement = float(
                np.linalg.norm(
                    source["puck"][positive_start + 25].astype(
                        np.float64
                    )
                    - source["puck"][positive_start].astype(
                        np.float64
                    )
                )
            )
            counterfactual_displacement = float(
                np.linalg.norm(
                    source["puck"][
                        counterfactual_start + 25
                    ].astype(np.float64)
                    - source["puck"][counterfactual_start].astype(
                        np.float64
                    )
                )
            )
            if positive_displacement + 1e-12 < 0.008:
                raise ValueError(
                    f"anchor {anchor_id} positive puck displacement changed"
                )
            if (
                counterfactual_role == "contact_divergent"
                and counterfactual_displacement + 1e-12 < 0.003
            ):
                raise ValueError(
                    f"anchor {anchor_id} counterfactual puck displacement changed"
                )
            if counterfactual_role == "no_contact":
                nodes = source["morph_nodes"][
                    counterfactual_start
                ].astype(np.float64)
                q0 = source["q"][counterfactual_start].astype(
                    np.float64
                )
                q1 = source["q"][
                    counterfactual_start + 25
                ].astype(np.float64)

                def end_effector(q: np.ndarray) -> np.ndarray:
                    angles = np.cumsum(nodes[:, 1] * q)
                    offsets = nodes[:, 0, None] * np.stack(
                        (np.cos(angles), np.sin(angles)), axis=-1
                    )
                    offsets *= active[:, None]
                    return offsets.sum(axis=0)

                retreat = float(
                    np.linalg.norm(end_effector(q1) - end_effector(q0))
                )
                if retreat + 1e-12 < 0.020:
                    raise ValueError(
                        f"anchor {anchor_id} no-contact retreat changed"
                    )
        self.episode_pairs = tuple(
            episode_pairs[anchor_id]
            for anchor_id in sorted(episode_pairs)
        )
        if len(groups) >= 5:
            missing_roles = {
                morphology: {
                    "contact_divergent",
                    "no_contact",
                }
                - roles
                for morphology, roles in roles_by_morphology.items()
                if roles
                != {"contact_divergent", "no_contact"}
            }
            if missing_roles:
                raise ValueError(
                    "v2.2 counterfactual role coverage incomplete: "
                    + ", ".join(
                        f"{name} missing {sorted(roles)}"
                        for name, roles in sorted(missing_roles.items())
                    )
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
        start = int(self.offsets[episode]) + local_start
        observation_rows = start + np.arange(
            self.transitions + 1, dtype=np.int64
        ) * self.frameskip
        sample: dict[str, torch.Tensor] = {}
        for key in self._keys:
            if key == "action":
                value = np.stack(
                    [
                        self.h5_file["action"][
                            start + step * self.frameskip :
                            start + (step + 1) * self.frameskip
                        ]
                        for step in range(self.transitions)
                    ]
                ).reshape(self.transitions, -1)
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
        if not bool(torch.isfinite(sample["action"]).all()):
            raise AssertionError(
                "terminal/non-finite action leaked into sample"
            )
        return self.transform(sample) if self.transform else sample

    def get_col_data(self, column: str) -> np.ndarray:
        self._open()
        if column not in self.h5_file:
            raise KeyError(column)
        return self.h5_file[column][:]

    def get_dim(self, column: str) -> int:
        data = self.get_col_data(column)
        return (
            int(np.prod(data.shape[1:]))
            if data.ndim > 1
            else 1
        )


def assert_disjoint_morphologies(
    train: AlignedCounterfactualMacroHDF5Dataset,
    validation: AlignedCounterfactualMacroHDF5Dataset,
) -> None:
    overlap = set(train.morphology_names) & set(
        validation.morphology_names
    )
    if overlap:
        raise ValueError(
            "training/validation morphology identities overlap: "
            + ", ".join(sorted(overlap))
        )
    logging.info(
        "Verified morphology-disjoint v2.2 split: %d train, %d validation",
        len(train.morphology_names),
        len(validation.morphology_names),
    )


class PairedAnchorBatchSampler(Sampler[list[int]]):
    """Shuffle anchors while keeping both branches in the same local batch."""

    def __init__(
        self,
        dataset: AlignedCounterfactualMacroHDF5Dataset,
        *,
        batch_size: int,
        generator: torch.Generator,
        drop_last: bool = True,
    ) -> None:
        if dataset.transitions != 5 or not dataset.anchor_pairs:
            raise ValueError("paired sampler requires the h=5 v2.2 dataset")
        if batch_size < 2 or batch_size % 2:
            raise ValueError("paired sampler batch_size must be positive/even")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.anchors_per_batch = self.batch_size // 2
        self.generator = generator
        self.drop_last = bool(drop_last)

    def __len__(self) -> int:
        anchors = len(self.dataset.anchor_pairs)
        if self.drop_last:
            return anchors // self.anchors_per_batch
        return (
            anchors + self.anchors_per_batch - 1
        ) // self.anchors_per_batch

    def __iter__(self):
        pair_count = len(self.dataset.anchor_pairs)
        order = torch.randperm(
            pair_count, generator=self.generator
        ).tolist()
        for start in range(0, pair_count, self.anchors_per_batch):
            selected = order[start : start + self.anchors_per_batch]
            if self.drop_last and len(selected) < self.anchors_per_batch:
                continue
            batch: list[int] = []
            for pair_index in selected:
                left, right = self.dataset.anchor_pairs[pair_index]
                if bool(
                    torch.randint(
                        0,
                        2,
                        (1,),
                        generator=self.generator,
                    ).item()
                ):
                    left, right = right, left
                batch.extend((left, right))
            yield batch


__all__ = [
    "AlignedCounterfactualMacroHDF5Dataset",
    "FORBIDDEN_MODEL_INPUT_COLUMNS",
    "PairedAnchorBatchSampler",
    "V22_COUNTERFACTUAL_SCHEMA_VERSION",
    "V2_SUCCESS_TOLERANCE_M",
    "V2_TASK_ID",
    "assert_disjoint_morphologies",
]
