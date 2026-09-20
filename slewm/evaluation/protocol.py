"""Build immutable, unique-episode evaluation banks for SimpleTT experiments."""

from __future__ import annotations

from slewm.paths import REPO_ROOT

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from slewm.evaluation.inloop import _sample_eval_rows
from slewm.protocols.demo_goal_formal_specs import (
    DEMO_GOAL_BANK_SEED_NAMESPACES,
    DEMO_GOAL_RESERVED_EPISODES_PER_TASK,
    DEMO_GOAL_STUDY_IDS,
)
from slewm.protocols.formal_training_completion import (
    EPISODE_SPLIT_SCHEMA_VERSION,
    is_sha256,
)
from slewm.training.utilities import (
    git_tracked_tree_sha256,
    runtime_environment_compatibility_fingerprint,
    runtime_environment_fingerprint,
)


FORMAL_EVAL_MANIFEST_SCHEMA_VERSION = 3
FORMAL_EVAL_RESULT_SCHEMA_VERSION = 4

FORMAL_TASK_PROTOCOL_SPECS = {
    "pusht": (
        {
            "name": "primary_fc_h10_o25",
            "horizon": 10,
            "goal_offset": 25,
            "receding_horizon": 10,
            "action_block": 5,
            "eval_budget": 100,
        },
        {
            "name": "secondary_stock_h5_o25",
            "horizon": 5,
            "goal_offset": 25,
            "receding_horizon": 5,
            "action_block": 5,
            "eval_budget": 50,
        },
    ),
    "tworoom": (
        {
            "name": "primary_stock_h5_o25",
            "horizon": 5,
            "goal_offset": 25,
            "receding_horizon": 5,
            "action_block": 5,
            "eval_budget": 50,
        },
        {
            "name": "secondary_fc_h10_o25",
            "horizon": 10,
            "goal_offset": 25,
            "receding_horizon": 10,
            "action_block": 5,
            "eval_budget": 100,
        },
    ),
}
FORMAL_TASK_PROTOCOL_NAMES = {
    task: tuple(spec["name"] for spec in specs)
    for task, specs in FORMAL_TASK_PROTOCOL_SPECS.items()
}
FORMAL_BANK_SEED_NAMESPACES = {
    "discovery": "lewm-simplett-discovery-v1",
    "confirmation": "lewm-simplett-confirmation-v1",
}


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_task_protocols(task: str, protocols) -> tuple[str, str]:
    """Require the exact, unique primary/secondary protocol pair for a task."""
    if task not in FORMAL_TASK_PROTOCOL_NAMES:
        raise RuntimeError(f"unknown formal evaluation task: {task!r}")
    if not isinstance(protocols, list):
        raise RuntimeError("formal manifest protocols must be a list")
    names = []
    for index, protocol in enumerate(protocols):
        if not isinstance(protocol, dict) or not isinstance(protocol.get("name"), str):
            raise RuntimeError(
                f"formal protocol at index {index} lacks a string name"
            )
        names.append(protocol["name"])
    if len(names) != len(set(names)):
        raise RuntimeError("formal manifest contains duplicate protocol names")
    expected = FORMAL_TASK_PROTOCOL_NAMES[task]
    if set(names) != set(expected) or len(names) != len(expected):
        raise RuntimeError(
            f"formal protocols for task={task!r} must be exactly {list(expected)}; "
            f"received {names}"
        )
    by_name = {protocol["name"]: protocol for protocol in protocols}
    for frozen in FORMAL_TASK_PROTOCOL_SPECS[task]:
        actual = by_name[frozen["name"]]
        mismatches = {
            field: (actual.get(field), wanted)
            for field, wanted in frozen.items()
            if field != "name"
            and (
                not isinstance(actual.get(field), int)
                or isinstance(actual.get(field), bool)
                or actual.get(field) != wanted
            )
        }
        if mismatches:
            raise RuntimeError(
                f"formal protocol spec differs for task={task!r} "
                f"name={frozen['name']!r}: {mismatches}"
            )
    return expected


def _validate_binary_episode_successes(
    values, *, expected_count: int, context: str = "episode_successes"
) -> np.ndarray:
    """Validate binary outcomes without permitting lossy dtype coercion."""
    outcomes = np.asarray(values)
    expected_shape = (int(expected_count),)
    if outcomes.shape != expected_shape:
        raise RuntimeError(
            f"{context} must have shape={expected_shape}; received {outcomes.shape}"
        )
    if outcomes.dtype.kind == "b":
        return outcomes.astype(np.int64, copy=False)
    if outcomes.dtype.kind not in "iuf":
        raise RuntimeError(
            f"{context} must contain numeric 0/1 values or bools; "
            f"received dtype={outcomes.dtype}"
        )
    if outcomes.dtype.kind == "f" and not np.isfinite(outcomes).all():
        raise RuntimeError(f"{context} contains NaN or Inf")
    if not np.logical_or(outcomes == 0, outcomes == 1).all():
        raise RuntimeError(f"{context} must contain only strict binary 0/1 values")
    return outcomes.astype(np.int64, copy=False)


def _derived_seeds(namespace: str, count: int) -> list[int]:
    return [
        int.from_bytes(
            hashlib.sha256(f"{namespace}:{index}".encode()).digest()[:4],
            "little",
        )
        for index in range(int(count))
    ]


def _validate_formal_bank_structure(
    payload,
    *,
    expected_discovery_banks: int = 3,
    expected_confirmation_banks: int = 5,
    expected_num_eval: int = 50,
    expected_goal_offset: int = 25,
):
    """Validate the complete deterministic, disjoint formal-bank contract."""
    expected_values = {
        "expected_discovery_banks": expected_discovery_banks,
        "expected_confirmation_banks": expected_confirmation_banks,
        "expected_num_eval": expected_num_eval,
        "expected_goal_offset": expected_goal_offset,
    }
    for name, value in expected_values.items():
        if type(value) is not int:
            raise RuntimeError(f"{name} must be a strict integer")
    if expected_discovery_banks < 0 or expected_confirmation_banks < 0:
        raise RuntimeError("expected formal bank counts cannot be negative")
    if expected_num_eval <= 0:
        raise RuntimeError("expected_num_eval must be positive")

    if not isinstance(payload, dict):
        raise RuntimeError("formal manifest payload must be a dictionary")
    if type(payload.get("schema_version")) is not int or payload.get(
        "schema_version"
    ) != FORMAL_EVAL_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            "formal bank structure requires schema_version="
            f"{FORMAL_EVAL_MANIFEST_SCHEMA_VERSION}"
        )
    if type(payload.get("num_eval_per_bank")) is not int:
        raise RuntimeError("num_eval_per_bank must be a strict integer")
    if payload["num_eval_per_bank"] != expected_num_eval:
        raise RuntimeError(
            "num_eval_per_bank differs from the formal contract: "
            f"expected={expected_num_eval}, actual={payload['num_eval_per_bank']}"
        )
    if type(payload.get("goal_offset")) is not int:
        raise RuntimeError("goal_offset must be a strict integer")
    if payload["goal_offset"] != expected_goal_offset:
        raise RuntimeError(
            "goal_offset differs from the formal contract: "
            f"expected={expected_goal_offset}, actual={payload['goal_offset']}"
        )

    study_id = payload.get("study_id")
    if study_id is None:
        expected_namespaces = FORMAL_BANK_SEED_NAMESPACES
    elif study_id in set(DEMO_GOAL_STUDY_IDS.values()):
        task = payload.get("task")
        if DEMO_GOAL_STUDY_IDS.get(task) != study_id:
            raise RuntimeError(
                "Demo-Goal task/study_id mapping mismatch: "
                f"task={task!r} study_id={study_id!r}"
            )
        expected_namespaces = DEMO_GOAL_BANK_SEED_NAMESPACES
        if payload.get("bank_seed_namespaces") != expected_namespaces:
            raise RuntimeError(
                "Demo-Goal manifest bank namespaces differ from the locked study"
            )
        exclusions = payload.get("cross_study_exclusions")
        if not isinstance(exclusions, list) or len(exclusions) != 1:
            raise RuntimeError(
                "Demo-Goal manifest requires exactly one complete Delta exclusion manifest"
            )
        excluded_ids = payload.get("cross_study_excluded_episode_ids")
        if (
            not isinstance(excluded_ids, list)
            or any(type(value) is not int for value in excluded_ids)
            or excluded_ids != sorted(set(excluded_ids))
        ):
            raise RuntimeError(
                "Demo-Goal excluded episode IDs must be a sorted unique integer list"
            )
        if len(excluded_ids) != int(
            payload.get("cross_study_excluded_episode_count", -1)
        ):
            raise RuntimeError("Demo-Goal excluded episode count is inconsistent")
        if len(excluded_ids) != DEMO_GOAL_RESERVED_EPISODES_PER_TASK:
            raise RuntimeError(
                "Demo-Goal must exclude exactly 400 legacy Delta episodes"
            )
        if _canonical_sha256(excluded_ids) != payload.get(
            "cross_study_excluded_episode_ids_sha256"
        ):
            raise RuntimeError("Demo-Goal excluded episode ID hash is invalid")
        if payload.get("cross_study_episode_disjoint") is not True:
            raise RuntimeError("Demo-Goal manifest lacks cross-study disjointness")
        exclusion = exclusions[0]
        expected_exclusion_keys = {
            "manifest_path",
            "manifest_file_sha256",
            "manifest_sha256",
            "episode_ids",
            "episode_ids_sha256",
            "episode_count",
        }
        if not isinstance(exclusion, dict) or set(exclusion) != expected_exclusion_keys:
            raise RuntimeError("Demo-Goal Delta exclusion binding is malformed")
        if exclusion.get("episode_ids") != excluded_ids:
            raise RuntimeError("Demo-Goal Delta exclusion list differs from global list")
        if exclusion.get("episode_count") != len(excluded_ids):
            raise RuntimeError("Demo-Goal Delta exclusion binding count is invalid")
        if exclusion.get("episode_ids_sha256") != _canonical_sha256(excluded_ids):
            raise RuntimeError("Demo-Goal Delta exclusion binding hash is invalid")
        for key in ("manifest_file_sha256", "manifest_sha256"):
            value = exclusion.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RuntimeError(f"Demo-Goal exclusion {key} is not SHA256")
        if not isinstance(exclusion.get("manifest_path"), str):
            raise RuntimeError("Demo-Goal exclusion manifest_path must be a string")
    else:
        raise RuntimeError(f"unknown formal study_id: {study_id!r}")

    banks = payload.get("banks")
    expected_groups = {
        "discovery": expected_discovery_banks,
        "confirmation": expected_confirmation_banks,
    }
    if not isinstance(banks, dict) or set(banks) != set(expected_groups):
        actual_groups = sorted(banks) if isinstance(banks, dict) else type(banks).__name__
        raise RuntimeError(
            "formal banks must contain exactly discovery and confirmation groups; "
            f"received={actual_groups}"
        )

    globally_used_episodes: set[int] = set()
    vector_fields = ("row_indices", "episode_ids", "start_steps", "goal_steps")
    for group_name, expected_bank_count in expected_groups.items():
        group_banks = banks[group_name]
        if not isinstance(group_banks, list):
            raise RuntimeError(f"formal banks[{group_name!r}] must be a list")
        if len(group_banks) != expected_bank_count:
            raise RuntimeError(
                f"formal banks[{group_name!r}] must contain exactly "
                f"{expected_bank_count} banks; received={len(group_banks)}"
            )
        expected_seeds = _derived_seeds(
            expected_namespaces[group_name], expected_bank_count
        )
        for expected_index, bank in enumerate(group_banks):
            context = f"banks[{group_name!r}][{expected_index}]"
            if not isinstance(bank, dict):
                raise RuntimeError(f"{context} must be a dictionary")
            if type(bank.get("bank_index")) is not int:
                raise RuntimeError(f"{context}.bank_index must be a strict integer")
            if bank["bank_index"] != expected_index:
                raise RuntimeError(
                    f"{context}.bank_index must equal its contiguous list index "
                    f"{expected_index}; received={bank['bank_index']}"
                )
            if type(bank.get("seed")) is not int:
                raise RuntimeError(f"{context}.seed must be a strict integer")
            if bank["seed"] != expected_seeds[expected_index]:
                raise RuntimeError(
                    f"{context}.seed differs from the derived formal seed: "
                    f"expected={expected_seeds[expected_index]}, actual={bank['seed']}"
                )

            vectors = {}
            for field in vector_fields:
                values = bank.get(field)
                if not isinstance(values, list):
                    raise RuntimeError(f"{context}.{field} must be a list")
                if len(values) != expected_num_eval:
                    raise RuntimeError(
                        f"{context}.{field} must contain exactly {expected_num_eval} "
                        f"entries; received={len(values)}"
                    )
                if any(type(value) is not int for value in values):
                    raise RuntimeError(
                        f"{context}.{field} must contain only strict integers"
                    )
                vectors[field] = values

            rows = vectors["row_indices"]
            if any(left >= right for left, right in zip(rows, rows[1:])):
                raise RuntimeError(
                    f"{context}.row_indices must be strictly increasing and unique"
                )
            episodes = vectors["episode_ids"]
            if len(set(episodes)) != expected_num_eval:
                raise RuntimeError(f"{context}.episode_ids must contain unique episodes")
            overlap = globally_used_episodes.intersection(episodes)
            if overlap:
                raise RuntimeError(
                    f"{context}.episode_ids violate cross-bank episode disjointness: "
                    f"reused={sorted(overlap)[:5]}"
                )
            starts = vectors["start_steps"]
            goals = vectors["goal_steps"]
            if any(
                goal != start + expected_goal_offset
                for start, goal in zip(starts, goals)
            ):
                raise RuntimeError(
                    f"{context}.goal_steps must equal start_steps + "
                    f"{expected_goal_offset} elementwise"
                )
            globally_used_episodes.update(episodes)

    if study_id in set(DEMO_GOAL_STUDY_IDS.values()):
        excluded = set(payload["cross_study_excluded_episode_ids"])
        overlap = globally_used_episodes.intersection(excluded)
        if overlap:
            raise RuntimeError(
                "Demo-Goal banks overlap the excluded Delta banks: "
                f"{sorted(overlap)[:5]}"
            )
        if len(globally_used_episodes) != DEMO_GOAL_RESERVED_EPISODES_PER_TASK:
            raise RuntimeError(
                "Demo-Goal study must reserve exactly "
                f"{DEMO_GOAL_RESERVED_EPISODES_PER_TASK} episodes per task"
            )


def verify_demo_goal_manifest_external_bindings(payload):
    """Reopen the method lock and legacy manifest bound by a Demo-Goal bank."""
    if payload.get("study_id") not in set(DEMO_GOAL_STUDY_IDS.values()):
        return None
    from slewm.protocols.demo_goal_formal_artifacts import verify_demo_goal_method_lock

    method_lock_path = Path(payload["method_lock_path"])
    lock = verify_demo_goal_method_lock(
        method_lock_path,
        expected_commit=payload.get("created_by_git_commit"),
        expected_tree=payload.get("source_tracked_tree_sha256"),
    )
    if _file_sha256(method_lock_path) != payload.get("method_lock_file_sha256"):
        raise RuntimeError("Demo-Goal method-lock file SHA256 mismatch")
    if lock.get("method_lock_sha256") != payload.get("method_lock_sha256"):
        raise RuntimeError("Demo-Goal manifest binds the wrong method lock")

    binding = payload["cross_study_exclusions"][0]
    exclusion_path = Path(binding["manifest_path"])
    if exclusion_path.is_symlink() or not exclusion_path.is_file():
        raise RuntimeError("Delta exclusion manifest must be a regular file")
    if _file_sha256(exclusion_path) != binding["manifest_file_sha256"]:
        raise RuntimeError("Delta exclusion manifest file SHA256 mismatch")
    exclusion = json.loads(exclusion_path.read_text())
    unsigned = dict(exclusion)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise RuntimeError("Delta exclusion manifest self-hash mismatch")
    if claimed != binding["manifest_sha256"]:
        raise RuntimeError("Demo-Goal manifest binds the wrong Delta manifest")
    if exclusion.get("study_id") is not None:
        raise RuntimeError("cross-study exclusion is not a legacy Delta manifest")
    if exclusion.get("task") != payload.get("task"):
        raise RuntimeError("cross-study exclusion task mismatch")
    if exclusion.get("dataset_file_sha256") != payload.get(
        "dataset_file_sha256"
    ):
        raise RuntimeError("cross-study exclusion dataset mismatch")
    _validate_formal_bank_structure(exclusion)
    episode_ids = sorted({
        int(episode)
        for banks in exclusion["banks"].values()
        for bank in banks
        for episode in bank["episode_ids"]
    })
    if episode_ids != binding["episode_ids"]:
        raise RuntimeError("Delta exclusion episode list differs from its manifest")
    return {"method_lock": lock, "delta_manifest": exclusion}


def _build_disjoint_banks(
    episode_idx,
    step_idx,
    valid_rows,
    *,
    groups,
    num_eval: int,
):
    """Build named banks with one row per episode and no cross-bank overlap."""
    episode_idx = np.asarray(episode_idx)
    step_idx = np.asarray(step_idx)
    remaining = np.asarray(valid_rows, dtype=np.int64)
    used_episodes = set()
    output = {}
    for group_name, seeds in groups.items():
        output[group_name] = []
        for bank_index, seed in enumerate(seeds):
            if used_episodes:
                keep = ~np.isin(episode_idx[remaining], list(used_episodes))
                remaining = remaining[keep]
            picks = _sample_eval_rows(
                episode_idx,
                remaining,
                num_eval=int(num_eval),
                seed=int(seed),
                unique_episodes=True,
            )
            # HDF5 fancy indexing and InLoopEvaluator both consume rows in
            # ascending order.  Freeze that exact execution order so episode
            # metadata and per-episode outcomes cannot become misaligned.
            picks = np.sort(picks)
            episodes = episode_idx[picks].astype(np.int64)
            starts = step_idx[picks].astype(np.int64)
            if len(np.unique(episodes)) != int(num_eval):
                raise RuntimeError("bank construction violated unique-episode contract")
            overlap = used_episodes & set(episodes.tolist())
            if overlap:
                raise RuntimeError(f"bank construction reused episodes: {sorted(overlap)[:5]}")
            used_episodes.update(episodes.tolist())
            output[group_name].append({
                "bank_index": int(bank_index),
                "seed": int(seed),
                "row_indices": picks.astype(np.int64).tolist(),
                "episode_ids": episodes.tolist(),
                "start_steps": starts.tolist(),
            })
    return output


def _dataset_fingerprint(
    dataset_name: str,
    dataset_windows: int,
    lengths,
    offsets,
    size_bytes: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset_name).encode())
    digest.update(str(int(dataset_windows)).encode())
    digest.update(np.asarray(lengths).tobytes())
    digest.update(np.asarray(offsets).tobytes())
    digest.update(str(int(size_bytes)).encode())
    return digest.hexdigest()


def build_formal_eval_manifest(
    *,
    task: str,
    split_manifest_path: Path,
    output_path: Path,
    dataset_path: Path | None = None,
    num_eval: int = 50,
    discovery_banks: int = 3,
    confirmation_banks: int = 5,
    study_family: str = "delta_triangle",
    method_lock_path: Path | None = None,
    exclusion_manifest_paths=(),
):
    repo = REPO_ROOT
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise RuntimeError(
            f"refusing to overwrite formal evaluation manifest: {output_path}"
        )
    split_manifest_path = Path(split_manifest_path).resolve()
    split_manifest = json.loads(split_manifest_path.read_text())
    if split_manifest.get("schema_version") != EPISODE_SPLIT_SCHEMA_VERSION:
        raise RuntimeError(
            "formal evaluation requires episode split schema_version="
            f"{EPISODE_SPLIT_SCHEMA_VERSION}"
        )
    expected_dataset_file_sha256 = split_manifest.get("dataset_file_sha256")
    if not is_sha256(expected_dataset_file_sha256):
        raise RuntimeError(
            "training split manifest lacks a lowercase dataset_file_sha256"
        )
    study_family = str(study_family)
    if study_family not in {"delta_triangle", "demo_goal"}:
        raise ValueError(
            "study_family must be either delta_triangle or demo_goal"
        )
    method_lock = None
    method_lock_file_sha256 = None
    excluded_episode_ids: set[int] = set()
    cross_study_exclusions = []
    if study_family == "delta_triangle":
        if method_lock_path is not None or tuple(exclusion_manifest_paths):
            raise ValueError(
                "legacy Delta manifest does not accept Demo-Goal lock/exclusions"
            )
        study_id = None
        bank_seed_namespaces = FORMAL_BANK_SEED_NAMESPACES
    else:
        from slewm.protocols.demo_goal_formal_artifacts import (
            file_sha256 as demo_file_sha256,
            verify_demo_goal_method_lock,
        )

        study_id = DEMO_GOAL_STUDY_IDS.get(str(task))
        if study_id is None:
            raise ValueError(f"unsupported Demo-Goal task: {task!r}")
        if method_lock_path is None:
            raise RuntimeError("Demo-Goal manifest requires --method-lock")
        method_lock_path = Path(method_lock_path).resolve()
        method_lock = verify_demo_goal_method_lock(
            method_lock_path,
            expected_commit=split_manifest.get("source_git_commit"),
            expected_tree=split_manifest.get("source_tracked_tree_sha256"),
        )
        method_lock_file_sha256 = demo_file_sha256(method_lock_path)
        exclusion_manifest_paths = tuple(exclusion_manifest_paths)
        if len(exclusion_manifest_paths) != 1:
            raise RuntimeError(
                "Demo-Goal manifest requires exactly one complete old Delta manifest"
            )
        for exclusion_path in exclusion_manifest_paths:
            exclusion_path = Path(exclusion_path).resolve()
            exclusion = json.loads(exclusion_path.read_text())
            claimed_hash = exclusion.get("manifest_sha256")
            unsigned = dict(exclusion)
            unsigned.pop("manifest_sha256", None)
            if claimed_hash != _canonical_sha256(unsigned):
                raise RuntimeError(
                    f"Delta exclusion manifest self-hash is invalid: {exclusion_path}"
                )
            if exclusion.get("study_id") is not None:
                raise RuntimeError(
                    "Demo-Goal exclusion must be a frozen legacy Delta manifest"
                )
            if exclusion.get("task") != str(task):
                raise RuntimeError("Delta exclusion manifest task mismatch")
            if exclusion.get("dataset_file_sha256") != expected_dataset_file_sha256:
                raise RuntimeError("Delta exclusion manifest dataset SHA mismatch")
            _validate_formal_bank_structure(exclusion)
            manifest_episode_ids = sorted({
                int(episode)
                for banks in exclusion["banks"].values()
                for bank in banks
                for episode in bank["episode_ids"]
            })
            if len(manifest_episode_ids) != DEMO_GOAL_RESERVED_EPISODES_PER_TASK:
                raise RuntimeError(
                    "legacy Delta exclusion must bind exactly 400 task episodes"
                )
            overlap = excluded_episode_ids.intersection(manifest_episode_ids)
            if overlap:
                raise RuntimeError(
                    "Delta exclusion manifests repeat episodes: "
                    f"{sorted(overlap)[:5]}"
                )
            excluded_episode_ids.update(manifest_episode_ids)
            cross_study_exclusions.append({
                "manifest_path": str(exclusion_path),
                "manifest_file_sha256": _file_sha256(exclusion_path),
                "manifest_sha256": claimed_hash,
                "episode_ids": manifest_episode_ids,
                "episode_ids_sha256": _canonical_sha256(manifest_episode_ids),
                "episode_count": len(manifest_episode_ids),
            })
        bank_seed_namespaces = DEMO_GOAL_BANK_SEED_NAMESPACES
    split_manifest_sha256 = hashlib.sha256(split_manifest_path.read_bytes()).hexdigest()
    if dataset_path is None:
        dataset_path = Path(split_manifest["dataset_path"])
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    cfg_path = repo / "config" / "eval" / f"{task}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"unknown task {task!r}: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    goal_offset = int(cfg.eval.goal_offset_steps)
    validation_episodes = set(
        int(ep) for ep in split_manifest["validation_episode_labels"]
    )

    dataset_stat_before = dataset_path.stat()
    with h5py.File(dataset_path, "r", swmr=True) as handle:
        episode_col = "episode_idx" if "episode_idx" in handle else "ep_idx"
        episode_idx = handle[episode_col][:]
        step_idx = handle["step_idx"][:]
        lengths = handle["ep_len"][:]
        offsets = handle["ep_offset"][:]
    size_bytes = dataset_stat_before.st_size
    fingerprint = _dataset_fingerprint(
        split_manifest["dataset_name"],
        split_manifest["dataset_windows"],
        lengths,
        offsets,
        size_bytes,
    )
    if fingerprint != split_manifest["dataset_fingerprint_sha256"]:
        raise RuntimeError(
            "dataset fingerprint differs from the training split manifest"
        )
    dataset_file_sha256 = _file_sha256(dataset_path)
    if dataset_file_sha256 != expected_dataset_file_sha256:
        raise RuntimeError(
            "evaluation dataset full-file SHA256 differs from training"
        )
    dataset_stat_after = dataset_path.stat()
    if (
        int(dataset_stat_after.st_size) != int(dataset_stat_before.st_size)
        or int(dataset_stat_after.st_mtime_ns)
        != int(dataset_stat_before.st_mtime_ns)
    ):
        raise RuntimeError(
            "evaluation dataset changed while formal banks were being built"
        )

    labels_by_internal_episode = episode_idx[offsets]
    sort_order = np.argsort(labels_by_internal_episode)
    sorted_labels = labels_by_internal_episode[sort_order]
    sorted_max_start = lengths[sort_order] - goal_offset - 1
    positions = np.searchsorted(sorted_labels, episode_idx)
    if np.any(positions >= len(sorted_labels)) or not np.array_equal(
        sorted_labels[positions], episode_idx
    ):
        raise RuntimeError("episode labels cannot be mapped to ep_len/ep_offset")
    max_start_per_row = sorted_max_start[positions]
    valid = np.nonzero(
        (step_idx <= max_start_per_row)
        & np.isin(episode_idx, list(validation_episodes))
        & ~np.isin(episode_idx, list(excluded_episode_ids))
    )[0]

    groups = {
        name: _derived_seeds(namespace, (
            discovery_banks if name == "discovery" else confirmation_banks
        ))
        for name, namespace in bank_seed_namespaces.items()
    }
    banks = _build_disjoint_banks(
        episode_idx,
        step_idx,
        valid,
        groups=groups,
        num_eval=num_eval,
    )
    for group_banks in banks.values():
        for bank in group_banks:
            bank["goal_steps"] = [
                int(start) + goal_offset for start in bank["start_steps"]
            ]

    solver_path = repo / "config" / "eval" / "solver" / "cem.yaml"
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    source_tree_sha256 = git_tracked_tree_sha256(repo)
    training_runtime_environment = split_manifest["runtime_environment"]
    training_runtime_environment_sha256 = _canonical_sha256(
        training_runtime_environment
    )
    if training_runtime_environment_sha256 != split_manifest[
        "runtime_environment_sha256"
    ]:
        raise RuntimeError("training runtime environment self-hash is invalid")
    runtime_environment = runtime_environment_fingerprint()
    runtime_environment_sha256 = _canonical_sha256(runtime_environment)
    runtime_compatibility = runtime_environment_compatibility_fingerprint(
        training_runtime_environment
    )
    runtime_compatibility_sha256 = _canonical_sha256(runtime_compatibility)
    current_runtime_compatibility_sha256 = _canonical_sha256(
        runtime_environment_compatibility_fingerprint(runtime_environment)
    )
    if source_commit != split_manifest["source_git_commit"]:
        raise RuntimeError("formal manifest must be built from the training source commit")
    if source_tree_sha256 != split_manifest["source_tracked_tree_sha256"]:
        raise RuntimeError("tracked source content differs from the training run")
    if current_runtime_compatibility_sha256 != runtime_compatibility_sha256:
        raise RuntimeError(
            "manifest-builder runtime is not inference-compatible with training"
        )
    protocols = (
        [
            {"name": "primary_fc_h10_o25", "horizon": 10, "goal_offset": goal_offset,
             "eval_budget": 100, "receding_horizon": 10,
             "action_block": int(cfg.plan_config.action_block)},
            {"name": "secondary_stock_h5_o25", "horizon": 5, "goal_offset": goal_offset,
             "eval_budget": 50, "receding_horizon": 5,
             "action_block": int(cfg.plan_config.action_block)},
        ]
        if task == "pusht"
        else [
            {"name": "primary_stock_h5_o25", "horizon": 5, "goal_offset": goal_offset,
             "eval_budget": 50, "receding_horizon": 5,
             "action_block": int(cfg.plan_config.action_block)},
            {"name": "secondary_fc_h10_o25", "horizon": 10, "goal_offset": goal_offset,
             "eval_budget": 100, "receding_horizon": 10,
             "action_block": int(cfg.plan_config.action_block)},
        ]
    )
    _validate_task_protocols(str(task), protocols)
    payload = {
        "schema_version": FORMAL_EVAL_MANIFEST_SCHEMA_VERSION,
        "task": str(task),
        "created_by_git_commit": source_commit,
        "source_tracked_tree_sha256": source_tree_sha256,
        "runtime_environment": training_runtime_environment,
        "runtime_environment_sha256": training_runtime_environment_sha256,
        "runtime_compatibility": runtime_compatibility,
        "runtime_compatibility_sha256": runtime_compatibility_sha256,
        "manifest_creation_runtime_environment": runtime_environment,
        "manifest_creation_runtime_environment_sha256": runtime_environment_sha256,
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": split_manifest_sha256,
        "training_dataset_path": (
            split_manifest["dataset_path"]
            if study_family == "delta_triangle"
            else str(dataset_path)
        ),
        "training_dataset_size_bytes": (
            split_manifest["dataset_size_bytes"]
            if study_family == "delta_triangle"
            else int(size_bytes)
        ),
        "training_dataset_mtime_ns": (
            split_manifest["dataset_mtime_ns"]
            if study_family == "delta_triangle"
            else int(dataset_stat_after.st_mtime_ns)
        ),
        "split_seed": int(split_manifest["split_seed"]),
        "train_split_sha256": split_manifest["train_sha256"],
        "validation_split_sha256": split_manifest["validation_sha256"],
        "dataset_name": split_manifest["dataset_name"],
        "dataset_path": str(dataset_path),
        "dataset_size_bytes": int(size_bytes),
        "dataset_mtime_ns": int(dataset_stat_after.st_mtime_ns),
        "dataset_windows": int(split_manifest["dataset_windows"]),
        "dataset_fingerprint_sha256": fingerprint,
        "dataset_file_sha256": dataset_file_sha256,
        "normalizer_fit": "train_episodes_only",
        "normalizer_stats": split_manifest["normalizer_stats"],
        "unique_episode_sampling": True,
        "cross_bank_episode_disjoint": True,
        "num_eval_per_bank": int(num_eval),
        "goal_offset": goal_offset,
        "eval_config_path": str(cfg_path),
        "eval_config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "solver_config_path": str(solver_path),
        "solver_config_sha256": hashlib.sha256(solver_path.read_bytes()).hexdigest(),
        "protocols": protocols,
        "banks": banks,
    }
    if study_family == "demo_goal":
        sorted_excluded_episode_ids = sorted(excluded_episode_ids)
        payload.update({
            "study_id": study_id,
            "method_lock_path": str(method_lock_path),
            "method_lock_file_sha256": method_lock_file_sha256,
            "method_lock_sha256": method_lock["method_lock_sha256"],
            "bank_seed_namespaces": copy.deepcopy(bank_seed_namespaces),
            "cross_study_exclusions": cross_study_exclusions,
            "cross_study_excluded_episode_ids": sorted_excluded_episode_ids,
            "cross_study_excluded_episode_count": len(
                sorted_excluded_episode_ids
            ),
            "cross_study_excluded_episode_ids_sha256": _canonical_sha256(
                sorted_excluded_episode_ids
            ),
            "cross_study_episode_disjoint": True,
        })
    _validate_formal_bank_structure(
        payload,
        expected_discovery_banks=discovery_banks,
        expected_confirmation_banks=confirmation_banks,
        expected_num_eval=num_eval,
        expected_goal_offset=goal_offset,
    )
    if study_family == "demo_goal":
        verify_demo_goal_manifest_external_bindings(payload)
    payload["manifest_sha256"] = _canonical_sha256(payload)
    if output_path.exists():
        raise RuntimeError(
            f"formal evaluation manifest output appeared during construction: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output_path, flags, 0o600)
    try:
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ).encode()
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("pusht", "tworoom"))
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument(
        "--study-family",
        choices=("delta_triangle", "demo_goal"),
        default="delta_triangle",
    )
    parser.add_argument("--method-lock", type=Path)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    payload = build_formal_eval_manifest(
        task=args.task,
        split_manifest_path=args.split_manifest,
        output_path=args.output,
        dataset_path=args.dataset_path,
        num_eval=args.num_eval,
        study_family=args.study_family,
        method_lock_path=args.method_lock,
        exclusion_manifest_paths=args.exclude_manifest,
    )
    print(
        f"FORMAL_EVAL_MANIFEST task={payload['task']} "
        f"sha256={payload['manifest_sha256']} output={args.output}"
    )


if __name__ == "__main__":
    main()
