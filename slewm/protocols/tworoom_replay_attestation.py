"""Pre-performance replay gate for the canonical TwoRoom dataset and env."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import subprocess
from pathlib import Path

import h5py
import numpy as np
import PIL
import torch
from PIL import Image

from slewm.protocols.demo_goal_formal_artifacts import (
    canonical_sha256,
    file_sha256,
    verify_demo_goal_method_lock,
    verify_self_hashed_json,
    write_exclusive_self_hashed_json,
)
from slewm.protocols.demo_goal_formal_specs import DEMO_GOAL_STUDY_IDS
from slewm.training.utilities import git_tracked_tree_sha256


REPLAY_SCHEMA_VERSION = 1
REPLAY_HASH_KEY = "replay_attestation_sha256"
GOAL_OFFSET = 25
FORMAL_QUOTA_PER_CATEGORY = 16
CATEGORIES = ("free_motion", "solid_central_wall_collision", "door_passage")
STATE_ATOL = 1e-6
NEXT_STATE_ATOL = 1e-4


class ReplayAttestationError(RuntimeError):
    pass


def _sha_episode_ids(values) -> str:
    return hashlib.sha256(",".join(map(str, values)).encode()).hexdigest()


def _array_sha256(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _python_tree_sha256(root: Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not paths:
        raise ReplayAttestationError(f"no Python sources found below {root}")
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True
    ).strip()


def _make_tworoom_env():
    import gymnasium as gym
    import stable_worldmodel.envs  # noqa: F401 - registers swm/TwoRoom-v1

    return gym.make("swm/TwoRoom-v1", render_mode="rgb_array")


def _base_env(env):
    return getattr(env, "unwrapped", env)


def _reset_fixed(env):
    return env.reset(seed=0, options={"variation": []})


def _space_value(space, dotted_name):
    value = space
    for component in dotted_name.split("."):
        value = value[component]
    return np.asarray(value.value)


def _variation_defaults(base):
    if hasattr(base, "variation_defaults"):
        return {
            str(key): np.asarray(value)
            for key, value in base.variation_defaults.items()
        }
    names = (
        "agent.color",
        "agent.radius",
        "agent.speed",
        "target.color",
        "target.radius",
        "wall.color",
        "wall.border_color",
        "wall.thickness",
        "wall.axis",
        "door.color",
        "door.number",
        "door.size",
        "door.position",
        "background.color",
        "rendering.render_target",
        "task.min_steps",
    )
    return {name: _space_value(base.variation_space, name) for name in names}


def _geometry_snapshot(base):
    defaults = _variation_defaults(base)
    return {
        "image_size": int(base.IMG_SIZE),
        "border_size": float(base.BORDER_SIZE),
        "wall_center": float(base.WALL_CENTER),
        "wall_axis": int(base.wall_axis),
        "wall_thickness": int(base.wall_thickness),
        "agent_radius": float(np.asarray(defaults["agent.radius"]).reshape(-1)[0]),
        "agent_speed": float(np.asarray(defaults["agent.speed"]).reshape(-1)[0]),
        "num_doors": int(base.num_doors),
        "door_positions": np.asarray(base.door_positions, dtype=np.float64).tolist(),
        "door_half_extents": np.asarray(base.door_sizes, dtype=np.float64).tolist(),
        "door_collision_margin": float(getattr(base, "DOOR_MARGIN", 1.75)),
    }


def _render_like_add_pixels(env):
    image = np.asarray(env.render())
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ReplayAttestationError(
            f"TwoRoom render must be HWC uint8 RGB; received {image.shape} {image.dtype}"
        )
    return np.asarray(
        Image.fromarray(image).resize((224, 224), Image.BILINEAR)
    )


def _classify_transition(base, p0, action, geometry):
    p0 = np.asarray(p0, dtype=np.float64)
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    proposal = p0 + action * geometry["agent_speed"]
    radius = geometry["agent_radius"]
    low = geometry["border_size"] + radius
    high = geometry["image_size"] - geometry["border_size"] - radius
    border_clipped = np.clip(proposal, low, high)
    border_collision = not np.array_equal(proposal, border_clipped)

    axis = geometry["wall_axis"]
    # TwoRoom encodes axis=1 as a vertical wall (motion coordinate x=0) and
    # axis=0 as a horizontal wall (motion coordinate y=1).
    coordinate = 0 if axis == 1 else 1
    transverse = 1 - coordinate
    center = geometry["wall_center"]
    half = geometry["wall_thickness"] // 2
    effective_low = center - half - radius
    effective_high = center + half + radius
    started_low = p0[coordinate] < center
    enters_wall_zone = (
        border_clipped[coordinate] > effective_low
        if started_low
        else border_clipped[coordinate] < effective_high
    )
    transverse_value = float(border_clipped[transverse])
    in_door = any(
        (float(door_center) - float(door_size) - geometry["door_collision_margin"])
        <= transverse_value
        <= (float(door_center) + float(door_size) + geometry["door_collision_margin"])
        for door_center, door_size in zip(
            geometry["door_positions"], geometry["door_half_extents"]
        )
    )
    predicted = np.asarray(
        base._apply_collisions(
            torch.as_tensor(p0, dtype=torch.float32),
            torch.as_tensor(proposal, dtype=torch.float32),
        ),
        dtype=np.float64,
    )
    collision_delta = float(np.max(np.abs(predicted - border_clipped)))
    normal_action = abs(float(action[coordinate]))
    crossed_entry = (
        started_low
        and p0[coordinate] <= effective_low
        and border_clipped[coordinate] > effective_low
    ) or (
        not started_low
        and p0[coordinate] >= effective_high
        and border_clipped[coordinate] < effective_high
    )
    realized_displacement = float(np.linalg.norm(predicted - p0))
    if (
        crossed_entry
        and normal_action > 1e-6
        and not in_door
        and collision_delta > 1e-6
    ):
        return "solid_central_wall_collision"
    if (
        crossed_entry
        and normal_action > 1e-6
        and in_door
        and not border_collision
        and collision_delta <= 1e-6
        and realized_displacement > 1e-6
    ):
        return "door_passage"
    if (
        not enters_wall_zone
        and not border_collision
        and collision_delta <= 1e-6
        and float(np.linalg.norm(action)) > 1e-6
        and realized_displacement > 1e-6
    ):
        return "free_motion"
    return None


def _h5_dataset_map(handle):
    datasets = {}

    def collect(name, value):
        if isinstance(value, h5py.Dataset):
            datasets[name] = value

    handle.visititems(collect)
    return datasets


def _validate_h5_descriptors(datasets):
    required = {
        "ep_len": (np.dtype("int32"), 1),
        "ep_offset": (np.dtype("int64"), 1),
        "pixels": (np.dtype("uint8"), 4),
        "action": (np.dtype("float32"), 2),
        "proprio": (np.dtype("float32"), 2),
        "step_idx": (None, 1),
    }
    missing = sorted(set(required) - set(datasets))
    if missing:
        raise ReplayAttestationError(f"TwoRoom HDF5 lacks required datasets: {missing}")
    episode_columns = [name for name in ("episode_idx", "ep_idx") if name in datasets]
    if len(episode_columns) != 1:
        raise ReplayAttestationError(
            "TwoRoom HDF5 requires exactly one of episode_idx or ep_idx"
        )
    for name, (dtype, ndim) in required.items():
        dataset = datasets[name]
        if dataset.ndim != ndim:
            raise ReplayAttestationError(
                f"{name} must have ndim={ndim}; received {dataset.ndim}"
            )
        if dtype is not None and dataset.dtype != dtype:
            raise ReplayAttestationError(
                f"{name} must have dtype={dtype}; received {dataset.dtype}"
            )
    if datasets["step_idx"].dtype.kind not in "iu":
        raise ReplayAttestationError("step_idx must have an integer dtype")
    episode_column = episode_columns[0]
    if datasets[episode_column].dtype.kind not in "iu":
        raise ReplayAttestationError("episode labels must have an integer dtype")
    if datasets["pixels"].shape[1:] != (224, 224, 3):
        raise ReplayAttestationError(
            f"pixels must have shape [N,224,224,3]; got {datasets['pixels'].shape}"
        )
    for name in ("action", "proprio"):
        if datasets[name].shape[1:] != (2,):
            raise ReplayAttestationError(f"{name} must have shape [N,2]")
    return episode_column


def _validate_split(split, *, dataset_sha, expected_commit, expected_tree, episodes):
    expected = {
        "dataset_name": "tworoom.h5",
        "dataset_file_sha256": dataset_sha,
        "source_git_commit": expected_commit,
        "source_tracked_tree_sha256": expected_tree,
        "transition_triangle_task": "tworoom",
        "transition_triangle_study_family": "demo_goal",
        "normalizer_fit": "train_episodes_only",
    }
    mismatches = {
        key: (split.get(key), wanted)
        for key, wanted in expected.items()
        if split.get(key) != wanted
    }
    train_ids = [int(value) for value in split.get("train_internal_episode_ids", [])]
    validation_ids = [
        int(value) for value in split.get("validation_internal_episode_ids", [])
    ]
    if (
        not train_ids
        or not validation_ids
        or len(train_ids) != len(set(train_ids))
        or len(validation_ids) != len(set(validation_ids))
        or set(train_ids).intersection(validation_ids)
        or set(train_ids).union(validation_ids) != set(range(episodes))
    ):
        mismatches["episode_partition"] = (
            {"train": len(train_ids), "validation": len(validation_ids)},
            f"disjoint exact partition of {episodes} episodes",
        )
    if split.get("train_sha256") != _sha_episode_ids(train_ids):
        mismatches["train_sha256"] = (split.get("train_sha256"), "recomputed")
    if split.get("validation_sha256") != _sha_episode_ids(validation_ids):
        mismatches["validation_sha256"] = (
            split.get("validation_sha256"), "recomputed"
        )
    if mismatches:
        raise ReplayAttestationError(f"TwoRoom split manifest mismatch: {mismatches}")
    return train_ids, validation_ids


def _validate_train_episode_layout(
    datasets, *, episode_column, lengths, offsets, train_ids
):
    total_rows = int(lengths.sum())
    if offsets.shape != lengths.shape or lengths.ndim != 1:
        raise ReplayAttestationError("ep_len and ep_offset must be aligned vectors")
    expected_offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
    )
    if not np.array_equal(offsets, expected_offsets):
        raise ReplayAttestationError("ep_offset is not the cumulative ep_len prefix")
    for name, dataset in datasets.items():
        if name not in {"ep_len", "ep_offset"} and dataset.shape[0] != total_rows:
            raise ReplayAttestationError(
                f"row dataset {name} has leading dim {dataset.shape[0]}, want {total_rows}"
            )
    for internal_episode in train_ids:
        start = int(offsets[internal_episode])
        length = int(lengths[internal_episode])
        stop = start + length
        steps = np.asarray(datasets["step_idx"][start:stop])
        labels = np.asarray(datasets[episode_column][start:stop])
        if not np.array_equal(steps, np.arange(length, dtype=steps.dtype)):
            raise ReplayAttestationError(
                f"train episode {internal_episode} has non-contiguous step_idx"
            )
        if labels.size == 0 or not np.all(labels == labels[0]):
            raise ReplayAttestationError(
                f"train episode {internal_episode} has inconsistent episode labels"
            )
        actions = np.asarray(datasets["action"][start:stop])
        if not np.isfinite(actions[:-1]).all():
            raise ReplayAttestationError(
                f"train episode {internal_episode} has non-finite non-terminal action"
            )


def _validate_variations(datasets, *, offsets, lengths, train_ids, defaults):
    allowed_position = {
        "variation.agent.position",
        "variation.target.position",
    }
    observed = {}
    for name, dataset in datasets.items():
        if not name.startswith("variation."):
            continue
        if name in allowed_position:
            observed[name] = "position_overridden_by_formal_setter"
            continue
        short_name = name[len("variation.") :]
        if short_name not in defaults:
            raise ReplayAttestationError(
                f"unrecognized non-position TwoRoom variation column: {name}"
            )
        expected = np.asarray(defaults[short_name])
        for internal_episode in train_ids:
            start = int(offsets[internal_episode])
            stop = start + int(lengths[internal_episode])
            actual = np.asarray(dataset[start:stop])
            try:
                matches = np.equal(actual, expected).all()
            except ValueError as error:
                raise ReplayAttestationError(
                    f"variation {name} has incompatible shape {actual.shape} vs {expected.shape}"
                ) from error
            if not bool(matches):
                raise ReplayAttestationError(
                    f"variation {name} differs from the current evaluator default"
                )
        observed[name] = "present_and_equal_to_current_default"
    for name in defaults:
        observed.setdefault(
            f"variation.{name}", "unrecorded_but_replay_verified"
        )
    return observed


def _select_transitions(
    datasets,
    *,
    lengths,
    offsets,
    train_ids,
    base,
    geometry,
    study_id,
    dataset_sha,
    train_sha,
    quota_per_category,
):
    candidates = {category: [] for category in CATEGORIES}
    for internal_episode in train_ids:
        start = int(offsets[internal_episode])
        length = int(lengths[internal_episode])
        max_t = length - GOAL_OFFSET - 1
        if max_t < 0:
            continue
        states = np.asarray(datasets["proprio"][start : start + max_t + 1])
        actions = np.asarray(datasets["action"][start : start + max_t + 1])
        for local_t, (p0, action) in enumerate(zip(states, actions)):
            category = _classify_transition(base, p0, action, geometry)
            if category is None:
                continue
            selection_key = hashlib.sha256(
                (
                    f"{study_id}|{dataset_sha}|{train_sha}|{category}|"
                    f"{internal_episode}|{local_t}"
                ).encode()
            ).hexdigest()
            candidates[category].append(
                (selection_key, int(internal_episode), int(local_t))
            )
    selected = []
    for category in CATEGORIES:
        used_episodes = set()
        category_selected = []
        for selection_key, internal_episode, local_t in sorted(candidates[category]):
            if internal_episode in used_episodes:
                continue
            used_episodes.add(internal_episode)
            category_selected.append({
                "category": category,
                "selection_key_sha256": selection_key,
                "internal_episode_id": internal_episode,
                "local_step": local_t,
                "global_row": int(offsets[internal_episode]) + local_t,
                "goal_local_step": local_t + GOAL_OFFSET,
            })
            if len(category_selected) == int(quota_per_category):
                break
        if len(category_selected) != int(quota_per_category):
            raise ReplayAttestationError(
                "FAIL_COVERAGE: "
                f"{category} provides {len(category_selected)} unique episodes; "
                f"requires {quota_per_category}"
            )
        selected.extend(category_selected)
    return selected


def _assert_close(actual, expected, *, atol, label):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ReplayAttestationError(
            f"{label} shape mismatch: {actual.shape} vs {expected.shape}"
        )
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    if not np.isfinite(error) or error > float(atol):
        raise ReplayAttestationError(
            f"{label} mismatch: max_abs={error} atol={atol}"
        )
    return error


def _replay_selected(datasets, selected):
    records = []
    for item in selected:
        row = int(item["global_row"])
        goal_row = row + GOAL_OFFSET
        p0 = np.asarray(datasets["proprio"][row])
        p1 = np.asarray(datasets["proprio"][row + 1])
        goal = np.asarray(datasets["proprio"][goal_row])
        action = np.asarray(datasets["action"][row])
        pixel0 = np.asarray(datasets["pixels"][row])
        pixel1 = np.asarray(datasets["pixels"][row + 1])

        env = _make_tworoom_env()
        try:
            _reset_fixed(env)
            base = _base_env(env)
            set_state = getattr(base, "_set_state", None)
            set_goal = getattr(base, "_set_goal_state", None)
            if not callable(set_state) or not callable(set_goal):
                raise ReplayAttestationError(
                    "TwoRoom formal replay requires callable _set_state and _set_goal_state"
                )
            set_state(p0)
            state_error = _assert_close(
                np.asarray(base.agent_position), p0, atol=STATE_ATOL, label="setter state"
            )
            agent_before_goal = np.asarray(base.agent_position).copy()
            set_goal(goal)
            goal_error = _assert_close(
                np.asarray(base.target_position), goal, atol=STATE_ATOL, label="setter goal"
            )
            goal_side_effect_error = _assert_close(
                np.asarray(base.agent_position),
                agent_before_goal,
                atol=STATE_ATOL,
                label="goal setter agent side effect",
            )
            rendered0 = _render_like_add_pixels(env)
            if rendered0.shape != pixel0.shape or rendered0.dtype != pixel0.dtype:
                raise ReplayAttestationError("initial replay pixel shape/dtype mismatch")
            initial_mismatch = int(np.count_nonzero(rendered0 != pixel0))
            initial_max_abs = int(
                np.max(np.abs(rendered0.astype(np.int16) - pixel0.astype(np.int16)))
            )
            if initial_mismatch:
                raise ReplayAttestationError(
                    "initial replay pixels are not byte-exact: "
                    f"mismatch_count={initial_mismatch} max_abs={initial_max_abs}"
                )

            _, _, _, _, info = env.step(action)
            next_state = info.get("proprio")
            if next_state is None:
                next_state = np.asarray(base.agent_position)
            next_state_error = _assert_close(
                next_state, p1, atol=NEXT_STATE_ATOL, label="next proprio"
            )
            rendered1 = _render_like_add_pixels(env)
            if rendered1.shape != pixel1.shape or rendered1.dtype != pixel1.dtype:
                raise ReplayAttestationError("next replay pixel shape/dtype mismatch")
            next_mismatch = int(np.count_nonzero(rendered1 != pixel1))
            next_max_abs = int(
                np.max(np.abs(rendered1.astype(np.int16) - pixel1.astype(np.int16)))
            )
            if next_mismatch:
                raise ReplayAttestationError(
                    "next replay pixels are not byte-exact: "
                    f"mismatch_count={next_mismatch} max_abs={next_max_abs}"
                )
            # The formal replay installs a new goal from proprio[t+25].  The
            # source HDF5 terminal flags were produced under the episode's
            # original goal and therefore are not comparable.  State and
            # byte-exact pixels are the dynamics/rendering contract here.

            records.append({
                **item,
                "action": action.astype(float).tolist(),
                "goal_source": f"proprio[t+{GOAL_OFFSET}]",
                "setter_state_max_abs": state_error,
                "setter_goal_max_abs": goal_error,
                "goal_setter_agent_side_effect_max_abs": goal_side_effect_error,
                "next_proprio_max_abs": next_state_error,
                "initial_expected_pixel_sha256": _array_sha256(pixel0),
                "initial_actual_pixel_sha256": _array_sha256(rendered0),
                "initial_pixel_mismatch_count": initial_mismatch,
                "initial_pixel_max_abs": initial_max_abs,
                "next_expected_pixel_sha256": _array_sha256(pixel1),
                "next_actual_pixel_sha256": _array_sha256(rendered1),
                "next_pixel_mismatch_count": next_mismatch,
                "next_pixel_max_abs": next_max_abs,
            })
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    return records


def attest_tworoom_dataset_env_replay(
    *,
    repo: Path,
    dataset_path: Path,
    split_manifest_path: Path,
    method_lock_path: Path,
    output_path: Path,
    expected_commit: str,
    expected_tree: str,
    smoke_job_id: str,
    quota_per_category: int = FORMAL_QUOTA_PER_CATEGORY,
):
    repo = Path(repo).resolve()
    dataset_path = Path(dataset_path).resolve()
    split_manifest_path = Path(split_manifest_path).resolve()
    method_lock_path = Path(method_lock_path).resolve()
    output_path = Path(output_path).resolve()
    if quota_per_category <= 0:
        raise ValueError("quota_per_category must be positive")
    if _git(repo, "rev-parse", "--show-toplevel") != str(repo):
        raise ReplayAttestationError("repo must be the exact git worktree root")
    if _git(repo, "rev-parse", "HEAD") != expected_commit:
        raise ReplayAttestationError("TwoRoom replay source commit mismatch")
    if git_tracked_tree_sha256(repo) != expected_tree:
        raise ReplayAttestationError("TwoRoom replay tracked tree mismatch")
    if _git(repo, "status", "--porcelain", "--untracked-files=normal"):
        raise ReplayAttestationError("TwoRoom replay requires a clean worktree")
    if not str(smoke_job_id).isdigit():
        raise ReplayAttestationError("smoke_job_id must be numeric")
    lock = verify_demo_goal_method_lock(
        method_lock_path,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
    )
    dataset_stat_before = dataset_path.stat()
    dataset_sha = file_sha256(dataset_path)
    split = json.loads(split_manifest_path.read_text())

    with h5py.File(dataset_path, "r", swmr=True) as handle:
        datasets = _h5_dataset_map(handle)
        episode_column = _validate_h5_descriptors(datasets)
        lengths = np.asarray(datasets["ep_len"][:])
        offsets = np.asarray(datasets["ep_offset"][:])
        train_ids, validation_ids = _validate_split(
            split,
            dataset_sha=dataset_sha,
            expected_commit=expected_commit,
            expected_tree=expected_tree,
            episodes=len(lengths),
        )
        _validate_train_episode_layout(
            datasets,
            episode_column=episode_column,
            lengths=lengths,
            offsets=offsets,
            train_ids=train_ids,
        )
        env = _make_tworoom_env()
        try:
            _reset_fixed(env)
            base = _base_env(env)
            defaults = _variation_defaults(base)
            if bool(getattr(base, "render_target_flag", False)):
                raise ReplayAttestationError("current TwoRoom evaluator renders target unexpectedly")
            geometry = _geometry_snapshot(base)
            env_source = Path(inspect.getsourcefile(base.__class__)).resolve()
            selected = _select_transitions(
                datasets,
                lengths=lengths,
                offsets=offsets,
                train_ids=train_ids,
                base=base,
                geometry=geometry,
                study_id=DEMO_GOAL_STUDY_IDS["tworoom"],
                dataset_sha=dataset_sha,
                train_sha=split["train_sha256"],
                quota_per_category=quota_per_category,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        variation_contract = _validate_variations(
            datasets,
            offsets=offsets,
            lengths=lengths,
            train_ids=train_ids,
            defaults=defaults,
        )
        replay_records = _replay_selected(datasets, selected)
        schema = {
            name: {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
            for name, dataset in sorted(datasets.items())
        }

    legacy_source = env_source.with_name("legacy_env.py")
    stable_worldmodel_root = next(
        (
            parent
            for parent in env_source.parents
            if parent.name == "stable_worldmodel"
        ),
        env_source.parent,
    )
    dataset_stat_after = dataset_path.stat()
    if (
        dataset_stat_after.st_size != dataset_stat_before.st_size
        or dataset_stat_after.st_mtime_ns != dataset_stat_before.st_mtime_ns
        or file_sha256(dataset_path) != dataset_sha
    ):
        raise ReplayAttestationError(
            "TwoRoom dataset changed while replay attestation was running"
        )
    payload = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "PASS",
        "study_id": DEMO_GOAL_STUDY_IDS["tworoom"],
        "task": "tworoom",
        "purpose": "canonical HDF5 to current evaluator one-step replay equivalence",
        "contains_performance_outcomes": False,
        "checkpoint_loaded": False,
        "formal_banks_read": False,
        "training_episodes_only": True,
        "source_git_commit": expected_commit,
        "source_tracked_tree_sha256": expected_tree,
        "source_worktree_clean": True,
        "smoke_slurm_job_id": str(smoke_job_id),
        "method_lock_path": str(method_lock_path),
        "method_lock_file_sha256": file_sha256(method_lock_path),
        "method_lock_sha256": lock["method_lock_sha256"],
        "dataset": {
            "path": str(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
            "file_sha256": dataset_sha,
            "schema": schema,
            "episode_label_column": episode_column,
        },
        "split_manifest": {
            "path": str(split_manifest_path),
            "file_sha256": file_sha256(split_manifest_path),
            "train_sha256": split["train_sha256"],
            "validation_sha256": split["validation_sha256"],
            "num_train_episodes": len(train_ids),
            "num_validation_episodes": len(validation_ids),
        },
        "selection": {
            "namespace": "sha256(study|dataset|train|category|episode|step)",
            "categories": list(CATEGORIES),
            "quota_per_category": int(quota_per_category),
            "goal_offset": GOAL_OFFSET,
            "does_not_read_next_state_or_pixels_for_selection": True,
            "records": selected,
        },
        "replay": {
            "action_alignment": "action[t]:proprio[t]->proprio[t+1]",
            "goal_source": f"proprio[t+{GOAL_OFFSET}]",
            "reset": {"seed": 0, "variation": []},
            "pixel_preprocess": "PIL.Image.BILINEAR resize to 224x224",
            "state_atol": STATE_ATOL,
            "next_state_atol": NEXT_STATE_ATOL,
            "pixels_require_byte_exact": True,
            "records": replay_records,
        },
        "geometry": geometry,
        "variation_contract": variation_contract,
        "dataset_extra_fields": sorted(
            set(schema)
            - {"ep_len", "ep_offset", "pixels", "action", "proprio", "step_idx", episode_column}
        ),
        "environment": {
            "class": f"{base.__class__.__module__}.{base.__class__.__qualname__}",
            "source_path": str(env_source),
            "source_file_sha256": file_sha256(env_source),
            "stable_worldmodel_package_root": str(stable_worldmodel_root),
            "stable_worldmodel_python_tree_sha256": _python_tree_sha256(
                stable_worldmodel_root
            ),
            "legacy_source_path": str(legacy_source) if legacy_source.is_file() else None,
            "legacy_source_file_sha256": (
                file_sha256(legacy_source) if legacy_source.is_file() else None
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pillow": PIL.__version__,
        },
    }
    return write_exclusive_self_hashed_json(
        payload, output_path, hash_key=REPLAY_HASH_KEY
    )


def verify_tworoom_replay_attestation(
    path: Path,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
    expected_dataset_sha256: str | None = None,
    expected_train_sha256: str | None = None,
    expected_method_lock_sha256: str | None = None,
    replay_current_environment: bool = False,
):
    payload = verify_self_hashed_json(path, hash_key=REPLAY_HASH_KEY)
    expected = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "PASS",
        "study_id": DEMO_GOAL_STUDY_IDS["tworoom"],
        "task": "tworoom",
        "contains_performance_outcomes": False,
        "checkpoint_loaded": False,
        "formal_banks_read": False,
        "training_episodes_only": True,
        "source_worktree_clean": True,
    }
    if expected_commit is not None:
        expected["source_git_commit"] = expected_commit
    if expected_tree is not None:
        expected["source_tracked_tree_sha256"] = expected_tree
    mismatches = {
        key: (payload.get(key), wanted)
        for key, wanted in expected.items()
        if payload.get(key) != wanted
    }
    dataset = payload.get("dataset", {})
    split = payload.get("split_manifest", {})
    selection = payload.get("selection", {})
    replay = payload.get("replay", {})
    replay_expected = {
        "action_alignment": "action[t]:proprio[t]->proprio[t+1]",
        "goal_source": f"proprio[t+{GOAL_OFFSET}]",
        "reset": {"seed": 0, "variation": []},
        "pixel_preprocess": "PIL.Image.BILINEAR resize to 224x224",
        "state_atol": STATE_ATOL,
        "next_state_atol": NEXT_STATE_ATOL,
        "pixels_require_byte_exact": True,
    }
    for key, wanted in replay_expected.items():
        if replay.get(key) != wanted:
            mismatches[f"replay.{key}"] = (replay.get(key), wanted)
    if expected_dataset_sha256 is not None and dataset.get(
        "file_sha256"
    ) != expected_dataset_sha256:
        mismatches["dataset.file_sha256"] = (
            dataset.get("file_sha256"), expected_dataset_sha256
        )
    if expected_train_sha256 is not None and split.get(
        "train_sha256"
    ) != expected_train_sha256:
        mismatches["split_manifest.train_sha256"] = (
            split.get("train_sha256"), expected_train_sha256
        )
    if expected_method_lock_sha256 is not None and payload.get(
        "method_lock_sha256"
    ) != expected_method_lock_sha256:
        mismatches["method_lock_sha256"] = (
            payload.get("method_lock_sha256"), expected_method_lock_sha256
        )
    if selection.get("quota_per_category") != FORMAL_QUOTA_PER_CATEGORY:
        mismatches["selection.quota_per_category"] = (
            selection.get("quota_per_category"), FORMAL_QUOTA_PER_CATEGORY
        )
    if selection.get("goal_offset") != GOAL_OFFSET:
        mismatches["selection.goal_offset"] = (
            selection.get("goal_offset"), GOAL_OFFSET
        )
    selected = selection.get("records")
    replayed = replay.get("records")
    if not isinstance(selected, list) or not isinstance(replayed, list):
        mismatches["records"] = (type(selected).__name__, "two lists")
    else:
        expected_count = FORMAL_QUOTA_PER_CATEGORY * len(CATEGORIES)
        if len(selected) != expected_count or len(replayed) != expected_count:
            mismatches["record_count"] = (
                (len(selected), len(replayed)), (expected_count, expected_count)
            )
        category_counts = {
            category: sum(
                item.get("category") == category for item in selected
            )
            for category in CATEGORIES
        }
        if category_counts != {
            category: FORMAL_QUOTA_PER_CATEGORY for category in CATEGORIES
        }:
            mismatches["category_counts"] = (
                category_counts,
                {category: FORMAL_QUOTA_PER_CATEGORY for category in CATEGORIES},
            )
        identity_keys = (
            "category",
            "selection_key_sha256",
            "internal_episode_id",
            "local_step",
            "global_row",
            "goal_local_step",
        )
        for index, (chosen, result) in enumerate(zip(selected, replayed)):
            if any(chosen.get(key) != result.get(key) for key in identity_keys):
                mismatches[f"records[{index}].identity"] = (result, chosen)
                break
            for key in (
                "initial_pixel_mismatch_count",
                "initial_pixel_max_abs",
                "next_pixel_mismatch_count",
                "next_pixel_max_abs",
            ):
                if result.get(key) != 0:
                    mismatches[f"records[{index}].{key}"] = (
                        result.get(key), 0
                    )
    try:
        method_lock_path = Path(payload["method_lock_path"])
        lock = verify_demo_goal_method_lock(
            method_lock_path,
            expected_commit=payload.get("source_git_commit"),
            expected_tree=payload.get("source_tracked_tree_sha256"),
        )
        if file_sha256(method_lock_path) != payload.get(
            "method_lock_file_sha256"
        ):
            mismatches["method_lock_file_sha256"] = ("recomputed", "bound")
        if lock.get("method_lock_sha256") != payload.get("method_lock_sha256"):
            mismatches["method_lock_binding"] = (
                lock.get("method_lock_sha256"), payload.get("method_lock_sha256")
            )
        split_path = Path(split["path"])
        if file_sha256(split_path) != split.get("file_sha256"):
            mismatches["split_manifest.file_sha256"] = ("recomputed", "bound")
        env_source = Path(payload["environment"]["source_path"])
        if file_sha256(env_source) != payload["environment"].get(
            "source_file_sha256"
        ):
            mismatches["environment.source_file_sha256"] = (
                "recomputed", "bound"
            )
        package_root = Path(
            payload["environment"]["stable_worldmodel_package_root"]
        )
        if _python_tree_sha256(package_root) != payload["environment"].get(
            "stable_worldmodel_python_tree_sha256"
        ):
            mismatches["environment.package_tree_sha256"] = (
                "recomputed", "bound"
            )
        dataset_path = Path(dataset["path"])
        if file_sha256(dataset_path) != dataset.get("file_sha256"):
            mismatches["dataset.file_sha256"] = ("recomputed", "bound")
        if dataset_path.stat().st_size != dataset.get("size_bytes"):
            mismatches["dataset.size_bytes"] = (
                dataset_path.stat().st_size, dataset.get("size_bytes")
            )
    except (KeyError, OSError, RuntimeError) as error:
        mismatches["external_bindings"] = (str(error), "all bindings valid")
    if replay_current_environment and not mismatches:
        try:
            _reverify_current_tworoom_environment(payload)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            mismatches["current_environment_replay"] = (
                str(error), "fresh deterministic replay PASS"
            )
    if mismatches:
        raise ReplayAttestationError(
            f"TwoRoom replay attestation mismatch: {mismatches}"
        )
    return payload


def _reverify_current_tworoom_environment(payload):
    """Recompute selection and all 48 transitions against the current env.

    This is deliberately stronger than checking a signed-looking JSON record:
    formal completion calls it after training so a fabricated self-consistent
    PASS artifact cannot authorize evaluation.
    """
    dataset_payload = payload["dataset"]
    split_payload = payload["split_manifest"]
    dataset_path = Path(dataset_payload["path"])
    split_path = Path(split_payload["path"])
    split = json.loads(split_path.read_text())
    dataset_sha = file_sha256(dataset_path)
    if dataset_sha != dataset_payload["file_sha256"]:
        raise ReplayAttestationError(
            "dataset bytes changed before current-environment replay"
        )
    with h5py.File(dataset_path, "r", swmr=True) as handle:
        datasets = _h5_dataset_map(handle)
        episode_column = _validate_h5_descriptors(datasets)
        lengths = np.asarray(datasets["ep_len"][:])
        offsets = np.asarray(datasets["ep_offset"][:])
        train_ids, validation_ids = _validate_split(
            split,
            dataset_sha=dataset_sha,
            expected_commit=payload["source_git_commit"],
            expected_tree=payload["source_tracked_tree_sha256"],
            episodes=len(lengths),
        )
        _validate_train_episode_layout(
            datasets,
            episode_column=episode_column,
            lengths=lengths,
            offsets=offsets,
            train_ids=train_ids,
        )
        env = _make_tworoom_env()
        try:
            _reset_fixed(env)
            base = _base_env(env)
            if bool(getattr(base, "render_target_flag", False)):
                raise ReplayAttestationError(
                    "current TwoRoom evaluator renders the target"
                )
            defaults = _variation_defaults(base)
            geometry = _geometry_snapshot(base)
            env_source = Path(inspect.getsourcefile(base.__class__)).resolve()
            selected = _select_transitions(
                datasets,
                lengths=lengths,
                offsets=offsets,
                train_ids=train_ids,
                base=base,
                geometry=geometry,
                study_id=DEMO_GOAL_STUDY_IDS["tworoom"],
                dataset_sha=dataset_sha,
                train_sha=split["train_sha256"],
                quota_per_category=FORMAL_QUOTA_PER_CATEGORY,
            )
            environment_class = (
                f"{base.__class__.__module__}.{base.__class__.__qualname__}"
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        variation_contract = _validate_variations(
            datasets,
            offsets=offsets,
            lengths=lengths,
            train_ids=train_ids,
            defaults=defaults,
        )
        replay_records = _replay_selected(datasets, selected)
        schema = {
            name: {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
            for name, dataset in sorted(datasets.items())
        }

    package_root = next(
        (
            parent
            for parent in env_source.parents
            if parent.name == "stable_worldmodel"
        ),
        env_source.parent,
    )
    legacy_source = env_source.with_name("legacy_env.py")
    current = {
        "dataset.schema": schema,
        "dataset.episode_label_column": episode_column,
        "split.num_train_episodes": len(train_ids),
        "split.num_validation_episodes": len(validation_ids),
        "selection.records": selected,
        "replay.records": replay_records,
        "geometry": geometry,
        "variation_contract": variation_contract,
        "dataset_extra_fields": sorted(
            set(schema)
            - {
                "ep_len",
                "ep_offset",
                "pixels",
                "action",
                "proprio",
                "step_idx",
                episode_column,
            }
        ),
        "environment.class": environment_class,
        "environment.source_file_sha256": file_sha256(env_source),
        "environment.stable_worldmodel_python_tree_sha256": (
            _python_tree_sha256(package_root)
        ),
        "environment.legacy_source_file_sha256": (
            file_sha256(legacy_source) if legacy_source.is_file() else None
        ),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pillow": PIL.__version__,
        },
    }
    recorded = {
        "dataset.schema": dataset_payload["schema"],
        "dataset.episode_label_column": dataset_payload[
            "episode_label_column"
        ],
        "split.num_train_episodes": split_payload["num_train_episodes"],
        "split.num_validation_episodes": split_payload[
            "num_validation_episodes"
        ],
        "selection.records": payload["selection"]["records"],
        "replay.records": payload["replay"]["records"],
        "geometry": payload["geometry"],
        "variation_contract": payload["variation_contract"],
        "dataset_extra_fields": payload["dataset_extra_fields"],
        "environment.class": payload["environment"]["class"],
        "environment.source_file_sha256": payload["environment"][
            "source_file_sha256"
        ],
        "environment.stable_worldmodel_python_tree_sha256": payload[
            "environment"
        ]["stable_worldmodel_python_tree_sha256"],
        "environment.legacy_source_file_sha256": payload["environment"][
            "legacy_source_file_sha256"
        ],
        "runtime": payload["runtime"],
    }
    mismatches = {
        key: (canonical_sha256(recorded[key]), canonical_sha256(current[key]))
        for key in current
        if recorded[key] != current[key]
    }
    if mismatches:
        raise ReplayAttestationError(
            "fresh TwoRoom replay differs from sealed evidence: "
            f"{mismatches}"
        )
    return {
        "dataset_file_sha256": dataset_sha,
        "train_sha256": split["train_sha256"],
        "selection_sha256": canonical_sha256(selected),
        "replay_records_sha256": canonical_sha256(replay_records),
        "environment_source_file_sha256": file_sha256(env_source),
        "environment_package_tree_sha256": _python_tree_sha256(package_root),
    }
