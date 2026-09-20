#!/usr/bin/env python3
"""LeWM-compatible evaluation environment for the SRSA gear-assembly task.

``stable_worldmodel.World`` drives an ``EnvPool``: a list of N independent
gymnasium environments that it resets, steps and teleports one at a time. Isaac
Sim cannot work that way — a process hosts exactly one simulation app, and its
environments are one batched tensor world. This module bridges the two:

* :class:`GearAssemblyEvalEnv` is the official ``AssemblySparseEnv`` plus the
  dataset RGB camera and a deterministic :meth:`set_state`, which writes a
  recorded simulator state directly instead of running the randomized
  multi-stage grasp reset. Termination is the strict insertion predicate, so
  ``World.terminateds`` means "really inserted", not the loose official one.
* :class:`GearEnvFacade` is the per-environment view ``_apply_callables``
  teleports through (``world.envs.envs[i].unwrapped.set_state(...)``).
* :class:`IsaacGearEnvPool` presents the batched world through the ``EnvPool``
  API, collapsing the per-env loop into one batched physics step.
* :class:`GearWorld` swaps that pool into ``swm.World`` so ``evaluate()``,
  ``_run()`` and the whole dataset-driven eval path work unchanged.

State column names match ``EvalTrajectoryHDF5Writer`` exactly, so a row of
``eval_trajectories.h5`` teleports the simulator with no renaming.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# A legacy v1 sidecar contains only these physical/reset fields.  Keep the
# public name for callers that were written against that schema.
STATE_COLUMNS: dict[str, int] = {
    "robot_joint_pos": 9,
    "robot_joint_vel": 9,
    "held_pos": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "fixed_pos_action_frame": 3,
    "init_fixed_pos_obs_noise": 3,
}

# A v2 sidecar additionally records the controller state that is otherwise
# hidden behind the same visible robot/object pose.  When all of these columns
# exist, a row can be restored directly.  A v1 row is still supported by
# replaying its expert prefix from episode row zero.
RESTORE_STATE_COLUMNS: dict[str, int] = {
    "held_root_vel": 6,
    "fixed_root_vel": 6,
    "action_ema": 6,
    "prev_action_ema": 6,
    "ctrl_target_joint_pos": 9,
    "ctrl_target_fingertip_midpoint_pos": 3,
    "ctrl_target_fingertip_midpoint_quat": 4,
    "ctrl_target_gripper_dof_pos": 1,
    "prev_joint_pos": 7,
    "prev_fingertip_pos": 3,
    "prev_fingertip_quat": 4,
    "ee_linvel_fd": 3,
    "ee_angvel_fd": 3,
    "joint_vel_fd": 7,
    "task_prop_gains": 6,
    "task_deriv_gains": 6,
}
FULL_STATE_COLUMNS: dict[str, int] = {**STATE_COLUMNS, **RESTORE_STATE_COLUMNS}

RAW_ACTION_DIM = 6
EFFECTIVE_ACTION_DIM = 4
# AutoMate consumes [dx, dy, dz, rx, ry, rz], but _apply_action overwrites the
# target roll and pitch with pi/0 and always keeps the gripper closed.  Thus the
# only physically effective controls are translation and yaw.  In particular,
# index 5 is yaw; it must never be treated as a gripper command.
EFFECTIVE_ACTION_INDICES = (0, 1, 2, 5)
INACTIVE_ACTION_INDICES = (3, 4)
EFFECTIVE_ACTION_SEMANTICS = "gear_effective_filtered_xyz_yaw_v1"
LEGACY_ACTION_SEMANTICS = "gear_raw_policy_xyz_rx_ry_rz_v1"
DEFAULT_MASTER_OFFSETS = (5, 15, 25)
SUPPORTED_GOAL_OFFSETS = DEFAULT_MASTER_OFFSETS


def validate_restoration_reconstruction(
    audit: Mapping[str, Any],
    *,
    max_mean_abs_error: float,
    max_pixels_over_8: float,
) -> None:
    """Reject a replay/restored start whose render is not the source row."""

    if max_mean_abs_error < 0:
        raise ValueError("max_mean_abs_error must be non-negative")
    if not 0.0 <= max_pixels_over_8 <= 1.0:
        raise ValueError("max_pixels_over_8 must be in [0, 1]")
    mean_error = float(audit.get("init_frame_mean_abs_error", float("nan")))
    pixel_fraction = float(audit.get("init_frame_pixels_over_8", float("nan")))
    if not np.isfinite(mean_error) or not np.isfinite(pixel_fraction):
        raise RuntimeError(f"restoration audit is incomplete or non-finite: {audit}")
    if mean_error > max_mean_abs_error or pixel_fraction > max_pixels_over_8:
        raise RuntimeError(
            "restored Gear start does not reproduce its source visual row: "
            f"mean_abs_error={mean_error:.4f} (limit {max_mean_abs_error}), "
            f"pixels_over_8={pixel_fraction:.4%} (limit {max_pixels_over_8:.4%})"
        )


def raw6_to_effective4(actions: Any) -> np.ndarray:
    """Project AutoMate's raw 6-D command onto its effective xyz+yaw space."""

    array = np.asarray(actions, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] != RAW_ACTION_DIM:
        raise ValueError(f"raw gear actions must end in dimension 6, got {array.shape}")
    return np.ascontiguousarray(array[..., EFFECTIVE_ACTION_INDICES])


def effective4_to_env6(actions: Any) -> np.ndarray:
    """Expand [dx,dy,dz,yaw] without ever confusing yaw for a gripper."""

    array = np.asarray(actions, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] != EFFECTIVE_ACTION_DIM:
        raise ValueError(
            f"effective gear actions must end in dimension 4, got {array.shape}"
        )
    expanded = np.zeros((*array.shape[:-1], RAW_ACTION_DIM), dtype=np.float32)
    expanded[..., EFFECTIVE_ACTION_INDICES] = array
    return expanded


def filter_raw_episode_to_effective4(
    raw_actions: Any, *, ema_factor: float = 0.2
) -> np.ndarray:
    """Return the exact applied xyz+yaw sequence for one reset-bounded episode.

    ``AssemblyEnv`` starts ``self.actions`` at zero and applies
    ``a_t = alpha * raw_t + (1-alpha) * a_(t-1)``.  Recording ``a_t`` makes the
    visual transition Markov in the stored action.  Evaluation can then set
    ``ema_factor=1`` and execute the planner's 4-D output directly.
    """

    raw = np.asarray(raw_actions, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != RAW_ACTION_DIM:
        raise ValueError(f"raw episode must have shape [T,6], got {raw.shape}")
    if not 0.0 < float(ema_factor) <= 1.0:
        raise ValueError("ema_factor must be in (0, 1]")
    if not np.isfinite(raw).all():
        raise ValueError("raw actions contain NaN or infinity")

    previous = np.zeros(RAW_ACTION_DIM, dtype=np.float32)
    filtered = np.empty_like(raw)
    alpha = np.float32(ema_factor)
    one_minus_alpha = np.float32(1.0 - ema_factor)
    for step, command in enumerate(raw):
        previous = alpha * command + one_minus_alpha * previous
        filtered[step] = previous
    return raw6_to_effective4(filtered)


def prepare_model_action(
    action: Any,
    *,
    protocol: str,
    clip: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map a planner action to the 6-D environment command and audit projection.

    The primary ``effective4`` protocol is lossless: the world model scores the
    same [xyz,yaw] values that the EMA-disabled environment executes.  The
    explicit ``legacy_raw6`` protocol exists only for old checkpoints; rx/ry are
    frozen because the environment discards them physically.
    """

    scored = np.asarray(action, dtype=np.float32).reshape(-1)
    if not np.isfinite(scored).all():
        raise ValueError("model action contains NaN or infinity")
    if protocol == "effective4":
        if scored.size != EFFECTIVE_ACTION_DIM:
            raise ValueError(
                f"effective4 protocol requires 4 model outputs [xyz,yaw], got {scored.size}"
            )
        model_effective = scored.copy()
    elif protocol == "legacy_raw6":
        if scored.size != RAW_ACTION_DIM:
            raise ValueError(f"legacy_raw6 protocol requires 6 model outputs, got {scored.size}")
        model_effective = raw6_to_effective4(scored)
    else:
        raise ValueError(f"unknown gear action protocol {protocol!r}")

    command_effective = model_effective.copy()
    if clip is not None:
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be a positive finite number or None")
        command_effective = np.clip(command_effective, -clip, clip)
    env_command = effective4_to_env6(command_effective)
    projection_delta = np.abs(command_effective - model_effective)
    audit = {
        "protocol": protocol,
        "model_scored_action": scored.tolist(),
        "model_scored_effective4": model_effective.tolist(),
        "command_effective4": command_effective.tolist(),
        "env_command6": env_command.tolist(),
        "model_to_command_max_abs": float(projection_delta.max(initial=0.0)),
        "inactive_rx_ry_frozen": True,
        "yaw_model_index": 3 if protocol == "effective4" else 5,
        "yaw_env_index": 5,
        "has_gripper_dimension": False,
    }
    return command_effective, env_command, audit


def complete_action_consistency_audit(
    audit: Mapping[str, Any], applied_env6: Any, *, tolerance: float = 1.0e-6
) -> dict[str, Any]:
    """Attach the environment's post-EMA action to a model/command audit."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    applied6 = np.asarray(applied_env6, dtype=np.float32).reshape(-1)
    if applied6.size != RAW_ACTION_DIM:
        raise ValueError(f"applied environment action must have 6 values, got {applied6.size}")
    applied4 = raw6_to_effective4(applied6)
    commanded4 = np.asarray(audit["command_effective4"], dtype=np.float32)
    command_delta = np.abs(applied4 - commanded4)
    inactive_max = float(np.abs(applied6[list(INACTIVE_ACTION_INDICES)]).max(initial=0.0))
    result = dict(audit)
    result.update(
        applied_env6=applied6.tolist(),
        applied_effective4=applied4.tolist(),
        command_to_applied_max_abs=float(command_delta.max(initial=0.0)),
        applied_inactive_rx_ry_max_abs=inactive_max,
        exact_model_scored_execution=bool(
            float(result["model_to_command_max_abs"]) <= tolerance
            and float(command_delta.max(initial=0.0)) <= tolerance
            and inactive_max <= tolerance
        ),
        tolerance=float(tolerance),
    )
    return result


def validate_gear_planner_config(
    reply: Mapping[str, Any],
    *,
    action_protocol: str,
    expected_horizon: int | None = None,
    expected_receding_horizon: int | None = None,
) -> dict[str, Any]:
    """Validate the policy-server reset handshake before any control step."""

    planner = reply.get("planner_config")
    if not isinstance(planner, dict):
        raise RuntimeError(f"server reset omitted planner_config: {reply}")
    schema = planner.get("task_action_schema")
    if not isinstance(schema, dict):
        raise RuntimeError(
            "server planner_config.task_action_schema must be a provenance object"
        )
    try:
        expected_name, expected_dim = {
            "effective4": ("gear_effective4", EFFECTIVE_ACTION_DIM),
            "legacy_raw6": ("gear_legacy6", RAW_ACTION_DIM),
        }[action_protocol]
    except KeyError as error:
        raise ValueError(f"unknown Gear action protocol {action_protocol!r}") from error
    actual = {
        "name": schema.get("name"),
        "action_dim": planner.get("action_dim"),
        "action_block": planner.get("action_block"),
    }
    expected = {
        "name": expected_name,
        "action_dim": expected_dim,
        "action_block": 5,
    }
    if actual != expected:
        raise RuntimeError(
            f"planning server is incompatible with action_protocol={action_protocol}: "
            f"expected {expected}, got {actual}"
        )
    if expected_horizon is not None and planner.get("horizon") != expected_horizon:
        raise RuntimeError(
            f"planner horizon must equal paired task horizon={expected_horizon}, "
            f"got {planner.get('horizon')!r}"
        )
    if (
        expected_receding_horizon is not None
        and planner.get("receding_horizon") != expected_receding_horizon
    ):
        raise RuntimeError(
            "planner receding_horizon must equal the formal full-commit value "
            f"{expected_receding_horizon}, got {planner.get('receding_horizon')!r}"
        )
    planner_mode = planner.get("planner_mode", "projected_cem")
    if planner_mode == "stock_cem":
        if schema.get("projection") is not None:
            raise RuntimeError("stock Gear CEM must not apply an action projection")
        if schema.get("bound_mode") != "none":
            raise RuntimeError("stock Gear CEM must use action_bound_mode=none")
    elif action_protocol == "effective4":
        if schema.get("continuous_action_indices") != [0, 1, 2, 3]:
            raise RuntimeError(
                "effective4 Gear must project full-plan TV over [x,y,z,yaw]"
            )
        tv_max = schema.get("continuous_plan_tv_max")
        if (
            not isinstance(tv_max, (int, float))
            or not np.isfinite(tv_max)
            or tv_max <= 0
        ):
            raise RuntimeError(
                "effective4 Gear must provenance a positive full-plan TV maximum"
            )
        if schema.get("continuous_plan_tv_quantile") != 0.99:
            raise RuntimeError(
                "effective4 Gear full-plan TV quantile must equal 0.99"
            )
    if planner_mode not in {"stock_cem", "projected_cem"}:
        raise RuntimeError(f"unsupported Gear planner mode {planner_mode!r}")
    return dict(planner)


def _decode_attr(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _strict_mask(
    held_pos: np.ndarray,
    held_quat: np.ndarray,
    fixed_pos: np.ndarray,
    fixed_quat: np.ndarray,
    *,
    max_height_m: float,
    max_xy_error_m: float,
    max_orientation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relative = held_pos - fixed_pos
    height = relative[:, 2]
    xy_error = np.linalg.norm(relative[:, :2], axis=1)
    denominator = np.linalg.norm(held_quat, axis=1) * np.linalg.norm(fixed_quat, axis=1)
    if not np.all(denominator > 0):
        raise ValueError("paired sidecar contains a zero-norm quaternion")
    dot = np.abs(np.sum(held_quat * fixed_quat, axis=1)) / denominator
    orientation_deg = np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))
    strict = (
        (height > 0.0)
        & (height <= max_height_m)
        & (xy_error <= max_xy_error_m)
        & (orientation_deg <= max_orientation_deg)
    )
    return strict, height, xy_error, orientation_deg


def load_paired_gear_cases(
    path: str | Path,
    *,
    count: int,
    seed: int,
    goal_offset_steps: int,
    master_offsets: Sequence[int] = DEFAULT_MASTER_OFFSETS,
    minimum_goal_step: int = 0,
    persistence_steps: int = 3,
    max_height_m: float = 0.003,
    max_xy_error_m: float = 0.003,
    max_orientation_deg: float = 5.0,
    legacy_source_ema_factor: float = 0.2,
    eligible_source_episodes: set[int] | None = None,
    selected_source_episodes: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a deterministic H1/H3/H5-paired sample from one master pool.

    A candidate's goal is *the episode's first strict frame*, that frame starts
    a strict run of at least ``persistence_steps``, and every goal-minus-master
    offset is non-successful.  Candidate construction and RNG selection are
    independent of ``goal_offset_steps``; therefore separate H1/H3/H5 runs with
    the same sidecar/count/seed receive exactly the same episode and goal row.
    """

    import h5py

    offsets_required = tuple(sorted({int(value) for value in master_offsets}))
    if count < 1:
        raise ValueError("count must be positive")
    if goal_offset_steps not in offsets_required:
        raise ValueError(
            f"goal_offset_steps={goal_offset_steps} is outside master_offsets={offsets_required}"
        )
    if not offsets_required or offsets_required[0] < 1:
        raise ValueError("master_offsets must contain positive offsets")
    if persistence_steps < 1:
        raise ValueError("persistence_steps must be positive")
    if minimum_goal_step < 0:
        raise ValueError("minimum_goal_step must be non-negative")

    path = Path(path)
    candidates: list[dict[str, Any]] = []
    with h5py.File(path, "r") as source:
        missing = ({"pixels", "action", "ep_len", "ep_offset"} | set(STATE_COLUMNS)) - set(source)
        if missing:
            raise ValueError(f"paired evaluation dataset is missing {sorted(missing)}")
        lengths = np.asarray(source["ep_len"][:], dtype=np.int64)
        episode_offsets = np.asarray(source["ep_offset"][:], dtype=np.int64)
        action_dim = int(source["action"].shape[1])
        semantics_attr = source.attrs.get("action_semantics")
        if semantics_attr is None:
            if action_dim != RAW_ACTION_DIM:
                raise ValueError("sidecar without action_semantics must use legacy 6-D actions")
            action_semantics = LEGACY_ACTION_SEMANTICS
            source_ema_factor = float(legacy_source_ema_factor)
        else:
            action_semantics = _decode_attr(semantics_attr)
            source_ema_factor = float(
                source.attrs.get(
                    "source_ema_factor",
                    1.0 if action_semantics == EFFECTIVE_ACTION_SEMANTICS else legacy_source_ema_factor,
                )
            )
        expected_dim = {
            EFFECTIVE_ACTION_SEMANTICS: EFFECTIVE_ACTION_DIM,
            LEGACY_ACTION_SEMANTICS: RAW_ACTION_DIM,
        }.get(action_semantics)
        if expected_dim is None or action_dim != expected_dim:
            raise ValueError(
                f"unsupported sidecar action semantics/dimension: {action_semantics!r}/{action_dim}"
            )

        available_restore = {
            name: width
            for name, width in RESTORE_STATE_COLUMNS.items()
            if name in source and source[name].shape[1:] == (width,)
        }
        direct_restore = set(available_restore) == set(RESTORE_STATE_COLUMNS)
        available_state = {**STATE_COLUMNS, **available_restore}

        for source_episode, (episode_offset, length) in enumerate(
            zip(episode_offsets, lengths)
        ):
            start = int(episode_offset)
            end = start + int(length)
            strict, height, xy_error, orientation_deg = _strict_mask(
                source["held_pos"][start:end],
                source["held_quat"][start:end],
                source["fixed_pos"][start:end],
                source["fixed_quat"][start:end],
                max_height_m=max_height_m,
                max_xy_error_m=max_xy_error_m,
                max_orientation_deg=max_orientation_deg,
            )
            strict_steps = np.flatnonzero(strict)
            if strict_steps.size == 0:
                continue
            goal_step = int(strict_steps[0])
            run_end = goal_step + persistence_steps
            if goal_step < max(offsets_required[-1], minimum_goal_step) or run_end > len(strict):
                continue
            if not bool(strict[goal_step:run_end].all()):
                # Never substitute a later strict run: that would violate the
                # first-strict goal contract.
                continue
            if any(bool(strict[goal_step - delta]) for delta in offsets_required):
                continue
            candidates.append(
                {
                    "source_episode": int(source_episode),
                    "goal_step": goal_step,
                    "goal_row": start + goal_step,
                    "goal_height_mm": float(height[goal_step] * 1000.0),
                    "goal_xy_mm": float(xy_error[goal_step] * 1000.0),
                    "goal_orientation_deg": float(orientation_deg[goal_step]),
                    "master_offset_initials": {
                        str(delta): {
                            "step": int(goal_step - delta),
                            "row": int(start + goal_step - delta),
                            "strict_success": bool(strict[goal_step - delta]),
                            "height_mm": float(height[goal_step - delta] * 1000.0),
                            "xy_mm": float(xy_error[goal_step - delta] * 1000.0),
                            "orientation_deg": float(
                                orientation_deg[goal_step - delta]
                            ),
                        }
                        for delta in offsets_required
                    },
                }
            )

        all_candidates = list(candidates)
        unfiltered_pool_array = np.asarray(
            [
                [item["source_episode"], item["goal_step"], item["goal_row"]]
                for item in all_candidates
            ],
            dtype="<i8",
        )
        if eligible_source_episodes is not None:
            eligible = {int(value) for value in eligible_source_episodes}
            candidates = [
                item for item in all_candidates
                if item["source_episode"] in eligible
            ]

        if selected_source_episodes is not None:
            requested = [int(value) for value in selected_source_episodes]
            if len(requested) != count:
                raise ValueError(
                    "explicit Gear source-episode count does not match count"
                )
            if len(set(requested)) != len(requested):
                raise ValueError("explicit Gear source episodes must be unique")
            by_episode = {
                int(item["source_episode"]): item for item in candidates
            }
            missing = [value for value in requested if value not in by_episode]
            if missing:
                raise ValueError(
                    "explicit Gear source episodes are outside the eligible pool: "
                    f"{missing}"
                )
            selected_candidates = [by_episode[value] for value in requested]
        elif len(candidates) < count:
            raise ValueError(
                f"only {len(candidates)} first-strict persistent goals are eligible for "
                f"{count} requested episodes"
            )
        else:
            selected_indices = np.random.default_rng(seed).choice(
                len(candidates), size=count, replace=False
            )
            selected_candidates = [
                candidates[int(index)] for index in selected_indices
            ]
        cases: list[dict[str, Any]] = []
        for selected_candidate in selected_candidates:
            metadata = dict(selected_candidate)
            episode = metadata["source_episode"]
            episode_row = int(episode_offsets[episode])
            init_step = metadata["goal_step"] - goal_offset_steps
            init_row = metadata["goal_row"] - goal_offset_steps
            strict_init, init_height, init_xy, init_orientation = _strict_mask(
                source["held_pos"][init_row : init_row + 1],
                source["held_quat"][init_row : init_row + 1],
                source["fixed_pos"][init_row : init_row + 1],
                source["fixed_quat"][init_row : init_row + 1],
                max_height_m=max_height_m,
                max_xy_error_m=max_xy_error_m,
                max_orientation_deg=max_orientation_deg,
            )
            if bool(strict_init[0]):
                raise RuntimeError("master-pool invariant broken: selected init is successful")
            metadata.update(
                init_step=int(init_step),
                init_row=int(init_row),
                init_height_mm=float(init_height[0] * 1000.0),
                init_xy_mm=float(init_xy[0] * 1000.0),
                init_orientation_deg=float(init_orientation[0]),
                restoration="full_state_v2" if direct_restore else "expert_prefix",
                source_action_semantics=action_semantics,
                source_ema_factor=source_ema_factor,
            )
            episode_actions = np.asarray(
                source["action"][episode_row : episode_row + int(length) - 1],
                dtype=np.float32,
            ).copy()
            if action_semantics == LEGACY_ACTION_SEMANTICS:
                episode_effective = filter_raw_episode_to_effective4(
                    episode_actions, ema_factor=source_ema_factor
                )
            else:
                episode_effective = episode_actions
            cases.append(
                {
                    "metadata": metadata,
                    "state": {
                        name: np.asarray(source[name][init_row], dtype=np.float32)
                        for name in available_state
                    },
                    "episode_initial_state": {
                        name: np.asarray(source[name][episode_row], dtype=np.float32)
                        for name in STATE_COLUMNS
                    },
                    "expert_prefix": np.asarray(
                        source["action"][episode_row:init_row], dtype=np.float32
                    ).copy(),
                    "expert_future_raw": np.asarray(
                        source["action"][init_row : episode_row + int(length) - 1],
                        dtype=np.float32,
                    ).copy(),
                    "expert_future_effective4": np.ascontiguousarray(
                        episode_effective[init_step:]
                    ),
                    "init_pixels": np.asarray(source["pixels"][init_row]).copy(),
                    "goal": np.asarray(source["pixels"][metadata["goal_row"]]).copy(),
                }
            )

    eligible_pool_array = np.asarray(
        [[item["source_episode"], item["goal_step"], item["goal_row"]] for item in candidates],
        dtype="<i8",
    )
    selected_array = np.asarray(
        [[case["metadata"]["source_episode"], case["metadata"]["goal_row"]] for case in cases],
        dtype="<i8",
    )
    pool_audit = {
        "master_offsets": list(offsets_required),
        "candidate_count": len(candidates),
        "unfiltered_candidate_count": len(all_candidates),
        "master_pool_sha256": hashlib.sha256(
            unfiltered_pool_array.tobytes()
        ).hexdigest(),
        "eligible_pool_sha256": hashlib.sha256(
            eligible_pool_array.tobytes()
        ).hexdigest(),
        "selection_sha256": hashlib.sha256(selected_array.tobytes()).hexdigest(),
        "selected_goal_rows": selected_array[:, 1].tolist(),
        "goal_contract": "episode_first_strict_frame_and_persists",
        "persistence_steps": int(persistence_steps),
        "source_action_semantics": action_semantics,
        "source_ema_factor": source_ema_factor,
        "restoration": "full_state_v2" if direct_restore else "expert_prefix",
    }
    return cases, pool_audit


def load_gear_expert_gate(
    path: str | Path,
    *,
    eval_dataset: str | Path,
    goal_offset_steps: int,
    eval_budget: int,
    seed: int,
    master_offsets: Sequence[int],
    action_protocol: str,
) -> tuple[set[int], dict[str, Any]]:
    """Load an exhaustive same-initialization Gear expert eligibility gate."""

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "policy": "expert",
        "eval_dataset": str(Path(eval_dataset).expanduser().resolve()),
        "goal_offset_steps": int(goal_offset_steps),
        "eval_budget": int(eval_budget),
        "seed": int(seed),
        "action_protocol": action_protocol,
        "paired_goal_selection": "episode_first_strict_frame_and_persists",
    }
    actual = {key: payload.get(key) for key in expected}
    # Early v2 expert-gate writers recorded the same protocol under the
    # mandatory action_contract block but omitted the duplicated top-level
    # field. Accept that schema without weakening the exact-action check below.
    if actual["action_protocol"] is None:
        action_contract = payload.get("action_contract")
        if isinstance(action_contract, dict):
            actual["action_protocol"] = action_contract.get("protocol")
    if actual != expected:
        raise ValueError(
            f"Gear expert gate protocol mismatch: expected {expected}, got {actual}"
        )
    pool = payload.get("paired_master_pool")
    if not isinstance(pool, dict):
        raise ValueError("Gear expert gate has no paired master-pool provenance")
    if pool.get("master_offsets") != sorted({int(x) for x in master_offsets}):
        raise ValueError("Gear expert gate master offsets do not match")
    if pool.get("goal_contract") != "episode_first_strict_frame_and_persists":
        raise ValueError("Gear expert gate uses the wrong goal contract")
    restoration_gate = payload.get("restoration_gate")
    if not isinstance(restoration_gate, dict) or restoration_gate.get("all_cases_passed") is not True:
        raise ValueError("Gear expert gate did not pass visual restoration checks")
    action_contract = payload.get("action_contract")
    if not isinstance(action_contract, dict) or action_contract.get("all_steps_exact") is not True:
        raise ValueError("Gear expert gate did not execute exact effective4 actions")
    records = payload.get("per_episode")
    if not isinstance(records, list) or not records:
        raise ValueError("Gear expert gate contains no episode records")
    if (
        int(payload.get("episodes", -1)) != len(records)
        or int(pool.get("candidate_count", -1)) != len(records)
        or int(pool.get("unfiltered_candidate_count", -1)) != len(records)
    ):
        raise ValueError(
            "Gear expert gate must exhaustively test every unfiltered master-pool case"
        )
    tested: list[int] = []
    eligible: set[int] = set()
    for record in records:
        source = record.get("paired_source") if isinstance(record, dict) else None
        if not isinstance(source, dict) or "source_episode" not in source:
            raise ValueError("Gear expert gate has an invalid paired episode record")
        episode = int(source["source_episode"])
        if episode in tested:
            raise ValueError(f"Gear expert gate repeats source episode {episode}")
        tested.append(episode)
        restoration = record.get("restoration")
        if not isinstance(restoration, dict):
            raise ValueError(f"Gear expert gate episode {episode} lacks restoration audit")
        if record.get("success") is True:
            eligible.add(episode)
    if not eligible:
        raise ValueError("Gear expert gate has no successful continuation")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return eligible, {
        "path": str(path),
        "sha256": digest,
        "master_pool_sha256": pool.get("master_pool_sha256"),
        "tested_source_episodes": tested,
        "eligible_source_episodes": sorted(eligible),
        "tested_count": len(tested),
        "eligible_count": len(eligible),
    }


def build_env_classes(args):
    """Build the env classes after Isaac Sim has started.

    Isaac's Python modules are only importable once ``AppLauncher`` has created
    the simulation app, so the class bodies cannot live at module import time.
    ``args`` supplies resolution, camera placement and strict-success thresholds
    and must match the collection run that produced the evaluation dataset.
    """

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCamera, TiledCameraCfg
    from isaaclab_tasks.direct.automate import automate_algo_utils as automate_algo
    from SRSA.tasks.direct.srsa.assembly_sparse_env import AssemblySparseEnv

    class GearAssemblyEvalEnv(AssemblySparseEnv):
        """Official sparse assembly env with a camera, set_state, and strict dones."""

        def __init__(self, *init_args, **init_kwargs):
            self.collection_post_step_frame = None
            self.collection_official_success = None
            self.collection_strict_success = None
            super().__init__(*init_args, **init_kwargs)
            # The collector aims the camera once immediately after reset and
            # then leaves that world pose stationary for the whole episode.
            # Keep the same episode-level target here.  Re-aiming from the
            # current/restored ``fixed_pos`` would silently change the camera
            # extrinsics for every paired mid-trajectory state.
            self._lewm_camera_target_local = self.fixed_pos.detach().clone()
            self._strict_persistence_steps = int(
                getattr(args, "strict_persistence_steps", 3)
            )
            if self._strict_persistence_steps < 1:
                raise ValueError("strict_persistence_steps must be positive")
            self._strict_success_streak = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.eval_last_height_m = torch.full(
                (self.num_envs,), float("nan"), device=self.device
            )
            self.eval_last_xy_error_m = self.eval_last_height_m.clone()
            self.eval_last_orientation_deg = self.eval_last_height_m.clone()
            self.eval_last_instant_success = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.eval_last_success_streak = torch.zeros_like(
                self._strict_success_streak
            )
            self.eval_last_applied_action = torch.zeros(
                (self.num_envs, RAW_ACTION_DIM),
                dtype=torch.float32,
                device=self.device,
            )

        def _setup_scene(self) -> None:
            super()._setup_scene()
            self.lewm_camera = TiledCamera(
                TiledCameraCfg(
                    prim_path="/World/envs/env_.*/LeWMDatasetCamera",
                    data_types=["rgb"],
                    width=args.resolution,
                    height=args.resolution,
                    update_period=0.0,
                    offset=TiledCameraCfg.OffsetCfg(),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=args.focal_length_mm,
                        focus_distance=0.20,
                        horizontal_aperture=20.955,
                        clipping_range=(0.01, 5.0),
                    ),
                )
            )
            self.scene.sensors["lewm_dataset_camera"] = self.lewm_camera

        # -- observation plumbing -------------------------------------------------

        def set_camera_target_local(self, env_ids, fixed_pos_local) -> None:
            """Set the stationary per-episode camera target in env-local space."""

            value = torch.as_tensor(
                fixed_pos_local,
                dtype=self.fixed_pos.dtype,
                device=self.device,
            ).reshape(len(env_ids), 3)
            self._lewm_camera_target_local[env_ids] = value

        def aim_camera_and_capture(self) -> np.ndarray:
            """Aim at the episode's initial socket pose and return one frame each.

            The collector sets this pose once after reset and its post-step
            frames retain those extrinsics.  Calling this method repeatedly is
            harmless: it reapplies the same stationary pose without deriving a
            new target from the current socket state.
            """

            target = (
                self._lewm_camera_target_local + self.scene.env_origins
            ).clone()
            target[:, 2] += args.camera_target_z_offset
            eye_offset = torch.tensor(
                args.camera_eye_offset, dtype=target.dtype, device=target.device
            )
            self.lewm_camera.set_world_poses_from_view(target + eye_offset, target)
            for _ in range(4):
                self.sim.render()
                self.scene.update(dt=self.physics_dt)
            return self._rgb()

        def _rgb(self) -> np.ndarray:
            rgb = self.lewm_camera.data.output["rgb"]
            return (
                rgb[..., :3].detach().to("cpu").numpy().astype(np.uint8, copy=False)
            )

        def strict_metrics(self):
            """Return the three geometric errors used by the strict contract."""
            relative = self.held_pos - self.fixed_pos
            height = relative[:, 2]
            xy_error = torch.linalg.norm(relative[:, :2], dim=1)
            dot = torch.abs((self.held_quat * self.fixed_quat).sum(dim=1))
            orientation_deg = torch.rad2deg(
                2.0 * torch.arccos(torch.clamp(dot, 0.0, 1.0))
            )
            return height, xy_error, orientation_deg

        def strict_success(self):
            """Instantaneous geometry predicate; persistence is applied in dones."""

            return self._strict_success_from_metrics(*self.strict_metrics())

        @staticmethod
        def _strict_success_from_metrics(height, xy_error, orientation_deg):
            return (
                (height > 0.0)
                & (height <= args.strict_max_height_m)
                & (xy_error <= args.strict_max_xy_error_m)
                & (orientation_deg <= args.strict_max_orientation_deg)
            )

        def _get_dones(self):
            """Terminate after strict insertion persists for the collection duration.

            The base class only ever reports a time-out, so ``World.terminateds``
            would otherwise be identically False and every evaluation would score
            zero regardless of the policy.  The collector accepts an episode only
            after the geometry predicate holds for the configured number of
            consecutive control steps; applying the same counter here keeps
            collection and evaluation on the same success contract.
            """

            _, time_out = super()._get_dones()
            height, xy_error, orientation_deg = self.strict_metrics()
            instant = self._strict_success_from_metrics(
                height, xy_error, orientation_deg
            )
            self._strict_success_streak = torch.where(
                instant,
                self._strict_success_streak + 1,
                torch.zeros_like(self._strict_success_streak),
            )

            # DirectRLEnv auto-resets terminated environments before step()
            # returns.  Preserve the just-evaluated state so the client never
            # reports the following reset pose as the terminal pose.
            self.eval_last_height_m = height.detach().clone()
            self.eval_last_xy_error_m = xy_error.detach().clone()
            self.eval_last_orientation_deg = orientation_deg.detach().clone()
            self.eval_last_instant_success = instant.detach().clone()
            self.eval_last_success_streak = self._strict_success_streak.detach().clone()
            self.eval_last_applied_action = self.actions.detach().clone()

            terminated = self._strict_success_streak >= self._strict_persistence_steps
            return terminated, time_out

        def _reset_idx(self, env_ids) -> None:
            super()._reset_idx(env_ids)
            if hasattr(self, "_lewm_camera_target_local"):
                self._lewm_camera_target_local[env_ids] = self.fixed_pos[
                    env_ids
                ].detach()
            # During base-class construction this override can be reached before
            # the evaluation-only buffer is allocated, hence the guard.
            if hasattr(self, "_strict_success_streak"):
                self._strict_success_streak[env_ids] = 0

        def _get_rewards(self):
            official = automate_algo.check_plug_inserted_in_socket(
                self.held_pos,
                self.fixed_pos,
                self.disassembly_dists,
                self.keypoints_held,
                self.keypoints_fixed,
                self.cfg_task.close_error_thresh,
                self.episode_length_buf,
            )
            reward = super()._get_rewards()
            self.collection_official_success = official.detach().clone()
            self.collection_strict_success = self.strict_success().detach().clone()
            rgb = self.lewm_camera.data.output.get("rgb")
            self.collection_post_step_frame = (
                None if rgb is None else rgb[..., :3].detach().clone()
            )
            return reward

        # -- deterministic teleport ----------------------------------------------

        def set_state(self, env_ids, state: dict) -> None:
            """Write a recorded state directly, bypassing the randomized reset.

            ``AssemblyEnv._reset_idx`` reaches its start pose through a stochastic
            multi-stage routine (randomize socket, randomize plug, servo the
            gripper to a grasp, close the fingers). It cannot be replayed, so an
            evaluation that must begin from a specific recorded frame has to
            write the state itself. Poses arrive env-local, exactly as
            ``EvalTrajectoryHDF5Writer`` stored them.
            """

            missing = set(STATE_COLUMNS) - set(state)
            if missing:
                raise ValueError(f"set_state is missing columns: {sorted(missing)}")

            def _t(name: str):
                width = FULL_STATE_COLUMNS[name]
                value = torch.as_tensor(
                    np.asarray(state[name], dtype=np.float32), device=self.device
                )
                value = value.reshape(-1, width)
                if value.shape[0] != len(env_ids):
                    raise ValueError(
                        f"'{name}' has {value.shape[0]} rows for {len(env_ids)} envs"
                    )
                return value

            origins = self.scene.env_origins[env_ids]
            zero_velocity = torch.zeros((len(env_ids), 6), device=self.device)

            fixed_pose = torch.cat((_t("fixed_pos") + origins, _t("fixed_quat")), dim=1)
            self._fixed_asset.write_root_pose_to_sim(fixed_pose, env_ids=env_ids)
            fixed_velocity = (
                _t("fixed_root_vel") if "fixed_root_vel" in state else zero_velocity
            )
            self._fixed_asset.write_root_velocity_to_sim(
                fixed_velocity, env_ids=env_ids
            )

            held_pose = torch.cat((_t("held_pos") + origins, _t("held_quat")), dim=1)
            self._held_asset.write_root_pose_to_sim(held_pose, env_ids=env_ids)
            held_velocity = (
                _t("held_root_vel") if "held_root_vel" in state else zero_velocity
            )
            self._held_asset.write_root_velocity_to_sim(
                held_velocity, env_ids=env_ids
            )

            joint_pos, joint_vel = _t("robot_joint_pos"), _t("robot_joint_vel")
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            self.ctrl_target_joint_pos[env_ids] = joint_pos
            self.ctrl_target_gripper_dof_pos = 0.0

            # The observation frame carries a per-episode noise term; without it
            # the policy sees a differently-biased socket than the recording did.
            self.fixed_pos_action_frame[env_ids] = _t("fixed_pos_action_frame")
            self.init_fixed_pos_obs_noise[env_ids] = _t("init_fixed_pos_obs_noise")

            self.episode_length_buf[env_ids] = 0
            self.ep_succeeded[env_ids] = False
            self.actions[env_ids] = 0.0
            self.prev_actions[env_ids] = 0.0
            self._strict_success_streak[env_ids] = 0

            self.scene.write_data_to_sim()
            self.sim.forward()
            self._compute_intermediate_values(dt=self.physics_dt)

            # v2 rows restore every controller/finite-difference field.  Legacy
            # rows take the conservative zero-velocity fallback here and are
            # restored by expert-prefix replay in the evaluation client.
            tensor_fields = {
                "action_ema": "actions",
                "prev_action_ema": "prev_actions",
                "ctrl_target_joint_pos": "ctrl_target_joint_pos",
                "ctrl_target_fingertip_midpoint_pos": "ctrl_target_fingertip_midpoint_pos",
                "ctrl_target_fingertip_midpoint_quat": "ctrl_target_fingertip_midpoint_quat",
                "prev_joint_pos": "prev_joint_pos",
                "prev_fingertip_pos": "prev_fingertip_pos",
                "prev_fingertip_quat": "prev_fingertip_quat",
                "ee_linvel_fd": "ee_linvel_fd",
                "ee_angvel_fd": "ee_angvel_fd",
                "joint_vel_fd": "joint_vel_fd",
                "task_prop_gains": "task_prop_gains",
                "task_deriv_gains": "task_deriv_gains",
            }
            for column, attribute in tensor_fields.items():
                target = getattr(self, attribute)
                if column in state:
                    # Some AutoMate gain tensors are constructed from integer
                    # config literals and therefore have an integer dtype,
                    # while every sidecar column is intentionally stored as
                    # float32.  Restore in the destination tensor's native
                    # dtype; otherwise PyTorch rejects the indexed assignment
                    # before the first v2 evaluation case can run.
                    target[env_ids] = _t(column).to(dtype=target.dtype)
                elif column == "prev_joint_pos":
                    target[env_ids] = self.joint_pos[env_ids, :7]
                elif column == "prev_fingertip_pos":
                    target[env_ids] = self.fingertip_midpoint_pos[env_ids]
                elif column == "prev_fingertip_quat":
                    target[env_ids] = self.fingertip_midpoint_quat[env_ids]
                elif column in {"ee_linvel_fd", "ee_angvel_fd", "joint_vel_fd"}:
                    target[env_ids] = 0.0

            if "ctrl_target_gripper_dof_pos" in state:
                gripper = _t("ctrl_target_gripper_dof_pos").reshape(-1)
                if not torch.allclose(gripper, gripper[:1]):
                    raise ValueError(
                        "AutoMate exposes one shared gripper target; restored rows disagree"
                    )
                self.ctrl_target_gripper_dof_pos = float(gripper[0].item())
            else:
                self.ctrl_target_gripper_dof_pos = 0.0
            self._robot.set_joint_position_target(
                self.ctrl_target_joint_pos[env_ids], env_ids=env_ids
            )

    return GearAssemblyEvalEnv


class GearEnvFacade:
    """Per-environment view of the shared batched world.

    ``_apply_callables`` reaches the simulator as
    ``world.envs.envs[i].unwrapped.<method>(...)``, so this exposes ``set_state``
    with the dataset's own column names and forwards to slot ``index``.
    """

    def __init__(self, pool: "IsaacGearEnvPool", index: int):
        self._pool = pool
        self._index = index

    @property
    def unwrapped(self) -> "GearEnvFacade":
        return self

    @property
    def action_space(self):
        return self._pool.single_action_space

    @property
    def observation_space(self):
        return self._pool.single_observation_space

    def set_state(self, **state: Any) -> None:
        """Teleport just this environment. Extra keyword arguments are ignored."""

        available = {
            name: np.asarray(state[name])[None, ...]
            for name in FULL_STATE_COLUMNS
            if name in state
        }
        self._pool.backend.set_state(
            [self._index],
            available,
        )

    def close(self) -> None:  # the pool owns the shared simulation app
        pass


class IsaacGearEnvPool:
    """``EnvPool``-compatible façade over one batched Isaac Lab world."""

    def __init__(self, backend, num_envs: int, action_dim: int = 6):
        import gymnasium as gym
        from gymnasium.vector.utils import batch_space

        self.backend = backend
        self._num_envs = int(num_envs)
        self.envs = [GearEnvFacade(self, i) for i in range(self._num_envs)]
        self.seeds = np.zeros(self._num_envs, dtype=np.int64)
        self._single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self._single_observation_space = gym.spaces.Box(
            low=0, high=255, shape=(backend.resolution, backend.resolution, 3),
            dtype=np.uint8,
        )
        self._action_space = batch_space(self._single_action_space, self._num_envs)
        self._observation_space = batch_space(
            self._single_observation_space, self._num_envs
        )
        self._stacked_infos: dict[str, Any] | None = None

    # -- EnvPool surface -----------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def action_space(self):
        return self._action_space

    @property
    def single_action_space(self):
        return self._single_action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def single_observation_space(self):
        return self._single_observation_space

    @property
    def variation_space(self):
        return None

    @property
    def single_variation_space(self):
        return None

    def _infos(self) -> dict[str, Any]:
        # (N, 1, H, W, 3): the leading time axis World and MegaWrapper assume.
        pixels = self.backend.observe()
        self._stacked_infos = {"pixels": pixels[:, None, ...]}
        return self._stacked_infos

    def reset(self, seed=None, options=None, mask=None):
        """Reset the whole world. Masked partial resets are not supported.

        Every environment shares one physics scene and one episode clock here, so
        a partial reset would silently desynchronize them. ``World.evaluate``'s
        dataset path only ever resets everything at once before teleporting.
        """

        if mask is not None and not np.all(mask):
            raise NotImplementedError(
                "IsaacGearEnvPool resets all environments together; "
                "a masked reset would desynchronize the shared physics scene"
            )
        if isinstance(seed, (int, np.integer)):
            self.seeds[:] = [int(seed) + i for i in range(self._num_envs)]
        elif isinstance(seed, (list, np.ndarray)):
            for i, value in enumerate(seed[: self._num_envs]):
                if value is not None:
                    self.seeds[i] = int(value)
        self.backend.reset()
        return None, self._infos()

    def step(self, actions, mask=None):
        """Step every environment with one batched physics step."""

        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape[0] != self._num_envs:
            raise ValueError(
                f"expected {self._num_envs} actions, got {actions.shape[0]}"
            )
        if mask is not None and not np.all(mask):
            # Freeze masked envs by commanding no motion rather than skipping
            # them: the batched scene has to advance as a whole.
            actions = actions.copy()
            actions[~np.asarray(mask, dtype=bool)] = 0.0
        rewards, terminateds, truncateds = self.backend.step(actions)
        return None, rewards, terminateds, truncateds, self._infos()

    def close(self) -> None:
        self.backend.close()


class GearAssemblyBackend:
    """Owns the one batched Isaac Lab world the whole pool shares."""

    def __init__(self, env, args):
        self._env = env
        self._args = args
        self.resolution = int(args.resolution)
        import torch

        self._torch = torch
        clip = getattr(args, "clip_actions", None)
        self._clip = None if clip is None else float(clip)
        self.last_action_consistency: list[dict[str, Any]] | None = None

    @property
    def env(self):
        return self._env

    def reset(self) -> None:
        self._env.reset()
        self._env.aim_camera_and_capture()

    def set_state(self, env_ids, state: dict) -> None:
        import torch

        ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self._env.device)
        self._env.set_state(ids, state)

    def observe(self) -> np.ndarray:
        """Read frames with the stationary per-episode camera extrinsics."""

        return self._env.aim_camera_and_capture()

    def step(self, actions: np.ndarray):
        torch = self._torch
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"gear actions must have shape [N,D], got {actions.shape}")
        protocol = "effective4" if actions.shape[1] == EFFECTIVE_ACTION_DIM else "legacy_raw6"
        prepared = [
            prepare_model_action(action, protocol=protocol, clip=self._clip)
            for action in actions
        ]
        command_rows = np.stack([item[1] for item in prepared])
        command = torch.as_tensor(
            command_rows, dtype=torch.float32, device=self._env.device
        )
        with torch.inference_mode():
            pass
        _, _, terminated, truncated, _ = self._env.step(command)
        applied = self._env.actions.detach().to("cpu").numpy()
        self.last_action_consistency = [
            complete_action_consistency_audit(item[2], applied[row])
            for row, item in enumerate(prepared)
        ]
        rewards = np.zeros(len(actions), dtype=np.float32)
        return (
            rewards,
            terminated.detach().to("cpu").numpy().astype(bool),
            truncated.detach().to("cpu").numpy().astype(bool),
        )

    def close(self) -> None:
        self._env.close()


def build_world(backend, num_envs: int, action_dim: int = 6):
    """Return a ``swm.World`` driving the batched Isaac pool.

    ``World.__init__`` would ``gym.make`` one environment per slot, which Isaac
    cannot provide, so the pool is installed directly and the five attributes
    ``World`` initializes are set by hand. Every method that matters —
    ``evaluate``, ``_evaluate_from_dataset``, ``_run``, ``set_policy`` — only
    touches ``self.envs`` and those attributes, so they work unchanged.
    """

    import stable_worldmodel as swm

    world = swm.World.__new__(swm.World)
    world.envs = IsaacGearEnvPool(backend, num_envs, action_dim=action_dim)
    world.policy = None
    world.infos = {}
    world.rewards = None
    world.terminateds = None
    world.truncateds = None
    return world
