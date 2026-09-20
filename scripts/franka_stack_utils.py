"""Pure data/timing helpers shared by Franka Stack conversion and evaluation.

This module deliberately has no Isaac Lab dependency so its alignment and success
semantics can be unit-tested in the normal project environment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

XY_THRESHOLD = 0.04
HEIGHT_DIFF = 0.0468
HEIGHT_THRESHOLD = 0.005
GRIPPER_OPEN_VAL = 0.04
GRIPPER_ATOL = 1e-4
GOAL_POSITION_THRESHOLD = 0.04
SPLIT_NAMES = ("train", "val", "eval")


def validate_franka_lewm_distance_contract(
    *, offset: int, budget: int, action_block: int, horizon: int
) -> None:
    """Require the Cube-style goal distance and two-pass evaluation budget."""

    expected_offset = int(action_block) * int(horizon)
    if int(offset) != expected_offset:
        raise ValueError(
            "formal LeWM start/goal distance must equal planner horizon * "
            f"action block: offset={offset}, expected {expected_offset}"
        )
    if int(budget) != 2 * int(offset):
        raise ValueError(
            f"formal LeWM budget must equal 2 * offset={2 * int(offset)}, got {budget}"
        )


def prefix_start_index(start: int, init_mode: str, prefix_steps: int) -> int:
    """Choose the source row from which an evaluation initialization starts."""
    if start < 0 or prefix_steps < 0:
        raise ValueError("start and prefix_steps must be non-negative")
    if init_mode == "teleport":
        return start
    if init_mode == "expert_prefix":
        return 0
    if init_mode == "hybrid_prefix":
        return max(0, start - prefix_steps)
    raise ValueError(f"unknown init_mode {init_mode!r}")


def deterministic_episode_splits(
    num_demos: int, seed: int = 3072, train_fraction: float = 0.8, val_fraction: float = 0.1
) -> np.ndarray:
    """Return split ids (0=train, 1=val, 2=eval), indexed by source demo id."""
    if num_demos <= 0:
        raise ValueError("num_demos must be positive")
    if train_fraction < 0 or val_fraction < 0 or train_fraction + val_fraction > 1:
        raise ValueError("fractions must be non-negative and train_fraction + val_fraction <= 1")
    order = np.random.default_rng(seed).permutation(num_demos)
    n_train = int(np.floor(num_demos * train_fraction))
    n_val = int(np.floor(num_demos * val_fraction))
    assignment = np.full(num_demos, 2, dtype=np.uint8)
    assignment[order[:n_train]] = 0
    assignment[order[n_train : n_train + n_val]] = 1
    return assignment


def observation_state_location(t: int) -> tuple[str, int]:
    """Return the raw-HDF5 state record corresponding to observation row ``t``.

    The recorder stores ``initial_state=s_0`` and ``states[t]=s_(t+1)``.
    """
    if t < 0:
        raise ValueError(f"observation index must be non-negative, got {t}")
    return ("initial_state", 0) if t == 0 else ("states", t - 1)


def cubes_stacked(cube_positions: np.ndarray, gripper_pos: np.ndarray, stage: int = 3) -> np.ndarray:
    """Vectorized strict stack predicate, matching Isaac Lab for stage 3."""
    cube_positions = np.asarray(cube_positions)
    gripper_pos = np.asarray(gripper_pos)
    if cube_positions.ndim != 2 or cube_positions.shape[1] != 9:
        raise ValueError(f"cube_positions must have shape (T, 9), got {cube_positions.shape}")
    if gripper_pos.ndim != 2 or gripper_pos.shape[1] != 2:
        raise ValueError(f"gripper_pos must have shape (T, 2), got {gripper_pos.shape}")
    if stage not in (2, 3):
        raise ValueError(f"stage must be 2 or 3, got {stage}")
    cubes = cube_positions.reshape(-1, 3, 3)
    ok = np.ones(len(cubes), dtype=bool)
    for lower_i, upper_i in ((0, 1), (1, 2))[: stage - 1]:
        delta = cubes[:, lower_i] - cubes[:, upper_i]
        ok &= np.linalg.norm(delta[:, :2], axis=1) < XY_THRESHOLD
        ok &= (np.abs(delta[:, 2]) - HEIGHT_DIFF) < HEIGHT_THRESHOLD
        ok &= delta[:, 2] < 0.0
    ok &= np.all(
        np.isclose(np.abs(gripper_pos), GRIPPER_OPEN_VAL, atol=GRIPPER_ATOL, rtol=GRIPPER_ATOL),
        axis=1,
    )
    return ok


def goal_stack_diagnostics(
    cube_positions: np.ndarray,
    gripper_pos: np.ndarray,
    goal_cube_positions: np.ndarray,
    stage: int = 3,
    goal_position_threshold: float = GOAL_POSITION_THRESHOLD,
) -> dict[str, object]:
    """Diagnose strict structure plus identity-preserving proximity to a goal state.

    Generic ``cubes_stacked`` accepts the right tower at any table location.  A
    goal-conditioned metric must additionally require each participating named cube
    to be close to its pose in the selected goal.  Orientation is intentionally not
    included because it is absent from Isaac Lab's task success definition.
    """
    current = np.asarray(cube_positions, dtype=np.float64).reshape(3, 3)
    goal = np.asarray(goal_cube_positions, dtype=np.float64).reshape(3, 3)
    grip = np.asarray(gripper_pos, dtype=np.float64).reshape(2)
    out: dict[str, object] = {}
    pair_ok = []
    for lower_i, upper_i in ((0, 1), (1, 2)):
        tag = f"{lower_i + 1}{upper_i + 1}"
        delta = current[lower_i] - current[upper_i]
        xy = float(np.linalg.norm(delta[:2]))
        h_error = float(abs(delta[2]) - HEIGHT_DIFF)
        order_ok = bool(delta[2] < 0.0)
        this_ok = bool(xy < XY_THRESHOLD and h_error < HEIGHT_THRESHOLD and order_ok)
        out.update({f"xy{tag}": xy, f"h{tag}": h_error, f"dz{tag}": float(delta[2]),
                    f"pair{tag}_ok": this_ok})
        pair_ok.append(this_ok)
    goal_dist = np.linalg.norm(current - goal, axis=1)
    for i, dist in enumerate(goal_dist, 1):
        out[f"goal_dist_{i}"] = float(dist)
    active = stage
    structure_ok = bool(all(pair_ok[: stage - 1]))
    gripper_open = bool(np.all(np.isclose(np.abs(grip), GRIPPER_OPEN_VAL,
                                                  atol=GRIPPER_ATOL, rtol=GRIPPER_ATOL)))
    goal_position_ok = bool(np.all(goal_dist[:active] <= goal_position_threshold))
    out.update(
        finger=[float(x) for x in grip],
        structure_ok=structure_ok,
        gripper_open=gripper_open,
        goal_position_ok=goal_position_ok,
        goal_max_dist=float(np.max(goal_dist[:active])),
        goal_position_threshold=float(goal_position_threshold),
        success=bool(structure_ok and gripper_open and goal_position_ok),
    )
    return out


def validate_franka_planner_config(
    config: dict,
    expected_action_block: int,
    *,
    expected_horizon: int | None = None,
    expected_receding_horizon: int | None = None,
    expected_planner_mode: str | None = None,
    expected_plan_bank_continuation: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the server provenance required by the Franka Stack client."""
    if not isinstance(config, dict):
        raise ValueError("planner_config is missing or is not a dictionary")
    if config.get("action_dim") != 7:
        raise ValueError(f"planner action_dim must be 7, got {config.get('action_dim')!r}")
    if config.get("action_block") != expected_action_block:
        raise ValueError(
            f"planner action_block must equal client action_block={expected_action_block}, "
            f"got {config.get('action_block')!r}"
        )
    if expected_horizon is not None and config.get("horizon") != expected_horizon:
        raise ValueError(
            f"planner horizon must be {expected_horizon}, got {config.get('horizon')!r}"
        )
    if (
        expected_receding_horizon is not None
        and config.get("receding_horizon") != expected_receding_horizon
    ):
        raise ValueError(
            "planner receding_horizon must equal the formal full-commit value "
            f"{expected_receding_horizon}, got {config.get('receding_horizon')!r}"
        )
    if (
        expected_planner_mode is not None
        and config.get("planner_mode") != expected_planner_mode
    ):
        raise ValueError(
            f"planner mode must be {expected_planner_mode!r}, "
            f"got {config.get('planner_mode')!r}"
        )
    if (
        expected_plan_bank_continuation is not None
        and config.get("plan_bank_continuation")
        != expected_plan_bank_continuation
    ):
        raise ValueError(
            "planner bank continuation does not match the requested formal arm"
        )
    schema = config.get("task_action_schema")
    if not isinstance(schema, dict) or schema.get("name") != "franka_stack":
        raise ValueError(f"planner task_action_schema.name must be 'franka_stack', got {schema!r}")
    if schema.get("action_dim") != 7 or schema.get("action_block") != expected_action_block:
        raise ValueError("nested Franka action schema dimensions do not match the client")
    low = np.asarray(schema.get("raw_low"), dtype=np.float64)
    high = np.asarray(schema.get("raw_high"), dtype=np.float64)
    if low.shape != (7,) or high.shape != (7,):
        raise ValueError(f"Franka raw bounds must have shape (7,), got {low.shape}/{high.shape}")
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
        raise ValueError("Franka raw action bounds are invalid")
    categorical = schema.get("raw_categorical", {})
    if set(map(float, categorical.get("6", ()))) != {-1.0, 1.0}:
        raise ValueError("Franka schema must provenance raw gripper categories {-1,+1} at dim 6")
    if schema.get("single_switch_categorical") != [6]:
        raise ValueError(
            "Franka schema must jointly project each five-step gripper macro "
            "to at most one open/close transition"
        )
    if schema.get("global_single_switch_categorical") != [6]:
        raise ValueError(
            "Franka schema must project gripper switching over the complete plan"
        )
    if schema.get("terminal_open_steps") != 6:
        raise ValueError(
            "Franka schema must leave six commands for strict physical opening"
        )
    if schema.get("terminal_switch_steps") != [16, 18, 19]:
        raise ValueError(
            "Franka schema must use the exact train terminal splits compatible "
            "with strict physical opening"
        )
    if schema.get("terminal_switch_counts") != {
        "16": 1,
        "18": 2,
        "19": 795,
    } or schema.get("terminal_support_sample_count") != 798:
        raise ValueError("Franka terminal split provenance does not match train-only data")
    latch = schema.get("episode_open_latch")
    if not isinstance(latch, dict) or latch.get("enabled") is not True:
        raise ValueError("Franka planner must enable the episode-global open latch")
    if (
        latch.get("trigger")
        != "client_feedback_stack_structure_and_goal_pose_ready"
        or latch.get("scope") != "episode_until_reset"
        or latch.get("projected_raw_value") != 1.0
        or latch.get("applies_before")
        != [
            "model_scoring",
            "elite_update",
            "finalist_scoring",
            "returned_plan",
        ]
    ):
        raise ValueError("Franka episode-global open latch contract is invalid")
    if expected_planner_mode is not None and (
        latch.get("activation") != "next_natural_replan_boundary"
        or latch.get("interrupts_committed_plan") is not False
    ):
        raise ValueError(
            "Franka feedback latch must preserve the committed 25-step plan"
        )
    if (
        latch.get("train_plan_length") != 25
        or latch.get("train_clip_stride") != 5
        or latch.get("train_sampled_plan_count") != 35308
        or latch.get("train_all_open_plan_count") != 9699
    ):
        raise ValueError("Franka all-open latch support provenance is invalid")
    if (
        latch.get("train_post_success_episode_count") != 800
        or latch.get("train_post_success_min_steps") != 5
        or latch.get("train_post_success_max_steps") != 9
        or latch.get("train_post_success_action_count") != 6397
        or latch.get("train_post_success_open_action_count") != 6397
        or latch.get("strict_minimum_open_commands") != 6
        or latch.get("train_observed_terminal_switch_steps") != [16, 18, 19, 20]
        or latch.get("train_observed_terminal_switch_counts")
        != {"16": 1, "18": 2, "19": 795, "20": 2}
        or latch.get("strict_ineligible_terminal_switch_counts") != {"20": 2}
    ):
        raise ValueError("Franka post-success open continuation provenance is invalid")
    if schema.get("continuous_action_indices") != list(range(6)):
        raise ValueError(
            "Franka schema must project plan-TV over all six arm dimensions"
        )
    tv_max = schema.get("continuous_plan_tv_max")
    tv_quantile = schema.get("continuous_plan_tv_quantile")
    if (
        not isinstance(tv_max, (int, float))
        or not np.isfinite(tv_max)
        or tv_max <= 0
    ):
        raise ValueError("Franka schema must provenance a positive continuous plan-TV maximum")
    if tv_quantile != 0.99:
        raise ValueError(
            f"Franka schema continuous plan-TV quantile must be 0.99, got {tv_quantile!r}"
        )
    if expected_planner_mode is not None:
        proposal = schema.get("plan_proposal")
        if not isinstance(proposal, dict):
            raise ValueError("Franka planner must provenance its plan proposal")
        if expected_planner_mode == "projected_cem":
            if proposal != {
                "mode": "projected_cem",
                "selection": "iterative_model_cost",
                "synthetic_cem_candidates": True,
                "world_model_reranks_candidates": True,
            }:
                raise ValueError("Franka projected-CEM proposal provenance is invalid")
        else:
            prefix = "train_terminal_"
            if not expected_planner_mode.startswith(prefix):
                raise ValueError(f"unknown Franka planner mode {expected_planner_mode!r}")
            expected_selection = expected_planner_mode[len(prefix) :]
            expected_goal_conditioned = expected_selection in {
                "behavior_retrieval", "wm_rerank"
            }
            if (
                proposal.get("mode") != "train_terminal_plan_bank"
                or proposal.get("schema_version")
                != "franka-train-terminal-plan-bank-v1"
                or proposal.get("selection") != expected_selection
                or proposal.get("continuation")
                != expected_plan_bank_continuation
                or proposal.get("synthetic_cem_candidates") is not False
                or proposal.get("world_model_reranks_candidates")
                is not (expected_selection == "wm_rerank")
                or proposal.get("goal_conditioned_retrieval")
                is not expected_goal_conditioned
            ):
                raise ValueError("Franka train terminal proposal mode is invalid")
            if (
                proposal.get("selected_split") != "train"
                or proposal.get("split_seed") != 3072
                or proposal.get("train_fraction") != 0.8
                or proposal.get("val_fraction") != 0.1
                or proposal.get("action_alignment")
                != "obs[t]=s_t, action[t]=a_t"
                or proposal.get("horizon") != 5
                or proposal.get("action_block") != 5
                or proposal.get("raw_plan_steps") != 25
                or proposal.get("action_dim") != 7
                or proposal.get("minimum_open_commands") != 6
                or proposal.get("allowed_terminal_switch_steps") != [16, 18, 19]
            ):
                raise ValueError("Franka train terminal bank data contract is invalid")
            if (
                proposal.get("observed_strict_candidate_count") != 798
                or proposal.get("excluded_by_raw_bounds") != 0
                or proposal.get("excluded_by_continuous_plan_tv") != 2
                or proposal.get("candidate_count") != 796
                or proposal.get("source_demo_count") != 796
                or proposal.get("candidate_macro_shape") != [796, 5, 35]
                or proposal.get("candidate_raw_shape") != [796, 25, 7]
                or proposal.get("terminal_switch_counts")
                != {"16": 1, "18": 2, "19": 793}
            ):
                raise ValueError("Franka executable terminal bank count is invalid")
            if proposal.get("raw_support_low") != schema.get("raw_low") or proposal.get(
                "raw_support_high"
            ) != schema.get("raw_high"):
                raise ValueError("Franka plan-bank and planner raw bounds disagree")
            if not np.isclose(
                proposal.get("continuous_plan_tv_max", np.nan), tv_max
            ) or proposal.get(
                "accepted_continuous_plan_tv_observed_max", np.inf
            ) > tv_max:
                raise ValueError("Franka plan-bank TV support provenance is invalid")
            source_ids = proposal.get("source_demo_ids")
            plan_hashes = proposal.get("candidate_plan_sha256")
            if (
                not isinstance(source_ids, list)
                or len(source_ids) != 796
                or len(set(map(int, source_ids))) != 796
                or not isinstance(plan_hashes, list)
                or len(plan_hashes) != 796
                or len(set(plan_hashes)) != 796
                or any(not _is_sha256(value) for value in plan_hashes)
                or not _is_sha256(proposal.get("source_demo_ids_sha256"))
                or not _is_sha256(proposal.get("bank_sha256"))
                or not _is_sha256(proposal.get("endpoint_pixels_sha256"))
            ):
                raise ValueError("Franka terminal bank identity provenance is invalid")
            if expected_goal_conditioned:
                scale = proposal.get("retrieval_scale")
                if (
                    proposal.get("retrieval_topk") != 300
                    or proposal.get("retrieval_distance")
                    != "projected_encoder_squared_l2_start_plus_goal"
                    or not isinstance(scale, dict)
                    or scale.get("fit")
                    != "train_leave_one_out_nearest_neighbor_median"
                    or not np.isfinite(scale.get("start", np.nan))
                    or scale.get("start", 0) <= 0
                    or not np.isfinite(scale.get("goal", np.nan))
                    or scale.get("goal", 0) <= 0
                    or not _is_sha256(
                        proposal.get("retrieval_embeddings_sha256")
                    )
                ):
                    raise ValueError("Franka endpoint retrieval provenance is invalid")
            continuation = proposal.get("continuation_provenance")
            if expected_plan_bank_continuation == "abstain":
                if (
                    not isinstance(continuation, dict)
                    or continuation.get("mode") != "abstain"
                    or continuation.get("synthetic_action_sequence") is not False
                    or continuation.get("exact_train_window_match_count") is not None
                ):
                    raise ValueError("Franka abstention provenance is invalid")
            elif expected_plan_bank_continuation == "open_hold":
                if (
                    not isinstance(continuation, dict)
                    or continuation.get("mode") != "open_hold"
                    or continuation.get("synthetic_action_sequence") is not True
                    or continuation.get("exact_train_window_match_count") != 0
                ):
                    raise ValueError("Franka synthetic continuation provenance is invalid")
            if (
                config.get("num_samples") != 796
                or config.get("cem_steps") != 0
                or config.get("topk") != 1
            ):
                raise ValueError("Franka plan-bank effective solver sizes are invalid")
    return low, high


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def validate_franka_action(action, raw_low: np.ndarray, raw_high: np.ndarray) -> np.ndarray:
    """Validate one raw environment action returned by the planner."""
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (7,):
        raise ValueError(f"Franka action must have shape (7,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Franka action contains non-finite values: {array}")
    low = np.asarray(raw_low)
    high = np.asarray(raw_high)
    if np.any(array[:6] < low[:6]) or np.any(array[:6] > high[:6]):
        raise ValueError(f"Franka arm action is outside provenance raw bounds: {array[:6]}")
    if float(array[6]) not in (-1.0, 1.0):
        raise ValueError(f"Franka gripper action must be bit-exact -1 or +1, got {array[6]!r}")
    return array


def load_expert_gate(
    path: str | Path,
    *,
    task: str,
    stage: int,
    offset: int,
    budget: int,
    source_dataset: str | Path,
    dataset_split: str,
    split_seed: int,
    train_fraction: float,
    val_fraction: float,
) -> tuple[set[int], dict[str, object]]:
    """Load the exact-teleport expert eligibility gate for formal evaluation."""

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text())
    expected = {
        "task": task,
        "policy": "expert",
        "stage": int(stage),
        "offset": int(offset),
        "budget": int(budget),
        "source_dataset": str(Path(source_dataset).expanduser().resolve()),
        "init_mode": "teleport",
        "dataset_split": dataset_split,
        "split_seed": int(split_seed),
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"expert gate protocol mismatch: expected {expected}, got {actual}"
        )
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or not (
        protocol.get("formal_init_mode") == "teleport"
        and protocol.get("requires_same_init_expert_continuation_gate") is True
        and protocol.get("diagnostic_only") is False
    ):
        raise ValueError("expert gate is not a formal exact-teleport run")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("expert gate contains no episodes")
    all_ids: list[int] = []
    eligible: set[int] = set()
    for record in episodes:
        if not isinstance(record, dict) or "demo" not in record:
            raise ValueError("expert gate contains an invalid episode record")
        demo = int(record["demo"])
        if demo in all_ids:
            raise ValueError(f"expert gate repeats demo {demo}")
        all_ids.append(demo)
        if record.get("init_mode") != "teleport":
            raise ValueError(f"expert gate demo {demo} did not use teleport")
        if int(record.get("budget", -1)) != int(budget):
            raise ValueError(f"expert gate demo {demo} used a different budget")
        if int(record.get("start", -1)) != int(record.get("goal_idx", -1)) - int(offset):
            raise ValueError(f"expert gate demo {demo} has inconsistent start/goal rows")
        if int(record.get("prefix_steps", -1)) != 0:
            raise ValueError(f"expert gate demo {demo} was not a pure teleport")
        if record.get("success") is True:
            eligible.add(demo)
    if not eligible:
        raise ValueError("expert gate has no successful continuation")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return eligible, {
        "path": str(path),
        "sha256": digest,
        "tested_demo_ids": all_ids,
        "eligible_demo_ids": sorted(eligible),
        "tested_count": len(all_ids),
        "eligible_count": len(eligible),
        "eligibility_rule": (
            "root expert reaches this episode's first goal-specific success "
            "from the identical row-exact teleport initialization"
        ),
    }
