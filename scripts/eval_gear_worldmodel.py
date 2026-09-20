#!/usr/bin/env python3
"""Closed-loop evaluation of a LeWM world model on SRSA gear insertion.

Two processes, for the reason scripts/lewm_policy_server.py documents: Isaac Lab
lives in the ``srsa_official_clean`` conda env and LeWM in ``le-wm/.venv``, and
neither can be imported into the other. This is the Isaac-side client; start the
planning server first and point ``--socket`` at it.

Protocol: either use a fixed goal with normal environment resets, or sample a
strictly successful goal row from the evaluation trajectory sidecar and restore
the initial simulator state from ``goal_offset_steps`` earlier in that episode.
Success is the simulator's strict insertion predicate - seated within 3 mm,
centred within 3 mm, aligned within 5 deg - held for three consecutive control
steps.  Only then does ``GearAssemblyEvalEnv._get_dones`` report ``terminated``.

Episodes run one at a time in a single-environment world; the planning server
keeps one receding-horizon action buffer, so concurrent episodes would share it.

Usage:
    # in le-wm/.venv
    python scripts/lewm_policy_server.py --checkpoint <dir>/weights_epoch_20.pt \
        --img-size 224 --action-dim 4 --task-action-schema gear_effective4 \
        --action-bound-mode data_quantile --action-block 5 \
        --train-dataset ~/.stable_worldmodel/datasets/gear_01036_effective4_balanced_train_v2.h5 \
        --horizon 1 --receding-horizon 1
    # in srsa_official_clean
    python scripts/eval_gear_worldmodel.py --eval_dataset <h5> \
        --goal_offset_steps 5 --episodes 10 ...
"""

from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import sys
import time
from collections import deque
from pathlib import Path

from isaaclab.app import AppLauncher

from result_commit import commit_json_result

sys.path.insert(0, str(Path(__file__).resolve().parent))

TASK_ID = "Assembly-Sparse-v0"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    goal_source = p.add_mutually_exclusive_group(required=True)
    goal_source.add_argument("--goal_image", type=Path)
    goal_source.add_argument(
        "--eval_dataset",
        type=Path,
        help="Evaluation trajectory sidecar containing pixels and simulator state.",
    )
    p.add_argument(
        "--goal_offset_steps",
        type=int,
        default=None,
        help="Paired mode only: restore state this many raw steps before the goal.",
    )
    p.add_argument(
        "--paired_min_goal_step",
        type=int,
        default=0,
        help="Paired mode only: require goal step >= this value for shared samples.",
    )
    p.add_argument(
        "--paired_required_offsets",
        type=int,
        nargs="+",
        default=(5, 15, 25),
        help=(
            "Master-pool offsets. Every listed goal-minus-offset row must be "
            "non-successful; keep 5 15 25 to pair H1/H3/H5 exactly."
        ),
    )
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument(
        "--paired_source_episodes",
        type=str,
        default=None,
        help=(
            "Comma-separated source episode IDs in exact manifest order. "
            "When supplied, --episodes must equal the list length."
        ),
    )
    p.add_argument("--eval_budget", type=int, default=150, help="Raw control steps per episode.")
    p.add_argument(
        "--policy", choices=("random", "zero", "expert", "server"), default="server"
    )
    p.add_argument("--socket", default="/tmp/lewm_gear_policy.sock")
    p.add_argument(
        "--expert_gate_json",
        type=Path,
        default=None,
        help="Formal paired server evaluation requires an exhaustive same-init expert gate.",
    )
    p.add_argument(
        "--goal_ablation",
        choices=("correct", "self_init"),
        default="correct",
        help=(
            "self_init supplies the case's own non-success start image as goal; "
            "paired with correct it tests visual-goal dependence on identical starts."
        ),
    )
    p.add_argument(
        "--action_protocol",
        choices=("effective4", "legacy_raw6"),
        default="effective4",
        help=(
            "effective4 is [dx,dy,dz,yaw] with environment EMA disabled. "
            "legacy_raw6 is an explicit old-checkpoint compatibility mode; "
            "rx/ry are still frozen and index 5 remains yaw, never gripper."
        ),
    )
    # jepa.rollout sets n_steps = horizon - pixels.size(2), but its inclusive
    # range(n_steps + 1) loop appends horizon - pixels.size(2) + 1 future states.
    # The shipped EnvPool sends one frame (`pixels[:, None, ...]`), for which the
    # number of predicted and executed action blocks both equal horizon.  With
    # more than one input frame the unmodified rollout treats the first candidate
    # actions as action context for already-observed frames while WorldModelPolicy
    # still executes every candidate as a future action.  That path is temporally
    # misaligned unless the interface also supplies/skips the true history actions,
    # so one frame is the only supported planning convention here.
    p.add_argument("--history_len", type=int, default=1)
    p.add_argument("--assembly_id", default="01036")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--planner_seed",
        type=int,
        default=None,
        help=(
            "Base seed sent to the planning server on episode reset. Defaults "
            "to --seed. Set this separately to repeat the exact paired cases "
            "with independent CEM samples without changing the expert gate, "
            "environment seed, or case selection."
        ),
    )
    p.add_argument(
        "--paired_planner_seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated per-case planner seeds in manifest order. This "
            "keeps seeds invariant when a multi-source pool is evaluated in groups."
        ),
    )
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--focal_length_mm", type=float, default=40.0)
    p.add_argument("--camera_eye_offset", type=float, nargs=3, default=(0.08, -0.08, 0.055))
    p.add_argument("--camera_target_z_offset", type=float, default=0.030)
    p.add_argument("--render_interval", type=int, default=1)
    p.add_argument("--strict_max_height_m", type=float, default=0.003)
    p.add_argument("--strict_max_xy_error_m", type=float, default=0.003)
    p.add_argument("--strict_max_orientation_deg", type=float, default=5.0)
    p.add_argument(
        "--strict_persistence_steps",
        type=int,
        default=3,
        help="Consecutive strict-success control steps required; collection used 3.",
    )
    p.add_argument(
        "--clip_actions",
        type=float,
        default=None,
        help=(
            "Optional client-side clipping in effective 4-D space. Omit for exact "
            "model-scored/executed equality; any projection is reported."
        ),
    )
    p.add_argument("--action_consistency_tolerance", type=float, default=1.0e-6)
    p.add_argument(
        "--max_init_frame_mae",
        type=float,
        default=5.0,
        help="Paired evaluation fails closed if restored RGB MAE exceeds this /255 value.",
    )
    p.add_argument(
        "--max_init_pixels_over_8",
        type=float,
        default=0.05,
        help="Paired evaluation fails closed if this fraction of pixels differs by >8.",
    )
    p.add_argument(
        "--restoration_audit_only",
        action="store_true",
        help=(
            "Paired zero-policy diagnostic: restore and render every requested case "
            "without executing control steps. This never produces policy scores."
        ),
    )
    p.add_argument(
        "--action_diagnostics",
        choices=("summary", "full"),
        default="summary",
        help="Write aggregate diagnostics, or additionally retain every control step.",
    )
    p.add_argument(
        "--legacy_source_ema_factor",
        type=float,
        default=0.2,
        help="EMA alpha used only while replaying a legacy v1 sidecar prefix.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gif_dir", type=Path, default=None)
    AppLauncher.add_app_launcher_args(p)
    return p


args = _parser().parse_args()
if (
    args.policy in {"server", "zero"}
    and args.eval_dataset is not None
    and args.expert_gate_json is None
):
    raise ValueError("formal paired server/zero evaluation requires --expert_gate_json")
if args.policy == "expert" and args.eval_dataset is None:
    raise ValueError("--policy expert requires --eval_dataset")
if args.policy == "expert" and args.action_protocol != "effective4":
    raise ValueError("the formal Gear expert gate requires --action_protocol effective4")
if args.restoration_audit_only and (
    args.policy != "zero" or args.eval_dataset is None
):
    raise ValueError(
        "--restoration_audit_only requires paired --policy zero evaluation"
    )
if args.goal_ablation != "correct" and (
    args.eval_dataset is None or args.policy != "server"
):
    raise ValueError("Gear goal ablations require paired --policy server evaluation")
args.enable_cameras = True
args.headless = True
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import SRSA  # noqa: E402, F401
import SRSA.tasks  # noqa: E402, F401
from isaaclab_tasks.direct.automate.assembly_tasks_cfg import ASSET_DIR  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from isaac_gear_eval_env import (  # noqa: E402
    EFFECTIVE_ACTION_SEMANTICS,
    LEGACY_ACTION_SEMANTICS,
    complete_action_consistency_audit,
    effective4_to_env6,
    load_paired_gear_cases,
    load_gear_expert_gate,
    prepare_model_action,
    validate_restoration_reconstruction,
    validate_gear_planner_config,
    build_env_classes,
)


def _log(m: str) -> None:
    print(f"[gear-eval] {m}", flush=True)


def _send(sock, obj) -> None:
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv(sock):
    head = b""
    while len(head) < 4:
        c = sock.recv(4 - len(head))
        if not c:
            return None
        head += c
    n = struct.unpack(">I", head)[0]
    buf = b""
    while len(buf) < n:
        c = sock.recv(min(1 << 20, n - len(buf)))
        if not c:
            return None
        buf += c
    return pickle.loads(buf)


def _restore_paired_case(env, case: dict) -> tuple[dict, np.ndarray]:
    """Restore a v2 row directly, or reconstruct a legacy row by expert replay."""

    metadata = case["metadata"]
    env_id = torch.tensor([0], dtype=torch.long, device=env.device)
    # Collection aims the camera once from the episode's reset-time socket pose
    # and leaves it stationary.  Restore that episode-level camera state before
    # rendering the selected mid-trajectory row.
    env.set_camera_target_local(
        env_id,
        case["episode_initial_state"]["fixed_pos"],
    )
    restoration = metadata["restoration"]
    replayed_steps = 0
    if restoration == "full_state_v2":
        env.set_state(env_id, case["state"])
    elif restoration == "expert_prefix":
        env.set_state(env_id, case["episode_initial_state"])
        source_semantics = metadata["source_action_semantics"]
        source_alpha = float(metadata["source_ema_factor"])
        replay_alpha = (
            source_alpha
            if source_semantics == LEGACY_ACTION_SEMANTICS
            else 1.0
        )
        online_alpha = float(env.cfg.ctrl.ema_factor)
        env.cfg.ctrl.ema_factor = replay_alpha
        try:
            for source_action in case["expert_prefix"]:
                if source_semantics == LEGACY_ACTION_SEMANTICS:
                    env_command = np.asarray(source_action, dtype=np.float32)
                elif source_semantics == EFFECTIVE_ACTION_SEMANTICS:
                    env_command = effective4_to_env6(source_action)
                else:
                    raise ValueError(
                        f"unsupported prefix action semantics {source_semantics!r}"
                    )
                _, _, terminated, truncated, _ = env.step(
                    torch.as_tensor(
                        env_command, dtype=torch.float32, device=env.device
                    ).reshape(1, 6)
                )
                replayed_steps += 1
                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    raise RuntimeError(
                        "expert prefix terminated before the selected non-success init row"
                    )
        finally:
            env.cfg.ctrl.ema_factor = online_alpha
    else:
        raise ValueError(f"unknown restoration mode {restoration!r}")

    # Roll/pitch are overwritten by AutoMate and are not part of the Markov
    # action. Remove any legacy EMA residue before online planning.
    env.actions[env_id, 3:5] = 0.0
    env.prev_actions[env_id, 3:5] = 0.0

    # The benchmark budget starts at the paired init regardless of how much
    # expert prefix was needed to reconstruct it.
    env.episode_length_buf[env_id] = 0
    env.ep_succeeded[env_id] = False
    env._strict_success_streak[env_id] = 0
    observed = env.aim_camera_and_capture()[0]
    difference = np.abs(
        observed.astype(np.int16) - case["init_pixels"].astype(np.int16)
    )
    return {
        "mode": restoration,
        "expert_prefix_steps": replayed_steps,
        "expert_prefix_replay_ema_factor": (
            replay_alpha if restoration == "expert_prefix" else None
        ),
        "init_frame_mean_abs_error": float(difference.mean()),
        "init_frame_max_abs_error": int(difference.max()),
        "init_frame_pixels_over_8": float((difference.max(axis=-1) > 8).mean()),
    }, observed


def _configure_env():
    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1, use_fabric=True)
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.sim.render_interval = args.render_interval
    cfg.seed = args.seed
    # Main protocol stores/optimizes the already-filtered effective action, so
    # applying another EMA here would reintroduce hidden state.  Legacy raw-6D
    # checkpoints retain the original controller smoothing explicitly.
    cfg.ctrl.ema_factor = (
        1.0 if args.action_protocol == "effective4" else args.legacy_source_ema_factor
    )
    # Collection episodes are 74 control steps because episode_length_s is 5 s.
    # A 150-step evaluation budget would silently trip that time-out and let
    # DirectRLEnv auto-reset the world half way through, so the horizon has to
    # be long enough to hold the whole budget.
    needed_s = (args.eval_budget + 5) * cfg.sim.dt * cfg.decimation
    cfg.episode_length_s = float(max(cfg.episode_length_s, needed_s))
    task = cfg.tasks[cfg.task_name]
    d = f"{ASSET_DIR}/{args.assembly_id}/"
    task.assembly_id = args.assembly_id
    task.assembly_dir = d
    task.disassembly_path_json = f"{d}/disassemble_traj.json"
    task.fixed_asset.spawn.usd_path = f"{d}{task.fixed_asset_cfg.usd_path}"
    task.held_asset.spawn.usd_path = f"{d}{task.held_asset_cfg.usd_path}"
    task.if_sbc = False
    task.if_logging_eval = False
    return cfg


def main() -> None:
    import os
    import tempfile

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    planner_seed = args.seed if args.planner_seed is None else int(args.planner_seed)
    paired_planner_seeds = None
    if args.paired_planner_seeds is not None:
        paired_planner_seeds = tuple(
            int(value.strip())
            for value in args.paired_planner_seeds.split(",")
            if value.strip()
        )
        if len(paired_planner_seeds) != args.episodes:
            raise ValueError("--paired_planner_seeds length must equal --episodes")
    if args.strict_persistence_steps < 1:
        raise ValueError("strict_persistence_steps must be positive")
    if args.history_len != 1:
        raise ValueError("Gear planning currently supports exactly --history_len 1")
    if args.action_consistency_tolerance < 0:
        raise ValueError("action_consistency_tolerance must be non-negative")
    if not 0.0 < args.legacy_source_ema_factor <= 1.0:
        raise ValueError("legacy_source_ema_factor must be in (0, 1]")
    if args.eval_dataset is None and args.goal_offset_steps is not None:
        raise ValueError("--goal_offset_steps requires --eval_dataset")
    if args.eval_dataset is not None and args.goal_offset_steps is None:
        raise ValueError("--eval_dataset requires --goal_offset_steps")
    expected_macro_horizon = None
    if args.eval_dataset is not None:
        if args.goal_offset_steps % 5:
            raise ValueError("paired goal offset must be a multiple of action_block=5")
        expected_macro_horizon = args.goal_offset_steps // 5

    paired_cases = None
    paired_candidate_count = None
    paired_pool = None
    fixed_goal = None
    expert_gate_audit = None
    expert_eligible_episodes = None
    selected_source_episodes = None
    if args.paired_source_episodes is not None:
        selected_source_episodes = tuple(
            int(value.strip())
            for value in args.paired_source_episodes.split(",")
            if value.strip()
        )
        if len(selected_source_episodes) != args.episodes:
            raise ValueError(
                "--paired_source_episodes length must equal --episodes"
            )
    if args.eval_dataset is not None:
        if args.expert_gate_json is not None:
            expert_eligible_episodes, expert_gate_audit = load_gear_expert_gate(
                args.expert_gate_json,
                eval_dataset=args.eval_dataset,
                goal_offset_steps=args.goal_offset_steps,
                eval_budget=args.eval_budget,
                seed=args.seed,
                master_offsets=args.paired_required_offsets,
                action_protocol=args.action_protocol,
            )
        paired_cases, paired_pool = load_paired_gear_cases(
            args.eval_dataset,
            count=args.episodes,
            seed=args.seed,
            goal_offset_steps=args.goal_offset_steps,
            master_offsets=args.paired_required_offsets,
            minimum_goal_step=args.paired_min_goal_step,
            persistence_steps=args.strict_persistence_steps,
            max_height_m=args.strict_max_height_m,
            max_xy_error_m=args.strict_max_xy_error_m,
            max_orientation_deg=args.strict_max_orientation_deg,
            legacy_source_ema_factor=args.legacy_source_ema_factor,
            eligible_source_episodes=expert_eligible_episodes,
            selected_source_episodes=selected_source_episodes,
        )
        if (
            expert_gate_audit is not None
            and expert_gate_audit["master_pool_sha256"]
            != paired_pool["master_pool_sha256"]
        ):
            raise ValueError("Gear expert gate was built from a different master pool")
        paired_candidate_count = paired_pool["candidate_count"]
        _log(
            f"paired goals: selected {len(paired_cases)} of "
            f"{paired_candidate_count} eligible episodes; "
            f"offset={args.goal_offset_steps}, min_goal_step="
            f"{max(args.paired_min_goal_step, args.goal_offset_steps)}, "
            f"master_offsets={args.paired_required_offsets}, "
            f"pool={paired_pool['master_pool_sha256'][:12]}, "
            f"selection={paired_pool['selection_sha256'][:12]}, "
            f"restore={paired_pool['restoration']}"
        )
    else:
        fixed_goal = np.asarray(
            Image.open(args.goal_image).convert("RGB"), dtype=np.uint8
        )
        if fixed_goal.shape != (args.resolution, args.resolution, 3):
            raise ValueError(
                f"goal must be {args.resolution}^2 RGB, got {fixed_goal.shape}"
            )

    cfg = _configure_env()
    _log(f"episode_length_s={cfg.episode_length_s:.2f} -> max {int(round(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)))} control steps "
         f"(budget {args.eval_budget})")

    env_class = build_env_classes(args)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="gear-eval-") as cache:
        os.chdir(cache)
        try:
            env = env_class(cfg, render_mode=None)
        finally:
            os.chdir(cwd)
    _log(f"env ready, max_episode_length={int(env.max_episode_length)}")
    _log(
        f"action protocol={args.action_protocol}: model dimensions are "
        f"{'[dx,dy,dz,yaw]' if args.action_protocol == 'effective4' else '[dx,dy,dz,rx,ry,yaw]'}; "
        f"env EMA alpha={cfg.ctrl.ema_factor}; rx/ry frozen; no gripper dimension"
    )

    sock = None
    if args.policy == "server":
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(args.socket)
        _log(f"connected to planning server at {args.socket}")
    else:
        _log(f"{args.policy} baseline: no planner")

    episodes = []
    server_planner_config = None
    started = time.monotonic()
    for ep in range(args.episodes):
        ep_started = time.monotonic()
        env.reset()
        source_metadata = None
        restoration_audit = None
        if paired_cases is not None:
            case = paired_cases[ep]
            restoration_audit, frame = _restore_paired_case(env, case)
            validate_restoration_reconstruction(
                restoration_audit,
                max_mean_abs_error=args.max_init_frame_mae,
                max_pixels_over_8=args.max_init_pixels_over_8,
            )
            goal = (
                case["goal"]
                if args.goal_ablation == "correct"
                else case["init_pixels"]
            )
            source_metadata = case["metadata"]
        else:
            goal = fixed_goal
            frame = env.aim_camera_and_capture()[0]
        if args.restoration_audit_only:
            record = {
                "episode": ep,
                "success": False,
                "success_step": None,
                "steps": 0,
                "seconds": time.monotonic() - ep_started,
                "final_height_mm": None,
                "final_xy_mm": None,
                "final_orientation_deg": None,
                "final_instant_success": None,
                "final_success_streak": None,
                "first_instant_success_step": None,
                "max_success_streak": 0,
                "expert_sequence_exhausted": False,
                "best": None,
                "action_consistency": {
                    "protocol": args.action_protocol,
                    "steps": 0,
                    "non_exact_steps": 0,
                    "max_model_to_command_abs": 0.0,
                    "max_command_to_applied_abs": 0.0,
                    "max_applied_inactive_rx_ry_abs": 0.0,
                },
                "paired_source": source_metadata,
                "restoration": restoration_audit,
                "restoration_audit_only": True,
            }
            episodes.append(record)
            _log(
                f"restoration {ep + 1}/{args.episodes}: "
                f"source_episode={source_metadata['source_episode']} "
                f"mae={restoration_audit['init_frame_mean_abs_error']:.4f} "
                f"pixels_over_8={restoration_audit['init_frame_pixels_over_8']:.4%}"
            )
            continue
        history = deque([frame] * args.history_len, maxlen=args.history_len)
        if sock is not None:
            reset_request = {"cmd": "reset"}
            if paired_cases is not None:
                reset_request["seed"] = (
                    paired_planner_seeds[ep]
                    if paired_planner_seeds is not None
                    else planner_seed + ep
                )
            _send(sock, reset_request)
            reply = _recv(sock)
            if not (reply and reply.get("ok")):
                raise RuntimeError(f"server reset failed: {reply}")
            reset_planner_config = validate_gear_planner_config(
                reply,
                action_protocol=args.action_protocol,
                expected_horizon=expected_macro_horizon,
                expected_receding_horizon=expected_macro_horizon,
            )
            if server_planner_config is None:
                server_planner_config = reset_planner_config
            elif reset_planner_config != server_planner_config:
                raise RuntimeError(
                    "planning server provenance changed between episode resets"
                )

        success = False
        success_step = None
        gif = [frame.copy()] if args.gif_dir else None
        best_score = float("inf")
        best_metrics = None
        final_metrics = None
        first_instant_success_step = None
        max_success_streak = 0
        expert_sequence_exhausted = False
        action_audit = {
            "protocol": args.action_protocol,
            "steps": 0,
            "non_exact_steps": 0,
            "max_model_to_command_abs": 0.0,
            "max_command_to_applied_abs": 0.0,
            "max_applied_inactive_rx_ry_abs": 0.0,
        }
        action_audit_steps = [] if args.action_diagnostics == "full" else None
        for t in range(args.eval_budget):
            if args.policy == "random":
                action_dim = 4 if args.action_protocol == "effective4" else 6
                action = rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            elif args.policy == "zero":
                action_dim = 4 if args.action_protocol == "effective4" else 6
                action = np.zeros(action_dim, dtype=np.float32)
            elif args.policy == "expert":
                key = (
                    "expert_future_effective4"
                    if args.action_protocol == "effective4"
                    else "expert_future_raw"
                )
                sequence = case[key]
                if t >= len(sequence):
                    # Exhausting the recorded continuation without reproducing
                    # success is exactly what the exhaustive expert gate is
                    # meant to classify as restoration-ineligible.  Preserve
                    # the failed record instead of aborting the whole pool.
                    expert_sequence_exhausted = True
                    break
                action = np.asarray(sequence[t], dtype=np.float32)
            else:
                stacked = np.stack(list(history)).astype(np.uint8)  # oldest first
                _send(sock, {
                    "cmd": "act",
                    "pixels": stacked.tobytes(), "pixels_shape": tuple(stacked.shape),
                    "goal": goal.tobytes(), "goal_shape": tuple(goal.shape),
                })
                reply = _recv(sock)
                if not (reply and reply.get("ok")):
                    raise RuntimeError(f"server act failed at ep{ep} t{t}: {reply}")
                action = np.asarray(reply["action"], dtype=np.float32)

            _, env_command, pending_action_audit = prepare_model_action(
                action,
                protocol=args.action_protocol,
                clip=args.clip_actions,
            )
            command = torch.as_tensor(
                env_command, dtype=torch.float32, device=env.device
            ).unsqueeze(0)
            _, _, terminated, truncated, _ = env.step(command)
            step_action_audit = complete_action_consistency_audit(
                pending_action_audit,
                env.eval_last_applied_action[0].detach().to("cpu").numpy(),
                tolerance=args.action_consistency_tolerance,
            )
            action_audit["steps"] += 1
            action_audit["non_exact_steps"] += int(
                not step_action_audit["exact_model_scored_execution"]
            )
            action_audit["max_model_to_command_abs"] = max(
                action_audit["max_model_to_command_abs"],
                step_action_audit["model_to_command_max_abs"],
            )
            action_audit["max_command_to_applied_abs"] = max(
                action_audit["max_command_to_applied_abs"],
                step_action_audit["command_to_applied_max_abs"],
            )
            action_audit["max_applied_inactive_rx_ry_abs"] = max(
                action_audit["max_applied_inactive_rx_ry_abs"],
                step_action_audit["applied_inactive_rx_ry_max_abs"],
            )
            if action_audit_steps is not None:
                action_audit_steps.append(step_action_audit)
            if (
                args.action_protocol == "effective4"
                and not step_action_audit["exact_model_scored_execution"]
            ):
                raise RuntimeError(
                    "effective4 contract violated: model-scored and applied actions differ: "
                    f"{step_action_audit}"
                )

            # These tensors were captured inside _get_dones(), before Isaac's
            # automatic reset.  Reading held_pos/fixed_pos after env.step() would
            # report the next episode's reset pose on a successful transition.
            final_metrics = {
                "height_mm": float(env.eval_last_height_m[0].item() * 1000.0),
                "xy_mm": float(env.eval_last_xy_error_m[0].item() * 1000.0),
                "orientation_deg": float(env.eval_last_orientation_deg[0].item()),
                "instant_success": bool(env.eval_last_instant_success[0].item()),
                "success_streak": int(env.eval_last_success_streak[0].item()),
            }
            if final_metrics["instant_success"] and first_instant_success_step is None:
                first_instant_success_step = t + 1
            max_success_streak = max(
                max_success_streak, final_metrics["success_streak"]
            )
            score = max(
                max(0.0, final_metrics["height_mm"]) / (args.strict_max_height_m * 1000.0),
                final_metrics["xy_mm"] / (args.strict_max_xy_error_m * 1000.0),
                final_metrics["orientation_deg"] / args.strict_max_orientation_deg,
            )
            if final_metrics["height_mm"] <= 0.0:
                score += 1.0 + abs(final_metrics["height_mm"]) / 3.0
            if score < best_score:
                best_score = score
                best_metrics = dict(final_metrics, step=t + 1, normalized_violation=score)

            if bool(terminated[0].item()):
                success, success_step = True, t + 1
                break
            if bool(truncated[0].item()):
                _log(f"  ep{ep}: env truncated at step {t + 1} before the budget ran out")
                break

            frame = env.aim_camera_and_capture()[0]
            history.append(frame)
            if gif is not None:
                gif.append(frame.copy())

        if final_metrics is None:
            raise RuntimeError("episode executed no control steps")
        record = {
            "episode": ep, "success": success, "success_step": success_step,
            "steps": action_audit["steps"], "seconds": time.monotonic() - ep_started,
            "final_height_mm": final_metrics["height_mm"],
            "final_xy_mm": final_metrics["xy_mm"],
            "final_orientation_deg": final_metrics["orientation_deg"],
            "final_instant_success": final_metrics["instant_success"],
            "final_success_streak": final_metrics["success_streak"],
            "first_instant_success_step": first_instant_success_step,
            "max_success_streak": max_success_streak,
            "expert_sequence_exhausted": expert_sequence_exhausted,
            "best": best_metrics,
            "action_consistency": action_audit,
        }
        if action_audit_steps is not None:
            record["action_consistency"]["per_step"] = action_audit_steps
        if source_metadata is not None:
            record["paired_source"] = source_metadata
            record["restoration"] = restoration_audit
        episodes.append(record)
        _log(f"episode {ep + 1}/{args.episodes}: success={success} "
             f"step={success_step} z={record['final_height_mm']:.2f}mm "
             f"xy={record['final_xy_mm']:.2f}mm "
             f"rot={record['final_orientation_deg']:.2f}deg "
             f"streak={record['final_success_streak']} ({record['seconds']:.0f}s)")
        if gif is not None:
            args.gif_dir.mkdir(parents=True, exist_ok=True)
            imgs = [Image.fromarray(f) for f in gif]
            imgs[0].save(args.gif_dir / f"ep{ep:02d}_success{int(success)}.gif",
                         save_all=True, append_images=imgs[1:], duration=100, loop=0)

    n = len(episodes)
    hits = sum(e["success"] for e in episodes)
    result = {
        "policy": args.policy, "socket": args.socket if sock else None,
        "action_protocol": args.action_protocol,
        "restoration_audit_only": args.restoration_audit_only,
        "goal_image": str(args.goal_image) if args.goal_image is not None else None,
        "goal_ablation": args.goal_ablation,
        "eval_dataset": (
            str(args.eval_dataset.expanduser().resolve())
            if args.eval_dataset is not None
            else None
        ),
        "goal_offset_steps": args.goal_offset_steps,
        "expected_macro_horizon": expected_macro_horizon,
        "formal_full_commit": args.eval_dataset is not None,
        "paired_min_goal_step": (
            max(
                args.paired_min_goal_step,
                args.goal_offset_steps,
                *(args.paired_required_offsets or ()),
            )
            if args.eval_dataset is not None
            else None
        ),
        "paired_candidate_count": paired_candidate_count,
        "paired_master_pool": paired_pool if args.eval_dataset is not None else None,
        "expert_gate": expert_gate_audit,
        "paired_required_non_success_offsets": list(args.paired_required_offsets),
        "paired_goal_selection": "episode_first_strict_frame_and_persists",
        "paired_source_episodes": (
            list(selected_source_episodes)
            if selected_source_episodes is not None
            else None
        ),
        "episodes": n, "seed": args.seed, "planner_seed": planner_seed,
        "paired_planner_seeds": (
            list(paired_planner_seeds)
            if paired_planner_seeds is not None
            else None
        ),
        "eval_budget": args.eval_budget, "history_len": args.history_len,
        "max_control_steps": int(env.max_episode_length),
        "strict_criterion": {
            "max_height_m": args.strict_max_height_m,
            "max_xy_error_m": args.strict_max_xy_error_m,
            "max_orientation_deg": args.strict_max_orientation_deg,
            "required_consecutive_control_steps": args.strict_persistence_steps,
        },
        "restoration_gate": (
            {
                "max_init_frame_mean_abs_error": args.max_init_frame_mae,
                "max_init_frame_pixels_over_8": args.max_init_pixels_over_8,
                "all_cases_passed": True,
            }
            if args.eval_dataset is not None
            else None
        ),
        "action_contract": {
            "protocol": args.action_protocol,
            "effective_order": ["dx", "dy", "dz", "yaw"],
            "environment_order": ["dx", "dy", "dz", "rx_frozen", "ry_frozen", "yaw"],
            "has_gripper_dimension": False,
            "environment_ema_factor": float(cfg.ctrl.ema_factor),
            "client_clip": args.clip_actions,
            "consistency_tolerance": args.action_consistency_tolerance,
            "all_steps_exact": all(
                episode["action_consistency"]["non_exact_steps"] == 0
                for episode in episodes
            ),
        },
        "server_planner_config": server_planner_config,
        "successes": hits, "success_rate_percent": 100.0 * hits / n,
        "wall_time_s": time.monotonic() - started,
        "per_episode": episodes,
    }
    completion = commit_json_result(
        args.output,
        result,
        schema_version="gear-h5-eval-v1",
        record_count=n,
    )
    _log(f"SUCCESS {hits}/{n} = {100.0 * hits / n:.1f}%  -> {args.output}")
    _log(f"completion hash={completion['result_sha256']}")

    if sock is not None:
        sock.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
