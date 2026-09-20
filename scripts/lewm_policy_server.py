#!/usr/bin/env python3
"""LeWM planning server — serves actions over a Unix socket.

Why a server: the Isaac Lab stack that renders the Franka cube-stacking environment lives in a
separate conda env (``srsa_official_clean``: python 3.12 + Isaac Sim 5.1), while LeWM lives in
``le-wm/.venv`` (python 3.10). The two cannot be imported into one process, and installing either
into the other's env would disturb a setup another workstream depends on. So the environment runs
as a client and asks this server for actions.

The server owns the model, the CEM solver and ``WorldModelPolicy`` (which manages the receding-
horizon action buffer), so the client stays a thin loop.

``WorldModelPolicy.set_env`` only ever reads ``action_space``, ``num_envs`` and
``single_action_space``, so a stub standing in for the env is enough — see ``_EnvStub``.

Protocol (pickle over SOCK_STREAM, 4-byte big-endian length prefix). Only built-in Python types
cross the socket — never numpy arrays: the two envs run different numpy majors (2.x here, 1.x in
``srsa_official_clean``) and a pickled array from one fails to unpickle in the other with
``ModuleNotFoundError: No module named 'numpy._core'``. Arrays travel as raw bytes plus shape.

    client -> server   {"cmd": "reset", "seed": int | None}
                       # optional seed resets the CEM generator for paired ablations
                       {"cmd": "act",
                        "pixels": bytes, "pixels_shape": (T, H, W, 3),   # uint8, oldest first
                        "goal":   bytes, "goal_shape":   (H, W, 3)}      # uint8
    server -> client   {"ok": True, "action": [float, ...]}
                       {"ok": False, "error": str}

Images arrive as raw uint8 HWC exactly as the camera produces them; this process applies the same
transform used in training (ToImage -> float32 scaled -> ImageNet normalize -> Resize), so the
client never has to know the preprocessing.

Usage:
    python scripts/lewm_policy_server.py \
        --checkpoint lewm_fstack1k_ep40/weights_epoch_39.pt \
        --task-action-schema franka_stack --action-bound-mode data_minmax \
        --train-dataset ~/.stable-wm/datasets/franka_stack_orig_1k.h5 \
        --img-size 196 --action-dim 7
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import sys
import traceback
from pathlib import Path

# Script entry points put ``scripts/`` rather than the repository root on
# sys.path. Add the root explicitly so the shared planner is importable from
# any launch directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from gymnasium import spaces
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms

from slewm.planning.projected_cem import (
    ACTION_BOUND_MODES,
    TASK_ACTION_DIMS,
    TASK_ACTION_SCHEMAS,
    ProjectedCEMSolver,
    ProjectedActionScaler,
    TrainingZScoreScaler,
    build_task_action_projection,
    resolve_raw_action_bounds,
    validate_action_scaler_contract,
    validate_model_action_contract,
)
from slewm.planning.behavior_plan_bank import (
    PLAN_BANK_CONTINUATIONS,
    PLAN_BANK_SELECTION_MODES,
    ProjectedPlanBankSolver,
    load_franka_terminal_plan_bank,
)


def img_transform(img_size: int):
    """Identical to eval.py::img_transform — the transform the model was trained under."""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


@torch.inference_mode()
def encode_plan_bank_endpoints(model, pixels: np.ndarray, transform, *, batch_size=64):
    """Encode immutable train endpoint images without action/dynamics rollout."""

    parameter = next(model.parameters())
    encoded_batches = []
    for start in range(0, len(pixels), batch_size):
        raw = pixels[start : start + batch_size]
        transformed = torch.stack(
            [
                transform(
                    tv_tensors.Image(torch.from_numpy(image).permute(2, 0, 1))
                )
                for image in raw
            ]
        ).to(device=parameter.device, dtype=parameter.dtype)
        encoded = model.encode({"pixels": transformed[:, None]})["emb"][:, -1]
        encoded_batches.append(encoded.float().cpu())
    return torch.cat(encoded_batches, dim=0)


class _EnvStub:
    """Stands in for a gym vector env. `WorldModelPolicy.set_env` reads only these three."""

    def __init__(self, action_dim: int, low, high):
        self.num_envs = 1
        raw_low = np.broadcast_to(np.asarray(low, dtype=np.float32), (action_dim,))
        raw_high = np.broadcast_to(np.asarray(high, dtype=np.float32), (action_dim,))
        self.single_action_space = spaces.Box(low=raw_low, high=raw_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=raw_low[None, :], high=raw_high[None, :], dtype=np.float32
        )


def parse_action_bound(value: str):
    """Parse either one scalar or a comma-separated per-dimension bound."""

    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid action bound {value!r}") from error
    if not values or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError(f"invalid action bound {value!r}")
    return values[0] if len(values) == 1 else values


def send(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv(sock):
    head = b""
    while len(head) < 4:
        chunk = sock.recv(4 - len(head))
        if not chunk:
            return None
        head += chunk
    n = struct.unpack(">I", head)[0]
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def fit_action_scaler(dataset_h5: str):
    """Refit the exact training-time z-score transform on training data only.

    `WorldModelPolicy.get_action` ends with
    ``action = self.process['action'].inverse_transform(action)``. Training normalised the action
    column, so the planner works in standardised space; without this scaler the returned actions
    would be handed to the environment still standardised. Fitting on the same file the model was
    trained on reproduces the training transform. Unlike sklearn
    ``StandardScaler`` (ddof=0), training uses torch's default
    sample standard deviation (ddof=1) and maps std<1e-6 to one.
    ``dataset_h5`` must be the training file, never a validation/evaluation
    file, so an external train/val setup cannot leak validation statistics.
    """
    a = load_action_data(dataset_h5)
    sc = TrainingZScoreScaler.fit(a)
    print(f"[server] action scaler 拟合自训练集 {dataset_h5}: n={len(a)} ddof=1 "
          f"mean={np.round(sc.mean_, 4)} scale={np.round(sc.scale_, 4)}", flush=True)
    return sc


def load_action_data(dataset_h5: str) -> np.ndarray:
    """Load finite actions that have a stored successor observation.

    Exact aligned files have one action-shaped placeholder on the final
    observation row of each episode. Training excludes those rows, so the
    policy server must exclude them from both z-scoring and support bounds.
    Legacy files without episode metadata retain their historical behavior.
    """
    import h5py

    with h5py.File(dataset_h5, "r") as handle:
        # Preserve stored precision: train.py passes np.array(col_data)
        # directly to torch, so promoting float32 here would change stats.
        if "ep_offset" in handle and "ep_len" in handle:
            offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
            lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
            if offsets.shape != lengths.shape:
                raise ValueError("ep_offset and ep_len have different shapes")
            chunks = [
                np.asarray(handle["action"][int(start) : int(start + length - 1)])
                for start, length in zip(offsets, lengths)
                if int(length) > 1
            ]
            if not chunks:
                raise ValueError("training dataset has no action with a successor")
            action = np.concatenate(chunks, axis=0)
        else:
            action = np.asarray(handle["action"][:])
    if action.ndim != 2:
        raise ValueError(f"training action column must be rank 2, got {action.shape}")
    action = action[np.isfinite(action).all(axis=1)]
    if len(action) == 0:
        raise ValueError("training action column has no finite rows")
    return action


def resolve_continuous_plan_tv_max(
    dataset_h5: str,
    scaler: TrainingZScoreScaler,
    *,
    total_steps: int,
    quantile: float,
    continuous_indices: tuple[int, ...],
) -> float:
    """Fit a task's sequence-level support bound without crossing episodes.

    The statistic is the L1 total variation of standardized continuous actions
    over windows matching one complete CEM plan. The final placeholder action
    in each aligned episode is excluded exactly as it is during training.
    """
    import h5py

    if total_steps < 2:
        raise ValueError("continuous plan-TV support needs at least two raw steps")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("temporal TV quantile must lie in (0,1]")
    if not continuous_indices:
        raise ValueError("continuous plan-TV support needs action indices")
    if len(set(continuous_indices)) != len(continuous_indices):
        raise ValueError("continuous plan-TV action indices are duplicated")
    if any(
        index < 0 or index >= scaler.n_features_in_
        for index in continuous_indices
    ):
        raise ValueError("continuous plan-TV action index is out of range")
    values: list[np.ndarray] = []
    with h5py.File(dataset_h5, "r") as handle:
        if "ep_offset" not in handle or "ep_len" not in handle:
            raise ValueError(
                "continuous plan-TV support requires exact episode metadata"
            )
        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        for start, stored_length in zip(offsets, lengths):
            action_length = int(stored_length) - 1
            if action_length < total_steps:
                continue
            raw = np.asarray(
                handle["action"][int(start) : int(start) + action_length]
            )
            standardized = scaler.transform(raw)[:, continuous_indices]
            step_tv = np.abs(np.diff(standardized, axis=0)).sum(axis=1)
            # A total_steps window contains total_steps-1 adjacent deltas.
            cumulative = np.concatenate(([0.0], np.cumsum(step_tv, dtype=np.float64)))
            width = total_steps - 1
            values.append(cumulative[width:] - cumulative[:-width])
    if not values:
        raise ValueError(
            f"training dataset has no episode with {total_steps} valid actions"
        )
    maximum = float(np.quantile(np.concatenate(values), quantile))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("resolved continuous plan-TV maximum is invalid")
    return maximum


def resolve_franka_terminal_support(
    dataset_h5: str,
    *,
    total_steps: int,
    clip_stride: int = 5,
    minimum_terminal_open_steps: int = 0,
) -> dict:
    """Fit exact terminal phases and strict-release support on train only."""

    import h5py

    if total_steps < 1:
        raise ValueError("Franka terminal support needs a positive plan length")
    if clip_stride < 1:
        raise ValueError("Franka terminal support clip_stride must be positive")
    if minimum_terminal_open_steps < 0:
        raise ValueError("minimum_terminal_open_steps must be non-negative")
    switch_counts: dict[int, int] = {}
    post_success_lengths: list[int] = []
    post_success_actions = 0
    post_success_open_actions = 0
    sampled_plan_count = 0
    all_open_plan_count = 0
    exact_open_hold_plan_count = 0
    with h5py.File(dataset_h5, "r") as handle:
        required = {
            "action",
            "ep_offset",
            "ep_len",
            "goal_idx",
            "ep_has_success",
        }
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(
                f"Franka terminal support metadata missing: {sorted(missing)}"
            )
        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        goals = np.asarray(handle["goal_idx"][:], dtype=np.int64)
        successes = np.asarray(handle["ep_has_success"][:], dtype=bool)
        if not (offsets.shape == lengths.shape == goals.shape == successes.shape):
            raise ValueError("Franka terminal metadata arrays have different shapes")
        for offset, stored_length, goal, has_success in zip(
            offsets, lengths, goals, successes
        ):
            if not has_success:
                continue
            offset = int(offset)
            stored_length = int(stored_length)
            goal = int(goal)
            valid_end = offset + stored_length - 1
            for local_start in range(
                0, max(0, stored_length - total_steps), clip_stride
            ):
                sampled_plan = np.asarray(
                    handle["action"][
                        offset + local_start : offset + local_start + total_steps
                    ]
                )
                sampled_gripper = sampled_plan[:, 6]
                sampled_plan_count += 1
                all_open_plan_count += int(np.all(sampled_gripper == 1.0))
                exact_open_hold_plan_count += int(
                    np.all(sampled_plan[:, :6] == 0.0)
                    and np.all(sampled_gripper == 1.0)
                )
            start = goal - total_steps
            if start < offset or goal > valid_end:
                raise ValueError(
                    "successful Franka episode cannot provide an exact terminal plan"
                )
            raw = np.asarray(handle["action"][start:goal])
            if raw.shape != (total_steps, 7):
                raise ValueError("Franka terminal plan has an unexpected action shape")
            if not np.isfinite(raw).all():
                raise ValueError("Franka terminal plan contains non-finite actions")
            gripper = raw[:, 6]
            if not np.all(np.isin(gripper, (-1.0, 1.0))):
                raise ValueError("Franka terminal gripper is not exact binary ±1")
            open_steps = np.flatnonzero(gripper == 1.0)
            if len(open_steps) == 0:
                raise ValueError("Franka terminal plan never opens the gripper")
            split = int(open_steps[0])
            if not (
                np.all(gripper[:split] == -1.0)
                and np.all(gripper[split:] == 1.0)
            ):
                raise ValueError(
                    "Franka terminal plan is not one close-to-open phase transition"
                )
            switch_counts[split] = switch_counts.get(split, 0) + 1

            post = np.asarray(handle["action"][goal:valid_end])
            post_success_lengths.append(len(post))
            post_success_actions += len(post)
            if len(post):
                post_success_open_actions += int(np.sum(post[:, 6] == 1.0))

    if not switch_counts:
        raise ValueError("training dataset has no successful Franka terminal plan")
    if post_success_actions != post_success_open_actions:
        raise ValueError(
            "training post-success continuation contains a non-open gripper action"
        )
    if all_open_plan_count == 0:
        raise ValueError(
            "training data has no all-open plan supporting the episode latch"
        )
    observed_switch_steps = tuple(sorted(switch_counts))
    eligible_switch_counts = {
        split: count
        for split, count in switch_counts.items()
        if total_steps - split >= minimum_terminal_open_steps
    }
    if not eligible_switch_counts:
        raise ValueError(
            "no train-observed Franka terminal phase leaves enough open commands "
            f"for the strict actuator criterion ({minimum_terminal_open_steps})"
        )
    switch_steps = tuple(sorted(eligible_switch_counts))
    excluded_switch_counts = {
        split: switch_counts[split]
        for split in observed_switch_steps
        if split not in eligible_switch_counts
    }
    return {
        "switch_steps": switch_steps,
        "switch_counts": tuple(
            (step, eligible_switch_counts[step]) for step in switch_steps
        ),
        "sample_count": sum(eligible_switch_counts.values()),
        "terminal_open_steps": total_steps - max(switch_steps),
        "minimum_terminal_open_steps": int(minimum_terminal_open_steps),
        "observed_switch_steps": observed_switch_steps,
        "observed_switch_counts": tuple(
            (step, switch_counts[step]) for step in observed_switch_steps
        ),
        "observed_sample_count": sum(switch_counts.values()),
        "excluded_switch_counts": tuple(sorted(excluded_switch_counts.items())),
        "sampled_plan_count": sampled_plan_count,
        "all_open_plan_count": all_open_plan_count,
        "exact_open_hold_plan_count": exact_open_hold_plan_count,
        "clip_stride": int(clip_stride),
        "post_success_episode_count": len(post_success_lengths),
        "post_success_min_steps": min(post_success_lengths),
        "post_success_max_steps": max(post_success_lengths),
        "post_success_action_count": post_success_actions,
        "post_success_open_action_count": post_success_open_actions,
    }


def build_policy(args):
    training_dataset = (
        str(Path(args.dataset).expanduser().resolve()) if args.dataset else None
    )
    stock_cem = args.planner_mode == "stock_cem"
    if args.task_action_schema not in TASK_ACTION_SCHEMAS:
        raise ValueError(f"unknown task action schema {args.task_action_schema!r}")
    if args.task_action_schema == "generic":
        if args.action_bound_mode != "none":
            raise ValueError("generic schema requires --action-bound-mode none")
    else:
        expected_dim = TASK_ACTION_DIMS[args.task_action_schema]
        if args.action_dim != expected_dim:
            raise ValueError(
                f"task action schema {args.task_action_schema!r} requires "
                f"action_dim={expected_dim}, got {args.action_dim}"
            )
        if stock_cem and args.action_bound_mode != "none":
            raise ValueError("stock_cem requires --action-bound-mode none")
        if not stock_cem and args.action_bound_mode == "none":
            raise ValueError(
                f"{args.task_action_schema} schema requires an explicit non-none action bound mode"
            )
        if training_dataset is None:
            raise ValueError(
                f"{args.task_action_schema} schema requires --train-dataset so raw constraints "
                "can be mapped with the exact training scaler"
            )

    process = {"action": fit_action_scaler(training_dataset)} if training_dataset else {}
    scaler = process.get("action")
    validate_action_scaler_contract(scaler, action_dim=args.action_dim)
    action_data = load_action_data(training_dataset) if training_dataset else None
    total_steps = args.horizon * args.action_block
    terminal_support = None
    if args.task_action_schema == "franka_stack" and not stock_cem:
        terminal_support = resolve_franka_terminal_support(
            training_dataset,
            total_steps=total_steps,
            clip_stride=5,
            # Five +1 commands only open the physical fingers to about 0.0387,
            # below the evaluator's strict 0.0399 threshold.  Intersect the
            # exact train-observed phase set with the six-command actuator
            # requirement instead of admitting an impossible-to-score split.
            minimum_terminal_open_steps=6,
        )
        print(
            "[server] Franka train-only terminal support: "
            f"splits={terminal_support['switch_steps']} "
            f"counts={dict(terminal_support['switch_counts'])} "
            f"excluded={dict(terminal_support['excluded_switch_counts'])} "
            f"all_open_plans={terminal_support['all_open_plan_count']}/"
            f"{terminal_support['sampled_plan_count']}",
            flush=True,
        )
    continuous_plan_tv_max = None
    continuous_indices = None if stock_cem else {
        "franka_stack": tuple(range(6)),
        "gear_effective4": tuple(range(4)),
    }.get(args.task_action_schema)
    if continuous_indices is not None:
        continuous_plan_tv_max = resolve_continuous_plan_tv_max(
            training_dataset,
            scaler,
            total_steps=total_steps,
            quantile=args.temporal_tv_quantile,
            continuous_indices=continuous_indices,
        )
        print(
            f"[server] {args.task_action_schema} continuous plan-TV support: "
            f"q={args.temporal_tv_quantile:g} max={continuous_plan_tv_max:.6f}",
            flush=True,
        )
    if args.task_action_schema == "generic" or stock_cem:
        raw_low, raw_high = args.action_low, args.action_high
    else:
        raw_low, raw_high = resolve_raw_action_bounds(
            args.action_bound_mode,
            action_dim=args.action_dim,
            physical_low=args.action_low,
            physical_high=args.action_high,
            action_data=action_data,
            quantile=args.action_bound_quantile,
        )
    projection = None if stock_cem else build_task_action_projection(
        args.task_action_schema,
        action_dim=args.action_dim,
        action_block=args.action_block,
        scaler=scaler,
        raw_low=raw_low,
        raw_high=raw_high,
        terminal_open_steps=(
            terminal_support["terminal_open_steps"] if terminal_support else 0
        ),
        terminal_switch_steps=(
            terminal_support["switch_steps"] if terminal_support else ()
        ),
        terminal_switch_counts=(
            terminal_support["switch_counts"] if terminal_support else ()
        ),
        terminal_support_sample_count=(
            terminal_support["sample_count"] if terminal_support else 0
        ),
        continuous_plan_tv_max=continuous_plan_tv_max,
        continuous_plan_tv_quantile=(
            args.temporal_tv_quantile
            if continuous_plan_tv_max is not None
            else None
        ),
    )

    model = swm.wm.utils.load_pretrained(args.checkpoint)
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    validate_model_action_contract(
        model, action_dim=args.action_dim, action_block=args.action_block
    )
    tf = img_transform(args.img_size)

    config = swm.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        action_block=args.action_block,
        warm_start=args.warm_start,
    )
    plan_bank = None
    if args.planner_mode in {"projected_cem", "stock_cem"}:
        solver = ProjectedCEMSolver(
            model=model,
            batch_size=1,
            num_samples=args.num_samples,
            var_scale=1.0,
            n_steps=args.n_steps,
            topk=args.topk,
            device=args.device,
            seed=args.seed,
            action_projection=projection,
        )
    else:
        if args.task_action_schema != "franka_stack":
            raise ValueError("train terminal plan-bank modes are Franka-only")
        if args.horizon != 5 or args.action_block != 5:
            raise ValueError("Franka terminal plan bank is defined for H5 x block5")
        if args.receding_horizon != args.horizon:
            raise ValueError("Franka terminal plan bank requires full-commit planning")
        if projection is None:
            raise ValueError("Franka terminal plan bank requires action projection")
        selection_mode = args.planner_mode.removeprefix("train_terminal_")
        plan_bank = load_franka_terminal_plan_bank(
            training_dataset,
            scaler,
            horizon=args.horizon,
            action_block=args.action_block,
            minimum_open_commands=6,
            terminal_switch_steps=tuple(terminal_support["switch_steps"]),
            raw_low=np.asarray(raw_low, dtype=np.float64),
            raw_high=np.asarray(raw_high, dtype=np.float64),
            continuous_plan_tv_max=float(continuous_plan_tv_max),
        )
        start_embeddings = None
        goal_embeddings = None
        if selection_mode in ("behavior_retrieval", "wm_rerank"):
            start_embeddings = encode_plan_bank_endpoints(
                model, plan_bank.start_pixels, tf
            )
            goal_embeddings = encode_plan_bank_endpoints(
                model, plan_bank.goal_pixels, tf
            )
        solver = ProjectedPlanBankSolver(
            model=model,
            plan_bank=plan_bank,
            selection_mode=selection_mode,
            continuation=args.plan_bank_continuation,
            device=args.device,
            action_projection=projection,
            start_embeddings=start_embeddings,
            goal_embeddings=goal_embeddings,
            retrieval_topk=args.plan_bank_topk,
        )
        print(
            "[server] Franka train-only terminal plan bank: "
            f"selection={selection_mode} continuation={args.plan_bank_continuation} "
            f"candidates={plan_bank.size} sha256="
            f"{plan_bank.provenance['bank_sha256'][:12]}",
            flush=True,
        )

    if not process:
        print("[server] 警告: 未提供 --dataset, 动作不会反标准化, 结果无效", flush=True)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform={"pixels": tf, "goal": tf}
    )
    policy.set_env(_EnvStub(args.action_dim, args.action_low, args.action_high))
    if projection is not None:
        # Keep the scaler exact while guaranteeing bit-exact raw fixed and
        # categorical values at the environment boundary.
        policy.process["action"] = ProjectedActionScaler(scaler, projection)
    policy.task_action_schema = (
        {
            "name": args.task_action_schema,
            "projection": None,
            "bound_mode": args.action_bound_mode,
            "action_dim": args.action_dim,
            "action_block": args.action_block,
            "checkpoint": args.checkpoint,
            "training_dataset": training_dataset,
            "scaler": scaler.to_dict() if scaler is not None else None,
        }
        if projection is None
        else {
            **projection.to_dict(),
            "bound_mode": args.action_bound_mode,
            "checkpoint": args.checkpoint,
            "training_dataset": training_dataset,
            "data_quantile": (
                args.action_bound_quantile
                if args.action_bound_mode == "data_quantile"
                else None
            ),
            "scaler": scaler.to_dict(),
            "episode_open_latch": (
                {
                    "enabled": True,
                    "trigger": "client_feedback_stack_structure_and_goal_pose_ready",
                    "scope": "episode_until_reset",
                    "activation": "next_natural_replan_boundary",
                    "interrupts_committed_plan": False,
                    "projected_raw_value": 1.0,
                    "applies_before": [
                        "model_scoring",
                        "elite_update",
                        "finalist_scoring",
                        "returned_plan",
                    ],
                    "train_plan_length": total_steps,
                    "train_clip_stride": terminal_support["clip_stride"],
                    "train_sampled_plan_count": terminal_support[
                        "sampled_plan_count"
                    ],
                    "train_all_open_plan_count": terminal_support[
                        "all_open_plan_count"
                    ],
                    "train_exact_open_hold_plan_count": terminal_support[
                        "exact_open_hold_plan_count"
                    ],
                    "train_post_success_episode_count": terminal_support[
                        "post_success_episode_count"
                    ],
                    "train_post_success_min_steps": terminal_support[
                        "post_success_min_steps"
                    ],
                    "train_post_success_max_steps": terminal_support[
                        "post_success_max_steps"
                    ],
                    "train_post_success_action_count": terminal_support[
                        "post_success_action_count"
                    ],
                    "train_post_success_open_action_count": terminal_support[
                        "post_success_open_action_count"
                    ],
                    "strict_minimum_open_commands": terminal_support[
                        "minimum_terminal_open_steps"
                    ],
                    "train_observed_terminal_switch_steps": list(
                        terminal_support["observed_switch_steps"]
                    ),
                    "train_observed_terminal_switch_counts": dict(
                        (str(step), count)
                        for step, count in terminal_support[
                            "observed_switch_counts"
                        ]
                    ),
                    "strict_ineligible_terminal_switch_counts": dict(
                        (str(step), count)
                        for step, count in terminal_support[
                            "excluded_switch_counts"
                        ]
                    ),
                }
                if terminal_support is not None
                else None
            ),
            "plan_proposal": (
                {
                    "mode": "projected_cem",
                    "selection": "iterative_model_cost",
                    "synthetic_cem_candidates": True,
                    "world_model_reranks_candidates": True,
                }
                if plan_bank is None
                else {
                    **solver.proposal_provenance(),
                    "continuation_provenance": {
                        "mode": args.plan_bank_continuation,
                        "shared_across_selection_arms": True,
                        "synthetic_action_sequence": (
                            args.plan_bank_continuation == "open_hold"
                        ),
                        "exact_train_window_match_count": (
                            terminal_support["exact_open_hold_plan_count"]
                            if args.plan_bank_continuation == "open_hold"
                            else None
                        ),
                        "claim": (
                            "transparent zero-arm/open-gripper settling control; "
                            "not an observed train continuation"
                            if args.plan_bank_continuation == "open_hold"
                            else (
                                "no second action plan is executed; the 50-step "
                                "budget is treated as an upper bound"
                                if args.plan_bank_continuation == "abstain"
                                else "reuses a demonstrated terminal action sequence "
                                "outside its original state context"
                            )
                        ),
                    },
                }
            ),
        }
    )
    return policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="path under $STABLEWM_HOME/checkpoints, e.g. lewm_fstack1k_ep40/weights_epoch_39.pt")
    p.add_argument("--socket", default="/tmp/lewm_policy.sock")
    p.add_argument("--train-dataset", "--dataset", dest="dataset", default=None,
        help="训练集 h5（不得传 validation/eval 文件），用于拟合与训练完全一致的 action z-score")
    p.add_argument(
        "--task-action-schema",
        required=True,
        choices=TASK_ACTION_SCHEMAS,
        help=(
            "显式动作投影协议: generic=原始无投影CEM; cube=逐维边界; "
            "gear_effective4=新Gear [x,y,z,yaw]; gear_legacy6/gear=旧6D且raw rx/ry固定为0; "
            "franka_stack=边界且raw夹爪为精确±1"
        ),
    )
    p.add_argument("--img-size", type=int, default=196)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument(
        "--action-low", type=parse_action_bound, default=-1.0,
        help="raw物理下界：一个标量或逗号分隔的逐维值",
    )
    p.add_argument(
        "--action-high", type=parse_action_bound, default=1.0,
        help="raw物理上界：一个标量或逗号分隔的逐维值",
    )
    p.add_argument(
        "--action-bound-mode",
        required=True,
        choices=ACTION_BOUND_MODES,
        help=(
            "none仅用于generic; physical用--action-low/high; data_minmax用训练集逐维极值; "
            "data_quantile用训练集逐维分位数"
        ),
    )
    p.add_argument(
        "--action-bound-quantile",
        type=float,
        default=0.001,
        help="data_quantile模式的双侧分位数q，边界为[q,1-q]",
    )
    p.add_argument(
        "--temporal-tv-quantile",
        type=float,
        default=0.99,
        help=(
            "Franka/Gear-effective4连续动作计划总变差的训练集分位支持边界；"
            "按episode计算，不跨边界（默认0.99）"
        ),
    )
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--receding-horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5, help="frameskip")
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument(
        "--planner-mode",
        choices=(
            "stock_cem",
            "projected_cem",
            *(f"train_terminal_{mode}" for mode in PLAN_BANK_SELECTION_MODES),
        ),
        default="projected_cem",
        help=(
            "stock_cem is the released unconstrained StableWorldModel CEM; "
            "projected_cem uses constrained Gaussian CEM; train_terminal_behavior_central "
            "returns the central demonstrated terminal plan; train_terminal_wm_rerank "
            "scores every strict train terminal plan with the frozen world model"
        ),
    )
    p.add_argument(
        "--plan-bank-continuation",
        choices=PLAN_BANK_CONTINUATIONS,
        default="abstain",
        help="Plan-bank behavior after the first full 25-step terminal proposal.",
    )
    p.add_argument(
        "--plan-bank-topk",
        type=int,
        default=300,
        help="Train-only endpoint-retrieval shortlist scored by WM reranking.",
    )
    p.add_argument("--warm-start", action="store_true")
    p.add_argument(
        "--snap-gripper",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.snap_gripper:
        p.error(
            "--snap-gripper is unsafe because it only changed model scoring. "
            "Use --task-action-schema franka_stack for solver-level projection."
        )

    print(f"[server] 载入 {args.checkpoint}", flush=True)
    policy = build_policy(args)
    effective_samples = getattr(policy.solver, "num_samples", args.num_samples)
    effective_steps = 0 if isinstance(policy.solver, ProjectedPlanBankSolver) else args.n_steps
    print(f"[server] 就绪: horizon={args.horizon} rh={args.receding_horizon} "
          f"action_block={args.action_block} samples={effective_samples} iters={effective_steps} "
          f"schema={args.task_action_schema} planner={args.planner_mode}", flush=True)

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    srv.listen(1)
    print(f"[server] 监听 {args.socket}", flush=True)

    n_act = 0
    try:
      while True:                      # 接受多次连接: 客户端崩溃后不必重启服务器
        conn, _ = srv.accept()
        print("[server] 客户端已连接", flush=True)
        while True:
            msg = recv(conn)
            if msg is None:
                break
            cmd = msg.get("cmd")
            if cmd == "close":
                conn.close()
                print(f"[server] 客户端断开 (累计 {n_act} 次动作)", flush=True)
                break
            if cmd == "reset":
                # clear the receding-horizon buffer so the next act() replans from scratch
                for buf in policy._action_buffer:
                    buf.clear()
                policy._next_init = None
                policy.solver.reset_action_latches()
                episode_seed = msg.get("seed")
                if episode_seed is not None:
                    # A per-episode seed makes paired goal ablations use identical CEM
                    # candidate streams even when an earlier episode terminates early.
                    policy.solver.torch_gen.manual_seed(int(episode_seed))
                send(conn, {
                    "ok": True,
                    "episode_seed": episode_seed,
                    "planner_config": {
                        "horizon": args.horizon,
                        "receding_horizon": args.receding_horizon,
                        "action_block": args.action_block,
                        "action_dim": args.action_dim,
                        "num_samples": effective_samples,
                        "cem_steps": effective_steps,
                        "topk": (1 if isinstance(policy.solver, ProjectedPlanBankSolver) else args.topk),
                        "seed": args.seed,
                        "planner_mode": args.planner_mode,
                        "plan_bank_continuation": args.plan_bank_continuation,
                        "task_action_schema": policy.task_action_schema,
                    },
                })
                continue
            if cmd != "act":
                send(conn, {"ok": False, "error": f"unknown cmd {cmd}"})
                continue

            try:
                pixels = np.frombuffer(msg["pixels"], dtype=np.uint8).reshape(
                    msg["pixels_shape"]
                ).copy()
                goal = np.frombuffer(msg["goal"], dtype=np.uint8).reshape(
                    msg["goal_shape"]
                ).copy()
                if msg.get("latch_open"):
                    # Preserve the official Cube full-commit contract: feedback
                    # constrains the *next* solve, but never truncates the queued
                    # 25-step plan midway through execution.
                    policy.solver.latch_raw_categorical(6, 1.0)
                T = pixels.shape[0]
                info = {
                    "pixels": pixels[None],                 # (1, T, H, W, 3)
                    "goal": np.repeat(goal[None, None], T, axis=1),
                    # `action` must be present: JEPA.get_cost does `goal.pop("action")`
                    # unconditionally. The value is irrelevant — JEPA.rollout overwrites it with
                    # the candidate sequence (`info["action"] = act_0`).
                    "action": np.zeros((1, T, args.action_dim), dtype=np.float32),
                    "terminated": np.zeros(1, dtype=bool),
                }
                action = policy.get_action(info)
                a = np.asarray(action, dtype=np.float32).reshape(-1)
                n_act += 1
                reply = {"ok": True, "action": [float(x) for x in a]}
                plan_diagnostics = getattr(
                    policy.solver, "last_solve_diagnostics", None
                )
                if plan_diagnostics is not None:
                    reply["plan_diagnostics"] = plan_diagnostics
                    reply["abstain"] = bool(plan_diagnostics.get("abstain", False))
                send(conn, reply)
            except Exception:
                send(conn, {"ok": False, "error": traceback.format_exc()})
    except KeyboardInterrupt:
        print("[server] 收到停止信号", flush=True)
    finally:
        srv.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)
        print(f"[server] 结束, 共服务 {n_act} 次动作请求", flush=True)


if __name__ == "__main__":
    sys.exit(main())
