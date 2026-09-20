#!/usr/bin/env python3
"""Closed-loop goal-conditioned evaluation of LeWM on IsaacLab Franka cube stacking.

Runs on the OFFICIAL legacy stack (IsaacLab 2.3 / Isaac Sim 5.1, conda env
``srsa_official_clean``) because that is the stack the released dataset was generated on. On
IsaacLab 3.0-beta the Franka asset 404s and the render does not reproduce the dataset — see
docs/2026-09-03_isaaclab_franka_stack_dataset.md.

The task id must be the **Cosmos** Mimic variant. It is the only one that sets
``sim.render.dome_light_upper_lower_strategy = 4`` ("to remove rendering noise", per Isaac Lab's own
comment) and that natively renders 200x200. With it the live render matches the dataset to a mean
absolute difference of 0.50/255 at the reset frame; with the plain Mimic variant the gap is ~45.

Protocol (per LeWM paper App. F.1): the start state is a frame sampled from a dataset trajectory and
the goal is a state later in that *same* trajectory, which keeps the goal reachable and consistent
with the dataset dynamics. Here the goal is pinned to the first frame satisfying Isaac Lab's
``cubes_stacked`` criterion — i.e. the goal image always shows three successfully stacked cubes —
and the start is ``goal_idx - offset``. Budget follows the paper's 2:1 budget:offset ratio.

The formal initialization is an exact teleport to that observation row. Because the HDF5 scene
state cannot encode PhysX contact caches, formal evaluation pools must first be filtered by an
expert-continuation gate under the identical teleport initialization. Prefix modes are diagnostic:
short prefixes may begin inside a grasp and lose contact, while full prefixes accumulate open-loop
simulation drift; neither is a universal replacement for the gate.

For a LeWM H5 model with five raw actions per macro, the formal distance is
``offset = horizon * action_block = 25`` and the budget is 50 raw actions.
Longer offsets are useful diagnostics, but are not an H5 matched-distance result.
For context, older diagnostic offsets measured over the 1000 demos behave as:

    offset  50   start ~t=194, robot already holding the green cube in the air; tests the final
                 placement only (covers the red-cube placement in just 22% of demos)
    offset 150   start ~t=94, covers the red placement in 97% of demos and the green in 100%;
                 tests two placements plus the regrasp between them

Actions come from a LeWM planning server in the other conda env — start
``scripts/lewm_policy_server.py`` first. See its docstring for why two processes.

Usage (after starting the server):
    python scripts/isaac_stack_eval.py --offset 25 --budget 50 --num-eval 5
"""

from __future__ import annotations
from repro_paths import resolve_path as _resolve_artifact

import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Mimic-v0")
p.add_argument("--src", default=_resolve_artifact('data:cosmos_dataset_1k.hdf5'))
p.add_argument("--offset", type=int, default=25, help="goal offset in ENV steps")
p.add_argument("--budget", type=int, default=None, help="default = 2 * offset (paper convention)")
p.add_argument("--num-eval", type=int, default=5)
p.add_argument("--action-block", type=int, default=5, help="frameskip; obs history is sampled this often")
p.add_argument("--expected-planner-horizon", type=int, default=5,
               help="Formal LeWM CEM horizon in macro-actions; training is supervised through H5.")
p.add_argument("--expected-receding-horizon", type=int, default=5,
               help="Formal full-commit macro-actions before replanning.")
p.add_argument(
    "--expected-planner-mode",
    default="projected_cem",
    choices=[
        "projected_cem",
        "train_terminal_behavior_central",
        "train_terminal_behavior_retrieval",
        "train_terminal_wm_rerank",
    ],
    help="Fail-closed server planner identity required by this evaluation arm.",
)
p.add_argument(
    "--expected-plan-bank-continuation",
    default="abstain",
    choices=["abstain", "open_hold", "repeat_terminal"],
    help="Fail-closed behavior after the first complete 25-step bank plan.",
)
p.add_argument("--history", type=int, default=1,
               help="送给 planner 的观测帧数；默认 1 与官方 EnvPool 对齐。"
                    "当前 rollout/policy 接口无法正确执行多帧历史对应的动作时序")
p.add_argument("--socket", default="/tmp/lewm_policy.sock")
p.add_argument("--policy", default="lewm", choices=["lewm", "expert", "zero", "drop"],
               help="expert = 回放数据集动作; zero = 全零(夹爪立即张开); "
                    "drop = 固定下降后张开夹爪")
p.add_argument("--drop-steps", type=int, default=4,
               help="drop 基线保持夹紧并竖直下降的步数")
p.add_argument("--drop-z", type=float, default=-0.02,
               help="drop 基线下降阶段每步的原始 z 动作(IK scale=0.5)")
p.add_argument("--shuffle-goals", action="store_true",
               help="在所选 episode 间做确定性的无固定点目标置换，用于检验目标条件是否有效")
p.add_argument("--planner-seed-per-demo", action="store_true",
               help="每个 demo 用 seed+demo 重置 CEM RNG，使原目标/乱序目标严格共享候选流")
p.add_argument(
    "--planner-seed-base",
    type=int,
    default=None,
    help="与环境 --seed 分离的 planner base seed；默认沿用 --seed。",
)
p.add_argument("--demos", type=str, default=None,
               help="逗号分隔的 demo id, 指定则不随机采样(用于精确复现某一条)")
p.add_argument(
    "--expert-gate-json",
    default=None,
    help=(
        "LeWM正式评测必需：同task/stage/offset/split、init_mode=teleport的expert "
        "continuation JSON；只从其中success=true的demo采样"
    ),
)
p.add_argument("--log-actions", action="store_true",
               help="把每步实际执行的动作和夹爪关节状态存成 npz, 用于分析动作分布")
p.add_argument("--record", type=int, default=0,
               help="每 N 步录一帧(table_cam+wrist_cam 并排)存成 gif; 0=不录。"
                    "仅调用 sim.render(), 不额外 app.update(), 避免注入物理步")
p.add_argument("--stage", type=int, default=3, choices=[2, 3],
               help="3 = 官方 cubes_stacked(三块); 2 = 只要求蓝+红两块叠好(子任务)")
p.add_argument("--auto-offset", action="store_true",
               help="起点固定 t=0(干净重置), offset = 该 demo 首次达成目标的帧号, budget = 2x offset。"
                    "配 --stage 2 即为「从头把红块放到蓝块上」这一子任务")
p.add_argument("--from-start", action="store_true",
               help="从 t=0 整条回放(而非 goal-offset), 用来判定发散是否由中途 reset_to 引起")
p.add_argument("--init-mode", default="teleport",
               choices=["hybrid_prefix", "expert_prefix", "teleport"],
               help="teleport: 精确恢复s_start（正式协议，case须另经expert continuation门禁）；"
                    "hybrid_prefix/expert_prefix仅用于诊断隐藏状态与开环漂移")
p.add_argument("--prefix-steps", type=int, default=25,
               help="hybrid_prefix 在计分起点前回放的专家步数，默认25")
p.add_argument("--trace", action="store_true",
               help="expert 模式下逐步比对回放状态与数据集记录状态, 用来定位开环发散")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--dataset-split", default="eval", choices=["train", "val", "eval", "all"],
               help="只从确定性的 source-demo split 取 episode；默认 held-out eval")
p.add_argument("--split-seed", type=int, default=3072)
p.add_argument("--train-fraction", type=float, default=0.8)
p.add_argument("--val-fraction", type=float, default=0.1)
p.add_argument("--out", default="./eval_out")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.policy == "lewm" and args.expert_gate_json is None:
    p.error("--policy lewm requires --expert-gate-json; ungated contact states are invalid")
if args.policy == "lewm" and args.init_mode != "teleport":
    p.error("formal --policy lewm evaluation requires --init-mode teleport")
if args.policy == "lewm" and (args.auto_offset or args.from_start):
    p.error("formal --policy lewm evaluation forbids --auto-offset/--from-start")
if args.policy == "lewm" and args.history != 1:
    p.error("the released LeWM planning interface supports exactly --history 1")
if args.policy == "lewm":
    expected_budget = 2 * args.offset if args.budget is None else args.budget
    expected_offset = args.expected_planner_horizon * args.action_block
    if args.offset != expected_offset:
        p.error(
            "formal LeWM start/goal distance must equal planner horizon * "
            f"action block: offset={args.offset}, expected {expected_offset}"
        )
    if expected_budget != 2 * args.offset:
        p.error("formal LeWM budget must be exactly 2 * offset")
args.enable_cameras = True
app = AppLauncher(args).app

import json
import os
import pickle
import socket
import struct
import time
from collections import deque

import gymnasium as gym
import h5py
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
try:
    import isaaclab_mimic.envs  # noqa: F401
except Exception as e:  # pragma: no cover
    print("[warn] isaaclab_mimic:", e, flush=True)
from isaaclab_tasks.utils import parse_env_cfg
from franka_stack_utils import (
    GOAL_POSITION_THRESHOLD,
    HEIGHT_DIFF as H_DIFF,
    HEIGHT_THRESHOLD as H_T,
    XY_THRESHOLD as XY_T,
    SPLIT_NAMES,
    cubes_stacked,
    deterministic_episode_splits,
    goal_stack_diagnostics,
    load_expert_gate,
    observation_state_location,
    prefix_start_index,
    validate_franka_action,
    validate_franka_planner_config,
)
from result_commit import commit_json_result


def active_stack_snapshot(cube_positions, goal_cube_positions, stage):
    """Return graded metrics for the pair and cube manipulated at this stage.

    Stage 2 places cube 2 on cube 1; stage 3 places cube 3 on cube 2.  Keeping
    this selection explicit prevents a final-stack evaluation from reporting
    progress only on the already-completed cube-1/cube-2 pair.
    """

    if stage not in (2, 3):
        raise ValueError(f"stage must be 2 or 3, got {stage}")
    current = np.asarray(cube_positions, dtype=np.float64)
    goal = np.asarray(goal_cube_positions, dtype=np.float64)
    if current.size != 9 or goal.size != 9:
        raise ValueError(
            f"cube and goal positions must each contain 9 values, got "
            f"{current.shape}/{goal.shape}"
        )
    current = current.reshape(3, 3)
    goal = goal.reshape(3, 3)
    lower_index, upper_index = stage - 2, stage - 1
    delta = current[lower_index] - current[upper_index]
    xy = float(np.linalg.norm(delta[:2]))
    return {
        "active_pair": f"{lower_index + 1}{upper_index + 1}",
        "active_cube": upper_index + 1,
        "pair_xy": xy,
        "pair_dz": float(delta[2]),
        "active_cube_z": float(current[upper_index, 2]),
        "active_cube_goal_distance": float(
            np.linalg.norm(current[upper_index] - goal[upper_index])
        ),
        "pair_condition_count": (
            int(xy < XY_T)
            + int((abs(delta[2]) - H_DIFF) < H_T)
            + int(delta[2] < 0)
        ),
    }


BUDGET = args.budget if args.budget is not None else 2 * args.offset
PLANNER_SEED_BASE = (
    args.seed if args.planner_seed_base is None else args.planner_seed_base
)
os.makedirs(args.out, exist_ok=True)

def send(sock, obj):
    d = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(d)) + d)


def recv(sock):
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


# ---------------------------------------------------------------- env
env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
env_cfg.seed = args.seed
env_cfg.observations.policy.concatenate_terms = False
env_cfg.terminations.time_out = None
env_cfg.recorders = None
env_cfg.terminations.success = None
# Any one of these terms used to trigger ManagerBasedRLEnv's automatic reset.  Because
# reset happens inside step(), ignoring the done flag silently changes the evaluated episode.
for dropping_name in ("cube_1_dropping", "cube_2_dropping", "cube_3_dropping"):
    if hasattr(env_cfg.terminations, dropping_name):
        setattr(env_cfg.terminations, dropping_name, None)
print("[OK] 已禁用 time_out/success/cube_{1,2,3}_dropping；任何 done 信号将使评测报错", flush=True)

env = gym.make(args.task, cfg=env_cfg).unwrapped
RES = env_cfg.scene.table_cam.width
CROP = (RES - 196) // 2
print(f"[OK] env {args.task} | 相机 {RES}x{RES} -> 裁剪 196 | offset={args.offset} budget={BUDGET}", flush=True)


def as_np(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    try:
        import warp as wp
        return wp.to_torch(x).cpu().numpy()
    except Exception:
        return np.asarray(x)


def grab(cam="table_cam", n_render=2):
    """Render and return the 196 centre crop.

    Call `sim.render()` only — never a bare `app.update()`. Isaac Lab's own render already wraps
    the app update in `/app/player/playSimulations = False` (simulation_context.py), so it refreshes
    the RTX sensor without letting the timeline advance physics. Adding a bare `app.update()` on top
    re-enables stepping and silently corrupts the rollout; pausing the timeline instead stops the
    camera updating at all. Measured on expert replay from t=0 (max joint error at step 40, and
    frame-to-frame variation of the captured images):

        no capture at all               0.17349 rad    —
        sim.render() only               0.17349 rad    frames vary, |Δ| vs dataset 0.28-5.53
        sim.render() + app.update()     0.22387 rad    frames vary, |Δ| vs dataset 0.39-8.84
        pause + render + app.update()   —              frames FROZEN (identical every step)
    """
    for _ in range(n_render):
        env.sim.render()
    env.scene[cam].update(env.physics_dt, force_recompute=True)
    img = as_np(env.scene[cam].data.output["rgb"])[0][..., :3].astype(np.uint8)
    return img[CROP:CROP + 196, CROP:CROP + 196]


src = h5py.File(args.src, "r")["data"]


def state_at_numpy(demo, t):
    group_name, row = observation_state_location(t)
    st = src[f"demo_{demo}"][group_name]
    d = {"articulation": {}, "rigid_object": {}}
    for kind in ("articulation", "rigid_object"):
        if kind not in st:
            continue
        for name, grp in st[kind].items():
            d[kind][name] = {k: np.asarray(v[row]) for k, v in grp.items()}
    return d


def state_at(demo, t):
    return {
        kind: {
            name: {key: torch.tensor(value[None], dtype=torch.float32, device=env.device)
                   for key, value in values.items()}
            for name, values in entities.items()
        }
        for kind, entities in state_at_numpy(demo, t).items()
    }


def reset_to(demo, t):
    env_ids = torch.tensor([0], device=env.device)
    env.reset_to(state_at(demo, t), env_ids, is_relative=True)
    # reset_to restores the measured velocity as both actual state and controller target.
    # The recorder action a_t assumes no residual velocity command; make that explicit.
    robot = env.scene["robot"]
    robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids)
    env.scene.write_data_to_sim()


def source_cube_positions(demo, t):
    state = state_at_numpy(demo, t)
    return np.stack([state["rigid_object"][f"cube_{k}"]["root_pose"][:3]
                     for k in (1, 2, 3)])


def live_diagnostics(goal_cube_positions):
    o = as_np(env.scene.env_origins)[0]
    cp = np.stack([as_np(env.scene[f"cube_{k}"].data.root_pos_w)[0] - o for k in (1, 2, 3)])
    gp = as_np(env.scene["robot"].data.joint_pos)[0, -2:]
    return goal_stack_diagnostics(cp, gp, goal_cube_positions, stage=args.stage)


def checked_step(action, *, phase, demo, step):
    result = env.step(action)
    terminated, truncated = result[2], result[3]
    if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
        raise RuntimeError(
            f"unexpected termination during {phase}: demo={demo} step={step}; "
            "the environment may have auto-reset, so this rollout is invalid"
        )
    return result


def endpoint_diagnostics(demo, t, rendered_image):
    expected = state_at_numpy(demo, t)
    origin = as_np(env.scene.env_origins)[0]
    actual_cube = np.stack([
        as_np(env.scene[f"cube_{k}"].data.root_pos_w)[0] - origin for k in (1, 2, 3)
    ])
    expected_cube = np.stack([
        expected["rigid_object"][f"cube_{k}"]["root_pose"][:3] for k in (1, 2, 3)
    ])
    actual_joint = as_np(env.scene["robot"].data.joint_pos)[0]
    expected_joint = expected["articulation"]["robot"]["joint_position"]
    raw = np.asarray(src[f"demo_{demo}"]["obs"]["table_cam"][t])[
        CROP:CROP + 196, CROP:CROP + 196, :3
    ]
    cube_error = np.linalg.norm(actual_cube - expected_cube, axis=1)
    joint_error = np.abs(actual_joint - expected_joint)
    return {
        "mode": args.init_mode,
        "source_row": int(t),
        "cube_error_m": [float(x) for x in cube_error],
        "cube_max_error_m": float(cube_error.max()),
        "joint_max_error_rad": float(joint_error.max()),
        "arm_joint_max_error_rad": float(joint_error[:7].max()),
        "pixel_mae_255": float(np.mean(np.abs(rendered_image.astype(np.float32) - raw.astype(np.float32)))),
    }


def state_error_diagnostics(demo, t):
    """Cheap state-only comparison used while tracing an expert prefix."""
    expected = state_at_numpy(demo, t)
    origin = as_np(env.scene.env_origins)[0]
    actual_cube = np.stack([
        as_np(env.scene[f"cube_{k}"].data.root_pos_w)[0] - origin for k in (1, 2, 3)
    ])
    expected_cube = np.stack([
        expected["rigid_object"][f"cube_{k}"]["root_pose"][:3] for k in (1, 2, 3)
    ])
    actual_joint = as_np(env.scene["robot"].data.joint_pos)[0]
    expected_joint = expected["articulation"]["robot"]["joint_position"]
    return np.linalg.norm(actual_cube - expected_cube, axis=1), np.abs(actual_joint - expected_joint)


# ---------------------------------------------------------------- pick episodes
rng = np.random.default_rng(args.seed)
cands = []
split_assignment = deterministic_episode_splits(
    1000, args.split_seed, args.train_fraction, args.val_fraction
)
wanted_split = None if args.dataset_split == "all" else SPLIT_NAMES.index(args.dataset_split)
source_demo_ids = [i for i in range(1000)
                   if wanted_split is None or split_assignment[i] == wanted_split]
expert_gate_audit = None
if args.expert_gate_json is not None:
    expert_eligible_ids, expert_gate_audit = load_expert_gate(
        args.expert_gate_json,
        task=args.task,
        stage=args.stage,
        offset=args.offset,
        budget=BUDGET,
        source_dataset=args.src,
        dataset_split=args.dataset_split,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    source_demo_ids = [i for i in source_demo_ids if i in expert_eligible_ids]
    print(
        f"[OK] expert gate {expert_gate_audit['eligible_count']}/"
        f"{expert_gate_audit['tested_count']} eligible; "
        f"sha256={str(expert_gate_audit['sha256'])[:12]}",
        flush=True,
    )
for i in source_demo_ids:
    o = src[f"demo_{i}"]["obs"]
    ok = cubes_stacked(o["cube_positions"][:], o["gripper_pos"][:], stage=args.stage)
    if not ok.any():
        continue
    g = int(np.argmax(ok))
    if args.auto_offset or g - args.offset >= 0:
        cands.append((i, g))
print(f"[OK] split={args.dataset_split} stage={args.stage}: {len(cands)}/{len(source_demo_ids)} 条 demo "
      "的 goal_idx-offset >= 0", flush=True)
if args.demos:
    requested = [int(x) for x in args.demos.split(",")]
    by_demo = {c[0]: c for c in cands}
    missing = [demo for demo in requested if demo not in by_demo]
    if missing:
        raise ValueError(f"指定 demo 不满足当前 stage/offset 条件: {missing}")
    # Preserve the requested order.  CEM owns a stateful RNG, so sorting the
    # demos here changes every later candidate stream and prevents exact replay.
    sel = [by_demo[demo] for demo in requested]
    print(f"[OK] 指定 demo: {[c[0] for c in sel]}", flush=True)
else:
    sel = [cands[k] for k in rng.choice(len(cands), size=min(args.num_eval, len(cands)), replace=False)]

goal_sel = list(sel)
if args.shuffle_goals:
    if args.policy != "lewm":
        raise ValueError("--shuffle-goals 只对 --policy lewm 有意义")
    if len(sel) < 2:
        raise ValueError("--shuffle-goals 至少需要两个 episode")
    # Deterministic derangement: every episode receives another selected demo's goal.
    # Keep this RNG independent of demo selection and of the planner RNG.
    goal_rng = np.random.default_rng(args.seed + 1_000_003)
    donor_idx = np.arange(len(sel))
    selected_goal_positions = [source_cube_positions(demo, goal_frame)
                               for demo, goal_frame in sel]
    for _ in range(10_000):
        goal_rng.shuffle(donor_idx)
        no_fixed_points = np.all(donor_idx != np.arange(len(sel)))
        # Reject visually/physically equivalent donor goals: a shuffled pair is only
        # informative if at least one participating named cube differs by >4 cm.
        distinct_goals = all(
            not np.all(np.linalg.norm(selected_goal_positions[i][:args.stage] -
                                      selected_goal_positions[int(j)][:args.stage], axis=1)
                       <= GOAL_POSITION_THRESHOLD)
            for i, j in enumerate(donor_idx)
        )
        if no_fixed_points and distinct_goals:
            break
    else:
        raise ValueError("无法在所选 episodes 中构造 goal-position 不等价的无固定点置换")
    goal_sel = [sel[int(k)] for k in donor_idx]
    print("[OK] 目标已无固定点置换: "
          f"{[(src_ep[0], goal_ep[0]) for src_ep, goal_ep in zip(sel, goal_sel)]}", flush=True)

# ---------------------------------------------------------------- connect
sock = None
if args.policy == "lewm":
  sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  for attempt in range(30):
    try:
        sock.connect(args.socket)
        break
    except (FileNotFoundError, ConnectionRefusedError):
        time.sleep(2)
  else:
    raise SystemExit(f"连不上策略服务器 {args.socket} — 先启动 scripts/lewm_policy_server.py")
  print("[OK] 已连接策略服务器", flush=True)
print(f"[OK] policy = {args.policy}", flush=True)

env.reset()
for _ in range(60):
    env.sim.render(); app.update()

results = []
planner_config = None
planner_raw_bounds = None
for ep, ((demo, goal_idx), (goal_demo, goal_frame)) in enumerate(zip(sel, goal_sel)):
    if args.auto_offset:
        start, budget = 0, 2 * goal_idx
    elif args.from_start:
        start, budget = 0, goal_idx + 10
    else:
        start, budget = goal_idx - args.offset, BUDGET

    # Use the dataset row itself as the goal image.  Re-rendering after a misaligned
    # state restore can otherwise make the claimed "first successful frame" untrue.
    raw_goal = np.asarray(src[f"demo_{goal_demo}"]["obs"]["table_cam"][goal_frame])
    goal_img = raw_goal[CROP:CROP + 196, CROP:CROP + 196, :3].astype(np.uint8)
    goal_cube_positions = source_cube_positions(goal_demo, goal_frame)
    goal_gripper = np.asarray(src[f"demo_{goal_demo}"]["obs"]["gripper_pos"][goal_frame])
    goal_reference_diag = goal_stack_diagnostics(
        goal_cube_positions, goal_gripper, goal_cube_positions, stage=args.stage
    )
    if not goal_reference_diag["success"]:
        raise RuntimeError(
            f"goal demo_{goal_demo}[{goal_frame}] is not strict-successful after row->state mapping: "
            f"{goal_reference_diag}"
        )
    prefix_t0 = time.time()
    prefix_start = prefix_start_index(start, args.init_mode, args.prefix_steps)
    reset_to(demo, prefix_start)
    if prefix_start < start:
        prefix_actions = np.asarray(src[f"demo_{demo}"]["actions"][prefix_start:start])
        for local_step, source_action in enumerate(prefix_actions):
            source_step = prefix_start + local_step
            prefix_action = torch.tensor(
                source_action, dtype=torch.float32, device=env.device
            ).reshape(1, -1)
            checked_step(prefix_action, phase=args.init_mode, demo=demo, step=source_step)
            if args.trace and (local_step % 5 == 0 or source_step + 1 == start):
                cube_err, joint_err = state_error_diagnostics(demo, source_step + 1)
                print(
                    f"        prefix {source_step:3d} -> s{source_step + 1}: "
                    f"cube={[round(x * 1000, 1) for x in cube_err]}mm "
                    f"joint_max={joint_err.max():.5f}rad",
                    flush=True,
                )
    prefix_seconds = time.time() - prefix_t0
    prefix_count = start - prefix_start
    obs_img = grab()
    init_diag = endpoint_diagnostics(demo, start, obs_img)
    print(
        f"        init={args.init_mode} prefix={prefix_count}步 "
        f"cube误差={[round(x * 1000, 2) for x in init_diag['cube_error_m']]}mm | "
        f"joint_max={init_diag['joint_max_error_rad']:.5f}rad | "
        f"pixel_MAE={init_diag['pixel_mae_255']:.3f}/255 ({prefix_seconds:.1f}s)",
        flush=True,
    )

    if sock is not None:
        planner_seed = (
            PLANNER_SEED_BASE + demo if args.planner_seed_per_demo else None
        )
        reset_msg = {"cmd": "reset"}
        if planner_seed is not None:
            reset_msg["seed"] = planner_seed
        send(sock, reset_msg)
        reset_reply = recv(sock)
        if reset_reply is None or not reset_reply.get("ok"):
            raise SystemExit(f"策略服务器 reset 失败: {reset_reply}")
        if planner_config is None:
            planner_config = reset_reply.get("planner_config")
            planner_raw_bounds = validate_franka_planner_config(
                planner_config,
                args.action_block,
                expected_horizon=args.expected_planner_horizon,
                expected_receding_horizon=args.expected_receding_horizon,
                expected_planner_mode=args.expected_planner_mode,
                expected_plan_bank_continuation=(
                    args.expected_plan_bank_continuation
                    if args.expected_planner_mode.startswith("train_terminal_")
                    else None
                ),
            )
            proposal = planner_config["task_action_schema"].get("plan_proposal", {})
            if args.expected_planner_mode.startswith("train_terminal_"):
                source_path = str(Path(proposal.get("source_dataset", "")).expanduser().resolve())
                if source_path != str(Path(args.src).expanduser().resolve()):
                    raise ValueError(
                        "terminal plan bank source dataset does not match evaluation source"
                    )
                bank_source_ids = set(map(int, proposal.get("source_demo_ids", [])))
                invalid_train_membership = {
                    demo
                    for demo in bank_source_ids
                    if demo < 0
                    or demo >= len(split_assignment)
                    or split_assignment[demo] != SPLIT_NAMES.index("train")
                }
                if invalid_train_membership:
                    raise ValueError(
                        "terminal plan bank contains IDs outside the recomputed "
                        f"train split: {sorted(invalid_train_membership)}"
                    )
            proposal_summary = proposal.get("mode", "unknown")
            if proposal_summary == "train_terminal_plan_bank":
                proposal_summary += (
                    f"/{proposal.get('selection')} n={proposal.get('candidate_count')} "
                    f"bank={str(proposal.get('bank_sha256', ''))[:12]}"
                )
            print(
                "[OK] planner_config: "
                f"mode={planner_config.get('planner_mode')} "
                f"H/RH/block={planner_config.get('horizon')}/"
                f"{planner_config.get('receding_horizon')}/"
                f"{planner_config.get('action_block')} proposal={proposal_summary}",
                flush=True,
            )

    initial_cubes = np.stack([
        as_np(env.scene[f"cube_{k}"].data.root_pos_w)[0]
        - as_np(env.scene.env_origins)[0]
        for k in (1, 2, 3)
    ])
    initial_active = active_stack_snapshot(
        initial_cubes, goal_cube_positions, args.stage
    )
    pair_tag = initial_active["active_pair"]
    pair_xy_key = f"xy{pair_tag}"
    pair_dz_key = f"dz{pair_tag}_at_best"
    best = {
        "active_pair": pair_tag,
        "active_cube": initial_active["active_cube"],
        "active_pair_xy_min": initial_active["pair_xy"],
        "active_pair_dz_at_xy_min": initial_active["pair_dz"],
        "active_cube_lifted": initial_active["active_cube_z"] > 0.05,
        "max_pair_conditions": initial_active["pair_condition_count"],
        "active_cube_goal_distance_initial": initial_active[
            "active_cube_goal_distance"
        ],
        "active_cube_goal_distance_min": initial_active[
            "active_cube_goal_distance"
        ],
        # Keep stage-specific and legacy aliases in the JSON for existing analysis.
        pair_xy_key: initial_active["pair_xy"],
        pair_dz_key: initial_active["pair_dz"],
        "lifted": initial_active["active_cube_z"] > 0.05,
        "n_cond": initial_active["pair_condition_count"],
    }

    def track():
        o = as_np(env.scene.env_origins)[0]
        cubes = np.stack([
            as_np(env.scene[f"cube_{k}"].data.root_pos_w)[0] - o
            for k in (1, 2, 3)
        ])
        snapshot = active_stack_snapshot(cubes, goal_cube_positions, args.stage)
        if snapshot["pair_xy"] < best["active_pair_xy_min"]:
            best["active_pair_xy_min"] = snapshot["pair_xy"]
            best["active_pair_dz_at_xy_min"] = snapshot["pair_dz"]
            best[pair_xy_key] = snapshot["pair_xy"]
            best[pair_dz_key] = snapshot["pair_dz"]
        if snapshot["active_cube_z"] > 0.05:
            best["active_cube_lifted"] = True
            best["lifted"] = True
        best["max_pair_conditions"] = max(
            best["max_pair_conditions"], snapshot["pair_condition_count"]
        )
        best["n_cond"] = best["max_pair_conditions"]
        best["active_cube_goal_distance_min"] = min(
            best["active_cube_goal_distance_min"],
            snapshot["active_cube_goal_distance"],
        )

    if args.policy == "expert":
        # Root actions[t] is a_t. obs/actions is mdp.last_action (a_(t-1)).
        acts = np.asarray(src[f"demo_{demo}"]["actions"][start:start + budget])
    hist = deque([obs_img] * args.history, maxlen=args.history)
    success, t_success = False, -1
    rec = []
    act_log, grip_log = [], []
    episode_plan_diagnostics = []
    seen_plan_diagnostics = set()
    if args.record:
        rec.append(np.concatenate([obs_img, grab("wrist_cam")], 1))
    t0 = time.time()
    for step in range(budget):
        px = np.ascontiguousarray(np.stack(hist), dtype=np.uint8)
        gl = np.ascontiguousarray(goal_img, dtype=np.uint8)
        if args.policy == "lewm":
            feedback = live_diagnostics(goal_cube_positions)
            latch_open = bool(
                feedback["structure_ok"] and feedback["goal_position_ok"]
            )
            send(sock, {"cmd": "act", "pixels": px.tobytes(), "pixels_shape": tuple(px.shape),
                        "goal": gl.tobytes(), "goal_shape": tuple(gl.shape),
                        "latch_open": latch_open})
            rep = recv(sock)
            if rep is None or not rep.get("ok"):
                raise SystemExit(f"策略服务器报错:\n{rep and rep.get('error')}")
            plan_diagnostics = rep.get("plan_diagnostics")
            if isinstance(plan_diagnostics, dict):
                diagnostic_index = int(
                    plan_diagnostics.get("episode_solve_index", -1)
                )
                if diagnostic_index not in seen_plan_diagnostics:
                    episode_plan_diagnostics.append(plan_diagnostics)
                    seen_plan_diagnostics.add(diagnostic_index)
            if rep.get("abstain"):
                print(
                    f"        planner在完整宏动作后于step {step} fail-closed abstain",
                    flush=True,
                )
                break
            action_np = validate_franka_action(rep["action"], *planner_raw_bounds)
            a = torch.tensor(action_np, dtype=torch.float32, device=env.device).reshape(1, -1)
        elif args.policy == "expert":
            src_a = acts[step] if step < len(acts) else acts[-1]
            a = torch.tensor(src_a, dtype=torch.float32, device=env.device).reshape(1, -1)
        elif args.policy == "drop":
            a = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
            if step < args.drop_steps:
                a[0, 2] = args.drop_z
                a[0, -1] = -1.0             # BinaryJointPositionAction: negative = close
            else:
                a[0, -1] = 1.0              # non-negative = open
        else:
            # This is also an immediate-release baseline: the binary gripper maps 0 to open.
            a = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
        checked_step(a, phase="evaluated_rollout", demo=demo, step=step)
        track()
        if args.log_actions:
            act_log.append(as_np(a)[0].copy())
            grip_log.append(as_np(env.scene["robot"].data.joint_pos)[0, -2:].copy())
        if args.trace and args.policy == "expert" and step % 5 == 0:
            gt = np.asarray(src[f"demo_{demo}"]["states"]["rigid_object"]["cube_3"]["root_pose"][
                min(start + step, goal_idx)])[:3]
            got = as_np(env.scene["cube_3"].data.root_pos_w)[0] - as_np(env.scene.env_origins)[0]
            jgt = np.asarray(src[f"demo_{demo}"]["states"]["articulation"]["robot"]["joint_position"][
                min(start + step, goal_idx)])[:7]
            jgot = as_np(env.scene["robot"].data.joint_pos)[0, :7]
            print(f"        step {step:3d} cube3 偏差 {np.linalg.norm(got-gt)*1000:6.1f} mm | "
                  f"关节最大偏差 {np.abs(jgot-jgt).max():.4f} rad", flush=True)
        if bool(live_diagnostics(goal_cube_positions)["success"]):
            success, t_success = True, step
            break
        if (step + 1) % args.action_block == 0:
            hist.append(grab())
        if args.record and (step + 1) % args.record == 0:
            rec.append(np.concatenate([grab("table_cam"), grab("wrist_cam")], 1))

    dt = time.time() - t0
    if args.log_actions and act_log:
        np.savez(os.path.join(args.out, f"actions_demo{demo}_{args.init_mode}_off{args.offset}.npz"),
                 actions=np.stack(act_log), gripper=np.stack(grip_log))
        print(f"        动作日志 -> actions_demo{demo}_{args.init_mode}_off{args.offset}.npz "
              f"({len(act_log)} 步)", flush=True)
    if rec:
        import imageio.v3 as iio
        # 末尾补上目标帧, 便于对照
        tail = np.concatenate([goal_img, np.zeros_like(goal_img)], 1)
        gif = os.path.join(
            args.out, f"rollout_demo{demo}_{args.init_mode}_s{args.stage}_off{args.offset}.gif"
        )
        iio.imwrite(gif, np.stack(rec + [tail] * 6), duration=80, loop=0)
        print(f"        已录 {len(rec)} 帧 -> {gif}", flush=True)
    diag = live_diagnostics(goal_cube_positions)
    active_goal_key = f"goal_dist_{best['active_cube']}"
    best["active_cube_goal_distance_final"] = float(diag[active_goal_key])
    best["active_cube_goal_progress_best"] = float(
        best["active_cube_goal_distance_initial"]
        - best["active_cube_goal_distance_min"]
    )
    best["active_cube_goal_progress_final"] = float(
        best["active_cube_goal_distance_initial"]
        - best["active_cube_goal_distance_final"]
    )
    results.append(dict(demo=demo, goal_idx=goal_idx, goal_demo=goal_demo, goal_frame=goal_frame,
                        planner_seed=(
                            PLANNER_SEED_BASE + demo
                            if args.planner_seed_per_demo
                            else None
                        ),
                        start=start, budget=budget,
                        init_mode=args.init_mode, prefix_steps=prefix_count, prefix_start=prefix_start,
                        prefix_seconds=round(prefix_seconds, 1), init_endpoint=init_diag,
                        success=success, t_success=t_success, seconds=round(dt, 1),
                        goal_cube_positions=goal_cube_positions.tolist(),
                        goal_reference=goal_reference_diag,
                        success_definition={"stage": args.stage,
                                            "goal_position_threshold": GOAL_POSITION_THRESHOLD,
                                            "named_cube_pose_match": True},
                        plan_diagnostics=episode_plan_diagnostics,
                        final=diag, best=dict(best)))
    print(f"  [{ep+1}/{len(sel)}] demo_{demo} start={start} goal={goal_idx} budget={budget} "
          f"-> {'成功' if success else '失败'}"
          f"{f' @step {t_success}' if success else ''}  ({dt:.0f}s)", flush=True)
    print(
        f"        活动对象: pair={best['active_pair']} cube{best['active_cube']} | "
        f"全程最近 xy={best['active_pair_xy_min']:.4f} "
        f"(此刻 dz={best['active_pair_dz_at_xy_min']:+.4f}) | "
        f"活动 cube 曾被抬起={best['active_cube_lifted']} | "
        f"最多满足 {best['max_pair_conditions']}/3 条 | "
        f"goal距离 initial/min/final="
        f"{best['active_cube_goal_distance_initial']:.4f}/"
        f"{best['active_cube_goal_distance_min']:.4f}/"
        f"{best['active_cube_goal_distance_final']:.4f}m",
        flush=True,
    )
    if not success:
        print(f"        末态 xy12={diag['xy12']} xy23={diag['xy23']} (需<0.04) | "
              f"h12={diag['h12']} h23={diag['h23']} (需<0.005) | "
              f"dz12={diag['dz12']} dz23={diag['dz23']} (需<0) | 指={diag['finger']} (需≈0.04)", flush=True)
        print(f"        goal距离(m): {[round(diag[f'goal_dist_{k}'], 4) for k in range(1, args.stage + 1)]} "
              f"(每个需<{GOAL_POSITION_THRESHOLD}) | structure={diag['structure_ok']} "
              f"gripper={diag['gripper_open']} goal_pose={diag['goal_position_ok']}", flush=True)

if sock is not None:
    send(sock, {"cmd": "close"}); sock.close()

sr = float(np.mean([r["success"] for r in results])) if results else float("nan")
bx = (np.array([r["best"]["active_pair_xy_min"] for r in results])
      if results else np.array([9.9]))
lift = (np.mean([r["best"]["active_cube_lifted"] for r in results])
        if results else 0.0)
progress_best = (np.array([r["best"]["active_cube_goal_progress_best"] for r in results])
                 if results else np.array([0.0]))
progress_final = (np.array([r["best"]["active_cube_goal_progress_final"] for r in results])
                  if results else np.array([0.0]))
active_pair = "12" if args.stage == 2 else "23"
active_cube = 2 if args.stage == 2 else 3
print(f"\n[RESULT] stage={args.stage} n={len(results)} 成功率 {sr*100:.1f}%", flush=True)
print(
    f"[GRADED] 活动 pair={active_pair} cube{active_cube} | 全程最近 xy: "
    f"中位 {np.median(bx):.4f} 最小 {bx.min():.4f} 最大 {bx.max():.4f} "
    f"(阈值 0.04) | <0.04 的比例 {100*(bx<0.04).mean():.0f}% | "
    f"活动 cube 曾被抬起 {100*lift:.0f}% | goal进展 best/final 中位 "
    f"{np.median(progress_best):+.4f}/{np.median(progress_final):+.4f}m | "
    f"best/final 正进展比例 "
    f"{100*(progress_best>0).mean():.0f}%/{100*(progress_final>0).mean():.0f}%",
    flush=True,
)
result_payload = dict(
    task=args.task, policy=args.policy, stage=args.stage, offset=args.offset, budget=BUDGET,
    source_dataset=str(Path(args.src).expanduser().resolve()),
    seed=args.seed, history=args.history, action_block=args.action_block,
    expected_planner_horizon=args.expected_planner_horizon,
    expected_receding_horizon=args.expected_receding_horizon,
    expected_planner_mode=args.expected_planner_mode,
    expected_plan_bank_continuation=(
        args.expected_plan_bank_continuation
        if args.expected_planner_mode.startswith("train_terminal_")
        else None
    ),
    init_mode=args.init_mode, prefix_steps_requested=args.prefix_steps,
    protocol=dict(
        formal_init_mode="teleport",
        requires_same_init_expert_continuation_gate=True,
        diagnostic_only=(
            args.init_mode != "teleport" or args.auto_offset or args.from_start
        ),
        note="Only teleport cases passing a separate expert continuation run belong in the formal pool.",
    ),
    dataset_split=args.dataset_split, split_seed=args.split_seed,
    expert_gate=expert_gate_audit,
    train_fraction=args.train_fraction, val_fraction=args.val_fraction,
    shuffle_goals=args.shuffle_goals,
    planner_seed_per_demo=args.planner_seed_per_demo,
    planner_seed_base=PLANNER_SEED_BASE,
    drop_steps=(args.drop_steps if args.policy == "drop" else None),
    drop_z=(args.drop_z if args.policy == "drop" else None),
    planner_config=planner_config, success_rate=sr, episodes=results,
)
result_path = os.path.join(
    args.out, f"eval_{args.policy}_{args.init_mode}_s{args.stage}_offset{args.offset}.json"
)
completion = commit_json_result(
    result_path,
    result_payload,
    schema_version="franka-stack-h5-eval-v1",
    record_count=len(results),
)
print(f"[COMMIT] {result_path} sha256={completion['result_sha256']}", flush=True)
if args.policy == "expert" and results and not all(r["success"] for r in results):
    failed = [r["demo"] for r in results if not r["success"]]
    print(f"[GATE FAILED] expert continuation failed for demos {failed}", flush=True)
    env.close()
    os._exit(2)
env.close()
# This evaluator is deliberately an isolated one-shot worker.  On the legacy
# Isaac Sim 5.1 stack ``app.close()`` can enter an unbounded, CPU-heavy plugin
# shutdown loop *after* the complete result JSON has already been committed.
# Let process teardown release Kit resources, matching the fail-closed expert
# path above, so a completed evaluation cannot silently occupy a GPU forever.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
