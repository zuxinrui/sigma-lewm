"""Goal-tolerance stress test: re-run the main-comparison episodes with termination DISABLED and
record the native goal error at every step, so success can be re-scored offline under any threshold.

Port of le-wm scripts/goal_tolerance_eval.py (058d998) to this repository, for Figure 5. It mirrors
slewm.evaluation.planning, the evaluator behind Table 1: the same config/eval/<task>.yaml, the same
episode draw and CEM seed, the same helpers and the same model loader (config.json + state dict),
except that (i) envs never freeze at the native threshold and (ii) per-step errors are saved. The T0
level of the rescored errors is therefore the Table 1 success rate up to GPU nondeterminism.
Success under threshold T is "error <= T at some step within the budget" (the native any-time rule).

Usage:
  python scripts/goal_tolerance_eval.py --task pusht --policy <abs path>/weights.pt --seed 0 --tag sigma_lewm_ts3072
Output: outputs/goal_tolerance/<task>_<tag>_s<seed>.npz
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, torch, hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn import preprocessing
import stable_worldmodel as swm
from stable_worldmodel.world.world import _extract_init_goal, _apply_callables
import gymnasium as gym


class NoTerminate(gym.Wrapper):
    """Innermost wrapper: hide the env's own termination so neither World nor the policy freezes the env.
    The native flag is kept on the unwrapped env as `_native_terminated` for the T0 cross-check."""
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.env.unwrapped._native_terminated = bool(term)
        return obs, r, False, trunc, info

    def reset(self, **kw):
        self.env.unwrapped._native_terminated = False
        return self.env.reset(**kw)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slewm.evaluation import benchmark as lewm_eval          # img_transform, get_dataset, get_episodes_length
from slewm.evaluation.planning import load_full_model       # Table 1's loader


def native_error(task, u):
    """Return the native goal-error components of one unwrapped env. Units documented per task."""
    if task == "pusht":       # window 512 px; state = [agent xy, block xy, block angle]
        cur = np.array(tuple(u.agent.position) + tuple(u.block.position) + (u.block.angle % (2 * np.pi),))
        g = np.asarray(u.goal_state)
        pos = float(np.linalg.norm(g[:4] - cur[:4]))
        ang = float(abs(g[4] - cur[4])); ang = min(ang, 2 * np.pi - ang)
        return [pos, ang]                                   # px, rad   (native: pos<20 and ang<pi/9)
    if task == "tworoom":     # 224 px image coordinates
        return [float(torch.norm(u.agent_position - u.target_position))]   # px (native: <16)
    if task == "reacher":     # per-joint |qpos - target| in rad
        d = np.abs(u.env.physics.data.qpos - u.env.task.target_qpos)
        return [float(d.max())]                             # rad (native: all < 0.05)
    if task == "cube":        # cube centre to mocap target, metres
        obj = u._data.joint("object_joint_0").qpos[:3]
        tar = u._data.mocap_pos[u._cube_target_mocap_ids[0]]
        return [float(np.linalg.norm(obj - tar))]           # m (native: <= 0.04)
    raise ValueError(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["pusht", "tworoom", "reacher", "cube"])
    ap.add_argument("--policy", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--cem", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "outputs/goal_tolerance"))
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        cfg = compose(config_name=a.task, overrides=[f"seed={a.seed}", f"policy={a.policy}",
                                                     f"solver.n_steps={a.cem}", *a.override])
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(int(cfg.world.get("height", 224)), int(cfg.world.get("width", 224))),
                      pre_wrappers=[NoTerminate])
    transform = {"pixels": lewm_eval.img_transform(cfg), "goal": lewm_eval.img_transform(cfg)}
    dataset = lewm_eval.get_dataset(cfg, cfg.eval.dataset_name)
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
    process = {}
    for c in cfg.dataset.keys_to_cache:
        if c == "pixels": continue
        p = preprocessing.StandardScaler(); d = dataset.get_col_data(c); d = d[~np.isnan(d).any(axis=1)]
        p.fit(d); process[c] = p
        if c != "action": process[f"goal_{c}"] = p
    model, _ = load_full_model(cfg.policy)
    model = model.to("cuda").eval(); model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(**cfg.plan_config),
                                         process=process, transform=transform)
    # --- identical episode sampling to eval.py
    ep_len = lewm_eval.get_episodes_length(dataset, ep_indices)
    max_start = ep_len - cfg.eval.goal_offset_steps - 1
    msd = {e: max_start[i] for i, e in enumerate(ep_indices)}
    max_start_per_row = np.array([msd[e] for e in dataset.get_col_data(col)])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= max_start_per_row)[0]
    g = np.random.default_rng(cfg.seed)
    rows = np.sort(valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)])
    eval_eps = dataset.get_row_data(rows)[col].tolist(); eval_start = dataset.get_row_data(rows)["step_idx"].tolist()
    world.set_policy(policy)
    # --- World._evaluate_from_dataset re-implemented with termination disabled + error logging
    n = len(eval_eps); goal_off, budget = cfg.eval.goal_offset_steps, cfg.eval.eval_budget
    init_state, goal_state, _ = _extract_init_goal(dataset, eval_eps, eval_start, goal_off)
    world.reset(seed=init_state.get("seed"))
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    merged = {**init_state, **goal_state}
    for i in range(n):
        _apply_callables(world.envs.envs[i].unwrapped, callables, {k: v[i] for k, v in merged.items()})
    if a.task == "reacher":   # also stop the dm_control task from ending sub-steps early once inside the tolerance
        for i in range(n):
            world.envs.envs[i].unwrapped.env.task.qpos_threshold = -1.0
    pref = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for k, v in src.items():
            if k in world.infos or k in goal_state:
                world.infos[k] = np.broadcast_to(v[:, None, ...], pref + v.shape[1:]).copy()
    snap = {k: world.infos[k].copy() for k in goal_state}
    err0 = np.array([native_error(a.task, world.envs.envs[i].unwrapped) for i in range(n)])
    errs, native = [], []
    def on_step(w):
        w.infos.update({k: v.copy() for k, v in snap.items()})
        native.append(np.array([w.envs.envs[i].unwrapped._native_terminated for i in range(n)]))
        errs.append(np.array([native_error(a.task, w.envs.envs[i].unwrapped) for i in range(n)]))
    t0 = time.time(); world._run(max_steps=budget, mode="wait", on_step=on_step); wall = time.time() - t0
    errs = np.stack(errs, 1); native = np.stack(native, 1)      # [n, T, k], [n, T]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.task}_{a.tag}_s{a.seed}.npz"
    np.savez(f, errors=errs, native_terminated=native, error_start=err0, rows=rows, episodes=np.array(eval_eps),
             start_steps=np.array(eval_start), cfg=OmegaConf.to_yaml(cfg), policy=cfg.policy, wall=wall)
    print(f"saved {f}  native any-time success {native.any(1).mean()*100:.1f}%  wall {wall:.0f}s")


if __name__ == "__main__":
    main()
