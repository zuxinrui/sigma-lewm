"""In-training closed-loop evaluation for LeWM multistep arms (PushT / OGBench-Cube).

Task-parameterized: `task` picks config/eval/<task>.yaml, from which env, dataset, goal
offset, callables and action_block are all derived — nothing else is task-specific.

Policy (2026-07-14 user directive): every epoch → h=5 eval; every 4 epochs → additionally h=10.
Runs INSIDE the training process on the live model (no ckpt reload, no separate job), using the
exact eval.py protocol: same episodes (fixed seed), same CEM solver config, same success metric.

Cost: the world/dataset/scalers build once lazily (~1 min); each h5 eval ≈ 2 CEM solves × 50
parallel envs ≈ a few minutes on A100. Failures NEVER crash training (see callback).
"""
from __future__ import annotations

from trim.paths import REPO_ROOT

import os
import json
import inspect
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

_REPO = REPO_ROOT


def _sample_eval_rows(
    episode_idx,
    valid_rows,
    *,
    num_eval: int,
    seed: int,
    unique_episodes: bool,
):
    """Sample evaluation rows, optionally at most one start per episode."""
    episode_idx = np.asarray(episode_idx)
    valid_rows = np.asarray(valid_rows, dtype=np.int64)
    g = np.random.default_rng(int(seed))
    if not unique_episodes:
        if len(valid_rows) < int(num_eval):
            raise RuntimeError(
                f"only {len(valid_rows)} valid rows; need {int(num_eval)}"
            )
        return np.sort(
            valid_rows[g.choice(len(valid_rows), size=int(num_eval), replace=False)]
        )

    valid_episode_ids = np.unique(episode_idx[valid_rows])
    if len(valid_episode_ids) < int(num_eval):
        raise RuntimeError(
            f"only {len(valid_episode_ids)} eligible episodes; need {int(num_eval)} unique episodes"
        )
    chosen_episodes = g.choice(
        valid_episode_ids, size=int(num_eval), replace=False
    )
    picks = []
    valid_row_episodes = episode_idx[valid_rows]
    for ep in chosen_episodes:
        episode_rows = valid_rows[valid_row_episodes == ep]
        picks.append(int(g.choice(episode_rows)))
    return np.sort(np.asarray(picks, dtype=np.int64))


def _standard_scaler_from_persisted(persisted, feature_dim: int):
    """Reconstruct sklearn scaler semantics from frozen training mean/std."""
    from sklearn import preprocessing

    mean = np.asarray(persisted["mean"], dtype=np.float64).reshape(-1)
    scale = np.asarray(persisted["std"], dtype=np.float64).reshape(-1)
    if mean.size != int(feature_dim) or scale.size != int(feature_dim):
        raise ValueError(
            f"persisted normalizer shape mismatch: mean={mean.shape}, "
            f"std={scale.shape}, expected={int(feature_dim)}"
        )
    scaler = preprocessing.StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = np.square(scale)
    scaler.n_features_in_ = int(mean.size)
    scaler.n_samples_seen_ = np.int64(1)
    return scaler


def _validate_callable_contract(world, callables, dataset_columns):
    """Fail before evaluation if dataset-to-environment setup cannot execute.

    ``stable_worldmodel.world._apply_callables`` intentionally skips a missing
    environment method.  That is convenient for generic interactive use but
    unsafe for a formal dataset-initialized rollout: silently skipping
    ``_set_state`` or ``_set_goal_state`` changes the evaluated episode.  Check
    every vectorized environment, every configured argument name, and every
    referenced dataset-derived key before any outcome is produced.
    """
    if not isinstance(callables, list) or not callables:
        raise RuntimeError("evaluation callable contract must be a non-empty list")

    vector_envs = getattr(getattr(world, "envs", None), "envs", None)
    if not vector_envs:
        raise RuntimeError("evaluation world exposes no vectorized environments")

    columns = {str(column) for column in dataset_columns}
    available_keys = set(columns)
    for column in columns:
        if column.startswith("goal"):
            continue
        available_keys.add("goal" if column == "pixels" else f"goal_{column}")

    for spec_index, spec in enumerate(callables):
        if not isinstance(spec, dict):
            raise RuntimeError(f"callable[{spec_index}] must be a mapping")
        method_name = spec.get("method")
        if not isinstance(method_name, str) or not method_name:
            raise RuntimeError(f"callable[{spec_index}] has no valid method name")
        args = spec.get("args", {})
        if not isinstance(args, dict):
            raise RuntimeError(f"callable[{spec_index}].args must be a mapping")

        prepared_names = {}
        for argument_name, source in args.items():
            if not isinstance(source, dict):
                raise RuntimeError(
                    f"callable[{spec_index}] argument {argument_name!r} "
                    "must have a mapping source"
                )
            if bool(source.get("in_dataset", True)):
                key = source.get("value")
                if key not in available_keys:
                    raise RuntimeError(
                        f"callable[{spec_index}] argument {argument_name!r} "
                        f"references unavailable dataset state {key!r}"
                    )
            prepared_names[str(argument_name)] = object()

        for env_index, wrapped_env in enumerate(vector_envs):
            env = getattr(wrapped_env, "unwrapped", wrapped_env)
            method = getattr(env, method_name, None)
            if not callable(method):
                raise RuntimeError(
                    f"callable[{spec_index}] method {method_name!r} is missing "
                    f"from environment {env_index}"
                )
            try:
                inspect.signature(method).bind(**prepared_names)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"callable[{spec_index}] arguments do not bind to "
                    f"{method_name!r}: {error}"
                ) from error


class InLoopEvaluator:
    """Lazy one-time setup of the eval world; evaluate(model, horizon) → success rate."""

    def __init__(self, num_eval: int = 50, seed: int = 42, task: str = "pusht",
                 exclude_episodes=None, unique_episodes: bool = False,
                 include_episodes=None, stats_episodes=None,
                 normalizer_stats=None, fixed_rows=None, dataset_path=None):
        self.num_eval = int(num_eval)
        self.seed = int(seed)
        self.unique_episodes = bool(unique_episodes)
        # 2026-08-21: episode-level 排除集。holdout bank 必须与 Bank S（seed 42）
        # 完全不共享 episode —— 只做行级排除不够，同一条 episode 的不同起点仍是同一段动力学。
        self.exclude_episodes = set(int(e) for e in (exclude_episodes or []))
        self.include_episodes = (
            None if include_episodes is None
            else set(int(e) for e in include_episodes)
        )
        self.stats_episodes = (
            None if stats_episodes is None
            else set(int(e) for e in stats_episodes)
        )
        self.normalizer_stats = normalizer_stats
        self.fixed_rows = (
            None if fixed_rows is None
            else tuple(int(row) for row in fixed_rows)
        )
        self.dataset_path = None if dataset_path is None else str(dataset_path)
        self.task = str(task)
        self._built = False

    # -- one-time heavy setup (world, dataset, scalers, episode selection) --
    def _build(self):
        import stable_pretraining as spt
        import stable_worldmodel as swm
        from sklearn import preprocessing
        from torchvision.transforms import v2 as transforms

        cfg_path = _REPO / f"config/eval/{self.task}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"no eval config for task={self.task!r}: {cfg_path}")
        cfg = OmegaConf.load(cfg_path)

        img_tf = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ])
        self.transform = {"pixels": img_tf, "goal": img_tf}

        # world sized for the largest horizon we run (h10 → budget 100)
        world_kw = {k: v for k, v in OmegaConf.to_container(cfg.world, resolve=True).items()
                    if k not in ("num_envs", "max_episode_steps")}
        self.world = swm.World(
            **world_kw,
            num_envs=self.num_eval,
            max_episode_steps=200,
            image_shape=(224, 224),
        )

        dataset_kwargs = {
            "keys_to_cache": list(cfg.dataset.keys_to_cache),
        }
        if self.dataset_path is None:
            dataset_kwargs.update({
                "name": cfg.eval.dataset_name,
                "cache_dir": Path(swm.data.utils.get_cache_dir()),
            })
        else:
            dataset_kwargs["path"] = self.dataset_path
        dataset = swm.data.HDF5Dataset(**dataset_kwargs)
        self.dataset = dataset

        col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        episode_idx = dataset.get_col_data(col_name)

        self.process = {}
        for col in cfg.dataset.keys_to_cache:
            if col == "pixels":
                continue
            scaler = preprocessing.StandardScaler()
            col_data = dataset.get_col_data(col)
            if self.normalizer_stats is not None:
                if col not in self.normalizer_stats:
                    raise KeyError(f"persisted normalizer is missing column {col!r}")
                scaler = _standard_scaler_from_persisted(
                    self.normalizer_stats[col], col_data.shape[1]
                )
            else:
                finite = ~np.isnan(col_data).any(axis=1)
                if self.stats_episodes is not None:
                    finite &= np.isin(episode_idx, list(self.stats_episodes))
                fit_data = col_data[finite]
                if len(fit_data) == 0:
                    raise RuntimeError(
                        f"no finite {col!r} rows remain for the requested stats episodes"
                    )
                scaler.fit(fit_data)
            self.process[col] = scaler
            if col != "action":
                self.process[f"goal_{col}"] = scaler

        # episode selection — identical logic + seed as eval.py → same episodes every epoch
        step_idx = dataset.get_col_data("step_idx")
        ep_ids = np.unique(episode_idx)
        lengths = np.array([np.max(step_idx[episode_idx == e]) + 1 for e in ep_ids])
        max_start = lengths - cfg.eval.goal_offset_steps - 1
        max_start_dict = {e: m for e, m in zip(ep_ids, max_start)}
        max_start_per_row = np.array([max_start_dict[e] for e in episode_idx])
        valid = np.nonzero(step_idx <= max_start_per_row)[0]
        if self.include_episodes is not None:
            keep = np.isin(episode_idx[valid], list(self.include_episodes))
            valid = valid[keep]
        if self.exclude_episodes:
            keep = ~np.isin(episode_idx[valid], list(self.exclude_episodes))
            valid = valid[keep]
            if len(valid) < self.num_eval:
                raise RuntimeError(
                    f"holdout bank seed={self.seed}: only {len(valid)} rows left after excluding "
                    f"{len(self.exclude_episodes)} episodes; need {self.num_eval}")
        if self.fixed_rows is not None:
            picks = np.sort(np.asarray(self.fixed_rows, dtype=np.int64))
            if len(picks) != self.num_eval:
                raise RuntimeError(
                    f"fixed bank contains {len(picks)} rows; expected {self.num_eval}"
                )
            invalid = np.setdiff1d(picks, valid)
            if len(invalid):
                raise RuntimeError(
                    f"fixed bank contains {len(invalid)} rows invalid under this protocol"
                )
        else:
            picks = _sample_eval_rows(
                episode_idx,
                valid,
                num_eval=self.num_eval,
                seed=self.seed,
                unique_episodes=self.unique_episodes,
            )
        rows = dataset.get_row_data(picks)
        self.eval_row_indices = picks.astype(np.int64).tolist()
        self.eval_episodes = rows[col_name].tolist()
        self.episode_set = set(int(e) for e in self.eval_episodes)
        self.eval_start_idx = rows["step_idx"].tolist()
        self.goal_offset = int(cfg.eval.goal_offset_steps)
        self.callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
        _validate_callable_contract(
            self.world, self.callables, dataset.column_names
        )
        self._cem_cfg = OmegaConf.load(_REPO / "config/eval/solver/cem.yaml")
        self.action_block = int(cfg.plan_config.action_block)
        self._built = True

    @torch.no_grad()
    def evaluate(
        self,
        model,
        horizon: int = 5,
        return_metrics: bool = False,
        *,
        receding_horizon: int | None = None,
        action_block: int | None = None,
        eval_budget: int | None = None,
    ):
        """Closed-loop success rate of the LIVE model at the given horizon.

        Protocol: FULL-COMMIT (receding_horizon = horizon), action_block 5,
        eval_budget = horizon * 10. Model must expose the JEPA planning API
        (encode/rollout/get_cost) — extra training heads are ignored.
        """
        import hydra as _hydra
        import stable_worldmodel as swm

        if not self._built:
            self._build()

        receding_horizon = (
            int(horizon) if receding_horizon is None else int(receding_horizon)
        )
        action_block = self.action_block if action_block is None else int(action_block)
        eval_budget = (
            int(horizon) * action_block * 2
            if eval_budget is None else int(eval_budget)
        )
        if action_block != self.action_block:
            raise RuntimeError(
                f"action_block={action_block} differs from task config {self.action_block}"
            )

        was_training = model.training
        model.eval()
        # eval.py sets this before planning
        model.interpolate_pos_encoding = True
        try:
            solver_cfg = OmegaConf.to_container(self._cem_cfg, resolve=False)
            solver_cfg = {k: v for k, v in solver_cfg.items() if k != "model"}
            solver_cfg["seed"] = self.seed
            solver = _hydra.utils.instantiate(solver_cfg, model=model)
            plan = swm.PlanConfig(
                # full-commit: rh = horizon (2026-07-15). rh=5 crushed h10 (rh=5 artifact,
                # see 2026-07-14_full_commit_h10_finding.md). h5 is unchanged (rh=5=horizon).
                horizon=int(horizon), receding_horizon=receding_horizon, history_len=1,
                action_block=action_block,
            )
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=plan, process=self.process, transform=self.transform,
            )
            self.world.set_policy(policy)
            metrics = self.world.evaluate(
                dataset=self.dataset,
                start_steps=list(self.eval_start_idx),
                goal_offset=self.goal_offset,
                eval_budget=eval_budget,
                episodes_idx=list(self.eval_episodes),
                callables=self.callables,
                video=None,
            )
            if return_metrics:
                return metrics
            # success key: mirror eval.py's reported metrics dict
            for k in ("success_rate", "success", "mean_success"):
                if k in metrics:
                    return float(metrics[k])
            # fallback: first scalar
            return float(next(iter(metrics.values())))
        finally:
            if was_training:
                model.train()


_EVALUATOR = {}

# 2026-08-21: in-loop holdout 种子。SHA256 确定性派生，**刻意区别于**
#   · Bank S 的 seed 42（历史 in-loop 曲线，用于挑峰值 ckpt）
#   · 预注册确认集的 5 个种子（若复用，等于每 epoch 偷看确认集）
HOLDOUT_SEEDS = (2046760368, 1349953925, 1741607489)   # sha256("lewm-inloop-holdout-v1:{0,1,2}")[:4]


def get_evaluator(num_eval: int = 50, seed: int = 42, task: str = "pusht",
                  exclude_episodes=None, unique_episodes: bool = False,
                  include_episodes=None, stats_episodes=None,
                  normalizer_stats=None, fixed_rows=None,
                  dataset_path=None) -> InLoopEvaluator:
    """Cached per (task, num_eval, seed, exclude) — 重的 world/dataset 构建只做一次。"""
    key = (
        str(task), int(num_eval), int(seed), bool(unique_episodes),
        tuple(sorted(int(e) for e in (exclude_episodes or []))),
        None if include_episodes is None else tuple(sorted(int(e) for e in include_episodes)),
        None if stats_episodes is None else tuple(sorted(int(e) for e in stats_episodes)),
        None if normalizer_stats is None else json.dumps(normalizer_stats, sort_keys=True),
        None if fixed_rows is None else tuple(int(row) for row in fixed_rows),
        None if dataset_path is None else str(dataset_path),
    )
    if key not in _EVALUATOR:
        _EVALUATOR[key] = InLoopEvaluator(num_eval=num_eval, seed=seed, task=task,
                                          exclude_episodes=exclude_episodes,
                                          unique_episodes=unique_episodes,
                                          include_episodes=include_episodes,
                                          stats_episodes=stats_episodes,
                                          normalizer_stats=normalizer_stats,
                                          fixed_rows=fixed_rows,
                                          dataset_path=dataset_path)
    return _EVALUATOR[key]


def get_dual_banks(num_eval: int = 50, task: str = "pusht", seen_seed: int = 42,
                   holdout_seeds=HOLDOUT_SEEDS, unique_episodes: bool = False,
                   include_episodes=None, stats_episodes=None,
                   normalizer_stats=None):
    """返回 (bank_S, [bank_H...])。

    Bank S = 历史 seed 42，保持与既有 in-loop 曲线可比 → 用于选模。
    Bank H = holdout_seeds 各一个 bank，**episode 级**与 S 及彼此互不重叠 → 只用于报告。
    互斥是累积的：每建一个 H bank，就把它用掉的 episode 加进排除集。
    """
    S = get_evaluator(
        num_eval=num_eval,
        seed=seen_seed,
        task=task,
        unique_episodes=unique_episodes,
        include_episodes=include_episodes,
        stats_episodes=stats_episodes,
        normalizer_stats=normalizer_stats,
    )
    if not S._built:
        S._build()
    used = set(S.episode_set)
    H = []
    for sd in holdout_seeds:
        h = get_evaluator(num_eval=num_eval, seed=int(sd), task=task,
                          exclude_episodes=used,
                          unique_episodes=unique_episodes,
                          include_episodes=include_episodes,
                          stats_episodes=stats_episodes,
                          normalizer_stats=normalizer_stats)
        if not h._built:
            h._build()
        assert not (h.episode_set & used), f"holdout seed={sd} 与已用 episode 重叠"
        used |= h.episode_set
        H.append(h)
    return S, H
