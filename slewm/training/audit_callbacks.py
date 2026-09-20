"""Audit callbacks support for the archived multistep recipes."""

from slewm.paths import REPO_ROOT
import os
import math
import hashlib
import json
import subprocess
import time
from pathlib import Path
import lightning as pl
import torch
import torch.distributed as dist
from slewm.protocols.formal_training_completion import verify_smoke_attestation, verify_stage_authorization
from slewm.training.utilities import git_tracked_tree_sha256
from slewm.training.distributed import _collective_device
from slewm.training.distributed import _dist_active
from slewm.training.data_split import _state_dict_sha256

class InLoopEvalCallback(pl.Callback):
    """训练内闭环评测（2026-08-21 双 bank 版）。

    每 epoch 跑**两组 h5**：
      · Bank S —— 历史的 seed 42，与既有 in-loop 曲线可比 → **用于挑峰值 checkpoint**
      · Bank H —— 3 个 SHA256 派生的 holdout 种子（各 50 episode），**episode 级**与 S
        及彼此互不重叠 → **从不用于选模，只用于诚实报告**

    这样训练结束后可以「用 S 选、用 H 报」，堵住 checkpoint-selection leakage
    （masterplan P0.4），而**不改动训练协议本身**。

    h10 默认关闭（h10_every=0）：它改了 horizon/预算/commit 三样却没把目标移远，
    不是"更长时程"，且每次要多花一倍 eval 时间。
    NEVER crashes training —— 评测失败只记录并跳过。
    """

    def __init__(
        self,
        h5_every=1,
        h10_every=0,
        num_eval=50,
        task="pusht",
        unique_episodes=False,
        include_episodes=None,
        stats_episodes=None,
        normalizer_stats=None,
    ):
        self.h5_every = int(h5_every)
        self.h10_every = int(h10_every)
        self.num_eval = int(num_eval)
        self.task = str(task)
        self.unique_episodes = bool(unique_episodes)
        self.include_episodes = include_episodes
        self.stats_episodes = stats_episodes
        self.normalizer_stats = normalizer_stats

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        ep = trainer.current_epoch + 1
        do_h5 = self.h5_every > 0 and ep % self.h5_every == 0
        do_h10 = self.h10_every > 0 and ep % self.h10_every == 0
        if not (do_h5 or do_h10):
            return
        try:
            from slewm.evaluation.inloop import get_dual_banks
            S, H = get_dual_banks(
                num_eval=self.num_eval,
                task=self.task,
                unique_episodes=self.unique_episodes,
                include_episodes=self.include_episodes,
                stats_episodes=self.stats_episodes,
                normalizer_stats=self.normalizer_stats,
            )
            model = pl_module.model
            msg = []
            if do_h5:
                s_val = S.evaluate(model, horizon=5)
                h_vals = [h.evaluate(model, horizon=5) for h in H]
                h_mean = sum(h_vals) / len(h_vals) if h_vals else float("nan")
                pl_module.log("eval/h5_seen", float(s_val), on_epoch=True, sync_dist=False)
                pl_module.log("eval/h5_holdout", float(h_mean), on_epoch=True, sync_dist=False)
                for k, v in enumerate(h_vals):
                    pl_module.log(f"eval/h5_holdout_{k}", float(v), on_epoch=True, sync_dist=False)
                msg.append(f"h5_seen={s_val:.1f}%")
                msg.append(f"h5_holdout={h_mean:.1f}% ({'/'.join(f'{v:.0f}' for v in h_vals)})")
            if do_h10:
                s10 = S.evaluate(model, horizon=10)
                pl_module.log("eval/h10_seen", float(s10), on_epoch=True, sync_dist=False)
                msg.append(f"h10_seen={s10:.1f}%")
            print("[INLOOP-EVAL] epoch=%d " % ep + "  ".join(msg), flush=True)
        except Exception as e:  # eval must never kill a training run
            import traceback
            print(f"[INLOOP-EVAL] WARN epoch={ep} failed: {e.__class__.__name__}: {e}", flush=True)
            traceback.print_exc()

class TransitionTriangleExecutionAuditCallback(pl.Callback):
    """Prove that a transition-triangle run used real DDP and trained live heads."""

    _ACTIVE_GROUPS = {
        "champion": {
            "backward_state": ("head_f.", "head_g."),
            "inverse_action": ("head_f_idm.", "head_g_idm."),
        },
        "concat_mse": {
            "backward_state": ("head_f.", "head_g."),
            "inverse_action": ("idm_mlp.",),
        },
        "delta_mse": {
            "backward_state": ("head_f.", "head_g."),
            "inverse_action": ("idm_mlp.",),
        },
        "dg_champion": {
            "backward_state": ("head_f.", "head_g."),
            "inverse_local": ("head_f_idm.", "head_g_idm."),
            "inverse_demo_goal": ("demo_goal_action_mlp.",),
        },
    }

    def __init__(
        self,
        *,
        enabled=False,
        method_name,
        study_family="delta_triangle",
        task="pusht",
        run_dir,
        expected_world_size=2,
        expected_max_epochs=None,
        training_seed=3072,
        per_device_batch_size=64,
        accumulate_grad_batches=1,
        require_clean_source=False,
        formal_execution=False,
        stage_authorization_path=None,
        smoke_attestation_path=None,
        expected_stage_authorization_sha256=None,
        expected_smoke_attestation_sha256=None,
        expected_slurm_job_id=None,
        expected_slurm_restart_count=0,
        expected_spt_cache_dir_sha256=None,
        expected_dataset_file_sha256=None,
        expected_method_lock_sha256=None,
    ):
        self.enabled = bool(enabled)
        self.method_name = str(method_name)
        self.study_family = str(study_family)
        if self.study_family not in {"delta_triangle", "demo_goal"}:
            raise ValueError(
                f"unsupported transition-triangle study family: {self.study_family!r}"
            )
        self.task = str(task)
        if self.task not in {"pusht", "tworoom"}:
            raise ValueError(f"unsupported transition-triangle task: {self.task!r}")
        self.run_dir = Path(run_dir)
        self.expected_world_size = int(expected_world_size)
        self.expected_max_epochs = (
            None if expected_max_epochs is None else int(expected_max_epochs)
        )
        self.training_seed = int(training_seed)
        self.per_device_batch_size = int(per_device_batch_size)
        self.accumulate_grad_batches = int(accumulate_grad_batches)
        self.require_clean_source = bool(require_clean_source)
        self.formal_execution = bool(formal_execution)
        self.stage_authorization_path = (
            None if stage_authorization_path is None else Path(stage_authorization_path)
        )
        self.smoke_attestation_path = (
            None if smoke_attestation_path is None else Path(smoke_attestation_path)
        )
        self.expected_stage_authorization_sha256 = expected_stage_authorization_sha256
        self.expected_smoke_attestation_sha256 = expected_smoke_attestation_sha256
        self.expected_slurm_job_id = (
            None if expected_slurm_job_id is None else str(expected_slurm_job_id)
        )
        self.expected_slurm_restart_count = int(expected_slurm_restart_count)
        self.expected_spt_cache_dir_sha256 = expected_spt_cache_dir_sha256
        self.expected_dataset_file_sha256 = expected_dataset_file_sha256
        self.expected_method_lock_sha256 = expected_method_lock_sha256
        if self.method_name not in self._ACTIVE_GROUPS:
            raise ValueError(
                f"transition-triangle execution audit does not support {self.method_name!r}"
            )
        self._initial_parameters = {}
        self._gradient_min_l2 = {}
        self._update_min_max_abs = {}
        self._ranks = []
        self._gradient_audited = False
        self._fit_started_monotonic = None
        self._completed_epochs = []
        self._source_at_fit_start = None
        self._initial_full_model_sha256 = None
        self._initial_shared_core_sha256 = None
        self._initial_champion_predecessor_sha256 = None

    def _group_parameters(self, pl_module, prefixes):
        return {
            name: parameter
            for name, parameter in pl_module.model.named_parameters()
            if name.startswith(prefixes)
        }

    def _active_groups(self):
        if self.study_family == "demo_goal" and self.method_name == "champion":
            return {
                "backward_state": ("head_f.", "head_g."),
                "inverse_local": ("head_f_idm.", "head_g_idm."),
            }
        return self._ACTIVE_GROUPS[self.method_name]

    @staticmethod
    def _distributed_min(value, device):
        dev = _collective_device(device)
        result = torch.tensor(float(value), device=dev, dtype=torch.float64)
        if _dist_active():
            dist.all_reduce(result, op=dist.ReduceOp.MIN)
        return float(result.item())

    @staticmethod
    def _distributed_max(value, device):
        dev = _collective_device(device)
        result = torch.tensor(float(value), device=dev, dtype=torch.float64)
        if _dist_active():
            dist.all_reduce(result, op=dist.ReduceOp.MAX)
        return float(result.item())

    @staticmethod
    def _source_provenance():
        repo = REPO_ROOT
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo,
            text=True,
        ).strip()
        return {
            "source_git_commit": commit,
            "source_tracked_tree_sha256": git_tracked_tree_sha256(repo),
            "source_worktree_clean": not bool(dirty),
        }

    @staticmethod
    def _assert_distributed_digest_equal(digest, device, label):
        raw = bytes.fromhex(digest)
        # 走与进程组后端匹配的设备（见 _collective_device 的说明）
        device = _collective_device(device)
        tensor = torch.tensor(list(raw), device=device, dtype=torch.uint8)
        world_size = int(dist.get_world_size()) if _dist_active() else 1
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        if _dist_active():
            dist.all_gather(gathered, tensor)
        else:
            gathered[0].copy_(tensor)
        observed = {bytes(item.cpu().tolist()).hex() for item in gathered}
        if len(observed) != 1:
            raise RuntimeError(
                f"transition-triangle ranks disagree on {label}: {sorted(observed)}"
            )

    def on_fit_start(self, trainer, pl_module):
        if not self.enabled:
            return
        actual_world_size = int(getattr(trainer, "world_size", 1))
        distributed_world_size = (
            int(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else 1
        )
        if actual_world_size != self.expected_world_size:
            raise RuntimeError(
                "transition-triangle execution world-size mismatch: "
                f"trainer={actual_world_size} expected={self.expected_world_size}"
            )
        if distributed_world_size != self.expected_world_size:
            raise RuntimeError(
                "transition-triangle execution process-group mismatch: "
                f"distributed={distributed_world_size} expected={self.expected_world_size}"
            )
        rank = int(dist.get_rank()) if _dist_active() else 0
        _cdev = _collective_device(pl_module.device)
        rank_tensor = torch.tensor([rank], device=_cdev, dtype=torch.long)
        gathered = [torch.empty_like(rank_tensor) for _ in range(distributed_world_size)]
        if _dist_active():
            dist.all_gather(gathered, rank_tensor)
        else:
            gathered[0].copy_(rank_tensor)
        self._ranks = sorted(int(item.item()) for item in gathered)
        if self._ranks != list(range(self.expected_world_size)):
            raise RuntimeError(
                f"transition-triangle execution has wrong distributed ranks: {self._ranks}"
            )
        if int(getattr(trainer, "current_epoch", 0)) != 0 or int(
            getattr(trainer, "global_step", 0)
        ) != 0:
            raise RuntimeError(
                "transition-triangle formal execution must start at epoch=0, step=0"
            )
        actual_max_epochs = int(getattr(trainer, "max_epochs", -1))
        if (
            self.expected_max_epochs is not None
            and actual_max_epochs != self.expected_max_epochs
        ):
            raise RuntimeError(
                "transition-triangle execution max-epoch mismatch: "
                f"trainer={actual_max_epochs} expected={self.expected_max_epochs}"
            )
        self._initial_full_model_sha256 = _state_dict_sha256(pl_module.model)
        self._initial_shared_core_sha256 = _state_dict_sha256(
            pl_module.model,
            excluded_prefixes=(
                "head_f_idm.",
                "head_g_idm.",
                "idm_mlp.",
                "demo_goal_action_mlp.",
            ),
        )
        if self.study_family == "demo_goal":
            # The locked candidate is the complete champion plus one training-only
            # head.  Excluding only that new head proves that the deployment core,
            # backward heads, and the existing DCL-IDM heads all start identically.
            self._initial_champion_predecessor_sha256 = _state_dict_sha256(
                pl_module.model,
                excluded_prefixes=("demo_goal_action_mlp.",),
            )
        self._assert_distributed_digest_equal(
            self._initial_full_model_sha256, pl_module.device, "full initialization"
        )
        self._assert_distributed_digest_equal(
            self._initial_shared_core_sha256,
            pl_module.device,
            "shared-core initialization",
        )
        if self._initial_champion_predecessor_sha256 is not None:
            self._assert_distributed_digest_equal(
                self._initial_champion_predecessor_sha256,
                pl_module.device,
                "champion-predecessor initialization",
            )
        if self.require_clean_source:
            self._source_at_fit_start = self._source_provenance()
            if not self._source_at_fit_start["source_worktree_clean"]:
                raise RuntimeError("transition-triangle source is dirty at fit start")
        if self.formal_execution:
            if self.stage_authorization_path is None or self.smoke_attestation_path is None:
                raise RuntimeError("formal execution lacks stage/smoke attestation paths")
            if self.study_family == "demo_goal":
                from slewm.protocols.demo_goal_training_artifacts import (
                    verify_demo_goal_smoke_attestation,
                    verify_demo_goal_stage_authorization,
                )

                authorization = verify_demo_goal_stage_authorization(
                    self.stage_authorization_path,
                    expected_commit=self._source_at_fit_start["source_git_commit"],
                    expected_tree=self._source_at_fit_start[
                        "source_tracked_tree_sha256"
                    ],
                    expected_arm=self.method_name,
                    expected_task=self.task,
                    expected_training_seed=self.training_seed,
                    expected_method_lock_sha256=self.expected_method_lock_sha256,
                    require_registry=True,
                    expected_registry_status="AUTHORIZED",
                )
                smoke = verify_demo_goal_smoke_attestation(
                    self.smoke_attestation_path,
                    expected_commit=self._source_at_fit_start["source_git_commit"],
                    expected_tree=self._source_at_fit_start[
                        "source_tracked_tree_sha256"
                    ],
                    expected_task=self.task,
                    expected_method_lock_sha256=self.expected_method_lock_sha256,
                )
                smoke_hash = smoke.get("smoke_attestation_sha256")
            else:
                authorization = verify_stage_authorization(
                    self.stage_authorization_path,
                    expected_commit=self._source_at_fit_start["source_git_commit"],
                    expected_arm=self.method_name,
                    expected_task=self.task,
                    expected_training_seed=self.training_seed,
                    require_registry=True,
                    expected_registry_status="AUTHORIZED",
                )
                smoke = verify_smoke_attestation(
                    self.smoke_attestation_path,
                    expected_commit=self._source_at_fit_start["source_git_commit"],
                    expected_tree=self._source_at_fit_start[
                        "source_tracked_tree_sha256"
                    ],
                    expected_task=self.task,
                )
                smoke_hash = smoke.get("attestation_sha256")
            observed_job_id = os.environ.get("SLURM_JOB_ID")
            observed_restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
            spt_cache_dir = os.environ.get("SPT_CACHE_DIR", "")
            spt_cache_sha256 = hashlib.sha256(spt_cache_dir.encode()).hexdigest()
            binding_mismatches = {}
            expected_bindings = {
                "authorization_sha256": self.expected_stage_authorization_sha256,
                "smoke_attestation_sha256": self.expected_smoke_attestation_sha256,
                "slurm_job_id": self.expected_slurm_job_id,
                "slurm_restart_count": self.expected_slurm_restart_count,
                "spt_cache_dir_sha256": self.expected_spt_cache_dir_sha256,
                "dataset_file_sha256": self.expected_dataset_file_sha256,
            }
            observed_bindings = {
                "authorization_sha256": authorization.get("authorization_sha256"),
                "smoke_attestation_sha256": smoke_hash,
                "slurm_job_id": observed_job_id,
                "slurm_restart_count": observed_restart_count,
                "spt_cache_dir_sha256": spt_cache_sha256,
                "dataset_file_sha256": authorization.get(
                    "dataset_file_sha256"
                ),
            }
            if self.study_family == "demo_goal":
                expected_bindings["method_lock_sha256"] = (
                    self.expected_method_lock_sha256
                )
                observed_bindings["method_lock_sha256"] = authorization.get(
                    "method_lock_sha256"
                )
                if self.training_seed == 3072:
                    smoke_arm = smoke.get("arms", {}).get(
                        self.method_name, {}
                    )
                    initialization_bindings = {
                        "initial_shared_core_sha256": (
                            self._initial_shared_core_sha256,
                            smoke_arm.get("initial_shared_core_sha256"),
                        ),
                        "initial_full_model_sha256": (
                            self._initial_full_model_sha256,
                            smoke_arm.get("initial_full_model_sha256"),
                        ),
                        "initial_champion_predecessor_sha256": (
                            self._initial_champion_predecessor_sha256,
                            smoke_arm.get(
                                "initial_champion_predecessor_sha256"
                            ),
                        ),
                    }
                    for key, (observed, expected) in (
                        initialization_bindings.items()
                    ):
                        if observed != expected:
                            binding_mismatches[
                                f"smoke_initialization.{key}"
                            ] = (observed, expected)
                if self.method_name == "dg_champion":
                    prior_predecessor = authorization.get(
                        "prior_initial_champion_predecessor_sha256"
                    )
                    if (
                        self._initial_champion_predecessor_sha256
                        != prior_predecessor
                    ):
                        binding_mismatches[
                            "candidate_to_champion_predecessor"
                        ] = (
                            self._initial_champion_predecessor_sha256,
                            prior_predecessor,
                        )
            for key, expected in expected_bindings.items():
                if expected is None or observed_bindings[key] != expected:
                    binding_mismatches[key] = (observed_bindings[key], expected)
            if authorization.get("smoke_attestation_sha256") != smoke_hash:
                binding_mismatches["authorization_to_smoke"] = (
                    authorization.get("smoke_attestation_sha256"),
                    smoke_hash,
                )
            authorization_bindings = {
                "authorization.slurm_job_id": (
                    authorization.get("slurm_job_id"), observed_job_id
                ),
                "authorization.source_tracked_tree_sha256": (
                    authorization.get("source_tracked_tree_sha256"),
                    self._source_at_fit_start["source_tracked_tree_sha256"],
                ),
                "authorization.dataset_fingerprint_sha256": (
                    authorization.get("dataset_fingerprint_sha256"),
                    smoke.get("dataset_fingerprint_sha256"),
                ),
                "authorization.dataset_file_sha256": (
                    authorization.get("dataset_file_sha256"),
                    smoke.get("dataset_file_sha256"),
                ),
                "authorization.train_sha256": (
                    authorization.get("train_sha256"),
                    smoke.get("train_sha256"),
                ),
                "authorization.validation_sha256": (
                    authorization.get("validation_sha256"),
                    smoke.get("validation_sha256"),
                ),
            }
            if self.study_family == "demo_goal" and self.task == "tworoom":
                authorization_bindings.update({
                    "authorization.tworoom_replay_attestation_sha256": (
                        authorization.get(
                            "tworoom_replay_attestation_sha256"
                        ),
                        smoke.get("tworoom_replay_attestation_sha256"),
                    ),
                    "authorization.tworoom_replay_attestation_file_sha256": (
                        authorization.get(
                            "tworoom_replay_attestation_file_sha256"
                        ),
                        smoke.get(
                            "tworoom_replay_attestation_file_sha256"
                        ),
                    ),
                })
            for key, (observed, expected) in authorization_bindings.items():
                if observed != expected:
                    binding_mismatches[key] = (observed, expected)
            if binding_mismatches:
                raise RuntimeError(
                    f"formal fit-start stage binding mismatch: {binding_mismatches}"
                )
        self._fit_started_monotonic = time.monotonic()
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)

        for group, prefixes in self._active_groups().items():
            parameters = self._group_parameters(pl_module, prefixes)
            if not parameters:
                raise RuntimeError(
                    f"transition-triangle execution lacks active parameter group {group}"
                )
            for name, parameter in parameters.items():
                self._initial_parameters[name] = parameter.detach().cpu().clone()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not self.enabled or self._gradient_audited:
            return
        local_ok = True
        local_details = []
        for group, prefixes in self._active_groups().items():
            parameters = self._group_parameters(pl_module, prefixes)
            missing = [
                name for name, parameter in parameters.items() if parameter.grad is None
            ]
            nonfinite = [
                name
                for name, parameter in parameters.items()
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            squared_l2 = sum(
                float(parameter.grad.detach().float().square().sum().item())
                for parameter in parameters.values()
                if parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
            )
            l2 = math.sqrt(squared_l2)
            min_l2 = self._distributed_min(l2, pl_module.device)
            self._gradient_min_l2[group] = min_l2
            group_ok = bool(parameters) and not missing and not nonfinite and min_l2 > 0.0
            local_ok = local_ok and group_ok
            if not group_ok:
                local_details.append(
                    f"{group}: missing={missing[:8]} nonfinite={nonfinite[:8]} "
                    f"min_l2={min_l2}"
                )
        globally_ok = self._distributed_min(float(local_ok), pl_module.device) == 1.0
        if not globally_ok:
            raise RuntimeError(
                "transition-triangle active-gradient audit failed: "
                + ("; ".join(local_details) or "another rank failed")
            )
        self._gradient_audited = True

    def on_train_epoch_end(self, trainer, pl_module):
        if self.enabled:
            self._completed_epochs.append(int(trainer.current_epoch) + 1)

    def on_fit_end(self, trainer, pl_module):
        if not self.enabled:
            return
        if not self._gradient_audited:
            raise RuntimeError(
                "transition-triangle execution ended without a gradient audit"
            )
        if self.expected_max_epochs is not None:
            expected_epochs = list(range(1, self.expected_max_epochs + 1))
            if self._completed_epochs != expected_epochs:
                raise RuntimeError(
                    "transition-triangle execution did not complete every epoch: "
                    f"actual={self._completed_epochs} expected={expected_epochs}"
                )
        source_at_fit_end = None
        if self.require_clean_source:
            source_at_fit_end = self._source_provenance()
            if source_at_fit_end != self._source_at_fit_start:
                raise RuntimeError(
                    "transition-triangle source changed between fit start and fit end"
                )
        local_ok = True
        for group, prefixes in self._active_groups().items():
            parameters = self._group_parameters(pl_module, prefixes)
            max_abs = 0.0
            for name, parameter in parameters.items():
                initial = self._initial_parameters.get(name)
                if initial is None:
                    local_ok = False
                    continue
                delta = (parameter.detach().cpu() - initial).abs()
                max_abs = max(max_abs, float(delta.max().item()))
            min_max_abs = self._distributed_min(max_abs, pl_module.device)
            self._update_min_max_abs[group] = min_max_abs
            local_ok = local_ok and min_max_abs > 0.0
        globally_ok = self._distributed_min(float(local_ok), pl_module.device) == 1.0
        if not globally_ok:
            raise RuntimeError(
                "transition-triangle active-head update audit failed: "
                f"{self._update_min_max_abs}"
            )

        actual_world_size = int(trainer.world_size)
        local_wall_time = time.monotonic() - self._fit_started_monotonic
        fit_wall_time = self._distributed_max(local_wall_time, pl_module.device)
        local_peak_memory = (
            int(torch.cuda.max_memory_allocated(pl_module.device))
            if pl_module.device.type == "cuda"
            else 0
        )
        peak_memory = int(self._distributed_max(local_peak_memory, pl_module.device))
        optimizer_steps = int(trainer.global_step)
        min_steps = int(self._distributed_min(optimizer_steps, pl_module.device))
        max_steps = int(self._distributed_max(optimizer_steps, pl_module.device))
        if min_steps != max_steps:
            raise RuntimeError(
                f"transition-triangle ranks disagree on optimizer steps: {min_steps}, {max_steps}"
            )

        if bool(getattr(trainer, "is_global_zero", False)):
            optimizer_contracts = []
            for optimizer in getattr(trainer, "optimizers", []):
                defaults = getattr(optimizer, "defaults", {})
                optimizer_contracts.append({
                    "class_name": optimizer.__class__.__name__,
                    "initial_lr": float(defaults.get("lr", float("nan"))),
                    "weight_decay": float(defaults.get("weight_decay", float("nan"))),
                })
            scheduler_contracts = []
            for scheduler_config in getattr(trainer, "lr_scheduler_configs", []):
                scheduler = scheduler_config.scheduler
                scheduler_contracts.append({
                    "class_name": scheduler.__class__.__name__,
                    "interval": str(scheduler_config.interval),
                    "last_epoch": int(getattr(scheduler, "last_epoch", -1)),
                })
            receipt = {
                "schema_version": 2,
                "method_name": self.method_name,
                "study_family": self.study_family,
                "task": self.task,
                "world_size": actual_world_size,
                "distributed_ranks": self._ranks,
                "strategy_class": trainer.strategy.__class__.__name__,
                "per_device_batch_size": self.per_device_batch_size,
                "accumulate_grad_batches": self.accumulate_grad_batches,
                "effective_global_batch_size": (
                    actual_world_size
                    * self.per_device_batch_size
                    * self.accumulate_grad_batches
                ),
                "optimizer_steps": optimizer_steps,
                "completed_epochs": self._completed_epochs,
                "max_epochs": int(getattr(trainer, "max_epochs", -1)),
                "num_training_batches": int(
                    getattr(trainer, "num_training_batches", -1)
                ),
                "estimated_stepping_batches": int(
                    getattr(trainer, "estimated_stepping_batches", -1)
                ),
                "optimizer_contracts": optimizer_contracts,
                "scheduler_contracts": scheduler_contracts,
                "initial_full_model_sha256": self._initial_full_model_sha256,
                "initial_shared_core_sha256": self._initial_shared_core_sha256,
                "initial_champion_predecessor_sha256": (
                    self._initial_champion_predecessor_sha256
                ),
                "source_at_fit_start": self._source_at_fit_start,
                "source_at_fit_end": source_at_fit_end,
                "formal_execution": self.formal_execution,
                "formal_stage_authorization_sha256": (
                    self.expected_stage_authorization_sha256
                ),
                "formal_smoke_attestation_sha256": (
                    self.expected_smoke_attestation_sha256
                ),
                "formal_slurm_job_id": self.expected_slurm_job_id,
                "formal_slurm_restart_count": self.expected_slurm_restart_count,
                "formal_spt_cache_dir_sha256": self.expected_spt_cache_dir_sha256,
                "formal_method_lock_sha256": self.expected_method_lock_sha256,
                "fit_wall_time_seconds": fit_wall_time,
                "optimizer_steps_per_second": optimizer_steps / fit_wall_time,
                "peak_cuda_memory_allocated_bytes": peak_memory,
                "active_gradient_audit_passed": True,
                "gradient_min_l2_by_group": self._gradient_min_l2,
                "active_update_audit_passed": True,
                "update_min_max_abs_by_group": self._update_min_max_abs,
            }
            self.run_dir.mkdir(parents=True, exist_ok=True)
            path = self.run_dir / "transition_triangle_execution_receipt.json"
            tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
            tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, path)

class IBRRoutingAuditCallback(pl.Callback):
    """Fail closed if an inactive IBR auxiliary head receives or applies updates."""

    def __init__(self, enabled=False, switch_epoch=1):
        self.enabled = bool(enabled)
        self.switch_epoch = int(switch_epoch)
        self._inactive_snapshot = {}
        self._active_gradient_epochs = set()

    def _phase(self, trainer):
        return 0 if int(trainer.current_epoch) < self.switch_epoch else 1

    def _inactive_prefixes(self, trainer):
        if self._phase(trainer) == 0:
            return ("head_f.", "head_g.")
        return ("head_f_idm.", "head_g_idm.")

    def _inactive_parameters(self, trainer, pl_module):
        prefixes = self._inactive_prefixes(trainer)
        return {
            name: parameter
            for name, parameter in pl_module.model.named_parameters()
            if name.startswith(prefixes)
        }

    def _active_prefixes(self, trainer):
        if self._phase(trainer) == 0:
            return ("head_f_idm.", "head_g_idm.")
        return ("head_f.", "head_g.")

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.enabled:
            return
        epoch = torch.tensor(
            int(trainer.current_epoch),
            device=_collective_device(pl_module.device), dtype=torch.long
        )
        if _dist_active():
            epoch_min = epoch.clone()
            epoch_max = epoch.clone()
            dist.all_reduce(epoch_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(epoch_max, op=dist.ReduceOp.MAX)
            if int(epoch_min) != int(epoch_max):
                raise RuntimeError("IBR-v1 ranks disagree on current_epoch")
        inactive = self._inactive_parameters(trainer, pl_module)
        if not inactive:
            raise RuntimeError("IBR-v1 inactive auxiliary head is missing")
        self._inactive_snapshot = {
            name: parameter.detach().cpu().clone()
            for name, parameter in inactive.items()
        }
        if int(getattr(trainer, "global_rank", 0)) == 0:
            phase_name = "inverse-idm" if self._phase(trainer) == 0 else "backward-state"
            print(
                f"[IBR-V1] epoch={int(trainer.current_epoch)} phase={phase_name} "
                f"inactive_parameters={len(inactive)}",
                flush=True,
            )

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not self.enabled:
            return
        leaked = [
            name
            for name, parameter in self._inactive_parameters(trainer, pl_module).items()
            if parameter.grad is not None
        ]
        if leaked:
            raise RuntimeError(
                "IBR-v1 inactive parameters received gradients: " + ", ".join(leaked[:8])
            )
        epoch = int(trainer.current_epoch)
        if epoch not in self._active_gradient_epochs:
            active = {
                name: parameter
                for name, parameter in pl_module.model.named_parameters()
                if name.startswith(self._active_prefixes(trainer))
            }
            missing = [
                name for name, parameter in active.items() if parameter.grad is None
            ]
            nonfinite = [
                name
                for name, parameter in active.items()
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            if not active or missing or nonfinite:
                raise RuntimeError(
                    "IBR-v1 active-gradient audit failed: "
                    f"missing={missing[:8]} nonfinite={nonfinite[:8]}"
                )
            self._active_gradient_epochs.add(epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.enabled:
            return
        current = self._inactive_parameters(trainer, pl_module)
        changed = [
            name
            for name, parameter in current.items()
            if name not in self._inactive_snapshot
            or not torch.equal(parameter.detach().cpu(), self._inactive_snapshot[name])
        ]
        if changed:
            raise RuntimeError(
                "IBR-v1 inactive parameters changed: " + ", ".join(changed[:8])
            )
        if int(trainer.current_epoch) not in self._active_gradient_epochs:
            raise RuntimeError("IBR-v1 epoch ended without an audited active-head gradient")
