"""Train-only behavior-plan proposals for constrained LeWM evaluation.

The Franka Stack controller uses 25 raw actions per H5 plan.  Independent
Gaussian CEM can satisfy every marginal bound while still proposing a joint
175-D sequence that never occurred in the demonstrations.  This module offers
the complementary, deliberately conservative experiment: score only complete
terminal plans that were observed in the episode-disjoint training split.

It does not read evaluation data and it does not infer success from file names.
Every bank member is reconstructed from ``goal_idx - total_steps : goal_idx``
and must have a single close-to-open transition with enough physical open
commands for the strict actuator threshold.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from trim.planning.projected_cem import ProjectedCEMSolver, StandardizedActionProjection


PLAN_BANK_SELECTION_MODES = (
    "behavior_central",  # deliberately goal-agnostic negative control
    "behavior_retrieval",
    "wm_rerank",
)
PLAN_BANK_CONTINUATIONS = ("abstain", "open_hold", "repeat_terminal")


@dataclass(frozen=True)
class FrankaTerminalPlanBank:
    """Canonical raw/standardized terminal plans and immutable provenance."""

    standardized_plans: np.ndarray
    raw_plans: np.ndarray
    source_demo_ids: np.ndarray
    source_episode_indices: np.ndarray
    start_rows: np.ndarray
    goal_rows: np.ndarray
    switch_steps: np.ndarray
    start_pixels: np.ndarray
    goal_pixels: np.ndarray
    central_index: int
    open_hold_standardized: np.ndarray
    provenance: dict[str, Any]

    @property
    def size(self) -> int:
        return int(self.raw_plans.shape[0])


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        canonical = np.ascontiguousarray(array)
        digest.update(str(canonical.dtype).encode("ascii"))
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def load_franka_terminal_plan_bank(
    dataset_h5: str | Path,
    scaler: Any,
    *,
    horizon: int,
    action_block: int,
    minimum_open_commands: int = 6,
    terminal_switch_steps: tuple[int, ...],
    raw_low: np.ndarray,
    raw_high: np.ndarray,
    continuous_plan_tv_max: float,
) -> FrankaTerminalPlanBank:
    """Load strict-release H5 plans from an aligned *training* HDF5 file.

    The returned standardized layout is exactly what LeWM consumes:
    ``[bank, horizon, action_block * 7]``.  A central real plan (the bank member
    nearest the standardized pointwise mean) supplies a goal-agnostic behavior
    baseline.  The same bank can instead be reranked by the frozen world model.
    """

    path = Path(dataset_h5).expanduser().resolve()
    total_steps = int(horizon) * int(action_block)
    if horizon < 1 or action_block < 1:
        raise ValueError("plan-bank horizon and action_block must be positive")
    if minimum_open_commands < 1 or minimum_open_commands > total_steps:
        raise ValueError("minimum_open_commands is outside the plan length")
    if (
        not terminal_switch_steps
        or tuple(sorted(set(terminal_switch_steps))) != terminal_switch_steps
        or terminal_switch_steps[0] < 0
        or terminal_switch_steps[-1] >= total_steps
    ):
        raise ValueError("Franka plan-bank terminal switch support is invalid")
    if getattr(scaler, "n_features_in_", None) != 7:
        raise ValueError("Franka plan bank requires a fitted seven-dimensional scaler")
    raw_low_array = np.asarray(raw_low, dtype=np.float64)
    raw_high_array = np.asarray(raw_high, dtype=np.float64)
    if (
        raw_low_array.shape != (7,)
        or raw_high_array.shape != (7,)
        or not np.isfinite(raw_low_array).all()
        or not np.isfinite(raw_high_array).all()
        or np.any(raw_low_array > raw_high_array)
    ):
        raise ValueError("Franka plan-bank raw support bounds are invalid")
    if not np.isfinite(continuous_plan_tv_max) or continuous_plan_tv_max <= 0:
        raise ValueError("Franka plan-bank TV support maximum must be positive")

    required = {
        "action",
        "ep_has_success",
        "ep_len",
        "ep_offset",
        "ep_source_demo_id",
        "ep_split_id",
        "goal_idx",
    }
    raw_plans: list[np.ndarray] = []
    source_demo_ids: list[int] = []
    source_episode_indices: list[int] = []
    start_rows: list[int] = []
    goal_rows: list[int] = []
    switch_steps: list[int] = []
    start_pixels: list[np.ndarray] = []
    goal_pixels: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(f"Franka plan-bank metadata missing: {sorted(missing)}")
        if handle.attrs.get("selected_split") != "train":
            raise ValueError("Franka plan bank must come from selected_split='train'")
        if handle.attrs.get("action_alignment") != "obs[t]=s_t, action[t]=a_t":
            raise ValueError("Franka plan bank has the wrong action alignment")

        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        goals = np.asarray(handle["goal_idx"][:], dtype=np.int64)
        successes = np.asarray(handle["ep_has_success"][:], dtype=bool)
        demo_ids = np.asarray(handle["ep_source_demo_id"][:], dtype=np.int64)
        split_ids = np.asarray(handle["ep_split_id"][:], dtype=np.int64)
        expected_shape = offsets.shape
        if not all(
            array.shape == expected_shape
            for array in (lengths, goals, successes, demo_ids, split_ids)
        ):
            raise ValueError("Franka plan-bank episode metadata shapes disagree")
        if np.any(split_ids != 0):
            raise ValueError("Franka plan bank contains a non-train episode")

        for episode_index, (
            offset,
            stored_length,
            goal,
            has_success,
            source_demo,
        ) in enumerate(zip(offsets, lengths, goals, successes, demo_ids)):
            if not has_success:
                continue
            offset = int(offset)
            goal = int(goal)
            valid_end = offset + int(stored_length) - 1
            start = goal - total_steps
            if start < offset or goal > valid_end:
                raise ValueError(
                    "successful train episode cannot provide its exact terminal plan"
                )
            raw = np.asarray(handle["action"][start:goal], dtype=np.float32)
            if raw.shape != (total_steps, 7) or not np.isfinite(raw).all():
                raise ValueError("Franka terminal bank member is malformed")
            gripper = raw[:, 6]
            if not np.all(np.isin(gripper, (-1.0, 1.0))):
                raise ValueError("Franka terminal bank gripper is not exact binary ±1")
            open_indices = np.flatnonzero(gripper == 1.0)
            if len(open_indices) == 0:
                raise ValueError("Franka terminal bank member never opens")
            split = int(open_indices[0])
            if not (
                np.all(gripper[:split] == -1.0)
                and np.all(gripper[split:] == 1.0)
            ):
                raise ValueError("Franka terminal bank member has multiple gripper phases")
            if total_steps - split < minimum_open_commands:
                continue
            if split not in terminal_switch_steps:
                raise ValueError(
                    f"strict terminal split {split} is outside train-fitted support"
                )
            raw_plans.append(raw)
            source_demo_ids.append(int(source_demo))
            source_episode_indices.append(int(episode_index))
            start_rows.append(start)
            goal_rows.append(goal)
            switch_steps.append(split)
            start_pixels.append(np.asarray(handle["pixels"][start], dtype=np.uint8))
            goal_pixels.append(np.asarray(handle["pixels"][goal], dtype=np.uint8))

        split_seed = int(handle.attrs.get("split_seed", -1))
        train_fraction = float(handle.attrs.get("train_fraction", np.nan))
        val_fraction = float(handle.attrs.get("val_fraction", np.nan))
        source_dataset = str(handle.attrs.get("source", ""))

    if not raw_plans:
        raise ValueError("Franka train split has no strict-release terminal plans")
    raw_array = np.ascontiguousarray(np.stack(raw_plans), dtype=np.float32)
    standardized_raw = np.asarray(scaler.transform(raw_array), dtype=np.float32)
    raw_support_mask = np.logical_and(
        raw_array >= raw_low_array[None, None, :],
        raw_array <= raw_high_array[None, None, :],
    ).all(axis=(1, 2))
    plan_tv = np.abs(np.diff(standardized_raw[:, :, :6], axis=1)).sum(
        axis=(1, 2), dtype=np.float64
    )
    tv_support_mask = plan_tv <= float(continuous_plan_tv_max)
    support_mask = raw_support_mask & tv_support_mask
    observed_candidate_count = int(len(raw_array))
    excluded_by_raw_bounds = int((~raw_support_mask).sum())
    excluded_by_plan_tv = int((raw_support_mask & ~tv_support_mask).sum())
    if not np.any(support_mask):
        raise ValueError("no strict-release terminal plan lies inside planner support")
    raw_array = np.ascontiguousarray(raw_array[support_mask], dtype=np.float32)
    standardized_raw = np.ascontiguousarray(
        standardized_raw[support_mask], dtype=np.float32
    )
    source_demo_ids = list(np.asarray(source_demo_ids)[support_mask])
    source_episode_indices = list(np.asarray(source_episode_indices)[support_mask])
    start_rows = list(np.asarray(start_rows)[support_mask])
    goal_rows = list(np.asarray(goal_rows)[support_mask])
    switch_steps = list(np.asarray(switch_steps)[support_mask])
    start_pixels = list(np.asarray(start_pixels)[support_mask])
    goal_pixels = list(np.asarray(goal_pixels)[support_mask])
    plan_tv = np.ascontiguousarray(plan_tv[support_mask], dtype=np.float64)
    source_sort = np.asarray(source_demo_ids, dtype=np.int64)
    starts_sort = np.asarray(start_rows, dtype=np.int64)
    order = np.lexsort((starts_sort, source_sort))
    raw_array = np.ascontiguousarray(raw_array[order])
    standardized_raw = np.ascontiguousarray(standardized_raw[order])
    source_demo_ids = list(np.asarray(source_demo_ids)[order])
    source_episode_indices = list(np.asarray(source_episode_indices)[order])
    start_rows = list(np.asarray(start_rows)[order])
    goal_rows = list(np.asarray(goal_rows)[order])
    switch_steps = list(np.asarray(switch_steps)[order])
    start_pixels = list(np.asarray(start_pixels)[order])
    goal_pixels = list(np.asarray(goal_pixels)[order])
    plan_tv = np.ascontiguousarray(plan_tv[order], dtype=np.float64)
    standardized = np.ascontiguousarray(
        standardized_raw.reshape(len(raw_array), horizon, action_block * 7),
        dtype=np.float32,
    )
    source_ids = np.asarray(source_demo_ids, dtype=np.int32)
    episode_ids = np.asarray(source_episode_indices, dtype=np.int32)
    starts = np.asarray(start_rows, dtype=np.int64)
    goals_array = np.asarray(goal_rows, dtype=np.int64)
    switches = np.asarray(switch_steps, dtype=np.int16)
    start_pixel_array = np.ascontiguousarray(np.stack(start_pixels), dtype=np.uint8)
    goal_pixel_array = np.ascontiguousarray(np.stack(goal_pixels), dtype=np.uint8)

    pointwise_mean = standardized.mean(axis=0, dtype=np.float64)
    central_index = int(
        np.square(standardized.astype(np.float64) - pointwise_mean)
        .sum(axis=(1, 2))
        .argmin()
    )
    open_hold_raw = np.zeros((total_steps, 7), dtype=np.float32)
    open_hold_raw[:, 6] = 1.0
    open_hold_standardized = np.ascontiguousarray(
        np.asarray(scaler.transform(open_hold_raw), dtype=np.float32).reshape(
            horizon, action_block * 7
        )
    )
    unique_splits, split_counts = np.unique(switches, return_counts=True)
    bank_sha256 = _sha256_arrays(
        raw_array.astype("<f4", copy=False),
        source_ids.astype("<i4", copy=False),
        starts.astype("<i8", copy=False),
        goals_array.astype("<i8", copy=False),
    )
    endpoint_sha256 = _sha256_arrays(start_pixel_array, goal_pixel_array)
    plan_sha256 = [
        _sha256_arrays(raw_array[index]) for index in range(len(raw_array))
    ]
    provenance = {
        "schema_version": "franka-train-terminal-plan-bank-v1",
        "training_dataset": str(path),
        "source_dataset": source_dataset,
        "selected_split": "train",
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "action_alignment": "obs[t]=s_t, action[t]=a_t",
        "horizon": int(horizon),
        "action_block": int(action_block),
        "raw_plan_steps": total_steps,
        "action_dim": 7,
        "minimum_open_commands": int(minimum_open_commands),
        "allowed_terminal_switch_steps": [
            int(value) for value in terminal_switch_steps
        ],
        "observed_strict_candidate_count": observed_candidate_count,
        "excluded_by_raw_bounds": excluded_by_raw_bounds,
        "excluded_by_continuous_plan_tv": excluded_by_plan_tv,
        "raw_support_low": [float(value) for value in raw_low_array],
        "raw_support_high": [float(value) for value in raw_high_array],
        "continuous_plan_tv_max": float(continuous_plan_tv_max),
        "accepted_continuous_plan_tv_observed_max": float(plan_tv.max()),
        "candidate_count": int(len(raw_array)),
        "source_demo_count": int(len(np.unique(source_ids))),
        "source_demo_ids": [int(value) for value in source_ids],
        "source_demo_ids_sha256": _sha256_arrays(
            source_ids.astype("<i4", copy=False)
        ),
        "ordering": "lexicographic(source_demo_id,start_row)",
        "candidate_macro_shape": [int(len(raw_array)), horizon, action_block * 7],
        "candidate_raw_shape": [int(len(raw_array)), total_steps, 7],
        "candidate_plan_sha256": plan_sha256,
        "terminal_switch_counts": {
            str(int(step)): int(count)
            for step, count in zip(unique_splits, split_counts)
        },
        "central_plan_index": central_index,
        "central_plan_source_demo_id": int(source_ids[central_index]),
        "bank_sha256": bank_sha256,
        "endpoint_pixels_sha256": endpoint_sha256,
        "open_hold_raw_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    return FrankaTerminalPlanBank(
        standardized_plans=standardized,
        raw_plans=raw_array,
        source_demo_ids=source_ids,
        source_episode_indices=episode_ids,
        start_rows=starts,
        goal_rows=goals_array,
        switch_steps=switches,
        start_pixels=start_pixel_array,
        goal_pixels=goal_pixel_array,
        central_index=central_index,
        open_hold_standardized=open_hold_standardized,
        provenance=provenance,
    )


class ProjectedPlanBankSolver(ProjectedCEMSolver):
    """Select a complete demonstrated plan, without synthesizing CEM actions."""

    def __init__(
        self,
        *,
        model: Any,
        plan_bank: FrankaTerminalPlanBank,
        selection_mode: str,
        continuation: str = "abstain",
        device: str | torch.device = "cpu",
        action_projection: StandardizedActionProjection,
        start_embeddings: torch.Tensor | None = None,
        goal_embeddings: torch.Tensor | None = None,
        retrieval_topk: int = 300,
    ) -> None:
        if selection_mode not in PLAN_BANK_SELECTION_MODES:
            raise ValueError(f"unknown plan-bank selection mode {selection_mode!r}")
        if continuation not in PLAN_BANK_CONTINUATIONS:
            raise ValueError(f"unknown plan-bank continuation {continuation!r}")
        super().__init__(
            model=model,
            batch_size=1,
            num_samples=plan_bank.size,
            var_scale=0.0,
            n_steps=1,
            topk=1,
            device=device,
            seed=0,
            action_projection=action_projection,
        )
        self.plan_bank = plan_bank
        self.selection_mode = selection_mode
        self.continuation = continuation
        self.retrieval_topk = int(retrieval_topk)
        if self.retrieval_topk < 1 or self.retrieval_topk > plan_bank.size:
            raise ValueError("retrieval_topk must lie inside the terminal bank")
        if selection_mode in ("behavior_retrieval", "wm_rerank"):
            if start_embeddings is None or goal_embeddings is None:
                raise ValueError("retrieval modes require train endpoint embeddings")
            start_embeddings = torch.as_tensor(start_embeddings, dtype=torch.float32)
            goal_embeddings = torch.as_tensor(goal_embeddings, dtype=torch.float32)
            if (
                start_embeddings.ndim != 2
                or goal_embeddings.shape != start_embeddings.shape
                or start_embeddings.shape[0] != plan_bank.size
                or not bool(torch.isfinite(start_embeddings).all())
                or not bool(torch.isfinite(goal_embeddings).all())
            ):
                raise ValueError("terminal endpoint embeddings are malformed")
            self.start_embeddings = start_embeddings.to(self.device)
            self.goal_embeddings = goal_embeddings.to(self.device)
            self.start_distance_scale = self._leave_one_out_scale(
                self.start_embeddings
            )
            self.goal_distance_scale = self._leave_one_out_scale(
                self.goal_embeddings
            )
            self.retrieval_embeddings_sha256 = _sha256_arrays(
                self.start_embeddings.detach().cpu().numpy(),
                self.goal_embeddings.detach().cpu().numpy(),
            )
        else:
            self.start_embeddings = None
            self.goal_embeddings = None
            self.start_distance_scale = None
            self.goal_distance_scale = None
            self.retrieval_embeddings_sha256 = None
        self._episode_solve_index = 0
        self.last_solve_diagnostics: dict[str, Any] | None = None

    @staticmethod
    def _leave_one_out_scale(embeddings: torch.Tensor) -> float:
        if embeddings.shape[0] < 2:
            raise ValueError("retrieval scaling requires at least two train endpoints")
        distances = torch.cdist(embeddings.float(), embeddings.float()).square()
        distances.fill_diagonal_(float("inf"))
        nearest = distances.min(dim=1).values
        scale = float(nearest.median().item())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("train-only retrieval distance scale is degenerate")
        return scale

    def _encode_query_endpoints(self, info_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = info_dict.get("pixels")
        goal = info_dict.get("goal")
        if not torch.is_tensor(pixels) or not torch.is_tensor(goal):
            raise ValueError("retrieval modes require transformed pixels and goal tensors")
        parameter = next(self.model.parameters())
        pixels = pixels.to(device=parameter.device, dtype=parameter.dtype)
        goal = goal.to(device=parameter.device, dtype=parameter.dtype)
        current_encoded = self.model.encode({"pixels": pixels[:, -1:]})["emb"][:, -1]
        goal_encoded = self.model.encode({"pixels": goal[:, -1:]})["emb"][:, -1]
        return current_encoded.float(), goal_encoded.float()

    def _retrieval_shortlist(
        self, info_dict: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current, goal = self._encode_query_endpoints(info_dict)
        start_distance = torch.cdist(current, self.start_embeddings).square()
        goal_distance = torch.cdist(goal, self.goal_embeddings).square()
        score = (
            start_distance / self.start_distance_scale
            + goal_distance / self.goal_distance_scale
        )
        shortlist_score, shortlist = score.topk(
            self.retrieval_topk, dim=1, largest=False, sorted=True
        )
        return shortlist, shortlist_score

    def reset_action_latches(self) -> None:
        super().reset_action_latches()
        self._episode_solve_index = 0
        self.last_solve_diagnostics = None

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        super().configure(action_space=action_space, n_envs=n_envs, config=config)
        expected = (self.plan_bank.size, self.horizon, self.action_dim)
        if self.plan_bank.standardized_plans.shape != expected:
            raise ValueError(
                "standardized terminal plan bank does not match planner config: "
                f"expected {expected}, got {self.plan_bank.standardized_plans.shape}"
            )
        if self.plan_bank.open_hold_standardized.shape != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("open-hold plan does not match planner config")

    def _expand_info(self, info_dict: dict, candidate_count: int) -> dict:
        total_envs = len(next(iter(info_dict.values())))
        expanded: dict[str, Any] = {}
        for key, value in info_dict.items():
            if torch.is_tensor(value):
                target_dtype = self.dtype if value.is_floating_point() else None
                batch = value.to(device=self.device, dtype=target_dtype)
                expanded[key] = batch.unsqueeze(1).expand(
                    total_envs, candidate_count, *batch.shape[1:]
                )
            elif isinstance(value, np.ndarray):
                expanded[key] = np.repeat(value[:, None, ...], candidate_count, axis=1)
            elif isinstance(value, list):
                expanded[key] = value
            else:
                expanded[key] = value
        return expanded

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        del init_action  # A real bank member, never a synthetic warm start, is returned.
        total_envs = len(next(iter(info_dict.values())))
        solve_index = self._episode_solve_index
        self._episode_solve_index += 1

        abstain = solve_index > 0 and self.continuation == "abstain"
        use_terminal_bank = solve_index == 0 or self.continuation == "repeat_terminal"
        if use_terminal_bank:
            all_candidates = torch.as_tensor(
                self.plan_bank.standardized_plans,
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0).expand(total_envs, -1, -1, -1)
            projected = self._project(all_candidates)
            projection_error = float(
                (projected - all_candidates).abs().max().detach().cpu().item()
            )
            if projection_error > 1e-6:
                raise ValueError(
                    "an exact terminal bank member was materially modified by "
                    f"projection (max standardized error={projection_error:.3e})"
                )
            # Preserve the canonical stored tensor. The projection calculation
            # may recreate a categorical float with sub-ulp roundoff, but a
            # bank proposal must score and return the exact standardized bytes.
            projected = all_candidates
            shortlist = None
            retrieval_scores = None
            if self.selection_mode in ("behavior_retrieval", "wm_rerank"):
                shortlist, retrieval_scores = self._retrieval_shortlist(info_dict)
                batch_rows = torch.arange(total_envs, device=self.device)[:, None]
                candidates = projected[batch_rows, shortlist]
            else:
                candidates = projected
            if self.selection_mode == "wm_rerank":
                expanded_info = self._expand_info(info_dict, self.retrieval_topk)
                costs = self.model.get_cost(expanded_info, candidates)
                if not isinstance(costs, torch.Tensor) or costs.shape != (
                    total_envs,
                    self.retrieval_topk,
                ):
                    raise ValueError("world model returned invalid plan-bank costs")
                if not bool(torch.isfinite(costs).all()):
                    raise ValueError("world model returned non-finite plan-bank costs")
                selected_local = costs.argmin(dim=1)
                selected_costs = costs.gather(1, selected_local[:, None])[:, 0]
                selected_indices = shortlist.gather(1, selected_local[:, None])[:, 0]
                returned = candidates[
                    torch.arange(total_envs, device=self.device), selected_local
                ]
            elif self.selection_mode == "behavior_retrieval":
                selected_indices = shortlist[:, 0]
                selected_costs = retrieval_scores[:, 0]
                returned = candidates[:, 0]
            else:
                selected_indices = torch.full(
                    (total_envs,),
                    self.plan_bank.central_index,
                    device=self.device,
                    dtype=torch.long,
                )
                selected_costs = torch.zeros(
                    total_envs, device=self.device, dtype=self.dtype
                )
                rows = torch.arange(total_envs, device=self.device)
                returned = candidates[rows, selected_indices]
            selected_list = [int(value) for value in selected_indices.cpu().tolist()]
            source_demo_ids = [
                int(self.plan_bank.source_demo_ids[index]) for index in selected_list
            ]
            plan_kind = "train_terminal"
        else:
            # After one complete terminal plan, a second close->open maneuver is
            # precisely the failure mode this ablation is intended to remove.
            # Use the transparent zero-arm/open-gripper baseline as a settling
            # continuation.  It is shared by behavior-only and WM-rerank arms.
            returned = torch.as_tensor(
                self.plan_bank.open_hold_standardized,
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0).expand(total_envs, -1, -1)
            returned = self.action_projection.project(returned)
            selected_costs = torch.zeros(
                total_envs, device=self.device, dtype=self.dtype
            )
            selected_list = [-1] * total_envs
            source_demo_ids = [-1] * total_envs
            plan_kind = "abstain" if abstain else "open_hold"

        self.last_solve_diagnostics = {
            "episode_solve_index": solve_index,
            "plan_kind": plan_kind,
            "selection_mode": self.selection_mode,
            "selected_plan_indices": selected_list,
            "selected_source_demo_ids": source_demo_ids,
            "selected_model_costs": [
                float(value) for value in selected_costs.detach().cpu().tolist()
            ],
            "candidate_count": self.plan_bank.size if use_terminal_bank else 1,
            "model_scored_candidate_count": (
                self.retrieval_topk if self.selection_mode == "wm_rerank" and use_terminal_bank else 0
            ),
            "abstain": abstain,
        }
        return {
            "actions": returned.detach().cpu(),
            "costs": selected_costs.detach().cpu().tolist(),
            "mean": [returned.detach().cpu()],
            "var": [torch.zeros_like(returned).detach().cpu()],
            "returned_plan_source": [plan_kind] * total_envs,
            "action_projection": self.action_projection.to_dict(),
            "plan_bank_diagnostics": self.last_solve_diagnostics,
        }

    def proposal_provenance(self) -> dict[str, Any]:
        return {
            "mode": "train_terminal_plan_bank",
            "selection": self.selection_mode,
            "continuation": self.continuation,
            "world_model_reranks_candidates": self.selection_mode == "wm_rerank",
            "goal_conditioned_retrieval": self.selection_mode in (
                "behavior_retrieval", "wm_rerank"
            ),
            "retrieval_topk": (
                self.retrieval_topk if self.start_embeddings is not None else None
            ),
            "retrieval_distance": "projected_encoder_squared_l2_start_plus_goal",
            "retrieval_scale": (
                {
                    "fit": "train_leave_one_out_nearest_neighbor_median",
                    "start": self.start_distance_scale,
                    "goal": self.goal_distance_scale,
                }
                if self.start_distance_scale is not None
                else None
            ),
            "retrieval_embeddings_sha256": self.retrieval_embeddings_sha256,
            "synthetic_cem_candidates": False,
            **self.plan_bank.provenance,
        }
