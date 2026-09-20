"""Stock-CEM safety arm plus a DCL-proposal CEM challenger.

The two optimizers run independently.  Candidate zero is the literal stock
CEM plan.  The challenger is accepted only when it improves the unchanged
terminal world-model cost and its query-local inverse-DCL support is no worse
than the stock plan.  This avoids comparing raw InfoNCE scores across distinct
transition queries, which is not statistically meaningful.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import torch

from stable_worldmodel.solver.cem import CEMSolver

from trim.planning.solver_dcl_beam_cem import DCLBeamCEMSolver


def select_safe_dcl_plan(
    costs: torch.Tensor,
    support: torch.Tensor,
    *,
    min_relative_goal_gain: float | Sequence[float] = 0.02,
    min_support_gain: float = 0.0,
    min_dcl_support: float = 0.0,
    secondary_costs: torch.Tensor | None = None,
    min_secondary_relative_gain: float = float("-inf"),
    robust_median_gain: torch.Tensor | None = None,
    min_robust_median_gain: float = float("-inf"),
    robust_win_fraction: torch.Tensor | None = None,
    min_robust_win_fraction: float = 0.0,
    residual_start_index: int | None = None,
    min_residual_relative_gain: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Choose the lowest-cost eligible arm; arm zero is always stock CEM.

    Every non-zero arm must independently clear the primary goal, inverse-DCL,
    optional second-world-model, and optional local-robustness gates.  Keeping
    the tensors candidate-shaped makes the same rule valid for a two-arm
    baseline and for a multi-retriever portfolio without task-specific logic.
    """
    if costs.ndim != 2 or costs.shape[1] < 2:
        raise ValueError(f"costs must be (B,C>=2), got {costs.shape}")
    if support.ndim != 3 or support.shape[:2] != costs.shape:
        raise ValueError(f"support must be (B,C,H), got {support.shape}")
    relative_gain = (
        costs[:, :1] - costs
    ) / costs[:, :1].abs().clamp_min(1e-8)
    support_mean = support.mean(dim=-1)
    support_floor = support.min(dim=-1).values
    if isinstance(min_relative_goal_gain, Sequence):
        challenger_thresholds = torch.as_tensor(
            list(min_relative_goal_gain), device=costs.device, dtype=costs.dtype
        )
        if challenger_thresholds.numel() != costs.shape[1] - 1:
            raise ValueError(
                "candidate-specific goal thresholds must have one value per "
                f"challenger ({costs.shape[1] - 1}), got "
                f"{challenger_thresholds.numel()}"
            )
        goal_thresholds = torch.cat(
            [torch.full_like(challenger_thresholds[:1], float("-inf")),
             challenger_thresholds]
        )[None]
    else:
        goal_thresholds = torch.full_like(costs, float(min_relative_goal_gain))
        goal_thresholds[:, 0] = float("-inf")
    eligible = (
        torch.isfinite(relative_gain)
        & (relative_gain >= goal_thresholds)
        & (support_mean >= support_mean[:, :1] + float(min_support_gain))
        & (support_floor >= float(min_dcl_support))
    )
    eligible[:, 0] = True

    secondary_relative_gain = None
    if secondary_costs is not None:
        if secondary_costs.shape != costs.shape:
            raise ValueError(
                f"secondary_costs must match {costs.shape}, got {secondary_costs.shape}"
            )
        secondary_relative_gain = (
            secondary_costs[:, :1] - secondary_costs
        ) / secondary_costs[:, :1].abs().clamp_min(1e-8)
        eligible[:, 1:] &= (
            torch.isfinite(secondary_relative_gain[:, 1:])
            & (
                secondary_relative_gain[:, 1:]
                >= float(min_secondary_relative_gain)
            )
        )

    for name, value in (
        ("robust_median_gain", robust_median_gain),
        ("robust_win_fraction", robust_win_fraction),
    ):
        if value is not None and value.shape != costs.shape:
            raise ValueError(f"{name} must match {costs.shape}, got {value.shape}")
    if robust_median_gain is not None:
        eligible[:, 1:] &= (
            torch.isfinite(robust_median_gain[:, 1:])
            & (robust_median_gain[:, 1:] >= float(min_robust_median_gain))
        )
    if robust_win_fraction is not None:
        eligible[:, 1:] &= (
            torch.isfinite(robust_win_fraction[:, 1:])
            & (robust_win_fraction[:, 1:] >= float(min_robust_win_fraction))
        )

    residual_relative_gain = None
    residual_reference = None
    if residual_start_index is not None:
        start = int(residual_start_index)
        if not 1 < start < costs.shape[1]:
            raise ValueError(
                "residual_start_index must leave stock plus at least one base arm"
            )
        base_costs = torch.where(
            eligible[:, :start] & torch.isfinite(costs[:, :start]),
            costs[:, :start],
            torch.full_like(costs[:, :start], float("inf")),
        )
        residual_reference = base_costs.argmin(dim=1)
        rows = torch.arange(costs.shape[0], device=costs.device)
        reference_cost = costs[rows, residual_reference]
        reference_support = support_mean[rows, residual_reference]
        residual_relative_gain = (
            reference_cost[:, None] - costs
        ) / reference_cost.abs().clamp_min(1e-8)[:, None]
        eligible[:, start:] &= (
            torch.isfinite(residual_relative_gain[:, start:])
            & (
                residual_relative_gain[:, start:]
                >= float(min_residual_relative_gain)
            )
            & (support_mean[:, start:] >= reference_support[:, None])
        )

    masked_costs = torch.where(
        eligible & torch.isfinite(costs),
        costs,
        torch.full_like(costs, float("inf")),
    )
    selected = masked_costs.argmin(dim=1)
    rows = torch.arange(costs.shape[0], device=costs.device)
    accept = selected != 0
    return selected, {
        "accepted": accept,
        "eligible": eligible,
        "relative_goal_gain": relative_gain[rows, selected],
        "candidate_relative_goal_gain": relative_gain,
        "secondary_relative_goal_gain": secondary_relative_gain,
        "robust_median_gain": robust_median_gain,
        "robust_win_fraction": robust_win_fraction,
        "support_mean": support_mean,
        "support_floor": support_floor,
        "residual_relative_gain": residual_relative_gain,
        "residual_reference": residual_reference,
    }


class SafeDCLPortfolioSolver(CEMSolver):
    """Run exact stock CEM and a DCL-beam challenger, then apply a safe gate."""

    def __init__(
        self,
        *args: Any,
        decoder: Any,
        min_relative_goal_gain: float = 0.02,
        challenger_goal_gains: Sequence[float] | None = None,
        min_support_gain: float = 0.0,
        min_dcl_support: float = 0.0,
        refine_steps: int = 0,
        refine_step_size: float = 0.05,
        refine_radius: float = 0.15,
        evaluator: Any | None = None,
        min_secondary_relative_gain: float = float("-inf"),
        robust_samples: int = 0,
        robust_noise_scale: float = 0.25,
        robust_scale_floor: float = 0.05,
        min_robust_median_gain: float = 0.02,
        min_robust_win_fraction: float = 0.75,
        robust_seed_stride: int = 65_537,
        challenger_variants: list[dict[str, Any]] | None = None,
        seeded_models: Sequence[Any] | None = None,
        residual_start_index: int | None = None,
        min_residual_relative_gain: float = 0.0,
        capture_plans: bool = False,
        force_arm: int | None = None,
        trace: bool = False,
        dcl_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        seed = int(kwargs.get("seed", 1234))
        super().__init__(*args, **kwargs)
        if getattr(decoder, "mode", None) != "contrastive":
            raise ValueError("SafeDCLPortfolioSolver requires a contrastive IDM")
        self.decoder = decoder
        self.min_relative_goal_gain = float(min_relative_goal_gain)
        self.challenger_goal_gains = (
            [float(value) for value in challenger_goal_gains]
            if challenger_goal_gains is not None
            else None
        )
        self.min_support_gain = float(min_support_gain)
        self.min_dcl_support = float(min_dcl_support)
        if refine_steps < 0:
            raise ValueError("refine_steps must be non-negative")
        if refine_step_size <= 0:
            raise ValueError("refine_step_size must be positive")
        if refine_radius < 0:
            raise ValueError("refine_radius must be non-negative")
        self.refine_steps = int(refine_steps)
        self.refine_step_size = float(refine_step_size)
        self.refine_radius = float(refine_radius)
        if robust_samples < 0:
            raise ValueError("robust_samples must be non-negative")
        if robust_noise_scale < 0 or robust_scale_floor < 0:
            raise ValueError("robust noise scale/floor must be non-negative")
        if not 0 <= min_robust_win_fraction <= 1:
            raise ValueError("min_robust_win_fraction must lie in [0,1]")
        self.evaluator = evaluator
        self.min_secondary_relative_gain = float(min_secondary_relative_gain)
        self.robust_samples = int(robust_samples)
        self.robust_noise_scale = float(robust_noise_scale)
        self.robust_scale_floor = float(robust_scale_floor)
        self.min_robust_median_gain = float(min_robust_median_gain)
        self.min_robust_win_fraction = float(min_robust_win_fraction)
        self.robust_gen = torch.Generator(device=self.device).manual_seed(
            seed + int(robust_seed_stride)
        )
        self.trace = bool(trace)
        self.residual_start_index = (
            int(residual_start_index) if residual_start_index is not None else None
        )
        self.min_residual_relative_gain = float(min_residual_relative_gain)
        self.capture_plans = bool(capture_plans)
        self.force_arm = int(force_arm) if force_arm is not None else None
        self.capture_records: list[dict[str, Any]] = []
        child = {
            "model": self.model,
            "batch_size": self.batch_size,
            "num_samples": self.num_samples,
            "var_scale": self.var_scale,
            "n_steps": self.n_steps,
            "topk": self.topk,
            "device": self.device,
            "seed": seed,
            "decoder": decoder,
        }
        child.update(dcl_kwargs or {})
        child["trace"] = False
        self.dcl_solvers = [DCLBeamCEMSolver(**child)]
        for variant in challenger_variants or []:
            variant_child = dict(child)
            variant_child.update(variant)
            self.dcl_solvers.append(DCLBeamCEMSolver(**variant_child))
        seeded_child = {
            "batch_size": self.batch_size,
            "num_samples": self.num_samples,
            "var_scale": self.var_scale,
            "n_steps": self.n_steps,
            "topk": self.topk,
            "device": self.device,
            # Common random numbers make the seeded arm a clean test of the
            # initial mean rather than a hidden extra-randomness restart.
            "seed": seed,
        }
        self.seeded_solvers = [
            CEMSolver(model=seeded_model, **seeded_child)
            for seeded_model in (seeded_models or [])
        ]
        if (
            self.challenger_goal_gains is not None
            and len(self.challenger_goal_gains)
            != len(self.dcl_solvers) + len(self.seeded_solvers)
        ):
            raise ValueError(
                "challenger_goal_gains must have one threshold per challenger: "
                f"expected {len(self.dcl_solvers) + len(self.seeded_solvers)}, got "
                f"{len(self.challenger_goal_gains)}"
            )
        # Backward-compatible name used by a few exploratory notebooks.
        self.dcl_solver = self.dcl_solvers[0]
        self._diag = {"solves": 0, "decisions": 0, "accepted": 0}

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        super().configure(action_space=action_space, n_envs=n_envs, config=config)
        for solver in [*self.dcl_solvers, *self.seeded_solvers]:
            solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    def _expand_infos(
        self, info_dict: dict, samples: int, start: int = 0, end: int | None = None
    ) -> dict[str, Any]:
        expanded: dict[str, Any] = {}
        for key, value in info_dict.items():
            batch = value[start:end]
            if torch.is_tensor(value):
                target_dtype = self.dtype if batch.is_floating_point() else None
                batch = batch.to(device=self.device, dtype=target_dtype)
                expanded[key] = batch.unsqueeze(1).expand(
                    batch.shape[0], samples, *batch.shape[1:]
                )
            elif isinstance(value, np.ndarray):
                expanded[key] = np.repeat(batch[:, None, ...], samples, axis=1)
            else:
                expanded[key] = batch
        return expanded

    def _score_plan_set(
        self, model: Any, info_dict: dict, plans: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a plan set against each environment's own goal.

        ``JEPA.criterion`` historically stores ``goal_emb`` without an
        explicit sample axis.  It is safe for the stock solver because its
        configured batch size is one, but a direct ``(B,2,...)`` portfolio
        call would right-align and mix the environment axis with the sample
        axis.  Preserve the model contract by evaluating portfolio pairs in
        the same B=1 chunks used by stock CEM.
        """
        all_costs = []
        all_chains = []
        all_goals = []
        # This must stay one regardless of the CEM optimization batch size:
        # JEPA.goal_emb has no explicit sample axis, so any B>1 pair scoring
        # can silently align the environment axis with S=2.
        for start in range(0, plans.shape[0], 1):
            end = start + 1
            expanded = self._expand_infos(
                info_dict, samples=plans.shape[1], start=start, end=end
            )
            cost = model.get_cost(expanded, plans[start:end]).float()
            predicted = expanded.get("predicted_emb")
            if predicted is None:
                raise RuntimeError("world model get_cost did not expose predicted_emb")
            all_costs.append(cost)
            all_chains.append(predicted[:, :, -(self.horizon + 1) :, :])
            if getattr(self, "capture_plans", False):
                goal_emb = expanded.get("goal_emb")
                if goal_emb is None:
                    raise RuntimeError("world model get_cost did not expose goal_emb")
                all_goals.append(goal_emb[:, -1].float())
        if all_goals:
            self._last_scored_goal = torch.cat(all_goals, dim=0)
        return torch.cat(all_costs, dim=0), torch.cat(all_chains, dim=0)

    def _score_plan_pair(
        self, info_dict: dict, plans: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible wrapper for the original two-arm tests."""
        return self._score_plan_set(self.model, info_dict, plans)

    def _robust_plan_gains(
        self,
        info_dict: dict,
        plans: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Paired common-noise local stability of each challenger vs stock.

        For each environment and challenger, both plans receive the exact same
        perturbation tensor.  Scoring remains B=1, so robustness probes cannot
        reintroduce the historical environment/goal broadcasting bug.
        """
        if self.robust_samples == 0:
            return None, None
        batch, candidates, horizon, action_dim = plans.shape
        medians = torch.zeros(batch, candidates, device=self.device)
        wins = torch.ones(batch, candidates, device=self.device)
        for env_index in range(batch):
            # One pooled neighborhood and one epsilon set are shared by every
            # arm.  This removes candidate-count-dependent Monte-Carlo draws
            # and makes all arms face the same local perturbation directions.
            shared_scale = scales[env_index].float().mean(dim=0).clamp_min(
                self.robust_scale_floor
            )
            epsilon = torch.randn(
                self.robust_samples,
                horizon,
                action_dim,
                generator=self.robust_gen,
                device=self.device,
                dtype=torch.float32,
            )
            delta = self.robust_noise_scale * shared_scale[None] * epsilon
            probes = (
                plans[env_index].float()[None, :, :, :] + delta[:, None, :, :]
            ).reshape(1, self.robust_samples * candidates, horizon, action_dim)
            expanded = self._expand_infos(
                info_dict,
                samples=self.robust_samples * candidates,
                start=env_index,
                end=env_index + 1,
            )
            probe_cost = self.model.get_cost(
                expanded, probes.to(device=self.device, dtype=self.dtype)
            ).float().reshape(self.robust_samples, candidates)
            gain = (probe_cost[:, :1] - probe_cost) / probe_cost[
                :, :1
            ].abs().clamp_min(1e-8)
            medians[env_index] = gain.median(dim=0).values
            wins[env_index] = (gain > 0).float().mean(dim=0)
            wins[env_index, 0] = 1.0
        return medians, wins

    def _refine_challenger(
        self, info_dict: dict, plans: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Projected terminal-cost descent around each DCL proposal basin.

        No model weights are updated.  Every environment is optimized alone
        to retain the same goal/sample alignment contract as stock CEM.  The
        starting DCL plan remains in the best-so-far set, so refinement cannot
        return a plan with a higher nominal world-model cost.
        """
        if self.refine_steps == 0 or self.refine_radius == 0:
            return plans, {"mean_rms": 0.0, "mean_model_gain": 0.0}

        refined = []
        rms_values = []
        gain_values = []
        for env_index in range(plans.shape[0]):
            # Disable the outer inference-mode region only for the leaf action
            # tensor.  Frozen model parameters still receive no gradients.
            with torch.inference_mode(False), torch.enable_grad():
                prototype = plans[env_index].detach().float().clone()
                current = prototype.clone()
                best = prototype.clone()
                best_cost = None
                initial_cost = None
                for iteration in range(self.refine_steps + 1):
                    action = current.detach().clone().requires_grad_(
                        iteration < self.refine_steps
                    )
                    expanded = self._expand_infos(
                        info_dict, samples=1, start=env_index, end=env_index + 1
                    )
                    cost = self.model.get_cost(
                        expanded, action[None, None].to(self.device, self.dtype)
                    )[0, 0].float()
                    if initial_cost is None:
                        initial_cost = float(cost.detach())
                    if best_cost is None or float(cost.detach()) < best_cost:
                        best_cost = float(cost.detach())
                        best = action.detach().clone()
                    if iteration == self.refine_steps:
                        break
                    grad = torch.autograd.grad(cost, action, only_inputs=True)[0]
                    grad_rms = grad.square().mean().sqrt().clamp_min(1e-8)
                    proposal = action - self.refine_step_size * grad / grad_rms
                    delta = proposal - prototype
                    delta_rms = delta.square().mean().sqrt().clamp_min(1e-8)
                    scale = min(1.0, self.refine_radius / float(delta_rms.detach()))
                    current = (prototype + scale * delta).detach()
                refined.append(best.to(device=self.device, dtype=self.dtype))
                rms_values.append(float((best - prototype).square().mean().sqrt()))
                gain_values.append(
                    (initial_cost - best_cost) / max(abs(initial_cost), 1e-8)
                )
        return torch.stack(refined, dim=0), {
            "mean_rms": float(np.mean(rms_values)),
            "mean_model_gain": float(np.mean(gain_values)),
        }

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start = time.time()
        stock = super().solve(info_dict, init_action)
        challengers = [
            solver.solve(info_dict, init_action)
            for solver in [*self.dcl_solvers, *self.seeded_solvers]
        ]
        refined_actions = []
        refine_diags = []
        for challenger in challengers:
            actions, diag = self._refine_challenger(
                info_dict,
                challenger["actions"].to(device=self.device, dtype=self.dtype),
            )
            refined_actions.append(actions)
            refine_diags.append(diag)
        plans = torch.stack(
            [stock["actions"].to(device=self.device, dtype=self.dtype)]
            + refined_actions,
            dim=1,
        ).to(device=self.device, dtype=self.dtype)
        costs, chain = self._score_plan_set(self.model, info_dict, plans)
        primary_goal = (
            self._last_scored_goal.detach().clone() if self.capture_plans else None
        )
        secondary_costs = None
        if self.evaluator is not None:
            secondary_costs, _ = self._score_plan_set(
                self.evaluator, info_dict, plans
            )
        z = chain[:, :, :-1, :].reshape(-1, chain.shape[-1])
        z_next = chain[:, :, 1:, :].reshape(-1, chain.shape[-1])
        action = plans.reshape(-1, self.action_dim).float()
        support = self.decoder.support_percentile(z, z_next, action).reshape(
            plans.shape[0], plans.shape[1], self.horizon
        )
        candidate_vars = torch.stack(
            [stock["var"][0]] + [challenger["var"][0] for challenger in challengers],
            dim=1,
        ).to(device=self.device, dtype=self.dtype)
        robust_median, robust_win = self._robust_plan_gains(
            info_dict, plans, candidate_vars
        )
        selected_idx, details = select_safe_dcl_plan(
            costs,
            support,
            min_relative_goal_gain=(
                self.challenger_goal_gains
                if self.challenger_goal_gains is not None
                else self.min_relative_goal_gain
            ),
            min_support_gain=self.min_support_gain,
            min_dcl_support=self.min_dcl_support,
            secondary_costs=secondary_costs,
            min_secondary_relative_gain=self.min_secondary_relative_gain,
            robust_median_gain=robust_median,
            min_robust_median_gain=self.min_robust_median_gain,
            robust_win_fraction=robust_win,
            min_robust_win_fraction=self.min_robust_win_fraction,
            residual_start_index=self.residual_start_index,
            min_residual_relative_gain=self.min_residual_relative_gain,
        )
        policy_selected_idx = selected_idx
        if self.force_arm is not None:
            if not 0 <= self.force_arm < plans.shape[1]:
                raise ValueError(
                    f"force_arm {self.force_arm} outside portfolio size {plans.shape[1]}"
                )
            selected_idx = torch.full_like(selected_idx, self.force_arm)
        rows = torch.arange(plans.shape[0], device=self.device)
        chosen = plans[rows, selected_idx]
        chosen_cost = costs[rows, selected_idx]

        stock["actions"] = chosen.detach().cpu()
        stock["costs"] = chosen_cost.detach().cpu().tolist()
        stock["mean"] = [stock["actions"]]
        stock["var"] = [candidate_vars[rows, selected_idx].detach().cpu()]
        self._diag["solves"] += 1
        self._diag["decisions"] += int(plans.shape[0])
        self._diag["accepted"] += int((selected_idx != 0).sum().item())
        if self.capture_plans:
            self.capture_records.append({
                "candidate_actions": plans.detach().cpu(),
                "candidate_costs": costs.detach().cpu(),
                "candidate_support": support.detach().cpu(),
                "current_latent": chain[:, 0, 0].detach().cpu(),
                "goal_latent": primary_goal.cpu(),
                "policy_selected_idx": policy_selected_idx.detach().cpu(),
                "executed_idx": selected_idx.detach().cpu(),
                "executed_actions": chosen.detach().cpu(),
                "force_arm": self.force_arm,
            })
        stock["safe_dcl_portfolio"] = {
            **self._diag,
            "accept_rate": self._diag["accepted"] / max(1, self._diag["decisions"]),
            "last_selected": selected_idx.detach().cpu().tolist(),
            "last_relative_goal_gain": details["relative_goal_gain"].detach().cpu().tolist(),
            "last_candidate_goal_gain": details[
                "candidate_relative_goal_gain"
            ].detach().cpu().tolist(),
            "last_eligible": details["eligible"].detach().cpu().tolist(),
            "last_support_mean": details["support_mean"].detach().cpu().tolist(),
            "last_support_floor": details["support_floor"].detach().cpu().tolist(),
            "last_residual_goal_gain": (
                details["residual_relative_gain"].detach().cpu().tolist()
                if details["residual_relative_gain"] is not None
                else None
            ),
            "last_residual_reference": (
                details["residual_reference"].detach().cpu().tolist()
                if details["residual_reference"] is not None
                else None
            ),
            "last_secondary_goal_gain": (
                details["secondary_relative_goal_gain"].detach().cpu().tolist()
                if details["secondary_relative_goal_gain"] is not None
                else None
            ),
            "last_robust_median_gain": (
                robust_median.detach().cpu().tolist()
                if robust_median is not None
                else None
            ),
            "last_robust_win_fraction": (
                robust_win.detach().cpu().tolist()
                if robust_win is not None
                else None
            ),
            "last_refine": refine_diags,
            "force_arm": self.force_arm,
        }
        if self.trace:
            print(
                "Safe-DCL-Portfolio "
                f"{time.time() - start:.3f}s selected={stock['safe_dcl_portfolio']['last_selected']} "
                f"goal_gain={stock['safe_dcl_portfolio']['last_relative_goal_gain']} "
                f"support_mean={stock['safe_dcl_portfolio']['last_support_mean']} "
                f"secondary_gain={stock['safe_dcl_portfolio']['last_secondary_goal_gain']} "
                f"robust={stock['safe_dcl_portfolio']['last_robust_median_gain']}/"
                f"{stock['safe_dcl_portfolio']['last_robust_win_fraction']} "
                f"refine={refine_diags}",
                flush=True,
            )
        return stock
