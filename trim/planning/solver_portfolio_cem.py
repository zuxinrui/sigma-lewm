"""Test-time-scaled CEM with cross-world-model rank verification.

The stock CEM solve is always candidate zero.  Additional candidates are
produced by genuinely independent CEM optimizers (independent RNG streams,
not extra samples from the same shrinking Gaussian).  Frozen world models
then cross-score the resulting plans.  Since latent terminal costs are not
calibrated across checkpoints, selection uses within-model ranks rather than
raw cost magnitudes.

With ``restarts=1`` this class returns the parent CEM output without even
running the verifier.  This makes the R=1 condition a literal baseline rather
than an approximate reimplementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from stable_worldmodel.solver.cem import CEMSolver


def normalized_ranks(costs: torch.Tensor) -> torch.Tensor:
    """Convert lower-is-better costs ``(M,B,R)`` to ranks in ``[0,1]``."""
    if costs.ndim != 3:
        raise ValueError(f"costs must be (models,batch,restarts), got {costs.shape}")
    restarts = costs.shape[-1]
    if restarts == 1:
        return torch.zeros_like(costs, dtype=torch.float32)
    finite = torch.where(
        torch.isfinite(costs), costs, torch.full_like(costs, float("inf"))
    )
    # Pairwise definition gives equal costs equal ranks.  Downstream argmin is
    # stable, so an exact score tie still preserves candidate zero.
    ranks = (
        finite.unsqueeze(-2) < finite.unsqueeze(-1)
    ).sum(dim=-1).float()
    return ranks / float(restarts - 1)


def select_portfolio_plan(
    plans: torch.Tensor,
    costs: torch.Tensor,
    *,
    mode: str = "consensus",
    disagreement_weight: float = 0.25,
    min_consensus_gain: float = 0.0,
    primary_cost_margin: float = float("inf"),
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Select plans while retaining candidate zero as a literal fallback.

    Args:
        plans: ``(B,R,H,A)``; candidate zero is stock CEM.
        costs: ``(M,B,R)`` terminal latent costs, primary model first.
        mode: ``consensus`` for rank aggregation or ``primary_best`` for an
            oracle-with-respect-to-the-generator control.
        disagreement_weight: penalty on cross-model rank standard deviation.
        min_consensus_gain: required score improvement over candidate zero.
        primary_cost_margin: optional primary-model safety gate.
    """
    if plans.ndim != 4:
        raise ValueError(f"plans must be (B,R,H,A), got {plans.shape}")
    if costs.ndim != 3 or costs.shape[1:] != plans.shape[:2]:
        raise ValueError(
            f"costs must be (M,{plans.shape[0]},{plans.shape[1]}), got {costs.shape}"
        )
    if mode not in ("consensus", "primary_best"):
        raise ValueError(f"unknown portfolio mode {mode!r}")
    if disagreement_weight < 0:
        raise ValueError("disagreement_weight must be non-negative")

    ranks = normalized_ranks(costs)
    rank_mean = ranks.mean(dim=0)
    rank_std = ranks.std(dim=0, unbiased=False)
    consensus_score = rank_mean + float(disagreement_weight) * rank_std
    selection_score = consensus_score if mode == "consensus" else costs[0].float()

    safe = torch.isfinite(costs[0]) & (
        costs[0] <= costs[0, :, :1] + float(primary_cost_margin)
    )
    safe[:, 0] = True
    masked_score = torch.where(
        safe & torch.isfinite(selection_score),
        selection_score,
        torch.full_like(selection_score, float("inf")),
    )
    best = masked_score.argmin(dim=1)
    rows = torch.arange(plans.shape[0], device=plans.device)
    gain = selection_score[:, 0] - selection_score[rows, best]
    accept = (best != 0) & torch.isfinite(gain) & (
        gain >= float(min_consensus_gain)
    )
    selected_idx = torch.where(accept, best, torch.zeros_like(best))
    selected = plans[rows, selected_idx]
    details = {
        "ranks": ranks,
        "rank_mean": rank_mean,
        "rank_std": rank_std,
        "consensus_score": consensus_score,
        "selection_score": selection_score,
        "gain": gain,
        "accepted": accept,
    }
    return selected, selected_idx, details


@dataclass
class _PortfolioDiagnostics:
    restarts: int
    solves: int = 0
    decisions: int = 0
    replacements: int = 0
    selected_counts: list[int] = field(default_factory=list)
    gain_sum: float = 0.0
    primary_improvement_sum: float = 0.0
    diversity_sum: float = 0.0
    agreement_sum: float = 0.0

    def __post_init__(self) -> None:
        if not self.selected_counts:
            self.selected_counts = [0] * self.restarts

    def as_dict(self) -> dict[str, Any]:
        denom = max(self.decisions, 1)
        return {
            "solves": self.solves,
            "decisions": self.decisions,
            "replacements": self.replacements,
            "replacement_rate": self.replacements / denom,
            "selected_counts": self.selected_counts,
            "mean_selection_gain": self.gain_sum / denom,
            "mean_primary_cost_improvement": self.primary_improvement_sum / denom,
            "mean_action_rms_from_baseline": self.diversity_sum / denom,
            "mean_evaluator_agreement": self.agreement_sum / denom,
        }


class PortfolioCEMSolver(CEMSolver):
    """Independent CEM restart portfolio with rank-consensus verification."""

    def __init__(
        self,
        *args: Any,
        evaluators: Sequence[Any] | None = None,
        evaluator_names: Sequence[str] | None = None,
        restarts: int = 2,
        restart_seed_stride: int = 100_003,
        mode: str = "consensus",
        disagreement_weight: float = 0.25,
        min_consensus_gain: float = 0.0,
        primary_cost_margin: float = float("inf"),
        trace: bool = False,
        **kwargs: Any,
    ) -> None:
        seed = int(kwargs.get("seed", 1234))
        super().__init__(*args, **kwargs)
        if restarts < 1:
            raise ValueError("restarts must be >= 1")
        if mode not in ("consensus", "primary_best"):
            raise ValueError(f"unknown portfolio mode {mode!r}")

        # The generator model must be evaluator zero because its raw cost is
        # the optional safety gate and the primary_best compute-scaling control.
        self.evaluators = [self.model]
        for evaluator in evaluators or []:
            if evaluator is not self.model:
                self.evaluators.append(evaluator)
        self.evaluator_names = list(evaluator_names or [])
        if self.evaluator_names and len(self.evaluator_names) != len(self.evaluators):
            raise ValueError("evaluator_names must match the de-duplicated evaluators")
        if not self.evaluator_names:
            self.evaluator_names = [f"model_{i}" for i in range(len(self.evaluators))]

        self.restarts = int(restarts)
        self.mode = mode
        self.disagreement_weight = float(disagreement_weight)
        self.min_consensus_gain = float(min_consensus_gain)
        self.primary_cost_margin = float(primary_cost_margin)
        self.trace = bool(trace)
        self.last_trace: list[dict[str, Any]] = []
        self._diag = _PortfolioDiagnostics(self.restarts)

        child_kwargs = {
            "model": self.model,
            "batch_size": self.batch_size,
            "num_samples": self.num_samples,
            "var_scale": self.var_scale,
            "n_steps": self.n_steps,
            "topk": self.topk,
            "device": self.device,
        }
        self._restart_solvers = [
            CEMSolver(
                **child_kwargs,
                seed=seed + restart_seed_stride * restart,
            )
            for restart in range(1, self.restarts)
        ]

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        super().configure(action_space=action_space, n_envs=n_envs, config=config)
        for solver in self._restart_solvers:
            solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    def _expand_infos(
        self, info_dict: dict, start: int, end: int, samples: int
    ) -> dict:
        expanded: dict[str, Any] = {}
        for key, value in info_dict.items():
            if torch.is_tensor(value):
                value = value[start:end]
                dtype = self.dtype if value.is_floating_point() else None
                batch = value.to(device=self.device, dtype=dtype)
                expanded[key] = batch.unsqueeze(1).expand(
                    batch.shape[0], samples, *batch.shape[1:]
                )
            elif isinstance(value, np.ndarray):
                batch = value[start:end]
                expanded[key] = np.repeat(batch[:, None, ...], samples, axis=1)
            elif isinstance(value, list):
                expanded[key] = value[start:end]
            else:
                expanded[key] = value
        return expanded

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        # Candidate zero follows the exact stock implementation and RNG stream.
        outputs = super().solve(info_dict, init_action)
        self._diag.solves += 1
        if self.restarts == 1:
            self._diag.decisions += int(outputs["actions"].shape[0])
            self._diag.selected_counts[0] += int(outputs["actions"].shape[0])
            self.last_trace = []
            return outputs

        candidate_outputs = [outputs]
        for solver in self._restart_solvers:
            candidate_outputs.append(solver.solve(info_dict, init_action))
        plans = torch.stack(
            [result["actions"] for result in candidate_outputs], dim=1
        ).to(device=self.device, dtype=self.dtype)

        all_costs = []
        for evaluator in self.evaluators:
            evaluator_costs = []
            # JEPA.get_cost is normally called by CEM on solver-sized batches.
            # Preserve that contract here; in particular its goal embedding
            # shape assumes the solver's batch slicing before sample expansion.
            for start in range(0, plans.shape[0], self.batch_size):
                end = min(start + self.batch_size, plans.shape[0])
                expanded = self._expand_infos(
                    info_dict, start, end, self.restarts
                )
                evaluator_costs.append(
                    evaluator.get_cost(expanded, plans[start:end]).float()
                )
            cost = torch.cat(evaluator_costs, dim=0)
            if cost.shape != plans.shape[:2]:
                raise RuntimeError(
                    f"evaluator returned {cost.shape}; expected {plans.shape[:2]}"
                )
            all_costs.append(cost.float())
        costs = torch.stack(all_costs, dim=0)
        selected, selected_idx, details = select_portfolio_plan(
            plans,
            costs,
            mode=self.mode,
            disagreement_weight=self.disagreement_weight,
            min_consensus_gain=self.min_consensus_gain,
            primary_cost_margin=self.primary_cost_margin,
        )

        rows = torch.arange(plans.shape[0], device=self.device)
        chosen_primary = costs[0, rows, selected_idx]
        primary_improvement = costs[0, :, 0] - chosen_primary
        diversity = (plans - plans[:, :1]).float().square().mean(dim=(2, 3)).sqrt()
        best_votes = (costs.argmin(dim=2) == selected_idx.unsqueeze(0)).float().mean(dim=0)
        trace = []
        for row in range(plans.shape[0]):
            index = int(selected_idx[row])
            trace.append({
                "selected_restart": index,
                "selection_gain": float(details["gain"][row]),
                "primary_costs": costs[0, row].detach().cpu().tolist(),
                "consensus_scores": details["consensus_score"][row].detach().cpu().tolist(),
                "evaluator_best_restarts": costs[:, row].argmin(dim=1).detach().cpu().tolist(),
                "selected_vote_fraction": float(best_votes[row]),
                "action_rms_from_baseline": diversity[row].detach().cpu().tolist(),
            })
            self._diag.selected_counts[index] += 1

        decisions = plans.shape[0]
        self._diag.decisions += decisions
        self._diag.replacements += int((selected_idx != 0).sum())
        self._diag.gain_sum += float(details["gain"].sum())
        self._diag.primary_improvement_sum += float(primary_improvement.sum())
        self._diag.diversity_sum += float(diversity[:, 1:].mean(dim=1).sum())
        self._diag.agreement_sum += float(best_votes.sum())
        self.last_trace = trace

        outputs["actions"] = selected.detach().cpu()
        outputs["costs"] = chosen_primary.detach().cpu().tolist()
        outputs["mean"] = [outputs["actions"]]
        candidate_vars = torch.stack(
            [result["var"][0] for result in candidate_outputs], dim=1
        ).to(device=self.device, dtype=self.dtype)
        outputs["var"] = [candidate_vars[rows, selected_idx].detach().cpu()]
        outputs["portfolio"] = self.diagnostics()
        if self.trace:
            print(f"[PortfolioCEM trace] {trace}")
        return outputs

    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "restarts": self.restarts,
            "evaluator_count": len(self.evaluators),
            "evaluators": self.evaluator_names,
            "disagreement_weight": self.disagreement_weight,
            "min_consensus_gain": self.min_consensus_gain,
            "primary_cost_margin": self.primary_cost_margin,
            **self._diag.as_dict(),
        }
