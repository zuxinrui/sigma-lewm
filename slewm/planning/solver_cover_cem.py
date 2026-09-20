"""Safe final-elite IDM verification for CEM.

CoVer-VLA samples action chunks from a frozen policy and lets a separately
trained contrastive verifier choose among them.  LeWM already has a stronger
task-conditioned generator: terminal-cost CEM.  This module therefore leaves
all CEM iterations untouched and applies the available IDM only once, to the
final elite set.  Contrastive heads use normalized NCE energy; regression heads
use action-reconstruction residual.  Both are transition witnesses rather than
task-success verifiers, so the task-cost and confidence gates remain mandatory.

The literal CEM mean is always candidate zero.  A different elite may be
executed only when

1. its unchanged JEPA terminal cost is no worse than the CEM mean; and
2. its sequence contrastive energy improves by a configurable number of
   within-set standard deviations.

Rejection returns the original CEM tensor exactly.  Unlike Energy-ACID, the
verifier never enters the optimization cost and cannot steer all 30 CEM steps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from slewm.planning.energy_acid import contrastive_energy_from_keys
from slewm.planning.idm_decode import IDMDecoder
from stable_worldmodel.solver.cem import CEMSolver


def select_verified_plan(
    plans: torch.Tensor,
    terminal_cost: torch.Tensor,
    verifier_energy: torch.Tensor,
    *,
    cost_margin: float = 0.0,
    cost_margin_sigma: float = 0.0,
    verifier_margin_sigma: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a verifier-preferred plan without discarding candidate zero.

    ``plans[:, 0]`` is the literal CEM mean.  Lower costs and energies are
    better.  Returns selected plans, acceptance mask, selected indices, and
    standardized verifier gains over the CEM mean.
    """
    if plans.ndim != 4:
        raise ValueError(f"plans must be (B,S,H,A), got {plans.shape}")
    if terminal_cost.shape != plans.shape[:2] or verifier_energy.shape != plans.shape[:2]:
        raise ValueError(
            "terminal_cost and verifier_energy must match plans[:2], got "
            f"{terminal_cost.shape}, {verifier_energy.shape}, {plans.shape[:2]}"
        )

    base_cost = terminal_cost[:, :1]
    task_spread = terminal_cost.float().std(dim=1, unbiased=False).unsqueeze(1)
    allowed_cost = (
        base_cost.float()
        + float(cost_margin)
        + float(cost_margin_sigma) * task_spread
    )
    safe = torch.isfinite(terminal_cost) & (
        terminal_cost.float() <= allowed_cost
    )
    safe[:, 0] = True
    finite_energy = torch.isfinite(verifier_energy)
    masked_energy = torch.where(
        safe & finite_energy,
        verifier_energy,
        torch.full_like(verifier_energy, float("inf")),
    )
    best_idx = masked_energy.argmin(dim=1)
    rows = torch.arange(plans.shape[0], device=plans.device)
    chosen_energy = verifier_energy[rows, best_idx]
    spread = verifier_energy.float().std(dim=1, unbiased=False).clamp_min(float(eps))
    gain_sigma = (verifier_energy[:, 0].float() - chosen_energy.float()) / spread
    accept = (
        (best_idx != 0)
        & torch.isfinite(chosen_energy)
        & (gain_sigma >= float(verifier_margin_sigma))
    )
    selected_idx = torch.where(accept, best_idx, torch.zeros_like(best_idx))
    selected = plans[rows, selected_idx]
    return selected, accept, selected_idx, gain_sigma


def aggregate_step_energy(
    per_step: torch.Tensor, worst_step_weight: float
) -> torch.Tensor:
    """Combine typical and bottleneck transition disagreement along a plan."""
    if per_step.ndim != 3:
        raise ValueError(f"per_step must be (B,S,H), got {per_step.shape}")
    if not 0.0 <= worst_step_weight <= 1.0:
        raise ValueError("worst_step_weight must be in [0,1]")
    mean = per_step.mean(dim=-1)
    worst = per_step.max(dim=-1).values
    return (1.0 - worst_step_weight) * mean + worst_step_weight * worst


class _FinalEliteCapture:
    """Minimal CEM callback that retains only the last elite set per batch."""

    output_key = "cover_final_elites"

    def __init__(self) -> None:
        self.batches: list[torch.Tensor] = []
        self.all_batches: list[torch.Tensor] = []
        self._current: torch.Tensor | None = None
        self._current_all: torch.Tensor | None = None

    @property
    def history(self) -> list:
        # Do not duplicate potentially large GPU tensors in solver outputs.
        return []

    def reset(self) -> None:
        self.batches = []
        self.all_batches = []
        self._current = None
        self._current_all = None

    def start_batch(self) -> None:
        if self._current is not None:
            self.batches.append(self._current)
            assert self._current_all is not None
            self.all_batches.append(self._current_all)
        self._current = None
        self._current_all = None

    def __call__(self, **kwargs: Any) -> None:
        self._current = kwargs["topk_candidates"].detach().clone()
        self._current_all = kwargs["candidates"].detach().clone()

    def end_solve(self) -> None:
        if self._current is not None:
            self.batches.append(self._current)
            assert self._current_all is not None
            self.all_batches.append(self._current_all)
            self._current = None
            self._current_all = None


@dataclass
class _Diagnostics:
    solves: int = 0
    considered: int = 0
    accepted: int = 0
    gain_sum: float = 0.0
    base_cost_sum: float = 0.0
    selected_cost_sum: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        denom = max(self.considered, 1)
        return {
            "solves": self.solves,
            "plans_considered": self.considered,
            "plans_accepted": self.accepted,
            "accept_rate": self.accepted / denom,
            "mean_best_verifier_gain_sigma": self.gain_sum / denom,
            "mean_base_terminal_cost": self.base_cost_sum / denom,
            "mean_selected_terminal_cost": self.selected_cost_sum / denom,
        }


class SafeCoVerCEMSolver(CEMSolver):
    """Run stock CEM, then conservatively verify its final elites."""

    def __init__(
        self,
        *args,
        verifier: IDMDecoder | Any,
        tau: float = 0.1,
        cost_margin: float = 0.0,
        cost_margin_sigma: float = 0.0,
        verifier_margin_sigma: float = float("inf"),
        worst_step_weight: float = 0.5,
        candidate_source: str = "elites",
        proposal_alphas: tuple[float, ...] = (),
        trace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if verifier.mode not in (
            "contrastive", "regression", "forward_dcl", "joint_contrastive",
            "goal_contrastive", "physics_contrastive"
        ):
            raise ValueError(
                "SafeCoVerCEM requires a contrastive/regression IDM or forward DCL, "
                f"got {verifier.mode!r}"
            )
        if not 0.0 <= worst_step_weight <= 1.0:
            raise ValueError("worst_step_weight must be in [0,1]")
        self.verifier = verifier
        self.tau = float(tau)
        self.cost_margin = float(cost_margin)
        self.cost_margin_sigma = float(cost_margin_sigma)
        self.verifier_margin_sigma = float(verifier_margin_sigma)
        self.worst_step_weight = float(worst_step_weight)
        if candidate_source not in {"elites", "all"}:
            raise ValueError("candidate_source must be 'elites' or 'all'")
        self.candidate_source = candidate_source
        self.proposal_alphas = tuple(float(alpha) for alpha in proposal_alphas)
        if any(not 0.0 < alpha <= 1.0 for alpha in self.proposal_alphas):
            raise ValueError("proposal_alphas must lie in (0,1]")
        if self.proposal_alphas and verifier.mode != "goal_contrastive":
            raise ValueError("retrieval proposals require Goal-XIDM")
        self.trace = bool(trace)
        self.last_trace: list[dict[str, float | int | bool]] = []
        self._diag = _Diagnostics()

    def _expand_infos(self, info_dict: dict, start: int, end: int, samples: int) -> dict:
        expanded: dict[str, Any] = {}
        for key, value in info_dict.items():
            batch = value[start:end]
            if torch.is_tensor(value):
                dtype = self.dtype if batch.is_floating_point() else None
                batch = batch.to(device=self.device, dtype=dtype)
                expanded[key] = batch.unsqueeze(1).expand(
                    batch.shape[0], samples, *batch.shape[1:]
                )
            elif isinstance(value, np.ndarray):
                expanded[key] = np.repeat(batch[:, None, ...], samples, axis=1)
            else:
                expanded[key] = batch
        return expanded

    def _sequence_energy(
        self,
        predicted_chain: torch.Tensor,
        plans: torch.Tensor,
        goal_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, samples, horizon, action_dim = plans.shape
        chain = predicted_chain[:, :, -(horizon + 1) :, :]
        if chain.shape[2] != horizon + 1:
            raise ValueError(
                f"expected {horizon + 1} rollout latents, got {chain.shape}"
            )
        if self.verifier.mode == "physics_contrastive":
            if goal_emb is None:
                raise RuntimeError("physics critic requires goal_emb")
            terminal = chain[:, :, -1, :]
            goal = goal_emb[..., -1, :]
            if goal.ndim == 2:
                goal = goal[:, None, :].expand(batch, samples, -1)
            elif goal.ndim != 3:
                raise RuntimeError(f"unexpected physics goal shape {goal_emb.shape}")
            return self.verifier.energy(
                terminal.reshape(-1, self.verifier.D),
                goal.reshape(-1, self.verifier.D),
            ).reshape(batch, samples)
        z_t = chain[:, :, :-1].reshape(-1, self.verifier.D)
        z_next = chain[:, :, 1:].reshape(-1, self.verifier.D)
        actions = plans.reshape(-1, action_dim).float()
        if self.verifier.mode == "goal_contrastive":
            if goal_emb is None:
                raise RuntimeError("Goal-XIDM requires goal_emb from world-model cost")
            goal = goal_emb[..., -1, :]
            if goal.ndim == 2:
                goal = goal[:, None, None, :].expand(
                    batch, samples, horizon, -1
                )
            elif goal.ndim == 3:
                goal = goal[:, :, None, :].expand(
                    batch, samples, horizon, -1
                )
            else:
                raise RuntimeError(f"unexpected goal embedding shape {goal_emb.shape}")
            goal = goal.reshape(-1, self.verifier.D)
            remaining = torch.arange(
                horizon, 0, -1, device=plans.device, dtype=torch.long
            ).view(1, 1, horizon).expand(batch, samples, -1).reshape(-1)
            per_step = self.verifier.energy(z_t, goal, remaining, actions)
        elif self.verifier.mode == "joint_contrastive":
            # X-IDM and its matched post-hoc DCL control share this exact
            # empirical-action-bank normalized NCE energy.  The verifier owns
            # the CEM->training action-scaler conversion.
            per_step = self.verifier.energy(z_t, z_next, actions)
        elif self.verifier.mode == "contrastive":
            query = self.verifier.f(torch.cat([z_t, z_next], dim=-1))
            positive = self.verifier.g(actions)
            per_step = contrastive_energy_from_keys(
                query,
                positive,
                bank_keys=self.verifier.pool_g,
                tau=self.tau,
                mode="nce",
            )
        elif self.verifier.mode == "regression":
            # The regression head predicts the standardized macro action from
            # each imagined transition.  This is an ACID-style witness, used
            # only for final-set selection rather than as a CEM steering cost.
            per_step = self.verifier.score(z_t, z_next, actions)
        else:
            # C_S forward DCL: h_f([z_t,E_a(a_t)]) <-> h_g(z_{t+1}).
            # ForwardDCLVerifier applies a fixed successor-marginal partition
            # function, making scores comparable across changing (z_t,a_t)
            # queries before final-set aggregation.
            per_step = self.verifier.score(z_t, z_next, actions)
        return aggregate_step_energy(
            per_step.reshape(batch, samples, horizon), self.worst_step_weight
        )

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        capture = _FinalEliteCapture()
        self.callbacks.append(capture)
        try:
            outputs = super().solve(info_dict, init_action)
        finally:
            self.callbacks.remove(capture)
        outputs.get("callbacks", {}).pop(capture.output_key, None)
        verify_start = time.time()

        baseline = outputs["actions"].to(self.device, self.dtype)
        selected = baseline.clone()
        solve_trace: list[dict[str, float | int | bool]] = []
        total = baseline.shape[0]
        expected_batches = (total + self.batch_size - 1) // self.batch_size
        candidate_batches = (
            capture.all_batches if self.candidate_source == "all" else capture.batches
        )
        if len(candidate_batches) != expected_batches:
            raise RuntimeError(
                f"captured {len(candidate_batches)} CEM batches, expected {expected_batches}"
            )

        offset = 0
        for candidates in candidate_batches:
            bs = candidates.shape[0]
            plans = torch.cat([baseline[offset : offset + bs, None], candidates], dim=1)
            if self.proposal_alphas:
                # One extra rollout of the literal CEM mean supplies the
                # contexts for bounded Goal-XIDM action retrieval.  Convex
                # blends remain near CEM while expanding its final candidate
                # support in a data-backed, goal-directed direction.
                base_plan = baseline[offset : offset + bs]
                base_info = self._expand_infos(info_dict, offset, offset + bs, 1)
                self.model.get_cost(base_info, base_plan[:, None])
                base_chain = base_info["predicted_emb"][:, :, -(base_plan.shape[1] + 1) :, :]
                z_retrieve = base_chain[:, :, :-1].reshape(-1, self.verifier.D)
                goal = base_info["goal_emb"][..., -1, :]
                if goal.ndim == 2:
                    goal = goal[:, None, None, :].expand(bs, 1, base_plan.shape[1], -1)
                elif goal.ndim == 3:
                    goal = goal[:, :, None, :].expand(bs, 1, base_plan.shape[1], -1)
                else:
                    raise RuntimeError(
                        f"unexpected retrieval goal embedding shape {base_info['goal_emb'].shape}"
                    )
                remaining = torch.arange(
                    base_plan.shape[1], 0, -1,
                    device=base_plan.device, dtype=torch.long,
                ).view(1, 1, -1).expand(bs, 1, -1).reshape(-1)
                retrieved = self.verifier.retrieve_action(
                    z_retrieve, goal.reshape(-1, self.verifier.D), remaining
                ).reshape_as(base_plan).to(base_plan.dtype)
                proposals = torch.stack(
                    [base_plan.lerp(retrieved, alpha) for alpha in self.proposal_alphas],
                    dim=1,
                )
                plans = torch.cat([plans, proposals], dim=1)
            expanded = self._expand_infos(
                info_dict, offset, offset + bs, plans.shape[1]
            )
            task_cost = getattr(self.model, "get_task_cost", self.model.get_cost)
            terminal = task_cost(expanded, plans)
            if "predicted_emb" not in expanded:
                raise RuntimeError("world model get_cost did not expose predicted_emb")
            energy = self._sequence_energy(
                expanded["predicted_emb"], plans, expanded.get("goal_emb")
            )
            chosen, accept, indices, gain = select_verified_plan(
                plans,
                terminal,
                energy,
                cost_margin=self.cost_margin,
                cost_margin_sigma=self.cost_margin_sigma,
                verifier_margin_sigma=self.verifier_margin_sigma,
            )
            selected[offset : offset + bs] = chosen
            rows = torch.arange(bs, device=self.device)
            chosen_cost = terminal[rows, indices]
            for row in range(bs):
                solve_trace.append({
                    "base_energy": float(energy[row, 0]),
                    "best_energy": float(energy[row].min()),
                    "gain_sigma": float(gain[row]),
                    "selected_index": int(indices[row]),
                    "accepted": bool(accept[row]),
                    "base_cost": float(terminal[row, 0]),
                    "selected_cost": float(chosen_cost[row]),
                })
            self._diag.considered += bs
            self._diag.accepted += int(accept.sum())
            self._diag.gain_sum += float(gain.sum())
            self._diag.base_cost_sum += float(terminal[:, 0].sum())
            self._diag.selected_cost_sum += float(chosen_cost.sum())
            offset += bs

        self._diag.solves += 1
        self.last_trace = solve_trace
        outputs["actions"] = selected.detach().cpu()
        outputs["mean"] = [outputs["actions"]]
        outputs["cover"] = self.diagnostics()
        print(
            "SafeCoVer-CEM verify time: "
            f"{time.time() - verify_start:.4f}s "
            f"(accepted={self._diag.accepted}/{self._diag.considered}, "
            f"margin_sigma={self.verifier_margin_sigma:g}, "
            f"cost_margin={self.cost_margin:g})"
        )
        if self.trace:
            print(f"[SafeCoVer-CEM trace] {solve_trace}")
        return outputs

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "verifier_mode": self.verifier.mode,
            "tau": self.tau,
            "cost_margin": self.cost_margin,
            "cost_margin_sigma": self.cost_margin_sigma,
            "verifier_margin_sigma": self.verifier_margin_sigma,
            "worst_step_weight": self.worst_step_weight,
            "candidate_source": self.candidate_source,
            "proposal_alphas": self.proposal_alphas,
            "candidate_count": (
                self.num_samples + 1 if self.candidate_source == "all" else self.topk + 1
            ) + len(self.proposal_alphas),
            **self._diag.as_dict(),
        }
