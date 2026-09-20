"""DCL inverse-retrieval proposals inside otherwise standard CEM.

The contrastive IDM in ``C_T_dcl_idm_sigreg`` is a set-valued inverse model:

    score(a | z, z_next) = cos(f([z, z_next]), g(a)).

Previous experiments used this score as a global planning penalty.  That is a
type error for control: transition compatibility is not task success, and the
penalty steered every CEM update away from the already calibrated terminal
goal objective.

This solver uses the IDM only in its trained direction.  At each CEM
iteration, it takes one world-model trajectory already present in the vanilla
population, treats its imagined successor latents as local waypoints, retrieves
multiple concrete action modes from the empirical action bank, and closes each
proposal through the frozen forward model.  A small beam keeps the inverse
solutions that best track the local waypoint chain.  The resulting plans are
then added as *extra* candidates and are judged exclusively by the unchanged
world-model goal cost.

Important invariants:

* all ``num_samples`` vanilla CEM candidates are retained;
* proposal generation has a separate deterministic path and consumes no CEM
  RNG state;
* no IDM energy is added to the task objective;
* ``proposal_beams=0`` dispatches literally to the parent CEM implementation.

This is therefore a defensive test-time-scaling arm, not a claim that an IDM
score is a calibrated success probability.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from stable_worldmodel.solver.cem import CEMSolver
from stable_worldmodel.solver.utils import prepare_init_action


def _gather_beam(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``value`` on beam dimension 1 with one index set per batch."""
    rows = torch.arange(value.shape[0], device=value.device)[:, None]
    return value[rows, indices]


class DCLBeamCEMSolver(CEMSolver):
    """CEM augmented with forward-closed, multi-modal DCL proposals."""

    def __init__(
        self,
        *args,
        decoder: Any,
        proposal_beams: int = 4,
        branch_topk: int = 8,
        proposal_start: int = 0,
        proposal_every: int = 1,
        target_source: str = "best",
        proposal_trust_rms: float = 0.0,
        source_chains: int = 1,
        source_pool: int = 30,
        retrieval_preselect: int = 0,
        retrieval_diversity: float = 0.0,
        proposal_mode: str = "reconstruct",
        output_mode: str = "mean",
        search_mode: str = "single",
        mixture_components: int = 4,
        mixture_initial_scale: float = 0.5,
        mixture_scale_floor: float = 0.05,
        mixture_elites: int = 0,
        history_size: int = 3,
        trace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if getattr(decoder, "mode", None) != "contrastive":
            raise ValueError(
                "DCLBeamCEMSolver needs the inverse contrastive IDM from "
                "C_T_dcl_idm_sigreg"
            )
        if proposal_beams < 0:
            raise ValueError("proposal_beams must be non-negative")
        if branch_topk <= 0:
            raise ValueError("branch_topk must be positive")
        if proposal_every <= 0:
            raise ValueError("proposal_every must be positive")
        if target_source not in {"best", "mean"}:
            raise ValueError("target_source must be 'best' or 'mean'")
        if proposal_trust_rms < 0:
            raise ValueError("proposal_trust_rms must be non-negative")
        if source_chains <= 0:
            raise ValueError("source_chains must be positive")
        if source_pool <= 0:
            raise ValueError("source_pool must be positive")
        if retrieval_preselect < 0:
            raise ValueError("retrieval_preselect must be non-negative")
        if not 0 <= retrieval_diversity <= 1:
            raise ValueError("retrieval_diversity must lie in [0,1]")
        if proposal_mode not in {"reconstruct", "repair", "stitch"}:
            raise ValueError(
                "proposal_mode must be 'reconstruct', 'repair', or 'stitch'"
            )
        if output_mode not in {"mean", "best_seen"}:
            raise ValueError("output_mode must be 'mean' or 'best_seen'")
        if search_mode not in {"single", "mixture"}:
            raise ValueError("search_mode must be 'single' or 'mixture'")
        if mixture_components <= 0:
            raise ValueError("mixture_components must be positive")
        if mixture_initial_scale <= 0:
            raise ValueError("mixture_initial_scale must be positive")
        if mixture_scale_floor < 0:
            raise ValueError("mixture_scale_floor must be non-negative")
        if mixture_elites < 0:
            raise ValueError("mixture_elites must be non-negative")
        self.decoder = decoder
        self.proposal_beams = int(proposal_beams)
        self.branch_topk = int(branch_topk)
        self.proposal_start = int(proposal_start)
        self.proposal_every = int(proposal_every)
        self.target_source = str(target_source)
        self.proposal_trust_rms = float(proposal_trust_rms)
        self.source_chains = int(source_chains)
        self.source_pool = int(source_pool)
        self.retrieval_preselect = int(retrieval_preselect)
        self.retrieval_diversity = float(retrieval_diversity)
        self.proposal_mode = str(proposal_mode)
        self.output_mode = str(output_mode)
        self.search_mode = str(search_mode)
        self.mixture_components = int(mixture_components)
        self.mixture_initial_scale = float(mixture_initial_scale)
        self.mixture_scale_floor = float(mixture_scale_floor)
        self.mixture_elites = int(mixture_elites)
        self.history_size = int(history_size)
        self.trace = bool(trace)
        self._diag = {
            "solves": 0,
            "proposal_candidates": 0,
            "proposal_elites": 0,
            "iterations_with_proposal_elite": 0,
            "trust_clipped": 0,
            "repair_candidates_evaluated": 0,
            "mixture_candidates": 0,
        }

    def _trust_project(
        self,
        proposals: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Temper inverse proposals inside the current CEM trust region.

        The diagonal CEM distribution supplies the local metric.  If a plan is
        farther than ``proposal_trust_rms`` in standardized RMS distance, it
        is interpolated back toward the CEM mean.  A zero radius disables this
        operation for exact reproduction of the original DCL-beam arm.
        """
        if self.proposal_trust_rms <= 0:
            alpha = torch.ones(
                proposals.shape[:2], device=proposals.device, dtype=proposals.dtype
            )
            return proposals, alpha

        delta = proposals - mean[:, None]
        normalized = delta / scale[:, None].clamp_min(1e-3)
        rms = normalized.float().pow(2).mean(dim=(-2, -1)).sqrt()
        alpha = (self.proposal_trust_rms / rms.clamp_min(1e-8)).clamp(max=1.0)
        projected = mean[:, None] + alpha.to(delta.dtype)[..., None, None] * delta
        return projected, alpha

    def _expand_infos(
        self, info_dict: dict, start: int, end: int, samples: int
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

    @torch.inference_mode()
    def _project_path(
        self, target_chain: torch.Tensor, beam_limit: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert an imagined latent chain with beam-search and forward closure.

        Args:
            target_chain: ``(B,H+1,D)`` trajectory selected from vanilla CEM.

        Returns:
            plans: ``(B,P,H,A)`` concrete action-bank proposals.
            tracking_cost: ``(B,P)`` cumulative local waypoint error.
        """
        batch, chain_len, latent_dim = target_chain.shape
        horizon = chain_len - 1
        if horizon != self.horizon:
            raise ValueError(
                f"target chain horizon {horizon} != solver horizon {self.horizon}"
            )

        # One initial beam starts from the exact latent exposed by the rollout.
        emb_hist = target_chain[:, None, :1, :].contiguous()  # (B,1,1,D)
        act_hist = torch.empty(
            batch,
            1,
            0,
            self.action_dim,
            device=target_chain.device,
            dtype=self.dtype,
        )
        cumulative = torch.zeros(
            batch, 1, device=target_chain.device, dtype=torch.float32
        )

        for step in range(horizon):
            beams = emb_hist.shape[1]
            current = emb_hist[:, :, -1, :]
            target = target_chain[:, step + 1, :]
            flat_current = current.reshape(batch * beams, latent_dim)
            flat_target = target[:, None, :].expand(
                batch, beams, latent_dim
            ).reshape(batch * beams, latent_dim)
            retrieval_kwargs = {}
            if self.retrieval_diversity > 0 and self.retrieval_preselect > 0:
                retrieval_kwargs = {
                    "preselect": self.retrieval_preselect,
                    "diversity_weight": self.retrieval_diversity,
                }
            hypotheses, _ = self.decoder.candidates(
                flat_current,
                flat_target,
                topk=self.branch_topk,
                **retrieval_kwargs,
            )
            branches = hypotheses.shape[1]
            hypotheses = hypotheses.reshape(
                batch, beams, branches, self.action_dim
            ).to(dtype=self.dtype)

            # Expand each live beam with every retrieved concrete action mode.
            expanded_emb = emb_hist[:, :, None].expand(
                batch, beams, branches, emb_hist.shape[2], latent_dim
            )
            expanded_act = act_hist[:, :, None].expand(
                batch, beams, branches, act_hist.shape[2], self.action_dim
            )
            next_act = torch.cat(
                [expanded_act, hypotheses.unsqueeze(3)], dim=3
            )

            flat_emb = expanded_emb.reshape(
                batch * beams * branches, emb_hist.shape[2], latent_dim
            )
            flat_act = next_act.reshape(
                batch * beams * branches, next_act.shape[3], self.action_dim
            )
            act_emb = self.model.action_encoder(flat_act)[:, -self.history_size :]
            pred = self.model.predict(
                flat_emb[:, -self.history_size :], act_emb
            )[:, -1]
            pred = pred.reshape(batch, beams, branches, latent_dim)

            step_error = (
                pred.float() - target[:, None, None, :].float()
            ).pow(2).sum(dim=-1)
            total = cumulative[:, :, None] + step_error
            flat_total = total.reshape(batch, beams * branches)
            keep = min(
                self.proposal_beams if beam_limit is None else int(beam_limit),
                flat_total.shape[1],
            )
            cumulative, chosen = torch.topk(
                flat_total, k=keep, dim=1, largest=False
            )

            candidate_act = next_act.reshape(
                batch, beams * branches, next_act.shape[3], self.action_dim
            )
            act_hist = _gather_beam(candidate_act, chosen)

            candidate_emb = torch.cat(
                [
                    expanded_emb,
                    pred.unsqueeze(3),
                ],
                dim=3,
            ).reshape(
                batch,
                beams * branches,
                emb_hist.shape[2] + 1,
                latent_dim,
            )
            emb_hist = _gather_beam(candidate_emb, chosen)

        return act_hist, cumulative

    def _select_source_indices(
        self, candidates: torch.Tensor, costs: torch.Tensor
    ) -> torch.Tensor:
        """Pick low-cost but action-diverse CEM source trajectories."""
        count = min(self.source_chains, candidates.shape[1])
        if count == 1:
            return costs.argmin(dim=1, keepdim=True)
        pool = min(max(self.source_pool, count), candidates.shape[1])
        _, ranked = torch.topk(costs, k=pool, dim=1, largest=False)
        rows = torch.arange(candidates.shape[0], device=candidates.device)
        plans = candidates[rows[:, None], ranked]
        flat = plans.float().flatten(2)
        distance = torch.cdist(flat, flat) / max(float(flat.shape[-1]) ** 0.5, 1.0)
        chosen = torch.zeros(
            candidates.shape[0], count, device=candidates.device, dtype=torch.long
        )
        used = torch.zeros(
            candidates.shape[0], pool, device=candidates.device, dtype=torch.bool
        )
        used[:, 0] = True
        min_distance = distance[:, 0]
        for slot in range(1, count):
            score = min_distance.masked_fill(used, float("-inf"))
            pick = score.argmax(dim=-1)
            chosen[:, slot] = pick
            used[rows, pick] = True
            min_distance = torch.minimum(min_distance, distance[rows, pick])
        return ranked[rows[:, None], chosen]

    @torch.inference_mode()
    def _project_stitched_paths(
        self, target_chains: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-close a beam that may switch source chain at every step.

        A fixed source chain can contain a strong approach but a poor contact
        suffix (or vice versa).  This search treats every source's next
        waypoint as an option from the beam's *actual predicted state*, so a
        plan can combine stages without ever splicing latent states directly.
        At least one live beam per current source is retained when capacity
        permits; this prevents one route from occupying the whole beam before
        later stages are considered.
        """
        if target_chains.ndim != 4:
            raise ValueError("stitched target chains must be (B,S,H+1,D)")
        batch, sources, chain_len, latent_dim = target_chains.shape
        horizon = chain_len - 1
        if sources < 2:
            raise ValueError("stitched projection needs at least two source chains")
        if horizon != self.horizon:
            raise ValueError("stitched target horizon does not match solver")

        emb_hist = target_chains[:, :1, :1].contiguous()
        act_hist = torch.empty(
            batch, 1, 0, self.action_dim,
            device=target_chains.device, dtype=self.dtype,
        )
        cumulative = torch.zeros(
            batch, 1, device=target_chains.device, dtype=torch.float32
        )
        retrieval_kwargs = {}
        if self.retrieval_diversity > 0 and self.retrieval_preselect > 0:
            retrieval_kwargs = {
                "preselect": self.retrieval_preselect,
                "diversity_weight": self.retrieval_diversity,
            }

        for step in range(horizon):
            beams = emb_hist.shape[1]
            current = emb_hist[:, :, -1]
            targets = target_chains[:, :, step + 1]
            flat_current = current[:, :, None].expand(
                batch, beams, sources, latent_dim
            ).reshape(batch * beams * sources, latent_dim)
            flat_target = targets[:, None].expand(
                batch, beams, sources, latent_dim
            ).reshape(batch * beams * sources, latent_dim)
            hypotheses, _ = self.decoder.candidates(
                flat_current,
                flat_target,
                topk=self.branch_topk,
                **retrieval_kwargs,
            )
            branches = hypotheses.shape[1]
            hypotheses = hypotheses.reshape(
                batch, beams, sources, branches, self.action_dim
            ).to(dtype=self.dtype)

            expanded_emb = emb_hist[:, :, None, None].expand(
                batch, beams, sources, branches, emb_hist.shape[2], latent_dim
            )
            expanded_act = act_hist[:, :, None, None].expand(
                batch, beams, sources, branches, act_hist.shape[2], self.action_dim
            )
            next_act = torch.cat(
                [expanded_act, hypotheses.unsqueeze(4)], dim=4
            )
            flat_emb = expanded_emb.reshape(
                batch * beams * sources * branches,
                emb_hist.shape[2],
                latent_dim,
            )
            flat_act = next_act.reshape(
                batch * beams * sources * branches,
                next_act.shape[4],
                self.action_dim,
            )
            act_emb = self.model.action_encoder(flat_act)[:, -self.history_size :]
            pred = self.model.predict(
                flat_emb[:, -self.history_size :], act_emb
            )[:, -1].reshape(batch, beams, sources, branches, latent_dim)
            step_error = (
                pred.float() - targets[:, None, :, None].float()
            ).pow(2).sum(dim=-1)
            total = cumulative[:, :, None, None] + step_error
            flat_total = total.reshape(batch, beams * sources * branches)
            keep = min(self.proposal_beams, flat_total.shape[1])

            # Reserve the best current branch for each source, then fill the
            # remaining capacity globally by tracking error.
            source_label = torch.arange(
                sources, device=self.device
            )[None, None, :, None].expand(
                batch, beams, sources, branches
            ).reshape(batch, -1)
            chosen_parts = []
            reserved = min(sources, keep)
            used = torch.zeros_like(flat_total, dtype=torch.bool)
            rows = torch.arange(batch, device=self.device)
            for source in range(reserved):
                source_cost = flat_total.masked_fill(
                    source_label != source, float("inf")
                )
                pick = source_cost.argmin(dim=1)
                chosen_parts.append(pick[:, None])
                used[rows, pick] = True
            if keep > reserved:
                remaining = flat_total.masked_fill(used, float("inf"))
                _, extra = torch.topk(
                    remaining, k=keep - reserved, dim=1, largest=False
                )
                chosen_parts.append(extra)
            chosen = torch.cat(chosen_parts, dim=1)
            cumulative = flat_total[rows[:, None], chosen]

            candidate_act = next_act.reshape(
                batch, beams * sources * branches,
                next_act.shape[4], self.action_dim,
            )
            act_hist = _gather_beam(candidate_act, chosen)
            candidate_emb = torch.cat(
                [expanded_emb, pred.unsqueeze(4)], dim=4
            ).reshape(
                batch, beams * sources * branches,
                emb_hist.shape[2] + 1, latent_dim,
            )
            emb_hist = _gather_beam(candidate_emb, chosen)

        order = cumulative.argsort(dim=1)
        return _gather_beam(act_hist, order), _gather_beam(cumulative, order)

    @torch.inference_mode()
    def _single_site_repairs(
        self,
        source_plans: torch.Tensor,
        target_chains: torch.Tensor,
    ) -> torch.Tensor:
        """Replace exactly one source macro-action with a retrieved mode.

        Whole-chain inverse reconstruction is useful for opening a new
        discrete basin, but it also quantizes every action in a CEM trajectory.
        Single-site repair is the complementary large-neighborhood move: keep
        H-1 continuous CEM actions intact and try bank-supported alternatives
        at the remaining site.  The unchanged task cost ranks the repairs.

        Args:
            source_plans: ``(B,S,H,A)`` vanilla CEM source trajectories.
            target_chains: ``(B,S,H+1,D)`` their imagined latent chains.

        Returns:
            ``(B,S,H*K,H,A)`` repair plans, grouped by source chain.
        """
        if source_plans.ndim != 4 or target_chains.ndim != 4:
            raise ValueError("repair sources and chains must both be rank four")
        batch, sources, horizon, action_dim = source_plans.shape
        if horizon != self.horizon or action_dim != self.action_dim:
            raise ValueError("repair source shape does not match configured solver")
        if target_chains.shape[:3] != (batch, sources, horizon + 1):
            raise ValueError("repair target chain must be (B,S,H+1,D)")

        latent_dim = target_chains.shape[-1]
        z = target_chains[:, :, :-1].reshape(batch * sources * horizon, latent_dim)
        z_next = target_chains[:, :, 1:].reshape(
            batch * sources * horizon, latent_dim
        )
        retrieval_kwargs = {}
        if self.retrieval_diversity > 0 and self.retrieval_preselect > 0:
            retrieval_kwargs = {
                "preselect": self.retrieval_preselect,
                "diversity_weight": self.retrieval_diversity,
            }
        actions, _ = self.decoder.candidates(
            z, z_next, topk=self.branch_topk, **retrieval_kwargs
        )
        branches = actions.shape[1]
        actions = actions.reshape(
            batch, sources, horizon, branches, action_dim
        ).to(dtype=self.dtype)
        repairs = source_plans[:, :, None, None].expand(
            batch, sources, horizon, branches, horizon, action_dim
        ).clone()
        for step in range(horizon):
            repairs[:, :, step, :, step] = actions[:, :, step]
        return repairs.reshape(
            batch, sources, horizon * branches, horizon, action_dim
        )

    def _select_repairs(
        self,
        info_dict: dict,
        start_idx: int,
        end_idx: int,
        repairs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Use only the original task cost to retain repairs from each source."""
        batch, sources, per_source_candidates, horizon, action_dim = repairs.shape
        flat = repairs.reshape(
            batch, sources * per_source_candidates, horizon, action_dim
        )
        repair_info = self._expand_infos(
            info_dict, start_idx, end_idx, flat.shape[1]
        )
        repair_cost = self.model.get_cost(repair_info, flat).reshape(
            batch, sources, per_source_candidates
        )
        keep_per_source = max(1, self.proposal_beams // sources)
        keep_per_source = min(keep_per_source, per_source_candidates)
        selected_cost, indices = torch.topk(
            repair_cost, k=keep_per_source, dim=2, largest=False
        )
        rows = torch.arange(batch, device=repairs.device)[:, None, None]
        source_rows = torch.arange(sources, device=repairs.device)[None, :, None]
        selected = repairs[rows, source_rows, indices].reshape(
            batch, sources * keep_per_source, horizon, action_dim
        )[:, : self.proposal_beams]
        selected_cost = selected_cost.reshape(
            batch, sources * keep_per_source
        )[:, : self.proposal_beams]
        return selected, selected_cost, int(batch * sources * per_source_candidates)

    @staticmethod
    def _round_robin_mode_indices(
        *, sources: int, proposals_per_source: int, count: int, device: torch.device
    ) -> torch.Tensor:
        """Retain early beams from every source before taking deeper beams.

        ``_project_path`` returns source-major proposals.  A plain prefix would
        therefore silently discard every later source whenever the mixture has
        fewer components than proposals.  Round-robin selection makes source
        preservation an explicit invariant rather than an accident of task
        cost ordering.
        """
        if sources <= 0 or proposals_per_source <= 0 or count <= 0:
            raise ValueError("mode index dimensions must be positive")
        count = min(int(count), int(sources * proposals_per_source))
        indices = []
        for rank in range(proposals_per_source):
            for source in range(sources):
                indices.append(source * proposals_per_source + rank)
                if len(indices) == count:
                    return torch.tensor(indices, device=device, dtype=torch.long)
        raise RuntimeError("round-robin mode selection produced too few indices")

    @torch.inference_mode()
    def _solve_mixture(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """Run one local CEM distribution per retrieved DCL action mode.

        The original DCL arm inserts a handful of discrete plans into one
        Gaussian and then averages all elites.  That is safe, but a contact
        task can have several incompatible basins whose average is not itself
        a useful action.  This experimental path spends the first ordinary CEM
        population on discovering waypoint chains, retrieves concrete DCL
        plans, and then gives each retained plan its own local distribution.

        The total stochastic population remains ``num_samples`` per iteration;
        it is partitioned evenly across components.  No IDM score enters the
        task objective, and the caller's separate Stock/base-DCL arms remain
        untouched.
        """
        if self.proposal_mode != "reconstruct":
            raise ValueError("mixture search currently requires reconstruct proposals")
        if self.proposal_beams < self.mixture_components:
            raise ValueError(
                "proposal_beams must be at least mixture_components in mixture mode"
            )

        start_time = time.time()
        total_envs = len(next(iter(info_dict.values())))
        init_action = prepare_init_action(
            self.model,
            info_dict,
            init_action,
            self.horizon,
            n_envs=total_envs,
            action_dim=self.action_dim,
        )
        global_mean, global_scale = self.init_action_distrib(total_envs, init_action)
        global_mean = global_mean.to(self.device)
        global_scale = global_scale.to(self.device)

        returned = torch.empty_like(global_mean)
        returned_mean = torch.empty_like(global_mean)
        returned_scale = torch.empty_like(global_scale)
        returned_costs = torch.empty(
            total_envs, device=self.device, dtype=torch.float32
        )
        archive_actions = torch.empty_like(global_mean)
        archive_costs = torch.empty_like(returned_costs)
        solve_candidates = 0
        solve_components = 0

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            batch = end_idx - start_idx
            batch_mean = global_mean[start_idx:end_idx]
            batch_scale = global_scale[start_idx:end_idx]

            # Discovery is a literal ordinary CEM population.  Apart from the
            # component path that follows it, this draw has the same shape and
            # seed contract as the stock/DCL solvers.
            discovery = torch.randn(
                batch,
                self.num_samples,
                self.horizon,
                self.action_dim,
                generator=self.torch_gen,
                device=self.device,
                dtype=self.dtype,
            )
            discovery = discovery * batch_scale[:, None] + batch_mean[:, None]
            discovery[:, 0] = batch_mean
            discovery_info = self._expand_infos(
                info_dict, start_idx, end_idx, self.num_samples
            )
            discovery_cost = self.model.get_cost(discovery_info, discovery).float()
            source_idx = self._select_source_indices(discovery, discovery_cost)
            rows = torch.arange(batch, device=self.device)
            target_chain = discovery_info["predicted_emb"][
                rows[:, None], source_idx, -(self.horizon + 1) :
            ]
            sources = target_chain.shape[1]
            per_source = max(1, self.proposal_beams // sources)
            flat_chain = target_chain.reshape(
                batch * sources, self.horizon + 1, target_chain.shape[-1]
            )
            proposals, _ = self._project_path(flat_chain, beam_limit=per_source)
            proposals = proposals.reshape(
                batch,
                sources * proposals.shape[1],
                self.horizon,
                self.action_dim,
            )
            actual_per_source = proposals.shape[1] // sources
            component_count = min(self.mixture_components, proposals.shape[1])
            if self.num_samples % component_count != 0:
                raise ValueError(
                    "num_samples must be divisible by the realized mixture component count"
                )
            mode_indices = self._round_robin_mode_indices(
                sources=sources,
                proposals_per_source=actual_per_source,
                count=component_count,
                device=self.device,
            )
            component_mean = proposals[:, mode_indices].contiguous()

            # The discovery elite spread supplies a task-local unit of scale;
            # each discrete proposal receives the same local radius but then
            # evolves independently.
            discovery_topk = min(self.topk, self.num_samples)
            _, elite_idx = torch.topk(
                discovery_cost, k=discovery_topk, dim=1, largest=False
            )
            discovery_elites = discovery[rows[:, None], elite_idx]
            local_scale = discovery_elites.std(dim=1, unbiased=False).clamp_min(
                self.mixture_scale_floor
            )
            component_scale = (
                self.mixture_initial_scale * local_scale[:, None]
            ).clamp_min(self.mixture_scale_floor).expand(
                batch,
                component_count,
                self.horizon,
                self.action_dim,
            ).clone()

            proposal_info = self._expand_infos(
                info_dict, start_idx, end_idx, proposals.shape[1]
            )
            proposal_cost = self.model.get_cost(proposal_info, proposals).float()
            all_initial = torch.cat([discovery, proposals], dim=1)
            all_initial_cost = torch.cat([discovery_cost, proposal_cost], dim=1)
            best_cost, best_index = all_initial_cost.min(dim=1)
            best_plan = all_initial[rows, best_index].clone()
            # A discovery winner has the global elite scale; a proposal winner
            # starts with the scale of its closest retained component.
            best_scale = local_scale.clone()
            proposal_winner = best_index >= self.num_samples
            if proposal_winner.any():
                winning_proposal = (
                    best_index[proposal_winner] - self.num_samples
                )
                selected_plans = proposals[proposal_winner, winning_proposal]
                centers = component_mean[proposal_winner]
                distance = (centers - selected_plans[:, None]).float().square().mean(
                    dim=(-2, -1)
                )
                closest = distance.argmin(dim=1)
                best_scale[proposal_winner] = component_scale[
                    proposal_winner, closest
                ]

            samples_per_component = self.num_samples // component_count
            elites_per_component = self.mixture_elites or max(
                1, int(np.ceil(self.topk / component_count))
            )
            elites_per_component = min(elites_per_component, samples_per_component)

            for _ in range(1, self.n_steps):
                candidates = torch.randn(
                    batch,
                    component_count,
                    samples_per_component,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                candidates = (
                    candidates * component_scale[:, :, None]
                    + component_mean[:, :, None]
                )
                candidates[:, :, 0] = component_mean
                flat_candidates = candidates.reshape(
                    batch,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                )
                candidate_info = self._expand_infos(
                    info_dict, start_idx, end_idx, self.num_samples
                )
                flat_cost = self.model.get_cost(
                    candidate_info, flat_candidates
                ).float()
                candidate_cost = flat_cost.reshape(
                    batch, component_count, samples_per_component
                )
                elite_cost, elite_index = torch.topk(
                    candidate_cost,
                    k=elites_per_component,
                    dim=2,
                    largest=False,
                )
                batch_rows = torch.arange(batch, device=self.device)[:, None, None]
                component_rows = torch.arange(
                    component_count, device=self.device
                )[None, :, None]
                elite = candidates[batch_rows, component_rows, elite_index]
                component_mean = elite.mean(dim=2)
                component_scale = elite.std(dim=2, unbiased=False).clamp_min(
                    self.mixture_scale_floor
                )

                iteration_cost, iteration_index = flat_cost.min(dim=1)
                improve = iteration_cost < best_cost
                if improve.any():
                    iteration_plan = flat_candidates[rows, iteration_index]
                    iteration_component = torch.div(
                        iteration_index,
                        samples_per_component,
                        rounding_mode="floor",
                    )
                    best_cost = torch.where(improve, iteration_cost, best_cost)
                    best_plan[improve] = iteration_plan[improve]
                    best_scale[improve] = component_scale[
                        rows[improve], iteration_component[improve]
                    ]
                solve_candidates += batch * self.num_samples

            # Never average across modes.  Choose the best concrete component
            # mean, optionally comparing it with the best sampled plan seen at
            # any stage under identical task-cost calls.
            mean_info = self._expand_infos(
                info_dict, start_idx, end_idx, component_count
            )
            mean_cost = self.model.get_cost(mean_info, component_mean).float()
            best_mean_cost, best_mean_index = mean_cost.min(dim=1)
            best_component_mean = component_mean[rows, best_mean_index]
            best_component_scale = component_scale[rows, best_mean_index]
            if self.output_mode == "best_seen":
                use_archive = best_cost < best_mean_cost
                batch_returned = torch.where(
                    use_archive[:, None, None], best_plan, best_component_mean
                )
                batch_returned_cost = torch.where(
                    use_archive, best_cost, best_mean_cost
                )
                batch_returned_scale = torch.where(
                    use_archive[:, None, None], best_scale, best_component_scale
                )
            else:
                batch_returned = best_component_mean
                batch_returned_cost = best_mean_cost
                batch_returned_scale = best_component_scale

            returned[start_idx:end_idx] = batch_returned
            returned_mean[start_idx:end_idx] = best_component_mean
            returned_scale[start_idx:end_idx] = batch_returned_scale
            returned_costs[start_idx:end_idx] = batch_returned_cost
            archive_actions[start_idx:end_idx] = best_plan
            archive_costs[start_idx:end_idx] = best_cost
            solve_components += batch * component_count
            solve_candidates += batch * (
                self.num_samples + proposals.shape[1]
            )

        self._diag["solves"] += 1
        self._diag["mixture_candidates"] += solve_candidates
        outputs = {
            "actions": returned.detach().cpu(),
            "costs": returned_costs.detach().cpu().tolist(),
            "mean": [returned_mean.detach().cpu()],
            "var": [returned_scale.detach().cpu()],
            "best_seen_actions": archive_actions.detach().cpu(),
            "best_seen_costs": archive_costs.detach().cpu().tolist(),
            "dcl_beam": {
                **self._diag,
                "search_mode": "mixture",
                "output_mode": self.output_mode,
                "solve_components": solve_components,
                "solve_mixture_candidates": solve_candidates,
                "mixture_components": self.mixture_components,
                "mixture_initial_scale": self.mixture_initial_scale,
                "mixture_scale_floor": self.mixture_scale_floor,
                "mixture_elites": self.mixture_elites,
            },
        }
        if self.trace:
            print(
                "DCL-Mixture-CEM "
                f"{time.time() - start_time:.3f}s components={solve_components} "
                f"candidates={solve_candidates}",
                flush=True,
            )
        return outputs

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        # This is the exact no-feature control and must retain parent RNG/output.
        if self.proposal_beams == 0:
            return super().solve(info_dict, init_action)
        if self.search_mode == "mixture":
            return self._solve_mixture(info_dict, init_action)

        start_time = time.time()
        outputs = {"costs": [], "mean": [], "var": []}
        total_envs = len(next(iter(info_dict.values())))
        init_action = prepare_init_action(
            self.model,
            info_dict,
            init_action,
            self.horizon,
            n_envs=total_envs,
            action_dim=self.action_dim,
        )
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean, var = mean.to(self.device), var.to(self.device)

        solve_proposals = 0
        solve_proposal_elites = 0
        solve_iterations_hit = 0
        solve_trust_clipped = 0
        solve_trust_alpha_sum = 0.0
        solve_trust_alpha_count = 0
        solve_repair_candidates = 0
        returned_actions = torch.empty_like(mean)
        best_seen_actions = torch.empty_like(mean)
        best_seen_costs = torch.empty(
            total_envs, device=self.device, dtype=torch.float32
        )

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            final_batch_cost = None
            best_seen_cost = torch.full(
                (current_bs,), float("inf"), device=self.device, dtype=torch.float32
            )
            best_seen_plan = batch_mean.detach().clone()

            for step in range(self.n_steps):
                # Literal vanilla population: proposal code does not consume this RNG.
                candidates = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                candidates = (
                    candidates * batch_var.unsqueeze(1)
                    + batch_mean.unsqueeze(1)
                )
                candidates[:, 0] = batch_mean

                vanilla_info = self._expand_infos(
                    info_dict, start_idx, end_idx, self.num_samples
                )
                vanilla_cost = self.model.get_cost(vanilla_info, candidates)
                all_candidates = candidates
                all_costs = vanilla_cost
                n_proposals = 0
                cached_proposal_cost = None

                enabled = (
                    step >= self.proposal_start
                    and (step - self.proposal_start) % self.proposal_every == 0
                )
                if enabled:
                    if self.target_source == "best":
                        source_idx = self._select_source_indices(
                            candidates, vanilla_cost
                        )
                    else:
                        source_idx = torch.zeros(
                            current_bs,
                            1,
                            device=self.device,
                            dtype=torch.long,
                        )
                    rows = torch.arange(current_bs, device=self.device)
                    predicted = vanilla_info["predicted_emb"]
                    target_chain = predicted[
                        rows[:, None], source_idx, -(self.horizon + 1) :
                    ]
                    sources = target_chain.shape[1]
                    if self.proposal_mode == "repair":
                        source_plans = candidates[rows[:, None], source_idx]
                        repairs = self._single_site_repairs(
                            source_plans, target_chain
                        )
                        proposals, cached_proposal_cost, repair_count = self._select_repairs(
                            info_dict,
                            start_idx,
                            end_idx,
                            repairs,
                        )
                        solve_repair_candidates += repair_count
                    elif self.proposal_mode == "stitch":
                        proposals, _ = self._project_stitched_paths(target_chain)
                    else:
                        per_source = max(1, self.proposal_beams // sources)
                        flat_chain = target_chain.reshape(
                            current_bs * sources,
                            self.horizon + 1,
                            target_chain.shape[-1],
                        )
                        proposals, _ = self._project_path(
                            flat_chain, beam_limit=per_source
                        )
                        proposals = proposals.reshape(
                            current_bs,
                            sources * proposals.shape[1],
                            self.horizon,
                            self.action_dim,
                        )[:, : self.proposal_beams]
                    n_proposals = proposals.shape[1]
                    if n_proposals:
                        proposals, trust_alpha = self._trust_project(
                            proposals, batch_mean, batch_var
                        )
                        solve_trust_clipped += int((trust_alpha < 0.999999).sum().item())
                        solve_trust_alpha_sum += float(trust_alpha.sum().item())
                        solve_trust_alpha_count += int(trust_alpha.numel())
                        proposal_info = self._expand_infos(
                            info_dict, start_idx, end_idx, n_proposals
                        )
                        if cached_proposal_cost is not None and self.proposal_trust_rms <= 0:
                            proposal_cost = cached_proposal_cost
                        else:
                            proposal_cost = self.model.get_cost(
                                proposal_info, proposals
                            )
                        all_candidates = torch.cat(
                            [candidates, proposals], dim=1
                        )
                        all_costs = torch.cat(
                            [vanilla_cost, proposal_cost], dim=1
                        )
                        solve_proposals += current_bs * n_proposals

                iteration_cost, iteration_index = all_costs.float().min(dim=1)
                improve = iteration_cost < best_seen_cost
                if improve.any():
                    rows_best = torch.arange(current_bs, device=self.device)
                    iteration_plan = all_candidates[rows_best, iteration_index]
                    best_seen_cost = torch.where(
                        improve, iteration_cost, best_seen_cost
                    )
                    best_seen_plan[improve] = iteration_plan[improve]

                topk_vals, topk_inds = torch.topk(
                    all_costs, k=self.topk, dim=1, largest=False
                )
                batch_indices = torch.arange(
                    current_bs, device=self.device
                )[:, None]
                topk_candidates = all_candidates[batch_indices, topk_inds]
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

                if n_proposals:
                    proposal_elite = topk_inds >= self.num_samples
                    n_hits = int(proposal_elite.sum().item())
                    solve_proposal_elites += n_hits
                    solve_iterations_hit += int(
                        proposal_elite.any(dim=1).sum().item()
                    )

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            best_seen_actions[start_idx:end_idx] = best_seen_plan
            best_seen_costs[start_idx:end_idx] = best_seen_cost
            if self.output_mode == "best_seen":
                # Score the final distribution mean and the concrete archive
                # under identical singleton-environment calls.  A direct
                # (B,2,...) call can mix the legacy goal and sample axes.
                pair = torch.stack([batch_mean, best_seen_plan], dim=1)
                pair_cost = []
                for env_offset in range(current_bs):
                    pair_info = self._expand_infos(
                        info_dict,
                        start_idx + env_offset,
                        start_idx + env_offset + 1,
                        2,
                    )
                    pair_cost.append(
                        self.model.get_cost(
                            pair_info, pair[env_offset : env_offset + 1]
                        ).float()
                    )
                pair_cost = torch.cat(pair_cost, dim=0)
                selected = pair_cost.argmin(dim=1)
                pair_rows = torch.arange(current_bs, device=self.device)
                returned_actions[start_idx:end_idx] = pair[pair_rows, selected]
                outputs["costs"].extend(
                    pair_cost[pair_rows, selected].detach().cpu().tolist()
                )
            else:
                returned_actions[start_idx:end_idx] = batch_mean
                # Preserve the parent CEM's historical elite-mean diagnostic.
                outputs["costs"].extend(final_batch_cost)

        self._diag["solves"] += 1
        self._diag["proposal_candidates"] += solve_proposals
        self._diag["proposal_elites"] += solve_proposal_elites
        self._diag["iterations_with_proposal_elite"] += solve_iterations_hit
        self._diag["trust_clipped"] += solve_trust_clipped
        self._diag["repair_candidates_evaluated"] += solve_repair_candidates
        outputs["actions"] = returned_actions.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        outputs["best_seen_actions"] = best_seen_actions.detach().cpu()
        outputs["best_seen_costs"] = best_seen_costs.detach().cpu().tolist()
        outputs["dcl_beam"] = {
            **self._diag,
            "solve_proposal_candidates": solve_proposals,
            "solve_proposal_elites": solve_proposal_elites,
            "solve_iterations_with_hit": solve_iterations_hit,
            "solve_trust_clipped": solve_trust_clipped,
            "solve_mean_trust_alpha": (
                solve_trust_alpha_sum / max(1, solve_trust_alpha_count)
            ),
            "solve_repair_candidates_evaluated": solve_repair_candidates,
            "proposal_mode": self.proposal_mode,
            "output_mode": self.output_mode,
            "search_mode": self.search_mode,
        }
        if self.trace:
            print(
                "DCL-Beam-CEM "
                f"{time.time() - start_time:.3f}s proposals={solve_proposals} "
                f"elite_hits={solve_proposal_elites} "
                f"iteration_hits={solve_iterations_hit} "
                f"trust_clipped={solve_trust_clipped} "
                f"mean_alpha={solve_trust_alpha_sum / max(1, solve_trust_alpha_count):.3f}",
                flush=True,
            )
        return outputs
