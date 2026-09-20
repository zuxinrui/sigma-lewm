"""Defensive IPI-MPPI v2 for LeWM/JEPA.

The original implementation replaced part of MPPI's population with a narrow
IDM proposal.  Its ESS fallback then removed importance correction *on that
mixed population*, which is not a fallback to vanilla MPPI.  In a 50-D action
trajectory, the default proposal covariance also had almost no density overlap
with the vanilla component.

This version keeps the complete vanilla MPPI arm and adds a small, explicit
IDM proposal arm.  The proposal is forward-verified, constrained to a Gaussian
KL trust region, and combined with balance-heuristic MIS.  At every iteration
the vanilla and MIS updates are evaluated by the unchanged terminal world-model
cost; MIS is accepted only when its ESS is adequate and its cost is lower.
Consequently a rejected proposal gives exactly the configured vanilla MPPI
update (same vanilla samples and top-k semantics), rather than a biased mixture.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from stable_worldmodel.solver.mppi import MPPISolver
from stable_worldmodel.solver.utils import prepare_init_action

from trim.planning.idm_decode import IDMDecoder


def log_normal(a: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """log N(a; mu, sigma^2 I) summed over (H, D); additive const dropped (cancels in ratios).
    a: (B,N,H,D); mu: (B,1,H,D) or broadcastable; sigma: same or scalar tensor."""
    return -0.5 * ((a - mu) / sigma).pow(2).sum((-2, -1)) - torch.log(sigma).expand_as(a).sum((-2, -1))


def isotropic_scale_kl(scale: float, dims: int) -> float:
    """KL covariance term for N(., (scale*sigma)^2) || N(., sigma^2)."""
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"proposal scale must be positive, got {scale}")
    return 0.5 * int(dims) * (scale * scale - 1.0 - 2.0 * math.log(scale))


def clip_scale_to_kl(scale: float, dims: int, max_kl: float) -> float:
    """Move a requested scalar covariance scale toward 1 until its KL fits."""
    if max_kl < 0:
        raise ValueError(f"max_kl must be non-negative, got {max_kl}")
    scale = float(scale)
    if isotropic_scale_kl(scale, dims) <= max_kl:
        return scale
    if max_kl == 0:
        return 1.0

    # KL is monotone on either side of 1.  Binary search along that side.
    lo, hi = (scale, 1.0) if scale < 1.0 else (1.0, scale)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if scale < 1.0:
            if isotropic_scale_kl(mid, dims) > max_kl:
                lo = mid
            else:
                hi = mid
        else:
            if isotropic_scale_kl(mid, dims) <= max_kl:
                lo = mid
            else:
                hi = mid
    return hi if scale < 1.0 else lo


def kl_trust_region_center(
    base: torch.Tensor,
    proposal: torch.Tensor,
    sigma: torch.Tensor,
    max_mean_kl: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bridge ``proposal`` toward ``base`` so Gaussian mean KL is bounded.

    For equal diagonal covariance ``sigma^2``, the mean contribution is
    ``0.5 * ||(mu_q-mu_p)/sigma||^2``.  Returns the bridged center and one
    interpolation coefficient per batch item.
    """
    delta = proposal - base
    full_kl = 0.5 * (delta / sigma.clamp_min(1e-6)).pow(2).sum((-2, -1))
    budget = torch.as_tensor(max(max_mean_kl, 0.0), device=base.device, dtype=base.dtype)
    alpha = torch.sqrt(budget / full_kl.clamp_min(1e-12)).clamp(max=1.0)
    alpha = torch.where(full_kl <= budget, torch.ones_like(alpha), alpha)
    return base + alpha[:, None, None] * delta, alpha


def select_defensive_update(
    vanilla_mean: torch.Tensor,
    mis_mean: torch.Tensor,
    vanilla_cost: torch.Tensor,
    mis_cost: torch.Tensor,
    ess: torch.Tensor,
    ess_threshold: float | torch.Tensor,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept MIS only when numerically healthy and better under task cost."""
    threshold = torch.as_tensor(
        ess_threshold, device=ess.device, dtype=ess.dtype
    )
    accept = (
        torch.isfinite(mis_mean).flatten(1).all(1)
        & torch.isfinite(mis_cost)
        & (ess >= threshold)
        & (mis_cost + float(margin) < vanilla_cost)
    )
    chosen = torch.where(accept[:, None, None], mis_mean, vanilla_mean)
    return chosen, accept


class IPIMPPISolver(MPPISolver):
    def __init__(self, *args, idm_frac: float = 0.1, idm_sigma: float = 1.0,
                 shell_radius: float = 3.0, ratio_clip: float = float("inf"),
                 ess_floor: float = 1.0, ess_fraction: float = 0.0,
                 ess_ratio_floor: float = 0.5,
                 proposal_kl: float = 2.0, covariance_kl_fraction: float = 0.5,
                 accept_margin: float = 0.0, forward_verify_topk: int = 8,
                 history_size: int = 3, cache_costs: bool = True,
                 verify_cost_cache: bool = False, idm_tau: float = 0.1,
                 idm_state_dict_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.idm_frac, self.idm_sigma = float(idm_frac), float(idm_sigma)
        self.shell_radius = float(shell_radius)
        self.ratio_clip, self.ess_floor = float(ratio_clip), float(ess_floor)
        self.ess_fraction = float(ess_fraction)
        self.ess_ratio_floor = float(ess_ratio_floor)
        self.proposal_kl = float(proposal_kl)
        self.covariance_kl_fraction = float(covariance_kl_fraction)
        self.accept_margin = float(accept_margin)
        self.forward_verify_topk = int(forward_verify_topk)
        self.history_size = int(history_size)
        self.cache_costs = bool(cache_costs)
        self.verify_cost_cache = bool(verify_cost_cache)
        self._cost_cache_verified = False
        self.idm_tau = float(idm_tau)
        self._idm_sd_path = idm_state_dict_path
        self._decoder: IDMDecoder | None = None
        # Proposal draws must not consume the parent's vanilla random stream.
        # A fully rejected proposal run is therefore bitwise plain MPPI.
        self.proposal_gen = torch.Generator(device=self.device).manual_seed(
            self.torch_gen.initial_seed() ^ 0x5EED1D1
        )
        if self.idm_frac < 0:
            raise ValueError("idm_frac must be non-negative")
        if self.proposal_kl <= 0:
            raise ValueError("proposal_kl must be positive")
        if not 0 <= self.covariance_kl_fraction <= 1:
            raise ValueError("covariance_kl_fraction must be in [0,1]")
        if not 0 <= self.ess_ratio_floor <= 1:
            raise ValueError("ess_ratio_floor must be in [0,1]")

    def _get_decoder(self) -> IDMDecoder:
        if self._decoder is None:
            dev = str(next(self.model.parameters()).device)
            sd = torch.load(self._idm_sd_path, map_location="cpu") if self._idm_sd_path \
                else self.model.state_dict()
            self._decoder = IDMDecoder(sd, tau=self.idm_tau, device=dev)
        return self._decoder

    @torch.no_grad()
    def _encode(self, px: torch.Tensor) -> torch.Tensor:
        m = self.model
        h = m.encoder(px.float(), interpolate_pos_encoding=True).last_hidden_state[:, 0]
        return m.projector(h)

    @torch.no_grad()
    def _idm_plan(self, info: dict, H: int) -> torch.Tensor:
        """Build an IDM draft, but advance only through forward predictions.

        Contrastive IDMs produce a set of concrete action-pool modes.  We take
        their top-k hypotheses and use P_f to choose the action whose predicted
        next latent is closest to the local waypoint.  Regression IDMs use one
        hypothesis but are still forward-closed, avoiding the old idealized
        ``z = z + delta`` bookkeeping error.
        """
        dev = next(self.model.parameters()).device
        px = info["pixels"]
        px = px[:, -1] if px.ndim == 5 else px                 # (B,3,224,224)
        goal = info["goal"]
        goal = goal[:, -1] if goal.ndim == 5 else goal
        z0 = self._encode(px.to(dev))
        zg = self._encode(goal.to(dev))
        dec = self._get_decoder()
        emb_hist = z0.unsqueeze(1)
        act_hist: torch.Tensor | None = None
        for t in range(H):
            z = emb_hist[:, -1]
            delta = (zg - z) / max(H - t, 1)
            nrm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = torch.clamp(nrm, max=self.shell_radius) / nrm   # cap per-step chord length
            delta = delta * scale
            waypoint = z + delta
            hypotheses, _ = dec.candidates(
                z, waypoint, topk=self.forward_verify_topk
            )                                                        # (B,K,A)
            B, K, A = hypotheses.shape

            emb = emb_hist[:, None].expand(B, K, *emb_hist.shape[1:])
            emb = emb.reshape(B * K, *emb_hist.shape[1:])[:, -self.history_size:]
            if act_hist is None:
                acts = hypotheses.unsqueeze(2)
            else:
                past = act_hist[:, None].expand(B, K, *act_hist.shape[1:])
                acts = torch.cat([past, hypotheses.unsqueeze(2)], dim=2)
            acts_flat = acts.reshape(B * K, acts.shape[2], A)
            act_emb = self.model.action_encoder(acts_flat)[:, -self.history_size:]
            pred = self.model.predict(emb, act_emb)[:, -1]
            pred = pred.reshape(B, K, -1)
            verify_cost = (pred - waypoint[:, None]).pow(2).sum(-1)
            chosen_idx = verify_cost.argmin(dim=1)
            rows = torch.arange(B, device=dev)
            chosen_action = hypotheses[rows, chosen_idx]
            chosen_pred = pred[rows, chosen_idx]
            act_hist = chosen_action.unsqueeze(1) if act_hist is None else torch.cat(
                [act_hist, chosen_action.unsqueeze(1)], dim=1
            )
            emb_hist = torch.cat([emb_hist, chosen_pred.unsqueeze(1)], dim=1)
        assert act_hist is not None
        return act_hist

    def _expand_infos(self, batch: dict[str, Any], n: int) -> dict[str, Any]:
        expanded = {}
        for k, value in batch.items():
            if torch.is_tensor(value):
                dtype = self.dtype if value.is_floating_point() else None
                value = value.to(device=self.device, dtype=dtype)
                expanded[k] = value.unsqueeze(1).expand(value.shape[0], n, *value.shape[1:])
            elif isinstance(value, np.ndarray):
                expanded[k] = np.repeat(value[:, None, ...], n, axis=1)
            else:
                expanded[k] = value
        return expanded

    @torch.no_grad()
    def _prepare_cost_cache(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Encode current/goal images once for all optimizer iterations.

        ``JEPA.get_cost`` re-encodes both ViT inputs on every call.  They are
        invariant inside one solve, and the deployed protocol has one observed
        latent token, so caching them preserves the objective while avoiding
        the dominant fixed cost of the two-candidate acceptance probes.
        Unknown model contracts safely fall back to ``model.get_cost``.
        """
        if (
            not self.cache_costs
            or "pixels" not in batch
            or "goal" not in batch
            or not all(hasattr(self.model, name) for name in (
                "encode", "predict", "action_encoder"
            ))
        ):
            return None
        pixels = batch["pixels"].to(self.device, self.dtype)
        goal = batch["goal"].to(self.device, self.dtype)
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        if goal.ndim == 4:
            goal = goal.unsqueeze(1)
        current_emb = self.model.encode({"pixels": pixels})["emb"]
        goal_emb = self.model.encode({"pixels": goal})["emb"][:, -1]
        if current_emb.shape[1] != 1:
            return None
        return current_emb, goal_emb

    @torch.no_grad()
    def _cached_terminal_cost(
        self,
        cache: tuple[torch.Tensor, torch.Tensor],
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Exact one-token-history equivalent of JEPA.rollout + criterion."""
        current_emb, goal_emb = cache
        B, S, T, A = candidates.shape
        emb = current_emb[:, None].expand(B, S, 1, current_emb.shape[-1])
        emb = emb.reshape(B * S, 1, current_emb.shape[-1]).clone()
        actions = candidates.reshape(B * S, T, A)
        for t in range(T):
            action_prefix = actions[:, :t + 1]
            act_emb = self.model.action_encoder(action_prefix)[:, -self.history_size:]
            pred = self.model.predict(
                emb[:, -self.history_size:], act_emb
            )[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
        terminal = emb[:, -1].reshape(B, S, -1)
        return (terminal - goal_emb[:, None]).pow(2).sum(-1)

    def _cost(
        self,
        batch: dict[str, Any],
        candidates: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if cache is None:
            return self.model.get_cost(
                self._expand_infos(batch, candidates.shape[1]), candidates
            )
        cached = self._cached_terminal_cost(cache, candidates)
        if self.verify_cost_cache and not self._cost_cache_verified:
            reference = self.model.get_cost(
                self._expand_infos(batch, candidates.shape[1]), candidates
            )
            error = (cached - reference).abs().max().item()
            if error > 1e-4:
                raise AssertionError(
                    f"cached JEPA cost differs from get_cost by {error:.3e}"
                )
            print(f"IPI-MPPI cached-cost verification max_abs={error:.3e}")
            self._cost_cache_verified = True
        return cached

    def _vanilla_update(
        self, candidates: torch.Tensor, costs: torch.Tensor,
        return_ess: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Match the parent MPPISolver update, including optional top-k."""
        n = candidates.shape[1]
        if self.topk is not None and self.topk < n:
            values, indices = torch.topk(costs, k=self.topk, dim=1, largest=False)
            rows = torch.arange(candidates.shape[0], device=self.device)[:, None]
            relevant = candidates[rows, indices]
            relevant_costs = values
        else:
            relevant = candidates
            relevant_costs = costs
        weights = torch.softmax(
            -(relevant_costs - relevant_costs.min(1, keepdim=True).values)
            / self.temperature,
            dim=1,
        )
        mean = (weights[:, :, None, None] * relevant).sum(1)
        if return_ess:
            return mean, 1.0 / weights.pow(2).sum(1)
        return mean

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        t0 = time.time()
        outputs = {
            "costs": [], "mean": [], "var": [], "ess": [],
            "fallbacks": 0, "accepted": 0, "proposal_alpha": [],
            "proposal_sigma_scale": None, "proposal_extra_samples": 0,
            "low_ess": 0, "not_better": 0, "gate_delta": [],
            "vanilla_ess": [],
        }
        total = len(next(iter(info_dict.values())))
        init_action = prepare_init_action(self.model, info_dict, init_action,
                                          self.horizon, n_envs=total, action_dim=self.action_dim)
        mean, var = self.init_action_distrib(total, init_action)
        mean, var = mean.to(self.device), var.to(self.device)
        # num_samples retains its parent meaning: the full vanilla budget.
        # IDM samples are defensive extras, never replacements.
        N = self.num_samples
        n_idm = int(round(self.idm_frac * N)) if self.idm_frac > 0 else 0
        outputs["proposal_extra_samples"] = n_idm
        d_traj = self.horizon * self.action_dim
        cov_budget = self.proposal_kl * self.covariance_kl_fraction
        sigma_scale = clip_scale_to_kl(self.idm_sigma, d_traj, cov_budget)
        cov_kl = isotropic_scale_kl(sigma_scale, d_traj)
        mean_kl_budget = max(self.proposal_kl - cov_kl, 0.0)
        outputs["proposal_sigma_scale"] = sigma_scale

        for s0 in range(0, total, self.batch_size):
            s1 = min(s0 + self.batch_size, total)
            bs = s1 - s0
            b_mean, b_var = mean[s0:s1], var[s0:s1]

            batch = {k: v[s0:s1] for k, v in info_dict.items()}
            cost_cache = self._prepare_cost_cache(batch)
            a_idm = None
            if n_idm:
                single = {k: batch[k] for k in ("pixels", "goal")}
                a_idm = self._idm_plan(single, self.horizon)
                a_idm = a_idm.to(self.device, self.dtype).view(
                    bs, self.horizon, self.action_dim
                )

            fcost = None
            for _ in range(self.n_steps):
                noise_v = torch.randn(
                    bs, N, self.horizon, self.action_dim,
                    generator=self.torch_gen, device=self.device, dtype=self.dtype,
                )
                vanilla = b_mean[:, None] + noise_v * b_var[:, None]
                vanilla[:, 0] = b_mean

                if not n_idm:
                    costs_v = self._cost(batch, vanilla, cost_cache)
                    b_mean = self._vanilla_update(vanilla, costs_v)
                    fcost = costs_v.mean(1).cpu().tolist()
                    continue

                assert a_idm is not None
                center, alpha = kl_trust_region_center(
                    b_mean, a_idm, b_var, mean_kl_budget
                )
                sig_i_base = (sigma_scale * b_var).clamp_min(1e-6)
                noise_i = torch.randn(
                    bs, n_idm, self.horizon, self.action_dim,
                    generator=self.proposal_gen, device=self.device, dtype=self.dtype,
                )
                proposal = center[:, None] + noise_i * sig_i_base[:, None]
                candidates = torch.cat([vanilla, proposal], dim=1)
                costs = self._cost(batch, candidates, cost_cache)
                costs_v, costs_i = costs[:, :N], costs[:, N:]
                vanilla_mean, vanilla_ess = self._vanilla_update(
                    vanilla, costs_v, return_ess=True
                )

                # Deterministic-mixture MIS.  vanilla[:,0] is deliberately
                # excluded because the forced mean is not a draw from p_v.
                stochastic = torch.cat([vanilla[:, 1:], proposal], dim=1)
                stochastic_cost = torch.cat([costs_v[:, 1:], costs_i], dim=1)
                nv = N - 1
                beta = n_idm / (nv + n_idm)
                sig_v = b_var[:, None].expand_as(stochastic).clamp_min(1e-6)
                sig_i = sig_i_base[:, None].expand_as(stochastic)
                lp_v = log_normal(stochastic, b_mean[:, None], sig_v)
                lp_i = log_normal(stochastic, center[:, None], sig_i)
                log_q = torch.logaddexp(
                    math.log1p(-beta) + lp_v, math.log(beta) + lp_i
                )
                log_ratio = lp_v - log_q
                if math.isfinite(self.ratio_clip):
                    log_ratio = log_ratio.clamp(max=math.log(self.ratio_clip))
                log_w = (
                    -(stochastic_cost - stochastic_cost.min(1, keepdim=True).values)
                    / self.temperature
                    + log_ratio
                )
                weights = torch.softmax(log_w, dim=1)
                ess = 1.0 / weights.pow(2).sum(1)
                mis_mean = (weights[:, :, None, None] * stochastic).sum(1)

                # The task-aligned gate adds only two rollouts.  It uses the
                # unchanged terminal objective, so it cannot silently reshape
                # the 25-frame commitment protocol.
                probes = torch.stack([vanilla_mean, mis_mean], dim=1)
                probe_cost = self._cost(batch, probes, cost_cache)
                ess_threshold = max(
                    self.ess_floor,
                    self.ess_fraction * stochastic.shape[1],
                )
                ess_threshold_t = torch.maximum(
                    torch.full_like(vanilla_ess, ess_threshold),
                    self.ess_ratio_floor * vanilla_ess,
                )
                b_mean, accept = select_defensive_update(
                    vanilla_mean, mis_mean,
                    probe_cost[:, 0], probe_cost[:, 1],
                    ess, ess_threshold_t, self.accept_margin,
                )
                outputs["accepted"] += int(accept.sum())
                outputs["fallbacks"] += int((~accept).sum())
                outputs["low_ess"] += int((ess < ess_threshold_t).sum())
                outputs["not_better"] += int(
                    (probe_cost[:, 1] + self.accept_margin >= probe_cost[:, 0]).sum()
                )
                outputs["ess"].append(ess.float().mean().item())
                outputs["vanilla_ess"].append(vanilla_ess.float().mean().item())
                outputs["proposal_alpha"].append(alpha.float().mean().item())
                outputs["gate_delta"].append(
                    (probe_cost[:, 0] - probe_cost[:, 1]).float().mean().item()
                )
                fcost = torch.where(
                    accept, probe_cost[:, 1], probe_cost[:, 0]
                ).cpu().tolist()

            mean[s0:s1] = b_mean
            outputs["costs"].extend(fcost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        avg_ess = float(np.mean(outputs["ess"])) if outputs["ess"] else float("nan")
        avg_vanilla_ess = (
            float(np.mean(outputs["vanilla_ess"]))
            if outputs["vanilla_ess"] else float("nan")
        )
        avg_alpha = (
            float(np.mean(outputs["proposal_alpha"]))
            if outputs["proposal_alpha"] else float("nan")
        )
        avg_delta = (
            float(np.mean(outputs["gate_delta"]))
            if outputs["gate_delta"] else float("nan")
        )
        print(f"IPI-MPPI solve time: {time.time() - t0:.3f}s  "
              f"(vanilla={N}, idm_extra={n_idm}, accepted={outputs['accepted']}, "
              f"fallbacks={outputs['fallbacks']}, low_ess={outputs['low_ess']}, "
              f"not_better={outputs['not_better']}, sigma_scale={sigma_scale:.3f}, "
              f"ess={avg_ess:.2f}/{avg_vanilla_ess:.2f}, alpha={avg_alpha:.3f}, "
              f"gate_delta={avg_delta:+.4f})")
        return outputs
