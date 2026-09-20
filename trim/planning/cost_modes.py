"""Alternative CEM planning costs that align the objective with the eval criterion.

The eval success is LATCHED: an episode counts as success if the T-block reaches the
goal at ANY env step within budget. But stock CEM optimizes only the TERMINAL latent
distance ||z_hat_T - z_goal||^2 — a mismatch that (with terminal-only cost) yields
dawdle-early/sprint-late plans and is fragile to P_f terminal hallucination.

These criteria reward getting close to the goal at ANY point along the imagined
rollout, matching the latched success metric. Pure P_f (no verifiers). Monkeypatched
onto the frozen model instance via types.MethodType in eval_ext (`+cost_mode=...`).

All return (B, S) per-candidate cost, lower = better — same contract as jepa.criterion.
"""
from __future__ import annotations

import torch


def _goal_and_traj(info_dict):
    pred = info_dict["predicted_emb"]              # (B,S,Tp,D)
    goal = info_dict["goal_emb"]
    g = goal[..., -1:, :]                          # (B,S,1,D) the goal latent
    per_t = (pred - g).pow(2).sum(dim=-1)          # (B,S,Tp) squared latent dist per step
    return per_t


def criterion_minovertime(self, info_dict):
    """min_t ||z_hat_t - z_goal||^2 — reward closest approach anywhere in the horizon."""
    return _goal_and_traj(info_dict).min(dim=-1).values


def criterion_softmin(self, info_dict, tau: float = 0.5):
    """-tau * logsumexp(-per_t/tau) — smooth min (differentiable-friendly soft closest approach)."""
    per_t = _goal_and_traj(info_dict)
    return -tau * torch.logsumexp(-per_t / tau, dim=-1)


def criterion_lastk_min(self, info_dict, k: int = 3):
    """min over the LAST k steps only — reward SUSTAINED arrival, not a transient early pass."""
    per_t = _goal_and_traj(info_dict)
    k = min(int(k), per_t.shape[-1])
    return per_t[..., -k:].min(dim=-1).values


def criterion_discounted(self, info_dict, gamma: float = 0.8):
    """sum_t gamma^(T-1-t) * per_t — dense, later-weighted; rewards early progress + arrival."""
    per_t = _goal_and_traj(info_dict)             # (B,S,Tp)
    T = per_t.shape[-1]
    w = gamma ** torch.arange(T - 1, -1, -1, device=per_t.device, dtype=per_t.dtype)
    return (per_t * w).sum(dim=-1)


BUILDERS = {
    "minovertime": lambda cfg: criterion_minovertime,
    "softmin": lambda cfg: (lambda self, info: criterion_softmin(self, info, float(cfg.get("cost_tau", 0.5)))),
    "lastk": lambda cfg: (lambda self, info: criterion_lastk_min(self, info, int(cfg.get("cost_k", 3)))),
    "discounted": lambda cfg: (lambda self, info: criterion_discounted(self, info, float(cfg.get("cost_gamma", 0.8)))),
}
