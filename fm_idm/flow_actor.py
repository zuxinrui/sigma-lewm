"""Horizon-conditioned flow IDM used as an AMORTIZED ACTOR (GC-IDM-style inference).

Wraps a frozen LeWM + a trained HC-FM-IDM / GC-ACID flow IDM. Exposes the `Actionable`
`get_action` so the CEM solver seeds/mixes from the IDM's amortized plan
(`stable_worldmodel.solver.utils.prepare_init_action`). Delegates get_cost/criterion to the
base world model, so CEM still SCORES by the WM — the IDM only proposes.

Inference knob (the point of this file, toward "beat CEM at less compute"):
  - CEM n_steps=0            -> pure react (the IDM plan, ~1 forward/step, no search)
  - CEM n_steps=k, mode=seed -> IDM seeds the CEM mean, then k cheap refinement iters
  - CEM n_steps=k, mode=mix  -> inject IDM samples as candidates (needs MixedCEMSolver)

The amortized plan is rolled GC-IDM-style: at macro-step t, generate a_t from
(current latent, goal latent, remaining horizon h), advance the frozen predictor, repeat.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fm_idm.flow import generate


class LeWMFlowActor(nn.Module):
    def __init__(self, base, idm, h_max: int = 12, ode_steps: int = 1,
                 h_mode: str = "remaining", h_fixed: int = 5,
                 cost_weight: str | None = None, arrival: int = 5, device: str = "cuda"):
        super().__init__()
        self.base = base
        self.idm = idm.eval().requires_grad_(False)
        self.h_max = int(h_max)
        self.ode_steps = int(ode_steps)
        self.h_mode = h_mode          # "remaining" (GC-IDM) | "fixed" (goal-distance) | "clamp"
        self.h_fixed = int(h_fixed)   # goal-distance in macro-steps (goal_offset/action_block=5)
        self.cost_weight = cost_weight  # None -> base terminal cost; else AdaJEPA cumulative (hold/linear/...)
        self.arrival = int(arrival)
        self.interpolate_pos_encoding = True

    # --- Costable: CEM scores by the WM (terminal, or AdaJEPA cumulative/HOLD cost) ---
    def criterion(self, info_dict):
        return self.base.criterion(info_dict)

    def get_cost(self, info_dict, action_candidates):
        term = self.base.get_cost(info_dict, action_candidates)   # also populates predicted_emb/goal_emb
        if not self.cost_weight or self.cost_weight == "terminal":
            return term
        # AdaJEPA weighted cumulative cost over the horizon (reuse cumcost semantics)
        pred = info_dict["predicted_emb"]                          # (B,S,L,D)
        goal = info_dict["goal_emb"][..., -1:, :]                  # (B,S,1,D)
        H = action_candidates.shape[2]
        zc = pred[:, :, -H:, :]
        sqd = (zc - goal).pow(2).sum(dim=-1)                       # (B,S,H)
        k = torch.arange(1, H + 1, device=sqd.device, dtype=sqd.dtype)
        if self.cost_weight == "uniform":
            w = torch.ones(H, device=sqd.device, dtype=sqd.dtype)
        elif self.cost_weight == "linear":
            w = k
        elif self.cost_weight == "hold":
            w = (k >= self.arrival).to(sqd.dtype)
            if float(w.sum()) == 0.0:
                w[-1] = 1.0
        else:
            raise ValueError(f"unknown cost_weight {self.cost_weight}")
        return (sqd * w).sum(dim=-1)

    # --- helpers (mirror knob_exp/actor.py, use the base WM's encode/predict) ---
    def _encode_emb(self, px):
        return self.base.encode({"pixels": px})["emb"]          # (B, T, D)

    def _advance(self, emb_hist, act_hist, HS):
        lo = max(0, len(emb_hist) - HS)
        emb_trunc = torch.stack(emb_hist[lo:], dim=1)           # (B, h, D)
        acts = torch.stack(act_hist[lo:], dim=1)                # (B, h, A)
        act_emb = self.base.action_encoder(acts)                # (B, h, A_emb)
        return self.base.predict(emb_trunc, act_emb)[:, -1]     # (B, D)

    def _h_for_step(self, step: int, horizon: int, dev, B):
        if self.h_mode == "fixed":
            h = self.h_fixed
        elif self.h_mode == "clamp":
            h = min(max(horizon - step, 1), self.h_max)         # remaining plan steps, clamped
        else:  # "remaining" — GC-IDM: steps left to the goal, clamped to train range
            h = min(max(self.h_fixed - step, 1), self.h_max)
        return torch.full((B,), int(h), device=dev, dtype=torch.long)

    # --- Actionable: roll the flow IDM as an amortized actor ---
    @torch.no_grad()
    def get_action(self, info, horizon: int = 1, prefix_actions=None):
        dev = next(self.base.parameters()).device
        px = info["pixels"].to(dev)
        goal = info["goal"].to(dev)
        if px.ndim == 4:
            px = px[:, None]
        if goal.ndim == 4:
            goal = goal[:, None]
        z0 = self._encode_emb(px)[:, -1]                        # (B, D)
        z_goal = self._encode_emb(goal)[:, -1]                  # (B, D)
        B = z0.shape[0]
        HS = int(getattr(self.base.predictor, "num_frames", 3))

        emb_hist = [z0]
        act_hist = []
        out = []
        n_prefix = 0
        if prefix_actions is not None:
            n_prefix = prefix_actions.shape[1]
            for t in range(n_prefix):
                act_hist.append(prefix_actions[:, t].to(dev))
                emb_hist.append(self._advance(emb_hist, act_hist, HS))
        for k in range(horizon):
            h = self._h_for_step(n_prefix + k, n_prefix + horizon, dev, B)
            if hasattr(self.idm, "act"):        # GCIDM (direct regressor) exposes .act
                a = self.idm.act(emb_hist[-1], z_goal, h,
                                 ode_steps=self.ode_steps, n_samples=1).squeeze(1)
            else:                               # flow IDM -> integrate the ODE
                a = generate(self.idm, emb_hist[-1], z_goal, h,
                             ode_steps=self.ode_steps, n_samples=1).squeeze(1)   # (B, A)
            out.append(a)
            act_hist.append(a)
            emb_hist.append(self._advance(emb_hist, act_hist, HS))
        return torch.stack(out, dim=1)                          # (B, horizon, A)
