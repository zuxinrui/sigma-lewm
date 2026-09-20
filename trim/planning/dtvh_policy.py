"""DTVH — Disagreement-Triggered Variable Horizon policy for LeWM PushT eval.

Baseline protocol executes the whole 25-env-step plan blind. DTVH re-encodes the observation
at every action-block boundary (5 env steps) and triggers an EARLY replan when either:
  g_t = ||z_obs - z_pred||^2 / D          (reality gap vs the plan's own rollout)
  d_t = IDM disagreement(z_prev, z_obs, a_sent_block)   (world responded to a different action)
exceeds dataset-floor-calibrated thresholds. Triggered replans are warm-started with the
remaining plan suffix. Failure containment: hysteresis + a per-episode extra-replan cap, so the
worst case is bounded extra compute, never a crash.

get_action reimplements WorldModelPolicy.get_action (stable_worldmodel/policy.py:351-439,
verified 2026-07-14) with three insertion points; the base machinery (buffers, solver call,
warm start, inverse_transform) is preserved bit-for-bit.
"""
from __future__ import annotations

import numpy as np
import torch

from stable_worldmodel.policy import WorldModelPolicy


@torch.no_grad()
def calibrate_thresholds(model, dataset, pixel_transform, action_scaler,
                         action_block: int = 5, n_pairs: int = 256,
                         q: float = 0.9, margin: float = 1.5, seed: int = 0,
                         idm_decoder=None, device: str = "cuda"):
    """Dataset floor for (g, d) on expert transitions: threshold = quantile_q * margin.

    g floor: 1-step P_f error on a block transition (context = single token, exactly the
    inference regime at H_ctx=1). d floor: IDM disagreement on TRUE (z_t, z_{t+b}, a_block).
    """
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep = np.asarray(dataset.get_col_data(col))
    st = np.asarray(dataset.get_col_data("step_idx"))
    idx = np.arange(len(ep) - action_block)
    ok = (ep[idx + action_block] == ep[idx]) & (st[idx + action_block] == st[idx] + action_block)
    idx = idx[ok]
    rng = np.random.default_rng(seed)
    picks = np.sort(rng.choice(idx, size=min(n_pairs, len(idx)), replace=False))

    D = 192
    g_all, d_all = [], []
    model.eval()
    for i in picks.tolist():
        rows = dataset.get_row_data(np.array([i, i + action_block]))
        px = torch.stack([pixel_transform(p) for p in rows["pixels"]]).to(device)   # (2,3,224,224)
        z = model.projector(model.encoder(px, interpolate_pos_encoding=True).last_hidden_state[:, 0])
        a_rows = dataset.get_row_data(np.arange(i, i + action_block))["action"]
        a_std = action_scaler.transform(np.asarray(a_rows)).reshape(1, -1).astype(np.float32)  # (1, 10)
        a_t = torch.from_numpy(a_std).to(device)
        a_emb = model.action_encoder(a_t.unsqueeze(1))                              # (1,1,A_emb)
        zhat = model.predict(z[:1].unsqueeze(1), a_emb)[:, -1]                      # (1,D)
        g_all.append(((zhat - z[1:2]).pow(2).mean() ).item())
        if idm_decoder is not None:
            d_all.append(idm_decoder.score(z[:1], z[1:2], a_t).item())
    tau_g = float(np.quantile(np.asarray(g_all), q)) * margin
    tau_d = float(np.quantile(np.asarray(d_all), q)) * margin if d_all else float("inf")
    return tau_g, tau_d


class DTVHPolicy(WorldModelPolicy):
    def __init__(self, *, tau_g, tau_d, hysteresis_ratio: float = 0.5,
                 max_extra_replans: int = 4, idm_decoder=None, var_head=None, **kwargs):
        super().__init__(**kwargs)
        self.tau_g, self.tau_d = float(tau_g), float(tau_d)
        self.hyst = float(hysteresis_ratio)
        self.max_extra = int(max_extra_replans)
        self.idm = idm_decoder
        self.var_head = var_head            # optional: calibrated per-state trigger
        self.stats = {"triggers": 0, "checks": 0, "replans": 0}

    # ---- per-env plan bookkeeping ----
    def set_env(self, env):
        super().set_env(env)
        n = getattr(env, "num_envs", 1)
        H = self.cfg.horizon
        self._plan_blocks = torch.zeros(n, H, self.solver.action_dim)   # standardized 10-d blocks
        self._pred_lat = torch.zeros(n, H + 1, 192)                     # rollout latents at boundaries
        self._planned = np.zeros(n, dtype=int)                          # env-steps in current plan
        self._z_prev = torch.zeros(n, 192)                              # latent at previous boundary
        self._extra = np.zeros(n, dtype=int)
        self._alarm = np.zeros(n, dtype=bool)
        # full-horizon warm-start store (base class would keep it None when rest is empty)
        ad = self.solver.action_dim
        self._next_init = torch.zeros(n, H, ad)

    @torch.no_grad()
    def _encode_frame(self, px_prepared):
        """PREPARED pixels (B, 3, 224, 224) (already transformed by _prepare_info)
        -> projector latents (B, 192)."""
        m = self.solver.model
        dev = next(m.parameters()).device
        px = px_prepared.to(dev).float()
        h = m.encoder(px, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        return m.projector(h).float().cpu()

    @torch.no_grad()
    def _rollout_plan(self, info_sliced, plan):
        """AR-rollout the chosen plan -> predicted latents (B, H+1, 192), via model.rollout
        (the exact inference path, jepa.py:61-110)."""
        m = self.solver.model
        dev = next(m.parameters()).device
        info = {}
        for k, v in info_sliced.items():
            if torch.is_tensor(v):
                info[k] = v.unsqueeze(1).to(dev)          # add S=1 sample dim
        acts = plan.unsqueeze(1).to(dev)                   # (B, 1, H, 10)
        out = m.rollout(dict(info), acts, history_size=3)
        pred = out["predicted_emb"][:, 0]                  # (B, H+1, 192)
        return pred.float().cpu()

    def get_action(self, info_dict: dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        n_envs = self.env.num_envs
        info = self._prepare_info(info_dict)
        # prepared pixels: (n_envs, T_hist, 3, 224, 224) — current frame = [:, -1]
        prep_px = info.get("pixels")

        needs_flush = info.pop("_needs_flush", None)
        if needs_flush is not None:
            for i in range(n_envs):
                if needs_flush[i]:
                    self._action_buffer[i].clear()
                    self._next_init[i] = 0
                    self._planned[i] = 0
                    self._extra[i] = 0
                    self._alarm[i] = False

        terminated = info.get("terminated")
        dead = (np.asarray(terminated, dtype=bool) if terminated is not None
                else np.zeros(n_envs, dtype=bool))

        ab = self.cfg.action_block

        # ---------- INSERTION 1: block-boundary disagreement triggers ----------
        chk = []
        for i in range(n_envs):
            if dead[i] or self._planned[i] == 0 or len(self._action_buffer[i]) == 0:
                continue
            popped = self._planned[i] - len(self._action_buffer[i])
            if popped > 0 and popped % ab == 0:
                chk.append((i, popped // ab))              # k-th boundary (1-based)
        if chk and prep_px is not None:
            envs = torch.as_tensor([i for i, _ in chk], dtype=torch.long)
            frames = prep_px[envs, -1] if prep_px.ndim == 5 else prep_px[envs]
            z_obs = self._encode_frame(frames)             # (m, 192)
            for row, (i, k) in enumerate(chk):
                zo = z_obs[row:row + 1]
                zp = self._pred_lat[i, k:k + 1]
                g = float((zo - zp).pow(2).mean())
                d = 0.0
                if self.idm is not None:
                    a_blk = self._plan_blocks[i, k - 1:k].to(zo.device)
                    dev = next(self.solver.model.parameters()).device
                    d = float(self.idm.score(self._z_prev[i:i + 1].to(dev), zo.to(dev),
                                             a_blk.to(dev)).item())
                tau_g = self.tau_g
                if self.var_head is not None:
                    # calibrated per-state floor: sigma^2 predicted for this transition (depth k)
                    dev = next(self.solver.model.parameters()).device
                    m = self.solver.model
                    a_emb = m.action_encoder(self._plan_blocks[i, k - 1:k].unsqueeze(0).to(dev))[:, 0]
                    lv = self.var_head(self._z_prev[i:i + 1].to(dev), a_emb,
                                       zp.to(dev), depth=k)
                    tau_g = max(tau_g, float(lv.exp().mean()) * 4.0)   # 2-sigma^2 slack, floor-capped
                self.stats["checks"] += 1
                hi = (g > tau_g) or (d > self.tau_d)
                lo = (g < self.hyst * tau_g) and (d < self.hyst * self.tau_d)
                if self._alarm[i] and lo:
                    self._alarm[i] = False
                if hi and not self._alarm[i] and self._extra[i] < self.max_extra:
                    # TRIGGER: flush remaining plan, warm-start with the suffix
                    self._alarm[i] = True
                    self._extra[i] += 1
                    self.stats["triggers"] += 1
                    H = self.cfg.horizon
                    suffix = self._plan_blocks[i, k:]                       # (H-k, 10)
                    init = torch.zeros(H, suffix.shape[-1])
                    init[: suffix.shape[0]] = suffix
                    self._next_init[i] = init
                    self._action_buffer[i].clear()
                    self._planned[i] = 0
                self._z_prev[i] = zo[0]

        # ---------- base replan machinery (verbatim semantics) ----------
        replan_idx = [i for i in range(n_envs)
                      if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan_idx:
            idx_tensor = torch.as_tensor(replan_idx, dtype=torch.long)
            sliced = {}
            for k_, v in info.items():
                if torch.is_tensor(v):
                    sliced[k_] = v[idx_tensor]
                elif isinstance(v, np.ndarray):
                    sliced[k_] = v[replan_idx]
                elif isinstance(v, list):
                    sliced[k_] = [v[i] for i in replan_idx]
                else:
                    sliced[k_] = v
            outputs = self.solver(sliced, init_action=self._next_init[idx_tensor])
            actions = outputs["actions"]                                    # (m, H, 10)
            keep = self.cfg.receding_horizon
            plan = actions[:, :keep]
            rest = actions[:, keep:]
            # the seeded init has been CONSUMED by this solve — reset, then re-seed
            # from the unexecuted tail if the config leaves one (h > receding).
            self._next_init[idx_tensor] = 0
            if self.cfg.warm_start and rest.shape[1] > 0:
                self._next_init[idx_tensor, : rest.shape[1]] = rest

            # ---------- INSERTION 2: record plan + its own rollout latents ----------
            self.stats["replans"] += len(replan_idx)
            pred = self._rollout_plan(sliced, actions)                      # (m, H+1, 192)
            flat = plan.reshape(len(replan_idx), self.flatten_receding_horizon, -1)
            for row, env_i in enumerate(replan_idx):
                self._plan_blocks[env_i] = actions[row]
                self._pred_lat[env_i] = pred[row]
                self._z_prev[env_i] = pred[row, 0]
                self._planned[env_i] = self.flatten_receding_horizon
                self._action_buffer[env_i].extend(flat[row])

        # ---------- base action pop (verbatim) ----------
        action_dim = self.env.single_action_space.shape[-1]
        action = torch.full((n_envs, action_dim), float("nan"))
        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        action = action.reshape(*self.env.action_space.shape)
        action = action.float().numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action
