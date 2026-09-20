"""Cross-step (goal-conditioned) IDM heads — GC-IDM [arXiv:2605.08732].

Two arms share the cross-step SAMPLER here; they differ only in the head/objective:
  C_T_gcidm_sigreg      -> GCIDMHead        (paper-faithful: 3-layer MLP 512 + LN + GELU +
                                             dropout 0.1, AdaLN-Zero sinusoidal horizon mod, MSE)
  C_T_flowgcidm_sigreg  -> FlowGCIDMHead    (ACID pi0 prefix-suffix transformer + CFM, prefix
                                             gains a horizon token; dropout 0.1 per GC-IDM)

PAPER FACTS (verified against the full text):
  sampling : (t,h) per example, h ~ Uniform[1, H_max], H_max = T = 50 ENV steps, tuple
             (z_t, z_{t+h}, a_t). No hindsight relabeling.
  head     : 3-layer MLP, hidden 512, LayerNorm, GELU, 10% dropout, no BatchNorm.
             Horizon: sinusoid(64) -> 2-layer MLP -> AdaLN-Zero scale/shift.
  loss     : E_(t,h)[ ||gc-idm(z_t, z_{t+h}, h) - a_t||^2 ]
  gradient : paper freezes the encoder. WE DEVIATE (user-approved): NO stop-grad on z, i.e.
             encoder-shaping, matching C_T_flowidm_sigreg's convention.

OUR DEVIATIONS (both user-approved, see docs/2026-07-15_gcidm_crossstep_variants_plan.md):
  * no-sg (encoder-shaping) instead of the paper's frozen encoder;
  * h <= T-1 = 7 MACRO steps. Our latents are macro frames (frameskip 5), so the paper's
    50 ENV steps ~= 10 macro steps -> we cover ~70% of their effective range, not 14%.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def sinusoidal_embedding(h: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """(N,) integer horizons -> (N, dim) sinusoidal encoding (paper: 64 dims)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=h.device, dtype=torch.float32) / half)
    ang = h.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def sample_crossstep(T: int, B: int, h_max: int, device, generator=None):
    """GC-IDM cross-step sampling, per example: h~U{1..min(h_max,T-1)}, t~U{0..T-1-h}.

    Returns (t_idx, h_idx) both (B,) long. Vectorized, independent per batch element."""
    hi = int(min(h_max, T - 1))
    h = torch.randint(1, hi + 1, (B,), device=device, generator=generator)
    # t uniform in [0, T-1-h] — inverse-CDF via a uniform float (per-row upper bound)
    span = (T - h).float()                                    # number of valid t values = T-h
    u = torch.rand(B, device=device, generator=generator)
    t = (u * span).floor().long().clamp_(min=0)
    t = torch.minimum(t, (T - 1 - h))
    return t, h


class GCIDMHead(nn.Module):
    """Paper-faithful GC-IDM: 3-layer MLP (512, LN, GELU, dropout .1) + AdaLN-Zero horizon."""

    def __init__(self, D=192, A=10, hidden=512, h_dim=64, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(2 * D, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, A)
        self.ln1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.ln2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.h_dim = h_dim
        self.h_mlp = nn.Sequential(nn.Linear(h_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        # AdaLN-Zero: two (scale, shift) pairs, zero-init -> ignores h at init, ramps up.
        self.adaLN = nn.Linear(hidden, 4 * hidden)
        nn.init.constant_(self.adaLN.weight, 0)
        nn.init.constant_(self.adaLN.bias, 0)

    def forward(self, z_t: torch.Tensor, z_goal: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """z_t,z_goal (N,D); h (N,) -> a_hat (N,A). z is a DIRECT input, so the encoder-shaping
        gradient is non-zero from step 1 (only the horizon modulation is zero-inited)."""
        c = self.h_mlp(sinusoidal_embedding(h, self.h_dim))                 # (N, hidden)
        g1, b1, g2, b2 = self.adaLN(c).chunk(4, dim=-1)
        x = torch.cat([z_t, z_goal], dim=-1)
        x = self.drop(torch.nn.functional.gelu(self.ln1(self.fc1(x)) * (1 + g1) + b1))
        x = self.drop(torch.nn.functional.gelu(self.ln2(self.fc2(x)) * (1 + g2) + b2))
        return self.out(x)


class FlowGCIDMHead(nn.Module):
    """ACID pi0 prefix-suffix transformer + CFM, conditioned CROSS-STEP.

    prefix = [proj(z_t), proj(z_{t+h}), horizon_token]  (attend only among themselves)
    suffix = proj(x_tau) + tau_embed                    (attends to prefix + itself)
    suffix output -> linear -> velocity. dropout 0.1 (GC-IDM's anti-overfitting; ACID used 0)."""

    def __init__(self, D=192, A=10, width=192, depth=4, heads=3, t_dim=32, h_dim=64, dropout=0.1):
        super().__init__()
        self.z_proj = nn.Linear(D, width)
        self.a_proj = nn.Linear(A, width)
        self.t_embed = nn.Sequential(nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, width))
        self.h_dim = h_dim
        self.h_embed = nn.Sequential(nn.Linear(h_dim, width), nn.SiLU(),
                                     nn.Dropout(dropout), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            batch_first=True, norm_first=True, dropout=dropout)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Linear(width, A)
        # 4 tokens: [z_t, z_{t+h}, horizon | x_tau]. Prefix rows (0,1,2) cannot see suffix col 3.
        m = torch.zeros(4, 4, dtype=torch.bool)
        m[0, 3] = m[1, 3] = m[2, 3] = True
        self.register_buffer("attn_mask", m)

    def forward(self, x_tau, tau, z_t, z_goal, h):
        """x_tau (N,A); tau (N,1); z_t,z_goal (N,D); h (N,) -> velocity (N,A)."""
        tok = torch.stack([
            self.z_proj(z_t),
            self.z_proj(z_goal),
            self.h_embed(sinusoidal_embedding(h, self.h_dim)),
            self.a_proj(x_tau) + self.t_embed(tau),
        ], dim=1)                                                # (N, 4, width)
        return self.out(self.blocks(tok, mask=self.attn_mask)[:, 3])
