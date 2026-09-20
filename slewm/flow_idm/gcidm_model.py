"""Faithful GC-IDM (arXiv:2605.08732): amortized goal-conditioned inverse dynamics.

MLP regressor  pi(z_t, z_{t+h}, h) -> a_t , exactly per the paper:
  * 3-layer MLP, hidden 512, LayerNorm, GELU, 10% dropout, input = [z_t ; z_goal]  (~1.5M params)
  * remaining-horizon h normalized to [0,1], sinusoidally encoded, 2-layer MLP -> c,
    applied via AdaLN-Zero modulation (gamma/beta zero-init) before the action head
  * plain MSE regression head (no flow / diffusion)

Exposes `.act(z_t, z_g, h, ...)` returning (B, n_samples, A) so it drops straight into
slewm.flow_idm.flow_actor.LeWMFlowActor as an amortized actor (single forward = GC-IDM's fast inference).
Trained with MSE (slewm.flow_idm.train --model gcidm --loss mse).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn, Tensor

if TYPE_CHECKING:
    from slewm.flow_idm.config import HCFMConfig


def _sinusoidal(x: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=x.device) / half)
    args = x.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _AdaLNBlock(nn.Module):
    """LayerNorm + Linear + GELU + Dropout, with AdaLN-Zero(h-cond) modulation of the LN output."""

    def __init__(self, dim: int, cond_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.lin = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.ada = nn.Linear(cond_dim, 2 * dim)          # -> (gamma, beta)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)                    # AdaLN-Zero: starts as identity modulation

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        gamma, beta = self.ada(c).chunk(2, dim=-1)
        h = self.norm(x) * (1 + gamma) + beta
        return x + self.drop(self.act(self.lin(h)))      # residual


class GCIDM(nn.Module):
    """Goal-conditioned inverse dynamics regressor.  pi(z_t, z_g, h) -> a  (standardized action)."""

    def __init__(self, cfg: "HCFMConfig"):
        super().__init__()
        self.cfg = cfg
        D, A = cfg.d_latent, cfg.d_action
        hid = 512
        drop = float(getattr(cfg, "dropout", 0.0))
        self.h_max = int(cfg.h_max)
        self.sin_dim = 64

        self.inp = nn.Sequential(nn.Linear(2 * D, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(drop))
        self.cond = nn.Sequential(nn.Linear(self.sin_dim, hid), nn.SiLU(), nn.Linear(hid, hid))
        self.blocks = nn.ModuleList([_AdaLNBlock(hid, hid, drop) for _ in range(3)])
        self.head = nn.Linear(hid, A)

    def forward(self, z_t: Tensor, z_g: Tensor, h: Tensor) -> Tensor:
        hn = (h.float().clamp(1, self.h_max) / self.h_max)          # normalized remaining horizon in [0,1]
        c = self.cond(_sinusoidal(hn, self.sin_dim))
        x = self.inp(torch.cat([z_t, z_g], dim=-1))
        for blk in self.blocks:
            x = blk(x, c)
        return self.head(x)                                         # (B, A)

    @torch.no_grad()
    def act(self, z_t: Tensor, z_g: Tensor, h: Tensor, ode_steps: int = 1, n_samples: int = 1) -> Tensor:
        """(B,D),(B,D),(B,) -> (B, n_samples, A).  Deterministic regressor -> replicate across samples."""
        a = self.forward(z_t, z_g, h)                               # (B, A)
        return a[:, None, :].expand(a.shape[0], n_samples, a.shape[-1]).contiguous()
