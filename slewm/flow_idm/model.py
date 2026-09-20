"""HC-FM-IDM velocity-field network. See docs/2026-07-14_hc_fm_idm_design.md §1, §8.

DiT (AdaLN-zero) flow-matching action head, horizon/goal conditioned:

    v_theta(a_s, s ; z_t, z_g, h) -> u_hat in R^10

The AdaLN condition is a SUM of three W-dim embeddings -- sinusoidal flow-time s,
a learned step-gap embedding h, and an MLP over [z_t; z_g]. The noised action a_s is
tokenized into n_sub sub-tokens (each d_action//n_sub-d), run through `depth`
ConditionalBlock's (reused from module.py), and read out per-token by a zero-init head
so the initial velocity is ~0 (DiT-style). LayerNorm only, pure PyTorch, no BatchNorm.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn, Tensor

from slewm.models.components import ConditionalBlock, MLP

if TYPE_CHECKING:  # avoid importing the slewm.flow_idm package at runtime (siblings may be absent)
    from slewm.flow_idm.config import HCFMConfig


class _SetConditionalBlock(ConditionalBlock):
    """ConditionalBlock with BIDIRECTIONAL (non-causal) self-attention.

    module.py's Attention defaults to `causal=True` (correct for ARPredictor's
    autoregressive frame prediction), and ConditionalBlock.forward inherits that
    default. For a DiT flow-matching action head the n_sub action sub-tokens are a
    SET denoised jointly, so every sub-token must attend to all others. We override
    only the attention call to be non-causal; the AdaLN-zero modulation is unchanged.
    """

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        from slewm.models.components import modulate
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), causal=False)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


def sinusoidal_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Sinusoidal positional embedding for continuous flow-time.

    t : (B,) float, scaled into a wide range before embedding so that s in [0,1]
        spans many frequencies (DiT/SiT convention).
    -> (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class HCFMIDM(nn.Module):
    """Horizon-conditioned flow-matching inverse-dynamics velocity field."""

    def __init__(self, cfg: "HCFMConfig"):
        super().__init__()
        self.cfg = cfg
        W = cfg.width
        self.width = W
        self.d_action = cfg.d_action
        self.n_sub = cfg.n_sub
        assert cfg.d_action % cfg.n_sub == 0, "d_action must be divisible by n_sub"
        self.sub_dim = cfg.d_action // cfg.n_sub          # 10 // 5 = 2
        self.time_scale = 1000.0                          # s in [0,1] -> [0,1000] for embedding

        # --- AdaLN condition: three W-dim embeddings, summed ---
        # sinusoidal(s) -> MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(W, W), nn.SiLU(), nn.Linear(W, W)
        )
        # step-gap h (discrete) -> embedding; h_max+1 rows cover indices 0..h_max
        self.h_embed = nn.Embedding(cfg.h_max + 1, W)
        # [z_t ; z_g] -> MLP -> W  (LayerNorm-based MLP from module.py)
        self.latent_mlp = MLP(2 * cfg.d_latent, W, W)

        # --- action tokenization ---
        drop = float(getattr(cfg, "dropout", 0.0))
        self.tok_embed = nn.Linear(self.sub_dim, W)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.n_sub, W) * 0.02)
        self.tok_dropout = nn.Dropout(drop)               # emb dropout (GC-IDM-style regularization)

        # --- DiT backbone (AdaLN-zero blocks conditioned on `cond`) ---
        self.blocks = nn.ModuleList([
            _SetConditionalBlock(W, cfg.heads, cfg.dim_head, cfg.mlp_dim, drop)   # attn + FFN dropout
            for _ in range(cfg.depth)
        ])

        # --- per-token velocity head (zero-init: initial velocity ~ 0) ---
        self.final_norm = nn.LayerNorm(W, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(W, self.sub_dim)
        nn.init.constant_(self.head.weight, 0.0)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, a_s: Tensor, s: Tensor, z_t: Tensor, z_g: Tensor, h: Tensor) -> Tensor:
        """
        a_s : (B, 10)   noised action
        s   : (B,)      float flow-time in [0,1]
        z_t : (B, 192)  current latent
        z_g : (B, 192)  goal latent (t+h)
        h   : (B,)      long step-gap
        ->  v : (B, 10) velocity estimate
        """
        B = a_s.shape[0]

        # AdaLN condition vector (B, W): sum of time / horizon / latent embeddings
        t_emb = self.time_mlp(sinusoidal_embedding(s * self.time_scale, self.width))
        h_emb = self.h_embed(h)
        zc = self.latent_mlp(torch.cat([z_t, z_g], dim=-1))
        cond = (t_emb + h_emb + zc)[:, None, :]           # (B, 1, W) -> broadcasts over tokens

        # tokenize action into n_sub sub-tokens
        x = a_s.reshape(B, self.n_sub, self.sub_dim)      # (B, n_sub, sub_dim)
        x = self.tok_dropout(self.tok_embed(x) + self.pos_embed)   # (B, n_sub, W)

        for blk in self.blocks:
            x = blk(x, cond)

        v = self.head(self.final_norm(x))                 # (B, n_sub, sub_dim)
        return v.reshape(B, self.d_action)                # (B, 10)
