"""ACID single-step inverse-dynamics velocity field (arXiv:2607.02403).

Faithful ACID IDM: a flow-matching velocity field v_phi(a_s, s; z_t, z_{t+1})
built as a pi0-style PREFIX-SUFFIX transformer.

    - PREFIX : z_t and z_{t+1} projected to two W-d tokens that self-attend (2x2).
    - SUFFIX : the noisy action a_s projected to ONE W-d token; the sinusoidal
               flow-time(s) embedding is injected on this suffix side (pi0 convention).
      The suffix attends to the prefix + itself (causal-after-prefix); the suffix
      output is linearly read out to the velocity in action space (zero-init head).

This shares the fm_idm forward signature `forward(a_s, s, z_t, z_g, h)` so
fm_idm/flow.py (fm_loss/generate/fm_residual) and fm_idm/train.py reuse UNCHANGED.
The ACID model IGNORES `h` and treats `z_g` as `z_{t+1}` (adjacent-pair IDM;
train with h_max=1 so LatentPairSampler emits adjacent pairs with z_g = z_{t+1}).
LayerNorm only; pure PyTorch; no BatchNorm. Defaults for ACID: depth=4, heads=3, W=192.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn, Tensor

if TYPE_CHECKING:  # avoid importing the fm_idm package at runtime (siblings may be absent)
    from fm_idm.config import HCFMConfig


def sinusoidal_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Sinusoidal positional embedding for continuous flow-time (DiT/SiT convention).

    t : (B,) float. -> (B, dim). Matches fm_idm/model.py's helper.
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


class _MaskedSelfAttention(nn.Module):
    """Pre-LN masked multi-head self-attention (own implementation, not module.py's).

    module.py's Attention is causal (is_causal=True) and cannot express the
    prefix-suffix mask, so we roll our own with an explicit additive/boolean mask.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0, "width must be divisible by heads"
        self.heads = heads
        self.dim_head = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """x:(B,T,D); attn_mask:(T,T) bool, True = query(row) may attend to key(col)."""
        B, T, D = x.shape
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.reshape(B, T, h, self.dim_head).transpose(1, 2) for t in (q, k, v)
        )  # (B, h, T, dim_head)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.to_out(out)


class _PrefixSuffixBlock(nn.Module):
    """LayerNorm + masked-MHA residual, then LayerNorm + MLP residual."""

    def __init__(self, dim: int, heads: int, mlp_dim: int):
        super().__init__()
        self.attn = _MaskedSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = x + self.attn(x, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ACIDIDM(nn.Module):
    """ACID prefix-suffix flow-matching inverse-dynamics velocity field."""

    def __init__(self, cfg: "HCFMConfig"):
        super().__init__()
        self.cfg = cfg
        W = cfg.width                       # 192
        heads = cfg.heads                   # 3 for ACID
        depth = cfg.depth                   # 4 for ACID
        self.width = W
        self.d_action = cfg.d_action        # 10
        self.time_scale = 1000.0            # s in [0,1] -> [0,1000] before sinusoid

        # --- prefix tokens: z_t and z_{t+1} (== z_g) projected to W ---
        self.z_t_proj = nn.Linear(cfg.d_latent, W)
        self.z_g_proj = nn.Linear(cfg.d_latent, W)
        # --- suffix token: noisy action a_s (10) -> W ---
        self.act_proj = nn.Linear(cfg.d_action, W)
        # --- flow-time injected on the suffix side (pi0): sinusoidal(s) -> MLP -> W ---
        self.time_mlp = nn.Sequential(nn.Linear(W, W), nn.SiLU(), nn.Linear(W, W))

        # --- prefix-suffix transformer stack ---
        mlp_dim = getattr(cfg, "mlp_dim", 4 * W)
        self.blocks = nn.ModuleList(
            [_PrefixSuffixBlock(W, heads, mlp_dim) for _ in range(depth)]
        )

        # --- suffix velocity read-out (zero-init: initial velocity ~ 0) ---
        self.final_norm = nn.LayerNorm(W)
        self.head = nn.Linear(W, cfg.d_action)
        nn.init.constant_(self.head.weight, 0.0)
        nn.init.constant_(self.head.bias, 0.0)

        # --- prefix-suffix attention mask (rows=query, cols=key; True = allowed) ---
        # token order = [ z_t (prefix0), z_{t+1} (prefix1), a_s (suffix2) ].
        # prefix rows see prefix cols only (2x2); suffix row sees prefix + itself.
        mask = torch.tensor(
            [[True,  True,  False],    # z_t   -> {z_t, z_{t+1}}
             [True,  True,  False],    # z_g   -> {z_t, z_{t+1}}
             [True,  True,  True]],    # a_s   -> {z_t, z_{t+1}, a_s}
            dtype=torch.bool,
        )
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, a_s: Tensor, s: Tensor, z_t: Tensor, z_g: Tensor, h: Tensor) -> Tensor:
        """
        a_s : (B, 10)   noised action
        s   : (B,)      float flow-time in [0,1]
        z_t : (B, 192)  current latent
        z_g : (B, 192)  z_{t+1} (ACID adjacent pair; h is IGNORED)
        h   : (B,)      accepted but UNUSED
        ->  v : (B, 10) velocity estimate (suffix read-out)
        """
        p_t = self.z_t_proj(z_t)                                        # (B, W)
        p_g = self.z_g_proj(z_g)                                        # (B, W)
        t_emb = self.time_mlp(sinusoidal_embedding(s * self.time_scale, self.width))
        suf = self.act_proj(a_s) + t_emb                               # (B, W)

        x = torch.stack([p_t, p_g, suf], dim=1)                        # (B, 3, W)
        for blk in self.blocks:
            x = blk(x, self.attn_mask)

        v = self.head(self.final_norm(x[:, 2]))                        # suffix -> (B, 10)
        return v
