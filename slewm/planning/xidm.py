"""X-IDM: cross-attention DiT-conditioned contrastive IDM (cross-encoder).

Design report: docs/2026-07-14_contrastive_idm_arch_upgrades.md (§ architecture one).

Replaces the SIMPLE bi-encoder DCL-IDM
    s(c,a) = cos( head_f_idm([z_t;z_{t+1}]) , head_g_idm(a) ) / tau        (late fusion, one dot product)
with a JOINT cross-encoder that fuses the transition c=[z_t;z_{t+1}] into the action tokens at EVERY
layer via AdaLN-zero (reusing module.ConditionalBlock), then reads out a scalar compatibility SCORE:
    s(c,a) = XIDM(c, a) in R           (higher = a explains the transition z_t -> z_{t+1})

Why: InfoNCE's optimum is the density ratio p(a|c)/p_neg, but a bi-encoder's cos/tau family can only
represent an angle in a 64-d bottleneck; a cross-encoder approximates the ratio far better
(Reimers 2019 / Khattab 2020). Cost: E(c,a) can't be precomputed, so training uses SAMPLED negatives
(all-pairs is O(BT^2) cross-encoder forwards — intractable). Encoder-shaping: NO stop-grad on z's.

Training semantics identical to L_DCL_IDM otherwise: raw standardized action block (bypasses the
action_encoder), in-batch negatives with a duplicate-action mask, learnable temperature.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from slewm.models.components import ConditionalBlock


class XIDMHead(nn.Module):
    def __init__(self, D=192, A=10, sub_dim=2, W=128, depth=4, heads=4,
                 dim_head=32, mlp_dim=256, tau_init=0.1):
        super().__init__()
        assert A % sub_dim == 0, f"action dim {A} not divisible by sub_dim {sub_dim}"
        self.n_sub = A // sub_dim                                   # 5 PushT sub-actions (2-D each)
        self.sub_dim = sub_dim
        self.cond = nn.Sequential(nn.Linear(2 * D, W), nn.SiLU(), nn.Linear(W, W))
        self.act_embed = nn.Linear(sub_dim, W)
        self.pos = nn.Parameter(torch.zeros(1, self.n_sub, W))
        self.blocks = nn.ModuleList(
            [ConditionalBlock(W, heads, dim_head, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(W, elementwise_affine=False, eps=1e-6)
        # multiplicative FiLM readout: x_pooled * (1 + film(c)) is a c-a INTERACTION, so its
        # gradient survives the InfoNCE softmax from step 1 (an ADDITIVE c term would cancel — all
        # M+1 candidates share the same anchor c, and a per-anchor constant drops out of softmax).
        self.film = nn.Linear(W, W)
        self.score = nn.Linear(W, 1)
        # learnable temperature (log-parameterized, clamped at use)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init), dtype=torch.float32))
        nn.init.trunc_normal_(self.pos, std=0.02)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.clamp(math.log(0.02), math.log(1.0)).exp()

    def forward(self, c: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """c: (N, 2D) transition context, a: (N, A) raw std action block -> score (N,)."""
        c_vec = self.cond(c)                                       # (N, W)
        sub = a.reshape(a.size(0), self.n_sub, self.sub_dim)       # (N, n_sub, sub_dim)
        x = self.act_embed(sub) + self.pos                         # (N, n_sub, W)
        for blk in self.blocks:
            x = blk(x, c_vec.unsqueeze(1))                         # AdaLN-zero cross-encoder fusion (ramps up)
        x = self.norm(x).mean(dim=1)                              # (N, W) mean-pool sub-action tokens
        # Multiplicative FiLM shortcut: c-a interaction → non-zero encoder-shaping gradient from
        # step 1 (AdaLN-zero blocks are identity at init, so their c-gradient is 0). The deep
        # cross-encoder fusion ramps up on top as AdaLN un-zeros.
        feat = x * (1.0 + self.film(c_vec))
        return self.score(feat).squeeze(-1)                       # (N,)


def xidm_infonce_loss(head: XIDMHead, c_flat: torch.Tensor, a_flat: torch.Tensor,
                      n_neg: int = 15, dup_cos: float = 0.98,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Sampled-negative InfoNCE for the cross-encoder. c_flat (BT,2D), a_flat (BT,A) with GRADS on c.

    For each positive i: candidates = [a_i, a_{neg_1..neg_M}]; score all M+1 via the cross-encoder
    (BT x (M+1) forwards); positive is column 0; duplicate-action negatives masked to -inf."""
    BT = c_flat.size(0)
    M = int(min(n_neg, BT - 1))
    dev = c_flat.device
    neg = torch.randint(0, BT, (BT, M), device=dev, generator=generator)   # random in-batch negatives
    cand = torch.cat([torch.arange(BT, device=dev).unsqueeze(1), neg], dim=1)  # (BT, M+1); col0=positive
    a_cand = a_flat[cand]                                          # (BT, M+1, A)
    c_rep = c_flat.unsqueeze(1).expand(BT, M + 1, c_flat.size(-1))  # (BT, M+1, 2D) — grads flow to encoder
    scores = head(c_rep.reshape(-1, c_flat.size(-1)),
                  a_cand.reshape(-1, a_flat.size(-1))).reshape(BT, M + 1)
    # duplicate-action mask: a sampled negative equal to the positive action is a false negative
    with torch.no_grad():
        an = torch.nn.functional.normalize(a_flat, dim=-1)
        pos_n = an.unsqueeze(1)                                    # (BT,1,A)
        cand_n = an[cand]                                         # (BT,M+1,A)
        dup = (cand_n * pos_n).sum(-1) > dup_cos                  # (BT,M+1)
        dup[:, 0] = False                                        # never mask the positive itself
    scores = scores.masked_fill(dup, -1e4) / head.tau
    labels = torch.zeros(BT, dtype=torch.long, device=dev)
    return torch.nn.functional.cross_entropy(scores, labels)
