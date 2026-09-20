"""Qantara Brownian-bridge state head — the STATE axis of Qantara, ported faithfully.

Qantara (arXiv:2607.04978) §2.1: the state axis is a BROWNIAN BRIDGE between two clean latents,

    z_tilde_tau = (1 - tau)*z_t + tau*z_{t+1} + gamma*sqrt(tau*(1 - tau))*eps,     gamma = 1

and the predictor does x-PREDICTION (predict z_{t+1}), NOT v-integration. At tau=0 the source is
EXACTLY z_t (the latent the planner feeds at deployment); at tau=1 it is the target z_{t+1}.

WHY THE gamma NOISE MATTERS (this is the one non-inert thing the bridge buys). With gamma=0 the
interpolant is deterministic, so z_{t+1} = (z_tilde_tau - (1-tau) z_t) / tau EXACTLY — a linear
readout that ignores z_<=t and a_t drives the loss to zero, i.e. every tau>0 locus is a
copy-the-answer shortcut and only tau=0 teaches dynamics. gamma>0 raises the shortcut's difficulty;
by Bishop (1995) input noise == Tikhonov, so this is a Jacobian penalty with schedule
sqrt(tau(1-tau)) — ZERO at both query loci tau in {0,1}, maximal at tau=1/2 which inference never
visits. Our study (docs/2026-07-17_contrastive_flow_unification_report.md) established that at the
CEM query locus this predictor equals the LeWM MSE forward model (K-sweep flat 89.8-91.2), so the
bridge buys no sampling — its value is exactly this query-locus-vanishing curriculum. gamma lives in
the noise term below, so the faithful bridge realises it automatically.

ON OUR STACK (important): this head is a TRAINING-TIME encoder-shaping loss, stripped at eval like
every aux head. CEM's forward model is the AR predictor (model.predict), trained separately by tf+AR
MSE, which we therefore KEEP ON. Qantara folds the forward model and the bridge into ONE
tau-conditioned network; we cannot without rewriting ARPredictor, so here the bridge ADDS the
tau-interior curriculum on top of the proven forward model rather than replacing it.

NO stop-grad on z (encoder-shaping), matching FlowIDM / DCL-IDM. SIGReg 0.09 floors the rank.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BridgeStateHead(nn.Module):
    """(z_tilde_tau, tau, context z_<=t, a_t) -> x-prediction of z_{t+1}.

    A small transformer over [z_ctx window, action token, noised-state token(+tau)], mirroring the
    shape of FlowPolicyHead so the two are comparable. Only the noised-state token's output is read;
    the context/action tokens are masked from attending to it (no answer leak into the context).
    """

    def __init__(self, D=192, A_emb=192, ctx=3, width=192, depth=4, heads=3, t_dim=32, dropout=0.0):
        super().__init__()
        self.ctx = int(ctx)
        self.z_proj = nn.Linear(D, width)            # context latents
        self.a_proj = nn.Linear(A_emb, width)        # action (encoded)
        self.zt_proj = nn.Linear(D, width)           # the noised interpolant token
        self.t_embed = nn.Sequential(nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, width))
        self.pos = nn.Parameter(torch.zeros(1, self.ctx, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            batch_first=True, norm_first=True, dropout=dropout)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Linear(width, D)               # x-prediction -> D-dim latent
        nn.init.trunc_normal_(self.pos, std=0.02)
        # [z_0..z_{ctx-1}, a, z_tilde]: the noised-state token (last) attends to everything; the
        # context + action tokens must NOT attend to it (else z_{t+1} leaks into the context).
        n = self.ctx + 2
        m = torch.zeros(n, n, dtype=torch.bool)
        m[:self.ctx + 1, self.ctx + 1] = True        # positions 0..ctx cannot see position ctx+1
        self.register_buffer("attn_mask", m)

    def forward(self, z_tilde, tau, z_ctx, a_emb):
        """z_tilde (N,D); tau (N,1); z_ctx (N,ctx,D); a_emb (N,A_emb) -> z_hat_{t+1} (N,D)."""
        tok = torch.cat([
            self.z_proj(z_ctx) + self.pos,                              # (N, ctx, W)
            self.a_proj(a_emb).unsqueeze(1),                            # (N, 1,   W)
            (self.zt_proj(z_tilde) + self.t_embed(tau)).unsqueeze(1),   # (N, 1,   W)
        ], dim=1)
        return self.out(self.blocks(tok, mask=self.attn_mask)[:, -1])


def bridge_loss(head: BridgeStateHead, emb: torch.Tensor, act_emb: torch.Tensor, gamma: float = 1.0):
    """L_bridge^z over every (z_ctx -> z_{t+1}) pair. emb (B,T,D) GRADS ON; act_emb (B,T,A_emb) too.

    For each t in [ctx-1, T-2]: context = z_{t-ctx+1..t} (last slot is z_t), action = a_t, target =
    z_{t+1}. Returns (loss, recon-diagnostic). No stop-grad: grads flow to the encoder AND the action
    encoder, consistent with the rest of our encoder-shaping recipe."""
    B, T, D = emb.shape
    ctx = head.ctx
    if T < ctx + 1:
        z = emb.new_zeros(())
        return z, z
    wins, tgts, acts = [], [], []
    for t in range(ctx - 1, T - 1):                 # need z_{t+1}, so stop at T-2
        wins.append(emb[:, t - ctx + 1:t + 1])      # (B, ctx, D) — includes z_t at the last slot
        tgts.append(emb[:, t + 1])                  # (B, D) = z_{t+1}
        acts.append(act_emb[:, t])                  # (B, A_emb) = a_t (encoded)
    z_ctx = torch.cat(wins, dim=0)                  # (N, ctx, D) — grads to encoder
    z_t = z_ctx[:, -1]                              # (N, D)
    z_tp1 = torch.cat(tgts, dim=0)                  # (N, D)
    a_e = torch.cat(acts, dim=0)                    # (N, A_emb)
    tau = torch.rand(z_t.size(0), 1, device=z_t.device, dtype=z_t.dtype)   # tau ~ U(0,1)
    eps = torch.randn_like(z_t)
    sig = (tau * (1.0 - tau)).clamp_min(0.0).sqrt()                        # sqrt(tau(1-tau))
    z_tilde = (1.0 - tau) * z_t + tau * z_tp1 + gamma * sig * eps          # Brownian bridge, gamma=1
    z_hat = head(z_tilde, tau, z_ctx, a_e)
    loss = (z_hat - z_tp1).pow(2).mean()            # x-prediction MSE
    with torch.no_grad():
        recon = loss.detach()
    return loss, recon
