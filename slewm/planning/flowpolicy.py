"""Flow-matching POLICY head — the one Qantara idea with a mechanism behind it.

Qantara (arXiv:2607.04978) ablation: dropping the policy mode collapses CEM PLANNING
(PushT 90.1 -> 27.2 +-32.5), and 1-mode{forward} -> 2-mode{forward,policy} moves Cube
76.3 -> 91.9. The policy mode stabilises PLANNING, not behaviour cloning — that is the
non-obvious part and the reason to port it.

WHY THIS IS NOT OUR ACID FlowIDM:

    FlowIDM:                  v(x_tau, tau | z_t, z_{t+1})   <- conditions on the FUTURE
    FlowPolicy (this file):   v(x_tau, tau | z_{<=t})        <- conditions on the PAST only

That difference is the whole point. Every verifier we built conditions on z_{t+1} and
is co-trained with P_f, so it tends to agree with P_f on P_f's own imagined rollouts
and carries little independent signal. A policy that never sees
z_{t+1} cannot be a function of P_f's transition, so the encoder-shaping pressure it applies is
genuinely a new direction.

Objective (ACID/Qantara-style CFM on the action axis):
    tau ~ Beta(1.5,1),  x_tau = tau*eps + (1-tau)*a_t,  target u = a_t - eps   [data-minus-noise]
    L_FM^a = || v_phi(x_tau, tau, z_{<=t}) - u ||^2
NO stop-grad on z -> encoder-shaping, consistent with the rest of our recipe.

Note the sign convention: Qantara Eq.3 targets (a_t - eps_a) while our FlowIDM targets
(eps - a). Both are valid (they differ by the direction of integration); this head follows
Qantara's so its lambda_a transfers, and samples with a = eps + v (Euler from tau=1 to 0).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlowPolicyHead(nn.Module):
    """Causal flow-matching policy: (z window, noisy action, tau) -> velocity.

    Conditions on the last `ctx` latents only — never on z_{t+1}. Small transformer over
    [z_{t-ctx+1..t}, x_tau] with a tau embedding, mirroring the shape of our FlowIDM so the
    two are comparable, but with the future token removed.
    """

    def __init__(self, D=192, A=10, ctx=3, width=192, depth=4, heads=3, t_dim=32, dropout=0.0):
        super().__init__()
        self.ctx = int(ctx)
        self.z_proj = nn.Linear(D, width)
        self.a_proj = nn.Linear(A, width)
        self.t_embed = nn.Sequential(nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, width))
        self.pos = nn.Parameter(torch.zeros(1, self.ctx, width))     # position within the z window
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            batch_first=True, norm_first=True, dropout=dropout)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Linear(width, A)
        nn.init.trunc_normal_(self.pos, std=0.02)
        # prefix (the z window) attends only among itself; the action token sees everything.
        n = self.ctx + 1
        m = torch.zeros(n, n, dtype=torch.bool)
        m[:self.ctx, self.ctx] = True                                # z tokens cannot see x_tau
        self.register_buffer("attn_mask", m)

    def forward(self, x_tau, tau, z_ctx):
        """x_tau (N,A); tau (N,1); z_ctx (N, ctx, D) -> velocity (N,A)."""
        tok = torch.cat([
            self.z_proj(z_ctx) + self.pos,                            # (N, ctx, W)
            (self.a_proj(x_tau) + self.t_embed(tau)).unsqueeze(1),    # (N, 1,  W)
        ], dim=1)
        return self.out(self.blocks(tok, mask=self.attn_mask)[:, -1])

    @torch.no_grad()
    def sample(self, z_ctx, a_dim=None):
        """1-step Euler from tau=1 (pure noise) to 0 — ACID reports one step suffices on PushT.

        NOT a sample, despite the name (corrected 2026-07-17). With target u = a - eps the optimum
        is v* = E[a - eps | x_tau, tau, c]; at tau=1 the input IS eps and eps _|_ a, so

            a_hat = eps + v*(eps,1,c) = eps + (E[a|c] - eps) = E[a|c]

        the eps cancels identically and Var_eps(a_hat) = 0. One Euler step from pure noise lands on
        the tau=1 slice, and that slice IS the conditional mean (i.e. MSE regression). Real sampling
        needs tau<1, hence >=2 steps. docs/2026-07-17_contrastive_flow_unification_report.md §2.2."""
        n = z_ctx.shape[0]
        a_dim = a_dim or self.out.out_features
        eps = torch.randn(n, a_dim, device=z_ctx.device, dtype=z_ctx.dtype)
        tau1 = torch.ones(n, 1, device=z_ctx.device, dtype=z_ctx.dtype)
        return eps + self.forward(eps, tau1, z_ctx)                   # target u = a - eps


def flowpolicy_loss(head: FlowPolicyHead, emb: torch.Tensor, a_raw: torch.Tensor):
    """CFM loss over every valid (z window -> a_t) pair. emb (B,T,D) with GRADS on; a_raw (B,T,A).

    For each t >= ctx-1 the window is z_{t-ctx+1..t} and the target action is a_t.

    Returns (loss, recon). `recon` is NOT a reconstruction score: since the 1-step readout equals
    E[a|z_{<=t}] exactly (see FlowPolicyHead.sample), at the optimum

        recon -> E[ (E[a|z_<=t] - a)^2 ] = E[ Var(a | z_<=t) ]

    i.e. it measures the PAST-conditioned action variance — how multimodal the policy conditional
    genuinely is, in the action's own units. Useful, but do NOT confuse it with the INVERSE
    conditional Var(a | z_t, z_{t+1}) that the IDM arms turn on: conditioning on z_{t+1} can make
    the action near-determined (in PushT it is close to the commanded pusher position) while the
    policy conditional stays broad. Different quantities; only the latter licenses claims about
    IDM mode-averaging."""
    B, T, D = emb.shape
    ctx = head.ctx
    if T < ctx + 1:
        z = emb.new_zeros(())
        return z, z
    wins, acts = [], []
    for t in range(ctx - 1, T - 1):                 # need a_t, so stop at T-2
        wins.append(emb[:, t - ctx + 1:t + 1])      # (B, ctx, D)
        acts.append(a_raw[:, t])                    # (B, A)
    z_ctx = torch.cat(wins, dim=0)                  # (N, ctx, D) — grads flow to the encoder
    a = torch.cat(acts, dim=0).float()              # (N, A)
    eps = torch.randn_like(a)
    tau = torch.distributions.Beta(1.5, 1.0).sample((a.shape[0], 1)).to(a.device, a.dtype)
    x_tau = tau * eps + (1.0 - tau) * a
    v = head(x_tau, tau, z_ctx)
    loss = (v - (a - eps)).pow(2).mean()            # Qantara Eq.3 convention: u = a - eps
    with torch.no_grad():
        a_hat = eps + head(eps, torch.ones_like(tau), z_ctx)
        recon = (a_hat - a).pow(2).mean()
    return loss, recon
