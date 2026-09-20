"""Post-hoc heteroscedastic variance head on P_f (projector space, D=192).

VarianceHead(z_t, a_emb_t, z_hat_next) -> per-dim log sigma^2 of the 1-step P_f residual.
Trained on FROZEN base features with the Gaussian NLL; AR-rollout depths are handled by a
multiplicative per-depth recalibration c_k (Levi et al. 2022):
    logvar_k(x) = logvar(x) + c_k,   c_k = log E[ res_k^2 * exp(-logvar) ].
c_0 == 0 by construction after head training (NLL optimum); c_k>0 grows with AR depth.
"""
from __future__ import annotations

import torch
import torch.nn as nn

LOGVAR_MIN, LOGVAR_MAX = -10.0, 5.0


class VarianceHead(nn.Module):
    """(z_t, a_emb_t, z_hat_next) -> logvar (B, D). ~0.44M params."""

    def __init__(self, D: int = 192, A_emb: int = 192, hidden: int = 512, max_depth: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D + A_emb + D, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, D),
        )
        # per-AR-depth log recalibration; identity (0) until fit_calibration
        self.register_buffer("c_k", torch.zeros(max_depth + 1))

    def forward(self, z_t: torch.Tensor, a_emb: torch.Tensor, z_hat_next: torch.Tensor,
                depth: int = 0) -> torch.Tensor:
        x = torch.cat([z_t, a_emb, z_hat_next], dim=-1)
        logvar = self.net(x).clamp(LOGVAR_MIN, LOGVAR_MAX)
        k = min(int(depth), self.c_k.numel() - 1)
        return logvar + self.c_k[k]


def gaussian_nll(z_next: torch.Tensor, z_hat_next: torch.Tensor,
                 logvar: torch.Tensor) -> torch.Tensor:
    """0.5 * mean_d[ res^2 / sigma^2 + log sigma^2 ]  (per-sample mean over dims, then batch mean)."""
    res2 = (z_next - z_hat_next).pow(2)
    return 0.5 * (res2 * torch.exp(-logvar) + logvar).mean()


@torch.no_grad()
def fit_calibration(head: VarianceHead, pairs_per_depth: dict[int, list]) -> dict[int, float]:
    """pairs_per_depth[k] = list of (res2, logvar_uncal) tensor pairs at AR depth k.
    Sets head.c_k[k] = log( E[res^2 * exp(-logvar)] ). Returns the fitted scales."""
    out = {}
    for k, pairs in sorted(pairs_per_depth.items()):
        num, cnt = 0.0, 0
        for res2, logvar in pairs:
            num += (res2 * torch.exp(-logvar)).sum().item()
            cnt += res2.numel()
        scale = max(num / max(cnt, 1), 1e-8)
        head.c_k[min(k, head.c_k.numel() - 1)] = float(torch.log(torch.tensor(scale)))
        out[k] = scale
    return out
