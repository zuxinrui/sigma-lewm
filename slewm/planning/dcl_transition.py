"""Planning-time adapter for the C_S forward-DCL transition head.

C_S trained a symmetric decoupled-contrastive objective on

    head_f([z_t, action_encoder(a_t)]) <-> head_g(z_{t+1}).

The raw cosine is not used directly to compare different ``(z_t, a_t)``
queries.  Instead, a fixed empirical successor-latent bank estimates the
query-dependent partition function:

    E(z, a, z') = -s(z, a, z') + log mean_j exp s(z, a, z_j^-),
    s = cosine(head_f([z, E_a(a)]), head_g(z')) / tau.

Lower energy means that the imagined successor is more compatible with the
current latent and action than ordinary successors from the dataset marginal.
This is a transition-density-ratio witness, not a goal-success verifier.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from slewm.planning.energy_acid import contrastive_energy_from_keys


def _load_head(
    state_dict: dict,
    prefix: str,
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    device: str,
) -> nn.Sequential:
    head = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )
    sub = {key[len(prefix):]: value for key, value in state_dict.items()
           if key.startswith(prefix)}
    if not sub:
        raise ValueError(f"state_dict has no {prefix} forward-DCL head")
    head.load_state_dict(sub)
    return head.to(device).eval().requires_grad_(False)


class ForwardDCLVerifier:
    """Frozen C_S DCL head with an empirical successor-bank normalizer."""

    mode = "forward_dcl"

    def __init__(
        self,
        state_dict: dict,
        *,
        action_encoder: nn.Module,
        successor_bank: torch.Tensor,
        tau: float = 0.1,
        device: str = "cuda",
        D: int = 192,
        A: int = 10,
        K: int = 64,
    ) -> None:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if successor_bank.ndim != 2 or successor_bank.shape[1] != D:
            raise ValueError(
                f"successor_bank must be (M,{D}), got {successor_bank.shape}"
            )
        if successor_bank.shape[0] < 2:
            raise ValueError("successor_bank needs at least two negative samples")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")

        self.D, self.A, self.K = int(D), int(A), int(K)
        self.tau = float(tau)
        self.device = str(device)
        self.action_encoder = action_encoder
        self.f = _load_head(
            state_dict,
            "head_f.net.",
            input_dim=2 * self.D,
            hidden_dim=2 * self.D,
            output_dim=self.K,
            device=self.device,
        )
        self.g = _load_head(
            state_dict,
            "head_g.net.",
            input_dim=self.D,
            hidden_dim=256,
            output_dim=self.K,
            device=self.device,
        )
        self.successor_bank = successor_bank.to(self.device).float()
        with torch.no_grad():
            self.pool_g = torch.nn.functional.normalize(
                self.g(self.successor_bank), dim=-1
            )

    @torch.no_grad()
    def score(
        self, z: torch.Tensor, z_next: torch.Tensor, a_std: torch.Tensor
    ) -> torch.Tensor:
        """Return normalized forward-DCL energy; lower is more compatible."""
        if z.shape != z_next.shape or z.shape[-1] != self.D:
            raise ValueError(
                f"z and z_next must have matching (*,{self.D}) shapes, "
                f"got {z.shape} and {z_next.shape}"
            )
        if a_std.shape[:-1] != z.shape[:-1] or a_std.shape[-1] != self.A:
            raise ValueError(
                f"a_std must match z leading dims and end in {self.A}, got {a_std.shape}"
            )
        action_emb = self.action_encoder(a_std.float().unsqueeze(-2)).squeeze(-2)
        query = self.f(torch.cat([z.float(), action_emb.float()], dim=-1))
        positive = self.g(z_next.float())
        return contrastive_energy_from_keys(
            query,
            positive,
            bank_keys=self.pool_g,
            tau=self.tau,
            mode="nce",
        )

    @torch.no_grad()
    def cosine(
        self, z: torch.Tensor, z_next: torch.Tensor, a_std: torch.Tensor
    ) -> torch.Tensor:
        """Raw trained alignment, exposed only for diagnostics."""
        action_emb = self.action_encoder(a_std.float().unsqueeze(-2)).squeeze(-2)
        query = torch.nn.functional.normalize(
            self.f(torch.cat([z.float(), action_emb.float()], dim=-1)), dim=-1
        )
        key = torch.nn.functional.normalize(self.g(z_next.float()), dim=-1)
        return (query * key).sum(dim=-1)

