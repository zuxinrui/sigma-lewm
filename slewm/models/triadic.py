"""Deployable JEPA classes carrying the article-champion training-only heads.

The auxiliary heads live on the model so Stable World Model's strict
``save_pretrained`` / ``load_pretrained`` round trip remains valid.  Planning
continues to use the unchanged JEPA encoder, predictor and latent-MSE
criterion; these heads are consulted only by the aligned training objective.
"""

from __future__ import annotations

import torch

from slewm.fmca.fmca import ProjHead
from slewm.models.world_model import JEPA
from slewm.models.residual import ResidualTransitionJEPA


def _attach_champion_heads(
    model: torch.nn.Module,
    *,
    embed_dim: int,
    action_input_dim: int,
    projection_dim: int,
) -> None:
    if min(embed_dim, action_input_dim, projection_dim) < 1:
        raise ValueError("champion head dimensions must be positive")
    # Backward: [z_(t+1); action_embedding_t] <-> z_t.
    model.head_f = ProjHead(2 * embed_dim, projection_dim, depth=2)
    model.head_g = ProjHead(embed_dim, projection_dim, depth=2)
    # Inverse: [z_t; z_(t+1)] <-> raw five-step action block.
    model.head_f_idm = torch.nn.Sequential(
        torch.nn.Linear(2 * embed_dim, 256),
        torch.nn.LayerNorm(256),
        torch.nn.GELU(),
        torch.nn.Linear(256, projection_dim),
    )
    model.head_g_idm = torch.nn.Sequential(
        torch.nn.Linear(action_input_dim, 64),
        torch.nn.LayerNorm(64),
        torch.nn.GELU(),
        torch.nn.Linear(64, projection_dim),
    )


class AlignedChampionJEPA(JEPA):
    """Released absolute-transition JEPA plus frozen champion auxiliary heads."""

    aligned_champion = True

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        *,
        champion_embed_dim: int = 192,
        champion_action_input_dim: int,
        champion_projection_dim: int = 64,
    ):
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
        )
        _attach_champion_heads(
            self,
            embed_dim=int(champion_embed_dim),
            action_input_dim=int(champion_action_input_dim),
            projection_dim=int(champion_projection_dim),
        )


class AlignedChampionResidualTransitionJEPA(ResidualTransitionJEPA):
    """Residual/stop-gradient task model plus champion auxiliary heads."""

    aligned_champion = True

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        *,
        champion_embed_dim: int = 192,
        champion_action_input_dim: int,
        champion_projection_dim: int = 64,
    ):
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
        )
        _attach_champion_heads(
            self,
            embed_dim=int(champion_embed_dim),
            action_input_dim=int(champion_action_input_dim),
            projection_dim=int(champion_projection_dim),
        )


__all__ = [
    "AlignedChampionJEPA",
    "AlignedChampionResidualTransitionJEPA",
]
