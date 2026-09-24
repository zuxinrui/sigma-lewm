"""Training-side implementations of the two auxiliary-objective baselines.

`slewm.baselines.inverse` covers the action-supervision baselines (SMWM, AC-MTM).
This module covers the two that change the *regulariser* rather than the action head, so
that they can be trained on Gear-Assembly and Cube-Stacking instead of only replayed from
the authors' released checkpoints.

Both implementations are thin adapters over the vendored upstream code in `external/`
(see `external/repos.lock.json` for the pinned commits); the objectives themselves are the
authors'.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

from slewm.models.world_model import JEPA

# The upstream objectives are not redistributed here.  Clone them at the commits in
# docs/external_repos.lock.json and point SLEWM_EXTERNAL at the directory holding them.
_EXTERNAL = Path(os.environ.get("SLEWM_EXTERNAL", Path(__file__).resolve().parents[2] / "external"))


def _vendored(repo: str, module: str):
    """Import `module` from the pinned checkout of `repo` without polluting sys.path."""
    import importlib.util

    path = _EXTERNAL / repo / f"{module}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Clone {repo} at the commit in docs/external_repos.lock.json "
            f"and set SLEWM_EXTERNAL to the directory that contains it."
        )
    spec = importlib.util.spec_from_file_location(f"_vendored_{repo}_{module}", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


class SubspaceSIGReg(nn.Module):
    """Sub-JEPA's SIGReg: the statistic is computed inside K random orthogonal subspaces.

    Drop-in for `module.SIGReg`, so `train.py` needs no change: it takes the same
    ``(T, B, D)`` input and returns a scalar. ``orthogonality_loss`` is added with weight
    ``theta``, as in the authors' `train.py`.

    The upstream module takes ``(B, T, D)`` (the authors call ``self.sigreg(emb)``), so the
    input is transposed back before the call. Passing ``(T, B, D)`` straight through makes
    the Epps-Pulley test average over the few time steps instead of the batch, and the
    regulariser then no longer penalises the embeddings shrinking towards a point.
    """

    def __init__(
        self,
        knots: int = 17,
        num_proj: int = 1024,
        num_subspaces: int = 32,
        subspace_dim: int | None = None,
        init_mode: str = "orthogonal_frozen",
        embed_dim: int = 192,
        theta: float = 0.0,
    ):
        super().__init__()
        upstream = _vendored("sub-jepa", "subjepa")
        self.inner = upstream.MultiSubspaceSIGReg(
            knots=knots,
            num_proj=num_proj,
            num_subspaces=num_subspaces,
            subspace_dim=subspace_dim,
            init_mode=init_mode,
            embed_dim=embed_dim,
        )
        self.theta = float(theta)

    def forward(self, proj: Tensor) -> Tensor:
        if proj.dim() != 3:
            raise ValueError(f"expected (T, B, D) embeddings, got {tuple(proj.shape)}")
        value = self.inner(proj.transpose(0, 1))  # (T, B, D) -> the authors' (B, T, D)
        if self.theta:
            value = value + self.theta * self.inner.orthogonality_loss()
        return value


class ReachabilityHead(nn.Module):
    """RC-aux's reachability scorer over a latent pair."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        width = 2 * latent_dim
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU()]
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z_from: Tensor, z_to: Tensor) -> Tensor:
        return self.net(torch.cat([z_from, z_to], dim=-1)).squeeze(-1)


def reachability_aux_loss(
    model,
    emb: Tensor,
    *,
    weight: float,
    margin: float = 1.0,
    temperature: float = 1.0,
) -> dict[str, Tensor]:
    """Time-contrastive reachability: adjacent latents outrank shuffled ones.

    ``emb`` is ``(B, T, D)``. Positives are consecutive frames of the same episode;
    negatives pair a frame with a frame of another episode at the same time index, which
    keeps the time marginal identical between the two sets.
    """
    head = getattr(model, "reachability_head", None)
    if head is None or weight <= 0:
        return {}
    if emb.dim() != 3 or emb.size(1) < 2:
        raise ValueError(f"reachability needs (B, T>=2, D) embeddings, got {tuple(emb.shape)}")
    z_from, z_to = emb[:, :-1], emb[:, 1:]
    positive = head(z_from, z_to)
    permutation = torch.randperm(emb.size(0), device=emb.device)
    negative = head(z_from, z_to[permutation])
    loss = torch.nn.functional.relu(margin - (positive - negative) / temperature).mean()
    return {
        "reachability_loss": loss,
        "reachability_weighted": weight * loss,
        "reachability_margin_satisfied": (positive > negative).float().mean().detach(),
    }


class RCAuxJEPA(JEPA):
    """Released LeWM with RC-aux's training-only reachability head.

    The head is auxiliary: planning discards it, which is the ``lambda=0`` arm we compare
    against on the standard benchmarks (the authors' headline TwoRoom number uses the head
    as a planning cost and therefore does not belong in an objective-only comparison).
    """

    rc_aux_baseline = True

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        *,
        reachability_latent_dim: int = 192,
        reachability_hidden_dim: int = 256,
        reachability_depth: int = 2,
    ) -> None:
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
        )
        self.reachability_head = ReachabilityHead(
            latent_dim=int(reachability_latent_dim),
            hidden_dim=int(reachability_hidden_dim),
            depth=int(reachability_depth),
        )

    def score_reachability(self, z_from: Tensor, z_to: Tensor) -> Tensor:
        return self.reachability_head(z_from, z_to)
