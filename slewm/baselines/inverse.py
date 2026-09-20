"""Training-only inverse objectives for the AC-MTM and SMWM baselines.

Both baselines deliberately reuse the released LeWM encoder, action encoder,
forward predictor, projector, and planner.  Their inverse head is discarded by
the planner and differs only in the training loss: in-batch action
discrimination for AC-MTM, and action regression for SMWM.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from slewm.models.world_model import JEPA
from slewm.objectives.forward import aligned_horizon_loss


class InverseActionHead(nn.Module):
    """Predict one normalized macro action from adjacent state latents."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int = 2,
        layer_norm: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if min(latent_dim, action_dim, hidden_dim, depth) < 1:
            raise ValueError("inverse-head dimensions and depth must be positive")
        activations = {"relu": nn.ReLU, "gelu": nn.GELU}
        if activation not in activations:
            raise ValueError("inverse activation must be 'relu' or 'gelu'")
        act = activations[activation]
        layers: list[nn.Module] = [nn.Linear(2 * latent_dim, hidden_dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act())
        for _ in range(depth - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), act()))
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: Tensor, z_next: Tensor) -> Tensor:
        return self.net(torch.cat((z_t, z_next), dim=-1))


class ExternalInverseJEPA(JEPA):
    """Released LeWM with a training-only inverse-action head."""

    external_inverse_baseline = True

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        *,
        inverse_action_dim: int,
        inverse_hidden_dim: int,
        inverse_depth: int = 2,
        inverse_layer_norm: bool = False,
        inverse_activation: str = "relu",
        inverse_latent_dim: int = 192,
    ) -> None:
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
        )
        self.inverse_model = InverseActionHead(
            latent_dim=int(inverse_latent_dim),
            action_dim=int(inverse_action_dim),
            hidden_dim=int(inverse_hidden_dim),
            depth=int(inverse_depth),
            layer_norm=bool(inverse_layer_norm),
            activation=str(inverse_activation),
        )

    def predict_action(self, z_t: Tensor, z_next: Tensor) -> Tensor:
        return self.inverse_model(z_t, z_next)


def action_nce_loss(
    predicted_action: Tensor,
    target_action: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Official AC-MTM in-batch negative-squared-distance classifier."""

    if temperature <= 0:
        raise ValueError("Action-NCE temperature must be positive")
    predicted = predicted_action.flatten(0, 1).float()
    target = target_action.flatten(0, 1).float()
    if predicted.shape != target.shape:
        raise ValueError("predicted and target macro actions must have equal shape")
    if predicted.shape[0] < 2:
        raise ValueError("Action-NCE needs at least two transition candidates")
    logits = -(
        predicted[:, None, :] - target[None, :, :]
    ).square().mean(dim=-1) / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return loss, accuracy


def external_inverse_horizon_loss(
    model: Any,
    emb: Tensor,
    act_emb: Tensor,
    raw_action: Tensor,
    *,
    max_history: int,
    method: str,
    inverse_weight: float,
    action_nce_temperature: float = 0.1,
) -> dict[str, Tensor]:
    """Matched LeWM forward loss plus one published inverse objective."""

    if not bool(getattr(model, "external_inverse_baseline", False)):
        raise TypeError("external inverse loss requires ExternalInverseJEPA")
    if inverse_weight < 0:
        raise ValueError("inverse weight must be non-negative")
    if raw_action.shape[:2] != (emb.shape[0], emb.shape[1] - 1):
        raise ValueError("one normalized macro action is required per transition")

    # Match the L0 lane exactly: all five positive teacher-forced transitions,
    # moving encoder targets, and no free-run/terminal training term.
    output = aligned_horizon_loss(
        model,
        emb,
        act_emb,
        max_history=max_history,
        teacher_forced_weight=1.0,
        free_run_weight=0.0,
        terminal_weight=0.0,
    )
    forward_loss = output["pred_loss"]
    predicted_action = model.predict_action(emb[:, :-1], emb[:, 1:])
    method = str(method).lower().replace("_", "-")
    if method == "ac-mtm":
        inverse_loss, inverse_accuracy = action_nce_loss(
            predicted_action,
            raw_action,
            temperature=action_nce_temperature,
        )
    elif method == "smwm":
        inverse_loss = (predicted_action - raw_action.float()).square().mean()
        inverse_accuracy = None
    else:
        raise ValueError("external inverse method must be 'ac-mtm' or 'smwm'")

    output["base_pred_loss"] = forward_loss
    output["external_inverse_loss"] = inverse_loss
    output["external_inverse_weight"] = raw_action.new_tensor(inverse_weight)
    if inverse_accuracy is not None:
        output["external_inverse_acc1"] = inverse_accuracy
    output["pred_loss"] = forward_loss + float(inverse_weight) * inverse_loss
    return output


__all__ = [
    "ExternalInverseJEPA",
    "InverseActionHead",
    "action_nce_loss",
    "external_inverse_horizon_loss",
]
