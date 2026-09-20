"""Article-champion objectives on six-state/five-transition aligned clips."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from slewm.objectives.contrastive import symmetric_dcl_loss, dcl_idm_loss
from slewm.objectives.forward import (
    free_running_predictions,
    teacher_forced_one_step_predictions,
)






def _teacher_predictions(
    model: Any,
    emb: Tensor,
    act_emb: Tensor,
    *,
    max_history: int,
    context_drop_probability: float,
    training: bool,
) -> Tensor:
    if not training or context_drop_probability <= 0:
        return teacher_forced_one_step_predictions(
            model, emb, act_emb, max_history=max_history
        )
    if context_drop_probability >= 1:
        raise ValueError("context drop probability must be below one")
    predictions = []
    transitions = emb.shape[1] - 1
    for step in range(transitions):
        first = max(0, step - max_history + 1)
        states = emb[:, first : step + 1]
        actions = act_emb[:, first : step + 1]
        keep = (
            torch.rand_like(states[..., :1]) >= float(context_drop_probability)
        ).to(states.dtype)
        predictions.append(model.predict(states * keep, actions)[:, -1])
    return torch.stack(predictions, dim=1)


def _diffusion_rollout(
    model: Any,
    emb: Tensor,
    act_emb: Tensor,
    *,
    max_history: int,
    sigma_max: float,
    training: bool,
) -> Tensor:
    if sigma_max < 0:
        raise ValueError("diffusion sigma must be non-negative")
    transitions = emb.shape[1] - 1
    history = [emb[:, 0]]
    predictions = []
    scale = emb[:, :1].detach().std().clamp_min(1e-6)
    for step in range(transitions):
        first = max(0, len(history) - max_history)
        state_window = torch.stack(history[first:], dim=1)
        action_window = act_emb[:, first : step + 1]
        prediction = model.predict(state_window, action_window)[:, -1]
        predictions.append(prediction)
        feedback = prediction
        if training and sigma_max > 0 and step < transitions - 1:
            sigma = float(sigma_max) * (step + 1) / max(1, transitions - 1)
            feedback = feedback + torch.randn_like(feedback) * (sigma * scale)
        history.append(feedback)
    return torch.stack(predictions, dim=1)


def aligned_champion_horizon_loss(
    model: Any,
    emb: Tensor,
    act_emb: Tensor,
    raw_action: Tensor,
    *,
    max_history: int,
    current_epoch: int,
    training: bool,
    random_horizon: bool = True,
    context_drop_probability: float = 0.1,
    diffusion_sigma_max: float = 0.3,
    teacher_weight: float = 1.0,
    autoregressive_weight: float = 1.0,
    autoregressive_warmup_epochs: int = 2,
    inverse_dcl_weight: float = 0.1,
    inverse_dcl_temperature: float = 0.1,
    dcl_idm_weight_value: float = 0.05,
    dcl_idm_temperature: float = 0.1,
    dcl_idm_duplicate_cosine: float = 0.98,
    dcl_idm_symmetric: bool = False,
) -> dict[str, Tensor]:
    """Forward prediction + backward state DCL + inverse action InfoNCE.

    Inputs contain six states and five aligned actions. Auxiliary heads shape
    the encoder; the planner still uses the forward model's latent MSE.
    ``inverse_dcl_*`` are historical configuration names for the backward term.
    They are retained so archived recipes load without changing their meaning.
    """

    if not bool(getattr(model, "aligned_champion", False)):
        raise TypeError("aligned champion loss requires an aligned champion model")
    if emb.ndim != 3 or act_emb.ndim != 3 or raw_action.ndim != 3:
        raise ValueError("champion inputs must be rank-three tensors")
    transitions = emb.shape[1] - 1
    if transitions != 5 or act_emb.shape[1] != 5 or raw_action.shape[1] != 5:
        raise ValueError("article champion is frozen to six states/five transitions")
    if min(
        teacher_weight,
        autoregressive_weight,
        inverse_dcl_weight,
        dcl_idm_weight_value,
    ) < 0:
        raise ValueError("champion weights must be non-negative")

    target = emb[:, 1:]
    if bool(getattr(model, "detach_prediction_targets", False)):
        target = target.detach()
    teacher = _teacher_predictions(
        model,
        emb,
        act_emb,
        max_history=max_history,
        context_drop_probability=context_drop_probability,
        training=training,
    )
    noisy_rollout = _diffusion_rollout(
        model,
        emb,
        act_emb,
        max_history=max_history,
        sigma_max=diffusion_sigma_max,
        training=training,
    )
    # The deterministic rollout is the method-independent validation/control
    # diagnostic and the input to task ranking.  During validation it is equal
    # to noisy_rollout because diffusion is disabled.
    deterministic_rollout = (
        free_running_predictions(
            model, emb, act_emb, max_history=max_history
        )
        if training and diffusion_sigma_max > 0
        else noisy_rollout
    )
    n_actual = (
        int(torch.randint(1, transitions + 1, (1,), device=emb.device).item())
        if training and random_horizon
        else transitions
    )
    teacher_loss = (teacher - target).square().mean()
    autoregressive_loss = (
        noisy_rollout[:, :n_actual] - target[:, :n_actual]
    ).square().mean()
    rollout_loss = (deterministic_rollout - target).square().mean()
    terminal_loss = (
        deterministic_rollout[:, -1] - target[:, -1]
    ).square().mean()
    if autoregressive_warmup_epochs > 0 and current_epoch < autoregressive_warmup_epochs:
        active_ar_weight = float(autoregressive_weight) * current_epoch / max(
            1, autoregressive_warmup_epochs
        )
    else:
        active_ar_weight = float(autoregressive_weight)
    forward_loss = float(teacher_weight) * teacher_loss + active_ar_weight * autoregressive_loss

    backward_query = torch.cat((emb[:, 1:], act_emb), dim=-1)
    backward_loss = symmetric_dcl_loss(
        model.head_f(backward_query),
        model.head_g(emb[:, :-1]),
        temperature=inverse_dcl_temperature,
    )
    inverse_loss, inverse_accuracy, duplicate_fraction = dcl_idm_loss(
        model,
        emb,
        raw_action,
        temperature=dcl_idm_temperature,
        duplicate_cosine=dcl_idm_duplicate_cosine,
        symmetric=bool(dcl_idm_symmetric),
    )
    total = (
        forward_loss
        + float(inverse_dcl_weight) * backward_loss
        + float(dcl_idm_weight_value) * inverse_loss
    )
    return {
        "pred_loss": total,
        "base_pred_loss": forward_loss,
        "teacher_pred_loss": teacher_loss,
        "rollout_pred_loss": rollout_loss,
        "terminal_pred_loss": terminal_loss,
        "champion_autoregressive_loss": autoregressive_loss,
        "champion_inverse_dcl_loss": backward_loss,
        "champion_dcl_idm_loss": inverse_loss,
        "champion_dcl_idm_acc1": inverse_accuracy,
        "champion_dcl_idm_duplicate_fraction": duplicate_fraction,
        "champion_active_ar_weight": emb.new_tensor(active_ar_weight),
        "champion_n_actual": emb.new_tensor(float(n_actual)),
        "teacher_pred_emb": teacher,
        "rollout_pred_emb": deterministic_rollout,
    }


__all__ = [
    "aligned_champion_horizon_loss",
    "dcl_idm_loss",
    "symmetric_dcl_loss",
]
