"""Source-only residual transition and exact-pair effect losses.

This is an opt-in diagnostic objective.  It does not alter the released v2.2
absolute-latent objective in :mod:`world_model_losses`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


EPS = 1e-6


def _require_pair_layout(batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Fail closed unless adjacent rows are the two branches of one anchor."""

    required = {
        "_pair_anchor_index",
        "_pair_branch_index",
        "pixels",
        "action",
        "q",
        "morph_nodes",
        "joint_mask",
    }
    missing = required - set(batch)
    if missing:
        raise ValueError(f"pair-delta batch missing {sorted(missing)}")
    anchors = batch["_pair_anchor_index"]
    branches = batch["_pair_branch_index"]
    if anchors.ndim != 1 or branches.ndim != 1 or anchors.shape != branches.shape:
        raise ValueError("pair identities must be one-dimensional and aligned")
    if anchors.numel() < 2 or anchors.numel() % 2:
        raise ValueError("pair-delta batch size must be positive and even")
    left = torch.arange(0, anchors.numel(), 2, device=anchors.device)
    right = left + 1
    if not torch.equal(anchors[left], anchors[right]):
        raise ValueError("adjacent rows do not share an exact anchor identity")
    observed_branches = torch.stack((branches[left], branches[right]), dim=1)
    if not torch.equal(
        observed_branches.sort(dim=1).values,
        torch.tensor([0, 1], device=branches.device).expand_as(observed_branches),
    ):
        raise ValueError("each adjacent pair must contain branches {0,1}")

    # Metadata alone is insufficient: verify the defining physical invariant
    # after preprocessing.  This also catches malicious/reordered metadata.
    for key in ("pixels", "q", "morph_nodes", "joint_mask"):
        value = batch[key]
        if value.shape[0] != anchors.numel() or value.shape[1] < 1:
            raise ValueError(f"{key} does not carry an initial state per row")
        if not torch.equal(value[left, 0], value[right, 0]):
            raise ValueError(f"paired initial {key} is not bitwise identical")
    action = batch["action"]
    if action.shape[0] != anchors.numel():
        raise ValueError("action batch and pair identities differ")
    same_action = (
        action[left].float() == action[right].float()
    ).reshape(len(left), -1).all(dim=1)
    if bool(same_action.any()):
        raise ValueError("same-anchor branches must have different actions")
    return left, right


def direction_log_norm_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    eps: float = EPS,
) -> Tensor:
    """Match a vector's direction and magnitude without fitted statistics."""

    if predicted.shape != target.shape or predicted.ndim < 2:
        raise ValueError("predicted and target effect tensors must align")
    if not predicted.is_floating_point() or not target.is_floating_point():
        raise TypeError("effect tensors must be floating point")
    predicted_f = predicted.float()
    target_f = target.detach().float()
    direction = 1.0 - F.cosine_similarity(
        predicted_f, target_f, dim=-1, eps=eps
    )
    predicted_norm = predicted_f.norm(dim=-1)
    target_norm = target_f.norm(dim=-1)
    magnitude = F.smooth_l1_loss(
        torch.log(predicted_norm + eps),
        torch.log(target_norm + eps),
        reduction="none",
    )
    return (direction + magnitude).mean()


def _residual_teacher_predictions(
    model: Any,
    states: Tensor,
    action_embeddings: Tensor,
) -> tuple[Tensor, Tensor]:
    batch, state_count, dim = states.shape
    transitions = state_count - 1
    current_context = states[:, :-1]
    current_base = current_context.detach()
    flat_current = current_context.reshape(batch * transitions, 1, dim)
    flat_action = action_embeddings.reshape(batch * transitions, 1, dim)
    if not callable(getattr(model, "predict_delta", None)):
        raise TypeError("E2 model must expose predict_delta")
    predicted_delta = model.predict_delta(flat_current, flat_action)[
        :, 0
    ].reshape(
        batch, transitions, dim
    )
    predicted_state = current_base + predicted_delta
    return predicted_state, predicted_delta


def _residual_free_run(
    model: Any,
    states: Tensor,
    action_embeddings: Tensor,
    *,
    max_history: int,
) -> tuple[Tensor, Tensor]:
    if max_history < 1:
        raise ValueError("max_history must be positive")
    # The observed state is a differentiable predictor context, while the
    # residual identity base is detached.  This is the JEPA asymmetry:
    # prediction gradients train the context encoder, but cannot win through
    # an identity shortcut or flow into the target branch.
    history = [states[:, 0]]
    predicted_states: list[Tensor] = []
    predicted_deltas: list[Tensor] = []
    for step in range(action_embeddings.shape[1]):
        first = max(0, len(history) - max_history)
        state_window = torch.stack(history[first:], dim=1)
        action_window = action_embeddings[:, first : step + 1]
        if state_window.shape[1] != action_window.shape[1]:
            raise AssertionError("residual rollout windows are misaligned")
        # The observed initial state is detached, but predicted recurrent
        # states retain their graph so the h=5 loss reaches every earlier
        # action.  Detaching here would silently reduce the endpoint objective
        # to the final action only.
        if not callable(getattr(model, "predict_delta", None)):
            raise TypeError("E2 model must expose predict_delta")
        delta = model.predict_delta(state_window, action_window)[:, -1]
        base = history[-1].detach() if step == 0 else history[-1]
        next_state = base + delta
        predicted_deltas.append(delta)
        predicted_states.append(next_state)
        history.append(next_state)
    return torch.stack(predicted_states, dim=1), torch.stack(
        predicted_deltas, dim=1
    )


def residual_transition_pair_delta_loss(
    model: Any,
    batch: Mapping[str, Tensor],
    states: Tensor,
    action_embeddings: Tensor,
    *,
    max_history: int,
    pair_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Compute the preregistered E2 objective on exact adjacent pairs.

    Target latents and the observed residual identity base are stop-gradient.
    The current latent remains a differentiable input to ``predict_delta`` so
    the dynamics objective trains the context encoder as well as the
    predictor/action interface.  This prevents both a moving SIGReg-only
    representation and a trivial identity-gradient shortcut.
    """

    if states.ndim != 3 or action_embeddings.ndim != 3:
        raise ValueError("states/actions must have shape [B,T,D]")
    if states.shape[0] != action_embeddings.shape[0]:
        raise ValueError("state/action batch sizes differ")
    if states.shape[1] != action_embeddings.shape[1] + 1:
        raise ValueError("E2 requires T+1 states and T actions")
    if states.shape[-1] != action_embeddings.shape[-1]:
        raise ValueError("state/action embedding widths differ")
    if float(pair_weight) < 0:
        raise ValueError("pair_weight must be nonnegative")
    left, right = _require_pair_layout(batch)

    target_states = states.detach()
    target_step_delta = target_states[:, 1:] - target_states[:, :-1]
    teacher_states, teacher_delta = _residual_teacher_predictions(
        model, states, action_embeddings
    )
    rollout_states, rollout_delta = _residual_free_run(
        model,
        states,
        action_embeddings,
        max_history=max_history,
    )
    teacher_loss = direction_log_norm_loss(teacher_delta, target_step_delta)
    rollout_effect = rollout_states - target_states[:, :1]
    target_effect = target_states[:, 1:] - target_states[:, :1]
    rollout_loss = direction_log_norm_loss(rollout_effect, target_effect)
    terminal_loss = direction_log_norm_loss(
        rollout_effect[:, -1], target_effect[:, -1]
    )

    predicted_pair_delta = rollout_states[left, -1] - rollout_states[right, -1]
    target_pair_delta = (
        target_states[left, -1] - target_states[right, -1]
    )
    pair_loss = direction_log_norm_loss(
        predicted_pair_delta, target_pair_delta
    )
    dynamics_loss = (teacher_loss + rollout_loss + terminal_loss) / 3.0
    prediction_loss = dynamics_loss + float(pair_weight) * pair_loss
    return {
        "pred_loss": prediction_loss,
        "teacher_pred_loss": teacher_loss,
        "rollout_pred_loss": rollout_loss,
        "terminal_pred_loss": terminal_loss,
        "pair_delta_loss": pair_loss,
        "teacher_pred_emb": teacher_states,
        "rollout_pred_emb": rollout_states,
        "predicted_step_delta_rms": rollout_delta.float().square().mean().sqrt(),
        "target_step_delta_rms": target_step_delta.float().square().mean().sqrt(),
        "predicted_pair_delta_rms": (
            predicted_pair_delta.float().square().mean().sqrt()
        ),
        "target_pair_delta_rms": target_pair_delta.float().square().mean().sqrt(),
    }


def action_gradient_rms(loss: Tensor, action_embeddings: Tensor) -> Tensor:
    """Measure, but do not optimize or tune on, the dynamics gradient to U."""

    gradient = torch.autograd.grad(
        loss,
        action_embeddings,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0]
    return gradient.detach().float().square().mean().sqrt()


__all__ = [
    "action_gradient_rms",
    "direction_log_norm_loss",
    "residual_transition_pair_delta_loss",
]
