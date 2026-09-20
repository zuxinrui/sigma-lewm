"""Auditable aligned one-step and free-running h-step JEPA losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


def _validate_transition_sequence(emb: Tensor, act_emb: Tensor) -> int:
    if emb.ndim != 3 or act_emb.ndim != 3:
        raise ValueError("emb and act_emb must both have shape [B,T,D]")
    if emb.shape[0] != act_emb.shape[0]:
        raise ValueError("state/action batch sizes differ")
    transitions = emb.shape[1] - 1
    if transitions < 1 or act_emb.shape[1] != transitions:
        raise ValueError(
            "an aligned sequence requires T+1 states and exactly T actions"
        )
    if emb.shape[-1] != act_emb.shape[-1]:
        raise ValueError("state/action embedding dimensions differ")
    return transitions


def teacher_forced_one_step_predictions(
    model: Any,
    emb: Tensor,
    act_emb: Tensor,
    *,
    max_history: int,
) -> Tensor:
    """Predict every ``z_{t+1}`` from a capped real state/action prefix.

    This mirrors the released Cube objective: predictor positions 0, 1 and 2
    are all supervised from genuine encoded observations when
    ``max_history=3``.  It is deliberately separate from free-running rollout,
    whose history after the first state consists only of model predictions.
    """

    transitions = _validate_transition_sequence(emb, act_emb)
    if max_history < 1:
        raise ValueError("max_history must be positive")
    predictions = []
    for step in range(transitions):
        first = max(0, step - max_history + 1)
        state_window = emb[:, first : step + 1]
        action_window = act_emb[:, first : step + 1]
        if state_window.shape != action_window.shape:
            raise AssertionError("teacher state/action windows are misaligned")
        predicted = model.predict(state_window, action_window)
        if predicted.shape != state_window.shape:
            raise ValueError("model.predict changed the teacher-window shape")
        predictions.append(predicted[:, -1])
    return torch.stack(predictions, dim=1)


def free_running_predictions(
    model: Any,
    emb: Tensor,
    act_emb: Tensor,
    *,
    max_history: int,
) -> Tensor:
    """Roll out from the real initial state using only previous predictions."""

    transitions = _validate_transition_sequence(emb, act_emb)
    if max_history < 1:
        raise ValueError("max_history must be positive")
    history = [emb[:, 0]]
    predictions = []
    for step in range(transitions):
        first = max(0, len(history) - max_history)
        state_window = torch.stack(history[first:], dim=1)
        action_window = act_emb[:, first : step + 1]
        if state_window.shape[1] != action_window.shape[1]:
            raise AssertionError("free-run state/action windows are misaligned")
        next_state = model.predict(state_window, action_window)[:, -1]
        predictions.append(next_state)
        history.append(next_state)
    return torch.stack(predictions, dim=1)


def aligned_horizon_loss(
    model: Any,
    emb: Tensor,
    act_emb: Tensor,
    *,
    max_history: int,
    teacher_forced_weight: float = 1.0,
    free_run_weight: float = 1.0,
    terminal_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Compute a simple loss matched to full-commit h-step CEM inference."""

    if min(teacher_forced_weight, free_run_weight, terminal_weight) < 0:
        raise ValueError("loss weights must be nonnegative")
    if teacher_forced_weight + free_run_weight + terminal_weight <= 0:
        raise ValueError("at least one prediction loss weight must be positive")
    target = emb[:, 1:]
    # ResidualTransitionJEPA has an exact identity path.  Unlike the released
    # Cube absolute predictor, allowing that model's future target branch to
    # move under prediction loss makes temporal persistence a direct shortcut.
    # The model explicitly opts in so legacy/absolute LeWM training remains
    # bit-for-bit unchanged.
    if bool(getattr(model, "detach_prediction_targets", False)):
        target = target.detach()
    teacher = teacher_forced_one_step_predictions(
        model, emb, act_emb, max_history=max_history
    )
    rollout = free_running_predictions(
        model, emb, act_emb, max_history=max_history
    )
    teacher_loss = (teacher - target).square().mean()
    rollout_loss = (rollout - target).square().mean()
    terminal_loss = (rollout[:, -1] - target[:, -1]).square().mean()
    total_weight = (
        teacher_forced_weight + free_run_weight + terminal_weight
    )
    prediction_loss = (
        teacher_forced_weight * teacher_loss
        + free_run_weight * rollout_loss
        + terminal_weight * terminal_loss
    ) / total_weight
    return {
        "pred_loss": prediction_loss,
        "teacher_pred_loss": teacher_loss,
        "rollout_pred_loss": rollout_loss,
        "terminal_pred_loss": terminal_loss,
        "teacher_pred_emb": teacher,
        "rollout_pred_emb": rollout,
    }
















__all__ = [
    "aligned_horizon_loss",
    "free_running_predictions",
    "teacher_forced_one_step_predictions",
    "action_contrastive_horizon_loss",
    "action_group_negative_actions",
    "delay_action_sequence",
    "cross_episode_derangement",
    "franka_terminal_listwise_loss",
]
