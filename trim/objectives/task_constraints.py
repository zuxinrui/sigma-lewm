"""Optional task-specific action and persistence constraints."""
from __future__ import annotations
from typing import Any
import torch
from torch import Tensor
import torch.nn.functional as F
from trim.objectives.forward import free_running_predictions, teacher_forced_one_step_predictions, _validate_transition_sequence

def cross_episode_derangement(episode_index: Tensor) -> Tensor:
    """Return deterministic donors with no same-episode pairs.

    A bijective derangement exists only when no episode occupies more than half
    of the batch.  Validation tail batches can legitimately violate that
    condition, so we prefer a cyclic permutation when one exists and otherwise
    reuse donors from another episode.  Reuse is still a valid cross-episode
    negative; silently pairing an item with its own episode is not.
    """

    episodes = episode_index.reshape(-1)
    if episodes.numel() < 2:
        raise ValueError("cross-episode action negatives require batch size >= 2")
    indices = torch.arange(episodes.numel(), device=episodes.device)
    for shift in range(1, episodes.numel()):
        donors = torch.roll(indices, shifts=shift)
        if bool(torch.all(episodes != episodes[donors])):
            return donors
    if torch.unique(episodes).numel() < 2:
        raise ValueError(
            "cross-episode action negatives require at least two source episodes"
        )
    donors = torch.empty_like(indices)
    for target in range(episodes.numel()):
        offsets = (target + torch.arange(1, episodes.numel(), device=episodes.device))
        candidates = indices[offsets.remainder(episodes.numel())]
        valid = candidates[episodes[candidates] != episodes[target]]
        donors[target] = valid[0]
    return donors

def action_group_negative_actions(
    action: Tensor,
    cross_action: Tensor,
    temporal_action: Tensor,
    *,
    action_schema: str,
) -> dict[str, Tensor]:
    """Build phase-controlled negatives for each physical action group.

    The final action dimension is one flattened five-controller-step macro.
    Franka arm negatives retain the true gripper sequence, so a release-timing
    shortcut cannot stand in for sensitivity to the six continuous controls.
    """

    raw_dim = {"gear_effective4": 4, "franka_stack": 7}.get(action_schema)
    if raw_dim is None:
        raise ValueError(f"unsupported action schema {action_schema!r}")
    if action.shape != cross_action.shape or action.shape != temporal_action.shape:
        raise ValueError("true and negative actions must have identical shapes")
    if action.shape[-1] % raw_dim:
        raise ValueError(
            f"flattened action width {action.shape[-1]} is not divisible by {raw_dim}"
        )
    block = action.shape[-1] // raw_dim
    true = action.reshape(*action.shape[:-1], block, raw_dim)
    cross = cross_action.reshape_as(true)
    temporal = temporal_action.reshape_as(true)

    if action_schema == "gear_effective4":
        cross_xyz = true.clone()
        cross_xyz[..., :3] = cross[..., :3]
        cross_yaw = true.clone()
        cross_yaw[..., 3] = cross[..., 3]
        temporal_xyz = true.clone()
        temporal_xyz[..., :3] = temporal[..., :3]
        temporal_yaw = true.clone()
        temporal_yaw[..., 3] = temporal[..., 3]
        variants = {
            "cross_episode_xyz_only": cross_xyz,
            "cross_episode_yaw_only": cross_yaw,
            "temporal_xyz_only": temporal_xyz,
            "temporal_yaw_only": temporal_yaw,
        }
    else:
        cross_arm = true.clone()
        cross_arm[..., :6] = cross[..., :6]
        temporal_gripper = true.clone()
        temporal_gripper[..., 6] = temporal[..., 6]
        variants = {
            "cross_episode_arm6_only_phase_matched": cross_arm,
            "temporal_gripper_only": temporal_gripper,
        }
    return {name: value.reshape_as(action) for name, value in variants.items()}

def delay_action_sequence(action: Tensor) -> Tensor:
    """Delay macro actions by one step without cyclic wraparound."""

    if action.ndim < 2 or action.shape[1] < 2:
        raise ValueError("temporal action delay requires at least two macro steps")
    return torch.cat((action[:, :1], action[:, :-1]), dim=1)

def _shift_franka_release(action: Tensor, *, direction: str) -> Tensor:
    """Advance or delay the 25-step gripper trace without cyclic wraparound."""

    if action.ndim != 3 or action.shape[-1] % 7:
        raise ValueError("Franka release shift expects flattened 7D action blocks")
    block = action.shape[-1] // 7
    raw = action.reshape(action.shape[0], action.shape[1] * block, 7)
    shifted = raw.clone()
    if direction == "delay":
        shifted[:, 1:, 6] = raw[:, :-1, 6]
        shifted[:, 0, 6] = raw[:, 0, 6]
    elif direction == "advance":
        shifted[:, :-1, 6] = raw[:, 1:, 6]
        shifted[:, -1, 6] = raw[:, -1, 6]
    else:
        raise ValueError("release direction must be advance or delay")
    return shifted.reshape_as(action)

def _franka_arm_baseline_actions(
    action: Tensor,
    standardized_zero_action: Tensor | None,
) -> dict[str, Tensor]:
    """Build no-motion and low-motion arms while retaining release timing."""

    raw = action.reshape(action.shape[0], -1, 7)
    if standardized_zero_action is None:
        zero = raw.new_zeros(7)
    else:
        zero = torch.as_tensor(
            standardized_zero_action, device=raw.device, dtype=raw.dtype
        ).reshape(-1)
        if zero.numel() != 7:
            raise ValueError("standardized Franka zero action must have width 7")
    zero_arm = raw.clone()
    zero_arm[..., :6] = zero[:6]
    low_motion = raw.clone()
    low_motion[..., :6] = zero[:6] + 0.25 * (
        raw[..., :6] - zero[:6]
    )
    return {
        "zero_arm_same_gripper_phase": zero_arm.reshape_as(action),
        "quarter_arm_same_gripper_phase": low_motion.reshape_as(action),
    }

def franka_terminal_listwise_loss(
    model: Any,
    emb: Tensor,
    action: Tensor,
    true_rollout: Tensor,
    episode_index: Tensor,
    *,
    max_history: int,
    temperature: float = 0.1,
    scale_floor: float = 1.0e-4,
    standardized_zero_action: Tensor | None = None,
) -> dict[str, Tensor]:
    """Rank the demonstrated Franka terminal plan above phase-valid negatives.

    The candidate set never cyclically wraps a release.  Arm donors preserve
    the target gripper phase; release negatives change only gripper timing by
    one raw controller step. Errors are normalized by the larger of the
    frozen-state terminal distance and detached median candidate error. The
    latter prevents a nearly static untrained encoder from turning tiny
    action-dependent differences into arbitrarily large cross-entropy.
    Persistence is a scale reference, not a ranked negative.
    """

    if temperature <= 0 or scale_floor <= 0:
        raise ValueError("Franka listwise temperature/scale floor must be positive")
    if emb.ndim != 3 or action.ndim != 3:
        raise ValueError("Franka listwise inputs must have shape [B,T,D]")
    if action.shape[:2] != (emb.shape[0], emb.shape[1] - 1):
        raise ValueError("Franka listwise loss requires T+1 states and T actions")
    if action.shape[-1] % 7:
        raise ValueError("Franka listwise actions must contain flattened 7D blocks")
    if true_rollout.shape != emb[:, 1:].shape:
        raise ValueError("Franka true rollout does not align with targets")

    negatives = {
        "release_delay_one_raw": _shift_franka_release(action, direction="delay"),
        "release_advance_one_raw": _shift_franka_release(action, direction="advance"),
        "macro_causal_delay": delay_action_sequence(action),
    }
    negatives.update(
        _franka_arm_baseline_actions(action, standardized_zero_action)
    )
    episodes = episode_index.reshape(-1)
    if torch.unique(episodes).numel() >= 2:
        donor = action[cross_episode_derangement(episodes)]
        grouped = action_group_negative_actions(
            action,
            donor,
            delay_action_sequence(action),
            action_schema="franka_stack",
        )
        negatives["arm6_same_gripper_phase"] = grouped[
            "cross_episode_arm6_only_phase_matched"
        ]

    negative_rollouts = {
        name: free_running_predictions(
            model,
            emb,
            model.encode_action(candidate),
            max_history=max_history,
        )
        for name, candidate in negatives.items()
    }
    target = emb[:, -1].detach()
    true_error = (true_rollout[:, -1] - target).square().mean(dim=-1)
    negative_errors = {
        name: (rollout[:, -1] - target).square().mean(dim=-1)
        for name, rollout in negative_rollouts.items()
    }
    persistence_error = (emb[:, 0].detach() - target).square().mean(dim=-1)
    ordered_names = list(negative_errors)
    errors = torch.stack(
        [true_error, *(negative_errors[name] for name in ordered_names)], dim=1
    )
    # ``Tensor.median(dim=...)`` requests an arg-index kernel that PyTorch does
    # not implement deterministically on CUDA.  The listwise scale needs only
    # the lower median value, so an explicit deterministic sort is equivalent
    # for both odd and even candidate counts and keeps formal deterministic
    # training fail-closed rather than silently switching to warn-only mode.
    sorted_errors = torch.sort(errors.detach(), dim=1).values
    candidate_scale = sorted_errors[:, (errors.shape[1] - 1) // 2]
    scale = torch.maximum(persistence_error, candidate_scale).clamp_min(
        float(scale_floor)
    )
    logits = -errors / scale.unsqueeze(1) / float(temperature)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        accuracy = (logits.argmax(dim=1) == 0).float().mean()
        best_negative = torch.stack(
            [negative_errors[name] for name in ordered_names], dim=1
        ).min(dim=1).values
        margin = ((best_negative - true_error) / scale).mean()
    output = {
        "franka_terminal_listwise_loss": loss,
        "franka_terminal_listwise_accuracy": accuracy,
        "franka_terminal_listwise_margin": margin,
        "franka_terminal_true_error": true_error.mean(),
        "franka_terminal_persistence_scale": scale.mean(),
        "franka_terminal_frozen_state_error": persistence_error.mean(),
        "franka_terminal_candidate_error_scale": candidate_scale.mean(),
    }
    for name in ordered_names:
        output[f"franka_terminal_{name}_error"] = negative_errors[name].mean()
    return output

def action_contrastive_horizon_loss(
    model: Any,
    emb: Tensor,
    action: Tensor,
    true_rollout: Tensor,
    episode_index: Tensor,
    *,
    max_history: int,
    margin_fraction: float = 0.1,
    margin_floor: float = 1.0e-4,
    persistence_weight: float = 0.0,
    action_schema: str | None = None,
) -> dict[str, Tensor]:
    """Rank aligned actions above counterfactual and persistence baselines.

    This is an observational-data regularizer, not a claim that the donors are
    exact simulator counterfactuals.  The held-out action gate and closed-loop
    goal ablation remain mandatory.  Scaling the hinge margin by the frozen
    persist-initial error keeps it meaningful as the latent scale evolves.
    """

    if margin_fraction < 0 or margin_floor < 0 or persistence_weight < 0:
        raise ValueError("action-ranking margins and weights must be non-negative")
    if emb.ndim != 3 or action.ndim != 3:
        raise ValueError("emb and action must both have shape [B,T,D]")
    transitions = emb.shape[1] - 1
    if (
        transitions < 1
        or action.shape[0] != emb.shape[0]
        or action.shape[1] != transitions
    ):
        raise ValueError("action ranking requires T+1 states and T actions")
    if true_rollout.shape != emb[:, 1:].shape:
        raise ValueError("true rollout and target state sequence do not align")
    if transitions < 2:
        raise ValueError("temporal action negatives require at least two transitions")

    has_cross_episode = bool(torch.unique(episode_index.reshape(-1)).numel() >= 2)
    cross_action = (
        action[cross_episode_derangement(episode_index)]
        if has_cross_episode
        else action
    )
    # A cyclic roll can turn a terminal Franka close→open release into an
    # impossible open→close→open sequence. Repeating the first action and
    # delaying the rest is causal and remains within the demonstrated global
    # single-switch support.
    temporal_action = delay_action_sequence(action)
    if action_schema is None:
        negative_actions = {
            "cross_episode_all": cross_action,
            "temporal_all": temporal_action,
        }
    else:
        negative_actions = action_group_negative_actions(
            action,
            cross_action,
            temporal_action,
            action_schema=action_schema,
        )
    if not has_cross_episode:
        # A sequential validation tail can contain clips from only one source
        # episode. There is no honest cross-episode donor in that batch. Keep
        # its causal temporal/persistence diagnostics and give the unavailable
        # cross term zero optimization weight instead of silently using a
        # same-episode donor.
        negative_actions = {
            name: value
            for name, value in negative_actions.items()
            if not name.startswith("cross_episode_")
        }
    negative_rollouts = {
        name: free_running_predictions(
            model,
            emb,
            model.encode_action(negative),
            max_history=max_history,
        )
        for name, negative in negative_actions.items()
    }

    target = emb[:, 1:].detach()
    true_error = (true_rollout - target).square().mean(dim=-1)
    negative_errors = {
        name: (rollout - target).square().mean(dim=-1)
        for name, rollout in negative_rollouts.items()
    }
    persist_error = (
        emb[:, :1].detach().expand_as(target) - target
    ).square().mean(dim=-1)
    margin = (float(margin_fraction) * persist_error).clamp_min(
        float(margin_floor)
    )
    negative_ranks = {
        name: F.relu(margin + true_error - error).mean()
        for name, error in negative_errors.items()
    }
    cross_values = [
        value for name, value in negative_ranks.items() if name.startswith("cross_episode_")
    ]
    temporal_values = [
        value for name, value in negative_ranks.items() if name.startswith("temporal_")
    ]
    if not temporal_values:
        raise AssertionError("action negatives must include a temporal group")
    zero = true_error.new_zeros(())
    cross_rank = torch.stack(cross_values).mean() if cross_values else zero
    temporal_rank = torch.stack(temporal_values).mean()
    persistence_rank = F.relu(margin + true_error - persist_error).mean()
    rank_denominator = (
        float(has_cross_episode) + 1.0 + float(persistence_weight)
    )
    output = {
        "action_rank_loss": (
            cross_rank
            + temporal_rank
            + float(persistence_weight) * persistence_rank
        ) / rank_denominator,
        "action_rank_cross_episode_loss": cross_rank,
        "action_rank_temporal_loss": temporal_rank,
        "action_rank_persistence_loss": persistence_rank,
        "action_cross_episode_effect_rms": torch.stack(
            [
                (true_rollout - rollout).square().mean().sqrt()
                for name, rollout in negative_rollouts.items()
                if name.startswith("cross_episode_")
            ]
        ).mean() if cross_values else zero,
        "action_temporal_effect_rms": torch.stack(
            [
                (true_rollout - rollout).square().mean().sqrt()
                for name, rollout in negative_rollouts.items()
                if name.startswith("temporal_")
            ]
        ).mean(),
    }
    for name, rank in negative_ranks.items():
        output[f"action_rank_{name}_loss"] = rank
    return output
