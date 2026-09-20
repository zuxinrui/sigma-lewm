from __future__ import annotations

import torch
from torch import nn

from slewm.objectives.paired_delta import residual_transition_pair_delta_loss
from slewm.models.residual import ResidualTransitionJEPA
from slewm.objectives.forward import aligned_horizon_loss, free_running_predictions


class AdditivePredictor(nn.Module):
    def forward(self, state, action):
        return action


def model(dim: int) -> ResidualTransitionJEPA:
    return ResidualTransitionJEPA(
        encoder=nn.Identity(),
        predictor=AdditivePredictor(),
        action_encoder=nn.Identity(),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
    )


def paired_batch(actions: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size = actions.shape[0]
    result = {
        "_pair_anchor_index": torch.tensor([0, 0, 1, 1]),
        "_pair_branch_index": torch.tensor([0, 1, 0, 1]),
        "pixels": torch.zeros(batch_size, 6, 3, 2, 2),
        "q": torch.zeros(batch_size, 6, 5),
        "morph_nodes": torch.zeros(batch_size, 6, 5, 8),
        "joint_mask": torch.ones(batch_size, 6, 5, dtype=torch.bool),
        "action": actions,
    }
    result["pixels"][2:] = 1
    result["q"][2:] = 1
    result["morph_nodes"][2:] = 1
    return result


def test_public_predict_is_absolute_and_private_head_is_delta():
    instance = model(3)
    state = torch.randn(2, 3, 3)
    action = torch.randn(2, 3, 3)
    assert torch.equal(instance.predict_delta(state, action), action)
    assert torch.equal(instance.predict(state, action), state + action)


def test_public_absolute_value_detaches_identity_but_not_delta_gradient():
    instance = model(3)
    state = torch.randn(2, 3, 3, requires_grad=True)
    action = torch.randn(2, 3, 3, requires_grad=True)
    instance.predict(state, action).sum().backward()
    # AdditivePredictor ignores state, so any state gradient here could only
    # have leaked through the forbidden identity base.
    assert state.grad is None
    torch.testing.assert_close(action.grad, torch.ones_like(action))


def test_aligned_residual_loss_detaches_future_targets_without_losing_actions():
    instance = model(2)
    states = torch.randn(3, 6, 2, requires_grad=True)
    actions = torch.randn(3, 5, 2, requires_grad=True)
    result = aligned_horizon_loss(
        instance,
        states,
        actions,
        max_history=3,
        teacher_forced_weight=1.0,
        free_run_weight=1.0,
        terminal_weight=1.0,
    )
    result["pred_loss"].backward()
    # With a delta head that ignores context, neither target nor identity-base
    # gradients may reach the state encoder.  Action supervision remains live.
    assert states.grad is None
    assert actions.grad is not None
    assert bool((actions.grad.abs().sum(dim=(0, 2)) > 0).all())


def test_training_free_run_equals_existing_semantic_audit_call_path():
    torch.manual_seed(19)
    instance = model(3)
    actions = torch.randn(4, 5, 3)
    actions[1] += 0.7
    actions[3] -= 0.5
    z0 = torch.randn(2, 3).repeat_interleave(2, dim=0)
    states = torch.cat(
        (z0[:, None], z0[:, None] + actions.cumsum(dim=1)), dim=1
    )
    result = residual_transition_pair_delta_loss(
        instance,
        paired_batch(actions),
        states,
        actions,
        max_history=3,
    )
    semantic_rollout = free_running_predictions(
        instance, states, actions, max_history=3
    )
    assert torch.equal(result["rollout_pred_emb"], semantic_rollout)
    assert torch.allclose(semantic_rollout, states[:, 1:], atol=1e-6)


def test_every_horizon_action_receives_endpoint_gradient():
    instance = model(2)
    actions = torch.randn(4, 5, 2, requires_grad=True)
    actions.data[1] += 0.6
    actions.data[3] -= 0.4
    z0 = torch.randn(2, 2).repeat_interleave(2, dim=0)
    target_actions = actions.detach() * 1.1
    states = torch.cat(
        (z0[:, None], z0[:, None] + target_actions.cumsum(dim=1)), dim=1
    )
    result = residual_transition_pair_delta_loss(
        instance,
        paired_batch(actions),
        states,
        actions,
        max_history=3,
    )
    result["terminal_pred_loss"].backward()
    per_step = actions.grad.abs().sum(dim=(0, 2))
    assert bool((per_step > 0).all())
