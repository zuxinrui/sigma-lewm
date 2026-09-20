import torch

from trim.baselines.inverse import (
    action_nce_loss,
    external_inverse_horizon_loss,
)


class ExactAdditiveInverse(torch.nn.Module):
    external_inverse_baseline = True

    def predict(self, state, action):
        return state + action

    def predict_action(self, z_t, z_next):
        return z_next - z_t


def _exact_batch():
    action = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[-1.0], [-2.0], [-4.0]],
        ]
    )
    states = [torch.zeros(2, 1)]
    for step in range(action.shape[1]):
        states.append(states[-1] + action[:, step])
    return torch.stack(states, dim=1), action


def test_action_nce_identifies_distinct_exact_actions():
    target = torch.tensor([[[0.0], [1.0]], [[2.0], [3.0]]])
    loss, accuracy = action_nce_loss(target, target, temperature=0.1)
    assert torch.isfinite(loss)
    assert accuracy.item() == 1.0


def test_acmtm_uses_macro_action_nce_without_sigreg():
    emb, action = _exact_batch()
    output = external_inverse_horizon_loss(
        ExactAdditiveInverse(),
        emb,
        action,
        action,
        max_history=3,
        method="ac-mtm",
        inverse_weight=0.3,
        action_nce_temperature=0.1,
    )
    assert output["teacher_pred_loss"].item() == 0.0
    assert output["external_inverse_acc1"].item() == 1.0
    torch.testing.assert_close(
        output["pred_loss"], 0.3 * output["external_inverse_loss"]
    )


def test_smwm_regresses_the_complete_macro_action():
    emb, action = _exact_batch()
    output = external_inverse_horizon_loss(
        ExactAdditiveInverse(),
        emb,
        action,
        action,
        max_history=3,
        method="smwm",
        inverse_weight=1.0,
    )
    assert output["external_inverse_loss"].item() == 0.0
    assert output["pred_loss"].item() == 0.0
    assert "external_inverse_acc1" not in output
