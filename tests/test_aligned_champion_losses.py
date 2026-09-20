import torch

from trim.objectives.triadic import aligned_champion_horizon_loss


class _ToyChampion(torch.nn.Module):
    aligned_champion = True

    def __init__(self, dim=8, action_dim=6, projection_dim=4):
        super().__init__()
        self.transition = torch.nn.Linear(dim, dim, bias=False)
        self.head_f = torch.nn.Linear(2 * dim, projection_dim)
        self.head_g = torch.nn.Linear(dim, projection_dim)
        self.head_f_idm = torch.nn.Linear(2 * dim, projection_dim)
        self.head_g_idm = torch.nn.Linear(action_dim, projection_dim)

    def predict(self, emb, act_emb):
        return self.transition(emb) + act_emb


def test_aligned_champion_consumes_five_real_actions_without_placeholder():
    torch.manual_seed(7)
    model = _ToyChampion()
    emb = torch.randn(4, 6, 8, requires_grad=True)
    act_emb = torch.randn(4, 5, 8, requires_grad=True)
    raw_action = torch.randn(4, 5, 6)

    output = aligned_champion_horizon_loss(
        model,
        emb,
        act_emb,
        raw_action,
        max_history=3,
        current_epoch=3,
        training=False,
    )

    assert output["teacher_pred_emb"].shape == (4, 5, 8)
    assert output["rollout_pred_emb"].shape == (4, 5, 8)
    assert output["champion_n_actual"].item() == 5
    assert all(
        torch.isfinite(value).all()
        for key, value in output.items()
        if "emb" not in key
    )
    output["pred_loss"].backward()
    assert emb.grad is not None
    assert act_emb.grad is not None


def test_aligned_champion_rejects_legacy_terminal_placeholder_shape():
    model = _ToyChampion()
    emb = torch.randn(3, 6, 8)
    act_emb = torch.randn(3, 6, 8)
    raw_action = torch.randn(3, 6, 6)

    try:
        aligned_champion_horizon_loss(
            model,
            emb,
            act_emb,
            raw_action,
            max_history=3,
            current_epoch=0,
            training=True,
        )
    except ValueError as error:
        assert "six states/five transitions" in str(error)
    else:
        raise AssertionError("legacy state/action cardinality was accepted")


def test_autoregressive_warmup_is_epoch_indexed_and_validation_uses_full_h5():
    torch.manual_seed(11)
    model = _ToyChampion()
    emb = torch.randn(4, 6, 8)
    act_emb = torch.randn(4, 5, 8)
    raw_action = torch.randn(4, 5, 6)

    epoch0 = aligned_champion_horizon_loss(
        model,
        emb,
        act_emb,
        raw_action,
        max_history=3,
        current_epoch=0,
        training=True,
        diffusion_sigma_max=0.0,
    )
    validation = aligned_champion_horizon_loss(
        model,
        emb,
        act_emb,
        raw_action,
        max_history=3,
        current_epoch=0,
        training=False,
        diffusion_sigma_max=0.0,
    )

    assert epoch0["champion_active_ar_weight"].item() == 0.0
    assert validation["champion_n_actual"].item() == 5
    assert validation["champion_active_ar_weight"].item() == 0.0
