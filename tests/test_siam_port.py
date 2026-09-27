"""The siam-jepa / Janus port (slewm.models.siam, slewm.objectives.siam) against le-wm (19e4f60):
identical losses and gradients on identical inputs, the Janus-OS recipe, stop-gradient in both
directions, and the relaxed window length used for the standard tasks (e.g. OGBench-Scene)."""
import copy
import os
import sys
from pathlib import Path

import pytest
import torch
from stable_pretraining.backbone.utils import vit_hf

from slewm.models.components import MLP, ARPredictor, Embedder
from slewm.models.siam import SiamJEPA
from slewm.objectives.siam import siam_horizon_loss

D, A = 192, 20
LEWM = Path(os.environ.get("LEWM_ROOT", "/scratch-shared/mprzystupa1/le-wm"))

# Janus-OS (le-wm config/train/paper_gear_janus_os.yaml and its parents)
JANUS_OS = dict(random_horizon=False, context_drop_probability=0.0, diffusion_sigma_max=0.0,
                teacher_weight=1.0, autoregressive_weight=0.0, forward_weight=0.5, backward_weight=0.5,
                inverse_weight=1.0, distance="mse", stop_gradient=True, backward_recipe="one_step")


def predictor_cfg():
    return dict(num_frames=3, input_dim=D, hidden_dim=D, output_dim=D, depth=2, heads=4, mlp_dim=256,
                dim_head=32, dropout=0.0)


def parts(seed=0):
    torch.manual_seed(seed)
    return dict(
        encoder=vit_hf(size="tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False),
        predictor=ARPredictor(**predictor_cfg()),
        action_encoder=Embedder(input_dim=A, emb_dim=D),
        projector=MLP(input_dim=D, hidden_dim=256, output_dim=D, norm_fn=torch.nn.BatchNorm1d),
        pred_proj=MLP(input_dim=D, hidden_dim=256, output_dim=D, norm_fn=torch.nn.BatchNorm1d),
    )


def janus(cls, p):
    torch.manual_seed(1)
    return cls(**copy.deepcopy(p), siam_embed_dim=D, siam_action_input_dim=A, siam_sphere=False,
               siam_inverse="dcl_idm", siam_cosine_planning=False, siam_backward_arch="transformer",
               siam_backward_predictor_cfg=ARPredictor(**predictor_cfg()),
               siam_backward_proj_cfg=MLP(input_dim=D, hidden_dim=256, output_dim=D, norm_fn=torch.nn.BatchNorm1d))


def inputs(t=6, seed=2):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(8, t, D, generator=g, requires_grad=True)
    act = torch.randn(8, t - 1, D, generator=g, requires_grad=True)
    raw = torch.randn(8, t - 1, A, generator=g)
    return emb, act, raw


def run(loss_fn, model, t=6, **kw):
    emb, act, raw = inputs(t)
    torch.manual_seed(3)
    out = loss_fn(model, emb, act, raw, max_history=3, current_epoch=5, training=True, **{**JANUS_OS, **kw})
    out["pred_loss"].backward()
    return out, emb.grad, act.grad


@pytest.mark.skipif(not (LEWM / "siam_jepa_losses.py").exists(), reason="le-wm checkout not available")
def test_janus_os_matches_le_wm():
    sys.path.insert(0, str(LEWM))
    try:
        from siam_jepa_losses import siam_horizon_loss as lewm_loss
        from siam_jepa_model import SiamJEPA as LewmSiam
    finally:
        sys.path.remove(str(LEWM))
    p = parts()
    ours, lewm = janus(SiamJEPA, p).train(), janus(LewmSiam, p).train()
    lewm.load_state_dict(ours.state_dict())
    a, ga, gaa = run(siam_horizon_loss, ours)
    b, gb, gbb = run(lewm_loss, lewm)
    for key in ("pred_loss", "siam_forward_loss", "siam_backward_loss", "siam_inverse_loss"):
        assert torch.allclose(a[key], b[key], atol=1e-6), key
    assert torch.allclose(ga, gb, atol=1e-6) and torch.allclose(gaa, gbb, atol=1e-6)


def test_stop_gradient_in_both_directions():
    """With both targets detached, the prediction terms reach the states only through the predictor
    inputs: a state that is only ever a target (the last one forward, the first one backward) gets no
    gradient from them."""
    model = janus(SiamJEPA, parts()).train()
    emb, act, raw = inputs()
    out = siam_horizon_loss(model, emb, act, raw, max_history=3, current_epoch=5, training=True,
                            **{**JANUS_OS, "inverse_weight": 0.0})
    torch.autograd.grad(out["pred_loss"], emb, retain_graph=True)
    fwd = torch.autograd.grad(out["siam_forward_loss"], emb, retain_graph=True)[0]
    bwd = torch.autograd.grad(out["siam_backward_loss"], emb)[0]
    assert fwd[:, -1].abs().sum() == 0 and fwd[:, 0].abs().sum() > 0
    assert bwd[:, 0].abs().sum() == 0 and bwd[:, -1].abs().sum() > 0


@pytest.mark.parametrize("t", [2, 4, 6])
def test_any_window_length(t):
    out, g, _ = run(siam_horizon_loss, janus(SiamJEPA, parts()).train(), t=t)
    assert torch.isfinite(out["pred_loss"]) and torch.isfinite(g).all()


def test_rejects_misaligned_actions():
    emb, act, raw = inputs()
    with pytest.raises(ValueError):
        siam_horizon_loss(janus(SiamJEPA, parts()), emb, act[:, :-1], raw, max_history=3, current_epoch=0,
                          training=True, **JANUS_OS)
