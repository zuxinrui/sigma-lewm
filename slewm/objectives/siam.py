"""Port of le-wm siam_jepa_losses.py (19e4f60): imports from slewm, the six-state check relaxed to
any clip (for the standard-task windows), and dcl_idm_loss without the act_emb key argument (SiamJEPA
uses the raw-action "head" key, the default in both repositories).

siam-jepa objective on six-state / five-transition aligned clips (Gear, Cube-Stacking).

Forward and backward follow SimSiam (Chen & He, arXiv 2011.10566; facebookresearch/simsiam):

    D(p, z) = 1 - cos(p, stopgrad(z))          SimSiam's -cos shifted by 1 (same gradient), so
                                               every logged component stays non-negative for the
                                               loss-EMA epoch selector
    L_fwd   = D(h_f(z_<=t, a_<=t), z_(t+1))    h_f = the AR predictor + pred_proj
    L_bwd   = D(h_b([z_(t+1); e(a_t)]), z_t)   h_b = SimSiam predictor MLP (siam_jepa_model.py)
    L_siam  = 1/2 L_fwd + 1/2 L_bwd            SimSiam's symmetrised loss, the view swap read as
                                               time reversal

The forward term keeps the c0 recipe around it - teacher forcing with context drop plus a noisy
autoregressive rollout with warm-up and a random horizon - and changes only the distance and the
stop-gradient on the target. The optional inverse term (dcl_idm) is the champion's, unchanged,
and like every inverse objective lets its gradient reach the encoder.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from slewm.objectives.triadic import _teacher_predictions
from slewm.objectives.contrastive import dcl_idm_loss, symmetric_dcl_loss
from slewm.objectives.forward import free_running_predictions


def siam_distance(prediction: Tensor, target: Tensor, *, kind: str = "cos", stop_gradient: bool = True) -> Tensor:
    """SimSiam D: 1 - cos(p, stopgrad(z)), averaged. Ablations (2026-09-26): kind="mse" uses the
    LeWM squared error mean((p - z)^2) instead; stop_gradient=False lets the target branch move
    (then anti-collapse must come from elsewhere, e.g. SIGReg)."""
    target = target.detach() if stop_gradient else target
    if kind == "mse":
        return (prediction.float() - target.float()).pow(2).mean()
    if kind != "cos":
        raise ValueError(f"unknown siam distance {kind!r}")
    return 1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1).mean()


def _noisy_rollout(model, emb, act_emb, *, max_history, sigma_max, training):
    transitions = emb.shape[1] - 1
    history = [emb[:, 0]]
    predictions = []
    scale = emb[:, :1].detach().std().clamp_min(1e-6)
    for step in range(transitions):
        first = max(0, len(history) - max_history)
        prediction = model.predict(torch.stack(history[first:], dim=1), act_emb[:, first : step + 1])[:, -1]
        predictions.append(prediction)
        feedback = prediction
        if training and sigma_max > 0 and step < transitions - 1:
            sigma = float(sigma_max) * (step + 1) / max(1, transitions - 1)
            feedback = model.sphere(feedback + torch.randn_like(feedback) * (sigma * scale))
        history.append(feedback)
    return torch.stack(predictions, dim=1)


@torch.no_grad()
def collapse_monitor(emb: Tensor) -> dict[str, Tensor]:
    """SimSiam Sec. 4.1: per-channel std of z/|z| over the batch; ~1/sqrt(d) healthy, -> 0 collapsed."""
    unit = F.normalize(emb.reshape(-1, emb.shape[-1]).float(), dim=-1)
    std = unit.std(dim=0).mean()
    return {"siam_unit_std": std, "siam_unit_std_ratio": std * emb.shape[-1] ** 0.5}


class _ReversedView:
    """Presents the Janus backward predictor through the forward helpers' interface (predict, sphere)."""

    def __init__(self, model):
        self._m = model

    def predict(self, emb, act_emb):
        return self._m.predict_reversed(emb, act_emb)

    def sphere(self, x):
        return self._m.sphere(x)


def siam_horizon_loss(
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
    forward_weight: float = 0.5,
    backward_weight: float = 0.5,
    inverse_weight: float = 1.0,
    dcl_idm_temperature: float = 0.1,
    dcl_idm_duplicate_cosine: float = 0.98,
    dcl_idm_symmetric: bool = False,
    distance: str = "cos",
    stop_gradient: bool = True,
    backward_recipe: str = "one_step",
    forward_stop_gradient: bool | None = None,
    backward_contrastive_weight: float = 0.0,
    backward_contrastive_temperature: float = 0.1,
) -> dict[str, Tensor]:
    if not bool(getattr(model, "siam_jepa", False)):
        raise TypeError("siam loss requires siam_jepa_model.SiamJEPA")
    transitions = emb.shape[1] - 1
    # le-wm fixes six states / five transitions (Gear, Cube-Stacking); the standard-task windows
    # (e.g. OGBench-Scene, history 3 + 1) are shorter. Any clip works with one action per transition.
    if transitions < 1 or act_emb.shape[1] != transitions or raw_action.shape[1] != transitions:
        raise ValueError("siam-jepa needs T states and T - 1 transition actions")

    target = emb[:, 1:]
    teacher = _teacher_predictions(
        model, emb, act_emb, max_history=max_history,
        context_drop_probability=context_drop_probability, training=training,
    )
    noisy = _noisy_rollout(model, emb, act_emb, max_history=max_history,
                           sigma_max=diffusion_sigma_max, training=training)
    deterministic = (
        free_running_predictions(model, emb, act_emb, max_history=max_history)
        if training and diffusion_sigma_max > 0 else noisy
    )
    n_actual = (
        int(torch.randint(1, transitions + 1, (1,), device=emb.device).item())
        if training and random_horizon else transitions
    )
    # Stop-gradient per direction (2026-09-27): `stop_gradient` is the backward default; the forward
    # target may instead receive gradient, as Sigma-LeWM's forward MSE does.
    fsg = stop_gradient if forward_stop_gradient is None else bool(forward_stop_gradient)
    D = lambda p, z: siam_distance(p, z, kind=distance, stop_gradient=stop_gradient)
    DF = lambda p, z: siam_distance(p, z, kind=distance, stop_gradient=fsg)
    teacher_loss = DF(teacher, target)
    autoregressive_loss = DF(noisy[:, :n_actual], target[:, :n_actual])
    with torch.no_grad():
        rollout_loss = D(deterministic, target)
        terminal_loss = D(deterministic[:, -1], target[:, -1])
    if autoregressive_warmup_epochs > 0 and current_epoch < autoregressive_warmup_epochs:
        ar_weight = float(autoregressive_weight) * current_epoch / max(1, autoregressive_warmup_epochs)
    else:
        ar_weight = float(autoregressive_weight)
    forward_loss = float(teacher_weight) * teacher_loss + ar_weight * autoregressive_loss

    if getattr(model, "siam_backward_arch", "mlp") == "transformer":
        # Janus: the backward predictor gets the forward recipe on the time-reversed clip. Reversed
        # position j holds z_(5-j) and the action a_(4-j) that led into it.
        rev_emb, rev_act = emb.flip(1), act_emb.flip(1)
        proxy = _ReversedView(model)
        rev_teacher = _teacher_predictions(
            proxy, rev_emb, rev_act, max_history=max_history,
            context_drop_probability=context_drop_probability if backward_recipe == "multi_step" else 0.0,
            training=training,
        )
        backward_loss = float(teacher_weight) * D(rev_teacher, rev_emb[:, 1:])
        bwd_pred, bwd_true = rev_teacher, rev_emb[:, 1:]
        if backward_recipe == "multi_step":
            rev_noisy = _noisy_rollout(proxy, rev_emb, rev_act, max_history=max_history,
                                       sigma_max=diffusion_sigma_max, training=training)
            backward_loss = backward_loss + ar_weight * D(rev_noisy[:, :n_actual], rev_emb[:, 1:][:, :n_actual])
        elif backward_recipe != "one_step":
            raise ValueError(f"unknown backward recipe {backward_recipe!r}")
    else:
        bwd_pred = model.predict_backward(emb[:, 1:], act_emb)
        bwd_true = emb[:, :-1]
        backward_loss = D(bwd_pred, bwd_true)
    backward_contrastive = emb.new_zeros(())
    if backward_contrastive_weight > 0:
        backward_contrastive = symmetric_dcl_loss(bwd_pred, bwd_true, temperature=backward_contrastive_temperature)
        backward_loss = backward_loss + float(backward_contrastive_weight) * backward_contrastive

    total = float(forward_weight) * forward_loss + float(backward_weight) * backward_loss
    out = {}
    if getattr(model, "siam_inverse", "none") == "dcl_idm":
        idm, idm_acc, dup = dcl_idm_loss(
            model, emb, raw_action, temperature=dcl_idm_temperature,
            duplicate_cosine=dcl_idm_duplicate_cosine, symmetric=bool(dcl_idm_symmetric),
        )
        total = total + float(inverse_weight) * idm
        out.update(siam_inverse_loss=idm, siam_inverse_acc1=idm_acc, siam_inverse_duplicate_fraction=dup)
    out.update(
        pred_loss=total,
        base_pred_loss=forward_loss,
        teacher_pred_loss=teacher_loss,
        rollout_pred_loss=rollout_loss,
        terminal_pred_loss=terminal_loss,
        siam_forward_loss=forward_loss,
        siam_backward_loss=backward_loss,
        siam_backward_contrastive=backward_contrastive,
        siam_autoregressive_loss=autoregressive_loss,
        siam_active_ar_weight=emb.new_tensor(ar_weight),
        teacher_pred_emb=teacher,
        rollout_pred_emb=deterministic,
    )
    out.update(collapse_monitor(emb))
    return out


__all__ = ["siam_distance", "siam_horizon_loss", "collapse_monitor"]
