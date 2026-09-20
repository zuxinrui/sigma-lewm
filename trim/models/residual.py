"""Residual-semantics JEPA with an absolute public prediction interface."""

from __future__ import annotations

from einops import rearrange

from trim.models.world_model import JEPA


class ResidualTransitionJEPA(JEPA):
    """Interpret the unchanged predictor head as a transition delta.

    ``predict_delta`` is the training primitive.  Public ``predict`` returns an
    absolute next latent, preserving the contract consumed by JEPA.rollout,
    CEM, and the existing source semantic evaluator.
    """

    prediction_semantics = "absolute_next_state_via_residual_delta_v1"
    # The residual head must learn physical change through ``predict_delta``.
    # Letting the absolute identity base or the future target carry prediction
    # gradients creates a direct persistence shortcut that does not exist in
    # the released Cube model's absolute predictor.
    detach_prediction_targets = True

    def predict_delta(self, emb, act_emb):
        if emb.shape != act_emb.shape:
            raise ValueError(
                "residual transition prediction requires aligned state/action "
                f"embeddings, got {tuple(emb.shape)} and {tuple(act_emb.shape)}"
            )
        delta = self.predictor(emb, act_emb)
        delta = self.pred_proj(rearrange(delta, "b t d -> (b t) d"))
        return rearrange(delta, "(b t) d -> b t d", b=emb.size(0))

    def predict(self, emb, act_emb):
        """Return absolute next latents without an identity-gradient shortcut.

        Detaching the base changes no forward value and therefore preserves
        every rollout/planner/checkpoint inference contract.  Gradients still
        reach the context encoder through ``predict_delta`` and still traverse
        earlier predicted deltas in a free-running rollout.
        """

        return emb.detach() + self.predict_delta(emb, act_emb)


__all__ = ["ResidualTransitionJEPA"]
