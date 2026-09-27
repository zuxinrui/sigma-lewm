"""Port of le-wm siam_jepa_model.py (19e4f60), unchanged apart from the JEPA import.

SiamJEPA: LeWM with SimSiam-style forward/backward prediction (Chen & He, arXiv 2011.10566).

The architecture is the released LeWM one (ViT-tiny encoder, BatchNorm-hidden projector, AR
transformer predictor + pred_proj). Two things are added:

* ``head_bwd`` - the backward predictor h_b([z_(t+1); e(a_t)]) -> z_t, built exactly like the
  SimSiam predictor: Linear(bias=False) -> BatchNorm1d -> ReLU -> Linear, a 1/4 bottleneck of its
  input width, and no BatchNorm on the output (SimSiam Table 3d: output BN makes training unstable).
  The forward predictor needs no new module: the AR predictor + pred_proj is already a learnable,
  non-identity h whose output carries no BN, which is what SimSiam requires of h (Table 1).
* an optional sphere constraint, z = sqrt(d) z~/|z~|, on every encoded state and every predicted
  state. The loss is scale-invariant (negative cosine), so the constraint does not change it; it
  keeps rolled-out predictions, which are fed back to the predictor, on the same shell as encoded
  states, and makes the planner's MSE cost equal to 2d(1 - cos).

Training-only inverse heads (``siam_inverse``) are attached when an inverse term is used; the
dcl_idm heads are the champion's, so that arm matches our current design.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from slewm.models.world_model import JEPA

SIAM_INVERSES = ("none", "dcl_idm")
# Encoder input (Phase-SIAM, 2026-09-27; not in le-wm): "single" is le-wm's o_t; the two-frame modes
# feed the encoder six channels, [o_t, o_t - o_(t-1)] ("obs_velocity") or [o_t, o_(t-1)] ("frame_pair").
SIAM_INPUT_MODES = ("single", "obs_velocity", "frame_pair")


class SiamJEPA(JEPA):
    siam_jepa = True

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        *,
        siam_embed_dim: int = 192,
        siam_action_input_dim: int = 1,
        siam_sphere: bool = True,
        siam_backward_bottleneck: float = 0.25,
        siam_inverse: str = "none",
        siam_projection_dim: int = 64,
        siam_cosine_planning: bool = True,
        siam_backward_arch: str = "mlp",
        siam_backward_predictor_cfg: dict | None = None,
        siam_backward_proj_cfg: dict | None = None,
        siam_backward_seed: int = 7919,
        siam_input_mode: str = "single",
    ) -> None:
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
        )
        d = int(siam_embed_dim)
        if siam_inverse not in SIAM_INVERSES:
            raise ValueError(f"siam_inverse must be one of {SIAM_INVERSES}, got {siam_inverse!r}")
        self.siam_embed_dim = d
        self.siam_sphere = bool(siam_sphere)
        self.siam_inverse = str(siam_inverse)
        self.siam_cosine_planning = bool(siam_cosine_planning)
        if siam_backward_arch not in ("mlp", "transformer"):
            raise ValueError(f"siam_backward_arch must be mlp or transformer, got {siam_backward_arch!r}")
        self.siam_backward_arch = siam_backward_arch
        if siam_backward_arch == "mlp":
            hidden = max(8, int(round(2 * d * float(siam_backward_bottleneck))))
            self.head_bwd = nn.Sequential(
                nn.Linear(2 * d, hidden, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, d),
            )
        else:
            # Janus-JEPA: a backward predictor built exactly like the forward one (same AR transformer
            # config and the same pred_proj MLP), run on time-reversed sequences; its own RNG stream,
            # so the shared components keep their per-component seeded initialisation.
            import hydra
            if siam_backward_predictor_cfg is None or siam_backward_proj_cfg is None:
                raise ValueError("siam_backward_arch=transformer needs the predictor and pred_proj configs")
            build = lambda c: c if isinstance(c, nn.Module) else hydra.utils.instantiate(c)  # config.json path
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(siam_backward_seed))
                self.backward_predictor = build(siam_backward_predictor_cfg)
                self.backward_proj = build(siam_backward_proj_cfg)
        if siam_input_mode not in SIAM_INPUT_MODES:
            raise ValueError(f"siam_input_mode must be one of {SIAM_INPUT_MODES}, got {siam_input_mode!r}")
        self.siam_input_mode = siam_input_mode
        if siam_input_mode != "single":
            with torch.random.fork_rng(devices=[]):
                self._widen_patch_embedding()
        if self.siam_inverse == "dcl_idm":
            # identical to aligned_champion_model._attach_champion_heads(idm_key="head")
            self.champion_idm_key = "head"
            self.head_f_idm = nn.Sequential(
                nn.Linear(2 * d, 256), nn.LayerNorm(256), nn.GELU(),
                nn.Linear(256, int(siam_projection_dim)),
            )
            self.head_g_idm = nn.Sequential(
                nn.Linear(int(siam_action_input_dim), 64), nn.LayerNorm(64), nn.GELU(),
                nn.Linear(64, int(siam_projection_dim)),
            )

    def sphere(self, x: torch.Tensor) -> torch.Tensor:
        if not self.siam_sphere:
            return x
        return F.normalize(x, dim=-1) * math.sqrt(x.shape[-1])

    def _widen_patch_embedding(self) -> None:
        """3 -> 6 input channels. The current frame keeps the pretrained-style RGB weights and the
        second half (velocity or previous frame) starts at zero, so at initialisation both two-frame
        modes compute the single-frame encoder exactly."""
        patch = self.encoder.embeddings.patch_embeddings
        old = patch.projection
        new = nn.Conv2d(2 * old.in_channels, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, : old.in_channels] = old.weight
            if old.bias is not None:
                new.bias.copy_(old.bias)
        patch.projection = new
        patch.num_channels = new.in_channels
        self.encoder.config.num_channels = new.in_channels

    def encode(self, info):
        if self.siam_input_mode == "single":
            info = super().encode(info)
            info["emb"] = self.sphere(info["emb"])
            return info
        # Two-frame state: z_k = E([o_k, o_k - o_(k-1)]) or E([o_k, o_(k-1)]). The predecessor of the
        # first frame comes either from info["pixels_prev"] [B, 1, C, H, W] (aligned clips) or from one
        # extra leading frame in info["pixels"] (standard windows, planning), which is then consumed.
        frames = info["pixels"]
        if "pixels_prev" in info:
            frames = torch.cat((info["pixels_prev"].to(frames.dtype), frames), dim=1)
        elif frames.shape[1] < 2:
            raise ValueError("two-frame encoding needs the previous frame: pixels_prev or T+1 frames")
        elif "action" in info and info["action"].shape[1] == frames.shape[1]:
            info["action"] = info["action"][:, 1:]        # the leading frame's action has no transition
        current, previous = frames[:, 1:], frames[:, :-1]
        second = current - previous if self.siam_input_mode == "obs_velocity" else previous
        staged = dict(info)
        staged["pixels"] = torch.cat((current, second), dim=2)
        staged.pop("pixels_prev", None)
        staged = super().encode(staged)
        info["emb"] = self.sphere(staged["emb"])
        if "act_emb" in staged:
            info["act_emb"] = staged["act_emb"]
        return info

    def rollout(self, info, action_sequence, history_size: int = None):
        if self.siam_input_mode == "single":
            return super().rollout(info, action_sequence, history_size)
        # Planning passes [o_(t-1), o_t] (one more frame than embeddings): encode it here and show
        # JEPA.rollout one frame per embedding, so its history length and action split stay right.
        if "emb" not in info:
            b, s = action_sequence.shape[:2]
            initial = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v) and k != "action"}
            encoded = self.encode({"pixels": initial["pixels"]})
            info["emb"] = encoded["emb"].detach().unsqueeze(1).expand(b, s, -1, -1)
        pixels = info["pixels"]
        info["pixels"] = pixels[:, :, 1:]
        try:
            return super().rollout(info, action_sequence, history_size)
        finally:
            info["pixels"] = pixels

    def predict(self, emb, act_emb):
        return self.sphere(super().predict(emb, act_emb))

    def criterion(self, info_dict: dict):
        """Planner cost. With cosine planning both sides are put on the sqrt(d) sphere first, so the
        released MSE cost becomes 2d(1 - cos) - the training distance - whether or not the states are
        sphere-constrained during training (a no-op when they are)."""
        if not self.siam_cosine_planning:
            return super().criterion(info_dict)
        info = dict(info_dict)
        for key in ("predicted_emb", "goal_emb"):
            v = info[key]
            info[key] = F.normalize(v, dim=-1) * math.sqrt(v.shape[-1])
        return super().criterion(info)

    def predict_reversed(self, emb_rev: torch.Tensor, act_rev: torch.Tensor) -> torch.Tensor:
        """Janus backward predictor on a time-reversed window: position j holds a state and the action
        that led *into* it; the output at j is its predecessor. (B, T, d) -> (B, T, d)."""
        preds = self.backward_predictor(emb_rev, act_rev)
        preds = self.backward_proj(rearrange(preds, "b t d -> (b t) d"))
        return self.sphere(rearrange(preds, "(b t) d -> b t d", b=emb_rev.size(0)))

    def predict_backward(self, emb_next: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """One backward step per transition: h_b([z_(t+1); e(a_t)]) for every t; (B, T, d) -> (B, T, d).
        With the transformer backward predictor every transition is a length-1 reversed window."""
        b = emb_next.shape[0]
        if self.siam_backward_arch == "transformer":
            flat = lambda x: rearrange(x, "b t d -> (b t) 1 d")
            return rearrange(self.predict_reversed(flat(emb_next), flat(act_emb)), "(b t) 1 d -> b t d", b=b)
        x = rearrange(torch.cat((emb_next, act_emb), dim=-1), "b t d -> (b t) d")
        return rearrange(self.head_bwd(x), "(b t) d -> b t d", b=b)
