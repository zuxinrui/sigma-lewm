"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode_action(self, action, context=None):
        """Encode actions while preserving legacy checkpoint behavior.

        Released LeWM action encoders accept only one tensor. Cross-morphology
        interfaces additionally need public morphology and physical state.
        Such encoders opt in with ``uses_context = True``; every legacy encoder
        follows the exact original call path.
        """

        if getattr(self.action_encoder, "uses_context", False):
            if context is None:
                raise ValueError("contextual action encoder requires context")
            return self.action_encoder(action, context=context)
        return self.action_encoder(action)

    def _candidate_action_context(self, initial, num_candidates):
        """Repeat only the context fields requested by the action encoder."""

        keys = tuple(getattr(self.action_encoder, "context_keys", ()))
        optional_keys = tuple(
            getattr(self.action_encoder, "optional_context_keys", ())
        )
        context = {}
        for key in keys + optional_keys:
            if key not in initial:
                if key in optional_keys:
                    continue
                raise KeyError(
                    f"contextual action encoder requires info[{key!r}]"
                )
            value = initial[key]
            if not torch.is_tensor(value):
                raise TypeError(f"info[{key!r}] must be a tensor")
            context[key] = value.repeat_interleave(num_candidates, dim=0)
        return context

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        parameter = next(self.encoder.parameters(), None)
        pixels = info['pixels'].to(
            parameter.dtype if parameter is not None else info['pixels'].dtype
        )
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.encode_action(info["action"], info)

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        if history_size is None:
            history_size = getattr(self.predictor, "num_frames", 3)
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # Encode initial state once, or reuse the cache across CEM iterations.
        initial = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        if "emb" not in info:
            encoded = self.encode(dict(initial))
            info["emb"] = (
                encoded["emb"].detach().unsqueeze(1).expand(B, S, -1, -1)
            )
        action_context = self._candidate_action_context(initial, S)

        # flatten batch and sample dimensions for rollout
        emb_init = rearrange(info["emb"], "b s ... -> (b s) ...")
        act_flat = rearrange(act_0, "b s ... -> (b s) ...")
        act_future_flat = rearrange(act_future, "b s ... -> (b s) ...")
        all_act_emb = self.encode_action(
            torch.cat((act_flat, act_future_flat), dim=1),
            action_context,
        )

        # Roll out n_steps future transitions plus the final state. Encoding
        # the whole candidate plan once is materially faster inside CEM.
        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))
        for t in range(n_steps + 1):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_act_emb[:, lo : H + t]
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])
        emb = torch.stack(emb_list, dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        # ``get_cost`` encodes one goal sequence per environment, yielding
        # [B,T,D]. CEM adds a distinct candidate dimension [B,S,...]. The
        # released implementation relied on right-aligned broadcasting, which
        # only works accidentally for B=1 (or B=S). Insert the candidate axis
        # explicitly so batched held-out-morphology evaluation is correct.
        if goal_emb.ndim == pred_emb.ndim - 1:
            goal_emb = goal_emb.unsqueeze(1)
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        parameter = next(self.parameters(), None)
        device = parameter.device if parameter is not None else action_candidates.device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        if "goal_emb" not in info_dict:
            goal = {
                k: v[:, 0]
                for k, v in info_dict.items()
                if torch.is_tensor(v)
            }
            goal["pixels"] = goal["goal"]

            for k in info_dict:
                if k.startswith("goal_"):
                    goal[k[len("goal_") :]] = goal.pop(k)

            goal.pop("action")
            goal = self.encode(goal)
            info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost
