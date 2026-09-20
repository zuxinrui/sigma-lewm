"""Unified IDM decoder for planning-time use (DTVH trigger + IPI-MPPI proposal).

Two trained IDM flavors exist across arms:
  regression  : q(cat(z_t, z_{t+1})) -> a (10-d standardized block)
                keys: idm_mlp. | q_idm_bn. | q_act.
  contrastive : head_f_idm(cat(z_t,z_{t+1})) . head_g_idm(a) / tau  (InfoNCE-trained score)
                keys: head_f_idm. | head_g_idm.   (e.g. the C_T_dcl_idm champion)

For the contrastive flavor, decoding is pool-softmax under the model's own conditional
    p(a | z, z') ∝ exp( f([z;z']) · g(a) / tau ),
approximated over a pool a_j ~ N(0, I) (valid: actions are StandardScaler-standardized, so
the marginal is ≈ N(0,I)):   â = Σ_j softmax_j(s_j) a_j.
The alignment score used by DTVH is the cosine between f([z;z']) and g(a_sent) — the exact
quantity InfoNCE trained to be high for true pairs.

All modules are built standalone from the RAW (unstripped) state_dict — no JEPA surgery.
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def select_kcenter_indices(
    features: torch.Tensor, size: int, *, seed: int = 0
) -> torch.Tensor:
    """Deterministic greedy cosine k-center subset on the feature device."""
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError("k-center features must be a non-empty matrix")
    count = min(max(int(size), 1), features.shape[0])
    normalized = torch.nn.functional.normalize(features.float(), dim=-1)
    generator = torch.Generator().manual_seed(int(seed))
    first = int(torch.randint(features.shape[0], (1,), generator=generator).item())
    selected = torch.empty(count, device=features.device, dtype=torch.long)
    selected[0] = first
    min_distance = 1.0 - normalized @ normalized[first]
    min_distance[first] = float("-inf")
    for slot in range(1, count):
        index = min_distance.argmax()
        selected[slot] = index
        distance = 1.0 - normalized @ normalized[index]
        min_distance = torch.minimum(min_distance, distance)
        min_distance[selected[: slot + 1]] = float("-inf")
    return selected


def _infer_dims(sd: dict) -> tuple[int | None, int | None]:
    """Read (D, A) off the trained IDM heads.

    The action dim is TASK-dependent (PushT 10, OGBench-Cube 25, ...) and the DCL-IDM's
    head_g_idm consumes the RAW action, so it must never be assumed: a wrong A silently
    fails as ``size mismatch for 0.weight`` when the ckpt is loaded into the template.
    """
    A = None
    for key, axis in (("q_idm_bn.3.weight", 0), ("q_act.3.weight", 0),
                      ("idm_mlp.2.weight", 0), ("head_g_idm.0.weight", 1)):
        if key in sd:
            A = int(sd[key].shape[axis])
            break
    D = None
    for key in ("head_f_idm.0.weight", "q_idm_bn.0.weight",
                "q_act.0.weight", "idm_mlp.0.weight"):
        if key in sd:                                   # all take cat(z_t, z_{t+1}) -> 2D
            D = int(sd[key].shape[1]) // 2
            break
    return D, A


def _seq_from_sd(sd: dict, prefix: str, template: nn.Sequential) -> nn.Sequential | None:
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not sub:
        return None
    template.load_state_dict(sub)
    template.eval()
    for p in template.parameters():
        p.requires_grad_(False)
    return template


class IDMDecoder:
    """decode(z, z_next) -> a_std (B, A);  score(z, z_next, a_std) -> alignment (B,)."""

    def __init__(self, state_dict: dict, D: int | None = None, A: int | None = None,
                 tau: float = 0.1, pool_size: int = 512, seed: int = 0,
                 device: str = "cuda", pool: torch.Tensor | None = None):
        """pool: optional (P, A) candidate actions in STANDARDIZED space. For contrastive
        IDMs the density-ratio interpretation of exp(f·g/τ) holds w.r.t. the TRAINING
        negative distribution = the empirical action marginal — so a dataset-sampled pool
        is the rigorous choice; N(0,I) is the fallback (matches mean/var but not shape).

        D/A default to whatever the checkpoint's own heads declare; pass them only to
        override."""
        D_sd, A_sd = _infer_dims(state_dict)
        D = int(D) if D is not None else (D_sd if D_sd is not None else 192)
        A = int(A) if A is not None else A_sd
        if A is None:
            raise ValueError(
                "cannot infer the action dim: state_dict has no recognisable IDM head "
                "(need q_idm_bn./q_act./idm_mlp. or head_g_idm.). Pass A= explicitly."
            )
        self.D, self.A, self.tau = D, A, float(tau)
        self.device = device
        self.mode = None

        # --- regression flavors (checked in preference order) ---
        for pref, net in [
            ("q_idm_bn.", nn.Sequential(nn.Linear(2 * D, 384), nn.BatchNorm1d(384),
                                        nn.ReLU(), nn.Linear(384, A))),
            ("q_act.",    nn.Sequential(nn.Linear(2 * D, 384), nn.BatchNorm1d(384),
                                        nn.ReLU(), nn.Linear(384, A))),
            ("idm_mlp.",  nn.Sequential(nn.Linear(2 * D, 256), nn.SiLU(),
                                        nn.Linear(256, A))),
        ]:
            m = None
            try:
                m = _seq_from_sd(state_dict, pref, net)
            except Exception:
                m = None
            if m is not None:
                self.reg = m.to(device)
                self.mode = "regression"
                break

        # --- contrastive flavor ---
        if self.mode is None:
            f = nn.Sequential(nn.Linear(2 * D, 256), nn.LayerNorm(256), nn.GELU(),
                              nn.Linear(256, 64))
            g = nn.Sequential(nn.Linear(A, 64), nn.LayerNorm(64), nn.GELU(),
                              nn.Linear(64, 64))
            fm = _seq_from_sd(state_dict, "head_f_idm.", f)
            gm = _seq_from_sd(state_dict, "head_g_idm.", g)
            if fm is not None and gm is not None:
                self.f, self.g = fm.to(device), gm.to(device)
                if pool is not None:
                    self.pool = pool.to(device).float()
                    self.pool_src = "dataset-marginal"
                else:
                    gen = torch.Generator().manual_seed(seed)
                    self.pool = torch.randn(pool_size, A, generator=gen).to(device)
                    self.pool_src = "N(0,I)-fallback"
                with torch.no_grad():
                    self.pool_g = torch.nn.functional.normalize(self.g(self.pool), dim=-1)
                self.mode = "contrastive"

        if self.mode is None:
            raise ValueError("state_dict has no IDM heads "
                             "(need q_idm_bn./q_act./idm_mlp. or head_f_idm.+head_g_idm.)")

    @torch.no_grad()
    def decode(self, z: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """(B,D),(B,D) -> (B,A) standardized action block."""
        pair = torch.cat([z, z_next], dim=-1)
        if self.mode == "regression":
            return self.reg(pair)
        q = torch.nn.functional.normalize(self.f(pair), dim=-1)          # (B,64)
        s = (q @ self.pool_g.T) / self.tau                                # (B,P)
        w = torch.softmax(s, dim=-1)
        return w @ self.pool                                              # (B,A)

    @torch.no_grad()
    def candidates(
        self,
        z: torch.Tensor,
        z_next: torch.Tensor,
        topk: int = 8,
        *,
        preselect: int | None = None,
        diversity_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return concrete action hypotheses and their IDM scores.

        A contrastive IDM represents a set-valued inverse, so averaging its
        posterior modes can produce an action that was never present in the
        dataset.  Planning code should use this method, then let the forward
        model verify the returned pool members.  Regression IDMs have one
        conditional-mean hypothesis and therefore return ``K=1``.

        Returns:
            actions: ``(B,K,A)`` standardized action blocks.
            scores: ``(B,K)``; larger is more compatible with ``(z,z_next)``.
        """
        pair = torch.cat([z, z_next], dim=-1)
        if self.mode == "regression":
            action = self.reg(pair).unsqueeze(1)
            return action, torch.zeros(
                action.shape[:2], device=action.device, dtype=action.dtype
            )

        q = torch.nn.functional.normalize(self.f(pair), dim=-1)
        scores = q @ self.pool_g.T
        k = min(max(int(topk), 1), self.pool.shape[0])
        if diversity_weight <= 0 or preselect is None or int(preselect) <= k:
            values, indices = scores.topk(k=k, dim=-1)
            return self.pool[indices], values

        if not 0 <= diversity_weight <= 1:
            raise ValueError("diversity_weight must lie in [0,1]")
        pre_k = min(max(int(preselect), k), self.pool.shape[0])
        values, indices = scores.topk(k=pre_k, dim=-1)
        actions = self.pool[indices]

        # MMR inside one fixed transition query.  First retain the best DCL
        # match, then trade normalized query quality against distance from the
        # already selected concrete action modes.  Because every candidate is
        # first restricted to query-local top-L, diversity cannot introduce an
        # arbitrary low-score action from the global bank.
        quality = (values - values[:, -1:]) / (
            values[:, :1] - values[:, -1:]
        ).clamp_min(1e-8)
        distance = torch.cdist(actions.float(), actions.float()) / max(
            float(self.A) ** 0.5, 1.0
        )
        distance_scale = distance.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        distance = distance / distance_scale

        chosen = torch.zeros(
            actions.shape[0], k, device=actions.device, dtype=torch.long
        )
        chosen[:, 0] = 0
        rows = torch.arange(actions.shape[0], device=actions.device)
        used = torch.zeros_like(values, dtype=torch.bool)
        used[:, 0] = True
        min_distance = distance[:, 0]
        weight = float(diversity_weight)
        for slot in range(1, k):
            objective = (1.0 - weight) * quality + weight * min_distance
            objective = objective.masked_fill(used, float("-inf"))
            pick = objective.argmax(dim=-1)
            chosen[:, slot] = pick
            used[rows, pick] = True
            min_distance = torch.minimum(
                min_distance, distance[rows, pick]
            )
        return actions[rows[:, None], chosen], values[rows[:, None], chosen]

    @torch.no_grad()
    def score(self, z: torch.Tensor, z_next: torch.Tensor, a_std: torch.Tensor) -> torch.Tensor:
        """Disagreement d in [0,2] (contrastive: 1-cos) or normalized L2 (regression).
        Higher = the sent action does NOT explain the observed transition."""
        pair = torch.cat([z, z_next], dim=-1)
        if self.mode == "regression":
            pred = self.reg(pair)
            return (pred - a_std).pow(2).mean(dim=-1)
        q = torch.nn.functional.normalize(self.f(pair), dim=-1)
        k = torch.nn.functional.normalize(self.g(a_std), dim=-1)
        return 1.0 - (q * k).sum(dim=-1)

    @torch.no_grad()
    def support_percentile(
        self, z: torch.Tensor, z_next: torch.Tensor, a_std: torch.Tensor
    ) -> torch.Tensor:
        """Query-local empirical rank of an action under the contrastive IDM.

        Raw InfoNCE logits from different transition queries are not globally
        calibrated because an arbitrary query-dependent offset cancels from
        the training loss.  This statistic compares the proposed action only
        with the fixed action bank under the *same* ``(z,z_next)`` query.  It
        is therefore invariant to that offset and lies in ``[0,1]``.
        """
        if self.mode != "contrastive":
            raise ValueError("support_percentile requires a contrastive IDM")
        pair = torch.cat([z, z_next], dim=-1)
        q = torch.nn.functional.normalize(self.f(pair), dim=-1)
        positive = (
            q * torch.nn.functional.normalize(self.g(a_std), dim=-1)
        ).sum(dim=-1)
        bank_scores = q @ self.pool_g.T
        return (bank_scores <= positive.unsqueeze(-1)).float().mean(dim=-1)


class BalancedIDMDecoder:
    """Query multiple empirical banks with an explicit per-bank quota.

    Concatenating a small and a large bank and taking one global top-k usually
    lets the large bank occupy every slot simply because it has more draws.
    This adapter preserves bank complementarity: each transition query gets
    concrete modes from every decoder, while the downstream forward model
    remains responsible for deciding which modes are useful.

    All wrapped decoders must be views of the same trained contrastive IDM.
    Their raw cosine scores therefore share a scale.  Query-local support is
    deliberately delegated to a fixed reference bank so adding a proposal
    bank cannot make its own gate easier to pass.
    """

    def __init__(
        self,
        decoders: list[IDMDecoder],
        *,
        retrieval_kwargs: list[dict] | None = None,
        reference: int = 0,
    ) -> None:
        if len(decoders) < 2:
            raise ValueError("BalancedIDMDecoder needs at least two decoders")
        if any(getattr(decoder, "mode", None) != "contrastive" for decoder in decoders):
            raise ValueError("all balanced-bank decoders must be contrastive")
        if not 0 <= int(reference) < len(decoders):
            raise ValueError("reference decoder index is out of range")
        action_dims = {int(decoder.A) for decoder in decoders}
        latent_dims = {int(decoder.D) for decoder in decoders}
        if len(action_dims) != 1 or len(latent_dims) != 1:
            raise ValueError("balanced-bank decoders must share action/latent dims")
        if retrieval_kwargs is None:
            retrieval_kwargs = [{} for _ in decoders]
        if len(retrieval_kwargs) != len(decoders):
            raise ValueError("retrieval_kwargs must have one entry per decoder")
        self.decoders = list(decoders)
        self.retrieval_kwargs = [dict(kwargs) for kwargs in retrieval_kwargs]
        self.reference = int(reference)
        self.mode = "contrastive"
        self.A = action_dims.pop()
        self.D = latent_dims.pop()
        self.device = decoders[self.reference].device
        self.pool_src = "balanced:" + "+".join(
            str(getattr(decoder, "pool_src", "unknown")) for decoder in decoders
        )

    @torch.no_grad()
    def candidates(
        self,
        z: torch.Tensor,
        z_next: torch.Tensor,
        topk: int = 8,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = max(int(topk), 1)
        banks = min(len(self.decoders), total)
        base, remainder = divmod(total, banks)
        actions = []
        scores = []
        for index in range(banks):
            quota = base + (1 if index < remainder else 0)
            bank_actions, bank_scores = self.decoders[index].candidates(
                z,
                z_next,
                topk=quota,
                **self.retrieval_kwargs[index],
            )
            actions.append(bank_actions)
            scores.append(bank_scores)
        return torch.cat(actions, dim=1), torch.cat(scores, dim=1)

    @torch.no_grad()
    def score(
        self, z: torch.Tensor, z_next: torch.Tensor, a_std: torch.Tensor
    ) -> torch.Tensor:
        return self.decoders[self.reference].score(z, z_next, a_std)

    @torch.no_grad()
    def support_percentile(
        self, z: torch.Tensor, z_next: torch.Tensor, a_std: torch.Tensor
    ) -> torch.Tensor:
        return self.decoders[self.reference].support_percentile(z, z_next, a_std)
