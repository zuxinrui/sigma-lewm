"""Contrastive Energy-ACID reranking for LeWM CEM planning.

Energy-ACID keeps the original LeWM execution contract unchanged: a horizon-H plan
is optimized and ``receding_horizon == horizon`` executes the whole plan.  It only
changes how the *same* CEM candidates are ranked.

For an imagined transition ``x_t = (z_hat_t, z_hat_{t+1})`` and its conditioning
standardized action block ``a_t``, the C_T_dcl_idm checkpoint supplies

    s(x_t, a_t) = cosine(f(x_t), g(a_t)) / tau.

The recommended ``nce`` energy estimates the context-dependent normalizer with a
fixed empirical action bank B:

    e_t = -s(x_t, a_t) + logmeanexp_{b in B} s(x_t, b).

The CEM cost is

    terminal_cost + lambda * std(terminal)/std(mean_t e_t) * mean_t e_t,

where the standard deviations are over the candidates of each CEM iteration, as in
ACID's scale-invariant adaptive weighting.  ``lambda == 0`` returns the base cost
bit-for-bit and is the correctness control.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from slewm.planning.idm_decode import IDMDecoder


def contrastive_energy_from_keys(
    query: torch.Tensor,
    positive_key: torch.Tensor,
    *,
    bank_keys: torch.Tensor | None = None,
    tau: float = 0.1,
    mode: str = "nce",
) -> torch.Tensor:
    """Return one energy per query from unnormalized embeddings.

    ``raw`` is the bounded disagreement ``1-cos``.  ``nce`` adds a fixed-bank
    log-normalizer so scores from different imagined transitions can be summed.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    q = F.normalize(query.float(), dim=-1)
    k = F.normalize(positive_key.float(), dim=-1)
    cosine = (q * k).sum(dim=-1)
    if mode == "raw":
        return 1.0 - cosine
    if mode != "nce":
        raise ValueError(f"unknown energy mode {mode!r}; expected 'raw' or 'nce'")
    if bank_keys is None or bank_keys.ndim != 2:
        raise ValueError("nce mode requires bank_keys with shape (M, K)")
    bank = F.normalize(bank_keys.float(), dim=-1)
    bank_logits = (q @ bank.T) / tau
    log_z = torch.logsumexp(bank_logits, dim=-1) - math.log(bank.shape[0])
    return -(cosine / tau) + log_z


def adaptive_acid_cost(
    terminal_cost: torch.Tensor,
    energy_cost: torch.Tensor,
    *,
    lam: float,
    eps: float = 1e-8,
    scale_max: float | None = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two ``(B,S)`` costs with per-environment candidate-spread matching."""
    if terminal_cost.shape != energy_cost.shape or terminal_cost.ndim != 2:
        raise ValueError(
            f"expected matching (B,S) tensors, got {terminal_cost.shape} and {energy_cost.shape}"
        )
    if lam == 0:
        scale = torch.zeros(
            terminal_cost.shape[0], 1, device=terminal_cost.device, dtype=terminal_cost.dtype
        )
        return terminal_cost, scale
    sigma_g = terminal_cost.float().std(dim=1, keepdim=True, unbiased=False)
    sigma_e = energy_cost.float().std(dim=1, keepdim=True, unbiased=False)
    scale = sigma_g / (sigma_e + eps)
    if scale_max is not None:
        scale = scale.clamp(max=float(scale_max))
    scale = scale.to(dtype=terminal_cost.dtype)
    return terminal_cost + float(lam) * scale * energy_cost, scale


@dataclass
class _RunningDiagnostics:
    calls: int = 0
    scale_sum: float = 0.0
    cosine_sum: float = 0.0
    overlap_sum: float = 0.0

    def update(self, scale: torch.Tensor, cosine: torch.Tensor, overlap: float) -> None:
        self.calls += 1
        self.scale_sum += float(scale.mean())
        self.cosine_sum += float(cosine.mean())
        self.overlap_sum += float(overlap)

    def as_dict(self) -> dict[str, float | int]:
        denom = max(self.calls, 1)
        return {
            "cost_calls": self.calls,
            "mean_adaptive_scale": self.scale_sum / denom,
            "mean_positive_cosine": self.cosine_sum / denom,
            "mean_terminal_combined_elite_overlap": self.overlap_sum / denom,
        }


class LeWMEnergyACID(nn.Module):
    """Wrap a frozen LeWM with a contrastive-IDM energy cost for CEM."""

    def __init__(
        self,
        base: nn.Module,
        raw_state_dict: dict,
        *,
        lam: float = 0.0,
        mode: str = "nce",
        action_bank: torch.Tensor | None = None,
        tau: float = 0.1,
        scale_max: float | None = 100.0,
        device: str = "cuda",
        D: int = 192,
        A: int = 10,
    ) -> None:
        super().__init__()
        self.base = base
        self.lam = float(lam)
        self.mode = str(mode)
        self.tau = float(tau)
        self.scale_max = scale_max
        self.D, self.A = int(D), int(A)
        self.interpolate_pos_encoding = True

        if "state_dict" in raw_state_dict:
            raw_state_dict = raw_state_dict["state_dict"]
        self.idm = IDMDecoder(
            raw_state_dict,
            D=self.D,
            A=self.A,
            tau=self.tau,
            device=device,
            pool=action_bank,
        )
        if self.idm.mode != "contrastive":
            raise ValueError(
                f"Energy-ACID requires head_f_idm/head_g_idm, got IDM mode {self.idm.mode!r}"
            )
        if self.mode not in {"raw", "nce"}:
            raise ValueError(f"unknown mode {self.mode!r}")
        self._diag = _RunningDiagnostics()
        self._logged = False

    def criterion(self, info_dict: dict) -> torch.Tensor:
        return self.base.criterion(info_dict)

    @staticmethod
    def _elite_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
        """Average overlap of the best 10% candidates under two costs."""
        sample_count = a.shape[1]
        k = max(1, sample_count // 10)
        ia = torch.topk(a, k=k, dim=1, largest=False).indices
        ib = torch.topk(b, k=k, dim=1, largest=False).indices
        overlap = (ia.unsqueeze(-1) == ib.unsqueeze(-2)).any(dim=-1).float().mean()
        return float(overlap)

    @torch.inference_mode()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        terminal = self.base.get_cost(info_dict, action_candidates)
        if self.lam == 0.0:
            return terminal

        predicted = info_dict["predicted_emb"]
        actions = action_candidates.to(predicted.device)
        batch, samples, horizon, action_dim = actions.shape
        if action_dim != self.A:
            raise ValueError(f"checkpoint expects {self.A}-D action blocks, got {action_dim}")

        # The final H+1 latents are the current imagined state plus H predicted states.
        chain = predicted[:, :, -(horizon + 1) :, :]
        if chain.shape[2] != horizon + 1 or chain.shape[-1] != self.D:
            raise ValueError(
                f"unexpected predicted chain {chain.shape}; expected (*,*,{horizon + 1},{self.D})"
            )
        z_t = chain[:, :, :-1].reshape(-1, self.D)
        z_next = chain[:, :, 1:].reshape(-1, self.D)
        a_t = actions.reshape(-1, self.A).float()

        pair = torch.cat([z_t, z_next], dim=-1)
        query = self.idm.f(pair)
        positive_key = self.idm.g(a_t)
        per_step = contrastive_energy_from_keys(
            query,
            positive_key,
            bank_keys=self.idm.pool_g if self.mode == "nce" else None,
            tau=self.tau,
            mode=self.mode,
        )
        energy = per_step.reshape(batch, samples, horizon).mean(dim=-1)
        combined, scale = adaptive_acid_cost(
            terminal,
            energy,
            lam=self.lam,
            scale_max=self.scale_max,
        )

        cosine = (F.normalize(query.float(), dim=-1) * F.normalize(positive_key.float(), dim=-1)).sum(-1)
        overlap = self._elite_overlap(terminal, combined)
        self._diag.update(scale, cosine, overlap)
        if not self._logged:
            self._logged = True
            print(
                "[energy_acid] "
                f"lam={self.lam:g} mode={self.mode} tau={self.tau:g} "
                f"bank={self.idm.pool.shape[0]}({self.idm.pool_src}) H={horizon} A={action_dim} "
                f"term_std={float(terminal.float().std(dim=1, unbiased=False).mean()):.4g} "
                f"energy_std={float(energy.float().std(dim=1, unbiased=False).mean()):.4g} "
                f"scale={float(scale.mean()):.4g} overlap@10%={overlap:.3f}"
            )
        return combined

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "lambda": self.lam,
            "mode": self.mode,
            "tau": self.tau,
            "bank_size": int(self.idm.pool.shape[0]),
            "bank_source": self.idm.pool_src,
            **self._diag.as_dict(),
        }
