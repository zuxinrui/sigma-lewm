"""GBW-Collocation (seam-only, goal-anchored backward-witness rerank).

RATIONALE:
  The two contrastive verifiers are co-trained with P_f and tend to agree with it
  on P_f-IMAGINED transitions. So re-scoring P_f's own imagined rollout
  carries little independent signal — the verifier and P_f were co-trained and
  concur. The ONE place a verifier gives INDEPENDENT, distribution-shift-safe
  signal is where its query is built from the REAL encoded goal, not an imagined
  latent. GBW exploits exactly that: instead of trusting the degraded terminal
  latent z_hat_H (P_f rolled 10 steps past its 3-token training window), it asks
  the inverse-DCL backward witness:

      "is the (less-degraded) penultimate imagined state z_hat_{H-1} a compatible
       PREDECESSOR of the REAL goal z_goal under the last planned action a_{H-1}?"

  Seam energy (lower = better), REAL-goal-anchored:
      E_inv(c) = ForwardDCLVerifier.score(z = z_goal, z_next = z_hat_{H-1}, a = a_{H-1})
  which is exactly the probe's s_inv(pred=z_hat_{H-1}, succ=z_goal, a) as an NCE
  energy (bank = 512 real dataset latents). Only the KEY z_hat_{H-1} is imagined;
  the QUERY side carries the real goal. gamma=0: the fully-imagined backward chain
  term E_bwd is DROPPED (both args imagined = the Energy-ACID failure mode).

SAFETY: reuse select_verified_plan VERBATIM — candidate 0 is the literal CEM mean
(= full-commit baseline). A different elite is executed only if its UNCHANGED JEPA
terminal cost <= base AND its standardized seam-energy gain >= margin_sigma. So the
method is structurally non-regressing on the terminal objective and reduces to the
baseline bit-for-bit at verifier_margin_sigma=inf (ablation 1).

ABLATION 2 (the load-bearing mechanism test): anchor='zhatH' replaces z_goal by the
imagined z_hat_H in the seam. If selection is identical to anchor='goal', inverse-DCL
adds no backward information beyond P_f's own forward terminal -> mechanism falsified.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from trim.planning.energy_acid import contrastive_energy_from_keys
from trim.planning.solver_cover_cem import (
    _FinalEliteCapture,
    _Diagnostics,
    select_verified_plan,
)
from stable_worldmodel.solver.cem import CEMSolver


def _zscore_over_candidates(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardize each env's energies across the candidate (S) axis. (B,S)->(B,S).

    Required because the inverse-DCL (latent bank) and DCL-IDM (action bank) energies
    live on different scales; standardizing within the elite set before combining is
    the gauge-safe way to mix them (context is fixed per env, ranks within it)."""
    mean = x.float().mean(dim=1, keepdim=True)
    std = x.float().std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    return (x.float() - mean) / std


class SafeGBWCEMSolver(CEMSolver):
    """Stock CEM generator + goal-anchored backward-witness seam rerank of final elites.

    verifier      : ForwardDCLVerifier (inverse-DCL head_f/head_g), called in backward mode.
    idm_verifier  : optional IDMDecoder (contrastive) for the secondary seam-IDM term (beta>0).
    anchor        : 'goal' (real goal, default) | 'zhatH' (imagined terminal, ablation 2).
    beta          : weight of the standardized DCL-IDM seam term (default 0 = pure inverse seam).
    """

    def __init__(
        self,
        *args,
        verifier: Any,
        idm_verifier: Any = None,
        anchor: str = "goal",
        beta: float = 0.0,
        tau: float = 0.1,
        cost_margin: float = 0.0,
        verifier_margin_sigma: float = float("inf"),
        trace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if getattr(verifier, "mode", None) != "forward_dcl":
            raise ValueError(
                f"GBW backward witness must be a ForwardDCLVerifier, got mode={getattr(verifier,'mode',None)!r}"
            )
        if anchor not in ("goal", "zhatH"):
            raise ValueError(f"anchor must be 'goal' or 'zhatH', got {anchor!r}")
        self.verifier = verifier
        self.idm_verifier = idm_verifier
        self.anchor = str(anchor)
        self.beta = float(beta)
        self.tau = float(tau)
        self.cost_margin = float(cost_margin)
        self.verifier_margin_sigma = float(verifier_margin_sigma)
        self.trace = bool(trace)
        self.last_trace: list[dict] = []
        self._diag = _Diagnostics()
        self._D = int(verifier.D)

    def _expand_infos(self, info_dict: dict, start: int, end: int, samples: int) -> dict:
        expanded: dict[str, Any] = {}
        for key, value in info_dict.items():
            batch = value[start:end]
            if torch.is_tensor(value):
                dtype = self.dtype if batch.is_floating_point() else None
                batch = batch.to(device=self.device, dtype=dtype)
                expanded[key] = batch.unsqueeze(1).expand(batch.shape[0], samples, *batch.shape[1:])
            elif isinstance(value, np.ndarray):
                expanded[key] = np.repeat(batch[:, None, ...], samples, axis=1)
            else:
                expanded[key] = value[start:end]
        return expanded

    def _real_goal_latent(self, expanded: dict, B: int, S: int) -> torch.Tensor:
        """(B,S,D) real encoded goal, from goal_emb that get_cost just populated.

        Defensive reducer: the goal is identical across the S (candidate) axis, and
        goal_emb may be (B,T,D) or (B,S,T,D) depending on convention; flatten to
        (B, -1, D) and take the last token -> (B,D), then broadcast to (B,S,D)."""
        zg = expanded["goal_emb"]
        zg = zg.reshape(zg.shape[0], -1, self._D)[:, -1, :]   # (B, D)
        return zg.unsqueeze(1).expand(B, S, self._D).contiguous()

    def _seam_energy(self, expanded: dict, predicted_chain: torch.Tensor, plans: torch.Tensor) -> torch.Tensor:
        B, S, H, A = plans.shape
        chain = predicted_chain[:, :, -(H + 1):, :]           # (B,S,H+1,D)
        if chain.shape[2] != H + 1:
            raise ValueError(f"expected {H+1} rollout latents, got {chain.shape}")
        z_hat_Hm1 = chain[:, :, -2, :]                        # (B,S,D) penultimate imagined (KEY)
        z_hat_H = chain[:, :, -1, :]                          # (B,S,D) terminal imagined (ablation)
        a_last = plans[:, :, -1, :].float()                  # (B,S,A) a_{H-1}
        z_succ = self._real_goal_latent(expanded, B, S) if self.anchor == "goal" else z_hat_H

        # inverse-DCL backward witness: is z_hat_{H-1} the predecessor of z_succ under a_last?
        # ForwardDCLVerifier.score(z=successor, z_next=predecessor, a) == probe s_inv(pred,succ,a).
        flatpred = z_hat_Hm1.reshape(-1, self._D)
        flatsucc = z_succ.reshape(-1, self._D)
        flata = a_last.reshape(-1, A)
        E_inv = self.verifier.score(flatsucc, flatpred, flata).reshape(B, S)  # lower=better
        energy = _zscore_over_candidates(E_inv)

        if self.beta > 0.0 and self.idm_verifier is not None:
            # DCL-IDM secondary: does a_last explain the seam step z_hat_{H-1} -> z_succ?
            q = self.idm_verifier.f(torch.cat([flatpred, flatsucc], dim=-1))
            pos = self.idm_verifier.g(flata)
            E_idm = contrastive_energy_from_keys(
                q, pos, bank_keys=self.idm_verifier.pool_g, tau=self.tau, mode="nce"
            ).reshape(B, S)
            energy = energy + self.beta * _zscore_over_candidates(E_idm)
        return energy

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        capture = _FinalEliteCapture()
        self.callbacks.append(capture)
        try:
            outputs = super().solve(info_dict, init_action)
        finally:
            self.callbacks.remove(capture)
        outputs.get("callbacks", {}).pop(capture.output_key, None)
        t0 = time.time()

        baseline = outputs["actions"].to(self.device, self.dtype)
        selected = baseline.clone()
        total = baseline.shape[0]
        expected = (total + self.batch_size - 1) // self.batch_size
        if len(capture.batches) != expected:
            raise RuntimeError(f"captured {len(capture.batches)} CEM batches, expected {expected}")

        trace: list[dict] = []
        offset = 0
        for elites in capture.batches:
            bs = elites.shape[0]
            plans = torch.cat([baseline[offset:offset + bs, None], elites], dim=1)  # (bs, 1+topk, H, A); [:,0]=CEM mean
            expanded = self._expand_infos(info_dict, offset, offset + bs, plans.shape[1])
            terminal = self.model.get_cost(expanded, plans)          # unchanged JEPA terminal cost; sets goal_emb+predicted_emb
            if "predicted_emb" not in expanded:
                raise RuntimeError("world model get_cost did not expose predicted_emb")
            if "goal_emb" not in expanded:
                raise RuntimeError("world model get_cost did not expose goal_emb")
            energy = self._seam_energy(expanded, expanded["predicted_emb"], plans)
            chosen, accept, indices, gain = select_verified_plan(
                plans, terminal, energy,
                cost_margin=self.cost_margin,
                verifier_margin_sigma=self.verifier_margin_sigma,
            )
            selected[offset:offset + bs] = chosen
            rows = torch.arange(bs, device=self.device)
            chosen_cost = terminal[rows, indices]
            for r in range(bs):
                trace.append({
                    "selected_index": int(indices[r]), "accepted": bool(accept[r]),
                    "gain_sigma": float(gain[r]), "base_cost": float(terminal[r, 0]),
                    "selected_cost": float(chosen_cost[r]),
                })
            self._diag.considered += bs
            self._diag.accepted += int(accept.sum())
            self._diag.gain_sum += float(gain.sum())
            self._diag.base_cost_sum += float(terminal[:, 0].sum())
            self._diag.selected_cost_sum += float(chosen_cost.sum())
            offset += bs

        self._diag.solves += 1
        self.last_trace = trace
        outputs["actions"] = selected.detach().cpu()
        outputs["mean"] = [outputs["actions"]]
        outputs["gbw"] = self.diagnostics()
        print(
            f"GBW-seam verify {time.time()-t0:.3f}s "
            f"(anchor={self.anchor} beta={self.beta:g} accepted={self._diag.accepted}/{self._diag.considered} "
            f"margin_sigma={self.verifier_margin_sigma:g} cost_margin={self.cost_margin:g})"
        )
        if self.trace:
            print(f"[GBW trace] {trace}")
        return outputs

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "anchor": self.anchor, "beta": self.beta, "tau": self.tau,
            "cost_margin": self.cost_margin, "verifier_margin_sigma": self.verifier_margin_sigma,
            "candidate_count": self.topk + 1, **self._diag.as_dict(),
        }
