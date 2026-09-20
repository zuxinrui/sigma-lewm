"""Best-of-K multi-restart CEM — targets the h10 CEM-optimization variance.

Long-horizon full-commit success varies across planner seeds. One source is CEM
OPTIMIZATION variance: a single 300x30 CEM lands in one basin of a sparsely-sampled
100-d (10 steps x 10-d) landscape. This solver runs K independent restarts (same env
state, different CEM RNG) and selects ONE plan.

selection:
  'cost'     : lowest JEPA terminal cost across restarts (pure optimization; risks
               objective-hacking = picking the most P_f-over-optimistic basin).
  'witness'  : lowest real-goal backward-witness energy (the CROSS-RESTART test — the
               elite-level version failed, but restarts are DIVERSE, not clustered).
  'consensus': plan closest (in action space) to the restart centroid (robustness /
               variance reduction without trusting either proxy).

Non-regression note: unlike the safe rerank, best-of-K has NO candidate-0 exemption;
K=1 reproduces stock CEM exactly (single restart, same seed).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from stable_worldmodel.solver.cem import CEMSolver


class MultiRestartCEMSolver(CEMSolver):
    def __init__(
        self,
        *args,
        n_restarts: int = 4,
        selection: str = "cost",
        verifier: Any = None,
        idm_verifier: Any = None,
        tau: float = 0.1,
        trace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        if selection not in ("cost", "witness", "consensus"):
            raise ValueError(f"unknown selection {selection!r}")
        if selection == "witness" and verifier is None:
            raise ValueError("selection='witness' needs a ForwardDCLVerifier")
        self.n_restarts = int(n_restarts)
        self.selection = str(selection)
        self.verifier = verifier
        self.idm_verifier = idm_verifier
        self.tau = float(tau)
        self.trace = bool(trace)
        self._base_seed = int(kwargs.get("seed", getattr(self, "seed", 1234)))
        self._diag = {"solves": 0, "restart_disagree_frac": 0.0}
        child_kwargs = {
            "model": self.model,
            "batch_size": self.batch_size,
            "num_samples": self.num_samples,
            "var_scale": self.var_scale,
            "n_steps": self.n_steps,
            "topk": self.topk,
            "device": self.device,
        }
        # Candidate zero remains the parent solver with its untouched stock RNG
        # stream.  Other restarts own persistent independent generators; they
        # are not re-seeded at every receding-horizon decision.
        self._restart_solvers = [
            CEMSolver(**child_kwargs, seed=self._base_seed + 7919 * k)
            for k in range(1, self.n_restarts)
        ]

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        super().configure(action_space=action_space, n_envs=n_envs, config=config)
        for solver in self._restart_solvers:
            solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    def _terminal_cost(self, info_dict: dict, plan: torch.Tensor) -> torch.Tensor:
        """(B,H,A) plan -> (B,) JEPA terminal cost; also leaves goal_emb/predicted_emb."""
        cand = plan.unsqueeze(1).to(self.device, self.dtype)          # (B,1,H,A)
        info = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                dt = self.dtype if v.is_floating_point() else None
                info[k] = v.to(self.device, dt)[:, None].expand(v.shape[0], 1, *v.shape[1:])
            elif isinstance(v, np.ndarray):
                info[k] = np.repeat(v[:, None, ...], 1, axis=1)
            else:
                info[k] = v
        cost = self.model.get_cost(info, cand)                        # (B,1)
        return cost[:, 0], info

    def _witness_energy(self, info: dict, plan: torch.Tensor) -> torch.Tensor:
        """Real-goal backward-witness energy of a single plan. (B,)  lower=better."""
        chain = info["predicted_emb"][:, 0, -(plan.shape[1] + 1):, :]  # (B,H+1,D)
        z_hat_Hm1 = chain[:, -2, :]                                    # (B,D)
        a_last = plan[:, -1, :].to(self.device, self.dtype)           # (B,A)
        D = self.verifier.D
        zg = info["goal_emb"].reshape(info["goal_emb"].shape[0], -1, D)[:, -1, :]  # (B,D) real goal
        return self.verifier.score(zg, z_hat_Hm1, a_last)             # (B,)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        t0 = time.time()
        # Literal stock solve first: same parent RNG state and implementation as
        # standalone CEM.  Extra restarts cannot perturb this stream.
        outs = [super().solve(info_dict, init_action)]
        outs.extend(
            solver.solve(info_dict, init_action) for solver in self._restart_solvers
        )
        plans = [out["actions"].to(self.device, self.dtype) for out in outs]
        stacked = torch.stack(plans, dim=1)                           # (B,K,H,A)
        B, K = stacked.shape[:2]

        # score each restart
        scores = torch.full((B, K), float("inf"), device=self.device)
        if self.selection == "consensus":
            centroid = stacked.mean(dim=1, keepdim=True)              # (B,1,H,A)
            scores = (stacked - centroid).pow(2).flatten(2).sum(-1)   # (B,K) distance to centroid
        else:
            # JEPA goal embeddings omit an explicit candidate axis.  Score one
            # environment at a time with all K restarts on the sample axis;
            # B>1 portfolio scoring can silently align env and sample axes.
            for env_index in range(B):
                info = {}
                for key, value in info_dict.items():
                    if torch.is_tensor(value):
                        batch = value[env_index : env_index + 1]
                        dtype = self.dtype if batch.is_floating_point() else None
                        info[key] = batch.to(self.device, dtype)[:, None].expand(
                            1, K, *batch.shape[1:]
                        )
                    elif isinstance(value, np.ndarray):
                        info[key] = np.repeat(
                            value[env_index : env_index + 1, None, ...], K, axis=1
                        )
                    else:
                        info[key] = value
                cost = self.model.get_cost(
                    info, stacked[env_index : env_index + 1]
                ).float()[0]
                if self.selection == "cost":
                    scores[env_index] = cost
                else:
                    chain = info["predicted_emb"][
                        0, :, -(stacked.shape[2] + 1) :, :
                    ]
                    z_hat_hm1 = chain[:, -2]
                    goal = info["goal_emb"].reshape(1, -1, chain.shape[-1])[
                        0, -1
                    ].expand(K, -1)
                    scores[env_index] = self.verifier.score(
                        goal, z_hat_hm1, stacked[env_index, :, -1]
                    )

        best = scores.argmin(dim=1)                                   # (B,)
        rows = torch.arange(B, device=self.device)
        selected = stacked[rows, best]                                # (B,H,A)

        # diagnostic: how often restarts disagree on the best plan vs restart 0
        self._diag["solves"] += 1
        self._diag["restart_disagree_frac"] += float((best != 0).float().mean())

        base = outs[0]
        base["actions"] = selected.detach().cpu()
        base["costs"] = scores[rows, best].detach().cpu().tolist()
        base["mean"] = [base["actions"]]
        candidate_vars = torch.stack([out["var"][0] for out in outs], dim=1).to(
            self.device, self.dtype
        )
        base["var"] = [candidate_vars[rows, best].detach().cpu()]
        base["multirestart"] = {
            "n_restarts": self.n_restarts, "selection": self.selection,
            **{k: (v / max(self._diag["solves"], 1) if k.endswith("frac") else v)
               for k, v in self._diag.items()},
        }
        print(f"MultiRestart-CEM {time.time()-t0:.2f}s "
              f"(K={self.n_restarts} sel={self.selection} chose!=0: "
              f"{float((best!=0).float().mean()):.2f})")
        return base
