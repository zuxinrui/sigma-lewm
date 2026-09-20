"""T7a proposal-only control: projected CEM whose world-model costs are random.

This module is deliberately *additive*: ``projected_cem.py`` and
``scripts/lewm_policy_server.py`` are hash-locked by the frozen paper protocol
(``outputs/paper_new_tasks_v1/manifest/experiment_lock.json``) and are left
byte-for-byte untouched.  ``RandomCostProjectedCEMSolver`` reuses
``ProjectedCEMSolver.solve`` unchanged -- sampling, train-support projection,
elite selection, distribution updates, latches and the final mean-vs-best
selection all run exactly as in ``projected_cem`` -- except that every call to
``model.get_cost`` is answered by i.i.d. ``torch.rand`` costs drawn from the
solver's planner-seeded generator (``self.torch_gen``, which the policy server
re-seeds per episode from the client's planner seed).
"""

from __future__ import annotations

from typing import Any

import torch

from slewm.planning.projected_cem import ProjectedCEMSolver


class _RandomCostModel:
    """Proxy that answers ``get_cost`` with uniform noise and delegates the rest."""

    def __init__(self, model: Any, solver: "RandomCostProjectedCEMSolver") -> None:
        object.__setattr__(self, "_wrapped_model", model)
        object.__setattr__(self, "_solver", solver)

    def get_cost(self, info_dict: dict, candidates: torch.Tensor) -> torch.Tensor:
        del info_dict  # the world model is never queried in this control
        solver = object.__getattribute__(self, "_solver")
        solver.random_cost_calls += 1
        return torch.rand(
            candidates.shape[0],
            candidates.shape[1],
            generator=solver.torch_gen,
            device=candidates.device,
            dtype=candidates.dtype,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped_model"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_wrapped_model"), name, value)


class RandomCostProjectedCEMSolver(ProjectedCEMSolver):
    """``ProjectedCEMSolver`` with the model cost replaced by planner-seeded noise."""

    PLAN_PROPOSAL_PROVENANCE = {
        "mode": "projected_random",
        "selection": "random_cost",
        "synthetic_cem_candidates": True,
        "world_model_reranks_candidates": False,
        "claim": (
            "T7a proposal-only control: projected CEM sampling, train-support "
            "projection, elite updates and latches identical to projected_cem; "
            "every world-model cost replaced by i.i.d. uniform noise drawn from "
            "the planner-seeded generator"
        ),
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.action_projection is None:
            raise ValueError("projected_random requires the projected CEM path")
        self.random_cost_calls = 0
        self.wrapped_model = self.model
        self.model = _RandomCostModel(self.model, self)
