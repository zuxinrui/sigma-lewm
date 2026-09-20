"""T7a proposal-only control: projected CEM with random costs (additive module)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trim.planning.projected_cem import ProjectedCEMSolver, build_task_action_projection  # noqa: E402
from trim.planning.t7a_projected_random import RandomCostProjectedCEMSolver  # noqa: E402
from tests.test_projected_cem import _RecordingCost, _configure, _scaler  # noqa: E402


def _gear_projection():
    scaler = _scaler([0.2, -0.1, 0.3, 0.7], [0.5, 0.25, 2.0, 0.4])
    return build_task_action_projection(
        "gear_effective4", action_dim=4, action_block=5, scaler=scaler,
        raw_low=[-0.6, -0.5, -0.4, -1.0], raw_high=[0.7, 0.8, 0.9, 1.0],
    )


def _solve(cls, seed: int):
    model = _RecordingCost()
    solver = cls(
        model=model, batch_size=1, num_samples=16, var_scale=1.0, n_steps=3,
        topk=4, device="cpu", seed=seed, action_projection=_gear_projection(),
    )
    _configure(solver, action_dim=4, action_block=5, horizon=5)
    return solver, model, solver.solve({"target": torch.zeros(1, 1)})


def test_random_cost_never_queries_the_world_model_and_stays_projected() -> None:
    solver, model, output = _solve(RandomCostProjectedCEMSolver, seed=11)
    assert model.scored == []  # the real model is never asked for a cost
    assert solver.random_cost_calls == 3 + 1  # n_steps iterations + final selection
    plan = output["actions"] if isinstance(output, dict) and "actions" in output else output
    plan = plan if torch.is_tensor(plan) else torch.as_tensor(np.asarray(plan))
    assert torch.isfinite(plan).all()
    # Returned plan is still a fixed point of the train-support projection.
    assert torch.allclose(solver._project(plan), plan, atol=1e-6)
    assert solver.model.anchor is model.anchor  # attribute delegation to the real model


def test_random_cost_is_planner_seed_deterministic() -> None:
    _, _, first = _solve(RandomCostProjectedCEMSolver, seed=5)
    _, _, second = _solve(RandomCostProjectedCEMSolver, seed=5)
    _, _, other = _solve(RandomCostProjectedCEMSolver, seed=6)
    a = first["actions"] if isinstance(first, dict) else first
    b = second["actions"] if isinstance(second, dict) else second
    c = other["actions"] if isinstance(other, dict) else other
    assert torch.equal(torch.as_tensor(a), torch.as_tensor(b))
    assert not torch.equal(torch.as_tensor(a), torch.as_tensor(c))


def test_frozen_projected_cem_unaffected() -> None:
    solver, model, _ = _solve(ProjectedCEMSolver, seed=11)
    assert len(model.scored) == 3 + 1
    assert not hasattr(solver, "random_cost_calls")
    assert not isinstance(solver.model, type(RandomCostProjectedCEMSolver.__mro__[0]))
