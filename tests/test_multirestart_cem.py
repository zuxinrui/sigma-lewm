"""Regression tests for compute-matched independent CEM restarts."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from trim.planning.multirestart_cem import MultiRestartCEMSolver
from stable_worldmodel.solver.cem import CEMSolver


class _QuadraticCost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.batch_sizes: list[int] = []

    def get_cost(self, info: dict, candidates: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(candidates.shape[0])
        target = info["goal"][..., 0]
        terminal = candidates[:, :, -1, 0]
        return (terminal - target).square()


def _configure(solver, n_envs: int = 2) -> None:
    solver.configure(
        action_space=Box(-1.0, 1.0, shape=(n_envs, 1), dtype=np.float32),
        n_envs=n_envs,
        config=SimpleNamespace(horizon=2, action_block=1),
    )


def test_one_restart_is_literal_stock_rng_stream() -> None:
    kwargs = dict(
        batch_size=2,
        num_samples=8,
        var_scale=1.0,
        n_steps=3,
        topk=3,
        device="cpu",
        seed=123,
    )
    stock = CEMSolver(model=_QuadraticCost(), **kwargs)
    multi = MultiRestartCEMSolver(
        model=_QuadraticCost(), n_restarts=1, selection="cost", **kwargs
    )
    _configure(stock)
    _configure(multi)
    info = {"goal": torch.tensor([[0.25], [-0.5]])}
    for _ in range(2):
        stock_out = stock.solve(info)
        multi_out = multi.solve(info)
        assert torch.equal(stock_out["actions"], multi_out["actions"])


def test_restart_portfolio_final_scoring_is_singleton_per_environment() -> None:
    model = _QuadraticCost()
    solver = MultiRestartCEMSolver(
        model=model,
        batch_size=7,
        num_samples=4,
        var_scale=1.0,
        n_steps=1,
        topk=2,
        device="cpu",
        seed=9,
        n_restarts=3,
        selection="cost",
    )
    _configure(solver, n_envs=3)
    info = {"goal": torch.tensor([[0.0], [4.0], [-3.0]])}
    out = solver.solve(info)
    assert out["actions"].shape == (3, 2, 1)
    # Three optimizer calls precede exactly three B=1 portfolio score calls.
    assert model.batch_sizes[-3:] == [1, 1, 1]


if __name__ == "__main__":
    test_one_restart_is_literal_stock_rng_stream()
    test_restart_portfolio_final_scoring_is_singleton_per_environment()
    print("test_multirestart_cem: PASS")
