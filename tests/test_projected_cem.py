"""Pure CPU contracts for task-aware projected CEM."""

from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import h5py
import pytest
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from stable_worldmodel.solver.cem import CEMSolver

from slewm.planning.projected_cem import (
    ProjectedCEMSolver,
    ProjectedActionScaler,
    TrainingZScoreScaler,
    build_task_action_projection,
    resolve_raw_action_bounds,
    validate_action_scaler_contract,
    validate_model_action_contract,
)
from scripts.lewm_policy_server import (
    load_action_data,
    resolve_continuous_plan_tv_max,
    resolve_franka_terminal_support,
)


class _RecordingCost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.scored: list[torch.Tensor] = []

    def get_cost(self, info: dict, candidates: torch.Tensor) -> torch.Tensor:
        self.scored.append(candidates.detach().clone())
        target = info["target"][..., 0]
        return (candidates - target[..., None, None]).square().flatten(2).sum(-1)


class _NegativeNormCost(_RecordingCost):
    def get_cost(self, info: dict, candidates: torch.Tensor) -> torch.Tensor:
        self.scored.append(candidates.detach().clone())
        return -candidates.square().flatten(2).sum(-1)


class _TraceCallback:
    output_key = "projection_trace"

    def __init__(self) -> None:
        self.history: list[dict[str, torch.Tensor]] = []

    def reset(self) -> None:
        self.history = []

    def start_batch(self) -> None:
        pass

    def __call__(self, **kwargs) -> None:
        self.history.append(
            {
                key: kwargs[key].detach().clone()
                for key in (
                    "candidates",
                    "costs",
                    "topk_inds",
                    "topk_candidates",
                    "mean",
                )
            }
        )

    def end_solve(self) -> None:
        pass


@contextmanager
def _raises(exception_type, match: str):
    try:
        yield
    except exception_type as error:
        assert re.search(match, str(error)), (match, str(error))
    else:
        raise AssertionError(f"expected {exception_type.__name__}: {match}")


def _scaler(mean: list[float], scale: list[float]) -> TrainingZScoreScaler:
    return TrainingZScoreScaler(
        np.asarray(mean, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
        sample_count=100,
    )


def _inverse_flat(
    actions: torch.Tensor, scaler, action_dim: int
) -> np.ndarray:
    array = actions.detach().cpu().numpy().reshape(-1, action_dim)
    return scaler.inverse_transform(array).reshape(*actions.shape[:-1], -1, action_dim)


def _configure(solver, *, action_dim: int, action_block: int, horizon: int = 2) -> None:
    solver.configure(
        action_space=Box(-1.0, 1.0, shape=(1, action_dim), dtype=np.float32),
        n_envs=1,
        config=SimpleNamespace(horizon=horizon, action_block=action_block),
    )


def test_gear_projection_repeats_bounds_and_fixed_raw_dims_per_block() -> None:
    scaler = _scaler(
        [0.2, -0.1, 0.3, 0.7, -0.4, 0.5],
        [0.5, 0.25, 2.0, 0.4, 0.8, 1.5],
    )
    projection = build_task_action_projection(
        "gear_legacy6", action_dim=6, action_block=5, scaler=scaler,
        raw_low=[-0.6, -0.5, -0.4, -1.0, -1.0, -0.3],
        raw_high=[0.7, 0.8, 0.9, 1.0, 1.0, 0.4],
    )
    assert projection is not None
    projected = projection.project(torch.linspace(-100, 100, 2 * 3 * 30).reshape(2, 3, 30))
    raw = _inverse_flat(projected, scaler, 6)
    low = np.asarray([-0.6, -0.5, -0.4, -1.0, -1.0, -0.3])
    high = np.asarray([0.7, 0.8, 0.9, 1.0, 1.0, 0.4])
    assert np.all(raw >= low - 1e-6)
    assert np.all(raw <= high + 1e-6)
    assert np.allclose(raw[..., 3], 0.0, atol=1e-7)
    assert np.allclose(raw[..., 4], 0.0, atol=1e-7)
    output_scaler = ProjectedActionScaler(scaler, projection)
    raw_exact = _inverse_flat(projected, output_scaler, 6)
    assert np.all(raw_exact[..., 3] == 0.0)
    assert np.all(raw_exact[..., 4] == 0.0)


def test_franka_every_scored_elite_update_and_return_is_the_same_projection() -> None:
    scaler = _scaler(
        [0.1, -0.2, 0.3, 0.0, 0.4, -0.6, -0.025],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.998],
    )
    projection = build_task_action_projection(
        "franka_stack", action_dim=7, action_block=3, scaler=scaler,
        terminal_open_steps=6,
        terminal_switch_steps=(0,),
        terminal_switch_counts=((0, 100),),
        terminal_support_sample_count=100,
        continuous_plan_tv_max=1.25, continuous_plan_tv_quantile=0.99,
    )
    assert projection is not None
    callback = _TraceCallback()
    model = _RecordingCost()
    solver = ProjectedCEMSolver(
        model=model,
        batch_size=1,
        num_samples=17,
        var_scale=8.0,
        n_steps=4,
        topk=5,
        device="cpu",
        seed=7,
        callbacks=[callback],
        action_projection=projection,
    )
    _configure(solver, action_dim=7, action_block=3)
    output = solver.solve({"target": torch.zeros(1, 1)})

    # Four optimization scores plus one explicit projected-mean score.
    assert len(model.scored) == 5
    assert len(callback.history) == 4
    for scored, trace in zip(model.scored[:-1], callback.history):
        # The model and callback receive the exact same already-projected tensor.
        assert torch.equal(scored, trace["candidates"])
        rows = torch.arange(scored.shape[0]).unsqueeze(1)
        expected_elites = scored[rows, trace["topk_inds"]]
        assert torch.equal(trace["topk_candidates"], expected_elites)
        assert torch.equal(trace["mean"], solver._project(trace["mean"]))

        raw_scored = _inverse_flat(scored, scaler, 7)
        assert np.all(raw_scored >= -1.0 - 1e-6)
        assert np.all(raw_scored <= 1.0 + 1e-6)
        assert set(np.unique(np.round(raw_scored[..., 6], 6))) <= {-1.0, 1.0}
        flat_gripper = raw_scored[..., 6].reshape(*raw_scored.shape[:-3], -1)
        assert np.all(
            np.sum(np.diff(flat_gripper, axis=-1) != 0, axis=-1) <= 1
        )
        assert np.allclose(flat_gripper, 1.0, atol=1e-6)
        continuous = scored.reshape(*scored.shape[:-2], -1, 7)[..., :6]
        tv = continuous.diff(dim=-2).abs().sum(dim=(-2, -1))
        assert torch.all(tv <= 1.25 + 1e-5)

    returned = output["actions"]
    assert torch.equal(returned, solver._project(returned))
    scored_selection = model.scored[-1]
    expected_mean = callback.history[-1]["mean"]
    assert torch.equal(scored_selection[:, 0], expected_mean)
    mean_cost = scored_selection[0, 0].square().sum().item()
    best_sample_cost, best_sample_index = callback.history[-1]["costs"][0].min(dim=0)
    assert torch.equal(
        scored_selection[0, 1],
        callback.history[-1]["candidates"][0, best_sample_index],
    )
    assert np.isclose(output["projected_mean_costs"][0], mean_cost)
    assert np.isclose(output["best_sample_costs"][0], best_sample_cost.item())
    assert np.isclose(output["costs"][0], min(mean_cost, best_sample_cost.item()))
    if output["returned_plan_source"][0] == "projected_mean":
        assert torch.equal(returned[0], expected_mean[0].cpu())
    else:
        assert output["returned_plan_source"][0] == "best_feasible_sample"
        assert torch.equal(
            returned[0],
            callback.history[-1]["candidates"][0, best_sample_index].cpu(),
        )
    output_scaler = ProjectedActionScaler(scaler, projection)
    raw_returned = _inverse_flat(returned, output_scaler, 7)
    assert set(np.unique(raw_returned[..., 6])) <= {-1.0, 1.0}
    assert np.all(raw_returned >= -1.0)
    assert np.all(raw_returned <= 1.0)
    assert output["action_projection"]["name"] == "franka_stack"
    assert output["action_projection"]["single_switch_categorical"] == [6]
    assert output["action_projection"]["global_single_switch_categorical"] == [6]
    assert output["action_projection"]["terminal_open_steps"] == 6
    assert output["action_projection"]["terminal_switch_steps"] == [0]
    assert output["action_projection"]["terminal_switch_counts"] == {"0": 100}
    assert output["action_projection"]["continuous_action_indices"] == list(range(6))
    assert output["action_projection"]["continuous_plan_tv_max"] == 1.25
    assert output["action_projection"]["continuous_plan_tv_quantile"] == 0.99


def test_plan_tv_statistic_excludes_terminal_placeholders_and_episode_boundaries(tmp_path) -> None:
    path = tmp_path / "aligned.h5"
    with h5py.File(path, "w") as handle:
        # Each episode has four valid actions plus one terminal placeholder.
        # The huge cross-episode and placeholder jumps must never enter a
        # three-action plan's two-delta support statistic.
        handle.create_dataset(
            "action",
            data=np.asarray(
                [[0.0] * 7, [1.0] * 7, [2.0] * 7, [3.0] * 7, [999.0] * 7,
                 [100.0] * 7, [101.0] * 7, [102.0] * 7, [103.0] * 7, [-999.0] * 7],
                dtype=np.float32,
            ),
        )
        handle.create_dataset("ep_offset", data=np.asarray([0, 5], dtype=np.int64))
        handle.create_dataset("ep_len", data=np.asarray([5, 5], dtype=np.int64))
    scaler = _scaler([0.0] * 7, [1.0] * 7)
    # Six continuous dims, two unit deltas per three-action window.
    assert np.isclose(
        resolve_continuous_plan_tv_max(
            str(path), scaler, total_steps=3, quantile=0.99,
            continuous_indices=tuple(range(6)),
        ),
        12.0,
    )
    # The same episode-safe statistic is schema-selectable: effective-4D Gear
    # uses all four of its native [x, y, z, yaw] coordinates.
    assert np.isclose(
        resolve_continuous_plan_tv_max(
            str(path), scaler, total_steps=3, quantile=0.99,
            continuous_indices=tuple(range(4)),
        ),
        8.0,
    )


def test_franka_alternating_gripper_is_projected_to_training_support() -> None:
    scaler = _scaler([0.0] * 7, [1.0] * 7)
    projection = build_task_action_projection(
        "franka_stack", action_dim=7, action_block=5, scaler=scaler
    )
    action = torch.zeros(2, 35)
    action[:, 6::7] = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0])
    raw = _inverse_flat(projection.project(action), scaler, 7)
    switches = np.sum(np.diff(raw[..., 6], axis=-1) != 0, axis=-1)
    assert np.all(switches <= 1)
    assert set(np.unique(raw[..., 6])) <= {-1.0, 1.0}


def test_franka_formal_h5_uses_only_four_observed_terminal_release_phases() -> None:
    scaler = _scaler([0.0] * 7, [1.0] * 7)
    projection = build_task_action_projection(
        "franka_stack", action_dim=7, action_block=5, scaler=scaler,
        terminal_open_steps=5,
        terminal_switch_steps=(16, 18, 19, 20),
        terminal_switch_counts=((16, 1), (18, 2), (19, 795), (20, 2)),
        terminal_support_sample_count=800,
    )
    solver = ProjectedCEMSolver(
        model=_RecordingCost(),
        num_samples=4,
        n_steps=1,
        topk=2,
        device="cpu",
        action_projection=projection,
    )
    _configure(solver, action_dim=7, action_block=5, horizon=5)
    candidate = torch.randn(64, 5, 35)
    projected = solver._project(candidate)
    raw_gripper = projected.reshape(64, 25, 7)[..., 6].numpy()
    for sequence in raw_gripper:
        open_indices = np.flatnonzero(sequence == 1.0)
        assert int(open_indices[0]) in {16, 18, 19, 20}
        assert np.all(sequence[: open_indices[0]] == -1.0)
        assert np.all(sequence[open_indices[0] :] == 1.0)

    for requested_split in (16, 17, 18, 19, 20):
        candidate = torch.zeros(1, 5, 35)
        raw = candidate.reshape(1, 25, 7)
        raw[..., 6] = -1.0
        raw[:, requested_split:, 6] = 1.0
        actual = solver._project(candidate).reshape(25, 7)[:, 6].numpy()
        actual_split = int(np.flatnonzero(actual == 1.0)[0])
        assert actual_split in {16, 18, 19, 20}
        assert actual_split != 17
        if requested_split in {16, 18, 19, 20}:
            assert actual_split == requested_split


def test_franka_open_latch_applies_to_every_later_scored_plan_until_reset() -> None:
    scaler = _scaler([0.0] * 7, [1.0] * 7)
    projection = build_task_action_projection(
        "franka_stack", action_dim=7, action_block=5, scaler=scaler,
        terminal_open_steps=5,
        terminal_switch_steps=(16, 18, 19, 20),
        terminal_switch_counts=((16, 1), (18, 2), (19, 795), (20, 2)),
        terminal_support_sample_count=800,
    )
    model = _RecordingCost()
    solver = ProjectedCEMSolver(
        model=model,
        num_samples=7,
        n_steps=2,
        topk=3,
        device="cpu",
        action_projection=projection,
    )
    _configure(solver, action_dim=7, action_block=5, horizon=5)

    solver.observe_returned_raw_action(np.asarray([0] * 6 + [-1], dtype=np.float32))
    assert not solver._latched_categorical
    solver.observe_returned_raw_action(np.asarray([0] * 6 + [1], dtype=np.float32))
    output = solver.solve({"target": torch.zeros(1, 1)})
    for scored in model.scored:
        gripper = scored.reshape(*scored.shape[:-2], 25, 7)[..., 6]
        assert torch.all(gripper == 1.0)
    returned_gripper = output["actions"].reshape(25, 7)[:, 6]
    assert torch.all(returned_gripper == 1.0)

    solver.reset_action_latches()
    assert not solver._latched_categorical
    unlatch_projected = solver._project(torch.zeros(1, 5, 35))
    assert torch.any(unlatch_projected.reshape(25, 7)[:, 6] == -1.0)


def test_feedback_latch_reports_only_the_first_state_change() -> None:
    scaler = _scaler([0.0] * 7, [1.0] * 7)
    projection = build_task_action_projection(
        "franka_stack", action_dim=7, action_block=5, scaler=scaler,
        terminal_open_steps=5,
        terminal_switch_steps=(16, 18, 19, 20),
        terminal_switch_counts=((16, 1), (18, 2), (19, 795), (20, 2)),
        terminal_support_sample_count=800,
    )
    solver = ProjectedCEMSolver(
        model=_RecordingCost(), num_samples=7, n_steps=2, topk=3,
        device="cpu", action_projection=projection,
    )
    _configure(solver, action_dim=7, action_block=5, horizon=5)
    assert solver.latch_raw_categorical(6, 1.0) is True
    assert solver.latch_raw_categorical(6, 1.0) is False
    with pytest.raises(ValueError, match="outside categorical support"):
        solver.latch_raw_categorical(6, 0.0)


def test_terminal_support_is_fitted_as_a_non_contiguous_train_only_set(tmp_path) -> None:
    path = tmp_path / "franka_train.h5"
    actions = np.zeros((20, 7), dtype=np.float32)
    offsets = np.asarray([0, 10], dtype=np.int64)
    goals = np.asarray([8, 18], dtype=np.int64)
    for offset, split in zip(offsets, (1, 3)):
        actions[offset : offset + 4, 6] = 1.0  # sampled all-open training plan
        actions[offset + 4 : offset + 8, 6] = -1.0
        actions[offset + 4 + split : offset + 8, 6] = 1.0
        actions[offset + 8, 6] = 1.0  # post-success continuation
        actions[offset + 9, 6] = 0.0  # terminal placeholder
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=actions)
        handle.create_dataset("ep_offset", data=offsets)
        handle.create_dataset("ep_len", data=np.asarray([10, 10], dtype=np.int64))
        handle.create_dataset("goal_idx", data=goals)
        handle.create_dataset("ep_has_success", data=np.asarray([True, True]))

    support = resolve_franka_terminal_support(
        str(path), total_steps=4, clip_stride=5
    )
    assert support["switch_steps"] == (1, 3)
    assert support["switch_counts"] == ((1, 1), (3, 1))
    assert support["sample_count"] == 2
    assert support["terminal_open_steps"] == 1
    assert support["all_open_plan_count"] > 0


def test_cube_projection_enforces_standardized_per_dimension_bounds() -> None:
    scaler = _scaler([0.2, 0.0, -0.4, 0.3, 0.1], [0.1, 0.5, 0.2, 2.0, 0.25])
    projection = build_task_action_projection(
        "cube",
        action_dim=5,
        action_block=2,
        scaler=scaler,
        raw_low=[-0.2, -0.8, -0.6, -0.4, -1.0],
        raw_high=[0.4, 0.7, 0.2, 0.9, 1.0],
    )
    assert projection is not None
    projected = projection.project(torch.randn(8, 4, 10) * 1000)
    raw = _inverse_flat(projected, scaler, 5)
    assert np.all(raw >= np.asarray([-0.2, -0.8, -0.6, -0.4, -1.0]) - 1e-6)
    assert np.all(raw <= np.asarray([0.4, 0.7, 0.2, 0.9, 1.0]) + 1e-6)


def test_gear_effective4_is_native_xyzyaw_without_legacy_fixed_dims() -> None:
    scaler = _scaler([0.1, -0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5])
    projection = build_task_action_projection(
        "gear_effective4",
        action_dim=4,
        action_block=5,
        scaler=scaler,
        raw_low=[-0.4, -0.5, -0.6, -0.7],
        raw_high=[0.4, 0.5, 0.6, 0.7],
        continuous_plan_tv_max=2.0,
        continuous_plan_tv_quantile=0.99,
    )
    assert projection is not None
    assert projection.fixed == ()
    assert projection.raw_fixed == ()
    assert projection.continuous_action_indices == tuple(range(4))
    raw = _inverse_flat(projection.project(torch.randn(3, 2, 20) * 50), scaler, 4)
    assert np.all(raw >= np.asarray([-0.4, -0.5, -0.6, -0.7]) - 1e-6)
    assert np.all(raw <= np.asarray([0.4, 0.5, 0.6, 0.7]) + 1e-6)

    with _raises(ValueError, match="requires action_dim=4"):
        build_task_action_projection(
            "gear_effective4", action_dim=6, action_block=5, scaler=scaler
        )

    solver = ProjectedCEMSolver(
        model=_RecordingCost(),
        num_samples=4,
        n_steps=1,
        topk=2,
        device="cpu",
        action_projection=projection,
    )
    _configure(solver, action_dim=4, action_block=5)
    alternating = torch.empty(1, 4, 2, 20)
    alternating[..., 0::2, :] = -100.0
    alternating[..., 1::2, :] = 100.0
    constrained = solver._project(alternating)
    continuous = constrained.reshape(1, 4, -1, 4)
    tv = continuous.diff(dim=-2).abs().sum(dim=(-2, -1))
    assert torch.all(tv <= 2.0 + 1e-5)


def test_returned_plan_uses_recorded_best_feasible_sample_when_mean_is_worse() -> None:
    scaler = _scaler([0.0] * 5, [1.0] * 5)
    projection = build_task_action_projection(
        "cube", action_dim=5, action_block=1, scaler=scaler,
        raw_low=-10.0, raw_high=10.0,
    )
    model = _NegativeNormCost()
    solver = ProjectedCEMSolver(
        model=model,
        batch_size=1,
        num_samples=5,
        var_scale=1.0,
        n_steps=1,
        topk=5,
        device="cpu",
        seed=13,
        action_projection=projection,
    )
    _configure(solver, action_dim=5, action_block=1)
    output = solver.solve({"target": torch.zeros(1, 1)})

    assert len(model.scored) == 2
    first_costs = -model.scored[0].square().flatten(2).sum(-1)
    best_cost, best_index = first_costs[0].min(dim=0)
    mean_cost = -model.scored[1][0, 0].square().sum()
    assert best_cost < mean_cost
    assert output["returned_plan_source"] == ["best_feasible_sample"]
    assert torch.equal(output["actions"][0], model.scored[0][0, best_index])
    assert np.isclose(output["costs"][0], best_cost.item())


def test_generic_mode_is_literal_stock_cem_compatible() -> None:
    assert build_task_action_projection(
        "generic", action_dim=2, action_block=1, scaler=None
    ) is None
    kwargs = dict(
        batch_size=1,
        num_samples=9,
        var_scale=1.0,
        n_steps=3,
        topk=3,
        device="cpu",
        seed=123,
    )
    stock = CEMSolver(model=_RecordingCost(), **kwargs)
    projected = ProjectedCEMSolver(
        model=_RecordingCost(), action_projection=None, **kwargs
    )
    _configure(stock, action_dim=2, action_block=1)
    _configure(projected, action_dim=2, action_block=1)
    info = {"target": torch.tensor([[0.25]])}
    assert torch.equal(stock.solve(info)["actions"], projected.solve(info)["actions"])


def test_schema_and_checkpoint_contracts_fail_fast() -> None:
    scaler = _scaler([0.0] * 7, [1.0] * 7)
    with _raises(ValueError, match="requires action_dim=7"):
        build_task_action_projection(
            "franka_stack", action_dim=6, action_block=5, scaler=scaler
        )
    with _raises(ValueError, match="requires a fitted action scaler"):
        build_task_action_projection(
            "gear", action_dim=6, action_block=5, scaler=None
        )

    model = SimpleNamespace(
        action_encoder=SimpleNamespace(patch_embed=nn.Conv1d(35, 4, 1))
    )
    assert validate_model_action_contract(model, action_dim=7, action_block=5) == 35
    with _raises(ValueError, match="expects 35"):
        validate_model_action_contract(model, action_dim=6, action_block=5)
    with _raises(ValueError, match="must have 6 features"):
        validate_action_scaler_contract(scaler, action_dim=6)


def test_projection_configuration_must_match_plan_config() -> None:
    scaler = _scaler([0.0] * 6, [1.0] * 6)
    projection = build_task_action_projection(
        "gear_legacy6", action_dim=6, action_block=5, scaler=scaler
    )
    solver = ProjectedCEMSolver(
        model=_RecordingCost(),
        num_samples=4,
        n_steps=1,
        topk=2,
        device="cpu",
        action_projection=projection,
    )
    with _raises(ValueError, match="action_block"):
        _configure(solver, action_dim=6, action_block=4)


def test_training_scaler_matches_torch_sample_std_and_round_trips() -> None:
    data = np.asarray(
        [[1.0, 4.0, 8.0], [2.0, 4.0, 10.0], [4.0, 4.0, 14.0]],
        dtype=np.float32,
    )
    scaler = TrainingZScoreScaler.fit(data)
    tensor = torch.from_numpy(data)
    expected_mean = tensor.mean(dim=0).numpy()
    expected_scale = tensor.std(dim=0).numpy()
    expected_scale[expected_scale < 1e-6] = 1.0
    assert scaler.ddof == 1
    assert np.array_equal(scaler.mean_, expected_mean)
    assert np.array_equal(scaler.scale_, expected_scale)
    assert np.allclose(scaler.inverse_transform(scaler.transform(data)), data)
    assert scaler.to_dict()["ddof"] == 1


def test_data_support_bounds_are_per_dimension_and_explicit() -> None:
    data = np.asarray(
        [[-0.5, 0.0], [-0.2, 10.0], [0.7, 20.0], [0.9, 30.0]],
        dtype=np.float32,
    )
    low, high = resolve_raw_action_bounds(
        "data_minmax",
        action_dim=2,
        physical_low=-1.0,
        physical_high=1.0,
        action_data=data,
    )
    assert np.array_equal(low, np.asarray([-0.5, 0.0]))
    assert np.allclose(high, [0.9, 30.0])

    low, high = resolve_raw_action_bounds(
        "physical",
        action_dim=2,
        physical_low=[-0.25, -3.0],
        physical_high=[0.75, 4.0],
    )
    assert np.array_equal(low, np.asarray([-0.25, -3.0]))
    assert np.array_equal(high, np.asarray([0.75, 4.0]))


def test_policy_action_data_excludes_terminal_placeholders(tmp_path) -> None:
    path = tmp_path / "actions.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "action", data=np.arange(12, dtype=np.float32).reshape(6, 2)
        )
        handle.create_dataset("ep_offset", data=np.asarray([0, 3]))
        handle.create_dataset("ep_len", data=np.asarray([3, 3]))

    # Episode-final rows 2 and 5 have no stored successor observation.
    expected = np.asarray([[0, 1], [2, 3], [6, 7], [8, 9]], dtype=np.float32)
    assert np.array_equal(load_action_data(str(path)), expected)


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} projected CEM tests")
