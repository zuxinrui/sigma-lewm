from types import SimpleNamespace

import h5py
import numpy as np
import torch
from gymnasium import spaces

from trim.planning.behavior_plan_bank import (
    FrankaTerminalPlanBank,
    ProjectedPlanBankSolver,
    load_franka_terminal_plan_bank,
)
from trim.planning.projected_cem import StandardizedActionProjection


class _IdentityScaler:
    n_features_in_ = 7

    def transform(self, values):
        return np.asarray(values)


def test_terminal_bank_filters_non_strict_and_non_idempotent_plans(tmp_path):
    path = tmp_path / "train.h5"
    actions = np.zeros((15, 7), dtype=np.float32)
    actions[:, 6] = -1.0
    # Episode 0: accepted split=2 terminal plan.
    actions[2:4, 6] = 1.0
    # Episode 1: strict release, but continuous TV exceeds the support limit.
    actions[5:9, 0] = [0.0, 10.0, 0.0, 10.0]
    actions[7:9, 6] = 1.0
    # Episode 2: only one open command, so it is not strict-release eligible.
    actions[13, 6] = 1.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=actions)
        handle.create_dataset("pixels", data=np.zeros((15, 2, 2, 3), dtype=np.uint8))
        handle.create_dataset("ep_has_success", data=np.ones(3, dtype=bool))
        handle.create_dataset("ep_len", data=np.full(3, 5, dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.array([0, 5, 10]))
        handle.create_dataset("ep_source_demo_id", data=np.array([7, 3, 9]))
        handle.create_dataset("ep_split_id", data=np.zeros(3, dtype=np.int8))
        handle.create_dataset("goal_idx", data=np.array([4, 9, 14]))
        handle.attrs["selected_split"] = "train"
        handle.attrs["action_alignment"] = "obs[t]=s_t, action[t]=a_t"
        handle.attrs["split_seed"] = 3072
        handle.attrs["train_fraction"] = 0.8
        handle.attrs["val_fraction"] = 0.1
        handle.attrs["source"] = "/source/eval.h5"

    bank = load_franka_terminal_plan_bank(
        path,
        _IdentityScaler(),
        horizon=2,
        action_block=2,
        minimum_open_commands=2,
        terminal_switch_steps=(2,),
        raw_low=np.full(7, -20.0),
        raw_high=np.full(7, 20.0),
        continuous_plan_tv_max=5.0,
    )
    assert bank.size == 1
    assert bank.source_demo_ids.tolist() == [7]
    assert bank.raw_plans.shape == (1, 4, 7)
    assert bank.standardized_plans.shape == (1, 2, 14)
    np.testing.assert_array_equal(
        bank.standardized_plans.reshape(1, 4, 7), bank.raw_plans
    )
    assert bank.provenance["observed_strict_candidate_count"] == 2
    assert bank.provenance["excluded_by_raw_bounds"] == 0
    assert bank.provenance["excluded_by_continuous_plan_tv"] == 1
    assert len(bank.provenance["candidate_plan_sha256"]) == 1


class _EndpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cost_calls = 0

    def encode(self, info):
        return {"emb": info["pixels"].float()}

    def get_cost(self, info, candidates):
        self.cost_calls += 1
        # Prefer the candidate with first action closest to 20.
        return (candidates[:, :, 0, 0] - 20.0).square()


def _solver_bank():
    plans = np.zeros((3, 1, 7), dtype=np.float32)
    plans[:, 0, 0] = [0.0, 10.0, 20.0]
    plans[:, 0, 6] = 1.0
    return FrankaTerminalPlanBank(
        standardized_plans=plans.copy(),
        raw_plans=plans.copy(),
        source_demo_ids=np.array([100, 101, 102], dtype=np.int32),
        source_episode_indices=np.arange(3, dtype=np.int32),
        start_rows=np.arange(3),
        goal_rows=np.arange(3) + 1,
        switch_steps=np.zeros(3, dtype=np.int16),
        start_pixels=np.zeros((3, 1, 1, 1), dtype=np.uint8),
        goal_pixels=np.zeros((3, 1, 1, 1), dtype=np.uint8),
        central_index=0,
        open_hold_standardized=np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32),
        provenance={"bank_sha256": "0" * 64},
    )


def _projection():
    return StandardizedActionProjection(
        name="franka_stack",
        action_dim=7,
        action_block=1,
        low=(-100.0,) * 6 + (-1.0,),
        high=(100.0,) * 6 + (1.0,),
        raw_low=(-100.0,) * 6 + (-1.0,),
        raw_high=(100.0,) * 6 + (1.0,),
        categorical=((6, (-1.0, 1.0)),),
        raw_categorical=((6, (-1.0, 1.0)),),
        single_switch_categorical=(6,),
        terminal_open_steps=1,
        terminal_switch_steps=(0,),
        terminal_switch_counts=((0, 3),),
        terminal_support_sample_count=3,
    )


def _configured_solver(selection):
    model = _EndpointModel()
    solver = ProjectedPlanBankSolver(
        model=model,
        plan_bank=_solver_bank(),
        selection_mode=selection,
        continuation="abstain",
        device="cpu",
        action_projection=_projection(),
        start_embeddings=torch.tensor([[0.0], [10.0], [20.0]]),
        goal_embeddings=torch.tensor([[0.0], [10.0], [20.0]]),
        retrieval_topk=2,
    )
    action_space = spaces.Box(
        low=np.full((1, 7), -100.0, dtype=np.float32),
        high=np.full((1, 7), 100.0, dtype=np.float32),
    )
    solver.configure(
        action_space=action_space,
        n_envs=1,
        config=SimpleNamespace(horizon=1, action_block=1),
    )
    return model, solver


def test_behavior_retrieval_is_goal_conditioned_and_does_not_roll_out_model():
    model, solver = _configured_solver("behavior_retrieval")
    info = {
        "pixels": torch.tensor([[[10.0]]]),
        "goal": torch.tensor([[[10.0]]]),
        "action": torch.zeros(1, 1, 7),
    }
    result = solver.solve(info)
    assert result["plan_bank_diagnostics"]["selected_plan_indices"] == [1]
    assert model.cost_calls == 0
    second = solver.solve(info)
    assert second["plan_bank_diagnostics"]["abstain"] is True
    assert model.cost_calls == 0


def test_wm_rerank_scores_only_the_shared_retrieval_shortlist():
    model, solver = _configured_solver("wm_rerank")
    info = {
        "pixels": torch.tensor([[[15.0]]]),
        "goal": torch.tensor([[[15.0]]]),
        "action": torch.zeros(1, 1, 7),
    }
    result = solver.solve(info)
    diagnostics = result["plan_bank_diagnostics"]
    # The shared retrieval shortlist is plans 1 and 2; WM must pick plan 2.
    selected = diagnostics["selected_plan_indices"][0]
    assert selected == 2
    assert diagnostics["model_scored_candidate_count"] == 2
    assert model.cost_calls == 1
    assert np.isclose(result["actions"][0, 0, 0].item(), selected * 10.0)
