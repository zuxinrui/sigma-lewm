from __future__ import annotations

import json

import pytest

from scripts.result_commit import commit_json_result
from scripts.run_paper_h5_checkpoint import (
    DEFAULT_POOL,
    _load_pool,
    _select_cases,
    _server_command,
    _verify_committed,
)


def test_frozen_pool_loads_exact_dev_and_test_cases():
    for task in ("gear", "franka"):
        payload, dev = _load_pool(DEFAULT_POOL, task=task, split="dev")
        _, test = _load_pool(DEFAULT_POOL, task=task, split="test")
        assert payload["schema_version"] == "paper-new-tasks-h5-pools-v2"
        assert len(dev) == len(test) == 30
        assert {item["case_id"] for item in dev}.isdisjoint(
            item["case_id"] for item in test
        )


def test_gear_pool_freezes_exhaustive_current_launch_restoration_audits():
    payload, _ = _load_pool(DEFAULT_POOL, task="gear", split="dev")
    sources = payload["gear"]["sources"]
    assert sum(int(source["eligible_count"]) for source in sources) == 64
    for source in sources:
        assert source["restoration_limits"] == {
            "all_cases_passed": True,
            "max_init_frame_mean_abs_error": 12.0,
            "max_init_frame_pixels_over_8": 0.4,
            "scope": "all_expert_eligible_cases_revalidated_without_control_steps",
        }
        assert source["restoration_revalidation"]["sha256"]
        assert source["restoration_revalidation_marker"]["sha256"]


def test_development_subset_is_explicit_and_never_allowed_on_test():
    cases = [{"case_id": str(index)} for index in range(30)]
    assert _select_cases(
        cases, split="dev", development_case_limit=None
    ) is cases
    assert _select_cases(
        cases, split="dev", development_case_limit=1
    ) == cases[:1]
    with pytest.raises(ValueError, match="sealed test"):
        _select_cases(cases, split="test", development_case_limit=1)
    with pytest.raises(ValueError, match="between 1 and 29"):
        _select_cases(cases, split="dev", development_case_limit=30)


def test_server_contracts_are_task_specific(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.touch()
    gear = _server_command("gear", checkpoint, tmp_path / "gear.sock", 42)
    franka = _server_command(
        "franka", checkpoint, tmp_path / "franka.sock", 42
    )
    assert gear[gear.index("--planner-mode") + 1] == "projected_cem"
    assert gear[gear.index("--task-action-schema") + 1] == "gear_effective4"
    assert franka[franka.index("--planner-mode") + 1] == (
        "train_terminal_wm_rerank"
    )
    assert franka[franka.index("--plan-bank-continuation") + 1] == "abstain"


def test_committed_result_verifier_is_hash_and_count_bound(tmp_path):
    path = tmp_path / "eval.json"
    commit_json_result(
        path,
        {"records": [{"success": True}]},
        schema_version="unit-eval-v1",
        record_count=1,
    )
    assert _verify_committed(path, schema="unit-eval-v1", count=1)[
        "records"
    ][0]["success"]
    marker = path.with_suffix(".json.done")
    payload = json.loads(marker.read_text())
    payload["record_count"] = 2
    marker.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="marker is invalid"):
        _verify_committed(path, schema="unit-eval-v1", count=1)
