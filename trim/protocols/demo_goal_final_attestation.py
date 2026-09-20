"""Cross-task gates for the locked Demo-Goal successor study."""

from __future__ import annotations

from pathlib import Path

from trim.protocols.demo_goal_formal_artifacts import (
    file_sha256,
    verify_demo_goal_method_lock,
    verify_self_hashed_json,
    write_exclusive_self_hashed_json,
)
from trim.protocols.demo_goal_formal_specs import (
    DEMO_GOAL_FINAL_CONFIRMATION_ID,
    DEMO_GOAL_STUDY_IDS,
)


DISCOVERY_QUALIFICATION_ID = "demo-goal-successor-discovery-qualified-v1"
DISCOVERY_QUALIFICATION_SCHEMA_VERSION = 1
FINAL_ATTESTATION_SCHEMA_VERSION = 1
FINAL_CLAIM = (
    "The locked training-only Demo-Goal successor beat the paired champion "
    "on both PushT and TwoRoom under preregistered gates."
)


def _verify_report(
    path: Path,
    *,
    task: str,
    mode: str,
    expected_commit: str,
    expected_tree: str,
    expected_method_lock_sha256: str,
):
    from scripts.compare_formal_eval_results import verify_comparison_report

    path = Path(path).resolve()
    report = verify_comparison_report(
        path,
        expected_task=task,
        expected_mode=mode,
        expected_baseline_method="champion",
        expected_candidate_method="dg_champion",
        expected_commit=expected_commit,
        expected_tree=expected_tree,
    )
    if not report.get("gate_pass"):
        raise RuntimeError(f"{task} {mode} report did not pass")
    if report.get("study_id") != DEMO_GOAL_STUDY_IDS[task]:
        raise RuntimeError(f"{task} {mode} report has the wrong study ID")
    if report.get("method_lock_sha256") != expected_method_lock_sha256:
        raise RuntimeError(f"{task} {mode} report has the wrong method lock")
    return path, report


def _report_binding(path: Path, report):
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "report_sha256": report["report_sha256"],
    }


def write_discovery_qualification(
    *,
    output_path: Path,
    method_lock_path: Path,
    pusht_discovery_path: Path,
    tworoom_discovery_path: Path,
    expected_commit: str,
    expected_tree: str,
):
    lock_path = Path(method_lock_path).resolve()
    lock = verify_demo_goal_method_lock(
        lock_path,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
    )
    reports = {}
    for task, supplied in (
        ("pusht", pusht_discovery_path),
        ("tworoom", tworoom_discovery_path),
    ):
        path, report = _verify_report(
            supplied,
            task=task,
            mode="discovery",
            expected_commit=expected_commit,
            expected_tree=expected_tree,
            expected_method_lock_sha256=lock["method_lock_sha256"],
        )
        reports[task] = _report_binding(path, report)
    payload = {
        "schema_version": DISCOVERY_QUALIFICATION_SCHEMA_VERSION,
        "status": "QUALIFIED",
        "qualification_id": DISCOVERY_QUALIFICATION_ID,
        "candidate_method": "dg_champion",
        "source_git_commit": expected_commit,
        "source_tracked_tree_sha256": expected_tree,
        "method_lock_path": str(lock_path),
        "method_lock_file_sha256": file_sha256(lock_path),
        "method_lock_sha256": lock["method_lock_sha256"],
        "task_study_ids": dict(DEMO_GOAL_STUDY_IDS),
        "discovery_reports": reports,
        "both_task_discovery_gates_passed": True,
    }
    return write_exclusive_self_hashed_json(
        payload,
        output_path,
        hash_key="qualification_sha256",
    )


def verify_discovery_qualification(
    path: Path,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
    expected_method_lock_sha256: str | None = None,
):
    path = Path(path).resolve()
    value = verify_self_hashed_json(path, hash_key="qualification_sha256")
    expected_keys = {
        "schema_version",
        "status",
        "qualification_id",
        "candidate_method",
        "source_git_commit",
        "source_tracked_tree_sha256",
        "method_lock_path",
        "method_lock_file_sha256",
        "method_lock_sha256",
        "task_study_ids",
        "discovery_reports",
        "both_task_discovery_gates_passed",
        "qualification_sha256",
    }
    expected = {
        "schema_version": DISCOVERY_QUALIFICATION_SCHEMA_VERSION,
        "status": "QUALIFIED",
        "qualification_id": DISCOVERY_QUALIFICATION_ID,
        "candidate_method": "dg_champion",
        "task_study_ids": DEMO_GOAL_STUDY_IDS,
        "both_task_discovery_gates_passed": True,
    }
    if expected_commit is not None:
        expected["source_git_commit"] = expected_commit
    if expected_tree is not None:
        expected["source_tracked_tree_sha256"] = expected_tree
    if expected_method_lock_sha256 is not None:
        expected["method_lock_sha256"] = expected_method_lock_sha256
    mismatches = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if set(value) != expected_keys:
        mismatches["keys"] = (sorted(value), sorted(expected_keys))
    try:
        lock_path = Path(value["method_lock_path"])
        lock = verify_demo_goal_method_lock(
            lock_path,
            expected_commit=value["source_git_commit"],
            expected_tree=value["source_tracked_tree_sha256"],
        )
        if file_sha256(lock_path) != value["method_lock_file_sha256"]:
            mismatches["method_lock_file_sha256"] = ("recomputed", "bound")
        if lock["method_lock_sha256"] != value["method_lock_sha256"]:
            mismatches["method_lock_sha256"] = (
                value["method_lock_sha256"], lock["method_lock_sha256"]
            )
        reports = value["discovery_reports"]
        if set(reports) != {"pusht", "tworoom"}:
            raise RuntimeError("qualification must bind exactly two tasks")
        for task, binding in reports.items():
            report_path, report = _verify_report(
                binding["path"],
                task=task,
                mode="discovery",
                expected_commit=value["source_git_commit"],
                expected_tree=value["source_tracked_tree_sha256"],
                expected_method_lock_sha256=value["method_lock_sha256"],
            )
            wanted = _report_binding(report_path, report)
            if binding != wanted:
                mismatches[f"discovery_reports.{task}"] = (binding, wanted)
    except (KeyError, OSError, RuntimeError) as error:
        mismatches["external_bindings"] = (str(error), "valid locked inputs")
    if mismatches:
        raise RuntimeError(
            f"Demo-Goal discovery qualification mismatch: {mismatches}"
        )
    return value


def write_two_task_confirmation(
    *,
    output_path: Path,
    discovery_qualification_path: Path,
    pusht_confirmation_path: Path,
    tworoom_confirmation_path: Path,
    expected_commit: str,
    expected_tree: str,
):
    qualification_path = Path(discovery_qualification_path).resolve()
    qualification = verify_discovery_qualification(
        qualification_path,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
    )
    reports = {}
    for task, supplied in (
        ("pusht", pusht_confirmation_path),
        ("tworoom", tworoom_confirmation_path),
    ):
        report_path, report = _verify_report(
            supplied,
            task=task,
            mode="confirmation",
            expected_commit=expected_commit,
            expected_tree=expected_tree,
            expected_method_lock_sha256=qualification[
                "method_lock_sha256"
            ],
        )
        reports[task] = _report_binding(report_path, report)
    payload = {
        "schema_version": FINAL_ATTESTATION_SCHEMA_VERSION,
        "status": "PASS",
        "final_confirmation_id": DEMO_GOAL_FINAL_CONFIRMATION_ID,
        "candidate_method": "dg_champion",
        "source_git_commit": expected_commit,
        "source_tracked_tree_sha256": expected_tree,
        "method_lock_sha256": qualification["method_lock_sha256"],
        "discovery_qualification_path": str(qualification_path),
        "discovery_qualification_file_sha256": file_sha256(
            qualification_path
        ),
        "discovery_qualification_sha256": qualification[
            "qualification_sha256"
        ],
        "confirmation_reports": reports,
        "both_task_confirmation_gates_passed": True,
        "claim": FINAL_CLAIM,
    }
    return write_exclusive_self_hashed_json(
        payload, output_path, hash_key="final_attestation_sha256"
    )


def verify_two_task_confirmation(
    path: Path,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
):
    value = verify_self_hashed_json(
        path, hash_key="final_attestation_sha256"
    )
    expected_keys = {
        "schema_version",
        "status",
        "final_confirmation_id",
        "candidate_method",
        "source_git_commit",
        "source_tracked_tree_sha256",
        "method_lock_sha256",
        "discovery_qualification_path",
        "discovery_qualification_file_sha256",
        "discovery_qualification_sha256",
        "confirmation_reports",
        "both_task_confirmation_gates_passed",
        "claim",
        "final_attestation_sha256",
    }
    expected = {
        "schema_version": FINAL_ATTESTATION_SCHEMA_VERSION,
        "status": "PASS",
        "final_confirmation_id": DEMO_GOAL_FINAL_CONFIRMATION_ID,
        "candidate_method": "dg_champion",
        "both_task_confirmation_gates_passed": True,
        "claim": FINAL_CLAIM,
    }
    if expected_commit is not None:
        expected["source_git_commit"] = expected_commit
    if expected_tree is not None:
        expected["source_tracked_tree_sha256"] = expected_tree
    mismatches = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if set(value) != expected_keys:
        mismatches["keys"] = (sorted(value), sorted(expected_keys))
    try:
        qualification_path = Path(value["discovery_qualification_path"])
        qualification = verify_discovery_qualification(
            qualification_path,
            expected_commit=value["source_git_commit"],
            expected_tree=value["source_tracked_tree_sha256"],
            expected_method_lock_sha256=value["method_lock_sha256"],
        )
        qualification_binding = {
            "discovery_qualification_file_sha256": file_sha256(
                qualification_path
            ),
            "discovery_qualification_sha256": qualification[
                "qualification_sha256"
            ],
        }
        for key, wanted in qualification_binding.items():
            if value.get(key) != wanted:
                mismatches[key] = (value.get(key), wanted)
        reports = value["confirmation_reports"]
        if set(reports) != {"pusht", "tworoom"}:
            raise RuntimeError("final receipt must bind exactly two tasks")
        for task, binding in reports.items():
            report_path, report = _verify_report(
                binding["path"],
                task=task,
                mode="confirmation",
                expected_commit=value["source_git_commit"],
                expected_tree=value["source_tracked_tree_sha256"],
                expected_method_lock_sha256=value["method_lock_sha256"],
            )
            wanted = _report_binding(report_path, report)
            if binding != wanted:
                mismatches[f"confirmation_reports.{task}"] = (
                    binding, wanted
                )
    except (KeyError, OSError, RuntimeError) as error:
        mismatches["external_bindings"] = (str(error), "valid locked inputs")
    if mismatches:
        raise RuntimeError(
            f"Demo-Goal two-task confirmation mismatch: {mismatches}"
        )
    return value
