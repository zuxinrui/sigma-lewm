import json

from scripts.result_commit import commit_json_result, file_sha256


def test_commit_publishes_hash_bound_done_marker(tmp_path):
    result = tmp_path / "eval.json"
    completion = commit_json_result(
        result,
        {"episodes": [{"success": True}]},
        schema_version="synthetic-eval-v1",
        record_count=1,
    )
    assert completion["result_sha256"] == file_sha256(result)
    assert json.loads(result.read_text())["schema_version"] == "synthetic-eval-v1"
    done = json.loads((tmp_path / "eval.json.done").read_text())
    assert done == completion
    assert done["record_count"] == 1
