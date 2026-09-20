"""Crash-safe JSON result publication with a hash-bound completion marker."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def commit_json_result(
    path: str | Path,
    payload: dict[str, Any],
    *,
    schema_version: str,
    record_count: int,
) -> dict[str, Any]:
    """Publish result then ``.done``; consumers trust only the latter's hash."""

    path = Path(path)
    if record_count < 0:
        raise ValueError("record_count must be non-negative")
    document = dict(payload)
    existing = document.get("schema_version")
    if existing not in (None, schema_version):
        raise ValueError("result schema version conflicts with commit contract")
    document["schema_version"] = schema_version
    _atomic_json(path, document)
    completion = {
        "schema_version": "paper-eval-completion-v1",
        "status": "complete",
        "result": str(path.resolve()),
        "result_sha256": file_sha256(path),
        "result_schema_version": schema_version,
        "record_count": int(record_count),
    }
    done_path = path.with_suffix(path.suffix + ".done")
    _atomic_json(done_path, completion)
    return completion


__all__ = ["commit_json_result", "file_sha256"]
