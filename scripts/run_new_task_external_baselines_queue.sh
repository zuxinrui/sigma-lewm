#!/usr/bin/env bash
# Compatibility entrypoint: the guarded deferred queue owns all new jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m scripts.new_task_deferred_queue --gpu "${1:-1}"
