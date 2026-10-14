#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
export PYTHON="${PYTHON:-/home/mehrab/venv/virne-venv/bin/python}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" scripts/thesis/experiments/pipeline.py "$@"
