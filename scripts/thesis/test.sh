#!/usr/bin/env bash
# Existing local Virne environment is Python 3.10; repository declares >=3.12.
# TEST_DEPS is a temporary pytest install; use a native >=3.12 env when available.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root:${TEST_DEPS:-/tmp/thesis-test-deps}${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-/home/mehrab/venv/virne-venv/bin/python}"
"$PYTHON" - "$@" <<'PY'
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli
    sys.modules['tomllib'] = tomli
import pytest
raise SystemExit(pytest.main(sys.argv[1:] or ['tests', '-q']))
PY
