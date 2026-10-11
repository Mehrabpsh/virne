#!/usr/bin/env bash
set -euo pipefail
export OFFLINE=true
export EPOCHS=${EPOCHS:-1}
exec "$(dirname -- "${BASH_SOURCE[0]}")/run.sh" "$@"
