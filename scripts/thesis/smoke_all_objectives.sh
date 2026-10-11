#!/usr/bin/env bash
set -euo pipefail
export PROFILE=smoke
export EPOCHS="${EPOCHS:-1}"
for OBJECTIVE in weighted_sum additive multiplicative geometric; do
    export OBJECTIVE
    "$(dirname -- "${BASH_SOURCE[0]}")/train_offline.sh" "$@"
done
