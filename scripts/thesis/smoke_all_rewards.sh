#!/usr/bin/env bash
set -euo pipefail
export PROFILE=smoke
export EPOCHS="${EPOCHS:-1}"
# potential_based is deliberately blocked until Phi is defined.
for REWARD in joint_terminal joint_fixed joint_adaptive joint_gradual; do
    export REWARD
    "$(dirname -- "${BASH_SOURCE[0]}")/train_offline.sh" "$@"
done
