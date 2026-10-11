#!/usr/bin/env bash
# Reproduce the successful DEV workflow, sequentially, in one fresh directory.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
dry_run=false
case "${1:-}" in
    --dry-run) dry_run=true; shift ;;
    --help|-h)
        printf 'Usage: bash scripts/thesis/run_all.sh [--dry-run]\n'
        printf 'Optional environment: PYTHON, RESULTS_DIR, SEED, TEST_DEPS\n'
        printf 'Runs 26 small DEV jobs, tests, reproducibility checks, and collection.\n'
        exit 0 ;;
esac
[[ $# == 0 ]] || { printf 'Unexpected argument: %s\n' "$1" >&2; exit 2; }
export PYTHON="${PYTHON:-/home/mehrab/venv/virne-venv/bin/python}"
RESULTS_DIR="$(realpath -m -- "${RESULTS_DIR:-$repo_root/results/thesis_suite/$(date -u +%Y%m%dT%H%M%S)_${RANDOM}}")"
export TEST_DEPS="${TEST_DEPS:-$RESULTS_DIR/test_dependencies}"
TEST_DEPS="$(realpath -m -- "$TEST_DEPS")"
export PYTHONPATH="$repo_root:$TEST_DEPS${PYTHONPATH:+:$PYTHONPATH}"
export SEED="${SEED:-17}"
# Freeze the previously used tiny budget independently of exported launcher options.
export REQUESTS=4 PN_NODES=12 VN_SIZE=2 VN_SIZE_HIGH=3 ALPHA=0.5 BETA=0.1
export OUTPUT_ROOT="$RESULTS_DIR/runs"
if ! "$dry_run"; then
    [[ ! -e "$RESULTS_DIR" ]] || { printf 'Refusing to overwrite: %s\n' "$RESULTS_DIR" >&2; exit 1; }
    "$PYTHON" -c 'import torch, hydra, pandas, torch_geometric, omegaconf'
    mkdir -p "$RESULTS_DIR/logs"
    printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$repo_root" > "$RESULTS_DIR/commands.sh"
    for name in PYTHON TEST_DEPS PYTHONPATH SEED REQUESTS PN_NODES VN_SIZE VN_SIZE_HIGH ALPHA BETA OUTPUT_ROOT; do
        printf 'export %s=%q\n' "$name" "${!name}" >> "$RESULTS_DIR/commands.sh"
    done
    git rev-parse HEAD > "$RESULTS_DIR/source_commit.txt"
    git diff --binary > "$RESULTS_DIR/tracked_changes.patch"
    git status --short > "$RESULTS_DIR/source_status.txt"
    printf 'step\tstatus\n' > "$RESULTS_DIR/status.tsv"
    trap 'code=$?; if ((code)); then printf "Workflow failed (exit %s). Logs: %s/logs\n" "$code" "$RESULTS_DIR" >&2; fi' EXIT
fi
step_number=0
step() {
    local label="$1"
    shift
    step_number=$((step_number + 1))
    printf '[%02d] %s\n' "$step_number" "$label"
    if "$dry_run"; then
        printf '  '; printf '%q ' "$@"; printf '\n'
        return
    fi
    printf '%q ' "$@" >> "$RESULTS_DIR/commands.sh"
    printf '\n' >> "$RESULTS_DIR/commands.sh"
    if "$@" > "$RESULTS_DIR/logs/$label.log" 2>&1; then
        printf '%s\tpassed\n' "$label" >> "$RESULTS_DIR/status.tsv"
    else
        local code=$?
        printf '%s\tfailed\n' "$label" >> "$RESULTS_DIR/status.tsv"
        tail -n 60 "$RESULTS_DIR/logs/$label.log" >&2
        return "$code"
    fi
}
job() {
    local label="$1" solver="$2" profile="$3" mode="$4" objective="$5" reward="$6" epochs="$7" offline="$8" model="$9"
    shift 9
    step "$label" env "RUN_ID=$label" "SOLVER=$solver" "PROFILE=$profile" \
        "MODE=$mode" "OBJECTIVE=$objective" "REWARD=$reward" "EPOCHS=$epochs" \
        "OFFLINE=$offline" "MODEL_PATH=$model" bash scripts/thesis/run.sh "$@"
}
printf 'Results directory: %s\nDEV smoke settings: seed=%s, 4 requests, CPU, one training epoch.\n' "$RESULTS_DIR" "$SEED"
# Install test-only dependencies into the output directory if unavailable.
# No packages are installed into or removed from the Python environment.
if "$dry_run" || ! "$PYTHON" -c 'import pytest; import sys; import importlib; importlib.import_module("tomllib" if sys.version_info >= (3,11) else "tomli")' 2>/dev/null; then
    step prepare_test_dependencies uv pip install --python "$PYTHON" --target "$TEST_DEPS" pytest tomli
fi
step tests bash scripts/thesis/test.sh

# Original baseline: repeat training and testing to check current-code determinism.
job baseline_train_a 'ppo_dual_gat+' baseline node_only multiplicative joint_terminal 1 true '' nn.num_gnn_layers=1
job baseline_train_b 'ppo_dual_gat+' baseline node_only multiplicative joint_terminal 1 true '' nn.num_gnn_layers=1
baseline_model="$OUTPUT_ROOT/ppo_dual_gat+/baseline_train_a/models/model.pkl"
job baseline_test_a 'ppo_dual_gat+' baseline node_only multiplicative joint_terminal 0 true "$baseline_model" nn.num_gnn_layers=1
job baseline_test_b 'ppo_dual_gat+' baseline node_only multiplicative joint_terminal 0 true "$baseline_model" nn.num_gnn_layers=1 reliability.enabled=false
step compare_baseline_records "$PYTHON" scripts/thesis/compare_baseline.py \
    "$OUTPUT_ROOT/ppo_dual_gat+/baseline_test_a" "$OUTPUT_ROOT/ppo_dual_gat+/baseline_test_b" --output "$RESULTS_DIR/baseline_records_comparison.json"
step compare_baseline_models "$PYTHON" scripts/thesis/compare_models.py \
    "$baseline_model" "$OUTPUT_ROOT/ppo_dual_gat+/baseline_train_b/models/model.pkl" --output "$RESULTS_DIR/baseline_models_comparison.json"

# Node+link training and reload in both systems; node-only online training.
job reliability_train 'ppo_dual_gat+' smoke node_and_link multiplicative joint_terminal 1 true ''
reliability_model="$OUTPUT_ROOT/ppo_dual_gat+/reliability_train/models/model.pkl"
job reliability_test_offline 'ppo_dual_gat+' smoke node_and_link multiplicative joint_terminal 0 true "$reliability_model"
job reliability_test_online 'ppo_dual_gat+' smoke node_and_link multiplicative joint_terminal 0 false "$reliability_model"
job reliability_train_online 'ppo_dual_gat+' smoke node_only multiplicative joint_terminal 1 false ''
for objective in weighted_sum additive multiplicative geometric; do
    job "objective_$objective" 'ppo_dual_gat+' smoke node_only "$objective" joint_terminal 1 true ''
done
for reward in joint_terminal joint_fixed joint_adaptive joint_gradual; do
    job "reward_$reward" 'ppo_dual_gat+' smoke node_only multiplicative "$reward" 1 true ''
done

# Original heuristics and explicitly parameterized reliability adaptations.
for solver in ffd_rank nrm_rank grc_rank rw_rank; do
    job "$solver" "$solver" smoke node_only multiplicative joint_terminal 0 false '' reliability_features.enabled=false
done
job ra_nrm ra_nrm smoke node_only multiplicative joint_terminal 0 false '' ra_heuristic.combine=product ra_heuristic.normalization=max
job ra_grc ra_grc smoke node_only multiplicative joint_terminal 0 false '' ra_heuristic.combine=weighted_sum ra_heuristic.normalization=max ra_heuristic.alpha=0.5
evne=(ra_evne.cpu_attribute=cpu ra_evne.bandwidth_attribute=bw ra_evne.empty_distance=1)
job ra_evne_a ra_evne_a smoke node_only multiplicative joint_terminal 0 false '' "${evne[@]}"
zeros='[0,0,0,0,0,0,0,0,0,0,0,0]'
job ra_evne_original ra_evne_original smoke node_only multiplicative joint_terminal 0 false '' "${evne[@]}" \
    ra_evne.weibull_scale=250000 ra_evne.weibull_shape=1.5 \
    "ra_evne.ages=$zeros" "ra_evne.failure_counts=$zeros" "ra_evne.last_failure_positions=$zeros"

# Executable thesis Cand-RL reconstruction; Paper-9 sanity remains blocked.
cand=(cand_rl.k=3 cand_rl.distance_weight=hops cand_rl.source=0 cand_rl.destination=1
    'cand_rl.resource_names=[cpu]' 'cand_rl.capacity_attributes=[max_cpu]'
    'cand_rl.hidden_dims=[16,16]' cand_rl.mask_padding=true cand_rl.redundancy=none
    reliability_features.enabled=false)
job cand_train cand_rl smoke node_only multiplicative joint_terminal 1 true '' "${cand[@]}"
job cand_test_online cand_rl smoke node_only multiplicative joint_terminal 0 false \
    "$OUTPUT_ROOT/cand_rl/cand_train/models/model.pkl" "${cand[@]}"

step collect_results "$PYTHON" scripts/thesis/collect_results.py "$OUTPUT_ROOT" --output-prefix "$RESULTS_DIR/run_manifest"
step verify_manifest "$PYTHON" -c 'import json, sys; rows=json.load(open(sys.argv[1])); assert len(rows) == 26, f"Expected 26 runs, found {len(rows)}"; assert all(row["completed"] for row in rows), "Incomplete runs"; print("Verified 26 completed runs")' "$RESULTS_DIR/run_manifest.json"
if ! "$dry_run"; then
    printf 'Completed all steps.\nThese are DEV functional results, not final scientific experiments.\n' > "$RESULTS_DIR/COMPLETE.txt"
fi
printf '\nFinal results directory: %s\nCombined results: %s/run_manifest.csv and run_manifest.json\n' "$RESULTS_DIR" "$RESULTS_DIR"
