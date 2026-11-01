#!/usr/bin/env bash
# Small functional runs. PROFILE=smoke parameters are DEV, never final experiments.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
PYTHON="${PYTHON:-/home/mehrab/venv/virne-venv/bin/python}"
SOLVER="${SOLVER:-ppo_dual_gat+}"
SEED="${SEED:-17}"
EPOCHS="${EPOCHS:-1}"
OFFLINE="${OFFLINE:-true}"
PROFILE="${PROFILE:-baseline}"
OBJECTIVE="${OBJECTIVE:-multiplicative}"
REWARD="${REWARD:-joint_terminal}"
MODE="${MODE:-node_only}"
RUN_ID="${RUN_ID:-${SOLVER}__${PROFILE}__${OBJECTIVE}__${REWARD}__${MODE}__offline_${OFFLINE}__seed_${SEED}__epochs_${EPOCHS}__$(date -u +%Y%m%dT%H%M%S)_${RANDOM}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$repo_root/results/thesis_smoke}"
run_dir="$OUTPUT_ROOT/$SOLVER/$RUN_ID"
for argument in "$@"; do
    case "$argument" in
        experiment.run_id=*|experiment.save_root_dir=*|hydra.run.dir=*)
            printf 'Set RUN_ID/OUTPUT_ROOT via environment to preserve overwrite checks\n' >&2
            exit 1 ;;
    esac
done
if [[ -e "$run_dir" ]]; then
    printf 'Refusing to overwrite existing run: %s\n' "$run_dir" >&2
    exit 1
fi
mkdir -p "$run_dir"
args=(
    "solver.solver_name=$SOLVER" "training.num_train_epochs=$EPOCHS"
    "system.if_offline_system=$OFFLINE" "experiment.seed=$SEED"
    "experiment.num_simulations=1" "experiment.run_id=$RUN_ID"
    "experiment.save_root_dir=$OUTPUT_ROOT" "hydra.run.dir=$run_dir"
    training.use_cuda=false training.num_workers=1 training.if_use_random_training_seed=false
    training.save_interval=1 training.eval_interval=1 training.batch_size=4
    rl.target_steps=4 rl.repeat_times=1 nn.embedding_dim=16
    "v_sim_setting.num_v_nets=${REQUESTS:-4}" "p_net_setting.topology.num_nodes=${PN_NODES:-12}"
    "v_sim_setting.v_net_size.low=${VN_SIZE:-2}" "v_sim_setting.v_net_size.high=${VN_SIZE_HIGH:-3}"
    'logger.backends=[console,file]' logger.level=WARNING
)
if [[ "$PROFILE" == smoke ]]; then
    args+=(reliability.enabled=true "reliability.mode=$MODE" reliability.config_id=DEV_SMOKE_EXPLICIT
        reliability.nodes.lambda_f=0.00001 reliability.nodes.mu_r=0.1
        reliability.links.lambda_f=0.000005 reliability.links.mu_r=0.1
        sla.enabled=true 'sla.probabilities=[0.25,0.25,0.25,0.25]'
        "joint_objective.name=$OBJECTIVE" "joint_objective.alpha=${ALPHA:-0.5}"
        "joint_objective.beta=${BETA:-0.1}" "rl.reward_calculator.name=$REWARD"
        reliability_features.enabled=true
        'reliability_features.names=[raw_availability,sla_requirement,sla_margin,partial_availability,log_risk,remaining_budget]')
elif [[ "$PROFILE" != baseline && "$PROFILE" != configured ]]; then
    printf 'PROFILE must be baseline, smoke or configured\n' >&2
    exit 1
fi
if [[ -n "${MODEL_PATH:-}" ]]; then
    MODEL_PATH="$(realpath -- "$MODEL_PATH")"
    args+=("solver.pretrained_model_path=$MODEL_PATH")
elif [[ "$EPOCHS" == 0 && ( "$SOLVER" == ppo* || "$SOLVER" == cand_rl ) ]]; then
    printf 'Testing PPO requires MODEL_PATH to avoid evaluating an untrained model\n' >&2
    exit 1
fi
args+=("$@")
printf '%q ' "$PYTHON" main.py "${args[@]}" > "$run_dir/command.sh"
printf '\n' >> "$run_dir/command.sh"
git rev-parse HEAD > "$run_dir/source_commit.txt"
printf 'Run: %s\nProfile: %s (small functional budget)\n' "$run_dir" "$PROFILE"
"$PYTHON" main.py "${args[@]}" --cfg job --resolve > "$run_dir/hydra_config.yaml"
#start_seconds=$SECONDS



#==== added-start ======


# start_seconds=$SECONDS

# if [[ "${DEBUG_VIRNE:-0}" == "1" ]]; then
#     echo "Starting Virne under debugpy..."
#     "$PYTHON" -u -m debugpy \
#         --listen 127.0.0.1:5678 \
#         --wait-for-client \
#         main.py "${args[@]}" 2>&1 | tee "$run_dir/stdout.log"
# else
#     "$PYTHON" -u main.py "${args[@]}" 2>&1 | tee "$run_dir/stdout.log"
# fi

# printf '%s\n' "$((SECONDS-start_seconds))" > "$run_dir/runtime_seconds.txt"









start_seconds=$SECONDS


if [[ "${DEBUG_VIRNE:-0}" == "1" ]]; then
    echo "Starting Virne under isolated debugpy..."

    env \
        -u DEBUGPY_RUNNING \
        -u BUNDLED_DEBUGPY_PATH \
        -u VSCODE_DEBUGPY_ADAPTER_ENDPOINTS \
        -u PYDEVD_USE_FRAME_EVAL \
        -u PYDEVD_DISABLE_FILE_VALIDATION \
        "$PYTHON" -u -m debugpy \
            --listen 127.0.0.1:5678 \
            --wait-for-client \
            main.py "${args[@]}" \
            2>&1 | tee "$run_dir/stdout.log"
else
    "$PYTHON" -u main.py "${args[@]}" \
        2>&1 | tee "$run_dir/stdout.log"
fi

printf '%s\n' "$((SECONDS-start_seconds))" > "$run_dir/runtime_seconds.txt"





#==== added-end ======

#"$PYTHON" -u main.py "${args[@]}" 2>&1 | tee "$run_dir/stdout.log"
#printf '%s\n' "$((SECONDS-start_seconds))" > "$run_dir/runtime_seconds.txt"
printf 'Run directory: %s\nModel (if trained): %s/models/model.pkl\n' "$run_dir" "$run_dir"
