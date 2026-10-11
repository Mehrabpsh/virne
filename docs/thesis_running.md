# Running the thesis implementation

## One command for the complete sequential workflow

```bash
cd /home/mehrab/Workspaces/Codex/Thesis/virne
bash scripts/thesis/run_all.sh
```

This runs the tests, baseline training/testing twice, exact policy/record
reproducibility checks, node-only and node+link reliability runs, all four
objectives, all four executable rewards, eight heuristics, and Cand-RL
training/online reload. The 26 jobs run sequentially and stop on the first error.
Missing pytest/tomli dependencies are installed with `uv` into the results
directory, without changing the Python environment (requires package access).

The final printed directory is `results/thesis_suite/<timestamp>_<id>/`:

- `run_manifest.csv` and `run_manifest.json`: combined results for every job.
- `runs/<solver>/<step>/`: configurations, records, logs, exact commands, models.
- `logs/`: tests and per-step output; `status.tsv`: passed/failed steps.
- `commands.sh`: the expanded commands in execution order.
- `baseline_*_comparison.json`: repeated-run reproducibility reports.
- `COMPLETE.txt`: written only after all steps succeed.

Preview every command without running or creating outputs:

```bash
bash scripts/thesis/run_all.sh --dry-run
# Optional custom environment/output directory (must not already exist):
PYTHON=/path/to/python RESULTS_DIR="$PWD/results/my_suite" bash scripts/thesis/run_all.sh
```

This reproduces the successful **DEV smoke workflow**, not historical failed
attempts or final thesis experiments. Fresh repeated baselines check current-code
determinism; they do not recreate the historical pre-modification checkout.
Potential-based shaping and Paper-9 sanity remain blocked as documented below.

These scripts default to four small requests, one CPU epoch, a 12-node substrate,
and 16-dimensional neural embeddings. **They are functional DEV runs, not the
final scientific experiment profile.** No final alpha/beta, objective, reward,
feature subset, candidate K, component-rate distribution or heuristic combine
has been selected. Infrastructure defaults keep reliability disabled; scientific
fields in `virne/configs/thesis.yaml` are null until explicitly supplied.

## Environment and tests

The existing usable environment is `/home/mehrab/venv/virne-venv/bin/python`
(Python 3.10). The project declares Python >=3.12. Validation here uses the
existing runtime; it does not certify the pinned `pyproject.toml`/`uv.lock`
dependency environment. The regression runner provides a process-local
`tomllib=tomli` compatibility alias for existing tests on Python 3.10.
pytest and compatibility dependencies were installed only in `/tmp/thesis-test-deps`.

```bash
scripts/thesis/test.sh
# Alternatively use your own >=3.12 environment with project dependencies:
PYTHON=/path/to/python TEST_DEPS=/path/to/test/dependencies scripts/thesis/test.sh
```

No environment or Git-history changes are required by the scripts. Temporary
pytest dependencies must be reinstalled or provided if /tmp is cleared.

## Baseline train / test

```bash
RUN_ID=my_baseline scripts/thesis/train_offline.sh
MODEL_PATH=results/thesis_smoke/ppo_dual_gat+/my_baseline/models/model.pkl scripts/thesis/test_offline.sh
```

Use absolute model paths or repo-relative paths; scripts resolve them before
Hydra changes working directory. Lowercase Hydra booleans are used. Values with
lists are shell-quoted. Solver key is literally `ppo_dual_gat+` (no escape or
special Hydra quoting needed for the plus within this value).

## Reliability-aware train / test

```bash
PROFILE=smoke MODE=node_and_link RUN_ID=my_thesis_model scripts/thesis/train_offline.sh
PROFILE=smoke MODE=node_and_link MODEL_PATH=results/thesis_smoke/ppo_dual_gat+/my_thesis_model/models/model.pkl scripts/thesis/test_offline.sh
PROFILE=smoke MODE=node_and_link MODEL_PATH=results/thesis_smoke/ppo_dual_gat+/my_thesis_model/models/model.pkl scripts/thesis/test_online.sh
PROFILE=smoke MODE=node_only scripts/thesis/train_online.sh
scripts/thesis/smoke_all_objectives.sh
scripts/thesis/smoke_all_rewards.sh
```

`PROFILE=smoke` explicitly selects scalar CTMC DEV rates, uniform SLA classes,
alpha=.5, beta=.1, all six features, and the chosen joint objective/reward.
`PROFILE=configured` takes scientific overrides directly, without these defaults:

```bash
PROFILE=configured scripts/thesis/train_online.sh \
  reliability.enabled=true reliability.mode=node_only reliability.config_id=MY_EXPLICIT_PROFILE \
  reliability.nodes.lambda_f=0.00001 reliability.nodes.mu_r=0.1 \
  sla.enabled=true 'sla.probabilities=[0.25,0.25,0.25,0.25]' \
  joint_objective.name=multiplicative joint_objective.beta=0.1 \
  rl.reward_calculator.name=joint_terminal
```

The numbers above are also **DEV examples**, not recommended thesis parameters.
A final validated profile should be version-controlled separately after selection.
Rate arrays follow physical graph iteration order; uniform rate distributions
require explicit low/high. SLA `assignment=explicit` requires one `values` entry
per request. Independent RNGs avoid perturbing resources/topology/workloads.

Control runs through `PYTHON`, `SOLVER`, `SEED`, `EPOCHS`, `REQUESTS`, `PN_NODES`,
`VN_SIZE`, `VN_SIZE_HIGH`, `PROFILE`, `MODE`, `OBJECTIVE`, `REWARD`, `ALPHA`,
`BETA`, `MODEL_PATH`, `OUTPUT_ROOT`, and `RUN_ID`; remaining arguments are Hydra
overrides. Run IDs are unique by default. Existing directories are rejected to
avoid model overwrite. Set identity/output fields through environment variables.
Each run saves its exact shell command, Git commit, Hydra/resolved config,
stdout/stderr, runtime, records and model when training.

## Ablations and comparator profiles

Feature-only: use an original reward with reliability features enabled.
Objective-only: use a joint reward and `reliability_features.enabled=false`.
A feature subset is selected by `reliability_features.names=[...]`; its order
is part of the checkpoint signature. Incompatible checkpoints fail explicitly.
Original baseline weights cannot load into a policy with added feature channels.

All six frozen features are available: raw component availability, request SLA,
local SLA margin, partial delivered availability, negative log availability,
and remaining log reliability budget. Physical edges receive applicable
component features in node+link mode. No extra feature candidates were added.

See `thesis_comparator_provenance.md` for original heuristics, RA adaptations,
and executable `cand_rl.profile=thesis_joint` reconstruction commands.
Potential-based reward and `cand_rl.profile=paper9_sanity` are explicitly blocked
pending scientific definitions/inputs; selecting them gives a specific error.

## Outputs and deterministic comparison

```bash
PYTHONPATH="$PWD" /home/mehrab/venv/virne-venv/bin/python scripts/thesis/collect_results.py results/thesis_smoke
/home/mehrab/venv/virne-venv/bin/python scripts/thesis/compare_baseline.py BEFORE_RUN_DIR AFTER_RUN_DIR --output results/baseline_comparison.json
```

Per-request records include required/delivered availability, normalized shortfall,
compliance, SLA status, original economic R2C, joint utility and configuration
identity. Rejected services have null delivered availability/shortfall/utility;
accepted-service SLA metrics and all-arrival SLA service rate use distinct,
explicit denominators. SLA-class statistics and shortfall histograms/quantiles
are machine-readable JSON columns alongside original Virne metrics.

Online execution preserves residual resources across requests and processes
all departures. Offline execution is the original Virne functional-solvability
mode that resets/scales substrate resources after each request; use online
runs for the thesis setting. No realized failure events are generated.
