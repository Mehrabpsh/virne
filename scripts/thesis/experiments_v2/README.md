# Thesis Experiment Framework v2

This is experiment engineering around the frozen Virne implementation. No solver,
reliability/SLA, reward or objective formulas are changed. All supplied numeric values
are **preliminary pipeline-validation settings**, not frozen scientific choices.

## Two independent tools

Use the Python environment that contains Virne's existing dependencies plus matplotlib.
On the verified laptop this is `/home/mehrab/venv/virne-venv/bin/python`; on a server use
the appropriate activated environment. Commands below run from the repository root.

```bash
# Execute concrete YAML: no calibration is performed.
python scripts/thesis/experiments_v2/run_experiment.py \
  --config scripts/thesis/experiments_v2/examples/local_dev.yaml

# Resolve, validate registry/capabilities and print the entire plan, with no jobs/output bundle.
python scripts/thesis/experiments_v2/run_experiment.py \
  --config scripts/thesis/experiments_v2/examples/local_dev.yaml --dry-run

# Optional explicit calibration: produces a NEW YAML and separate calibration bundle.
python scripts/thesis/experiments_v2/calibrate.py \
  --config scripts/thesis/experiments_v2/examples/local_dev.yaml \
  --output results/main.calibrated.yaml

# Run that calibrated environment without searching again.
python scripts/thesis/experiments_v2/run_experiment.py --config results/main.calibrated.yaml

# Larger, editable candidate budgets; review all values before scientific execution.
python scripts/thesis/experiments_v2/run_experiment.py \
  --config scripts/thesis/experiments_v2/examples/gpu_server.yaml
```

`--output NEW_DIRECTORY` selects an execution bundle. Existing directories/files are
refused. Failures preserve FAILED.json, status.tsv, logs and the partial manifest;
COMPLETE.txt is written only after required stages and artifacts succeed. SIGTERM/SIGINT
are recorded when Python can receive the signal; SIGKILL/power loss cannot be journaled.

## Schema, overrides and protection

The sections are `experiment`, `runtime`, `environment`, `calibration`, `protocol`,
`learning_defaults`, `methods`, `metrics`, `reporting`. Calibrated output additionally
contains `calibration_result` and `calibration_suggestions`. `calibration` is a recipe
for the explicit calibration tool, never an automatic execution stage.

```bash
python scripts/thesis/experiments_v2/run_experiment.py \
  --config scripts/thesis/experiments_v2/examples/local_dev.yaml \
  --set protocol.training.seeds='[31,37]' \
  --set environment.workload.num_requests=32 \
  --set methods.reliability_ppo.objective.beta=0.4 \
  --virne-set rl.gamma=0.995 \
  --methods original_ppo,reliability_ppo,nrm --dry-run
```

Values are YAML values (numbers, booleans, lists, mappings or null). Overrides target
existing paths; misspelled/unknown paths fail preflight. YAML anchor sharing is removed
before mutation, so a method-specific override never changes another anchored method.
The original file is immutable. `--summary-output NEW_FILE` optionally saves a dry-run
summary; without that flag dry-run creates no experiment artifacts.

Precedence is Virne defaults → experiment YAML/`--set` → structured method settings and
method `overrides` → global `--virne-set` → runner-owned fields. Native mapping overrides
merge known subkeys; lists replace lists. Raw keys must exist in the current composed
Virne schema; `+` additions and unchecked custom schema extensions are deliberately
unsupported. This is the advanced Virne escape hatch, not an alternate framework schema.

Protected exogenous fields are `p_net_setting.*`, `v_sim_setting.*`, `reliability.*`,
`sla.*`. Method native overrides cannot change them. Global raw overrides can change
common environment values consistently; their effects also precede derived settings
such as EVNE history-array sizing. Final resolved environment contracts must agree.
The broad implementation switches `reliability.enabled` and `sla.enabled` are controlled
through method awareness, not raw global overrides. Runner-owned experiment identity,
seeds, output paths, lifecycle mode, checkpoint paths and deterministic worker controls
are applied last and cannot be overridden.

Changing a calibrated path through `--set` or changing the translated environment with
raw overrides is permitted and prominently recorded as `CALIBRATED PARAMETER OVERRIDDEN`,
with old/new values. This run no longer uses the calibrated environment unchanged.

Environment resources use actual names (`cpu`, `bw`) and the four owner groups:
`physical_node`, `physical_link`, `virtual_node`, `virtual_link`. Scalar rates are native;
`{distribution: fixed, value: ...}` is translated to a scalar, and uniform low/high
rates are translated to Virne's actual provider schema. There is no invented `.scale`
rate path: search/override concrete scalar rates or existing uniform bounds explicitly.
The initial schema exposes uniform resources, existing default topology keys, and the
existing workload settings. Unsupported distribution/extra-key semantics fail preflight.
PPO learning rates map to `rl.learning_rate.actor/critic`, not `training.learning_rate`.

## Methods and capabilities

A method ID is a treatment; `solver` is its registry implementation. Arbitrary method
names and multiple treatments of one solver are supported. Enabled registry solvers
are inspected dynamically; the runner contains no fixed thesis method list.

`kind: auto` infers the current RLSolver lifecycle. `kind: learned` requires a concrete
solve API, RLSolver training/learn_singly and save/load methods. `kind: heuristic` must
not conceal an untrained RLSolver. Static capability checks establish API compatibility,
not a guarantee about every solver constructor or successful scientific experiment.
Unknown/incomplete native configs and known thesis-specific requirements fail preflight;
other solver-specific runtime failures remain fatal with logs. Native `overrides` expose
existing solver settings without adding new launcher branches.

MCTS illustrates why registry labels are insufficient: it is labelled `r_learning` but
inherits Solver. It is evaluation-only search, not a checkpointed learned treatment.
Requesting MCTS as learned fails preflight. Other registry classes are inspected on
selection; none are declared universally executable merely because they are registered.

Original PPO uses `awareness: {reliability: false, sla: false}`, original reward and no
reliability features. Its common-world SLA assessment is derived post hoc from recorded
mappings. This preserves the verified baseline execution switch and checkpoint signature.
Aware methods must explicitly provide the objective and reward required by the frozen
recorder. Heuristics may ignore reliability and still receive the same post-hoc assessment.
`reporting.assessment_beta` is an explicit API parameter for the post-hoc calculator;
only A/D/satisfaction are retained, so this value never changes baseline rewards.

Cand-RL retains thesis_joint, explicit endpoints/candidates/resources/MLP hidden dimensions,
mask padding and redundancy=none. RA combine/normalization/alpha and EVNE initialization
are explicit. `history: {initialization: zero}` creates deterministic arrays matching the
resolved PN size. No Paper-9 exact reward, backup, potential reward or solver redesign is added.

## Calibration contract

Calibration searches **only** `calibration.search_space` environment paths and uses only
configured evaluation-only methods and calibration seeds. Seeds are disjoint from both
training and evaluation seeds. Every grid point is validated before launch. CPU is forced.

The declared loss reuses v1's transparent score: normalized distance from acceptance and
SLA-satisfaction intervals, plus resource-utilization shortfall, plus increases in SLA
satisfaction between ascending classes (minus the explicit tolerance, when requested).
Qualifying points additionally require all accepted SLA classes when configured. Prefer
qualifying points, then smallest loss, then grid ID. If none qualifies, report PROVISIONAL
and select the least-degenerate point under that same rule. No learned outcomes enter.

`calibration.requests` is a calibration measurement budget, not an implicitly searched
parameter or an overwrite of the experiment request budget. Only searched paths are
changed in the output environment. Reports retain candidate values, metrics, criteria,
seeds, raw runs, input fingerprints and full provenance.

An optional beta suggestion uses `-log(target_compliance)/median_positive_D`. By default
it is stored ONLY in `calibration_suggestions`; methods' beta values are untouched.
No positive D yields an unavailable suggestion with warning. Only a nonempty, explicit
`calibration.adopt_beta_for: [method_name,...]` adopts the suggested scale, recording old
and new values. Adoption remains provisional, not scientific freezing.

## Protocol, fairness and statistics

All learned methods use `protocol.training.seeds`. All methods use the same
`protocol.evaluation.seeds`; no separate validation-seed stage is introduced. Evaluation
uses OnlineSystem, zero training epochs, expected checkpoint reload and no optimizer
telemetry. Learned job counts are T training + T×E evaluations; heuristics have E runs.
A second training seed can be enabled without changing any YAML file using `--set`.

The existing seeded generator, Counter, analytic evaluator and full input fingerprint
machinery are reused. Per-seed snapshots include PN topology/capacities/reliability and
VNR topology/demands/times/lifetimes/SLA. Contract equality is checked after overrides;
full generated hashes must match after execution. Arrival records are checked against
regenerated IDs, timing, graph sizes and demand totals. Enabled-method reliability
metrics must match the shared evaluator applied to recorded placements. This verifies
deterministic regeneration, not runtime instrumentation of every input draw. Different
evaluation seeds define different paired substrate/workload scenarios.

Outputs preserve the crossed structure:

- `results_long.csv`: every evaluation, with train_seed and eval_seed.
- `results_by_train_seed.csv`: average evaluation metrics for each trained model.
- `results_by_eval_seed.csv`: average models within each shared scenario.
- `results_summary.csv/.json`: hierarchical mean, train_std, eval_std, counts.
- `paired_differences_by_eval_seed.csv`: method differences on shared scenario means.

For learned methods the overall mean averages model means. train_std is sample SD
(ddof=1) across model means; eval_std is sample SD across scenario means after averaging
models. For heuristics only scenario means/SD apply. These observations are dependent:
SDs are descriptive, not confidence intervals. Single-observation SD is null/NaN. Missing
conditional SLA means remain null with counts; available values are averaged without
inventing unavailable services. Figures show separate model/scenario SD bars.

## Layout, provenance and compatibility

The result root contains input/CLI/raw/resolved configs, execution summary, source status,
commit/patch and both framework source snapshots, hardware, commands/status, manifest,
aggregates, figures/reports, and COMPLETE.txt. Every planned final Virne config is also
saved before launching. Actual process config.yaml remains authoritative.

Method grouping preserves the original launcher's normal solver/run-ID tier:

```
methods/<method>/train/<solver>/<unique_run_id>/
methods/<method>/eval/train_<training_seed>/<solver>/<unique_run_id>/
methods/<heuristic>/eval/<solver>/<unique_run_id>/
```

Each leaf retains config.yaml, hydra_config.yaml, command, stdout/logs, records, runtime,
source commit and model where applicable. Heuristics have no fake training/model branch.
The extra solver tier is intentional reuse of the verified launcher, rather than moving
artifacts after execution and invalidating recorded paths.

`scripts/thesis/experiments/run_pipeline.sh` remains the v1 compatibility workflow. It
still explicitly performs its original calibration→execution sequence. V2's normal
execution never invokes it. V2 examples map v1 LOCAL_DEV topology/resources/seeds/budgets
and provisional comparator settings; beta is explicit in each aware method instead of
silently recalculated on each execution. The shared collector/checks/telemetry helpers
preserve v1 defaults and are covered by regression tests.

## Tests

```bash
TEST_DEPS=/path/to/pytest-and-tomli-dependencies \
  bash scripts/thesis/test.sh tests scripts/thesis/experiments/test_pipeline.py \
  scripts/thesis/experiments_v2/tests -q

# Or include tests as an execution stage:
python scripts/thesis/experiments_v2/run_experiment.py \
  --config scripts/thesis/experiments_v2/examples/local_dev.yaml --run-tests
```

Use smaller request budgets for a complete CPU check while retaining all enabled methods:
`--set environment.workload.num_requests=4`. Scientific GPU budgets, workload/resource
models, CTMC distributions, beta/objective/reward and comparator settings require explicit
review/freezing before real experiments. CUDA is detected and gracefully falls back to CPU.
