# Canonical thesis experiment pipeline (v1 compatibility)

For the independent calibration and execution tools, see [Framework v2](../experiments_v2/README.md).
This original calibration-plus-execution workflow remains available for compatibility.

These are **preliminary pipeline-validation experiments**, not scientific superiority claims.
The frozen solvers, objectives, rewards and existing scripts are unchanged.

From the repository root:

```bash
./scripts/thesis/experiments/run_pipeline.sh --profile local_dev
./scripts/thesis/experiments/run_pipeline.sh --profile local_dev --quick
./scripts/thesis/experiments/run_pipeline.sh --profile gpu_server
./scripts/thesis/experiments/run_pipeline.sh --profile local_dev --dry-run
```

Use `PYTHON=/path/to/python` to select an environment containing the existing Virne
requirements plus matplotlib. Default is the previously verified local Python 3.10
virtual environment. `--profile /absolute/path/to/custom.yaml` loads a custom profile.
`--output /new/path` selects the bundle; existing paths are refused. No job is resumed
or overwritten. Failure leaves logs, FAILED.json, status.tsv and a partial manifest,
without COMPLETE.txt. Commands are shell-quoted and journaled before execution.

`--run-tests` also invokes the existing `scripts/thesis/test.sh`. Set `TEST_DEPS`
to a directory containing pytest/tomli if they are not in the selected Python environment:

```bash
TEST_DEPS=/path/to/test_dependencies ./scripts/thesis/experiments/run_pipeline.sh --run-tests
```

Stages run sequentially: preflight → optional tests → CPU NRM/GRC calibration → freeze
resolved config → offline pretraining → checkpoint reload and online evaluation →
aggregation → structural validation → matplotlib PNG/PDF plots → reports → COMPLETE.
GPU detection is recorded and GPU_SERVER gracefully falls back to CPU. LOCAL_DEV always
uses CPU, one epoch and one thread; GPU_SERVER has explicit editable larger budgets.
GPU_SERVER defaults are candidate budgets **requiring scientific review**, not frozen
scientific choices. It uses the same pipeline and no implicit smoke launcher profile.

## Inputs and reproducibility

Actual schema inspected: `virne/configs/main.yaml`, `learning.yaml`, `thesis.yaml`,
`p_net_setting/default.yaml`, `v_sim_setting/default.yaml`; generator and lifecycle:
`BaseSystem.load_dataset`, `Environment.reset`, `PhysicalNetwork.from_setting`,
`VirtualNetworkRequestSimulator.renew`, reliability model/evaluator and Counter.
Every job calls existing `scripts/thesis/run.sh` with PROFILE=configured and explicit
overrides, including those that replace that launcher's small-budget defaults.
The original baseline retains disabled reliability/SLA/features and fixed_intermediate
reward. Its learned checkpoint therefore has the original feature signature.

For every test seed, each method regenerates PN and VNR from identical topology,
resource and workload settings. The generator resets seeds independently before PN
and VNR generation and Environment.reset renews VNR with the test seed. Reliability
and SLA use independent NumPy generators and do not perturb workload draws.
No cached datasets are loaded. The pipeline regenerates each final job's inputs,
writes full topology/resource/request/SLA snapshots under `input_fingerprints/`,
and requires identical SHA256 hashes across all methods and training replicas for
each test seed. Record request IDs, arrival times, lifetimes, node/edge counts and resource-demand totals are
checked against regenerated inputs. This is deterministic generation verification,
not runtime instrumentation of every resource draw. Changes to generators require
rerunning these checks. Distinct test seeds also generate distinct substrates;
this experiment evaluates paired seed-specific environments rather than a single
fixed substrate. Training seeds and test seeds must be disjoint.

Baseline SLA/availability is assessed after execution using its recorded node_slots
and link_paths, deterministic SLA labels and the same CTMC attributes as other methods.
The frozen ReliabilityEvaluator counts unique hosts, transit nodes and links. Enabled
methods' recorded availability/shortfall/satisfaction must agree with this computation.
Raw Virne CSVs stay unchanged; `assessment_arrivals.csv` contains derived measurements.
Acceptance/economic metrics come directly from Counter. `avg_r2c_ratio` follows Virne's
all-arrival definition; SLA averages are among accepted services, SLA service rate uses
all arrivals. Missing metrics stay NaN/null; single-replicate std is NaN (ddof=1).
Original heuristic reward labels describe the active assessment config, not learning.

## Calibration

LOCAL_DEV runs NRM and GRC over 2 arrival rates × 2 CTMC failure rates × 3 seeds = 24
CPU jobs, 24 requests each. Arrival intensity uses the existing continuous Poisson
`arrival_rate.rate`; lifetime uses exponential `lifetime.scale`. SLA levels and equal
probabilities are fixed. Scalar CTMC rates are deliberately homogeneous for this
pipeline exercise. Calibration records resource utilization and A/D distributions,
including per-class metrics and per-arrival derived data.

Selection minimizes explicit distance penalties in calibrate.score: normalized distance
of acceptance and satisfaction from configured intervals, load shortfall and increases
in satisfaction across ascending SLA classes. Qualification additionally requires all
four accepted SLA classes. Qualifying points are preferred; ties use grid ID. If none
qualifies, the least-degenerate point is marked provisional. No proposed-method outcome
enters the score. Beta is calibrated to exp(-beta * median positive D) = configured
target compliance. No positive D is a hard calibration failure requiring grid expansion.
Thresholds, grid and target are in the profile and report; no beta is silently frozen.
The configured reference_beta is used only for heuristic calibration metrics and is
replaced by the selected shortfall-derived beta before learned-model training.

## Methods and scientific limits

All 11 ready methods execute: original PPO-DualGAT, reliability-aware PPO-DualGAT,
Cand-RL, FFD, NRM, GRC, RW, RA-NRM, RA-GRC, RA-EVNE-A, RA-EVNE-original.
Cand-RL uses thesis_joint, a Cand-specific MLP actor/critic and Virne PPO, redundancy=none.
RA combine/normalization/alpha and EVNE empty-distance/Weibull/history assumptions are
explicit profile settings based on the earlier verified DEV suite. Zero history arrays
are expanded to substrate size and stored in actual job configs. They are not tuned
scientific settings or exact Paper-9 reproductions. No backup/potential reward is added.

Before scientific GPU experiments, review/freeze workload/topology/resource distributions,
CTMC heterogeneity and mode, objective/reward and beta calibration, RA ranking parameters,
EVNE histories, Cand K/representation/architecture, training budgets and held-out seeds.
Edit `profiles/gpu_server.yaml` explicitly. Aggregation pools training and test seeds;
reported std is descriptive, not a confidence interval or independence claim.

## Outputs and validation

Bundles default to `results/thesis_experiments/<UTC timestamp>_<profile>/` and include
source status/commit/patch plus a snapshot of the experiment scripts, hardware/device,
active profile, resolved environment config, commands/status, calibration records,
training/evaluation configs/logs/models, full input fingerprints, manifests, long and
mean/std/count tables, machine-readable validation and experiment reports, human reports,
and preliminary PNG/PDF figures. Run directories are `runs/<method>/<solver>/<run_id>/`.
`COMPLETE.txt` requires every structural stage and required output. Scientific warnings
(e.g. provisional calibration, degenerate tiny samples) remain visible without failing
LOCAL_DEV. Subprocess failures are fatal; failed comparators are never silently omitted.

Training reward/loss plots use training_info.csv update telemetry; epoch plots use available epoch logs. One epoch cannot show convergence;
`training_telemetry.json` explains available data. The pipeline does not manufacture a
reward series. Test streams and job counts are verified, learned evaluations must show
successful checkpoint load in stdout, and final departures must release all services.
The validator checks counts, ranges, finiteness, source/config/model/record artifacts
and paired stream hashes. Calibration uses CPU even for GPU_SERVER.

Quick keeps all methods, two held-out test seeds and EPOCHS=1, shrinking calibration to
8 requests and one seed, and training/evaluation to 4 requests. CPU runtime is dominated
by Python/PyTorch startup and PPO training; see COMPLETE.txt for measured run duration.
Each stage script can be rerun on a bundle for inspection; the top-level pipeline never
overwrites a bundle. Do not rerun raw commands.sh into an existing bundle: overwrite
guards will refuse those jobs. Use the top-level command to create a fresh full run.
