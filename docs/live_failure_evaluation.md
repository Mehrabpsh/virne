# Live failure/repair evaluation (approach 2)

This implements live chronological failures during frozen-policy evaluation.
There is no separate failure-replay experiment. Existing analytic availability,
training rewards, observation dimensions, resource accounting and candidate
semantics remain unchanged when the mode is disabled (the default).

## Execution contract

- Independent alternating exponential up/down processes use the existing node
  and link `lambda_f`/`mu_r` values. `node_only` realizes node failures only;
  `node_and_link` also realizes link failures. There is no new software-VNF
  failure process or correlated-domain model.
- Requests, departures and component transitions share simulation time. Placement
  is atomic at its arrival time. Component transitions at time t precede workload
  events at t; the existing departure-before-arrival workload ordering is kept.
  Service observation intervals are half-open: a failure exactly at departure
  contributes no downtime or service interruption.
- Failed components retain their IDs, capacities and current reservations.
  They cannot host new placements or carry new traffic. Transit-node failures
  disable paths too. Existing services wait for all required repairs, then resume.
  Original departures release resources exactly once, even during an outage.
- Overlapping outages count the union of unavailable time, not a sum per failed
  component. One failed host affects every service that depends on it.
- There is no rerouting, migration, replica creation, or change to policy weights.
  Nonzero training epochs and OfflineSystem are rejected for live execution.
  The experiment-v2 planner disables the mode for every training job regardless
  of the evaluation setting. The MCF routing backend is not supported in live mode.
- Common controller checks, action-candidate filtering and operational routing
  enforce feasibility for all methods. Cand-RL's existing endpoints and top-K
  restrictions remain; a down endpoint produces no candidates. No new feature
  channel is added to trained networks. In particular, a GNN is not assumed to
  infer status automatically: the current implementation supplies feasibility
  through the common masks/checks, while retaining the trained feature schema.

Each component has an isolated RNG derived from its identity, workload seed and
failure seed offset. Traces are independent of policy RNG draws and admission
decisions. The common physical model is also used for unaware baselines, without
adding reliability/SLA observation channels or changing their reward.

The default is explicit **all-up initialization**. `stationary` independently
samples initial component states from their stationary availabilities. Initial
states, complete traces and a SHA-256 trajectory fingerprint are saved. A hard
event limit aborts rather than silently truncates an experiment.

For an accepted service arriving at a with lifetime L, departure remains a+L.
Outages never pause the lifetime. Observed availability is uptime/L on [a,a+L);
a service that is down near departure leaves on schedule and its reservations
are released without waiting for repair. This ratio is empirical availability,
not the probability of experiencing no failures over the whole lifetime.

`seed_offset` is an RNG namespace mixed with the scenario seed and component
identity. It changes sampled event times, not lambda/mu, and does not shift the
simulation clock. Up durations have mean 1/lambda_f and down durations mean
1/mu_r. `event_limit` counts total failure plus repair transitions over all
modeled components through the final request departure in one evaluation run.
It is neither a request limit nor a time limit; exceeding it aborts the run.

## Evaluate an existing checkpoint without retraining

From the repository root, using the server's established Python environment:

```bash
python -m scripts.thesis.evaluate_live_failures \
  --config /absolute/path/to/training-run/config.yaml \
  --checkpoint /absolute/path/to/training-run/models/model.pkl \
  --seed 100 \
  --initial-state all_up \
  --output /absolute/path/to/new/live-eval-seed100
```

Use the **saved per-run Virne config**, not an experiment-v2 input YAML. The
output root must not exist. The command reuses the architecture, resource and
reliability settings, selects zero training epochs and an online environment,
and gives logs and runtime audits new output paths. It reloads and evaluates the
checkpoint, verifies unchanged policy/checkpoint hashes and no optimizer
updates, and writes `COMPLETE.txt` only after these checks pass. Choose a separate
new output for each evaluation seed. Device settings are inherited from the saved
config. This command does not run a new training phase.

Output includes `evaluation_report.md`, original-config and source-code hashes,
`frozen_policy_verification.json`, and a native `<solver>/eval_live_seed_<seed>/`
run containing the artifacts below. Common post-hoc analytic assessment is also
written for ordinary PPO, keeping predicted and observed results distinct.

## Experiment-v2 integration

Existing input YAMLs gain these compatible defaults during resolution:

```yaml
protocol:
  evaluation:
    live_failures:
      enabled: false
      seed_offset: 1000003
      initial_state: all_up
      event_limit: 1000000
```

Enable with:

```bash
python -m scripts.thesis.experiments_v2.run_experiment \
  --config scripts/thesis/experiments_v2/examples/gpu_server.calibrated.v2.yaml \
  --methods original_ppo,reliability_ppo,cand_rl \
  --set protocol.evaluation.live_failures.enabled=true \
  --set protocol.training.online=true
```

This runner uses a **training-then-paired-evaluation** matrix: training has
no realized failures. Failure-free held-out evaluation always runs; enabling
live failures adds an online failure/repair evaluation with no further training.
Use the standalone checkpoint command above when training has already finished.
No rates, objective parameters, workload or training budgets are changed by
turning on the evaluation mode. Each pair uses the same saved checkpoint,
topology, requests, resource demands, SLA assignments and workload seed.

Enabled now means **both failure-free and live held-out evaluations**; disabled
means failure-free held-out evaluation only. This supersedes commit d7af1a0's
live-only behavior: historical results from that commit contain live held-out
evaluations only and do not acquire a failure-free reference retroactively.
Usual internal training validation stays failure-free. All selected methods receive the same modes;
heuristics skip training. Methods run separately with matched exogenous traces,
not together as competing policies in one shared resource ledger.

## Full experiment configuration

`scripts/thesis/experiments_v2/examples/gpu_server.live_failures.full.v2.yaml`
is a ready-to-run full proposal, retaining the calibrated physical environment,
comparator objectives and architectures. It sets:

- Nine active methods: original PPO, reliability-aware PPO, Cand-RL, FFD,
  NRM, GRC, RW, RA-NRM and RA-GRC. Both RA-EVNE variants are disabled in all
  shipped experiment-v2 examples and excluded from legacy default runners.
- Three independent training seeds `[31,37,43]`, 50 epochs each; batch size 16
  and target steps 32, matching the corrected PPO's 50-epoch test settings.
- Both PPO variants use provisional actor LR 0.0001 and guarded target KL 0.02;
  see [PPO stability](ppo_stability.md). Cand-RL keeps its existing optimizer.
- Evaluation scenarios `[100,110,120]`, 1,000 requests per scenario, live
  failures enabled, all-up initialization and offset 1000003.
- 9 training jobs, 45 failure-free evaluations and45 live evaluations:
  **99 total jobs**. Training/internal validation is analytic; both held-out modes
  are frozen. The two modes reuse one trained checkpoint; no extra training occurs.

Every evaluation manifest/aggregate row has `evaluation_mode` equal to
`failure_free` or `live_failures`. Failure-free run IDs/layout remain compatible;
additional live jobs use `eval_live__...` and an `eval/live_failures/` directory.
Summary tables, model/scenario variation, method pairing, SLA-class plots and
interpretations are separated by mode. `paired_evaluation_modes.csv` reports
live minus failure-free for each method/model/workload pair. Observed outage
metrics are not measured in failure-free runs and remain null there; they are
never filled with invented zeros or pooled into the live measurements.

From a fresh checkout, reuse the server's established virtual environment rather
than replacing its scientific dependencies:

```bash
export SIM_PYTHON=/local_home/pashanejad/ThesisProject/virne/.venv/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$SIM_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

"$SIM_PYTHON" -m scripts.thesis.experiments_v2.run_experiment \
  --config scripts/thesis/experiments_v2/examples/gpu_server.live_failures.full.v2.yaml \
  --dry-run

# This next command launches the full experiment.
"$SIM_PYTHON" -m scripts.thesis.experiments_v2.run_experiment \
  --config scripts/thesis/experiments_v2/examples/gpu_server.live_failures.full.v2.yaml
```

The runner prints the new result directory. Archive the entire directory, not
only reports/plots. `COMPLETE.txt` and a passed `reports/validation_report.json`
are the completion checks. Live component and per-service files reside in each
native evaluation run. Inspect component-event and affected-service counts:
nominal calibrated rates can produce relatively few outages in short scenarios.
Do not infer resilience merely from a high observed uptime with little exposure.

Observed fields may be added to the experiment's configured `metrics` list:

- `observed_service_availability`: fulfilled / admitted service-time (lifetime
  weighted, conditional on acceptance; null when no service was accepted).
- `fulfilled_service_time_ratio`: fulfilled / all requested service-time;
  rejected requests contribute zero fulfillment, not perfect availability.
- `observed_sla_satisfaction_rate`: accepted services whose observed lifetime
  availability meets their own SLA / accepted services (null for zero accepted).
- `observed_sla_service_rate_all_arrivals`: same compliant count / all arrivals.
- `total_service_downtime`, `interrupted_service_fraction`,
  `service_downtime_p95` (accepted-service population, including zero downtime).
- `component_failure_count`, `component_repair_count` (physical transitions,
  distinct from request placement/routing rejections).
- `requested_service_time`, `admitted_service_time`, `fulfilled_service_time`.

When the mode is disabled, configured observed outage metrics stay null, the
resolver records a warning, and their plots are omitted. No live jobs run and
unmeasured observations are never fabricated as zero. The existing revenue/cost
definitions are retained and are not automatically converted to uptime billing.

## Artifacts, interpretation and validation

Every native live run writes:

- `live_failure_trace.json`: component identities/rates, initial states, full
  trajectory, observation horizon and trace hash.
- `observed_services.csv`: one row per arrival with acceptance, lifetime,
  accumulated downtime, interruption count and empirical SLA outcome.
- `live_failure_summary.json`: aggregate denominators, service delivery and
  per-SLA-class observed outcomes.

All requests are observed through their original departures. The v2 collector
regenerates the deterministic clock as an accounting audit, checks the archived
trace hash and per-service downtime, verifies accepted mappings were operational
at admission, and compares aggregate live metrics. This audit is not an exposed
replay evaluation mode. The validator requires matching trace fingerprints for
all methods/models sharing an evaluation seed, and retains frozen-policy and
resource-restoration checks.

HTML and Markdown reports add evidence-based interpretation and distinguish
predicted steady-state availability from observed lifetime uptime. A zero median
shortfall does not imply no violations. Zero component failure transitions are
explicitly identified as inadequate evidence of outage robustness.

A finite lifetime can have 100% observed uptime even when predicted steady-state
availability is below SLA. Initial all-up versus stationary conditions also affect
finite observations. Component failure counts, affected-service counts, repeated
traces and independent training seeds are needed before comparative conclusions.
The live simulator shares the analytic model's independence/exponential
assumptions; model consistency is not independent evidence of real-world validity.

## Validation scope

Tests exercise overlapping host/transit/link outages, failure/repair ordering,
departure while down, rollback and unsafe placement paths, retained capacities,
trajectory RNG isolation, stationary initialization, event limits, and original
PPO/reliability-aware PPO/Cand-RL checkpoint evaluation without weight changes.
Synthetic test rates intentionally exercise outages and are not thesis settings.
See the implementation verification artifacts for exact commands and outcomes;
no GPU scientific experiment is implied by local test success.
