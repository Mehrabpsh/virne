# Heterogeneous reliability implementation and review boundary

Implemented on the current working tree; numerical settings remain provisional.
The shared component availability, unique-domain evaluator, normalized SLA shortfall,
and multiplicative objective formulas are unchanged. No redundancy or realized
failure/repair-event simulation has been added to Cand-RL.

## Configuration

The shared rate provider and v2 adapter accept:

```yaml
environment:
  reliability:
    backend: analytic
    mode: node_and_link
    config_id: PIPELINE_V2_PROVISIONAL
    nodes:
      lambda_f:
        distribution: categorical
        values: [0.00001, 0.00005, 0.0001, 0.0005]
        probabilities: [0.25, 0.25, 0.25, 0.25]
      mu_r: 0.1
    links: {lambda_f: 0.000025, mu_r: 0.1}
```

Node probabilities and request SLA probabilities are independent settings. The GPU
example preserves request probabilities `[0.4, 0.3, 0.2, 0.1]`. Rates have a common
abstract time unit; no physical time-unit calibration is claimed.

In categorical mode, SeedSequence child streams separate node failure, node repair,
link failure and link repair draws. These are separate from global topology/resource
RNGs and the SLA generator. Classes are drawn once per PN, in stable component-ID
order. Stored attributes include `lambda_f`, `mu_r`, `availability`, zero-based
`lambda_f_class` and `reliability_class`; categorical repair adds `mu_r_class`.
Existing GML persistence and input snapshots preserve them. Class IDs index the
configured values vector. Zero-probability classes are retained as IDs but not sampled.
Legacy scalar/uniform configurations retain their historical draws.

## Shared scaling and observations

`LogRiskScaler` uses fixed bounds from the configured active rate support, never the
minimum/maximum of the sampled PN. Uniform bounds and explicit scalar/array support
are also supported. Homogeneous risk maps to zero and preference to one.

For risk `r=-log(A)`, normalized risk is `(r-best)/(worst-best)` and ranking preference
is one minus normalized risk. Values outside the configured support are rejected.

The optional RA-NRM/RA-GRC combination is:

```yaml
ra_heuristic:
  normalization: max
  combine: normalized_risk_weighted_sum
  alpha: 0.5  # illustrative, not scientifically frozen
```

It computes `alpha * normalized_base + (1-alpha) * risk_preference`. Original product
and raw-availability weighted-sum modes remain available. Homogeneous availability
preserves base ordering.

DualGAT adds optional `normalized_log_risk` to physical-node features only, and
`remaining_budget_fraction` to physical/virtual context. Existing raw/log features
remain selectable. Fractional budget is `1 + log(A_partial)/(-log(A_required))`:
one initially, zero at exhaustion, negative after violation. No clipping is applied.
Partial availability includes all distinct used hosts, transit nodes and links.

Cand opts in separately:

```yaml
observation_features: [normalized_log_risk, remaining_budget_fraction]
```

Candidate rows retain resource fractions and true availability, then append normalized
node risk. Remaining budget is appended after demand and SLA context. Padding remains
zero and masked. For K=10 and one resource, input width changes from 22 to 33;
actor output remains 10, critic output one. Empty feature lists preserve legacy shapes.
No candidate-routing lookahead or endpoint behavior was added.

Normalized-risk checkpoint signatures contain their configured scaling bounds.
New observations need new checkpoints; changing bounds is also incompatible. Old
Cand checkpoints remain supported with optional observation features disabled.

## Rewards and effective controls

The GPU example selects existing `joint_fixed` with intermediate reward 0.1: successful
nonterminal placement/routing +0.1, failure -0.1, completed service U. Original PPO
retains `fixed_intermediate` and terminal economic R2C. `joint_terminal` remains available
for ablation. Beta is unchanged and remains provisional; suggestions are never adopted
unless `adopt_beta_for` names methods explicitly.

PPO now consumes configured repeat_times, gae_lambda and eps_clip instead of constructor
defaults alone. Standard DualGAT consumes nn.num_gnn_layers. Effective repeat count,
GAT depth and reward name are written into runtime audit metadata. Checkpoints formerly
trained with ignored depth settings may disagree with corrected configured depth:
retrain or explicitly configure the old *actual* depth; do not claim their saved
configuration accurately described the old training execution.

## Distribution-aware calibration

`calibration.reliability_scenarios` maps names to complete environment reliability
sections. Scenarios cross with the ordinary environment search_space, but reliability
search paths and named scenarios cannot be combined. This keeps class distributions
atomic. Existing scalar grids still work.

The GPU example supplies only the proposed node classes with unchanged link rates,
crossed with two arrival rates. These are candidate settings, not calibrated values.

Diagnostics distinguish:

* proven_impossible: an optimistic availability bound is below SLA;
* witnessed_attainable: a recorded feasible accepted embedding satisfies SLA;
* unknown: neither conclusion was established.

For an n-node connected request, an optimistic bound multiplies the n best PN node
availabilities and the n-1 best link availabilities. It ignores topology/resource
constraints and additional transit domains; passing this bound never proves feasibility.
Disconnected requests use n minus virtual component count as the edge lower bound.
This relies on the shared non-reusable host contract.

Collection records per-class attainability counts, observed utility/shortfall quantiles,
utility-near-zero fractions, class counts and availability spread. Diagnostic utilities
use method objective beta when present, otherwise reporting assessment beta; each JSON
entry identifies its beta. Existing evaluation metrics and denominators are unchanged.

`criteria.require_no_proven_impossible: true` adds the proven-impossible arrival fraction
to selection loss. This is distinct from accepted-class coverage. It is not an exact
feasibility certificate. `require_qualified: true` refuses to export calibrated YAML
or mark completion when every point fails qualification; diagnostic JSON and FAILED.json
remain. Legacy provisional selection is retained when this flag is false.

The proposed best hosts and current links cannot attain SLA .9999. The resolver warns
even in dry-run. The illustrative GPU calibration deliberately uses strict qualification,
so it cannot export a qualifying configuration while that impossibility remains.
Whether to improve node/link support or designate intentional stress classes remains
a scientific review decision. No replacement rates or increased beta were invented.

## Live input evidence

Runner-enabled environments capture runtime_inputs.json after resetting/generating the
initial PN and full request stream, before any solver decision. Collection compares all
captured node/edge resources, reliability attributes, request graphs, demands, times,
lifetimes and SLA values against regeneration. A mismatch fails collection. v2 validation
requires this evidence for every evaluation. Captures are overwritten at reset; the
final evaluation capture remains in each run directory. Historical bundles can still
be collected without a capture, but do not pass the new live-input requirement.

Disabled-awareness baseline captures remain raw and contain no policy-visible reliability
metadata. Only after their raw runtime inputs are verified does post-hoc assessment add
the deterministic shared reliability/SLA world for cross-method comparison.

## Review commands

Run from the repository root with the Virne Python environment:

```bash
python scripts/thesis/experiments_v2/run_experiment.py \
  --config scripts/thesis/experiments_v2/examples/gpu_server.yaml --dry-run
python scripts/thesis/experiments_v2/calibrate.py \
  --config scripts/thesis/experiments_v2/examples/gpu_server.yaml --dry-run
```

These commands plan only. Scientific calibration/training should follow review of the
scenario support, SLA attainability policy, alpha, endpoint protocol and budgets.
