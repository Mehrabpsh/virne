# Comparator provenance and reconstruction boundaries

Current source of truth: integrated checkpoint 1405-07-16. Comparator parameters
below are configuration candidates, not final thesis experimental choices.

## Virne originals

Unchanged original registrations: `ppo_dual_gat+`, `ffd_rank`, `nrm_rank`,
`grc_rank`, `rw_rank`. All can be measured through the shared recorder/evaluator
with reliability metadata enabled; their placement criteria remain original.
RW is used as the checkpoint's RW/MaxMatch representative; no new MaxMatch claim.

## Reliability-aware heuristics

- `ra_nrm`, `ra_grc`: **thesis adaptations**. Original virtual-node ranking;
  physical base rank with explicitly configured `max` or `none` normalization, then configured
  `product` or `weighted_sum` with shared singleton availability. Weighted sum
  requires explicit `ra_heuristic.alpha`. Winner/combine remains validation-open.
- `ra_evne_original`: **published ranking equation port**, not full reproduction.
  Su et al. (2018), *Research on Node Reliability-aware Efficient Virtual Network
  Embedding Algorithm*, Advanced Engineering Sciences 50(2), 126–132.
  Local article inspected from `/home/mehrab/Workspaces/Latex/Thesis_Plan/Thesis-virne-as-baseline/last_hope/ppo-dual-gat-failure-awareness-idea/papers/Research on Node Reliability-aware Efficient Virtual Network Embedding Algorithm.pdf`.
  Virtual importance `(CPU + incident bandwidth) / sum hop distances` (eq.13);
  physical importance multiplied by Weibull-age/environment reliability and
  divided by hop distance to all placed hosts (eqs.12,15–18). Explicit static
  age/failure history inputs, no realized failure-event simulation. Ranking
  metadata is distinct from thesis CTMC availability used for final measurement.
- `ra_evne_a`: same importance/placement pipeline, replaces published reliability
  factor with shared CTMC singleton availability. **Thesis adaptation**.
- Required explicit CPU/bandwidth attribute names. The paper's LS denominator is
  zero before the first placement; `ra_evne.empty_distance` is an **explicit
  reconstruction convention**. DEV=1 is for smoke only. Disconnected global
  centrality is zero. Stable iteration order breaks ties. Virne preserves its
  own feasibility constraints and k-shortest routing; geographic constraints
  are only enforced when supplied by the workload/controller.

## Cand-RL

Primary recovered evidence: `../cand_rl_forensic_reconstruction.md`, especially
§§3,7,8 and Algorithm G (Paper-9 equations 12–14). No new claim of exact author
code or policy recovery. `spider/` and `predecessor/` are evidence, not baselines.

| Component | Paper recovered | Code / SPIDER lineage | Reconstruction implemented | Validation selection | Still unknown |
|---|---|---|---|---|---|
| Cand score | d(n,current source)+d(n,destination), top K | Similar score/filter skeleton | Stable feasible/controller-order ties; guard disconnected nodes; isolated nodes retained | Hop/delay setting explicit | Published weight/tie details |
| Resource filtering | Resource/type/bandwidth | Resource/type checks | Virne candidate feasibility plus max incident VNR link demand for bandwidth prefilter | Workload profiles | Exact mapping for arbitrary graph VNR versus ordered SFC |
| Endpoints | Substrate source/destination; source advances | SPIDER uses fixed original source | Request graph endpoints preferred; explicit fixed config fallback; previous placed host advances source | Endpoint workload provider pending | Exact original endpoint distribution |
| Candidate observation | K rows CPU/memory/storage fractions + A_n; current demands + SLA | Similar candidate resource state | Configured resource/capacity names; flattened matrix and current demands/SLA | Resource workload, MLP widths | Exact author serialization/network |
| PPO architecture | Actor/critic PPO | No trained policy recovered | Independent configurable MLP actor and critic; Virne PPO updater | hidden_dims and shared PPO config | Author exact widths/initialization/update settings |
| Action | Candidate index + replication bit | Different replica semantics in SPIDER | Main comparison currently explicit `redundancy=none`, candidate-index action | K and padding-mask bool | Backup accounting/routing needed for replication bit |
| Execution | Sequential VNF placement | Sequential path pipeline | Shared Virne place-and-route, rollback, deployment and departure; processes graph VNR edges | None | Ingress/egress flow resource model (not present in shared Virne workload) |
| Reward main | Thesis checkpoint shared objective | N/A | Same joint reward registry/evaluator as PPO-DualGAT | Four objectives/four implemented shapings | Potential Phi |
| Paper9Reward | Symbolic equations and -10/-100 | No matching implementation recovered | Standalone exact printed formula, no sign repair; constants all explicit | Constant calibration/gate | Operational cost/energy metadata, acceptance semantics, SPN and redundancy |
| Episode | Approx.1000-hour training episode in paper | Predecessor different behavior | Main thesis one request per episode | Fixed checkpoint semantics | Reproduction episode boundary |
| Save/load | Not recovered | No author weights | Strict checkpoint + full Cand config signature | None | Exact author weights |

`cand_rl.profile=thesis_joint` is executable with explicit reconstruction inputs.
`cand_rl.profile=paper9_sanity` is deliberately **BLOCKED** with an informative
error; enabling it without recovered/reconstructed backup accounting,
operational cost/energy inputs, acceptance gate and calibrated reward constants
would silently change the published comparator. `Paper9Reward` is independently
implemented/testable for the recovered symbolic formula.

No claims of Cand-RL superiority or exact reproduction follow from DEV smoke runs.

## DEV-only example commands

```bash
PROFILE=smoke SOLVER=ra_nrm EPOCHS=0 scripts/thesis/test_online.sh ra_heuristic.combine=product ra_heuristic.normalization=max
PROFILE=smoke SOLVER=ra_grc EPOCHS=0 scripts/thesis/test_online.sh ra_heuristic.combine=weighted_sum ra_heuristic.normalization=max ra_heuristic.alpha=0.5
PROFILE=smoke SOLVER=ra_evne_a EPOCHS=0 scripts/thesis/test_online.sh ra_evne.cpu_attribute=cpu ra_evne.bandwidth_attribute=bw ra_evne.empty_distance=1
PROFILE=smoke SOLVER=cand_rl scripts/thesis/train_offline.sh cand_rl.k=3 cand_rl.distance_weight=hops cand_rl.source=0 cand_rl.destination=1 'cand_rl.resource_names=[cpu]' 'cand_rl.capacity_attributes=[max_cpu]' 'cand_rl.hidden_dims=[16,16]' cand_rl.mask_padding=true cand_rl.redundancy=none reliability_features.enabled=false
```
