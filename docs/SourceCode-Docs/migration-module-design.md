# Migration Module for PPO_Dual_GAT+ (Virne) — Architecture & Implementation Design

Scope decision (agreed): rule-based trigger + rule-based VNF selection + **frozen** PPO_Dual_GAT+ policy reused at inference for destination selection. No separate migration reward, no additional training loop in this phase. Evaluation via the R2C ledger (Layer 2), as defined previously.

**Scope narrowing (this revision):** migration is triggered **only by node/link failure** (hard trigger). Overload-based/soft migration (proactive migration to relieve a congested-but-alive node) is explicitly out of scope for this thesis and left as future work. This removes an entire class of design questions (utilization thresholds, "how often can we churn migrations" budgeting, live vs. cold transfer) and lets every migration be modeled as **restore-from-checkpoint onto a new host**, since the source is by definition unreachable when the trigger fires.

---

## 1. Architecture

### 1.1 Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Online VNE Event Loop                        │
│  (existing Virne simulation clock: SFC arrivals + departures)        │
└───────────────┬─────────────────────────────────┬────────────────────┘
                │                                 │
                ▼                                 ▼
     ┌─────────────────────┐          ┌─────────────────────────┐
     │  Embedding Path      │          │  Infra Event Generator   │
     │  (unchanged)         │          │  (NEW: node/link fail,   │
     │  PPO_Dual_GAT+       │          │   join, load spikes)     │
     │  places new SFCs     │          └───────────┬──────────────┘
     └──────────┬───────────┘                      │
                │                                  ▼
                │                       ┌─────────────────────────┐
                │                       │  1. Trigger Detector      │
                │                       │  (threshold rules)        │
                │                       └───────────┬──────────────┘
                │                                  │ fires
                │                                  ▼
                │                       ┌─────────────────────────┐
                │                       │  2. VNF/SFC Selector       │
                │                       │  (priority heuristic)      │
                │                       └───────────┬──────────────┘
                │                                  │ chosen VNF(s)
                │                                  ▼
                │                       ┌─────────────────────────┐
                │                       │  3. Destination Selector   │
                │                       │  (frozen PPO_Dual_GAT+,    │
                │                       │   inference-only, action    │
                │                       │   space masked to eligible │
                │                       │   PNs)                     │
                │                       └───────────┬──────────────┘
                │                                  │ target PN
                │                                  ▼
                │                       ┌─────────────────────────┐
                │                       │  4. Migration Executor     │
                │                       │  (cost/downtime model,     │
                │                       │   routing update, rollback)│
                │                       └───────────┬──────────────┘
                │                                  │
                ▼                                  ▼
     ┌───────────────────────────────────────────────────────────┐
     │            5. Cost/Revenue Ledger (R2C accounting)          │
     │  Revenue_total, Cost_embedding, Cost_migration → R2C(t)      │
     └───────────────────────────────────────────────────────────┘
```

### 1.2 Component responsibilities

**Trigger Detector**
- Subscribes to the infra event stream only (no utilization polling — no soft/overload trigger in this scope).
- Fires exclusively on: node failure (100% forced trigger for every VNF hosted there) or link failure (forced trigger for every SFC whose current path traverses that link).
- **Link failure runs a re-route-first check before invoking migration at all** (see §1.2a below) — the trigger only escalates to the full migration pipeline if no feasible alternate path exists.
- Stateless, deterministic, cheap, and now also *rare and bursty* rather than continuous — it only does anything on the (infrequent) timesteps where an infra event actually lands, which simplifies the runtime loop considerably (see §3).

**VNF/SFC Selector**
- Given a node failure affecting PN_x, collects **all** VNFs that were hosted on PN_x (not top-k — a dead node forces every VNF on it to move, there's no "leave it and re-evaluate later" option like there would be under a soft/overload trigger).
- Orders them by a priority score for **sequential processing** (this ordering is what avoids the race condition — see §1.2a):
  `priority = w1 · SLA_criticality + w2 · resource_demand − w3 · remaining_lifetime`
- For link failure, the "selection" is really just the set of SFCs whose active path used that link — no ranking needed for the reroute-first check, since reroute is attempted per-SFC independently; priority ordering matters only for the subset that fails reroute and falls through to full migration.

**Destination Selector (the actual reused component)**
- **Grounded in the actual codebase (inspected directly, not assumed):** this is not a new wrapper to design from scratch — the pieces already exist and compose almost directly.
  - Feasibility masking: `controller.find_candidate_nodes(v_net, p_net, curr_v_node_id, filter=...)` (in `rl_core/rl_enviroment_base.py`) already returns the feasible physical-node set for a given virtual node — it goes through the same constraint checker as ordinary placement, so a failed/zero-capacity node is automatically excluded with no special-case code.
  - Deterministic masked inference: `rl_solver.py`'s `select_action(observation, sample=False)` already does `action_logits = policy.act(observation)` → `apply_mask_to_logit(action_logits, observation['action_mask'])` → argmax. Passing `sample=False` gets you greedy, deterministic inference for free.
  - The policy forward pass itself (`BiGnnBaseModel.forward`, in `dual_gnn_policy.py`) takes `obs = {v_net, p_net, curr_v_node_id}` and returns one logit per physical node from graph structure alone — it has no baked-in assumption that this is a *fresh* placement, so scoring "where should this already-placed VNF go now" is the same call shape as scoring "where should this new VNF go."
- **Implementation shape:** construct `observation = {'v_net': the_sfc_graph, 'p_net': current_p_net_state, 'curr_v_node_id': the_migrating_vnf_id, 'action_mask': controller.find_candidate_nodes(v_net, p_net, curr_v_node_id, filter=[source_node] + already_committed_targets_this_pass)}`, then call the existing `select_action(observation, sample=False)`. The `filter` argument is also where the sequential-commit loop's "exclude nodes already claimed earlier in this pass" logic plugs in naturally — no extra state tracking beyond a running list.
- No gradient updates happen here — this is inference-only reuse of the frozen policy's `act()`, repurposed for a structurally identical decision (node scoring given a resource-demand vector and a graph).
- **Fallback:** if `find_candidate_nodes` returns an empty set (no feasible destination at all — plausible under a large-scale failure), fall back to a simple heuristic (most-free-capacity feasible-by-hard-constraints PN, or none available → the VNF's SFC is terminated early per the Executor's failure handling). Log fallback frequency as a robustness metric.

**Migration Executor**
- **Grounded in the actual codebase:** don't hand-roll allocation/deallocation — `virne/core/controller/controller.py` already exposes exactly the primitives needed: `release` (frees a VNF's current resources — call this on the dead node's VNFs first), `place_and_route` (places one virtual node and routes its adjacent links in a single call, with `undo_place_and_route` for rollback), and a `safely_*` variant family that runs constraint checking inline rather than requiring you to reimplement feasibility checks. Concretely: `controller.release(...)` the dead placement, then `controller.place_and_route(v_net, p_net, vnf_id, dest_p_node_id, solution, ...)` for the new one, per VNF, in priority order.
- Since the source node/link is down, there is **no live state to pull** — every migration is a **restore-from-checkpoint** onto the destination, not a bandwidth-proportional live transfer. Assumption to state explicitly in the thesis: VNF state is periodically checkpointed to a control-plane store independent of any single physical node (standard practice cited in the migration literature — the survey's own taxonomy names "checkpointing or log-replay" as the mechanism for stateful migration). You are not implementing the checkpoint store itself, just its cost consequence.
- Computes a fixed `C_restore` cost (time/resource to re-instantiate from the last checkpoint) plus a `downtime_penalty` that now includes **failure-detection time** in addition to restore time — this is larger than a live-migration downtime would be, and that's an intentional, defensible part of the model (see cost formula in §1.2a).
- Emits success/failure. On failure (`place_and_route` returns infeasible for every candidate, or downtime exceeds a hard SLA bound), mark the SFC's remaining lifetime as terminated early (same accounting as a rejection) — this is the mechanism that keeps migration honest in the ledger.

### 1.2a Sequential processing and the race condition

Do **not** spawn an independent migration-algorithm instance per affected VNF/SFC — that's exactly what creates the race condition (multiple instances reading "PN_7 has 40 free units" before any of them commit, all three picking PN_7, over-allocating it). Instead, since the simulator is a single-threaded event loop, process the priority-ordered VNF list **sequentially, committing each allocation before evaluating the next**:

```
affected = VNF_Selector.rank(vnfs_on_failed_node)     # priority order
for vnf in affected:
    dest = Destination_Selector.select(vnf, current_graph_state)  # state reflects all prior commits this pass
    Migration_Executor.migrate(vnf, dest)              # allocates at dest, updates ledger, before next iteration
```

Because each iteration commits before the next reads state, capacity is correctly decremented in time for the next decision — no locking or transaction machinery needed, just loop ordering. This also means a node failure that took down *k* VNFs is fully resolved within a single migration pass (bounded by *k*, not by the "one migration per timestep" pacing that a soft-trigger design would need — see the runtime note in §3 on why that pacing rule is no longer needed here at all).

**Link failure re-route-first check:** for a link failure affecting multiple SFCs' paths, attempt pure path re-routing first using the **same routing machinery Virne already runs for fresh embeddings** — `controller`'s `link_mapping` / `route` methods (backed by `link_mapper.py`'s k-shortest-path search, the same `shortest_method`/`k_shortest` config your baseline already uses) — with the failed link excluded from the candidate graph and the existing (unmoved) node placement held fixed. This is much cheaper than full migration (no VNF relocation, no restore cost, no policy inference) and should resolve the common case. Only SFCs for which no feasible alternate path exists (all detours violate latency/bandwidth/SLA constraints — `route`'s existing constraint checking already tells you this) fall through to the full VNF-Selector → Destination-Selector → Executor pipeline above — and for those, the same sequential-commit loop applies.

**Cost model, consolidated (failure-only scope):**

```
Cost_migration_i =
    C_restore                                    (fixed, checkpoint-restore cost)
  + downtime_penalty_i(detection+restore time, criticality_i)
  + rerouting_cost_i                             (new path bandwidth allocation)
```

No live-transfer bandwidth term — that only applied to the soft/overload case this revision removes.

**Cost/Revenue Ledger**
- Single source of truth for R2C. Every embedding event and every migration event (success or failed) writes into it. Nothing bypasses this component — this is what makes the baseline-vs-migration comparison valid.

### 1.3 Where this lives in Virne's codebase (verified by direct inspection, not assumption)

| Component | Concrete hook point |
|---|---|
| Event loop | `virne/core/environment.py` — `BaseEnvironment` processes a **sorted event list** (`v_net_simulator.events`, currently arrival/leave only, generated in `virtual_network_request_simulator.py`), one `ready(event_id)` at a time. It is **not** a fixed-timestep `for t in range(T)` loop. Your infra event generator should emit failure/recovery events in the same `{id, v_net_id/p_net_element_id, time, type}` schema and merge them into this sorted list — the dispatch logic mostly just needs a new branch for the new `type` values, not a parallel scanner. |
| Trigger Detector + VNF/SFC Selector | New module (e.g. `virne/migration/`), invoked from the event dispatch when a failure-type event is processed. |
| Destination Selector | Not a new wrapper — calls existing pieces directly: `controller.find_candidate_nodes(...)` (in `rl_core/rl_enviroment_base.py`) for feasibility masking, and the trained solver's `select_action(observation, sample=False)` (in `rl_core/rl_solver.py`) for deterministic masked inference. The policy forward pass itself (`BiGnnBaseModel.forward` in `solver/learning/rl_policy/dual_gnn_policy.py`) is a generic "score every p_net node given this v_net node" function — no placement-specific assumptions to work around. |
| Migration Executor | Orchestrates existing `Controller` methods (`virne/core/controller/controller.py`): `release` (free the dead placement) then `place_and_route` (place + route the new one), per VNF, with `undo_place_and_route` available for rollback on failure. |
| Reroute-first check | Existing routing machinery: `controller`'s `link_mapping`/`route`, backed by `link_mapper.py`'s k-shortest-path search — same mechanism the baseline already uses for fresh embeddings, just re-invoked with the failed link excluded and node placement held fixed. |
| Ledger | `Solution` (`virne/core/solution.py`) already has `v_net_cost`, `v_net_revenue`, `v_net_r2c_ratio` as first-class fields, split into node/link components, feeding `Recorder` (`virne/core/recorder.py`). Extending this to a `v_net_migration_cost` field that folds into the existing cost sum before `v_net_r2c_ratio` is computed keeps you on the *same* code path as the baseline's R2C — which is exactly what the baseline-vs-treatment comparison requires. |

**One structural note worth flagging for your thesis writing:** `JointPRStepEnvironment`/`JointPRStepRLEnv` place **one virtual node per environment step**, not a whole SFC atomically — the policy is called once per VNF, sequentially, within a single SFC's embedding. This is actually a gift for your design: it means "call the destination selector once per migrating VNF, sequentially, committing before the next" (your §1.2a loop) is not a new pattern you're introducing — it's the *same* control-flow shape the baseline already uses for ordinary placement, just re-entered at migration time instead of arrival time.

---

## 2. Training

**Key simplification from the scoping decision:** there is no new training loop for this phase.

- **Embedding policy:** trained exactly as your current baseline reproduction — unchanged. This is the model you already have (or are validating).
- **Migration Trigger / Selector:** rule-based, hand-tuned thresholds (τ_node, τ_link, priority weights w1-w3). "Tuning" here means a small grid/sensitivity sweep on a validation slice of your traffic trace, not gradient training — pick thresholds that trigger migrations at a reasonable rate (neither near-zero nor thrashing every step) and report the sweep in your thesis as justification.
- **Destination Selector:** frozen weights from the embedding policy, used purely at inference. Nothing to train.

This buys you a big methodological simplification: the *only* thing that differs between baseline and treatment is the presence/absence of the migration module — the embedding policy itself is bit-identical in both conditions. That is exactly what you want for a clean ablation.

*(Stretch goal, not required for the thesis core: once this works, fine-tuning the destination selector with a small auxiliary reward, or training a dedicated lightweight migration-target head, is the natural next step — flag it as future work.)*

---

## 3. Runtime (per-timestep control flow)

```
for t in simulation_timeline:
    1. Apply infra events scheduled at t (node/link fail, join, recovery)
    2. Process SFC arrivals at t → embedding path (PPO_Dual_GAT+, unchanged)
    3. Process SFC departures at t → release resources
    4. Migration pass — runs ONLY if step 1 included a node/link failure at t:
         a. Node failure → collect all hosted VNFs, priority-order,
            sequential-commit loop (VNF Selector → Destination Selector → Executor)
         b. Link failure → for each affected SFC, attempt reroute-first;
            for the subset that fails, same sequential-commit loop
    5. Ledger snapshot logged (Revenue_total, Cost_total, R2C, migration stats)
```

Notes:
- Because triggering is failure-only (not continuous threshold scanning), the migration pass is **idle on the vast majority of timesteps** and only does work exactly when an infra event fires — this is simpler and cheaper than the soft-trigger version, and it means you no longer need a "one migration per timestep" pacing rule: a failure that took down *k* VNFs is resolved fully, within the same pass, via the sequential-commit loop (§1.2a) — bounded by *k*, not by an artificial per-step cap.
- Step 4 runs **after** arrivals/departures so the affected-VNF set and destination capacities reflect the current true state, not a stale one.
- Infra Event Generator is new: you'll need a trace generator (e.g., node fails with rate λ_fail, recovers after a random interval; occasional new-node joins; link failures similarly). Keep it seeded and reproducible — you'll reuse the exact same trace across baseline and treatment runs.

---

## 4. Metrics

| Metric | Role |
|---|---|
| **R2C (revenue/cost)** | Primary — includes migration cost in the denominator, as defined earlier |
| Acceptance ratio | Context for R2C, shows admission behavior isn't the thing changing |
| Migration count | Shows how active the mechanism is |
| Reroute count (link-failure fast path) | Shows how often the cheaper reroute-first check resolved a link failure without a full migration |
| Migration cost (C_restore + downtime + rerouting) | Shows the mechanism isn't free — pairs with R2C to show net benefit |
| SLA violations avoided | The causal story: *why* R2C improves (fewer forced early-terminations from unresolved failures) |
| Fallback rate (destination selector) | Robustness diagnostic — how often masked inference had zero feasible actions and fell back to the heuristic |
| Decision latency (Trigger→Executor) | Only if you want to argue the mechanism is online-feasible |

Report R2C **as a time series** (e.g., windowed over the simulation) in addition to a final scalar — this is what will visually demonstrate the mechanism's value under sustained load/failure conditions, which a single endpoint number won't show.

---

## 5. Comparing against the baseline

**Controlled variable:** migration module on vs. off. Everything else held fixed:

1. **Same trained embedding policy weights** in both runs (load once, freeze, reuse — do not retrain per condition).
2. **Same SFC arrival trace** (same seed) for both runs.
3. **Same infra event trace** (same seed) for both runs — this is new; make sure the infra event generator is seeded independently of the SFC trace so you can vary one without the other later.
4. **Same physical topology.**

**Experimental design:**
- Run N ≥ 10–20 seeds per condition (baseline, baseline+migration) to get a distribution, not a point estimate — R2C under stochastic arrivals/failures will have real variance, and your committee will ask about it.
- Report mean ± std (or a boxplot) of final R2C, plus the time-series plot mentioned above.
- Use a paired test (same seeds across conditions → paired t-test or Wilcoxon signed-rank) rather than an unpaired test, since you control the randomness pairing.
- **Stress sweep:** vary the infra-event (failure) rate λ_fail across a few settings — this is now your primary stress dimension since triggering is failure-only. Migration should show negligible/no benefit at low λ_fail (few failures to resolve) and growing benefit as λ_fail increases — that shape of result is much more convincing than a single fixed-condition delta, and it directly demonstrates the mechanism is doing what you claim. Optionally cross with SFC arrival rate as a secondary dimension if time allows.
- **Ablations** (if time permits): (a) destination-selector-uses-policy vs. destination-selector-uses-fallback-only (does reusing the trained policy actually beat "pick the most-free-capacity node"? — this is the ablation that most directly tests your core novelty claim); (b) reroute-first enabled vs. disabled for link failures (does the fast path meaningfully reduce migration cost/count, or is it rarely triggered in practice?).

**What "success" looks like:** migration-enabled R2C ≥ baseline R2C at low stress (no regression from overhead) and significantly higher at moderate/high stress, with migration cost visibly present but not dominating the cost term. If migration cost ever *erases* the R2C gain, that's a real and reportable finding too — it directly speaks to the "realistic cost modeling" gap the survey flagged, and is a legitimate thesis contribution even if less flattering.

---

## 6. Implementation pipeline / workflow

**Phase 0 — Prerequisite**
- [ ] Close out baseline reproduction (per your cutoff) with a working, seeded PPO_Dual_GAT+ run producing stable R2C numbers on your target topology.

**Phase 1 — Infra event generator**
- [ ] Confirmed by inspection: `virtual_network_request_simulator.py` currently generates only arrival/leave events (`type` 1/0) — no failure/recovery event type exists yet, so this is genuinely new, not a config flag to flip.
- [ ] Implement seeded generator for node/link failure and recovery events, in the *same schema* as existing events (`{id, time, type, ...}`), merged into `v_net_simulator.events` sorted by time — extends the existing dispatch rather than replacing it.
- [ ] Unit-test in isolation: verify failed nodes/links are actually excluded from `controller.find_candidate_nodes(...)` (this must already work correctly via zero/unavailable capacity, or the baseline itself is invalid under failures).

**Phase 2 — Ledger extension**
- [ ] Extend Virne's existing metrics recorder to accept migration cost events.
- [ ] Verify: with the migration module fully stubbed out (no-op), R2C numbers match Phase 0 exactly — this is your regression check that you haven't broken anything.

**Phase 3 — Trigger Detector + VNF Selector**
- [ ] Wire Trigger Detector to the infra event stream (node/link failure only — no threshold polling needed).
- [ ] Implement priority scoring and full-set (not top-k) selection for node-failure VNFs.
- [ ] Implement the link-failure reroute-first check (attempt alternate path via existing embedding path-finder before falling through to migration).
- [ ] Test on a synthetic small topology with manually forced node/link failures — verify every affected VNF/SFC is resolved (either rerouted or migrated), none left orphaned.

**Phase 4 — Destination Selector (policy reuse)** — *lower-risk than initially estimated, per source inspection*
- [ ] Construct the `observation` dict (`v_net`, `p_net`, `curr_v_node_id`, `action_mask`) for a migrating VNF, reusing `controller.find_candidate_nodes(...)` for the mask (excluding source + already-committed-this-pass targets).
- [ ] Call the trained solver's existing `select_action(observation, sample=False)` for deterministic inference — confirm it runs correctly with a *migration-time* observation rather than the fresh-arrival observations it was trained on (this is the one real unknown left — the mechanism exists, but has never been exercised on this kind of input, so budget a few days for surprises here specifically, not for building the mechanism itself).
- [ ] Implement and test the most-free-capacity fallback path for the empty-candidate-set case.

**Phase 5 — Migration Executor**
- [ ] Implement resource release/allocate + path re-routing.
- [ ] Implement cost model (bandwidth, downtime) and wire it into the Ledger.
- [ ] Implement failure handling (early termination accounting).

**Phase 6 — End-to-end integration on toy topology**
- [ ] Small topology (10-20 nodes), short trace, both infra events and SFC churn enabled.
- [ ] Sanity-check: migrations happen, ledger numbers move sensibly, no crashes, no resource-accounting leaks (sum of allocated resources across PNs matches expected after many migrations — write this as an explicit invariant check/assertion).

**Phase 7 — Full-scale experiments**
- [ ] Run the seed × condition × stress-level matrix described in Section 5.
- [ ] Collect R2C time series + scalar summaries + migration stats.

**Phase 8 — Analysis & writing**
- [ ] Statistical comparison, plots, ablations if time allows.
- [ ] Write up Sections 3–5 of the thesis (Proposed Method, Experimental Design, Evaluation) using this document as the backbone.

Each phase has a cheap, explicit verification step before moving on — that's deliberate. The single biggest risk in a systems-integration thesis like this is silent bugs in resource accounting (a migration that "succeeds" but leaks capacity, or a ledger that double-counts cost) that don't crash anything but quietly invalidate your R2C numbers. Catching those at Phase 2 and Phase 6 is much cheaper than catching them at Phase 7.