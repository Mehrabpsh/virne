# `Counter` and `Recorder` in Virne — Architecture & Control Flow

> Verified against `GeminiLight/virne` @ `main`: `virne/core/counter.py`, `virne/core/recorder.py`, `virne/core/environment.py`, `virne/core/solution.py`, `virne/system/base_system.py`, `virne/solver/learning/rl_core/online_rl_environment.py`.

---

## 1. One-Line Distinction

| | `Counter` | `Recorder` |
|---|---|---|
| Nature | **Stateless calculator** | **Stateful ledger** |
| Holds across calls | Only config-derived attribute lists (fixed at construction) | Cumulative simulation state, full history (`memory`), lookup indices |
| Input | `(v_net, solution)` or `(network)` | `(v_net, p_net, solution)` + event metadata |
| Output | A dict of computed numeric fields (revenue, cost, r2c ratio, resource sums) | A persisted, timestamped record row + updated running totals |
| Called by | `Recorder` (owns a `Counter` instance) **and** `BaseEnvironment`/`SolutionStepEnvironment` directly | `BaseEnvironment` (via `count_and_add_record` / `release`) |

The key relationship: **`Recorder` owns a `Counter`** (`self.counter = counter` in `Recorder.__init__`), and delegates all revenue/cost *arithmetic* to it, while `Recorder` itself owns *bookkeeping* — the running totals, the CSV-backed history, and the id-lookup tables.

They are constructed once, in `BaseSystem.from_config`, and the **same `Counter` instance** is shared by the environment and the recorder:

```python
# BaseSystem.from_config
counter = Counter(node_attrs_setting, link_attrs_setting, graph_attrs_setting, config)
controller = Controller(node_attrs_setting, link_attrs_setting, graph_attrs_setting, config)
recorder = Recorder(counter, config)          # <- Recorder takes the Counter by reference
...
env = SolutionStepEnvironment(p_net, v_net_simulator, controller, recorder, counter, logger, config)
                                                # <- env also holds the same counter directly
```

Notably, `Controller` does **not** receive the `Counter` (grep confirms only commented-out `# self.counter.count_solution(...)` lines in `controller.py`) — constraint checking during embedding is a separate concern (`ConstraintChecker`), and `Counter` is purely a *post-hoc accounting* utility, never consulted to decide whether a placement is feasible.

---

## 2. `Counter` — Architecture

```
Counter
│
├── construction-time state (fixed, from config)
│     ├── all_node_attrs           (list of NodeAttribute objects)
│     ├── all_link_attrs           (list of LinkAttribute objects)
│     ├── node_resource_attrs      (filtered: type == 'resource')
│     ├── link_resource_attrs      (filtered: type == 'resource')
│     ├── num_node_resource_attrs  (len, used as a normalizer divisor)
│     └── num_link_resource_attrs  (len)
│
├── per-solution accounting  (mutates & returns a Solution dict)
│     ├── count_solution(v_net, solution)
│     └── count_partial_solution(v_net, solution)
│
├── raw resource aggregation  (pure functions over a network)
│     ├── calculate_sum_network_resource(network, node=True, link=True)
│     ├── calculate_sum_node_resource(network)
│     ├── calculate_sum_link_resource(network)
│     ├── calculate_v_net_cost(v_net, solution)
│     ├── calculate_v_net_revenue(v_net, solution=None)
│     └── calculate_v_net_link_cost(v_net, solution)
│
└── run-level summarization  (operates on the full record history)
      ├── summary_records(records: list | DataFrame)
      └── summary_csv(fpath)          [classmethod, reads a saved CSV]
```

`Counter` never touches `self` state beyond the attribute lists set in `__init__` — every method is effectively a pure function of its arguments. This is why the *same* `Counter` object can safely be called from both `Recorder` and `BaseEnvironment` without any risk of state leakage between VN requests.

### 2.1 `count_solution` — the core accounting function

Called once per VN **arrival** event, after the solver has returned a (complete) `Solution`. It mutates the `Solution` object in place and returns `solution.to_dict()`:

```
solution['num_placed_nodes']   = len(node_slots)
solution['num_routed_links']   = len(link_paths)
solution['v_net_node_demand']  = calculate_sum_node_resource(v_net) / num_node_resource_attrs   # normalized
solution['v_net_link_demand']  = calculate_sum_link_resource(v_net)

if solution['result']:                       # ── SUCCESS branch ──
    place_result = route_result = True; early_rejection = False
    v_net_node_revenue = v_net_node_demand
    v_net_link_revenue = v_net_link_demand
    v_net_node_cost    = v_net_node_revenue
    v_net_link_cost    = calculate_v_net_link_cost(v_net, solution)   # sums actual path resource usage
    v_net_path_cost     = v_net_link_cost - v_net_link_revenue         # "extra" cost from multi-hop routing
    v_net_revenue        = v_net_node_revenue + v_net_link_revenue
    v_net_cost            = v_net_revenue + v_net_path_cost
    v_net_r2c_ratio         = v_net_revenue / v_net_cost   (0 if cost == 0)
else:                                        # ── FAILURE branch ──
    v_net_node_revenue = v_net_link_revenue = v_net_revenue = 0
    v_net_path_cost = v_net_cost = v_net_r2c_ratio = 0

# time-weighted variants, regardless of branch:
v_net_time_revenue  = v_net_revenue  * v_net.lifetime
v_net_time_cost     = v_net_cost     * v_net.lifetime
v_net_time_rc_ratio = v_net_r2c_ratio * v_net.lifetime
```

Key modeling idea: **revenue = resource demand actually satisfied**; **cost = resource actually consumed on the substrate**, which for links can exceed revenue because a virtual link's bandwidth demand might be routed over a multi-hop physical path (`v_net_path_cost` captures that overhead). `v_net_r2c_ratio` (revenue-to-cost) is the central efficiency metric used throughout the framework (feeds `long_term_r2c_ratio` in `Recorder`, and RL reward shaping elsewhere).

### 2.2 `count_partial_solution` — same idea, mid-embedding

Used by the **joint-step** RL environment (`JointPRStepInstanceRLEnv`, via `RewardCalculator`/`recorder.counter.count_partial_solution` in `online_rl_environment.py`) to compute revenue/cost *incrementally* as nodes get placed one at a time, rather than waiting for a fully completed `Solution`. It's structurally similar to `count_solution` but reads directly off whatever `solution['node_slots']` / `solution['link_paths']` currently contain, rather than branching on `solution['result']`.

> **Observed quirk, not confirmed as consequential:** `count_solution` contains `solution['v_net_demand'] = solution['v_net_node_demand'] + solution['v_net_demand']` — this reads `v_net_demand` on the right-hand side before overwriting it, rather than adding `v_net_link_demand` (which is what every neighboring line's naming pattern would suggest). Since `Solution.__init__` defaults `v_net_demand` to `0.0`, a *single* call behaves as intended; if `count_solution` is ever invoked twice on the same `Solution` object, this line would silently accumulate incorrectly. Worth a quick look if you ever see `v_net_demand` figures drift from `node_demand + link_demand`.

### 2.3 Resource-sum helpers

These three just delegate to `BaseNetwork.get_node_attrs_data` / `get_link_attrs_data` (from the `network` module you already reviewed), filtered to `type == 'resource'` attributes, and `.sum()` over a NumPy array:

```python
calculate_sum_node_resource(network) = Σ over all nodes, all resource attrs
calculate_sum_link_resource(network) = Σ over all links, all resource attrs
calculate_sum_network_resource(network, node=True, link=True) = node_sum + link_sum  (either can be excluded)
```

These are what `BaseEnvironment`/`Recorder` use to track substrate-wide **available resource**, and — importantly — they're called on the *physical* network both before and after a deployment/rollback to sanity-check resource conservation (see §4.2).

### 2.4 `summary_records` — turning a run into headline metrics

Takes the full `Recorder.memory` list (or a loaded CSV as a DataFrame) and computes run-level KPIs: `acceptance_rate`, `avg_r2c_ratio`, `long_term_r2c_ratio`, `long_term_time_r2c_ratio`, `total_revenue`/`total_cost`, failure breakdowns (`early_rejection_count`, `place_failure_count`, `route_failure_count`), substrate-utilization minima (`min_p_net_available_resource`, etc.), constraint-violation totals, and — if present — mean RL reward. This is what ultimately lands in `summary.csv` / `solver_summary.csv` / `global_summary.csv`.

`summary_csv` is a `@classmethod` convenience wrapper: load a saved CSV, then call the (otherwise instance) `summary_records` — this works because `summary_records` doesn't actually touch `self` beyond nothing at all; it's usable as if static.

---

## 3. `Recorder` — Architecture

```
Recorder
│
├── construction-time (from config + injected Counter)
│     ├── counter                    (shared Counter instance)
│     ├── save_root_dir / summary_dir / record_dir   (path layout)
│     └── if_temp_save_records       (whether to stream rows to a temp CSV as you go)
│
├── per-run mutable state  (rebuilt on reset())
│     ├── curr_record : dict                    — the row being assembled right now
│     ├── memory : list[dict]                    — full ordered history, one dict per event
│     ├── v_net_event_dict : {v_net_id: event_id}         — reverse lookup
│     ├── p_net_nodes_for_v_net_dict : {p_node_id: [v_net_id, ...]}  — which VNs occupy which substrate node
│     └── state : dict                          — running totals (see §3.1)
│
├── event-driven counting
│     ├── count_state(v_net, p_net, solution)     — updates `state` based on event_type (0=leave, 1=enter)
│     └── count(v_net, p_net, solution)           — orchestrates counter.count_solution + count_state, returns a merged record
│
├── persistence
│     ├── add_record / add_info                   — append to `memory`, optionally stream to temp CSV
│     ├── temp_save_record                         — incremental CSV append (crash-safety / long-run visibility)
│     ├── save_records(fname)                       — dump `memory` to a final CSV, delete temp file
│     └── save_summary(summary_info, fname)          — write summary rows to multiple CSVs (per-run, per-solver, global)
│
└── lookup / introspection
      ├── get_record(event_id=None, v_net_id=None)
      ├── get_running_p_net_nodes()
      └── display_record(record, display_items, extra_items)
```

### 3.1 `state` — the running totals dict

Initialized in `reset()`:

```python
self.state = {
    'v_net_count': 0, 'success_count': 0, 'inservice_count': 0,
    'total_revenue': 0, 'total_cost': 0,
    'total_time_revenue': 0, 'total_time_cost': 0,
    'long_term_r2c_ratio': 0, 'long_term_time_r2c_ratio': 0,
    'num_running_p_net_nodes': 0,
}
```

Then updated by `update_state(info_dict)` (called by `BaseEnvironment.ready()` each event, injecting `event_id`/`event_type`/`event_time`) and by `count_state(...)` (see below).

### 3.2 `count_state` — the event-type branch

This is where **arrival vs. departure** semantics live, keyed off `self.state['event_type']` (`0` = leave, `1` = enter — matching the closed 2-state `VirtualNetworkEvent.type` you already found in the `network` module):

**Leave event (`type == 0`):**
```python
if the original arrival's solution was successful:
    inservice_count -= 1
    for each (v_node_id, p_node_id) in that solution['node_slots']:
        remove v_net_id from p_net_nodes_for_v_net_dict[p_node_id]
    num_running_p_net_nodes = len(get_running_p_net_nodes())
```
Note it looks up the **original arrival record** via `self.get_record(v_net_id=solution['v_net_id'])['result']` — i.e., a leave event's bookkeeping depends on what happened at that VN's *arrival*, retrieved through `v_net_event_dict`.

**Enter event (`type == 1`):**
```python
v_net_event_dict[v_net_id] = event_id     # register so a future leave event can look this up
v_net_count += 1
if solution['result']:
    success_count += 1; inservice_count += 1
    total_revenue += v_net_revenue;   total_cost += v_net_cost
    total_time_revenue += v_net_time_revenue;  total_time_cost += v_net_time_cost
    long_term_r2c_ratio = total_revenue / total_cost
    long_term_time_r2c_ratio = total_time_revenue / total_time_cost   # asserted <= 1
    for each (v_node_id, p_node_id) in solution['node_slots']:
        p_net_nodes_for_v_net_dict[p_node_id].append(v_net_id)
    num_running_p_net_nodes = len(get_running_p_net_nodes())
```

Both branches also unconditionally refresh substrate-utilization fields at the top of `count_state` (not shown above): `p_net_available_resource`, `p_net_node_available_resource`, `p_net_link_available_resource` (all via `self.counter.calculate_sum_network_resource(...)`), and derived utilization ratios against the cached `init_p_net_info` baseline (set once via `count_init_p_net_info`, called from `BaseEnvironment.reset()`).

### 3.3 `count` — the orchestrator

```python
def count(self, v_net, p_net, solution):
    if event_type == 0:   # leave
        solution_info = solution.to_dict()                       # no revenue/cost recompute
    elif event_type == 1: # enter
        solution_info = self.counter.count_solution(v_net, solution)   # delegates arithmetic to Counter
    self.count_state(v_net, p_net, solution)                      # always update running totals
    return {**self.state, **solution_info}                        # merged flat record
```

This is the single call site where `Counter` arithmetic and `Recorder` bookkeeping actually meet — and it clarifies an important asymmetry: **leave events do not recompute revenue/cost** (there's nothing new to compute — the VN is just vacating resources), they only trigger `count_state`'s leave-branch bookkeeping.

### 3.4 `p_net_nodes_for_v_net_dict` — substrate occupancy index

A `defaultdict(list)` mapping **physical node id → list of VN ids currently placed there**. This is maintained incrementally (append on successful arrival, remove on leave) rather than recomputed, and is exactly the kind of index you'd want to *extend* for a migration feature — e.g., to quickly answer "which VNs are affected if physical node X needs to be drained/evacuated," which is precisely the trigger condition for a proactive migration policy.

---

## 4. Control Flow

### 4.1 Where `Counter`/`Recorder` sit in the event loop

```
OnlineSystem.run
  └─ env.step(solution)                                  [SolutionStepEnvironment.step]
       ├─ constraint / admission checks
       ├─ counter.count_solution(...)                      [DIRECT — used only for a resource-conservation assert,
       │                                                     not merged into the record]
       ├─ if success: controller.deploy(v_net, p_net, solution)
       │  else:        rollback_for_failure(reason)
       ├─ record = count_and_add_record()                  [BaseEnvironment]
       │    ├─ record = self.recorder.count(v_net, p_net, solution)     [Recorder.count]
       │    │    ├─ if enter: solution_info = counter.count_solution(v_net, solution)   [Counter]
       │    │    ├─ (if leave: solution_info = solution.to_dict())
       │    │    ├─ count_state(v_net, p_net, solution)                 [Recorder — running totals]
       │    │    └─ return {**state, **solution_info}
       │    └─ record = self.add_record(record, extra_info)
       │         └─ recorder.add_record(record, extra_info)             [Recorder]
       │              ├─ memory.append(deepcopy(curr_record))
       │              └─ if if_temp_save_records: temp_save_record(...)  [streaming CSV append]
       └─ done = transit_obs()                              [advances event pointer; leave events call release()]
```

### 4.2 Detail: why `counter.count_solution` appears *twice* conceptually

`SolutionStepEnvironment.step` calls `self.counter.calculate_sum_network_resource(...)` (not `count_solution`) directly, purely to assert that total physical-network resource is conserved across a deploy: it snapshots the sum before deploying (`total_p_resource_2`), deploys, snapshots after (`total_p_resource_1`), and asserts the delta matches `solution['v_net_node_cost']` / `solution['v_net_link_cost']` — which were themselves computed **inside** `recorder.count(...) → counter.count_solution(...)` earlier in the same `step()` call, via `self.counter.count_solution` being invoked by `Recorder`. So `Counter` participates in the request twice: once (via `Recorder`) to *compute* the solution's cost figures, and once (directly, via raw resource sums) to *verify* those figures against the actual physical-network state change. This is essentially a runtime consistency check baked into the environment.

### 4.3 Detail: leave events (`release()`)

```
BaseEnvironment.transit_obs()
  └─ loop: ready(next_event_id)
       if curr_event['type'] == 0 (leave):
           record = self.release()
             ├─ solution = recorder.get_record(v_net_id=self.v_net.id)     [Recorder — pull back the ORIGINAL
             │                                                                arrival record via v_net_event_dict]
             ├─ controller.release(v_net, p_net, solution)                  [Controller — actually frees p_net resources]
             ├─ solution['description'] = 'Leave Event'
             └─ record = count_and_add_record()                             [same path as §4.1 — event_type is now 0]
       else (enter): return False   # stop transit, hand control back to the solver
```

This is the mechanism by which a VN's **departure** is entirely driven by the recorder's memory of its **arrival** — `Recorder.get_record(v_net_id=...)` is the join key. Note that `release()` re-derives the `Solution` object from the recorder rather than the environment holding a live reference — meaning the "solution" being released is whatever was last recorded for that `v_net_id`, not a fresh object.

### 4.4 End-to-end sequence diagram (single VN lifecycle: arrival → success → later departure)

```
 Solver            SolutionStepEnv          Controller        Recorder              Counter          p_net_nodes_for_v_net_dict
   │                     │                       │                │                     │                        │
   │  solve(instance)    │                       │                │                     │                        │
   │────────────────────>│                       │                │                     │                        │
   │   Solution           │                       │                │                     │                        │
   │<────────────────────│                       │                │                     │                        │
   │                     │  step(solution)        │                │                     │                        │
   │                     │───┐ constraint checks   │                │                     │                        │
   │                     │<──┘                    │                │                     │                        │
   │                     │  calculate_sum_..(before)                                     │                        │
   │                     │───────────────────────────────────────────────────────────────>│                        │
   │                     │  deploy(v_net,p_net,solution)                                   │                        │
   │                     │───────────────────────>│                │                     │                        │
   │                     │  calculate_sum_..(after)                                        │                        │
   │                     │───────────────────────────────────────────────────────────────>│                        │
   │                     │  count_and_add_record()                                         │                        │
   │                     │────────────────────────────────────────>│                     │                        │
   │                     │                       │                │ count(v_net,p_net,solution)                    │
   │                     │                       │                │  count_solution(v_net, solution) ─────────────>│
   │                     │                       │                │  <── revenue/cost/r2c dict ─────────────────────│
   │                     │                       │                │  count_state(...)   [event_type==1: enter]      │
   │                     │                       │                │───update p_net_nodes_for_v_net_dict[p_node]────>│  (append v_net_id)
   │                     │                       │                │  v_net_event_dict[v_net_id] = event_id           │
   │                     │                       │                │  add_record → memory.append(...) + temp CSV     │
   │                     │  <── record ───────────────────────────│                     │                        │
   │                     │  transit_obs() → done=False (enter consumed; loop resumes)     │                        │
   │                     │                       │                │                     │                        │
   │  ...  (time passes; simulator's pre-scheduled LEAVE event for this v_net comes up) ...                        │
   │                     │                       │                │                     │                        │
   │                     │  transit_obs() encounters type==0 event                        │                        │
   │                     │───┐ release()          │                │                     │                        │
   │                     │   │  get_record(v_net_id) ──────────────>│                     │                        │
   │                     │   │  <── original arrival Solution ──────│                     │                        │
   │                     │   │  controller.release(v_net,p_net,solution)                  │                        │
   │                     │   │─────────────────────>│                │                     │                        │
   │                     │   │  count_and_add_record() → recorder.count(...) [event_type==0: leave]                 │
   │                     │   │                       │                │  no counter.count_solution (skipped)         │
   │                     │   │                       │                │  count_state: inservice_count -= 1           │
   │                     │   │                       │                │───remove v_net_id from p_net_nodes_dict─────>│
   │                     │<──┘                       │                │                     │                        │
```

---

## 5. Design Notes Relevant to a Migration Feature

Since you're planning a migration module, a few things from this trace are directly load-bearing:

1. **`p_net_nodes_for_v_net_dict` is the closest thing Virne has to a live occupancy map.** It's currently only mutated on enter/leave (append/remove), and only tracks *node* placement, not link paths. A migration operation changes a VN's node/link mapping *without* an enter or leave event — so this dict (and its `link_paths` analogue, which doesn't currently exist) would need a third mutation path: "re-place without changing `v_net_count`/`success_count`/`inservice_count`."
2. **`v_net_event_dict` is a single `{v_net_id: event_id}` mapping**, implicitly assuming exactly one arrival event per VN. A migration event would need either to *not* touch this dict (if migration doesn't get its own event type) or to be handled carefully so a later leave event still resolves to the *original* arrival's `event_id` for `get_record` lookups — since `count_state`'s leave branch depends on that original record's `['result']` and `['node_slots']`.
3. **`Counter.count_solution` computes cost/revenue once, treating the whole VN as atomic.** A migration that moves only *some* of a VN's virtual nodes would need something closer to `count_partial_solution`'s incremental style, or a new `count_migration(...)` method that diffs old vs. new `node_slots`/`link_paths` to compute a *migration cost* (distinct from ongoing deployment cost) — there's no such concept anywhere in `Counter` today.
4. **Resource-conservation assertions in `SolutionStepEnvironment.step`** (§4.2) are a useful safety net you'll likely want to replicate for a migration step: snapshot total substrate resource before/after a re-placement and assert nothing was lost or double-counted, exactly the same pattern already used for fresh deployments.