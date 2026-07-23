# Virne Migration Feature — Implementation Plan

> Grounded in the verified architecture of `network`, `solver`, `system`, and `core` (`Counter`/`Recorder`/`Controller`/`Environment`) from the earlier deep-dives, plus direct inspection of `settings/*.yaml`, `virne/utils/dataset.py` (`generate_data_with_distribution`), and `virne/core/logger.py` for this plan specifically.

---

## 0. Scope Recap

Migration in Virne means: an already-embedded VN's placement changes mid-lifetime, without a new arrival or leave event. The trigger you're designing for (§1) is **physical infrastructure failure** — a substrate node or link goes down, and any VN occupying it needs to be relocated (or is dropped if relocation fails). §2 is the measurement layer needed to judge whether a migration policy is any good, wired through the same `Counter → Recorder → Logger` pipeline already used for embedding metrics.

These two pieces are sequenced deliberately: **you can't define migration metrics until failure events exist to trigger migrations**, so §1 is the prerequisite for §2.

---

## 1. Physical Failure Events — Config, Sampling, Integration

### 1.1 Config schema

Virne's existing distribution config pattern (confirmed in `settings/v_sim_setting/v_sim_setting.yaml`) is a flat dict of `{distribution, dtype, ...params}`, consumed by `generate_data_with_distribution(size, distribution, dtype, **kwargs)` (`virne/utils/dataset.py`), which already supports `uniform`, `normal`, `exponential`, `poisson`. **Reuse this exact schema shape** rather than inventing a new one — it's what `arrival_rate` and `lifetime` already use in `v_sim_setting.yaml`, so a `failure_setting` block is a natural sibling, not a new pattern to learn.

Add a new composed config file, mirroring `p_net_setting/` and `v_sim_setting/`:

```yaml
# settings/failure_setting/default.yaml
if_enable: false          # off by default — must not change behavior for existing experiments

node_failure:
  mtbf:                    # mean time between failures, per node
    distribution: exponential
    dtype: float
    scale: 5000
  mttr:                    # mean time to repair (recovery), per failure
    distribution: exponential
    dtype: float
    scale: 200
  affected_fraction:        # fraction of p_net nodes eligible for failure at all
    distribution: uniform
    dtype: float
    low: 0.05
    high: 0.05              # point value; widen for randomized experiments

link_failure:
  mtbf:
    distribution: exponential
    dtype: float
    scale: 8000
  mttr:
    distribution: exponential
    dtype: float
    scale: 150
  affected_fraction:
    distribution: uniform
    dtype: float
    low: 0.05
    high: 0.05

output:
  save_dir: dataset/p_net_failures
  events_file_name: failure_events.yaml
  setting_file_name: failure_setting.yaml
```

Wire it into `settings/main.yaml`'s `defaults` list, same way `v_sim_setting/default` and `p_net_setting/default` are composed:

```yaml
defaults:
  - _self_
  - learning
  - v_sim_setting/default
  - p_net_setting/default
  - failure_setting/default    # new
```

`if_enable: false` as the default is important — it means every existing experiment config, script, and saved run continues to reproduce byte-for-byte unless someone opts in, which matters for anyone (including you) comparing pre/post-migration results later.

### 1.2 A parallel event simulator — don't overload `VirtualNetworkRequestSimulator`

The existing `VirtualNetworkRequestSimulator` (`virne/network/virtual_network_request_simulator.py`) is purpose-built around VN arrival/leave semantics — `arrange_v_nets()`, `_renew_v_nets()`, `_renew_events()` all assume the thing being scheduled is a `VirtualNetwork`. Don't extend it; **add a sibling class** with the same shape:

```
virne/network/
├── virtual_network_request_simulator.py     [existing — VN arrivals/leaves]
└── physical_network_failure_simulator.py    [new]
```

```python
# physical_network_failure_simulator.py
from dataclasses import dataclass

@dataclass
class PhysicalFailureEvent:
    id: int
    type: int          # 1 = fail, 0 = recover  (mirrors VirtualNetworkEvent's 1=arrival/0=leave convention)
    target_type: str   # 'node' | 'link'
    target_id: object  # node id (int) or link id (tuple[int, int])
    time: float

class PhysicalNetworkFailureSimulator:
    def __init__(self, failure_setting: dict, p_net: 'PhysicalNetwork', seed=None):
        self.failure_setting = failure_setting
        self.p_net = p_net
        self.seed = seed
        self.events: list[PhysicalFailureEvent] = []

    @classmethod
    def from_setting(cls, failure_setting, p_net, seed=None):
        return cls(failure_setting, p_net, seed)

    def arrange_failures(self):
        # For each eligible node/link: sample failure times as a renewal process
        # (successive MTBF draws, cumsum), and for each failure sample an MTTR
        # to get the matching recovery time — same generate_data_with_distribution
        # calls already used for v_net arrival_rate/lifetime.
        ...

    def renew(self, seed=None):
        # mirrors VirtualNetworkRequestSimulator.renew(v_nets=, events=, seed=)
        ...

    def save_dataset(self, dataset_dir): ...
    def load_dataset(self, dataset_dir): ...
```

Keeping it separate means: (a) `if_enable: false` short-circuits cleanly — no failure simulator is even constructed; (b) the two event streams can be tested, seeded, and saved/loaded independently, which matters for reproducibility (you'll want to run the same VN arrival sequence against different failure sequences, and vice versa); (c) it doesn't risk destabilizing the VN request pipeline you've already deeply verified.

### 1.3 Merging the two event streams — the real integration point

This is the part with no existing precedent to lean on, so it deserves the most care. Currently `BaseEnvironment.ready(event_id)` (`virne/core/environment.py`) reads a single sorted list: `self.curr_event = self.v_net_simulator.events[event_id]`. With a second independent event source, you need one unified, time-sorted timeline.

**Recommended approach**: build the merge once, in `BaseEnvironment.reset()`, right after both simulators are (re)generated:

```python
def reset(self, seed=None):
    ...
    self.v_net_simulator.renew(v_nets=True, events=True, seed=seed)
    if self.config.failure_setting.if_enable:
        self.p_net_failure_simulator.renew(seed=seed)
        self.merged_events = self._merge_event_streams(
            self.v_net_simulator.events, self.p_net_failure_simulator.events
        )
    else:
        self.merged_events = self.v_net_simulator.events   # unchanged path when disabled
    ...
```

`_merge_event_streams` sorts by `time`, tags each event with a `source` field (`'v_net'` or `'p_net'`) so `ready()` and `transit_obs()` know how to dispatch it, and needs an explicit **tie-break rule** for same-timestamp events (recommend: recoveries before failures before VN arrivals before VN leaves, so a node that fails and is instantly repaired at the same tick doesn't leave a dangling inconsistent state — but pick a rule and document it, since simultaneous events are exactly where subtle bugs hide).

`BaseEnvironment.ready()` and `transit_obs()` then branch on `event['source']`:
- `source == 'v_net'` → existing behavior, unchanged.
- `source == 'p_net', type == 1 (fail)` → new path: mark the node/link unavailable, identify affected VNs, invoke the migration policy.
- `source == 'p_net', type == 0 (recover)` → new path: restore capacity, no VN action needed (a migrated VN doesn't automatically migrate back — that's a separate policy decision, out of scope unless you want "migrate back on recovery" as a feature too).

This is a **change to `BaseEnvironment`**, not a subclass — because `transit_obs()`'s job (advance to the next *actionable* event) needs to understand both event types to decide whether to stop and hand control back to the solver, or process-and-continue (the way it already silently processes leave events today).

### 1.4 Controller: marking resources unavailable

`Controller` currently has no notion of "down" — only consumed/available capacity via `ResourceUpdator`. Add two narrowly-scoped methods to `controller.py` (not a new class — this is squarely within Controller's existing "commit/revert p_net state" responsibility, confirmed in the corrected solver-module interaction diagram):

```python
def fail_node(self, p_net, node_id):
    # zero out available resource (don't delete the node — keep topology intact
    # so a later recover_node can restore it) and set a status flag, e.g.
    # p_net.nodes[node_id]['status'] = 'down'
    ...

def recover_node(self, p_net, node_id):
    # restore resource to pre-failure capacity; status = 'up'
    ...

def fail_link(self, p_net, u, v): ...
def recover_link(self, p_net, u, v): ...
```

The "don't delete, zero-and-flag" approach matters: existing VN mappings reference these nodes/links by id in `solution['node_slots']` / `solution['link_paths']`, and deleting a node would break every downstream `get_record`/`p_net_nodes_for_v_net_dict` lookup already documented in the Counter/Recorder trace.

### 1.5 Identifying affected VNs — reuse, don't rebuild

You already have the exact index needed: `Recorder.p_net_nodes_for_v_net_dict` (`{p_node_id: [v_net_id, ...]}`), flagged in the earlier Counter/Recorder review as "the closest thing Virne has to a live occupancy map." On a node-failure event:

```python
affected_v_net_ids = self.recorder.p_net_nodes_for_v_net_dict.get(failed_node_id, [])
```

For link failures you'd need an equivalent index that doesn't currently exist — **`p_net_links_for_v_net_dict`**, maintained the same way (append on successful placement's `link_paths`, remove on leave), which is a small, direct extension of the existing `count_state` bookkeeping in `Recorder`, not a new subsystem.

### 1.6 The migration decision itself — reuse the `Solver` interface

Once affected VNs are known, deciding *where* to re-place them is structurally identical to fresh embedding — same `solve(instance) -> Solution` contract from the `solver` module. Recommended shape: a `MigrationSolver` (or a `migrate()` method bolted onto existing solvers via a mixin, following the exact precedent of `SafeRLSolver` mixed into `PPOSolver`) that takes `(v_net, p_net, current_solution)` instead of just `(v_net, p_net)`, and either:
- re-solves from scratch, ignoring the current (now partially invalid) placement, or
- solves incrementally, keeping whichever virtual nodes/links are still validly placed and only re-placing the ones that were on the failed component.

The second is cheaper and more realistic (a real migration doesn't re-place a whole VN just because one node failed) but requires `Solution` to support a "partial re-solve" entry point — none of the existing `InstanceEnv` variants (`PlaceStepInstanceRLEnv`, `JointPRStepInstanceRLEnv`, etc.) currently support resuming from a partially-filled `solution['node_slots']`, so this is new ground, not a reuse of an existing code path. Flagging this now because it's the single largest actual implementation gap in the whole feature — worth prototyping the "incremental re-solve" `InstanceEnv` variant early, since everything else in this plan can be built and tested without it (falling back to full re-solve as an MVP).

---

## 2. Migration Evaluation Metrics — Counter, Recorder, Logger

### 2.1 What to measure, and why each metric matters

| Metric | What it captures | Why it matters for judging a migration policy |
|---|---|---|
| `migration_count` | Total migrations triggered | Baseline volume — is the policy migrating too eagerly or too rarely? |
| `migration_success_rate` | Fraction of triggered migrations that found a feasible new placement | The core "did it work" number |
| `forced_termination_count` | VNs dropped because no migration was possible | The cost of migration *failure* — should feed into acceptance-rate-style accounting, since a forced termination is effectively a mid-lifetime rejection |
| `avg_migration_cost` | Resource cost of the re-placement itself (via `Counter`, same shape as embedding cost) | Migrations aren't free — moving a VN consumes the same node/link resources a fresh embedding would |
| `service_disruption_time` | Time between failure detection and successful re-placement | The metric a real operator actually cares about — embedding cost alone doesn't capture downtime |
| `migration_overhead_ratio` | `avg_migration_cost / v_net_cost` (original embedding cost) | Normalizes migration cost the same way `v_net_r2c_ratio` normalizes embedding cost — lets you compare across VN sizes |
| `failure_survival_rate` | Fraction of failure-affected VNs that survived (successfully migrated) vs. were dropped | The headline number for comparing migration policies against a no-migration baseline |

### 2.2 Where each piece lives — following the existing division of labor

This maps directly onto the `Counter` (stateless calculator) / `Recorder` (stateful ledger) / `Logger` (surfacing) split already verified:

**`Solution` (`core/solution.py`) — new fields**, alongside the existing `v_net_cost`/`v_net_revenue` fields:
```python
self.is_migration: bool = False
self.migration_trigger: str = ''          # 'node_failure' | 'link_failure' | ''
self.migration_cost: float = 0.0
self.migration_result: bool = True         # False = forced termination
self.pre_migration_node_slots: OrderedDict = OrderedDict()   # for diffing old vs new placement
```

**`Counter` — a new `count_migration` method**, structurally parallel to `count_solution` (§2 of the Counter/Recorder deep-dive already covers `count_solution`'s revenue/cost pattern in detail — reuse it, don't reinvent):
```python
def count_migration(self, v_net, old_solution, new_solution) -> dict:
    # diff old_solution['node_slots'] vs new_solution['node_slots'] to find what
    # actually moved (not the whole VN — just the delta), sum resource cost of
    # the delta using self.node_resource_attrs / self.link_resource_attrs
    # (already available as instance attributes), same normalization pattern
    # used in count_solution (divide by num_node_resource_attrs, etc.)
    new_solution['migration_cost'] = ...
    new_solution['v_net_migration_overhead_ratio'] = (
        new_solution['migration_cost'] / new_solution['v_net_cost'] if new_solution['v_net_cost'] else 0
    )
    return new_solution.to_dict()
```

Extend `summary_records` (already computes `acceptance_rate`, `long_term_r2c_ratio`, etc. from the full `records` DataFrame) with the run-level rollups:
```python
summary_info['migration_count'] = (records['event_type'] == 2).sum()   # see §2.3 on event_type
summary_info['migration_success_rate'] = records.loc[records['event_type']==2, 'migration_result'].mean()
summary_info['avg_migration_cost'] = records.loc[records['event_type']==2, 'migration_cost'].mean()
summary_info['forced_termination_count'] = ((records['event_type']==2) & (records['migration_result']==False)).sum()
summary_info['failure_survival_rate'] = 1 - (summary_info['forced_termination_count'] / summary_info['migration_count']) if summary_info['migration_count'] else None
```

**`Recorder` — new `state` fields and a third branch in `count_state`.** This is the point flagged repeatedly across the earlier reviews: `count_state`'s branching is currently a closed `{0: leave, 1: enter}` switch, and `v_net_event_dict` assumes one arrival per VN. A migration event needs to be **event_type == 2**, added as a new branch:

```python
# Recorder.reset() — add to self.state
'migration_count': 0,
'total_migration_cost': 0,
'forced_termination_count': 0,

# Recorder.count_state() — new branch
elif self.state['event_type'] == 2:  # migration
    self.state['migration_count'] += 1
    self.state['total_migration_cost'] += solution['migration_cost']
    if not solution['migration_result']:
        self.state['forced_termination_count'] += 1
        # treat like a leave event for occupancy bookkeeping — the VN is gone
    else:
        # update p_net_nodes_for_v_net_dict: remove old node_slots entries,
        # add new ones — do NOT touch v_net_event_dict, since a future leave
        # event must still resolve back to the ORIGINAL arrival record
        ...
```

That last comment is important enough to restate: **`v_net_event_dict[v_net_id]` must keep pointing at the arrival event**, not get overwritten by the migration event, because the leave-event branch of `count_state` depends on `get_record(v_net_id=...)['result']` resolving to the *original* placement outcome. This was flagged as a risk in the Counter/Recorder deep-dive before migration was even designed — now it's the concrete rule to enforce.

**`Logger` — surface metrics as they happen, not just at run end.** `Logger.log(message, level, data, step)` already supports structured `data` dicts to file/tensorboard/wandb backends (confirmed — same mechanism `InstanceAgent.learn_singly` uses for training metrics). Add a call at the point a migration is resolved, inside whatever `BaseEnvironment` method handles the new `event_type == 2` path:

```python
self.logger.log(
    data={
        'migration/success': int(solution['migration_result']),
        'migration/cost': solution['migration_cost'],
        'migration/overhead_ratio': solution['v_net_migration_overhead_ratio'],
        'migration/trigger': solution['migration_trigger'],
    },
    step=self.recorder.state['migration_count'],
)
```

And extend the live progress bar (`BaseSystem.update_process_bar`, which already shows `ac` / `r2c` / `inservice` during a run) with a `mig` field, so migration behavior is visible during `system.run()` itself, not only in the post-run summary CSV.

### 2.3 Why `event_type == 2`, not a separate flag

You could instead keep migrations as a boolean flag on top of the existing enter/leave events rather than a third `event_type`. Don't — here's why the third-type approach is more consistent with what's already there:

- `VirtualNetworkEvent.type` (network module) and `Recorder.state['event_type']` are already the single dispatch key that both `count()` (Recorder) and `count_state()` branch on. Bolting migration onto `type == 1` (enter) would conflate "this is a fresh placement" with "this is a re-placement," corrupting `success_count`/`v_net_count` accounting that assumes every `type == 1` event is a genuinely new arrival.
- A dedicated `type == 2` keeps `summary_records`' existing formulas (`acceptance_rate = success_count / v_net_count`, etc.) correct **without modification**, since migrations simply don't touch `v_net_count`/`success_count` at all — they only touch the new migration-specific counters. This is the cleanest way to add a whole new capability without risking a silent regression in every metric you already validated.

---

## 3. Suggested Build Order

Given the dependency chain above, this is the order that lets you test each piece in isolation before the next depends on it:

1. **Config + `PhysicalNetworkFailureSimulator`** (§1.1–1.2), tested standalone — generate and inspect a failure/recovery event stream against a `PhysicalNetwork`, no environment integration yet.
2. **Event stream merge + `Controller.fail_node/recover_node`** (§1.3–1.4) — verify a failure correctly zeroes capacity and a recovery correctly restores it, using the existing resource-conservation assertion pattern from `SolutionStepEnvironment.step` as a template for a new "failure conserves nothing, recovery restores everything" test.
3. **Affected-VN lookup** (§1.5) — extend `Recorder` with `p_net_links_for_v_net_dict`; verify against `p_net_nodes_for_v_net_dict`'s existing tests/usage.
4. **`Solution` fields + `Counter.count_migration` + `Recorder` `event_type == 2` branch** (§2.2) — build the accounting before the actual migration solver exists, using a stub that just marks every migration `migration_result=True` with cost 0. This lets you validate the whole metrics pipeline (Recorder → summary_records → Logger) independent of solver quality.
5. **Full re-solve migration path** (§1.6, MVP) — wire a real solver in, initially just doing a complete re-embedding of the affected VN, ignoring its old placement.
6. **Incremental re-solve** (§1.6, stretch) — the one genuinely novel piece of solver-module work; revisit the `InstanceEnv` family once the MVP is validated end-to-end.