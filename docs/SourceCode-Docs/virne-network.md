# The `network` Module in Virne

## Overview

The **`network`** module is the data model of the entire Virne framework. It is responsible for representing every graph, resource, and request that exists during the simulation.

Unlike the `system` module, which controls the execution flow, or the `solver` module, which implements embedding algorithms, the `network` module **contains almost no embedding logic**. Instead, it provides a unified representation of physical and virtual networks together with their attributes.

Every solver, controller, recorder, and environment component eventually interacts with one or more objects defined in this module.

---

## Responsibilities

The `network` module is responsible for:

- Representing physical networks (substrate networks)
- Representing virtual network requests (VNs, tagged with temporal attributes)
- Managing graph topologies
- Managing node, link, and graph attributes
- Generating synthetic network topologies
- Generating virtual network requests
- Computing topological metrics

It **does not** decide where virtual nodes are embedded or how resources are allocated. Those responsibilities belong primarily to the **Controller** and **Solver** modules.

---

## Overall Architecture

```
                           networkx.Graph
                                  │
                                  ▼
                         ┌────────────────┐
                         │  BaseNetwork   │
                         └────────────────┘
                                  │
               ┌──────────────────┴──────────────────┐
               │                                      │
               ▼                                      ▼
      PhysicalNetwork                          VirtualNetwork
                                          (id, arrival_time, lifetime
                                           injected via graph_attrs_setting
                                           by the request simulator —
                                           no separate "Request" subclass)
```

There is **no `VirtualNetworkRequest` class**. A "virtual network request" in Virne is simply a `VirtualNetwork` instance whose graph attributes have been populated with `id`, `arrival_time`, and `lifetime` at construction time. These three fields are declared as class-level type annotations on `VirtualNetwork` itself:

```python
class VirtualNetwork(BaseNetwork):
    id: int
    arrival_time: float
    lifetime: float
```

The module also contains helper classes:

```
                 BaseNetwork
                 ┌──────────┐
                 │          │
                 │Topology  │────────► TopologyGenerator
                 │Attributes│────────► Attribute Objects (Node/Link/Graph)
                 │Metrics   │────────► TopologicalMetricCalculator
                 └──────────┘
```

---

## `BaseNetwork`

`BaseNetwork` is the fundamental abstraction of the module.

It inherits directly from **NetworkX's `Graph`**, meaning every network in Virne is also a fully functional NetworkX graph.

```
networkx.Graph
        │
        ▼
BaseNetwork
```

Consequently, all standard NetworkX operations are immediately available:

```python
network.nodes
network.edges
network.neighbors(node)
nx.shortest_path(network)
```

Besides graph functionality, `BaseNetwork` manages:

- graph topology
- attribute objects
- topology generation
- attribute generation
- topology metrics (via a companion calculator, not a self-owned method)

---

## Main Components of BaseNetwork

```
BaseNetwork
│
├── node_attrs   (dict[str, NodeAttribute])
├── link_attrs   (dict[str, LinkAttribute])
├── graph        (native nx dict — holds graph_attrs + settings)
│
├── uses → TopologyGenerator        (topology construction)
└── uses → TopologicalMetricCalculator  (external, called on demand)
```

`node_attrs` / `link_attrs` hold **attribute objects** (schema + behavior), not the values themselves. The actual per-node/per-link values live in standard NetworkX storage: `G.nodes[n][attr_name]` and `G.edges[u, v][attr_name]`.

---

## Attribute System

Virne separates attributes into **three independent categories**, by *owner*:

```
BaseAttribute (abstract)
│
├── NodeAttribute
├── LinkAttribute
└── GraphAttribute
```

### 1. Node Attributes

Describe properties of individual nodes (CPU, memory, GPU, position, node type). Stored in `G.nodes[node_id]`.

```python
{
    "cpu": 80,
    "gpu": 4,
    "pos": (10, 20)
}
```

### 2. Link Attributes

Describe properties of individual edges (bandwidth, delay, distance). Stored in `G.edges[u, v]`.

```python
{
    "bw": 100,
    "delay": 3
}
```

### 3. Graph Attributes

Describe the network as a whole (topology type, arrival rate, seed, id/arrival_time/lifetime for VNs). Stored in `G.graph`.

```python
{
    "id": 12,
    "arrival_time": 104.0,
    "lifetime": 38.5
}
```

---

## Attribute Hierarchy — Mixin Composition, Not a Class-per-Metric Tree

There is **no dedicated `CPUAttribute`, `MemoryAttribute`, or `BandwidthAttribute` class**. "cpu" and "bandwidth" are just the `name` field of a generic attribute object, entirely config-driven. The real hierarchy composes **owner classes** with **behavior mixins**:

```
                          BaseAttribute (abc)
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      NodeAttribute        LinkAttribute       GraphAttribute
              │                  │                  │
     mixed in with:      mixed in with:      mixed in with:
     ┌────────┴────────┐ ┌────────┴────────┐ ┌────────┴────────┐
     ▼                 ▼ ▼                 ▼ ▼                 ▼
ResourceAttribute  Extrema  ResourceAttribute Extrema  ResourceAttribute Extrema
Method              Method  Method             Method  Method             Method
     +                  +        +                +         +                +
ConstraintAttribute Information ConstraintAttribute Information ConstraintAttribute Information
Method              Method      Method              Method     Method              Method
     │                  │        │                  │         │                  │
     ▼                  ▼        ▼                  ▼         ▼                  ▼
NodeResource      NodeStatus  LinkResource     LinkStatus  GraphResource   GraphStatus
Attribute         /Extrema    Attribute        /Extrema    Attribute       /Extrema
                   Attribute                    Attribute                  Attribute
                                                      │
                                              LinkLatencyAttribute
                                          (Resource+Constraint mixin,
                                           extra owner-specific class)
```

Concrete classes that actually exist in `virne/network/attribute/`:

| Owner | Concrete classes |
|---|---|
| Node | `NodeStatusAttribute`, `NodeExtremaAttribute`, `NodeResourceAttribute`, `NodePositionAttribute` |
| Link | `LinkStatusAttribute`, `LinkExtremaAttribute`, `LinkResourceAttribute`, `LinkLatencyAttribute` |
| Graph | `GraphStatusAttribute`, `GraphExtremaAttribute`, `GraphResourceAttribute` |

Behavior mixins, defined in `attribute_method.py`:

- **`ResourceAttributeMethod`** — consumable capacity/demand semantics
- **`ExtremaAttributeMethod`** — min/max bound tracking
- **`InformationAttributeMethod`** — static, non-consumable metadata
- **`ConstraintAttributeMethod`** — feasibility-checking logic (used to test whether a candidate node/link satisfies a resource requirement)

So, e.g.:
```python
class NodeResourceAttribute(ResourceAttributeMethod, ConstraintAttributeMethod, NodeAttribute):
    ...
```
"Resource-ness" and "constraint-checkability" are orthogonal, composed behaviors — not separate leaf types per metric name.

### Resource vs. Information Attributes

```
NodeAttribute
├── NodeResourceAttribute    (consumable: cpu, ram, gpu — mutated during embedding)
└── NodeStatusAttribute      (static: position, label, type — untouched during embedding)

LinkAttribute
├── LinkResourceAttribute    (consumable: bandwidth)
├── LinkLatencyAttribute     (consumable/constrained: delay-type resource)
└── LinkStatusAttribute      (static: distance, cost, metadata)
```

Only **Resource-type** attributes (mixed with `ResourceAttributeMethod`) are modified during embedding. Status/Information attributes remain unchanged.

---

## `TopologyGenerator`

Responsible **only** for constructing graph structure (nodes + edges) — no resource values.

```
Configuration
     │
     ▼
TopologyGenerator.generate(type, num_nodes, **kwargs)
     │
     ▼
networkx.Graph (structure only)
     │
     ▼
spliced into BaseNetwork._node / ._adj
```

Actual supported `type` values (from the source `match` statement — this is the complete list):

- `'path'`
- `'star'`
- `'grid_2d'` (requires `m`, `n`)
- `'waxman'` (retries until connected)
- `'random'` (Erdős–Rényi, retries until connected)

There is **no Barabási–Albert generator**. Named topologies like **GEANT** or **Abilene** are **not** generator types — they are pre-built `.gml` files loaded directly via `config.topology.file_path` in `PhysicalNetwork.from_setting()`, which bypasses `TopologyGenerator` entirely.

---

## `create_attrs_from_setting()`

Constructs Attribute **objects** (schema, no data) from configuration.

```yaml
node_attrs_setting:
  - name: cpu
    type: resource
    distribution: uniform
    low: 50
    high: 100
```

This produces a `NodeResourceAttribute(name="cpu", ...)` instance and stores it in `node_attrs["cpu"]`. It does **not** generate numerical values yet.

---

## `generate_attrs_data()`

After topology exists, this method asks every attribute object to populate the graph with sampled values:

```
Topology
   │
   ▼
node_attrs["cpu"].generate_data() ──► written into G.nodes[n]["cpu"]
   │
   ▼
link_attrs["bw"].generate_data()  ──► written into G.edges[u,v]["bw"]
   │
   ▼
Completed Network
```

Each attribute object is responsible only for generating its own values (polymorphism over the mixin behaviors described above).

---

## `TopologicalMetricCalculator`

Computes **node-level centrality metrics only** — this is narrower than a general "graph statistics" toolkit. The complete set of fields it produces (`TopologicalMetrics` dataclass):

- `node_degree_centrality`
- `node_closeness_centrality`
- `node_eigenvector_centrality`
- `node_betweenness_centrality`

Each is optional and gated by a boolean flag at call time, and each is min-max normalized by default. There is **no** diameter, density, clustering coefficient, average degree, or connectivity computation in this class.

```
BaseNetwork
     │
     ▼
TopologicalMetricCalculator.calculate(network, degree=, closeness=, eigenvector=, betweenness=)
     │
     ▼
TopologicalMetrics(node_degree_centrality, node_closeness_centrality,
                    node_eigenvector_centrality, node_betweenness_centrality)
```

It never modifies the graph, and results can be cached via `add_to_cache` / `get_from_cache` keyed by an arbitrary cache key.

---

## `PhysicalNetwork`

```
BaseNetwork
      │
      ▼
PhysicalNetwork
```

Represents the substrate infrastructure. Nodes = physical servers, edges = physical links. Node/link resource attributes represent **capacities**.

```
Server                     Link
CPU capacity = 64          Bandwidth capacity = 100
RAM capacity = 128         Delay = 5ms
```

Key behavior beyond `BaseNetwork`:
- `generate_topology(..., type='waxman')` — different default type than the base class (`'path'`).
- `from_setting(config, seed)` — the real construction entry point. Loads a `.gml` file if `config.topology.file_path` exists; otherwise generates procedurally via `TopologyGenerator`. If loading a GML, any attribute key present in the file but not declared in config is back-filled as a generic `NodeStatusAttribute` / `LinkStatusAttribute`.
- `save_dataset` / `load_dataset` — GML persistence under a `dataset_dir/p_net.gml` convention.

---

## `VirtualNetwork`

```
BaseNetwork
      │
      ▼
VirtualNetwork
```

Structurally identical to `PhysicalNetwork` — same base class, same attribute machinery. The difference is purely semantic: resource attribute values represent **demands**, not capacities.

```
VNode
CPU demand = 8      (vs. Server CPU capacity = 64)
```

`VirtualNetwork` additionally declares `id`, `arrival_time`, `lifetime` as instance fields (populated externally by the simulator, not computed internally), and exposes derived properties:
- `total_node_resource_demand`
- `total_link_resource_demand`
- `total_resource_demand` (cached)

**There is no `VirtualNetworkRequest` subclass.** A "request" is just a `VirtualNetwork` with these graph attributes set — confirmed by an exhaustive search of the codebase.

---

## Virtual Network Attribute Hierarchy (corrected)

```
                              BaseNetwork
                                   │
                    ┌──────────────┼──────────────┐
                    │              │               │
                    ▼              ▼               ▼

              node_attrs      link_attrs      graph (native dict)
                    │              │               │
                    ▼              ▼               ▼

             Virtual Nodes   Virtual Links    VN Metadata
                    │              │               │
          ┌─────────┴───────┐  ┌───┴──────────┐    │
          ▼                 ▼  ▼              ▼    ▼
   NodeResource      NodeStatus  LinkResource  LinkStatus   id
   Attribute         Attribute   Attribute     Attribute    arrival_time
          │                 │        │              │       lifetime
          ▼                 ▼        ▼              ▼       (+ any custom
     cpu demand       vnf type   bw demand      delay/QoS     graph_attrs_setting
     ram demand       location                  cost/class    entries)
```

---

## `VirtualNetworkRequestSimulator`

The simulator is **not a network** — it's a generator/orchestrator that produces a stream of `VirtualNetwork` instances plus a chronological event log.

```
Environment
     │
     ▼
VirtualNetworkRequestSimulator
     │
     ├──► arrange_v_nets()     samples size / lifetime / arrival_time per VN
     ├──► _renew_v_nets()      builds VirtualNetwork objects (topology + attrs)
     └──► _renew_events()      builds one arrival + one leave VirtualNetworkEvent
                                per VN, time-sorted
     │
     ▼
Controller (consumes self.v_nets / self.events)
```

`VirtualNetworkEvent` is a tightly-typed dataclass with only two event kinds:
```python
type: int   # 1 = arrival, 0 = leave  — validated, no third state
```

This 2-state closed model (arrival/leave only) is the key limitation to be aware of if extending the simulator — e.g., for a migration feature, since there is currently no built-in event type for "relocate."

---

## Complete Network Module Architecture (corrected)

```
                                      networkx.Graph
                                             │
                                             ▼
                                   ┌────────────────┐
                                   │  BaseNetwork   │
                                   └────────────────┘
                                             │
              ┌──────────────────────────────┴─────────────────────────┐
              │                                                        │
              ▼                                                        ▼

   ┌────────────────────┐                              ┌─────────────────────┐
   │  PhysicalNetwork    │                              │   VirtualNetwork     │
   └────────────────────┘                              │ (id, arrival_time,   │
              │                                          │  lifetime as fields) │
              │                                          └─────────────────────┘
              │                                                        │
              ▼                                                        ▼

     Substrate Graph                                    Service Graph

     Nodes                                               Nodes
       │                                                   │
       ├ CPU capacity                                      ├ CPU demand
       ├ RAM capacity                                      ├ RAM demand
       └ (status attrs: position, label)                   └ (status attrs: vnf type)

     Links                                               Links
       │                                                   │
       ├ Bandwidth capacity                                ├ Bandwidth demand
       └ Delay (LinkLatencyAttribute)                       └ Delay demand


                                      ▲
                                      │
                                      │ generated by

                     ┌─────────────────────────────────┐
                     │  VirtualNetworkRequestSimulator  │
                     └─────────────────────────────────┘
                                      │
                                      ▼
                    self.v_nets  +  self.events (arrival/leave only)
```

---

## Construction Pipeline

```
Configuration
     │
     ▼
create_attrs_from_setting()      → builds node_attrs / link_attrs objects (no data)
     │
     ▼
generate_topology()              → TopologyGenerator builds nodes + edges (no resources)
     │
     ▼
generate_attrs_data()            → each attribute object samples & writes its own values
     │
     ▼
Complete Network
```

For `VirtualNetwork` instances specifically, this pipeline runs once per request inside `VirtualNetworkRequestSimulator._renew_v_nets()`, with `id` / `arrival_time` / `lifetime` injected as `graph_attrs_setting` at construction time — not computed by `VirtualNetwork` itself.

---

## Design Philosophy

- **`BaseNetwork`** represents graph data, subclassing `networkx.Graph` directly.
- **`TopologyGenerator`** builds graph structures only (`path`, `star`, `grid_2d`, `waxman`, `random`); named real-world topologies (GEANT, Abilene) are loaded from `.gml` files, not generated.
- **Attribute objects** compose owner (Node/Link/Graph) with behavior mixins (Resource/Extrema/Information/Constraint) to generate and manage values — not one class per metric.
- **`TopologicalMetricCalculator`** computes four node-centrality metrics on demand; it never modifies the graph.
- **`PhysicalNetwork`** represents substrate resources as capacities.
- **`VirtualNetwork`** represents service demands, and doubles as the "request" object once temporal graph attributes are injected — there is no separate request class.
- **`VirtualNetworkRequestSimulator`** produces the request stream (`v_nets` + arrival/leave `events`) but is itself not a network.

The module intentionally contains very little embedding logic, providing a flexible, extensible graph abstraction for the Environment, Controller, and Solver modules to consume.

This also clarifies the extension surface for new concepts like migration: the graph representation itself doesn't need to change — the two things that do need extending are the **attribute system** (e.g. tracking a VN's current physical placement, migration count) and the **event model** (`VirtualNetworkEvent.type` is currently a closed `{0, 1}` set with no "relocate" state).