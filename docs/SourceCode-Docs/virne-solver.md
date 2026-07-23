# The `solver` Module in Virne — Architecture & Control Flow

> Verified against `GeminiLight/virne` @ `main`. Checked: `virne/solver/base_solver.py`, `virne/solver/heuristic/*.py`, `virne/solver/exact/*.py`, `virne/solver/meta_heuristic/*.py`, `virne/solver/rank/*.py`, `virne/solver/learning/rl_core/{rl_solver,instance_agent,online_agent,policy_builder,feature_constructor,reward_calculator,buffer}.py`, `virne/solver/learning/rl_policy/*.py`, `virne/solver/learning/reinforcement_learning/mlp_solver.py`.
>
> The original write-up's high-level framing (composition over inheritance, five subsystems, solve → Solution as the universal interface) is directionally correct and kept. Several specifics needed correction — most importantly, **the actual extensibility mechanism is a decorator-based registry, not inheritance-driven polymorphism**, and **the "Environment → Controller → Solver" call direction was backwards**. Corrections are called out explicitly below.

---

## 0. Diagrams

### 0.1 Class hierarchy — four independent branches off `Solver`

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Solver (base class)                          │
│                                                                        │
│  ┌────────────────────────┐        ┌────────────────────────┐        │
│  │        Exact            │        │        Heuristic         │        │
│  │  MIP, deterministic     │        │  5 independent families  │        │
│  │  and randomized         │        │  rank, BFS, fit, ego-net,│        │
│  │  rounding               │        │  joint-PR                │        │
│  └────────────────────────┘        └────────────────────────┘        │
│                                                                        │
│  ┌────────────────────────┐        ┌────────────────────────┐        │
│  │     Meta-heuristic       │        │  Reinforcement learning  │        │
│  │  one shared base class  │        │  PG, A2C, PPO, DQN, DDPG │        │
│  │  GA, ACO, PSO, SA, tabu │        │  + Safe RL mixin family  │        │
│  └────────────────────────┘        └────────────────────────┘        │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
   Exact / Heuristic / Meta-heuristic = classical solvers (direct subclasses)
   Reinforcement learning = learning-based (composed via multiple-inheritance mixins)
```

### 0.2 Interaction flow — who calls whom for a single VN request

```
                    ┌───────────────────────────┐
                    │    System / Environment     │
                    │    drives the event loop     │
                    └──────────────┬────────────┘
                                   │ instance
                                   ▼
                    ┌───────────────────────────┐
                    │           Solver             │
                    │  solve(instance) → Solution  │
                    └──────────────┬────────────┘
                                   │ place / route
                                   ▼
                    ┌───────────────────────────┐
                    │          Controller          │
    ┌───────────────┤  used by solver AND          │
    │  ↻ next event  │  environment                 │
    │                └──────────────┬────────────┘
    │                               │ count_solution()
    │                               ▼
    │                ┌───────────────────────────┐
    │                │           Counter             │
    │                │  computes revenue and cost    │
    │                └──────────────┬────────────┘
    │                               │ add_record()
    │                               ▼
    │                ┌───────────────────────────┐
    │                │          Recorder             │
    │                │  persists history and totals  │
    │                └──────────────┬────────────┘
    │                               │
    └───────────────────────────────┘
```

Key point encoded in the diagram: `Controller` sits **downstream of both** `Solver` (called during placement, while building a `Solution`) **and** `System`/`Environment` (called again afterward, to deploy or roll back) — it is never the thing that decides to invoke a solver, correcting the original doc's `Environment → Controller → Solver` framing (see §9).

### 0.3 RL solver composition — Agent Wrapper + RL Algorithm + Policy Builder → concrete solver

```
 ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
 │    RL algorithm      │   │    Agent wrapper     │   │    Policy builder    │
 │  PG / A2C / PPO       │   │  instance or online   │   │  MLP / CNN / GNN      │
 │  (+Safe RL)           │   │                       │   │  family               │
 └──────────┬─────────┘   └──────────┬─────────┘   └──────────┬─────────┘
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     ▼
                    ┌───────────────────────────────┐
                    │        Concrete RL solver        │
                    │  multiple inheritance +           │
                    │  injected policy                  │
                    └──────────────┬────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
 ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
 │     PgMlpSolver       │   │  A2CDualGcnSolver     │   │  PpoDualGcnSolver     │
 │  instance + PG + MLP  │   │  instance + A2C +      │   │  instance + PPO +      │
 │                        │   │  dual-GCN               │   │  dual-GCN               │
 └────────────────────┘   └────────────────────┘   └────────────────────┘
```

`RLSolver` (§5) supplies rollout-buffer handling and the gradient-update logic (`update()`, `update_grad()`); it does not know how the environment is stepped or which network architecture the policy uses. The **Agent Wrapper** (§4) supplies environment interaction (per-node `InstanceAgent` vs. whole-VN `OnlineAgent`); it does not know the algorithm's loss function. The **Policy Builder** (§6) supplies the network + optimizer; it does not know the training loop. A concrete solver — e.g. `PgMlpSolver(InstanceAgent, PGSolver)` with `build_policy = PolicyBuilder.build_mlp_policy` — is the point where all three get wired together via multiple inheritance plus constructor injection (see §8 for the fully traced example).

---

## 1. Overview

The `solver` module is the algorithmic core of Virne — every optimization technique capable of solving a VNE instance lives here. It doesn't model graphs (`network` module) or orchestrate execution (`system` module); it makes embedding decisions.

```
Embedding Instance {v_net, p_net}
            │
            ▼
    solver.solve(instance)
            │
            ▼
        Solution
```

Every solver — heuristic, exact, meta-heuristic, or RL — exposes this same `solve(instance) -> Solution` interface, confirmed as the sole abstract contract in `Solver` (`base_solver.py`):

```python
class Solver:
    type: str
    def __init__(self, controller, recorder, counter, logger, config, **kwargs): ...
    def ready(self): pass
    def solve(self, instance: dict) -> Solution:
        raise NotImplementedError
```

---

## 2. Correction: How Solvers Actually Get Selected — `SolverRegistry`, Not Inheritance

`base_solver.py` also defines `SolverRegistry`, a **decorator-based, string-keyed registry** — this, not a shared abstract subclass tree, is what lets the rest of the framework "invoke different algorithms through a common interface":

```python
class SolverRegistry:
    _registry: Dict[str, Type[Solver]] = {}

    @classmethod
    def register(cls, solver_name, solver_type='unknown', env_cls=SolutionStepEnvironment):
        def decorator(handler_cls):
            setattr(handler_cls, 'type', solver_type)
            cls._registry[solver_name] = handler_cls
            return handler_cls
        return decorator

    @classmethod
    def get(cls, name) -> Type[Solver]:
        return cls._registry[name]
```

Every concrete solver is registered like this (from `mlp_solver.py`):
```python
@SolverRegistry.register(solver_name='pg_mlp', solver_type='r_learning')
class PgMlpSolver(InstanceAgent, PGSolver):
    ...
```

And resolved by string name in `BaseSystem.from_config`:
```python
solver_cls = SolverRegistry.get(config.solver.solver_name)
solver = solver_cls(controller, recorder, counter, logger, config)
```

This same registry pattern recurs throughout the module — `FeatureConstructorRegistry`, `RewardCalculatorRegistry`, `ActorCriticRegistry`, `NodeRankRegistry` all follow the identical `@XRegistry.register(name)` decorator idiom. **This is the real mechanism behind Virne's "swap a component without touching the rest" extensibility claim** — not polymorphic dispatch through a common ancestor class. It's a flat, config-driven factory pattern layered *on top of* a shallow, fragmented inheritance tree (see §3).

---

## 3. Correction: The "Solver Hierarchy" Is Not a Clean Tree

The original diagram implied a tidy `Solver → {Exact, Heuristic, Meta-Heuristic, RL}` tree. The actual class layout, confirmed by grepping every `class` declaration in each subdirectory, is flatter and more fragmented — most families branch **directly off `Solver`**, with no shared intermediate "HeuristicSolver" abstract class:

```
Solver  (base_solver.py — the only universal ancestor)
│
├── exact/            [each is an independent direct subclass of Solver]
│     ├── MipSolver
│     ├── DeterministicRoundingSolver
│     └── RandomizedRoundingSolver
│
├── heuristic/         [FIVE unrelated family roots, all direct children of Solver]
│     ├── BaseNodeRankSolver → OrderRankSolver, RandomRankSolver, GRCRankSolver,
│     │                        FFDRankSolver, NRMRankSolver, PLRankSolver,
│     │                        NEARankSolver, RandomWalkRankSolver
│     ├── BaseJointPRSolver  → RandomJointPRSolver, OrderJointPRSolver, FFDJointPRSolver
│     ├── BfsSolver          → OrderRankBfsSolver, RandomRankBfsSolver, RandomWalkRankBfsSolver
│     ├── FitSolver          → FirstFitSolver   (RandomFitSolver, NearestFitSolver subclass Solver directly, not FitSolver)
│     └── EgoNetworkSolver
│
├── meta_heuristic/    [one real shared base]
│     └── BaseMetaHeuristicSolver → GeneticAlgorithmSolver → ISGeneticAlgorithmSolver
│                                  → AntColonyOptimizationSolver
│                                  → ParticleSwarmOptimizationSolver
│                                  → SimulatedAnnealingSolver
│                                  → TabuSearchSolver
│                                  → LocalSearch
│
└── learning/rl_core/  [one real shared base, richest family]
      └── RLSolver → PGSolver, A2CSolver, PPOSolver → A3CSolver
                    → ARPPOSolver, DPGSolver, DDPGSolver, DQNSolver → DoubleDQNSolver
      └── SafeRLSolver (mixed in alongside PPOSolver via multiple inheritance):
            FixedPenaltyPPOSolver(PPOSolver, SafeRLSolver)
            LagrangianPPOSolver(PPOSolver, SafeRLSolver)
            NeuralLagrangianPPOSolver(PPOSolver, SafeRLSolver)
            RobustNeuralLagrangianPPOSolver(PPOSolver, SafeRLSolver)
            RewardCPOSolver(PPOSolver, SafeRLSolver)
            AdaptiveStateWiseSafePPOSolver(PPOSolver, SafeRLSolver)
```

Two things worth internalizing from this:

1. **Heuristics are organized by "what strategy the base class encodes"** (rank-based, joint-place-route, BFS-based, best-fit-based, ego-network-based) — not by a common `HeuristicSolver` ancestor. If you were adding a migration-aware heuristic, you'd most naturally extend whichever of these five families matches your placement strategy, not a nonexistent shared heuristic base.
2. **The `Safe RL` family — entirely absent from the original write-up — uses multiple inheritance to *compose* a base algorithm (`PPOSolver`) with a cross-cutting concern (`SafeRLSolver`, constraint-safety objectives/Lagrangian penalties)**. This is the module's clearest real-world precedent for "add a new orthogonal concern to an existing algorithm via a mixin" — arguably the most relevant existing pattern to study before adding a `MigrationAware` mixin of your own.

Separately, `rank/node_rank.py` and `rank/link_rank.py` define **`NodeRank`/`LinkRank`** — small ranking-strategy classes (`ABC`-based, not `Solver` subclasses at all) consumed *by* the heuristic solvers above (e.g. `BaseNodeRankSolver` uses a `NodeRank` strategy object internally) via yet another registry, `NodeRankRegistry`. So "ranking" is itself a pluggable strategy nested one level inside the heuristic solver family, not part of the solver hierarchy proper.

---

## 4. Agent Wrappers — Correction: `InstanceAgent` and `OnlineAgent` Are Not Parallel/Equivalent

The original doc presented `InstanceAgent` and `OnlineAgent` as two interchangeable implementations of the same `reset → observe → select_action → step → collect_transition` loop. Reading both source files shows they operate at **different granularities and serve different decision types**:

| | `InstanceAgent` | `OnlineAgent` |
|---|---|---|
| Docstring | "Training and Inference Methods for **Resource Allocation** Agent" | "Training and Inference Methods for **Admission Control** Agent" |
| Decision type | Sequential — one action per virtual node, across multiple steps | Single-shot — exactly one action per VN request |
| Environment used | Creates a **fresh inner `InstanceEnv`** per VN request (`self.InstanceEnv(p_net, v_net, ...)`) | No inner environment at all — acts directly on the outer system env's observation |
| `solve(instance)` | `instance_env = self.InstanceEnv(...); return self.searcher.find_solution(instance_env)` | `instance = preprocess_obs(instance); action, _ = select_action(instance, sample=False); return action` |
| What "action" means | A physical node id (one placement decision) | A whole accept/reject-style decision for the VN |

So these are two different **decision granularities for two different problems** — embedding (where to place each virtual node) vs. admission control (whether to accept a VN at all) — not two flavors of the same wrapper. A migration-triggering policy is conceptually closer to `OnlineAgent`'s shape (a coarse, whole-VN decision — "migrate or don't, and if so where") than to `InstanceAgent`'s node-by-node placement loop, though it could reuse `InstanceAgent`'s per-instance environment machinery if migration is modeled as a re-placement task.

---

## 5. RL Algorithms — Corrected, Complete List

Confirmed classes in `rl_solver.py` and `safe_rl_solver.py` (the original list of PG/PPO/A2C/DQN/DDPG was roughly right but incomplete — A3C, ARPPO, DPG, DoubleDQN, and the entire Safe-RL family were missing):

```
RLSolver (abstract base: buffer, optimizer, device, distributed-training scaffolding)
│
├── PGSolver            — Policy Gradient (REINFORCE); actor-only loss
├── A2CSolver            — Advantage Actor-Critic; actor + critic (MSE) + entropy
├── PPOSolver            — Proximal Policy Optimization; clipped surrogate objective
│     └── A3CSolver       — Asynchronous variant built on PPOSolver
├── ARPPOSolver          — (Action-Rejection / Auto-Regressive-flavored PPO variant)
├── DPGSolver            — Deterministic Policy Gradient
├── DDPGSolver           — Deep Deterministic Policy Gradient
└── DQNSolver            — Deep Q-Network
      └── DoubleDQNSolver — Double-DQN variant

SafeRLSolver  (mixed in with PPOSolver via multiple inheritance — constraint-aware objectives)
├── FixedPenaltyPPOSolver
├── LagrangianPPOSolver
├── NeuralLagrangianPPOSolver
├── RobustNeuralLagrangianPPOSolver
├── RewardCPOSolver
└── AdaptiveStateWiseSafePPOSolver
```

These define loss functions, update rules, and (for the Safe-RL family) constraint-penalty terms. They never touch the environment directly — as the original doc correctly noted, environment interaction is delegated to the Agent Wrapper (§4), and the concrete solver class multiply-inherits from both (e.g. `PgMlpSolver(InstanceAgent, PGSolver)`), with `InstanceAgent` positioned first in the MRO so its `solve()` wins.

---

## 6. Policy Architectures — Corrected Names and Build Pattern

Confirmed classes across `rl_policy/*.py`. The original "GCN / GAT / MLP / CNN / Seq2Seq / Dual-GCN / Dual-GAT" list is close but the real class names and construction pattern are worth being precise about, since you'll be instantiating one of these (or adding a new one) for any migration-aware policy:

```
BaseActorCritic (base_policy.py)
│
├── MlpActorCritic
├── CnnActorCritic
├── AttActorCritic          (attention-based — this is the "GAT-flavored" single-view policy)
├── GcnSeq2SeqActorCritic   (GCN encoder + RNN Seq2Seq decoder, for sequential/ordered placement)
│
├── Single-GNN family, built via a `_make_actor_critic(GNNClass)` factory function:
│     ├── GcnMlpActorCritic              = _make_actor_critic(GCNConvNet)
│     ├── GatMlpActorCritic              = _make_actor_critic(GATConvNet)
│     └── DeepEdgeFeatureGATActorCritic  = _make_actor_critic(DeepEdgeFeatureGAT)
│
└── Dual-view ("Bi-") GNN family, same factory pattern, operating jointly over p_net + v_net:
      ├── BiGcnActorCritic                        = _make_actor_critic(GCNConvNet)
      ├── BiGatActorCritic                        = _make_actor_critic(GATConvNet)
      ├── BiDeepEdgeFeatureGatActorCritic          = _make_actor_critic(DeepEdgeFeatureGAT)
      └── Safe-RL variants, via _make_advanced_actor_critic(GNNClass, use_cost_critic=, use_lambda_net=):
            BiGcnWithCostActorCritic, BiGatWithCostActorCritic,
            BiDeepEdgeFeatureGatWithCostActorCritic,
            BiGcnWithCostAndLambdaActorCritic, BiGatWithCostAndLambdaActorCritic,
            BiDeepEdgeFeatureGatWithCostAndLambdaActorCritic
```

Two structural points the original diagram missed:

- **Policies aren't hand-written per architecture — most are generated by a factory function** (`_make_actor_critic(gnn_cls)` / `_make_advanced_actor_critic(gnn_cls, **flags)`) that takes a GNN backbone class and produces an `ActorCritic` subclass around it. Adding a new GNN backbone (e.g. a Graph Transformer, as the original doc's "Extensibility" section suggested) means writing the GNN layer and passing it through this factory — not writing a new `ActorCritic` class from scratch.
- **The Safe-RL cost-critic/lambda-net variants are built the same way**, with extra boolean flags — this is the concrete mechanism by which "add a new optimization objective without touching the base architecture" (claimed generically in the original doc) actually happens: flags into a factory function, not subclass proliferation.

Construction is centralized in **`PolicyBuilder`** (`policy_builder.py`), a class of `@staticmethod`s (e.g. `PolicyBuilder.build_mlp_policy`) that reads feature dimensions from config (`get_feature_dim_config`) and returns `(policy, optimizer)` — this is what the original doc's "Policy Builder" box represents, confirmed accurate.

---

## 7. RL Infrastructure — Confirmed, With Registry Detail Added

```
FeatureConstructorRegistry   (feature_constructor.py)
  BaseFeatureConstructor
  ├── 'p_net'          → PNetFeatureConstructor
  ├── 'p_net_v_node'   → PNetVNodeFeatureConstructor   (used by PgMlpInstanceRLEnv, e.g.)
  └── 'p_net_v_net'    → PNetVNetFeatureConstructor

RewardCalculatorRegistry     (reward_calculator.py)
  BaseRewardCalculator (ABC)
  ├── 'vanilla'              → VanillaRewardCalculator
  ├── 'fixed_intermediate'   → FixedWeightRewardCalculator
  ├── 'adaptive_intermediate'→ AdaptiveWeightRewardCalculator
  └── 'gradual_intermediate' → GradualIntermediateRewardCalculator

RolloutBuffer                (buffer.py — already covered in the Counter/Recorder + training-trace docs)
  .add(...) / .compute_returns_and_advantages(...) / .merge(...) / .clear()

Instance / Online RL Environments  (instance_rl_environment.py, online_rl_environment.py, rl_enviroment_base.py)
  InstanceRLEnv → PlaceStepInstanceRLEnv, JointPRStepInstanceRLEnv,
                  SolutionStepInstanceRLEnv, NodePairStepInstanceRLEnv, NodeSlotsStepInstanceRLEnv
```

Both `feature_constructor.name` and `reward_calculator.name` are set as config strings and resolved through their registries at construction — e.g. `PgMlpInstanceRLEnv.__init__` sets `config.rl.feature_constructor.name = 'p_net_v_node'` and `config.rl.reward_calculator.name = 'vanilla'` before calling `super().__init__`. This confirms the original doc's high-level "Feature Constructor → Observation Tensor" / "Reward Calculator → Reward" framing, just clarifies that selection is registry/config-driven rather than a fixed pipeline.

---

## 8. Concrete Solver Composition (Corrected Example)

Using `PgMlpSolver` as the traced concrete example (already verified in the RL-training-trace document):

```
PgMlpSolver(InstanceAgent, PGSolver)              [multiple inheritance — MRO puts InstanceAgent first]
│
├── from InstanceAgent:   .solve(), .learn_with_instance(), .merge_instance_experience(), .learn_singly()
│                          → InstanceEnv = PgMlpInstanceRLEnv  (subclass of PlaceStepInstanceRLEnv)
│
├── from PGSolver/RLSolver: .learn(), .eval(), .update(), .update_grad(), .select_action(),
│                            .estimate_value(), .evaluate_actions(), .buffer (RolloutBuffer)
│
├── policy = build_policy(self)                    [PolicyBuilder.build_mlp_policy → MlpActorCritic + optimizer]
├── preprocess_obs = obs_as_tensor                  [TensorConvertor.obs_as_tensor_for_mlp]
│
└── PgMlpInstanceRLEnv.__init__ pins config:
      feature_constructor.name = 'p_net_v_node'      → PNetVNodeFeatureConstructor
      reward_calculator.name   = 'vanilla'            → VanillaRewardCalculator
```

Every RL solver in the codebase follows this same shape: **two base classes** (an Agent Wrapper + an RL Algorithm) **composed via multiple inheritance**, plus **three externally-wired, registry-selected components** (policy, feature constructor, reward calculator) chosen through config strings rather than hardcoded. This is the accurate version of the original "Concrete Solver Composition" diagram.

---

## 9. Correction: Interaction With Other Modules — the Call Direction Was Backwards

The original doc's diagram read `Environment → Controller → Solver → Solution → Counter → Recorder`, implying the **Controller invokes the Solver**. Cross-checked against the previously-verified RL execution trace and `base_system.py` / `environment.py`, the actual call direction is the reverse of that middle step — **the System/Environment layer calls the Solver directly; the Solver calls the Controller, not the other way around**:

```
OnlineSystem.run()                                   [System layer]
   │
   ├─ instance = env.reset(seed)                       [Environment generates the instance]
   │
   ├─ solution = solver.solve(instance)                 [System calls Solver DIRECTLY — Controller not involved here]
   │      │
   │      └─ (inside solving) solver calls Controller repeatedly:
   │            controller.node_mapper.place(...)
   │            controller.link_mapper.link_mapping(...)
   │            controller.place_and_route(...)          [joint variant]
   │         → these build up the Solution incrementally
   │
   └─ env.step(solution)                                [System hands the finished Solution back to the Environment]
          ├─ controller.deploy(...)   / rollback_for_failure(...)   [Controller — commits/reverts p_net state]
          ├─ recorder.count(...) → counter.count_solution(...)      [Counter computes revenue/cost]
          └─ recorder.add_record(...)                                [Recorder persists the row]
```

So `Controller` is a **service used by both the Solver (during solving, to attempt placements) and the Environment (after solving, to commit/rollback)** — it is never the thing that decides to call a solver. `Counter`/`Recorder` sit strictly downstream of `env.step(solution)`, not the solver itself.

---

## 10. Runtime Control Flow (Corrected)

### 10.1 Training

```
BaseSystem.ready()
  └─ solver.learn(env, num_epochs)              [RLSolver.learn → learn_singly, per InstanceAgent]
       loop per VN instance:
         instance_env.step(action)  ×N            [one per virtual node — Instance RL Env]
            → feature_constructor.construct(...)   [Observation]
            → policy.act(...) / select_action(...) [Policy Network → Action]
            → controller.node_mapper.place / controller.link_mapper.link_mapping   [or place_and_route]
            → reward_calculator.compute(...)        [Reward]
         instance_buffer.add(...)                   ×N   [Rollout Buffer]
         merge_instance_experience → buffer.compute_returns_and_advantages → buffer.merge
         if buffer.size() >= target_steps: solver.update()   [RL Algorithm.update() → Policy gradient step]
         env.step(solution)                          [System env — advances to next VN event]
```

### 10.2 Inference

```
OnlineSystem.run()
  └─ solver.solve(instance)                     [InstanceAgent.solve or OnlineAgent.solve — no learning]
       InstanceAgent path:
         instance_env = InstanceEnv(...)
         searcher.find_solution(instance_env)     [Greedy / Sample / Beam — no gradient, no buffer]
       OnlineAgent path:
         action = select_action(preprocess_obs(instance), sample=False)   [single decision, no inner env]
  └─ env.step(solution)                          [System env — deploy/rollback, Counter/Recorder]
```

No optimizer, no rollout buffer, and (for `InstanceAgent`-based solvers) no `learn_with_instance` involvement during inference — confirmed identical to the dedicated RL-training-trace document's findings.

---

## 11. Extensibility — What's Actually True

Restating the original doc's extensibility claims against what's now confirmed:

| Original claim | Verified mechanism |
|---|---|
| "Add a new policy architecture without modifying PPO" | True — swap the `make_policy` function passed into any `RLSolver.__init__`; PPO's loss code never references a concrete architecture. |
| "Introduce a new RL algorithm while reusing existing policy networks" | True — any `ActorCritic` class is architecture-agnostic w.r.t. which `RLSolver` subclass calls it; confirmed by `PgMlpSolver` and `PPOSolver`-based solvers sharing `MlpActorCritic`-family policies in practice. |
| "Create a new Agent Wrapper for a different interaction model" | True, but note §4's correction — a new wrapper needs to decide its **decision granularity** (per-node like `InstanceAgent`, or whole-VN like `OnlineAgent`) since these aren't interchangeable, just two existing points on a spectrum. |
| "Replace the Feature Constructor without changing the algorithm" | True — registry-driven via a config string, confirmed in `feature_constructor.py`. |
| "Design new Reward Calculators for different objectives" | True — same registry pattern, confirmed in `reward_calculator.py`; the Safe-RL family additionally shows how to add a *second* objective (cost/safety) alongside the primary reward via `use_cost_critic`/`use_lambda_net` flags rather than a new reward calculator. |

For a **migration feature** specifically, the Safe-RL mixin pattern (`class XPPOSolver(PPOSolver, SafeRLSolver)`) is the most directly reusable precedent in the codebase: it shows how Virne already composes an orthogonal concern (constraint-safety) into an existing algorithm via multiple inheritance and factory-built policy variants, without touching `PPOSolver` itself — the same shape you'd likely want for a `MigrationAwarePPOSolver` or similar.