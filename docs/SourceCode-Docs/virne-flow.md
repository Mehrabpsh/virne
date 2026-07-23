# Virne Solver Execution Trace: Simulation vs. RL Training

> Verified against `GeminiLight/virne` @ `main` (source cloned and read directly, not just GitHub UI/API). Every method reference below was confirmed to exist with the described behavior. No corrections were needed to the underlying claims — this is a cleanup/reformat pass, not an error-fix pass.

---

## 1. Two Environment Layers

Virne runs **two environment layers simultaneously** whenever an RL solver (`pg_mlp`, `ppo_*`, etc.) is used. They operate at different granularities and must not be confused.

| | System (outer) env | Instance (inner) env |
|---|---|---|
| Class | `SolutionStepEnvironment` (`virne/core/environment.py`) | `PlaceStepInstanceRLEnv` / `JointPRStepInstanceRLEnv` (`virne/solver/learning/rl_core/instance_rl_environment.py`) |
| Owned by | The `System` (`BaseSystem.from_config`) | The solver, per VN request (`InstanceAgent`) |
| Lifetime | Whole simulation run | One VN embedding attempt |
| `step()` consumes | A complete `Solution` | A single action (e.g. a physical node id) |
| Role | Evolves the simulation across many VN requests; commits or rolls back physical-network resources | Gym-style loop: agent picks actions, builds observations, computes step rewards, assembles a `Solution` incrementally |

### 1.1 System environment

Constructed once, in `BaseSystem.from_config`:

```python
env = SolutionStepEnvironment(p_net, v_net_simulator, controller, recorder, counter, logger, config)
```

This is then wrapped by `OnlineSystem` (or `OfflineSystem` / `ChangeableSystem` / `TimeWindowSystem`, depending on config). It owns the event stream (VN arrivals/leaves) and the authoritative physical network state.

Its observation is a pair of **deep copies**, so a solver can plan freely without mutating the live physical network:

```python
# SolutionStepEnvironment.get_observation
return {'v_net': copy.deepcopy(self.v_net), 'p_net': copy.deepcopy(self.p_net)}
```

### 1.2 Instance environment

Created fresh **inside the solver**, once per VN request, in `InstanceAgent.solve`:

```python
instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, self.logger, self.config)
```

`self.InstanceEnv` is set by whichever concrete solver subclass calls `InstanceAgent.__init__` — e.g. `PgMlpSolver` passes `PgMlpInstanceRLEnv` (a subclass of `PlaceStepInstanceRLEnv`).

---

## 2. Evaluation-Time Trace (`OnlineSystem.run`)

This is what happens during **inference/decoding** — i.e., when the simulation is just running a (possibly pretrained) policy, not learning.

### 2.1 System-level call order

```python
# OnlineSystem.run
self.ready()
instance = self.env.reset(seed)
while True:
    solution = self.solver.solve(instance)
    next_instance, _, done, info = self.env.step(solution)
    if done: break
    instance = next_instance
```

- `env.reset(seed)` prepares the event stream and returns `get_observation()` → `{v_net copy, p_net copy}`.
- `env.step(solution)` (`SolutionStepEnvironment.step`) consumes the solver's returned `Solution`:
  1. checks hard-constraint violations / admission-control thresholds,
  2. **on success** → `controller.deploy(v_net, p_net, solution)`,
  3. **on failure** → rolls back the physical network based on `place_result` / `route_result` / `early_rejection` (`rollback_for_failure`),
  4. records metrics (`count_and_add_record`) and advances to the next event (`transit_obs()`).

### 2.2 What `solver.solve(instance)` resolves to for RL solvers

For a concrete RL solver like `PgMlpSolver`:

```python
class PgMlpSolver(InstanceAgent, PGSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, PgMlpInstanceRLEnv)
        PGSolver.__init__(self, controller, recorder, counter, logger, config, build_policy, obs_as_tensor, **kwargs)
```

`InstanceAgent` is listed first in the MRO, and neither `PGSolver` nor its parent `RLSolver` defines `solve()` — confirmed by grepping `rl_solver.py`, which has no `def solve` at all. So `OnlineSystem.run()` always resolves `solver.solve(instance)` to **`InstanceAgent.solve`** for every RL solver in this family (`pg_mlp`, `a2c_*`, `ppo_*`, `a3c_*` unless individually overridden).

### 2.3 `InstanceAgent.solve` — the per-instance rollout is hidden behind a "searcher"

```python
# InstanceAgent.solve
def solve(self, instance):
    v_net, p_net = instance['v_net'], instance['p_net']
    instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, self.logger, self.config)
    solution = self.searcher.find_solution(instance_env)
    return solution
```

The solver makes multiple actions internally while building the solution, but those actions happen inside `searcher.find_solution(...)`, never at the system level.

### 2.4 Searcher's internal rollout loop

`RLSolver.eval(...)` builds a searcher via `get_searcher(decode_strategy, ...)`, selecting among strategies defined in `searcher.py`: `GreedySearcher`, `SingleSampleSearcher`, `SampleSearcher`, `BeamSearcher`, `GreedyWithRestartSearcher`, `RecovableSearcher`, `RandomSearcher`, `OneShotSearcher`.

For greedy decode, the per-instance action loop is:

```python
obs = instance_env.get_observation()
while not done:
    mask = instance_env.generate_action_mask()
    tensor_obs = preprocess_obs_func(obs, device)
    action = policy.act(tensor_obs, mask, sample=False)
    obs, reward, done, info = instance_env.step(action)
return instance_env.solution
```

### 2.5 Full evaluation-time control flow

```
OnlineSystem.run
 -> SolutionStepEnvironment.reset          (gets {v_net copy, p_net copy})
 -> InstanceAgent.solve
    -> instance_env = InstanceEnv(p_net, v_net)    [PlaceStepInstanceRLEnv / JointPRStepInstanceRLEnv]
    -> searcher.find_solution(instance_env)         [Searcher subclass]
       -> obs = instance_env.get_observation()
       -> loop until done:
          -> tensor_obs = preprocess_obs(obs)        [RLSolver, bound to TensorConvertor]
          -> action = policy.act(tensor_obs)          [Policy / ActorCritic]
          -> instance_env.step(action)                [PlaceStepInstanceRLEnv.step / JointPRStepInstanceRLEnv.step]
             -> controller.node_mapper.place(...)      [two-stage variant]
             -> controller.link_mapper.link_mapping(...) [two-stage variant, once all nodes placed]
             -> (or) controller.place_and_route(...)   [joint variant, single call]
          -> reward = reward_calculator.compute(...)   [via instance_env.compute_reward]
       -> return instance_env.solution
 -> SolutionStepEnvironment.step(solution)
    -> if success: controller.deploy(...)
    -> else: rollback_for_failure(...)
    -> count_and_add_record(...)
    -> transit_obs() -> next event
 -> repeat
```

**Confirmed source detail:** the two-stage vs. joint distinction is real and lives in `instance_rl_environment.py`:
- `PlaceStepInstanceRLEnv.step` → calls `controller.node_mapper.place(...)` first; once every virtual node is placed, calls `controller.link_mapper.link_mapping(...)` once.
- `JointPRStepInstanceRLEnv.step` → calls `controller.place_and_route(...)` per action, combining node placement and link routing in a single controller call.

---

## 3. Training-Time Trace (rollout → buffer → update)

This is where the classic RL loop — **actions → rollout saved → actor/critic updated** — actually happens.

### 3.1 Where training is triggered

Training runs inside `BaseSystem.ready()`, **before** `OnlineSystem.run()` starts the evaluation loop:

```python
# BaseSystem.ready
if isinstance(self.solver, RLSolver) and num_train_epochs > 0:
    self.solver.learn(self.env, num_epochs=num_train_epochs)
self.solver.eval()
```

So `solver.learn(...)` is not called from inside the simulation loop — it's a distinct pretraining phase that runs first, using the **same system environment** (`self.env`) that evaluation will later use.

### 3.2 Training call order

```
RLSolver.learn(env, num_epochs)
 -> learn_singly(env, num_epochs)        [or learn_distributedly, if config.training.distributed_training]
    -> instance = env.reset(...)
    -> loop over VN instances:
       -> solution, instance_buffer, last_value = learn_with_instance(instance)   [InstanceAgent]
       -> merge_instance_experience(...)                                          [InstanceAgent]
       -> if self.buffer.size() >= self.target_steps:
             loss = self.update()                                                 [PGSolver.update, etc.]
       -> instance, reward, done, info = env.step(solution)                       [SolutionStepEnvironment]
```

**Important nuance:** even during training, the *outer* loop advances the system environment with `env.step(solution)` — one call per completed VN — while the *inner* loop advances the instance RL environment with `instance_env.step(action)` — one call per placement decision.

### 3.3 Inner rollout loop — where actions happen and the buffer fills

```python
# InstanceAgent.learn_with_instance
instance_env = self.InstanceEnv(p_net, v_net, ...)
instance_obs = instance_env.reset()
while True:
    tensor_obs = self.preprocess_obs(instance_obs, self.device)
    action, action_logprob = self.select_action(tensor_obs, sample=True)
    value = self.estimate_value(tensor_obs)
    next_obs, reward, done, info = instance_env.step(action)
    instance_buffer.add(instance_obs, action, reward, done, action_logprob, value=value, next_obs=next_obs)
    if done: break
    instance_obs = next_obs
last_value = self.estimate_value(self.preprocess_obs(next_obs, self.device))
return instance_env.solution, instance_buffer, last_value
```

Method ownership in this loop:

| Method | Owning class |
|---|---|
| `learn_with_instance` | `InstanceAgent` |
| `preprocess_obs`, `select_action`, `estimate_value` | `RLSolver` (bound via `TensorConvertor`, used by the agent mixin) |
| `instance_env.step` | `PlaceStepInstanceRLEnv` / `JointPRStepInstanceRLEnv` |
| `instance_buffer.add` | `RolloutBuffer` |

### 3.4 Turning episode transitions into returns/advantages

```python
# InstanceAgent.merge_instance_experience
instance_buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda, method=self.compute_advantage_method)
self.buffer.merge(instance_buffer)
```

`RolloutBuffer.compute_returns_and_advantages(...)` (in `buffer.py`) supports GAE, Monte Carlo, and TD return computation, selected via `method`.

There's also a config-gated negative-sampling branch: if `config.rl.if_use_negative_sample` is set, a baseline solver's result is checked before deciding whether to merge the episode at all — otherwise, only *feasible* solutions get merged into the training buffer.

### 3.5 When the update happens, and what gets updated

Trigger, in `learn_singly`:

```python
if self.buffer.size() >= self.target_steps:
    loss = self.update()
```

**`PGSolver.update()`** (Policy Gradient):

```python
observations = self.preprocess_obs(self.buffer.observations, self.device)
actions = torch.LongTensor(self.buffer.actions).to(self.device)
returns = torch.FloatTensor(self.buffer.returns).to(self.device)
_, action_logprobs, _, _ = self.evaluate_actions(observations, actions, return_others=True)
loss = -(action_logprobs * returns).mean()          # REINFORCE loss
grad_clipped = self.update_grad(loss)
self.buffer.clear()
```

For `A2CSolver` / `PPOSolver`, the loss additionally includes an advantage-weighted actor term plus an MSE critic term (`self.criterion_critic`) and an entropy bonus — so those variants genuinely update **both actor and critic**, whereas plain `PGSolver` only has an actor term in its loss (the critic, if instantiated, isn't referenced there).

| Solver | What's actually in the loss |
|---|---|
| `PGSolver` | Actor only: `-(logprob * return).mean()` |
| `A2CSolver` / `PPOSolver` | Actor (advantage-weighted) + critic (MSE vs. return) + entropy bonus |

The actual gradient step is centralized in `RLSolver.update_grad(loss)`:

```python
self.optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)  # if clip_grad
self.optimizer.step()
```

(In distributed mode, this additionally zeros/steps a shared optimizer and syncs gradients via `sync_gradients`.)

---

## 4. Method-by-Method Ordered Trace

### 4.1 Evaluation iteration (inside `OnlineSystem.run`)

```
OnlineSystem.run                                    [OnlineSystem / BaseSystem]
 -> env.reset(seed)                                  [SolutionStepEnvironment / BaseEnvironment.reset]
    -> get_observation() -> {v_net copy, p_net copy}
 -> solver.solve(instance)                           [InstanceAgent.solve]
    -> instance_env = InstanceEnv(p_net, v_net)       [PlaceStepInstanceRLEnv / JointPRStepInstanceRLEnv __init__]
    -> searcher.find_solution(instance_env)           [Searcher subclass]
       -> obs = instance_env.get_observation()        [InstanceRLEnv]
       -> feature_constructor.construct(...)          [FeatureConstructor]
       -> obs['action_mask'] = ...                    [InstanceRLEnv]
       -> loop until done:
          -> tensor_obs = preprocess_obs(...)          [RLSolver, bound to TensorConvertor]
          -> action = policy.act(...)                  [Policy / ActorCritic]
          -> instance_env.step(action)                 [PlaceStepInstanceRLEnv.step / JointPRStepInstanceRLEnv.step]
             -> controller.node_mapper.place(...)       [Controller]
             -> controller.link_mapper.link_mapping(...) [Controller, when applicable]
          -> reward_calculator.compute(...)            [RewardCalculator, via InstanceRLEnv.compute_reward]
       -> return instance_env.solution
 -> env.step(solution)                                [SolutionStepEnvironment.step]
    -> if success: controller.deploy(...)              [Controller]
    -> else: rollback_for_failure(...)                 [BaseEnvironment]
    -> count_and_add_record(...)                       [BaseEnvironment / Recorder / Counter]
    -> transit_obs() -> next event                     [BaseEnvironment]
 -> repeat
```

### 4.2 Training iteration (rollout saving + parameter update)

```
BaseSystem.ready                                     [BaseSystem]
 -> solver.learn(system_env, num_epochs)              [RLSolver.learn]
    -> learn_singly(system_env, num_epochs)           [InstanceAgent.learn_singly]
       -> instance = system_env.reset(...)            [SolutionStepEnvironment.reset]
       -> loop over VN instances:
          -> learn_with_instance(instance)            [InstanceAgent.learn_with_instance]
             -> instance_env = InstanceEnv(...)         [PlaceStepInstanceRLEnv / JointPRStepInstanceRLEnv]
             -> obs = instance_env.reset()               [InstanceRLEnv.reset -> RLBaseEnv.reset]
             -> loop steps:
                -> tensor_obs = preprocess_obs(...)       [RLSolver binding]
                -> action, logprob = select_action(...)   [RLSolver.select_action]
                -> value = estimate_value(...)            [RLSolver.estimate_value]
                -> next_obs, r, done = instance_env.step(action)  [Instance RL env]
                -> instance_buffer.add(...)               [RolloutBuffer.add]
             -> return solution, instance_buffer, last_value
          -> merge_instance_experience(...)             [InstanceAgent.merge_instance_experience]
             -> instance_buffer.compute_returns_and_advantages(...)  [RolloutBuffer]
             -> self.buffer.merge(instance_buffer)
          -> if self.buffer.size() >= target_steps:
                self.update()                            [PGSolver.update / A2CSolver.update / ...]
                -> evaluate_actions(...)                  [RLSolver.evaluate_actions]
                -> loss = ...
                -> update_grad(loss)                      [RLSolver.update_grad]
                   -> optimizer.step()
          -> system_env.step(solution)                    [SolutionStepEnvironment.step]
```

This is the exact "actions → rollout saved → update" sequence.

---

## 5. Why `solver.solve` Is Not the Update Boundary

A subtlety worth stating explicitly:

- `solver.solve(instance)` is an **inference/decode** operation — it builds one `Solution` for one VN — even though internally it steps the instance RL env many times.
- The **parameter update** is driven entirely by the training loop (`solver.learn → learn_singly → update`) and is normally invoked from `BaseSystem.ready()`, strictly *before* `OnlineSystem.run()` starts the evaluation simulation loop.
- That's why rollout-saving and update logic live in `learn_with_instance`, `merge_instance_experience`, and `update()` — never inside `solve()` itself. `solve()` is decode-only and is shared, unmodified, between training-time instance rollouts (via `learn_with_instance`, which duplicates similar logic rather than calling `solve`) and evaluation-time calls.

---

## 6. Source References

All claims above were checked directly against these files in `GeminiLight/virne` (`main` branch):

- `virne/system/base_system.py` — `BaseSystem.from_config`, `.ready`, `OnlineSystem.run`
- `virne/core/environment.py` — `BaseEnvironment`, `SolutionStepEnvironment`
- `virne/solver/learning/rl_core/instance_agent.py` — `InstanceAgent.solve` / `.learn_with_instance` / `.merge_instance_experience` / `.learn_singly`
- `virne/solver/learning/rl_core/instance_rl_environment.py` — `PlaceStepInstanceRLEnv`, `JointPRStepInstanceRLEnv`
- `virne/solver/learning/reinforcement_learning/mlp_solver.py` — `PgMlpSolver`, `PgMlpInstanceRLEnv`
- `virne/solver/learning/rl_core/searcher.py` — `get_searcher`, `Searcher` subclasses
- `virne/solver/learning/rl_core/rl_solver.py` — `RLSolver`, `PGSolver`, `A2CSolver`, `PPOSolver`, `A3CSolver`
- `virne/solver/learning/rl_core/buffer.py` — `RolloutBuffer`

---

## 7. Open Extension: Two-Stage vs. Joint Trace

The step-level logic genuinely differs between the two mapping strategies:

- **Two-stage** (`PlaceStepInstanceRLEnv`): `controller.node_mapper.place(...)` for the current virtual node → once *all* virtual nodes are placed, a single `controller.link_mapper.link_mapping(...)` call routes every virtual link at once.
- **Joint** (`JointPRStepInstanceRLEnv`): `controller.place_and_route(...)` handles node placement and link routing together, per action — plus revocable-action and interaction-limit handling (`num_interactions > 10 * v_net.num_nodes` triggers a forced reject) that has no equivalent in the two-stage path.

A dedicated side-by-side trace of these two step() implementations would be a natural follow-up once you're ready to decide which mapping style your migration feature should hook into.