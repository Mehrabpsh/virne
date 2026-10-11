# Thesis implementation status — 2026-10-09

| Phase | Status | Tests | Notes |
|---|---|---|---|
| Audit | VERIFIED | Current source paths and integrated checkpoint | Starting commit/branch preserved |
| 0 baseline | VERIFIED | Original train + reload; exact disabled regression | 8 event rows × 61 columns equal; 92 policy tensors bitwise equal after training |
| 1 reliability data | VERIFIED | Rate math, heterogeneous provider, RNG isolation, persistence | Explicit/scalar/configured uniform rates; no final distribution selected |
| 2 request SLA | VERIFIED | Assignment/reseed/reload and workload isolation | All four levels supported; probabilities explicit |
| 3 evaluator | VERIFIED | Unique host/transit nodes/undirected edges | Same evaluator for every solver |
| 4 shortfall | VERIFIED | SLA equality, D=1, monotonicity, near-one validation | Compliance beta explicit |
| 5 objectives | VERIFIED | Four formulas and tiny train/inference runs | Final objective/alpha/beta validation-open |
| 6 rewards | VERIFIED | Four adapters + train/inference matrix | Original calculators unchanged |
| Potential reward | BLOCKED | Explicit pending-definition error tested | Phi and terminal convention not frozen |
| 7 features | VERIFIED | 128 mode/subset combinations; values/dimensions; strict checkpoint signature | Six frozen candidates; subset validation-open |
| 8 metrics | VERIFIED | Recorder, SLA classes, rejected-service handling, CSV/JSON manifest | Original metrics preserved |
| 9 regression | VERIFIED | Full suite 307 passed; latest thesis suite 32 passed | Python 3.10 existing environment; declared >=3.12 environment not validated |
| 10 heuristics | VERIFIED | Four original + four RA online smokes; RA contracts | RA-EVNE ranking port has explicit reconstruction conventions |
| 11 Cand-RL main | VERIFIED | Candidate actions/masks; PPO train/test; save/load; online reload | Configured no-redundancy thesis reconstruction, not exact Paper-9 reproduction |
| Cand-RL Paper-9 sanity | BLOCKED | Symbolic reward unit test; blocked-profile test | Backup accounting, operational cost/energy, reward constants/gate and episode fidelity unresolved |
| Offline smoke | VERIFIED | Baseline, node-only/node+link, four objectives, four rewards | DEV only, four requests, one epoch |
| Online lifecycle | VERIFIED | Overlapping requests and CLI train/test | Consumption, residual observations, departures and full release verified |
| Scientific validation / final test | NOT STARTED | None claimed | No large sweeps or final scientific choices authorized/frozen |

## Source of truth and repository state

### Sequential runner verification — 2026-10-11

Added `scripts/thesis/run_all.sh` to execute the successful DEV workflow in
sequence, reload its own checkpoints, check baseline reproducibility, and collect
one final results directory. `--dry-run` prints the command plan. Expanded
commands and per-step logs are retained in the results directory.

Executed:

```bash
TEST_DEPS=/tmp/thesis-test-deps RESULTS_DIR=results/thesis_suite/runner_verification_20261011 bash scripts/thesis/run_all.sh
```

Verified: **309 tests passed**, **26 completed runs** in the collected manifest,
baseline records equal (8 event rows), and all **92 policy tensors** equal.
Outputs: `results/thesis_suite/runner_verification_20261011/run_manifest.csv`
and `.json`; all 31 execution/check steps passed, with `COMPLETE.txt` present.
These repeated current-code baselines check determinism rather than recreating
the historical pre-change checkout. DEV values and scientific blockers remain
unchanged. Missing pytest/tomli dependencies were restored into `/tmp` using
`uv pip install --python /home/mehrab/venv/virne-venv/bin/python --target /tmp/thesis-test-deps pytest tomli`.

- Repository: `/home/mehrab/Workspaces/Codex/Thesis/virne`.
- Branch: `main`; starting commit: `cbd19aa223c628cc91345c7e7005e28cf7775929`.
- Initial `git status --short`: `?? Solvability-Evaluation.sh`. Preserved untouched; no commits/history operations authorized.
- Read source of truth: `/home/mehrab/Downloads/Thesis_Integrated_Checkpoint_1405-07-16_FA.pdf`, dated 2026-10-08 (16 content pages plus cover). Text extracted with `pdftotext -layout` and Unicode-normalized for Persian reading.
- Checkpoint SHA-256: `3e8e972b49c53f40a79a69a012abe452f5e5d12d8b787f382580df1c16b51111`.
- Older references inspected: `../Docs/thesis_implementable_items.md`; older memory navigation notes. Current checkpoint supersedes mandatory SPN, old SLA-dependent revenue, and earlier inspection-only restriction.
- Python: `/home/mehrab/venv/virne-venv/bin/python` (existing environment). System Python lacks torch; existing environment lacks pytest.

## Exact execution path audit

| Concern | Current files / classes / functions |
|---|---|
| CLI / Hydra | `main.py` -> `virne/cli.py:run`; `virne/configs/main.yaml`, `learning.yaml`, `p_net_setting/default.yaml`, `v_sim_setting/default.yaml` |
| Config derivation / output | `virne/utils/config.py:add_simulation_into_config`, `get_run_id_dir`; Hydra `job.chdir=true`, default working directory `results`; save root relative to that directory |
| Registry | `virne/solver/base_solver.py:SolverRegistry`; `virne/solver/__init__.py` imports registrations |
| Baseline | `virne/solver/learning/reinforcement_learning/dual_gnn_solver.py`: dynamically registered `ppo_dual_gat+`, `DualGnnInstanceEnv`; `solver_maker.py:make_solver_class` |
| Policy | `rl_core/policy_builder.py:PolicyBuilder.build_dual_gat_policy`; `rl_policy/dual_gnn_policy.py:BiGatActorCritic`; tensor conversion `rl_core/tensor_convertor.py:obs_as_tensor_for_dual_gnn` |
| Instance episode / action | `rl_core/instance_rl_environment.py:InstanceRLEnv`, `JointPRStepInstanceRLEnv.step`; `rl_enviroment_base.py:RLBaseEnv` generates masks; action selects physical node for next ranked virtual node |
| Features | `rl_core/feature_constructor.py:FeatureConstructorRegistry`, `PNetVNetFeatureConstructor`; DualGnn env changes feature constructor to `p_net_v_net` |
| Rewards | `rl_core/reward_calculator.py:RewardCalculatorRegistry`; original `vanilla`, `fixed_intermediate`, `adaptive_intermediate`, `gradual_intermediate` (config comment lists stale names) |
| Feasibility / deployment | `virne/core/controller/controller.py:Controller`; `virne/core/environment.py:SolutionStepEnvironment`, rollback, departure/release |
| Solution / accounting | `virne/core/solution.py:Solution`; `virne/core/counter.py:Counter.count_solution`, `count_partial_solution`, `summary_records`; resource schema 2 retained |
| Recorder | `virne/core/recorder.py:Recorder.count`, `count_state`, `save_records`, `save_summary` |
| Substrate | `virne/network/physical_network.py:PhysicalNetwork.from_setting`, `generate_attrs_data`; `network/attribute/` factories |
| Workload | `virne/network/virtual_network_request_simulator.py:VirtualNetworkRequestSimulator.from_setting`, `renew`, `_renew_v_nets`, `_renew_events` |
| Offline / online | Both in `virne/system/base_system.py`, classes `OfflineSystem`, `OnlineSystem`. `system/offline_system.py` is empty. Offline replaces substrate with 75%-scaled shuffled resources after each request; online preserves residual resources and releases departures |
| Training / inference | `BaseSystem.ready`; `rl_core/instance_agent.py:InstanceAgent.learn_singly`, `learn_with_instance`, `solve`; each request is one rollout episode |
| Save/load | `rl_core/rl_solver.py:RLSolver.save_model`, `load_model`: strict state dict; checkpoint feature schema 1/2; missing metadata falls back to legacy schema 1 |

## Baseline run evidence (before thesis edits)

Tiny DEV run: 12-node Waxman substrate; four requests of 2–3 nodes; CPU; seed 17; one training epoch; embedding_dim=16. This is functional verification, not scientific validation. Existing `nn.num_gnn_layers` is not passed by dual-GAT builder (recorded rather than changed).

Original training command:

```bash
/home/mehrab/venv/virne-venv/bin/python main.py solver.solver_name=ppo_dual_gat+ training.num_train_epochs=1 training.use_cuda=false training.num_workers=1 training.if_use_random_training_seed=false training.save_interval=1 training.eval_interval=1 v_sim_setting.num_v_nets=4 v_sim_setting.v_net_size.low=2 v_sim_setting.v_net_size.high=3 p_net_setting.topology.num_nodes=12 system.if_offline_system=true experiment.seed=17 experiment.num_simulations=1 experiment.run_id=thesis_baseline_before_train_20261009 logger.backends='[console,file]' nn.embedding_dim=16 nn.num_gnn_layers=1 rl.target_steps=4 training.batch_size=4 rl.repeat_times=1 hydra.run.dir=./results/thesis_smoke_before
```

- Training runtime reported: 3.25 seconds (excludes startup imports).
- Training acceptance=1.0, mean R2C=0.7574487895716946, LT-R2C=0.676056338028169.
- Post-training offline inference acceptance=0.75, mean R2C=0.4369530368028901, LT-R2C=0.5412844036697247; one route failure, no crash.
- Model: `results/thesis_smoke_before/virne/ppo_dual_gat+/thesis_baseline_before_train_20261009/models/model.pkl`.
- Per-run resolved config, record CSVs, log and training CSV saved under that run directory.
- Separate baseline test uses same topology/workload/architecture, `training.num_train_epochs=0`, absolute `solver.pretrained_model_path` above, run ID `thesis_baseline_before_test_20261009`, same Hydra output directory. Full stdout/stderr initially `/tmp/thesis_baseline_before_test.log`.

## Scientific boundaries / unresolved choices

- No final alpha/beta, component-rate distribution, feature subset, reward or objective winner selected. Test/smoke parameters must remain explicitly DEV.
- SLA skeleton shows uniform class probabilities; expose explicit probability configuration and label smoke assignments.
- Checkpoint defines potential shaping form but no potential Φ: interface may be added; execution must reject absent definition.
- RA-NRM/RA-GRC combine function must be configured/validated, not silently selected.
- Cand-RL K, exact policy/state encoding/reward constants/SPN/redundancy details are explicitly reconstructed or unknown; consult `../cand_rl_forensic_reconstruction.md` before comparator work.
- No failure events, mandatory redundancy, migration/recovery, SPN dependency or LT-R2C training objective.

## Phases 1–4 report

- Files inspected: physical/request generation, attribute factories, graph persistence and solution representation.
- Files changed: `virne/reliability/{model,evaluator,__init__}.py`, `virne/configs/{main,thesis}.yaml`, `virne/utils/config.py`, `virne/network/{physical_network,virtual_network_request_simulator}.py`; `tests/thesis/test_reliability.py`.
- Implemented explicit heterogeneous rate arrays, caller-configured uniform distributions, mandatory nodes / toggleable links, deterministic isolated generators, SLA metadata, shared analytic evaluator, shortfall/compliance.
- Command: `/home/mehrab/venv/virne-venv/bin/python -m unittest tests.thesis.test_reliability -v`. 11 tests passed (0.045 seconds).
- Interpretation recorded: UNIQUE used nodes include physical routing transit nodes as well as VNF hosts. Transit node availability is required in both modes; link factors only in node+link mode.
- No scientific component distribution selected. `config_id` identifies provider inputs; DEV smoke rates are separately labeled.

## Phases 5–8 report

- Files inspected: original rewards, feature constructor/ObservationHandler, policy dimension functions, dual-GAT edge attributes, checkpoint IO, Recorder and Counter.
- Files changed: `virne/reliability/{objective,service,features,metrics}.py`, `rl_core/{thesis_reward,instance_rl_environment,feature_constructor,policy_builder,rl_solver}.py`, `core/{counter,recorder}.py`. Original reward calculator file unchanged.
- Tests added: `tests/thesis/test_objectives_rewards.py`, `test_integration.py`.
- Command: `/home/mehrab/venv/virne-venv/bin/python -m unittest discover -s tests/thesis -v`. 16 tests passed before pytest integration tests were added.
- Smoke command: `PROFILE=smoke RUN_ID=core_first_node_link MODE=node_and_link scripts/thesis/train_offline.sh`. Passed full train + inference; model saved at `results/thesis_smoke/ppo_dual_gat+/core_first_node_link/models/model.pkl`. Exact executed CLI and resolved config saved in that run directory.
- Features: configured order on physical nodes; request/partial/budget context also broadcast to virtual nodes; component features on physical edges in node+link mode. Dimensions come from same feature provider. Signature metadata rejects same-width reordered/different features as well as different modes.
- Gradual reward delegates original 0.1 progress factor with partial joint utility substituted for partial R2C; negative utility remains negative. Fixed/adaptive/failure rewards delegate original semantics.
- Failed attempts record null delivered availability/shortfall/compliance/utility and false SLA status; no fabricated delivered service. SLA satisfaction summaries condition on accepted service; all-arrival SLA service rate reported separately.
- Potential strategy registered but raises a clear BLOCKED error: exact Phi and terminal convention absent from checkpoint.
- Next: integration tests, disabled regression comparison, online lifecycle smoke, comparator implementation where scientific definitions are available.

## Phase 9 report — deterministic compatibility

- Files inspected/changed: checkpoint signature code, dataset-path identity, generator disable cleanup, regression scripts and tests. Added `scripts/thesis/compare_baseline.py`, `test.sh`.
- After-change baseline test command: original separate test command above, `experiment.run_id=thesis_baseline_after_test_20261009`, `reliability.enabled=false`, `hydra.run.dir=./results/thesis_smoke_after`; same original model, seed, topology, workload and architecture.
- Comparison command: `/home/mehrab/venv/virne-venv/bin/python scripts/thesis/compare_baseline.py results/thesis_smoke_before/virne/ppo_dual_gat+/thesis_baseline_before_test_20261009 results/thesis_smoke_after/virne/ppo_dual_gat+/thesis_baseline_after_test_20261009 --output results/thesis_smoke_after/baseline_comparison.json`.
- Result: all 8 event rows and 61 original record columns matched exactly; no excluded record columns were necessary. Acceptance=0.75, mean R2C=0.4369530368028901, LT-R2C=0.5412844036697247 unchanged.
- Training command: `PROFILE=baseline RUN_ID=baseline_after_train scripts/thesis/train_offline.sh nn.num_gnn_layers=1`. Compared saved `policy` state tensors against the original pre-edit training checkpoint using `torch.equal`: **all 92 tensors bitwise equal**. Evidence: `results/thesis_smoke_after/training_comparison.json`.
- Full test command: `scripts/thesis/test.sh`; **307 passed in 31.56 seconds**. Subsequent targeted thesis/CLI tests after explicit RA normalization config: **36 passed in 29.08 seconds**. Two extra cache/validation tests added: latest thesis suite **32 passed in 25.60 seconds**. A final objective input/overflow guard was checked by the five objective/reward tests.
- Initial broader test attempt found Python 3.10 lacks `tomllib`; process-local alias to `tomli` resolves this. An initial CLI subprocess check selected stale installed Virne outside the checkout; setting repository-root PYTHONPATH resolved it and all six CLI tests pass. No product-code workaround or existing-test edits made.
- Existing environment: Python **3.10.22**, torch **2.6.0+cu124**, Hydra **1.3.7**, PyG **2.8.0.post1**. Project declares >=3.12 and different pinned dependencies. This run verifies the existing environment only; no full dependency reinstall was attempted.
- Optional generator configurations receive SHA-256-derived dataset path suffixes. Changing rates/mode/SLA probabilities cannot collide with baseline or other provider profiles. Disabled mode removes derived provider settings and keeps original cache paths.

## Phases 10–11 report — comparator implementation

- Inspected: original heuristic registrations/rankers/mapper; local RA-EVNE paper equations 12–18 and pseudocode; `../cand_rl_forensic_reconstruction.md` recovered Cand/state/reward evidence; original PPO/searcher/environment interfaces.
- Changed: `virne/solver/heuristic/reliability_rank.py`, `virne/solver/learning/reinforcement_learning/cand_rl_solver.py`, solver registration imports, `virne/reliability/paper9.py`, comparator config sections, `tests/thesis/test_comparators.py`.
- Added `docs/thesis_comparator_provenance.md` with published/reconstructed/unknown/validation-open components and exact DEV commands.
- Original `ffd_rank`, `nrm_rank`, `grc_rank`, `rw_rank` criteria preserved; all four ran online with shared final reliability measurement.
- `ra_nrm`, `ra_grc` expose explicit combine and normalization choices. `ra_evne_original` ports published age/environment ranking, while `ra_evne_a` substitutes shared analytic availability. Age/history data are static inputs; no realized failure simulation added. First-placement LS convention, channel names, disconnected/tie handling and Virne routing boundaries explicitly documented.
- Cand main profile has separate `CandidateSelector`, `CandRLObservationBuilder`, `CandRLPolicy` (independent MLP actor/critic), `PlacementExecutor`, PPO solver and joint reward adapter. K, widths, channels/capacities, endpoint fallback, hop/delay ranking, padding mask and no-redundancy selection all explicit. It uses the same substrate/workload/evaluator/objective/accounting/metrics, with its own decision representation.
- First Cand train succeeded but greedy inference incorrectly inherited a physical-node action mask. Fixed by overriding candidate-mask generation for both training and inference; automated test now exercises both. Failed first run retained as `cand_main_dev` with `completed=false` in manifest. Rerun `cand_main_dev_fixed` and online saved-model reload `cand_main_reload` passed.
- `Paper9Reward` implements the recovered printed symbolic formula and -10/-100 penalties with explicit constants; no unannounced sign correction. `paper9_sanity` refuses execution until its unresolved accounting/reconstruction inputs are addressed. Main Cand profile is a **thesis reconstruction**, not an exact published reproduction.
- Limitation: comparator main currently supports explicit `redundancy=none`; full Paper-9 replication action, backup/ingress/egress accounting, SPN semantics and clock-spanning reproduction episodes are not implemented. The shared thesis graph-VNR pipeline is documented as an adaptation.

## Smoke execution and artifacts

- Executed objective sweep: `scripts/thesis/smoke_all_objectives.sh` (weighted_sum, additive, multiplicative, geometric), all passed.
- Executed reward sweep: `scripts/thesis/smoke_all_rewards.sh` (joint_terminal, joint_fixed, joint_adaptive, joint_gradual), all passed.
- Node+link model train/reload: `core_first_node_link`, `core_reload_node_link`, both passed; inference acceptance=1.0, LT-R2C=0.9795918367, accepted-service SLA satisfaction=0.75.
- Online same node+link model: `core_online_node_link`, passed; acceptance=0.75, LT-R2C=0.5108359133, accepted-service SLA satisfaction=1/3. Online training: `PROFILE=smoke RUN_ID=core_online_train scripts/thesis/train_online.sh`, passed.
- Cand DEV main train/inference: `cand_main_dev_fixed`, passed; offline acceptance=0.75, LT-R2C=0.8398576512. Online saved-model inference `cand_main_reload`, passed. No algorithm-quality conclusion follows from these tiny samples.
- Four original and four RA heuristic online smokes passed. Exact overrides/commands/configs/outputs are retained within each run directory.
- Collection command: `PYTHONPATH=/home/mehrab/Workspaces/Codex/Thesis/virne /home/mehrab/venv/virne-venv/bin/python scripts/thesis/collect_results.py results/thesis_smoke`.
- Manifest: `results/thesis_smoke/run_manifest.csv` and `.json`, 24 scripted run directories: **23 completed**, one initial failed Cand inference retained. Original before/after direct baseline runs additionally live in `results/thesis_smoke_before` and `results/thesis_smoke_after`.
- Per-run runtimes include Python startup and are roughly 15–22 seconds; actual model training reported roughly 0.6–3.5 seconds for these tiny budgets. No expensive training/factorial scientific experiment launched.
- Canonical instructions: `docs/thesis_running.md`. Scripts expose solver/seed/epochs/objective/reward/mode, reject existing run IDs, use absolute model paths and save exact command/resolved configuration.
- Verification: shell syntax checks, Python compilation checks and `git diff --check` passed. Initial unrelated `Solvability-Evaluation.sh` remains untracked and untouched. No commits, rebase, reset or history rewrite performed.

## Remaining implementation blockers and next work

1. **BLOCKED potential reward**: exact Phi(s) and terminal-potential convention need a scientific definition. Only interface/config/explicit rejection implemented.
2. **BLOCKED Paper-9 reproduction/sanity execution**: define/recover operational cost/energy metadata and constants/gate, replication/backup/routing semantics, SPN/failure dependency assumptions and reproduction episode boundary. Symbolic reward and executable main comparator do not resolve those gaps.
3. **Validation-open**: final component distributions, SLA probabilities, alpha/beta, feature subset, objective/reward winner, heuristic combine/normalization and Cand K/widths. Current DEV values are not final selections.
4. **Environment limitation**: declared Python >=3.12 / pinned lockfile dependency environment has not been validated. Existing Python 3.10 environment was sufficient for the reported checks.
5. Next authorized independent work is complete for this implementation pass. Scientific validation should use separately approved/frozen workload and search profiles; final-test data must remain held out. No superiority claims made.
