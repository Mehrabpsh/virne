# PPO stability safeguards and validation

The November 11 GPU run completed, but both PPO variants developed extreme
stored-versus-recomputed action log-probability differences and variable internal
evaluation performance. Original PPO seed 31's failure-free internal evaluation
peaked at 86.5% at epoch 17 and ended at 52.2%. This deterioration therefore cannot
be attributed solely to failure/repair evaluation. The historical run had no
matched failure-free held-out reference.

The local investigation distinguishes established defects from remaining
uncertainty. A 64-request native probe with the same 100-node topology generator,
seed 31, 128-dimensional three-layer DualGAT and original optimizer settings
executed 210 optimizer steps. Before updates, stored/single/batched log
probabilities agreed within 8.2e-6. Its largest measured full-rollout KL was 0.02285.
It did not reproduce the late-training catastrophe; the historical checkpoints
and complete optimizer telemetry were not available locally, and SSH timed out.
The exact chain that produced each historical model is not established.

## Implementation changes

- `InstanceAgent.learn_singly` restores training mode at each epoch. Validation
  switches the policy to eval mode and previously left later training epochs
  there. Historical dropout=0 and batch_norm=false mean this defect alone does
  not explain the reported collapse.
- RL initialization now honors configured advantage normalization, gradient
  clipping and maximum gradient norm instead of replacing them with hardcoded
  defaults. Historical defaults matched those hardcoded values; this correction
  makes future stability settings effective rather than explaining that run.
- Nonfinite loss or gradients abort before an optimizer step.
- Optional `rl.kl_guard` checks an exact categorical KL over the full recorded
  rollout, including the same action masks used by the behavior policy. A
  pre-update consistency check rejects stale/mutated observations or mismatched
  rollout log probabilities before updating.
- After each proposed optimizer step, the guard checks mean KL(old||new) against
  `rl.target_kl`. Oversized/nonfinite proposals restore both policy and Adam
  moments/counters, then retry the same clipped gradient at a smaller learning
  rate. Exhausting retries restores the complete pre-step state and stops reuse
  of that rollout. Original learning rates are restored after the transaction.
  Adam timesteps advance only for accepted updates. The reference distribution
  remains fixed across all minibatches of one rollout.
- The guard defaults off to retain existing update behavior. It currently
  requires single-worker deterministic forwards (dropout=0, batch_norm=false).
  No observation dimensions, action candidates, mapping rules, SLA calculation,
  rewards, or reliability objectives are changed.

PPO surrogate clipping is not a hard bound on policy divergence, as also
described in the [official Stable Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters).
The guard adds an explicit acceptance condition on recorded rollout states; it
is not a guarantee over unseen states or a theorem of convergence/SLA compliance.

## Configuration

Native Virne settings:

```yaml
rl:
  learning_rate:
    actor: 0.0001
    critic: 0.001
  target_kl: 0.02
  kl_guard:
    enabled: true
    max_backtracks: 4
    backtrack_factor: 0.5
    rollout_logprob_atol: 0.001
    rollout_distribution_kl_atol: 0.000001
```

The actor learning rate and KL target are provisional stability settings, not
claimed optimal values. In experiment-v2 these can be set per PPO method:

```yaml
overrides:
  rl.learning_rate.actor: 0.0001
  rl.target_kl: 0.02
  rl.kl_guard.enabled: true
```

The full paired-evaluation example applies these overrides to original PPO and
reliability-aware PPO. Cand-RL retains its existing optimizer settings. Other
examples retain their settings unless explicitly overridden. Physical rates,
workload, objectives, candidate semantics and the 99-job matrix are unchanged.

## Diagnostics and evidence

`logs/training_info.csv` adds accepted-update exact KL, maximum state KL,
backtrack count and accepted step scale. TensorBoard exposes corresponding
`ppo/*` scalars. Training `runtime_system.json` records the guard settings,
effective optimizer learning rates, accepted update count, rejected proposals,
skipped rollout batches and maximum accepted KL. Sampled pre-step log-probability
differences remain available but must not be mistaken for exact distribution KL.

The controlled guarded 64-request probe retained all 210 updates, rejected and
reduced 7 oversized proposals, and bounded accepted mean KL at 0.0199872 under a
0.02 target. It used actor LR 0.001 in both runs to isolate the guard's effect;
the full example's lower actor LR was separately exercised in native training
tests. These are engineering checks, not an independent scientific comparison.

Regression tests exercise oversized steps, complete rejection, initialized Adam
state restoration, cumulative rollout KL, stale log probabilities, masked
actions, nonfinite gradients with clipping enabled/disabled, and train/eval mode
restoration. Native training tests for both PPO variants verify real accepted
updates and checkpoint changes. Paired synthetic execution verifies reload,
failure-free/live separation, reporting and no evaluation optimizer updates.

A new GPU training run is still required to assess long-horizon stability and
held-out performance. Preserve the historical results; do not select checkpoints
or hyperparameters using held-out test outcomes. These safeguards are not a
claim that the 50-epoch performance problem has already been eliminated.

A focused first GPU check retains all workload/physical settings and all three
training/evaluation seeds, but uses five epochs and only the two PPO methods:

```bash
python -m scripts.thesis.experiments_v2.run_experiment \
  --config scripts/thesis/experiments_v2/examples/gpu_server.live_failures.full.v2.yaml \
  --methods original_ppo,reliability_ppo \
  --set protocol.training.epochs=5 \
  --set experiment.name=ppo_stability_5epochs \
  --dry-run
```

This plans six training jobs,18 failure-free evaluations and18 live evaluations
(42 jobs). Removing `--dry-run` launches that experiment. No such GPU experiment
was launched during the local diagnosis. Check accepted updates, policy hashes,
maximum accepted KL and validation outcomes before interpreting performance.

## November 13 three-epoch GPU run

The first GPU follow-up used three epochs despite its `ppo_stability_5epochs`
experiment label. Original PPO seed 31 completed 9,508 accepted updates with
maximum accepted rollout mean KL 0.01999821. Seed 37 logged 9,818 accepted
updates before the pre-update consistency check stopped its third epoch:
stored-versus-batched action log-probability error was 0.00119352, above 0.001.
Seed 43, reliability-aware PPO and all held-out evaluations were not reached.
This failed bundle cannot establish comparative evaluation performance.

The archived epoch-1 seed-37 checkpoint has finite parameters. A frozen CPU
probe collected 202 observations over 32 requests without optimizer updates;
its largest single-versus-batched action log-probability difference was
0.0000552535. It did not reproduce the exact CUDA failure. The original archive
contains no exact failing rollout or failure-time policy state, so the cause of
that particular mismatch remains unconfirmed.

The guard now distinguishes a collection-path mismatch from batch-dependent
roundoff. If the initial check fails, it replays every observation individually,
using the same forward shape as collection. It proceeds only if stored action
log probabilities pass the original 0.001 tolerance AND the maximum per-state
KL(single||batch), over complete masked action distributions, is at most
`rollout_distribution_kl_atol` (default 1e-6). The replayed distributions become
the fixed old-policy reference. The optimizer acceptance target stays 0.02.
This is a bounded numerical allowance, not automatic acceptance of every batch
mismatch. Recheck counts and maximum batching discrepancy enter runtime audits.

Unresolved mismatches still abort. They save `logs/ppo_rollout_mismatch.pt` and
an accompanying JSON diagnostic with the failure-time policy/optimizer state,
observations, actions, stored/replayed distributions and precision metadata.
Failed training now also finalizes runtime update counts, policy hashes and
guard statistics instead of retaining the initialization-time zero-update audit.
GPU validation of this new verification path is still required.
