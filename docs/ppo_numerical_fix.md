# PPO numerical failure in the 6 November 2026 run

## Diagnosis

The earlier 2 November report records 83.7% original-PPO acceptance after ten
training epochs. Both reports identify source commit
`3490fa32d85f65e87f1d5b973b494b069eb1c95d`; their intended original-PPO method
configurations match. The later experiment extended training to 50 epochs.
The issue is a numerical bug exposed by the longer training, not evidence that
the original PPO method cannot embed the workload.

Direct CPU replay of the archived checkpoints reproduces the first request's
forbidden-node selections, including the final checkpoint's `[50, 5, 5]` sequence.
The physical-node mask is correct. The old mask replaces forbidden logits with
the finite value `-1e8`, which becomes greater than the allowed action scores:

| Checkpoint | Maximum valid logit at first decision | Old forbidden logit | Old decision |
| --- | ---: | ---: | --- |
| Epoch 9 | about 4,004 | -100,000,000 | Valid |
| Epoch 42 | about 6,830,465 | -100,000,000 | Valid |
| Epoch 43 | -135,020,928 | -100,000,000 | Forbidden |
| Final epoch 49 | about -1,474,690,304 | -100,000,000 | Forbidden |

Two other verified numerical defects explain why large offsets are problematic
even before crossing the masking threshold:

1. `Categorical(logits=...)` normalizes by subtracting a logsumexp. With a large
   common float32 offset, that subtraction loses precision. Rollout and update
   code used different normalization paths. At the actual epoch-42 first
   decision, the old PPO likelihood ratio is approximately **0.83335 without
   any parameter update**, although the ratio must be one.
2. The update path constructed `Categorical(probs=softmax(logits))`. Converting
   probabilities back to log probabilities clamps small probabilities. A valid
   action whose true log probability is approximately -40 is reported as
   approximately -15.94, and its log-probability gradient is clipped away.

There is also a demonstrable spurious gradient with respect to a common logit
offset. For float32 logits `[6830456, 6830457]`, the old negative log-probability
has offset derivative approximately `-0.17034`; the mathematically correct
derivative is zero because adding a constant cannot change the distribution.
Centering the logits gives the correct zero derivative.

These establish the invalid-action mechanism and numerical inconsistencies.
They do not constitute a controlled causal decomposition of every contribution
to the original 50-epoch optimization trajectory.

## Correction

- Forbidden actions receive negative infinity, so no finite valid logit can
  rank below a masked action.
- A shared `categorical_from_logits` helper subtracts each action row's maximum
  before normalization and constructs the distribution directly from logits.
- Rollout collection, PPO action evaluation, and inference use that same
  numerical construction, including temperature.
- Empty action support and non-finite normalizers raise an explicit error.
  Existing environments retain their explicit sentinel action for genuine
  no-feasible-node states.
- Absolute Q values are not centered by the generic masking utility. Only the
  categorical-distribution helper performs centering.

No reward, reliability/SLA formula, candidate rule, architecture, checkpoint
tensor layout, or experimental hyperparameter was changed. These operations
preserve the intended categorical probabilities mathematically; finite-precision
results and corrected training trajectories can differ from the old code.

## Verification

`tests/solver/learning/test_action_numerics.py` covers the archived negative
logits, greedy and sampled selection, rollout/update equality, rare-action log
probabilities and gradients, common-offset invariance, temperature, masked
gradients, empty support, and preservation of absolute Q values.

Before the patch, the first four regression tests all fail. The corrected code
passes all ten numerical regression cases and the complete
repository/thesis/framework suite: **408 passed**.
Exact test output and archived checkpoint replay results are retained under
`../../analysis_results/gpu_calibration_v2_20261106/ppo_diagnosis/` in the parent
Thesis workspace.

The replay command is:

```bash
python -m scripts.thesis.diagnose_ppo_checkpoint \
  --bundle /path/to/archived/bundle \
  --output /path/to/new/diagnostic/output \
  --device cpu --max-requests 50
```

It preserves the original 1,000-request generated inputs and evaluates only the
requested prefix, checking regenerated inputs against the archived runtime
snapshots. It evaluates the epoch-9 and final checkpoints on seeds 100, 110, and
120 without optimization. Every placement action is audited against its mask
and the already used hosts. Saved policy hashes must remain unchanged.

Omit `--max-requests` for complete streams; use `--device cuda` on a CUDA host.
The CPU diagnostic is not a substitute for a complete corrected 50-epoch CUDA
training/evaluation experiment. The initial unbounded CPU attempt was stopped
after eight minutes before completing its first evaluation; its log and source
snapshot are preserved separately. No completed full-stream result is claimed
for that attempt.

## Cand-RL scope

The candidate filtering, distance ranking, endpoints, K, and no-redundancy
semantics remain unchanged. Their observed rejection behavior is a limitation
of the configured comparator, rather than a defect addressed by this patch.
The shared PPO numerical correction applies to Cand-RL's PPO implementation too.

A configurable policy architecture is a useful separate extension. Keep the
current MLP as the default and preserve checkpoint signatures. A CNN can consume
ordered candidate rows with the same information. GCN/GAT variants should encode
an actual physical graph and gather scores/embeddings at the K candidate IDs;
reshaping the existing flat vector alone does not supply graph connectivity.
Graph-based observations may introduce additional information, so document that
change separately from a network-only comparison. All variants should retain
the same candidate selector, action mask, reward, and evaluation streams.

No new Cand-RL architecture switch is implemented by this numerical bug fix.
