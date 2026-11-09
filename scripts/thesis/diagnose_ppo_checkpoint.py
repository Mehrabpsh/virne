"""Replay archived PPO numerics and evaluate checkpoints without training.

Run from the repository root with the usual Virne Python environment:
python -m scripts.thesis.diagnose_ppo_checkpoint --bundle PATH --output NEW_PATH

The archived bundle is read-only. Output includes exact resolved replay configs,
legacy versus fixed numerical probes, per-checkpoint metrics and input checks.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from virne.system.base_system import BaseSystem
from virne.solver.learning.utils import apply_mask_to_logit, categorical_from_logits
from virne.utils.dataset import set_seed


def policy_digest(policy):
    digest = hashlib.sha256()
    for name, value in policy.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configured_system(bundle, output, seed, device='cpu'):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    job = next(j for j in manifest if j['method'] == 'original_ppo' and j['phase'] == 'eval' and j['seed'] == seed)
    config = OmegaConf.load(bundle / job['relative_directory'] / 'config.yaml')
    # Scientific environment, features, architecture, reward and action rules
    # are identical. Only device, logging/output and training execution change.
    config.training.use_cuda = device == 'cuda'
    config.training.num_train_epochs = 0
    config.logger.backends = []
    config.logger.verbose = -1
    config.logger.runtime_audit_path = None
    config.experiment.if_save_config = False
    config.experiment.save_root_dir = str(output)
    config.experiment.run_id = f'seed_{seed}'
    config.recorder.if_save_records = False
    config.recorder.if_temp_save_records = False
    set_seed(seed)
    system = BaseSystem.from_config(config)
    OmegaConf.save(config, output / f'replay_config_seed_{seed}.yaml')
    archived = json.loads((bundle / job['relative_directory'] / 'runtime_inputs.json').read_text())
    # Verify generation did not drift across server/local dependency versions.
    initial = system.env.reset(seed=seed)
    p_net = initial['p_net']
    for node, attributes in archived['substrate']['nodes']:
        for name in ('cpu', 'max_cpu'):
            assert p_net.nodes[node][name] == attributes[name]
    for u, v, attributes in archived['substrate']['edges']:
        assert p_net.edges[u, v]['bw'] == attributes['bw']
    assert set(p_net.edges) == {(u, v) for u, v, _ in archived['substrate']['edges']}
    for generated, recorded in zip(system.env.v_net_simulator.v_nets, archived['requests'], strict=True):
        assert generated.id == recorded['id']
        assert np.isclose(generated.arrival_time, recorded['arrival_time'], rtol=0, atol=1e-10)
        assert np.isclose(generated.lifetime, recorded['lifetime'], rtol=0, atol=1e-10)
        for node, attributes in recorded['nodes']:
            assert generated.nodes[node]['cpu'] == attributes['cpu']
        assert set(generated.edges) == {(u, v) for u, v, _ in recorded['edges']}
        for u, v, attributes in recorded['edges']:
            assert generated.edges[u, v]['bw'] == attributes['bw']
    return system


def probe(system, checkpoint):
    system.solver.load_model(str(checkpoint))
    system.solver.eval()
    instance = system.env.reset(seed=system.config.experiment.seed)
    results = []
    for mode in ('legacy', 'fixed'):
        env = system.solver.InstanceEnv(copy.deepcopy(instance['p_net']), instance['v_net'],
                                       system.controller, system.recorder, system.counter,
                                       system.logger, system.config)
        observation, _ = env.reset()
        steps = []
        for index in range(instance['v_net'].num_nodes):
            tensor = system.solver.preprocess_obs(observation, system.solver.device)
            with torch.no_grad():
                logits = system.solver.policy.act(tensor)
            mask = tensor['action_mask'].bool()
            legacy = torch.where(mask, logits, torch.full_like(logits, -1e8))
            masked = apply_mask_to_logit(logits, mask)
            legacy_rollout = torch.distributions.Categorical(logits=legacy)
            legacy_update = torch.distributions.Categorical(probs=legacy.softmax(-1))
            stable = categorical_from_logits(masked)
            action = (legacy if mode == 'legacy' else masked).argmax(-1)
            log_ratio = (legacy_update.log_prob(action) - legacy_rollout.log_prob(action)).item()
            ratio = float(np.exp(np.float64(log_ratio))) if log_ratio < 700 else None
            row = {'step': index, 'action': action.item(), 'mask_valid': bool(mask[0, action.item()]),
                   'valid_logits_min': logits[mask].min().item(), 'valid_logits_max': logits[mask].max().item(),
                   'legacy_mask_value': -1e8,
                   'legacy_rollout_log_prob': legacy_rollout.log_prob(action).item(),
                   'legacy_update_log_prob': legacy_update.log_prob(action).item(),
                   'legacy_log_ratio_without_weight_update': log_ratio,
                   'legacy_ratio_without_weight_update': ratio,
                   'legacy_ratio_overflows_float64': ratio is None}
            if row['mask_valid']:
                row['fixed_log_prob'] = stable.log_prob(action).item()
                with torch.no_grad():
                    _, update_log_prob, _ = system.solver.evaluate_actions(tensor, action)
                row['fixed_ratio_without_weight_update'] = (update_log_prob - stable.log_prob(action)).exp().item()
            steps.append(row)
            observation, _, terminated, truncated, _ = env.step(action.item())
            if terminated or truncated:
                break
        results.append({'checkpoint': checkpoint.name, 'selection': mode, 'steps': steps,
                        'embedding_succeeded': bool(env.solution['result'])})
    return results


def evaluate(system, checkpoint, max_requests=None):
    system.solver.load_model(str(checkpoint))
    system.solver.eval()
    before = policy_digest(system.solver.policy)
    trace = {'placement_steps': 0, 'invalid_mask_selections': 0, 'reused_host_selections': 0}
    original_env = system.solver.InstanceEnv

    class AuditedEnv(original_env):
        def generate_action_mask(self):
            self.last_mask = super().generate_action_mask()
            return self.last_mask

        def step(self, action):
            trace['placement_steps'] += 1
            trace['invalid_mask_selections'] += int(not self.last_mask[int(action)])
            if not self._no_feasible_action:
                physical = self.action_to_p_node_id(action)
                trace['reused_host_selections'] += int(physical in self.selected_p_net_nodes)
            return super().step(action)

    system.solver.InstanceEnv = AuditedEnv
    try:
        start = time.monotonic()
        instance = system.env.reset(seed=system.config.experiment.seed)
        while True:
            solution = system.solver.solve(instance)
            instance, _, done, info = system.env.step(solution)
            if info['v_net_count'] % 10 == 0:
                print(f'{checkpoint.name} seed {system.config.experiment.seed}: '
                      f"{info['v_net_count']} arrivals, {info['success_count']} accepted", flush=True)
            if done or (max_requests is not None and info['v_net_count'] >= max_requests):
                break
        summary = system.env.summary_records()
        result = {'checkpoint': checkpoint.name, 'eval_seed': system.config.experiment.seed,
                  'device': str(system.solver.device), 'runtime_seconds': time.monotonic() - start,
                  'evaluated_arrivals': int(info['v_net_count']), 'complete_stream': bool(done),
                  'optimizer_updates': system.solver.update_time,
                  'policy_sha256': before, 'policy_unchanged': before == policy_digest(system.solver.policy),
                  **trace, **{key: float(summary[key]) for key in ['acceptance_rate', 'avg_r2c_ratio',
                       'long_term_r2c_ratio', 'success_count', 'place_failure_count', 'route_failure_count']}}
        assert result['policy_unchanged'] and trace['invalid_mask_selections'] == trace['reused_host_selections'] == 0
        return result
    finally:
        system.solver.InstanceEnv = original_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[100, 110, 120])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--max-requests', type=int, default=None,
                        help='Evaluate only this prefix of each original stream; no input regeneration changes.')
    parser.add_argument('--probe-checkpoints', nargs='+', default=['model-9.pkl', 'model-42.pkl', 'model-43.pkl', 'model.pkl'])
    parser.add_argument('--evaluate-checkpoints', nargs='*', default=['model-9.pkl', 'model.pkl'])
    args = parser.parse_args()
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error('--max-requests must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('--device=cuda requires an available CUDA device')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    training = next(j for j in manifest if j['method'] == 'original_ppo' and j['phase'] == 'train')
    model_dir = args.bundle / training['relative_directory'] / 'models'
    provenance = {'command': list(__import__('sys').argv), 'source_commit': (args.bundle / 'source_commit.txt').read_text().strip(),
                  'python': __import__('sys').version, 'torch': torch.__version__, 'numpy': np.__version__,
                  'device': args.device, 'training_performed': False, 'verified_input_seeds': [],
                  'max_requests': args.max_requests,
                  'diagnostic_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'checkpoint_sha256': {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
                                        for name in set(args.probe_checkpoints + args.evaluate_checkpoints)}}
    probes, results = [], []
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for index, seed in enumerate(args.seeds):
        system = configured_system(args.bundle, args.output, seed, args.device)
        provenance['verified_input_seeds'].append(seed)
        if index == 0:
            for name in args.probe_checkpoints:
                probes.extend(probe(system, model_dir / name))
            (args.output / 'numerical_probes.json').write_text(json.dumps(probes, indent=2, allow_nan=False) + '\n')
        for name in args.evaluate_checkpoints:
            result = evaluate(system, model_dir / name, args.max_requests)
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.output / 'checkpoint_evaluations.json').write_text(json.dumps(results, indent=2) + '\n')
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
