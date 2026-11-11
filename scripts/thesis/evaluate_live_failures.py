"""Evaluate an existing learned checkpoint with live outages, without retraining.

Requires the saved per-training-run config.yaml, not an experiment-v2 input YAML.
Writes a new output root and never changes the supplied checkpoint/config.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from omegaconf import OmegaConf


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def evaluate(saved_config, checkpoint, output, seed, initial_state='all_up', seed_offset=1000003):
    from virne.system import BaseSystem
    from virne.reliability.config import validate_reliability_config
    from virne.utils.config import add_simulation_into_config
    saved_config, checkpoint, output = map(lambda p: Path(p).resolve(), (saved_config, checkpoint, output))
    if not saved_config.is_file() or not checkpoint.is_file():
        raise FileNotFoundError('Both the saved run config and checkpoint must exist')
    checkpoint_hash = file_hash(checkpoint)
    config = OmegaConf.load(saved_config)
    OmegaConf.set_struct(config, False)
    config.experiment.update(seed=seed, save_root_dir=str(output), run_id=f'eval_live_seed_{seed}',
                             num_simulations=1, if_load_p_net=False, if_load_v_nets=False,
                             if_save_config=True)
    config.training.update(num_train_epochs=0, num_workers=1, distributed_training=False, seed=None)
    config.system.update(if_offline_system=False, if_changeable_v_nets=False, if_time_window=False)
    config.live_failures = {'enabled': True, 'initial_state': initial_state,
                            'seed_offset': seed_offset, 'event_limit': 1000000}
    config.solver.pretrained_model_path = str(checkpoint)
    directory = output / config.solver.solver_name / config.experiment.run_id
    config.logger.runtime_audit_path = str(directory / 'runtime_system.json')
    config.logger.tensorboard_dir = str(directory / 'tensorboard')
    config.recorder.if_save_records = True
    validate_reliability_config(config)
    add_simulation_into_config(config)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(saved_config, output / 'source_config.yaml')
    source_root = Path(__file__).resolve().parents[2]
    metadata = {'source_config': str(saved_config), 'source_config_sha256': file_hash(saved_config),
                'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
                'evaluation_seed': seed, 'command': sys.argv,
                'source_sha256': {str(p.relative_to(source_root)): file_hash(p)
                                  for p in (source_root / 'virne').rglob('*.py')}}
    metadata['source_sha256'][str(Path(__file__).resolve().relative_to(source_root))] = file_hash(__file__)
    metadata['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip()
    (output / 'evaluation_provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    system = BaseSystem.from_config(config)
    def policy_hash():
        digest = hashlib.sha256()
        for name, value in system.solver.policy.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    try:
        if not hasattr(system.solver, 'policy'):
            raise ValueError('This command requires a learned solver and its trained checkpoint')
        system.solver.load_model(str(checkpoint))
        before = policy_hash()
        updates = system.solver.update_time
        system.run()
        after = policy_hash()
        if before != after or system.solver.update_time != updates:
            raise RuntimeError('Policy or optimizer updates changed during frozen evaluation')
        if file_hash(checkpoint) != checkpoint_hash:
            raise RuntimeError('Source checkpoint changed during evaluation')
        evidence = {'policy_sha256_before': before, 'policy_sha256_after': after,
                    'optimizer_updates': system.solver.update_time - updates,
                    'checkpoint_unchanged': True, 'run_directory': str(directory),
                    'device': str(system.solver.device)}
        (output / 'frozen_policy_verification.json').write_text(json.dumps(evidence, indent=2) + '\n')
        write_assessment(system, directory, output)
        (output / 'COMPLETE.txt').write_text('Frozen-checkpoint live failure evaluation completed; no retraining.\n')
        return directory
    except Exception as exc:
        (output / 'FAILED.json').write_text(json.dumps({'error': str(exc)}) + '\n')
        raise
    finally:
        system.logger.close()


def write_assessment(system, directory, output):
    """Common post-hoc analytic assessment, including the unaware baseline."""
    import statistics
    from virne.reliability.model import ReliabilityAttributeProvider
    from virne.reliability.evaluator import ReliabilityEvaluator, SLAShortfallCalculator
    world = copy.deepcopy(system.env.init_p_net)
    ReliabilityAttributeProvider(system.config.reliability).apply(world, seed=system.config.experiment.seed)
    evaluator = ReliabilityEvaluator(system.config.reliability.mode)
    process = system.env._live_failures
    rows = []
    for record in system.env.recorder.memory:
        if record['event_type'] != 1:
            continue
        row = {'v_net_id': int(record['v_net_id']), 'accepted': bool(record['result']),
               'availability_requirement': process.requirements[int(record['v_net_id'])],
               'predicted_availability': None, 'predicted_shortfall': None, 'predicted_sla_satisfied': False}
        if row['accepted']:
            # beta=1 only satisfies the shared calculator API; utility is neither
            # retained nor used. Availability, shortfall and satisfaction do not depend on beta.
            result = SLAShortfallCalculator.calculate(row['availability_requirement'],
                        evaluator.availability(world, record), 1.)
            row.update(predicted_availability=result.delivered_availability,
                       predicted_shortfall=result.sla_shortfall, predicted_sla_satisfied=result.sla_satisfied)
        rows.append(row)
    accepted = [r for r in rows if r['accepted']]
    compliant = sum(r['predicted_sla_satisfied'] for r in accepted)
    summary = {'arrivals': len(rows), 'accepted': len(accepted), 'predicted_compliant': compliant,
               'predicted_sla_service_rate_all_arrivals': compliant / len(rows) if rows else None,
               'predicted_sla_satisfaction_rate': compliant / len(accepted) if accepted else None,
               'mean_predicted_availability': statistics.mean(r['predicted_availability'] for r in accepted) if accepted else None,
               'mean_predicted_shortfall': statistics.mean(r['predicted_shortfall'] for r in accepted) if accepted else None}
    with (directory / 'predicted_services.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['v_net_id'])
        writer.writeheader()
        writer.writerows(rows)
    (directory / 'predicted_service_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    observed = process.summary()
    text = ('# Frozen-checkpoint evaluation with live component failures\n\n'
            f'Arrivals: {len(rows)}; accepted: {len(accepted)}; predicted SLA compliant: {compliant}. '
            'Predicted compliance assesses the analytic steady-state model for the placements actually chosen under live failures.\n\n'
            f"Observed fulfilled service-time: {observed['fulfilled_service_time']:.6g} / "
            f"{observed['requested_service_time']:.6g} requested time units. "
            f"Downtime: {observed['total_service_downtime']:.6g} admitted service-time units.\n\n"
            f"Component failure transitions: {observed['component_failure_count']}; repair transitions: {observed['component_repair_count']}. "
            'These are physical component events; rejected requests are a separate outcome.\n\n'
            'Policy weights and source checkpoint were verified unchanged; no optimizer updates occurred. '
            'Reservations persist through outages; service waits for repair; no rerouting or replicas are introduced. '
            'Observed lifetime uptime and predicted steady-state availability have different measurement windows. '
            'One trace cannot establish superiority or a long-run SLA guarantee.\n')
    if observed['component_failure_count'] == 0:
        text+='\nNo component failure transitions occurred; this sample provides no evidence of robustness to newly occurring outages.\n'
    (output / 'evaluation_report.md').write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Saved training-run config.yaml')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output root; must not exist')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--initial-state', choices=['all_up','stationary'], default='all_up')
    parser.add_argument('--seed-offset', type=int, default=1000003)
    args = parser.parse_args(argv)
    print(evaluate(args.config,args.checkpoint,args.output,args.seed,args.initial_state,args.seed_offset))


if __name__ == '__main__':
    main()
