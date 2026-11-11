"""Runner wiring and real checkpoint-backed evaluation under component outages."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from scripts.thesis.experiments.common import ROOT
from scripts.thesis.experiments_v2.config.resolver import resolve
from scripts.thesis.experiments_v2.config.schema import ConfigError
from scripts.thesis.experiments_v2.execution.planner import jobs, final_config
from scripts.thesis.experiments_v2.analysis.report import result_interpretations
from scripts.thesis.experiments_v2.analysis.validate import fairness_errors

EXAMPLE = ROOT / 'scripts/thesis/experiments_v2/examples/local_dev.yaml'


def resolved(method='original_ppo'):
    return resolve(EXAMPLE, ['protocol.training.online=true',
        'protocol.evaluation.live_failures.enabled=true',
        'environment.workload.num_requests=6', 'environment.workload.arrival_rate=0.2',
        'environment.reliability.nodes.lambda_f=0.04',
        'environment.reliability.nodes.mu_r=0.2',
        'environment.reliability.links.lambda_f=0.02',
        'environment.reliability.links.mu_r=0.2'], subset=method)[0]


def test_only_evaluation_gets_live_failures(tmp_path):
    r = resolved()
    train = next(j for j in jobs(r) if j['phase'] == 'train')
    evaluation = next(j for j in jobs(r) if j['phase'] == 'eval')
    tc = final_config(r, train, tmp_path, 'cpu')
    ec = final_config(r, evaluation, tmp_path, 'cpu')
    assert not tc['live_failures']['enabled']
    assert ec['live_failures']['enabled'] and ec['training']['num_train_epochs'] == 0
    assert not ec['reliability']['enabled']  # ordinary baseline observation/reward unchanged


def test_disabled_mode_rejects_unmeasured_live_metric():
    with pytest.raises(ConfigError, match='Observed failure metrics require'):
        resolve(EXAMPLE, ['metrics=[acceptance_rate,observed_service_availability]'])


def test_trajectory_mismatch_is_a_fairness_error():
    frame = pd.DataFrame({'eval_seed': [100,100], 'input_hash': ['same','same'],
                          'failure_trace_sha256': ['first','second']})
    assert 'trajectory fingerprint' in fairness_errors(frame)[0]


def test_interpretation_does_not_treat_no_admissions_as_perfect_availability():
    rows = pd.DataFrame({'method': ['ppo'], 'success_count': [0], 'v_net_count': [10],
        'sla_service_rate_all_arrivals': [0.], 'observed_service_availability': [None],
        'admitted_service_time': [0.], 'total_service_downtime': [0.], 'component_failure_count': [0]})
    text = result_interpretations(rows)[0]['interpretation']
    assert 'undefined' in text and 'does not establish resilience' in text


@pytest.mark.parametrize('method', ['original_ppo','reliability_ppo','cand_rl'])
def test_trained_policy_is_frozen_in_live_evaluation(tmp_path, method):
    import torch
    from virne.system import BaseSystem
    from virne.utils.config import add_simulation_into_config
    from virne.reliability.config import validate_reliability_config
    torch.set_num_threads(1)
    r = resolved(method)
    train = next(j for j in jobs(r) if j['phase'] == 'train')
    def system_for(job):
        config = OmegaConf.create(final_config(r, job, tmp_path, 'cpu'))
        validate_reliability_config(config)
        add_simulation_into_config(config)
        return BaseSystem.from_config(config)
    training = system_for(train)
    try:
        training.run()
    finally:
        training.logger.close()
    evaluation = next(j for j in jobs(r) if j['phase'] == 'eval')
    system = system_for(evaluation)
    system.solver.load_model(system.config.solver.pretrained_model_path)
    before = {k: v.clone() for k,v in system.solver.policy.state_dict().items()}
    updates = system.solver.update_time
    try:
        system.run()
        assert system.solver.update_time == updates
        for key, value in before.items():
            torch.testing.assert_close(system.solver.policy.state_dict()[key], value, rtol=0, atol=0)
        directory = tmp_path / evaluation['relative_directory']
        measured = json.loads((directory / 'live_failure_summary.json').read_text())
        assert measured['component_failure_count'] > 0
        observed = pd.read_csv(directory / 'observed_services.csv')
        assert len(observed) == 6 and observed.completed.all()
        assert system.env.recorder.state['inservice_count'] == 0
        assert measured['requested_service_time'] == pytest.approx(sum(v.lifetime for v in system.env.v_net_simulator.v_nets))
        audit = json.loads((directory / 'runtime_system.json').read_text())
        assert not audit['training_requested'] and audit.get('optimizer_updates', 0) == 0
        assert not (directory / 'models/model.pkl').exists()
    finally:
        system.logger.close()

    if method == 'original_ppo':
        from scripts.thesis.evaluate_live_failures import evaluate
        root = tmp_path / 'standalone'
        saved = tmp_path / train['relative_directory']
        result = evaluate(saved / 'config.yaml', saved / 'models/model.pkl', root, 103)
        proof = json.loads((root / 'frozen_policy_verification.json').read_text())
        assert proof['policy_sha256_before'] == proof['policy_sha256_after']
        assert proof['optimizer_updates'] == 0 and proof['checkpoint_unchanged']
        assert (root / 'evaluation_report.md').exists()
        predicted = json.loads((result / 'predicted_service_summary.json').read_text())
        assert predicted['arrivals'] == 6
        assert len(pd.read_csv(result / 'predicted_services.csv')) == 6
        with pytest.raises(FileExistsError):
            evaluate(saved / 'config.yaml', saved / 'models/model.pkl', root, 103)
