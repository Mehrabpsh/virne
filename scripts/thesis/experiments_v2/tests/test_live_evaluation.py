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
from scripts.thesis.experiments_v2.analysis.report import result_interpretations, evaluation_description
from scripts.thesis.experiments_v2.analysis.validate import fairness_errors
from scripts.thesis.experiments_v2.analysis.aggregate import tables, paired_differences, paired_modes

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
    evaluations = [j for j in jobs(r) if j['phase'] == 'eval']
    evaluation = next(j for j in evaluations if j['evaluation_mode']=='live_failures')
    tc = final_config(r, train, tmp_path, 'cpu')
    ec = final_config(r, evaluation, tmp_path, 'cpu')
    assert not tc['live_failures']['enabled']
    assert ec['live_failures']['enabled'] and ec['training']['num_train_epochs'] == 0
    assert not ec['reliability']['enabled']  # ordinary baseline observation/reward unchanged
    free=next(j for j in evaluations if j['evaluation_mode']=='failure_free' and j['seed']==evaluation['seed'])
    fc=final_config(r,free,tmp_path,'cpu')
    assert not fc['live_failures']['enabled'] and fc['training']['num_train_epochs']==0
    assert fc['solver']['pretrained_model_path']==ec['solver']['pretrained_model_path']
    assert free['relative_directory']!=evaluation['relative_directory']


def test_disabled_mode_retains_unmeasured_live_metric_without_running_it():
    r=resolve(EXAMPLE, ['metrics=[acceptance_rate,observed_service_availability]'])[0]
    assert any('unmeasured/null' in warning for warning in r['warnings'])
    assert not any(j.get('evaluation_mode')=='live_failures' for j in jobs(r))


def test_trajectory_mismatch_is_a_fairness_error():
    frame = pd.DataFrame({'eval_seed': [100,100], 'input_hash': ['same','same'],
                          'failure_trace_sha256': ['first','second']})
    assert 'trajectory fingerprint' in fairness_errors(frame)[0]


def test_dual_mode_fairness_compares_traces_only_in_live_mode():
    frame=pd.DataFrame({'evaluation_mode':['failure_free','live_failures','live_failures'],
        'eval_seed':[100]*3,'input_hash':['same']*3,
        'failure_trace_sha256':[None,'trace','trace']})
    assert not fairness_errors(frame)
    frame.loc[2,'failure_trace_sha256']='different'
    assert 'trajectory fingerprint' in fairness_errors(frame)[0]
    frame.loc[2,'failure_trace_sha256']='trace'
    frame.loc[0,'input_hash']='different-workload'
    assert 'Exogenous input' in fairness_errors(frame)[0]


def test_modes_are_not_pooled_and_differences_match_models():
    rows=[]
    for mode,offset in [('failure_free',0),('live_failures',-2)]:
        for method,base,kind in [('ppo',10,'learned'),('nrm',4,'heuristic')]:
            for seed in [100,110]:
                for train in ([31,37] if kind=='learned' else [None]):
                    rows.append(dict(evaluation_mode=mode,method=method,label=method,category=kind,
                                     train_seed=train,eval_seed=seed,metric=base+offset))
    frame=pd.DataFrame(rows)
    train,evaluation,summary=tables(frame,['metric'])
    means=summary.set_index(['evaluation_mode','method']).metric_mean
    assert means.loc['failure_free','ppo']==10 and means.loc['live_failures','ppo']==8
    assert len(summary)==4 and len(train)==4
    assert paired_differences(evaluation,['metric']).metric.eq(6).all()
    differences=paired_modes(frame,['metric'])
    assert len(differences)==6 and differences.metric.eq(-2).all()
    assert differences.difference_direction.eq('live_failures_minus_failure_free').all()
    with pytest.raises(ValueError,match='lacks a failure-free pair'):
        paired_modes(frame.loc[~((frame.evaluation_mode=='failure_free') & (frame.method=='nrm'))],['metric'])
    texts=result_interpretations(frame.assign(success_count=1,v_net_count=2,sla_service_rate_all_arrivals=.5))
    assert len(texts)==4 and {x['evaluation_mode'] for x in texts}=={'failure_free','live_failures'}


def test_full_defaults_exclude_evne_and_plan_both_modes(tmp_path):
    config=ROOT/'scripts/thesis/experiments_v2/examples/gpu_server.live_failures.full.v2.yaml'
    r=resolve(config)[0];plan=jobs(r)
    assert len(plan)==99
    assert len([j for j in plan if j['phase']=='train'])==9
    assert len([j for j in plan if j.get('evaluation_mode')=='failure_free'])==45
    assert len([j for j in plan if j.get('evaluation_mode')=='live_failures'])==45
    assert not any(j['solver'].startswith('ra_evne') for j in plan)
    assert len({j['run_id'] for j in plan})==len(plan)
    assert len({j['relative_directory'] for j in plan})==len(plan)
    free=next(j for j in plan if j.get('evaluation_mode')=='failure_free')
    live=next(j for j in plan if j.get('evaluation_mode')=='live_failures')
    a=final_config(r,free,tmp_path,'cpu');b=final_config(r,live,tmp_path,'cpu')
    assert a['solver']['pretrained_model_path']==b['solver']['pretrained_model_path']
    for key in ['p_net_setting','v_sim_setting','reliability','sla','nn','joint_objective']:
        assert a[key]==b[key]
    # Disabling live evaluations keeps the same failure-free jobs.
    r['experiment']['protocol']['evaluation']['live_failures']['enabled']=False
    assert jobs(r)==[j for j in plan if j.get('evaluation_mode')!='live_failures']


def test_live_only_metrics_stay_unmeasured_and_csv_schema_is_stable(tmp_path):
    frame=pd.DataFrame([
        dict(evaluation_mode=mode,method='nrm',label='NRM',category='heuristic',
             train_seed=None,eval_seed=100,acceptance_rate=.5,
             observed_service_availability=None if mode=='failure_free' else .9,
             live_failures_enabled=None if mode=='failure_free' else True)
        for mode in ['failure_free','live_failures']])
    expected=tables(frame,['acceptance_rate','observed_service_availability'])
    frame.to_csv(tmp_path/'results.csv',index=False)
    actual=tables(pd.read_csv(tmp_path/'results.csv'),['acceptance_rate','observed_service_availability'])
    for a,b in zip(actual,expected):pd.testing.assert_frame_equal(a,b,check_dtype=False)
    summary=actual[2].set_index('evaluation_mode')
    assert pd.isna(summary.loc['failure_free','observed_service_availability_mean'])
    assert summary.loc['failure_free','observed_service_availability_count']==0
    assert summary.loc['live_failures','observed_service_availability_mean']==.9
    assert not any(c.startswith('live_failures_enabled_') for c in summary)


def test_interpretation_does_not_treat_no_admissions_as_perfect_availability():
    rows = pd.DataFrame({'method': ['ppo'], 'success_count': [0], 'v_net_count': [10],
        'sla_service_rate_all_arrivals': [0.], 'observed_service_availability': [None],
        'admitted_service_time': [0.], 'total_service_downtime': [0.], 'component_failure_count': [0]})
    text = result_interpretations(rows)[0]['interpretation']
    assert 'undefined' in text and 'does not establish resilience' in text


def test_historical_live_only_bundle_does_not_acquire_a_failure_free_reference():
    r=resolved()
    assert 'legacy bundle used live-only' in evaluation_description(pd.DataFrame(),r)
    rows=pd.DataFrame({'evaluation_mode':['failure_free','live_failures']})
    assert 'matched failure-free' in evaluation_description(rows,r)
    assert 'failure-free held-out evaluations only' in evaluation_description(rows.iloc[:1],r)


@pytest.mark.parametrize('method',['original_ppo','reliability_ppo'])
def test_guarded_native_training_records_real_accepted_steps(tmp_path,method):
    import torch
    from virne.system import BaseSystem
    from virne.utils.config import add_simulation_into_config
    torch.set_num_threads(1)
    r=resolve(EXAMPLE,['protocol.training.online=true','protocol.training.epochs=2',
                      'environment.workload.num_requests=6','environment.workload.arrival_rate=0.2'],
              ['rl.kl_guard.enabled=true','rl.target_kl=0.02','rl.learning_rate.actor=0.0001',
               'rl.max_grad_norm=0.25','rl.norm_advantage=false'],subset=method)[0]
    train=next(j for j in jobs(r) if j['phase']=='train')
    cfg=OmegaConf.create(final_config(r,train,tmp_path,'cpu'))
    add_simulation_into_config(cfg)
    system=BaseSystem.from_config(cfg)
    try:
        assert system.solver.config.rl.max_grad_norm==.25
        assert system.solver.config.rl.norm_advantage is False
        system.run()
        audit=json.loads(Path(cfg.logger.runtime_audit_path).read_text())
        measured=audit['ppo_stability']
        assert audit['optimizer_updates']==measured['accepted_updates']>0
        assert 0<=measured['max_accepted_exact_kl']<=measured['target_kl']
        assert measured['optimizer_learning_rates'][0]==.0001
        assert audit['policy_sha256_before']!=audit['policy_sha256_after']
        checkpoint=torch.load(Path(cfg.logger.runtime_audit_path).parent/'models/model.pkl',weights_only=False)
        assert checkpoint['policy'] and checkpoint['optimizer']['state']
    finally:system.logger.close()


def test_failed_native_training_preserves_completed_updates_and_guard_audit(tmp_path):
    import torch
    from virne.system import BaseSystem
    from virne.utils.config import add_simulation_into_config
    torch.set_num_threads(1)
    r=resolve(EXAMPLE,['protocol.training.online=true','protocol.training.epochs=2',
                      'environment.workload.num_requests=6','environment.workload.arrival_rate=0.2'],
              ['rl.kl_guard.enabled=true','rl.target_kl=0.02'],subset='original_ppo')[0]
    train=next(j for j in jobs(r) if j['phase']=='train')
    cfg=OmegaConf.create(final_config(r,train,tmp_path,'cpu'))
    add_simulation_into_config(cfg)
    system=BaseSystem.from_config(cfg)
    update=system.solver.update
    def fail_after_update():
        update()
        raise RuntimeError('injected failure after real accepted updates')
    system.solver.update=fail_after_update
    try:
        with pytest.raises(RuntimeError,match='injected failure'):
            system.run()
        audit=json.loads(Path(cfg.logger.runtime_audit_path).read_text())
        assert audit['training_completed'] is False
        assert audit['optimizer_updates']==audit['ppo_stability']['accepted_updates']>0
        assert audit['policy_sha256_before']!=audit['policy_sha256_after']
        assert audit['training_error']['type']=='RuntimeError'
    finally:system.logger.close()


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
    initial_inputs=[]
    for mode in ('failure_free','live_failures'):
        evaluation = next(j for j in jobs(r) if j.get('evaluation_mode')==mode)
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
            initial_inputs.append(json.loads((directory/'runtime_inputs.json').read_text()))
            if mode=='live_failures':
                measured = json.loads((directory / 'live_failure_summary.json').read_text())
                assert measured['component_failure_count'] > 0
                observed = pd.read_csv(directory / 'observed_services.csv')
                assert len(observed) == 6 and observed.completed.all()
                assert measured['requested_service_time'] == pytest.approx(sum(v.lifetime for v in system.env.v_net_simulator.v_nets))
            else:
                assert not (directory/'live_failure_summary.json').exists()
            assert system.env.recorder.state['inservice_count'] == 0
            audit = json.loads((directory / 'runtime_system.json').read_text())
            assert not audit['training_requested'] and audit.get('optimizer_updates', 0) == 0
            assert not (directory / 'models/model.pkl').exists()
        finally:
            system.logger.close()
    assert initial_inputs[0]==initial_inputs[1]

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
