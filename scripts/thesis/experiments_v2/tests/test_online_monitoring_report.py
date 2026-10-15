"""Mode contracts, native TensorBoard routing and real online PPO/lifecycle proof."""
import json
from pathlib import Path
import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf
from scripts.thesis.experiments.common import ROOT
from scripts.thesis.experiments_v2.config.resolver import resolve
from scripts.thesis.experiments_v2.config.schema import ConfigError
from scripts.thesis.experiments_v2.execution.planner import jobs,final_config,summary,tensorboard_path
from scripts.thesis.experiments_v2.analysis.validate import mode_errors
from scripts.thesis.experiments_v2.analysis.report import regenerate,render_html

EXAMPLE=ROOT/'scripts/thesis/experiments_v2/examples/local_dev.yaml'


def resolved(*sets):
    return resolve(EXAMPLE,list(sets),subset='reliability_ppo,nrm')[0]


@pytest.mark.parametrize('online',[True,False])
def test_training_mode_planned_and_final(tmp_path,online):
    r=resolved('protocol.training.online='+str(online).lower())
    j=jobs(r)[0];config=final_config(r,j,tmp_path,'cpu')
    mode='OnlineSystem' if online else 'OfflineSystem'
    assert config['system']['if_offline_system'] is not online
    assert summary(r)['Training system']==mode
    audit={'system':mode,'training_completed':True}
    assert not mode_errors(r,j,OmegaConf.create(config),audit)
    audit['system']='OfflineSystem' if online else 'OnlineSystem'
    assert 'actual runtime' in mode_errors(r,j,OmegaConf.create(config),audit)[0]
    config['system']['if_offline_system']=online
    assert 'final config' in mode_errors(r,j,OmegaConf.create(config),audit)[0]


def test_evaluation_mode_and_preflight(tmp_path):
    r=resolved('protocol.training.online=true');j=next(j for j in jobs(r) if j['phase']=='eval')
    c=final_config(r,j,tmp_path,'cpu')
    assert not c['system']['if_offline_system'] and c['training']['num_train_epochs']==0
    assert summary(r)['Evaluation system']=='OnlineSystem'
    with pytest.raises(ConfigError,match='resets and scales substrate'):
        resolved('protocol.evaluation.online=false')


def test_legacy_default_and_override():
    data=yaml.safe_load(EXAMPLE.read_text());data['protocol']['training'].pop('online',None)
    data['runtime'].pop('tensorboard',None)
    r=resolve(EXAMPLE,document=data)[0]
    assert not r['experiment']['protocol']['training']['online']
    assert not r['experiment']['runtime']['tensorboard']['enabled']
    assert resolve(EXAMPLE,['protocol.training.online=true','runtime.tensorboard.enabled=true'],document=data)[0]['experiment']['protocol']['training']['online']


def test_unique_routes_under_shared_root(tmp_path):
    r=resolved('runtime.tensorboard.enabled=true')
    routes=[Path(final_config(r,j,tmp_path,'cpu')['logger']['tensorboard_dir']) for j in jobs(r)]
    assert len(routes)==len(set(routes))
    assert all(p.is_relative_to(tmp_path/'tensorboard') for p in routes)
    assert all(j['run_id'] in str(p) and j['method'] in str(p) for j,p in zip(jobs(r),routes))


def test_real_online_ppo_updates_checkpoint_and_lifecycle(tmp_path):
    import torch
    from virne.system import BaseSystem
    from virne.utils.config import add_simulation_into_config
    from virne.reliability.config import validate_reliability_config
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    torch.set_num_threads(1)
    r=resolved('protocol.training.online=true','runtime.tensorboard.enabled=true',
               'environment.workload.num_requests=6','environment.workload.arrival_rate=0.2')
    j=jobs(r)[0];config=OmegaConf.create(final_config(r,j,tmp_path,'cpu'))
    validate_reliability_config(config);add_simulation_into_config(config)
    system=BaseSystem.from_config(config)
    assert type(system).__name__=='OnlineSystem'
    original_step=system.env.step;steps=[];instances=[];resources=[];returns=[]
    original_learn=system.solver.learn_with_instance
    original_merge=system.solver.merge_instance_experience
    def learn(instance):
        # Each SFC sees the current residual substrate and has its own buffer.
        resources.append(system.counter.calculate_sum_network_resource(instance['p_net']))
        solution,buffer,last_value=original_learn(instance)
        instances.append((instance['v_net'].id,buffer.size(),len(buffer.returns)))
        return solution,buffer,last_value
    def merge(instance,solution,buffer,last_value):
        output=original_merge(instance,solution,buffer,last_value)
        returns.append((buffer.size(),len(buffer.returns),buffer.dones[-1]))
        return output
    def step(solution):
        output=original_step(solution)
        if system.logger.runtime_phase=='train':steps.append(dict(output[3]))
        return output
    system.env.step=step;system.solver.learn_with_instance=learn;system.solver.merge_instance_experience=merge
    try:
        system.run()
        audit=json.loads(Path(config.logger.runtime_audit_path).read_text())
        assert audit['system']=='OnlineSystem' and audit['training_completed']
        assert audit['optimizer_updates']>0
        assert audit['policy_sha256_before']!=audit['policy_sha256_after']
        checkpoint=Path(system.solver.model_dir)/'model.pkl'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['policy'] and saved['optimizer']['state']
        assert max(float(s['step']) for s in saved['optimizer']['state'].values())>0
        assert len(instances)==6 and len({i[0] for i in instances})==6
        assert all(size in (2,3) for _,size,_ in instances)
        # The existing learner may discard failed SFCs; every retained buffer's
        # returns end at that SFC boundary before global batching.
        assert len(returns)==6 and any(count for _,count,_ in returns)
        assert all(size==count and terminated for size,count,terminated in returns if count)
        assert min(resources[1:])<resources[0]
        assert max(s['inservice_count'] for s in steps)>=2
        assert max(s['p_net_node_resource_utilization'] for s in steps)>0
        # All departures are processed, and final resources are restored.
        final=system.env.recorder.memory[-1]
        assert final['inservice_count']==0
        assert final['p_net_node_resource_utilization']==pytest.approx(0)
        assert final['p_net_link_resource_utilization']==pytest.approx(0)
        events=EventAccumulator(str(tensorboard_path(tmp_path,j))).Reload()
        assert len(events.Scalars('train/episode_return'))==6
        assert events.Scalars('ppo/total_loss')
        assert events.Scalars('system/active_requests')
    finally:system.logger.close()


def test_report_regeneration_reads_frozen_bundle_only(tmp_path,monkeypatch):
    from scripts.thesis.experiments_v2.execution import launcher
    monkeypatch.setattr(launcher,'launch',lambda *a,**k:pytest.fail('report must never execute jobs'))
    r=resolved();s=summary(r);bundle=tmp_path/'bundle';bundle.mkdir();(bundle/'reports').mkdir();(bundle/'aggregated').mkdir()
    (bundle/'resolved_config.yaml').write_text(yaml.safe_dump(r))
    (bundle/'execution_summary.yaml').write_text(yaml.safe_dump(s))
    (bundle/'source_commit.txt').write_text('frozen-commit')
    (bundle/'hardware.json').write_text('{}');(bundle/'manifest.json').write_text('[]')
    (bundle/'reports/validation_report.json').write_text(json.dumps({'passed':True,'warnings':[]}))
    metrics=r['experiment']['metrics']
    pd.DataFrame([{'label':'<unsafe>','method':'nrm',**{m+'_mean':.5 for m in metrics}}]).to_csv(bundle/'aggregated/results_summary.csv',index=False)
    (bundle/'COMPLETE.txt').write_text('completed')
    before=(bundle/'aggregated/results_summary.csv').read_bytes()
    path=regenerate(bundle)
    text=path.read_text()
    assert all(title in text for title in ['Experiment Overview','Training Diagnostics','Reliability / SLA','Artifact Index'])
    assert '&lt;unsafe&gt;' in text and '<unsafe>' not in text
    assert (bundle/'reports/experiment_report.json').is_file()
    path.unlink();assert regenerate(bundle).is_file()
    assert (bundle/'aggregated/results_summary.csv').read_bytes()==before
