"""Contracts at configuration, planning, statistics and execution boundaries."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from scripts.thesis.experiments.common import digest, ROOT
from scripts.thesis.experiments_v2.config.schema import ConfigError, parse_assignments, set_path
from scripts.thesis.experiments_v2.config.resolver import resolve
from scripts.thesis.experiments_v2.execution.capabilities import inspect_solver
from scripts.thesis.experiments_v2.execution.planner import jobs, final_config, summary
from scripts.thesis.experiments_v2.analysis.aggregate import tables, paired_differences
from scripts.thesis.experiments_v2.run_experiment import main
from scripts.thesis.experiments_v2.calibrate import plan_calibration, select

EXAMPLE=ROOT/'scripts/thesis/experiments_v2/examples/local_dev.yaml'


def document():return yaml.safe_load(EXAMPLE.read_text())


def resolved(data=None,sets=(),raw=(),subset=None):
    return resolve(EXAMPLE,sets,raw,subset,document=data)[0]


@pytest.mark.parametrize('text,value',[('x=200',200),('x=0.4',.4),('x=false',False),('x=[101,103]',[101,103]),
                                        ('x=null',None),('x={low: 2, high: 3}',{'low':2,'high':3}),('x=hello','hello')])
def test_typed_yaml_overrides(text,value):assert parse_assignments([text])==[('x',value)]


def test_resolver_input_immutable_and_method_identity():
    before=EXAMPLE.read_bytes();r=resolved(sets=['methods.reliability_ppo.objective.beta=0.4'])
    assert EXAMPLE.read_bytes()==before
    methods=r['selected_methods']
    assert methods['original_ppo']['solver']==methods['reliability_ppo']['solver']=='ppo_dual_gat+'
    assert methods['reliability_ppo']['virne_config']['joint_objective']['beta']==.4
    # YAML anchors must not leak a targeted override into other treatments.
    assert methods['cand_rl']['virne_config']['joint_objective']['beta']!=.4


def test_precedence_and_correct_rl_learning_rate_mapping():
    d=document();d['methods']['reliability_ppo']['overrides']={'rl.gamma':.91}
    r=resolved(d,sets=['learning_defaults.learning_rate.actor=0.0002'],raw=['rl.gamma=0.995'])
    assert all(m['virne_config']['rl']['gamma']==.995 for m in r['selected_methods'].values())
    assert r['selected_methods']['original_ppo']['virne_config']['rl']['learning_rate']['actor']==.0002
    assert r['selected_methods']['original_ppo']['virne_config']['training']['learning_rate']==.001


@pytest.mark.parametrize('path',['p_net_setting.topology.num_nodes','p_net_setting.node_attrs_setting.0.low',
    'v_sim_setting.num_v_nets','v_sim_setting.arrival_rate.rate','v_sim_setting.lifetime.scale',
    'v_sim_setting.node_attrs_setting.0.low','reliability.nodes.lambda_f','sla.probabilities'])
def test_protected_method_override_rejected(path):
    d=document();d['methods']['original_ppo']['overrides']={path:1}
    with pytest.raises(ConfigError,match='protected environment'):resolved(d)


@pytest.mark.parametrize('path',['experiment.seed','experiment.save_root_dir','experiment.run_id',
                                  'solver.pretrained_model_path','training.num_train_epochs','system.if_offline_system','hydra.run.dir'])
def test_runner_owned_raw_override_rejected(path):
    with pytest.raises(ConfigError,match='runner-owned'):resolved(raw=[path+'=1'])


def test_global_environment_override_shared_and_derived_history_rebuilt():
    r=resolved(raw=['p_net_setting.topology.num_nodes=14','v_sim_setting.arrival_rate.rate=0.1'])
    assert all(m['virne_config']['p_net_setting']['topology']['num_nodes']==14 for m in r['selected_methods'].values())
    assert len(r['selected_methods']['ra_evne_original']['virne_config']['ra_evne']['ages'])==14
    assert r['resolved_environment']['v_sim_setting']['arrival_rate']['rate']==.1


def test_calibrated_override_warning_for_experiment_and_raw():
    d=document();d['calibration_result']={'selected':{'environment.workload.arrival_rate':.02}}
    r=resolved(d,sets=['environment.workload.arrival_rate=0.2'])
    assert any('CALIBRATED PARAMETER OVERRIDDEN' in w for w in r['warnings'])
    assert r['calibrated_overrides'][0]['calibrated']==.02
    assert r['calibrated_overrides'][0]['runtime']==.2
    assert resolved(d,raw=['v_sim_setting.arrival_rate.rate=0.3'])['calibrated_overrides']


def test_generic_registry_method_and_capability_contract():
    d=document();d['methods']['custom_order']={'enabled':True,'solver':'order_rank','kind':'auto','overrides':{'solver.k_shortest':2}}
    r=resolved(d,subset='custom_order')
    assert r['selected_methods']['custom_order']['capabilities']['evaluation_only']
    assert len(jobs(r))==2
    assert inspect_solver('mcts')['kind']=='heuristic'
    with pytest.raises(ConfigError,match='unsupported learned execution contract'):inspect_solver('mcts','learned')
    with pytest.raises(ConfigError,match='Unknown Virne solver'):inspect_solver('not_a_registered_solver')
    with pytest.raises(ConfigError,match='untrained model'):inspect_solver('ppo_dual_gat+','heuristic')


def test_job_matrix_subset_layout_and_runner_fields(tmp_path):
    r=resolved(sets=['protocol.training.seeds=[31,37]'],subset='original_ppo,reliability_ppo,nrm')
    plan=jobs(r)
    assert len(plan)==4+8+2
    assert len({j['run_id'] for j in plan})==len(plan)
    assert all(j['phase']=='eval' for j in plan if j['method']=='nrm')
    eval_job=next(j for j in plan if j['phase']=='eval' and j['method']=='original_ppo')
    c=final_config(r,eval_job,tmp_path,'cpu')
    assert c['training']['num_train_epochs']==0 and not c['system']['if_offline_system']
    assert 'methods/original_ppo/train/' in c['solver']['pretrained_model_path']
    assert c['experiment']['seed']==eval_job['eval_seed']
    assert 'methods/original_ppo/eval/train_31/' in eval_job['relative_directory']


@pytest.mark.parametrize('sets',[['protocol.training.seeds=[101]'],['protocol.evaluation.seeds=[11]'],
                                ['protocol.training.seeds=[31,31]'],['protocol.training.epochs=null']])
def test_seed_and_budget_validation(sets):
    with pytest.raises(ConfigError):resolved(sets=sets)


def test_subset_errors_and_unknown_keys_fail_preflight():
    for subset in ('missing','nrm,nrm'):
        with pytest.raises(ConfigError):resolved(subset=subset)
    with pytest.raises(ConfigError):resolved(sets=['methods.nrm.enabled=false'],subset='nrm')
    with pytest.raises(ConfigError,match='Unknown configuration path'):resolved(raw=['rl.nonexistent=1'])
    with pytest.raises(ConfigError,match='Unknown configuration path'):resolved(sets=['environment.workload.typo=1'])
    d=document();d['methods']['cand_rl']['endpoints']={}
    with pytest.raises(ConfigError,match='Cand'):resolved(d)
    d=document();d['methods']['reliability_ppo']['objective']['beta']=None
    with pytest.raises(ConfigError,match='beta'):resolved(d)


def test_baseline_contract_preserved():
    r=resolved();c=r['selected_methods']['original_ppo']['virne_config']
    assert not c['reliability']['enabled'] and not c['sla']['enabled']
    assert c['reliability_features']=={'enabled':False,'names':[]}
    assert c['rl']['reward_calculator']['name']=='fixed_intermediate'
    assert c['joint_objective']=={'name':None,'alpha':None,'beta':None}
    assert r['resolved_environment']['reliability']['enabled']


def test_fixed_rate_adapter_and_no_invented_scale():
    r=resolved(sets=['environment.reliability.nodes.mu_r={distribution: fixed, value: 0.1}'])
    assert r['resolved_environment']['reliability']['nodes']['mu_r']==.1
    with pytest.raises(ConfigError):resolved(sets=['environment.reliability.nodes.lambda_f={scale: 2}'])


def test_crossed_hierarchical_statistics_and_pairing():
    rows=[]
    for train,values in [(31,[1.,3.]),(37,[5.,7.])]:
        for seed,value in zip([101,103],values):rows.append({'method':'learned','label':'Learned','category':'learned','train_seed':train,'eval_seed':seed,'metric':value})
    for seed,value in [(101,2.),(103,4.)]:rows.append({'method':'heuristic','label':'Heuristic','category':'heuristic','train_seed':None,'eval_seed':seed,'metric':value})
    train,evaluation,summary=tables(pd.DataFrame(rows),['metric'])
    assert train.metric.tolist()==[2.,6.]
    learned=summary.set_index('method').loc['learned']
    assert learned.metric_mean==4
    assert np.isclose(learned.metric_train_std,np.std([2,6],ddof=1))
    assert np.isclose(learned.metric_eval_std,np.std([3,5],ddof=1))
    heuristic=summary.set_index('method').loc['heuristic']
    assert pd.isna(heuristic.metric_train_std)
    assert paired_differences(evaluation,['metric']).metric.tolist()==[1.,1.]


@pytest.mark.parametrize('component',['topology','resources','reliability','requests','arrival','lifetime','sla'])
def test_fairness_fingerprint_detects_any_exogenous_change(component):
    original={name:[1,2] for name in ('topology','resources','reliability','requests','arrival','lifetime','sla')}
    changed=copy.deepcopy(original);changed[component][0]=3
    assert digest(original)!=digest(changed)


def test_dry_run_has_no_outputs_or_launches(tmp_path,monkeypatch,capsys):
    import scripts.thesis.experiments_v2.run_experiment as module
    monkeypatch.setattr(module,'execute',lambda *a,**k:pytest.fail('dry-run executed jobs'))
    output=tmp_path/'must_not_exist'
    result=main(['--config',str(EXAMPLE),'--dry-run','--output',str(output),'--methods','nrm'])
    assert result['job_counts']['total']==2 and not result['calibration_executed']
    assert not output.exists()
    assert 'selected_device' in capsys.readouterr().out


def test_calibration_rejects_method_search_and_learned_methods(tmp_path):
    d=document();d['calibration']['search_space']={'methods.reliability_ppo.objective.beta':[.1,.2]}
    file=tmp_path/'draft.yaml';file.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError,match='only environment'):plan_calibration(file,[])
    d=document();d['calibration']['methods']=['original_ppo'];file.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError,match='cannot train or evaluate learned'):plan_calibration(file,[])


def test_calibration_dry_plan_preserves_method_beta_and_search_paths():
    r,original,d,points=plan_calibration(EXAMPLE,[])
    assert len(points)==4
    for p in points:
        assert set(p['selected'])==set(d['calibration']['search_space'])
        assert p['resolved']['experiment']['methods']['reliability_ppo']['objective']['beta']==d['methods']['reliability_ppo']['objective']['beta']


def test_calibration_selection_transparent_and_provisional():
    criteria=document()['calibration']['criteria']
    common={'acceptance_rate':.5,'sla_satisfaction_rate':.5,'peak_utilization':.4,'class_order_penalty':0,'class_coverage':4,'expected_class_count':4}
    selected=select([dict(common,point_id='grid_002'),dict(common,point_id='grid_001')],criteria)
    assert selected['point_id']=='grid_001' and selected['qualifies']
    assert not select([dict(common,point_id='grid_001',class_coverage=2)],criteria)['qualifies']


def test_hierarchical_tables_stable_after_csv_roundtrip(tmp_path):
    frame=pd.DataFrame([{'method':'nrm','label':'NRM','category':'heuristic','train_seed':None,'eval_seed':101,
                         'metric':.5,'objective':None,'beta':None,'model_path':'','model_reload_observed':None}])
    expected=tables(frame,['metric'])
    file=tmp_path/'long.csv';frame.to_csv(file,index=False)
    actual=tables(pd.read_csv(file),['metric'])
    for a,b in zip(actual,expected):pd.testing.assert_frame_equal(a,b,check_dtype=False)
    assert pd.isna(actual[2].iloc[0].metric_eval_std)


def test_unknown_nested_raw_mapping_rejected():
    with pytest.raises(ConfigError,match='Unknown configuration path'):
        resolved(raw=['nn={invented_field: 3}'])


def test_launcher_serializes_resource_mappings_for_hydra():
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from scripts.thesis.experiments_v2.execution.launcher import hydra_value
    value=[{'name':'cpu','low':10,'high':30,'generative':True,'dtype':'int'}]
    parsed=OverridesParser.create().parse_override('v_sim_setting.node_attrs_setting='+hydra_value(value))
    assert parsed.value()==value


@pytest.mark.parametrize('adopt',[False,True])
def test_calibration_export_native_yaml_and_beta_adoption_is_explicit(tmp_path,monkeypatch,adopt):
    import scripts.thesis.experiments_v2.calibrate as module
    d=document();d['calibration']['search_space']={'environment.workload.arrival_rate':[.02]}
    d['calibration']['seeds']=[11];d['calibration']['adopt_beta_for']=['reliability_ppo'] if adopt else []
    source=tmp_path/'source.yaml';source.write_text(yaml.safe_dump(d));original=source.read_bytes()
    bundle=tmp_path/'calibration_bundle';bundle.mkdir();(bundle/'logs').mkdir();(bundle/'reports').mkdir()
    (bundle/'source_commit.txt').write_text('frozen-source\n')
    monkeypatch.setattr(module,'new_bundle',lambda *a,**k:bundle)
    monkeypatch.setattr(module,'install_signals',lambda:None)
    launched=[]
    def launch_stub(bundle,resolved,job,device):
        assert job['phase']=='calibration' and job['method'] in ('nrm','grc') and device=='cpu'
        launched.append(job)
        directory=tmp_path/job['run_id'];directory.mkdir()
        pd.DataFrame({'sla_shortfall':[1.,2.]}).to_csv(directory/'assessment_arrivals.csv',index=False)
        return {**job,'run_directory':str(directory),'completed':True}
    def collect_stub(row,*a,**k):
        classes={str(level):{'accepted':1,'sla_satisfaction_rate':value} for level,value in zip([.99,.999,.9995,.9999],[1.,.7,.4,0.])}
        return {**row,'acceptance_rate':.5,'sla_satisfaction_rate':.5,'peak_p_net_node_resource_utilization':.4,
                'peak_p_net_link_resource_utilization':.3,'max_inservice_count':2.,'input_hash':'same',
                'sla_class_metrics_json':json.dumps(classes)}
    monkeypatch.setattr(module,'launch',launch_stub);monkeypatch.setattr(module,'collect_row',collect_stub)
    monkeypatch.setattr(module,'evaluation_errors',lambda *a:[]);monkeypatch.setattr(module,'metric_errors',lambda *a:[])
    output=tmp_path/'new.calibrated.yaml'
    module.execute(source,[],output)
    result=yaml.safe_load(output.read_text())
    assert source.read_bytes()==original and len(launched)==2
    assert (bundle/'COMPLETE.txt').exists() and (bundle/'reports/calibration_report.md').exists()
    suggestion=result['calibration_suggestions']['multiplicative_beta_scale']['value']
    original_beta=d['methods']['reliability_ppo']['objective']['beta']
    assert result['methods']['reliability_ppo']['objective']['beta']==(suggestion if adopt else original_beta)
    assert result['methods']['cand_rl']['objective']['beta']==original_beta
    assert result['calibration_result']['source_config_sha256']==__import__('hashlib').sha256(original).hexdigest()


def test_unknown_metric_and_invalid_generator_values_fail_preflight():
    with pytest.raises(ConfigError,match='configured metrics'):resolved(sets=['metrics=[invented_metric]'])
    with pytest.raises(ConfigError,match='topology'):resolved(sets=['environment.substrate.topology.type=imaginary'])
    with pytest.raises(ConfigError,match='Poisson'):resolved(raw=['v_sim_setting.arrival_rate.time_model=discrete'])


def test_actual_fairness_validator_rejects_mismatched_shared_seed():
    from scripts.thesis.experiments_v2.analysis.validate import fairness_errors
    frame=pd.DataFrame({'eval_seed':[101,101,103,103],'input_hash':['a','a','b','b']})
    assert not fairness_errors(frame)
    frame.loc[1,'input_hash']='wrong'
    assert fairness_errors(frame)==['Exogenous input fingerprint mismatch for evaluation seed 101']


def test_node_only_assessment_mode_is_resolved_without_link_features():
    r=resolved(sets=['environment.reliability.mode=node_only'])
    assert r['resolved_environment']['reliability']['mode']=='node_only'
    assert all(m['virne_config']['reliability']['mode']=='node_only' for m in r['selected_methods'].values())
