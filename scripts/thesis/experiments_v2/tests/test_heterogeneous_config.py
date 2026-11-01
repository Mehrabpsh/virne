"""No-launch calibration planning and Hydra composition for categorical rates."""
import copy
import json
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir

from scripts.thesis.experiments.common import ROOT
from scripts.thesis.experiments_v2.config.resolver import resolve
from scripts.thesis.experiments_v2.config.schema import ConfigError
from scripts.thesis.experiments_v2.execution.launcher import flatten, hydra_value
from scripts.thesis.experiments_v2.calibrate import plan_calibration, select
from tests.thesis.test_heterogeneous import classes

GPU=ROOT/'scripts/thesis/experiments_v2/examples/gpu_server.yaml'


def test_gpu_categorical_configuration_hydra_and_baseline():
    r,_=resolve(GPU)
    baseline=r['selected_methods']['original_ppo']['virne_config']
    assert baseline['rl']['reward_calculator']['name']=='fixed_intermediate'
    assert not baseline['reliability']['enabled']
    assert r['experiment']['environment']['sla']['probabilities']==[.4,.3,.2,.1]
    c=r['selected_methods']['reliability_ppo']['virne_config']
    args=[key+'='+hydra_value(value) for key,value in flatten(c)]
    assert any(arg.startswith('reliability.nodes.lambda_f={') for arg in args)
    with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'virne/configs')):
        actual=compose(config_name='main',overrides=args)
    assert actual.reliability.nodes.lambda_f.distribution=='categorical'
    assert actual.rl.reward_calculator.name=='joint_fixed'
    assert any('structurally unattainable' in warning for warning in r['warnings'])


def test_named_calibration_scenarios_are_atomic_and_beta_preserved():
    _,_,data,points=plan_calibration(GPU,[])
    assert len(points)==2
    for point in points:
        assert point['reliability_scenario']=='proposed_node_classes'
        assert point['selected']['environment.reliability']['nodes']['lambda_f']==classes()
        assert point['resolved']['experiment']['methods']['reliability_ppo']['objective']['beta']==data['methods']['reliability_ppo']['objective']['beta']


def test_no_proven_impossible_criterion_is_distinct_from_class_coverage():
    criteria=yaml.safe_load(GPU.read_text())['calibration']['criteria']
    point={'point_id':'x','acceptance_rate':.5,'sla_satisfaction_rate':.5,'peak_utilization':.4,
           'class_order_penalty':0.,'class_coverage':4,'expected_class_count':4,
           'proven_impossible_fraction':.1}
    assert not select([point],criteria)['qualifies']
    criteria['require_no_proven_impossible']=False
    assert select([point],criteria)['qualifies']


def test_scenario_and_reliability_grid_cannot_conflict(tmp_path):
    data=yaml.safe_load(GPU.read_text())
    data['calibration']['search_space']['environment.reliability.nodes.mu_r']=[.1,.2]
    file=tmp_path/'conflict.yaml';file.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError,match='cannot be crossed'):plan_calibration(file,[])
