"""Contract tests for calibration, baseline isolation, input identity and failure output."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from calibrate import score
from common import clean, digest, read_yaml, settings
from validate_results import run as validate


def profile():
    p=read_yaml(Path(__file__).parent/'profiles/local_dev.yaml')
    p['selected_device']='cpu'
    return p


def test_calibration_penalizes_degeneracy_and_nonmonotonic_classes():
    rules=profile()['calibration']
    good={'acceptance_rate':.5,'sla_satisfaction_rate':.5,'peak_utilization':.4,'class_order_penalty':0}
    assert score(good,rules)==0
    for change in ({'acceptance_rate':1.},{'acceptance_rate':0.},
                   {'sla_satisfaction_rate':1.},{'peak_utilization':0.},
                   {'class_order_penalty':.1}):
        assert score({**good,**change},rules)>score(good,rules)


def test_baseline_keeps_original_reward_and_observation_contract():
    p=profile();point={'arrival_rate':.02,'lambda_f':.00005,'beta':.3}
    d=settings(p,point,baseline=True)
    assert not d['reliability.enabled'] and not d['sla.enabled']
    assert not d['reliability_features.enabled'] and d['reliability_features.names']==[]
    assert d['rl.reward_calculator.name']=='fixed_intermediate'
    assert d['joint_objective.name'] is None and d['joint_objective.beta'] is None
    enabled=settings(p,point)
    assert enabled['joint_objective.beta']==point['beta']
    for key in d:
        if key.startswith(('p_net_setting.','v_sim_setting.')):assert d[key]==enabled[key]


@pytest.mark.parametrize('field',['topology','demands','arrivals','lifetime','sla'])
def test_input_fingerprint_detects_each_kind_of_stream_drift(field):
    a={'topology':[0,1],'demands':[10,20],'arrivals':[1.,2.],'lifetime':[3.,4.],'sla':[.99,.999]}
    b=json.loads(json.dumps(a));b[field][0]+=0.01
    assert digest(a)!=digest(b)
    assert digest(a)==digest(dict(reversed(list(a.items()))))


def test_missing_structural_input_emits_failure_report(tmp_path):
    with pytest.raises(FileNotFoundError):validate(tmp_path)
    report=json.loads((tmp_path/'validation_report.json').read_text())
    assert not report['passed'] and report['hard_errors']
    assert (tmp_path/'reports/validation_report.md').exists()
    assert not (tmp_path/'COMPLETE.txt').exists()


def test_nonfinite_json_is_null():
    assert clean({'missing':float('nan'),'bad':float('inf')})=={'missing':None,'bad':None}
