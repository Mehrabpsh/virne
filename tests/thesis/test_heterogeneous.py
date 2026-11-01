"""Heterogeneous assignments, shared scaling, compatibility and effective controls."""
import copy
import json
from unittest.mock import Mock

import networkx as nx
import numpy as np
import pytest
from omegaconf import OmegaConf, open_dict

from virne.network import PhysicalNetwork
from virne.reliability.model import ReliabilityAttributeProvider, component_availability, categorical_rates
from virne.reliability.scaling import LogRiskScaler
from virne.reliability.features import feature_signature
from virne.reliability.diagnostics import availability_upper_bound, classify_attainability
from virne.solver import SolverRegistry
from virne.solver.heuristic.reliability_rank import combine_rank
from virne.solver.learning.reinforcement_learning.cand_rl_solver import CandRLInstanceEnv, CandRLObservationBuilder
from tests.thesis.test_comparators import cand_config, infrastructure
from tests.thesis.test_integration import config_for_smoke


def classes():
    return {'distribution':'categorical','values':[1e-5,5e-5,1e-4,5e-4],
            'probabilities':[.25,.25,.25,.25]}


def test_assignments_are_stable_isolated_and_preserved():
    config={'mode':'node_and_link','nodes':{'lambda_f':classes(),'mu_r':.1},
            'links':{'lambda_f':{'distribution':'uniform','low':1e-5,'high':2e-5},'mu_r':.1}}
    a=nx.path_graph(40);b=nx.Graph();b.add_nodes_from(reversed(list(a)));b.add_edges_from(a.edges)
    np.random.seed(9);state=np.random.get_state()
    ReliabilityAttributeProvider(config).apply(a,seed=101)
    ReliabilityAttributeProvider(config).apply(b,seed=101)
    assert dict(a.nodes(data=True))==dict(b.nodes(data=True))
    assert np.array_equal(state[1],np.random.get_state()[1])
    assert {d['reliability_class'] for _,d in a.nodes(data=True)}==set(range(4))
    for _,d in a.nodes(data=True):
        assert d['lambda_f']==classes()['values'][d['reliability_class']]
        assert d['availability']==component_availability(d['lambda_f'],d['mu_r'])
    changed=copy.deepcopy(config);changed['links']['lambda_f']=.02
    c=nx.path_graph(40);ReliabilityAttributeProvider(changed).apply(c,seed=101)
    assert dict(c.nodes(data=True))==dict(a.nodes(data=True))
    changed=copy.deepcopy(config);changed['nodes']['lambda_f']['probabilities']=[1.,0.,0.,0.]
    d=nx.path_graph(40);ReliabilityAttributeProvider(changed).apply(d,seed=101)
    assert list(d.edges(data=True))==list(a.edges(data=True))


def test_legacy_uniform_draws_unchanged():
    config={'mode':'node_and_link','nodes':{'lambda_f':{'distribution':'uniform','low':.001,'high':.002},'mu_r':.1},
            'links':{'lambda_f':{'distribution':'uniform','low':.003,'high':.004},'mu_r':.1}}
    rng=np.random.default_rng(17);nodes=rng.uniform(.001,.002,4);edges=rng.uniform(.003,.004,3)
    p=nx.path_graph(4);ReliabilityAttributeProvider(config).apply(p,seed=17)
    np.testing.assert_array_equal(nodes,[p.nodes[n]['lambda_f'] for n in p])
    np.testing.assert_array_equal(edges,[d['lambda_f'] for _,_,d in p.edges(data=True)])


@pytest.mark.parametrize('changes',[{'values':[0,1]}, {'values':[1,1]}, {'probabilities':[.2]*4},
                                    {'probabilities':[-.1,.3,.4,.4]}, {'values':[float('nan')]*4}])
def test_invalid_classes(changes):
    with pytest.raises(ValueError):categorical_rates({**classes(),**changes})


def test_fixed_support_scaling_and_homogeneous_ranking():
    scaler=LogRiskScaler({'lambda_f':classes(),'mu_r':.1})
    values=[component_availability(v,.1) for v in classes()['values']]
    np.testing.assert_allclose([scaler.risk(v) for v in values],[0,.0818,.184,1],atol=.001)
    assert combine_rank({0:1.,1:.99},{0:values[-1],1:values[0]},
        'normalized_risk_weighted_sum',.5,'max',scaler)[1]>.5
    homogeneous=LogRiskScaler({'lambda_f':1e-5,'mu_r':.1})
    scores=combine_rank({0:2.,1:1.},{0:values[0],1:values[0]},
        'normalized_risk_weighted_sum',.5,'max',homogeneous)
    assert scores[0]>scores[1]
    with pytest.raises(ValueError):scaler.risk(.5)


def test_persistence_and_checkpoint_scaling_signature(tmp_path):
    config=config_for_smoke(tmp_path)
    config.reliability.nodes.lambda_f=classes()
    config.reliability_features.names=['normalized_log_risk']
    from virne.utils.config import add_simulation_into_config
    add_simulation_into_config(config)
    p=PhysicalNetwork.from_setting(config.p_net_setting,seed=101)
    p.save_dataset(str(tmp_path/'pn'));loaded=PhysicalNetwork.load_dataset(str(tmp_path/'pn'))
    for node in p:
        for key in ('lambda_f','mu_r','availability','lambda_f_class','reliability_class','cpu','max_cpu'):
            assert p.nodes[node][key]==loaded.nodes[node][key]
    signature=feature_signature(config)
    config.reliability.nodes.lambda_f['values'][0]=1e-6
    assert signature!=feature_signature(config)


def test_effective_depth_and_ppo_repeat_config(tmp_path):
    config=config_for_smoke(tmp_path);config.nn.num_gnn_layers=2;config.rl.repeat_times=3
    solver=SolverRegistry.get('ppo_dual_gat+')(Mock(),Mock(),Mock(),Mock(),config)
    assert solver.repeat_times==3
    assert solver.policy.actor.encoder.p_net_encoder.net_gnn.num_layers==2
    assert solver.policy.critic.encoder.v_net_encoder.net_gnn.num_layers==2
    solver.buffer.logprobs=[0]*256;solver.batch_size=128
    assert solver.calculate_update_sample_times()==6


def test_cand_opt_in_representation_and_shared_budget(tmp_path):
    config=cand_config(tmp_path)
    with open_dict(config.cand_rl):
        config.cand_rl.observation_features=['normalized_log_risk','remaining_budget_fraction']
    p,v,controller,recorder,counter=infrastructure(config)
    env=CandRLInstanceEnv(p,v,controller,recorder,counter,Mock(),config)
    obs,_=env.reset()
    assert obs['candidate_x'].shape==(CandRLObservationBuilder.dimension(config.cand_rl),)==(12,)
    assert obs['candidate_x'][-1]==1.
    if not env._no_feasible_action:
        _,_,done,_,_=env.step(int(np.flatnonzero(obs['action_mask'])[0]))
        if not done:
            assert env.get_observation()['candidate_x'][-1]<1.


def test_bounds_do_not_claim_attainability():
    p=nx.path_graph(5);nx.set_node_attributes(p,.9999,'availability');nx.set_edge_attributes(p,.99975,'availability')
    v=nx.path_graph(3);bound=availability_upper_bound(p,v,'node_and_link')
    assert bound==pytest.approx(.9999**3*.99975**2)
    assert classify_attainability(bound,.9999)=='proven_impossible'
    assert classify_attainability(bound,.99)=='unknown'
    assert classify_attainability(bound,.99,True)=='witnessed_attainable'
    with pytest.raises(ValueError):classify_attainability(bound,.9999,True)


def test_live_snapshot_collection_and_tamper_detection(tmp_path,monkeypatch):
    import pandas as pd
    from types import SimpleNamespace
    from scripts.thesis.experiments import common
    from virne.reliability.provenance import capture_inputs
    config=cand_config(tmp_path)
    p,v,controller,recorder,counter=infrastructure(config)
    initial=copy.deepcopy(p)
    sim=SimpleNamespace(v_nets=[v])
    directory=tmp_path/'run';directory.mkdir();(directory/'records').mkdir()
    config.logger.runtime_audit_path=str(directory/'runtime_system.json')
    capture_inputs(config,initial,sim)
    solution=SolverRegistry.get('ffd_rank')(controller,recorder,counter,Mock(),config).solve({'p_net':p,'v_net':v})
    assert solution.result
    row=counter.count_solution(v,solution)
    row.update(event_type=1,p_net_node_resource_utilization=.1,p_net_link_resource_utilization=.1)
    pd.DataFrame([row]).to_csv(directory/'records/final.csv',index=False)
    (directory/'runtime_seconds.txt').write_text('1')
    (directory/'source_commit.txt').write_text('fixture')
    monkeypatch.setattr(common,'generated',lambda *a,**k:(config,copy.deepcopy(initial),sim))
    from virne.core import Counter
    monkeypatch.setattr(Counter,'summary_records',staticmethod(lambda frame:{}))
    metadata={'run_directory':str(directory),'phase':'eval','category':'heuristic','run_id':'fixture'}
    output=common.collect_row(metadata,tmp_path,assessment_beta=1.,assessment_mode='node_and_link')
    assert output['runtime_inputs_verified']
    assert json.loads(output['attainability_summary_json'])
    assert json.loads(output['utility_diagnostics_json'])
    captured=json.loads((directory/'runtime_inputs.json').read_text())
    captured['substrate']['nodes'][0][1]['cpu']+=1
    (directory/'runtime_inputs.json').write_text(json.dumps(captured))
    with pytest.raises(ValueError,match='Live runtime inputs differ'):
        common.collect_row(metadata,tmp_path,assessment_beta=1.,assessment_mode='node_and_link')
