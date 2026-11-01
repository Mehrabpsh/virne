"""Feature ablations, persistence, workload isolation and online lifecycle."""

import copy
import itertools
from pathlib import Path
import time
from unittest.mock import Mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
import numpy as np
import pytest
import torch

from virne.core import Controller, Counter, Recorder, Solution
from virne.core.environment import SolutionStepEnvironment
from virne.network import PhysicalNetwork, VirtualNetworkRequestSimulator
from virne.network.virtual_network_request_simulator import VirtualNetworkEvent
from virne.reliability.features import FEATURE_NAMES, feature_dimensions, ReliabilityFeatureProvider
from virne.reliability.model import ReliabilityAttributeProvider
from virne.solver import SolverRegistry
from virne.solver.learning.rl_core.feature_constructor import PNetVNetFeatureConstructor
from virne.solver.learning.rl_core.policy_builder import PolicyBuilder
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.utils.config import add_simulation_into_config
from tests.core.test_environment import make_config, make_networks, make_success_solution, NODE_ATTRS, LINK_ATTRS


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / 'virne/configs')


def config_for_smoke(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base='1.3'):
        config = compose(config_name='main', overrides=[
            'training.use_cuda=false', 'training.num_train_epochs=0',
            'v_sim_setting.num_v_nets=4', 'v_sim_setting.v_net_size.low=2',
            'v_sim_setting.v_net_size.high=3', 'p_net_setting.topology.num_nodes=12',
            'logger.backends=[]', 'logger.verbose=-1', 'nn.embedding_dim=16',
            'reliability.enabled=true', 'reliability.mode=node_and_link',
            'reliability.nodes.lambda_f=0.00001', 'reliability.nodes.mu_r=0.1',
            'reliability.links.lambda_f=0.000005', 'reliability.links.mu_r=0.1',
            'sla.enabled=true', 'sla.probabilities=[0.25,0.25,0.25,0.25]',
            'joint_objective.name=multiplicative', 'joint_objective.alpha=0.5',
            'joint_objective.beta=0.1', 'reliability_features.enabled=true',
        ])
    config.experiment.save_root_dir = str(tmp_path)
    config.experiment.run_id = 'unit-smoke'
    config.rl.feature_constructor.name = 'p_net_v_net'
    add_simulation_into_config(config)
    return config


def test_all_feature_subsets_and_dimensions(tmp_path):
    config = config_for_smoke(tmp_path)
    p_net = PhysicalNetwork.from_setting(config.p_net_setting, seed=17)
    sim = VirtualNetworkRequestSimulator.from_setting(config.v_sim_setting, seed=17)
    sim.renew(seed=17)
    v_net = sim.v_nets[0]
    solution = Solution(v_net)
    for mode in ('node_only', 'node_and_link'):
        config.reliability.mode = mode
        for flags in itertools.product((False, True), repeat=len(FEATURE_NAMES)):
            config.reliability_features.names = [name for name, enabled in zip(FEATURE_NAMES, flags) if enabled]
            constructor = PNetVNetFeatureConstructor(p_net, v_net, config)
            observation = constructor.construct(p_net, v_net, solution, 0)
            dimensions = PolicyBuilder.get_feature_dim_config(config)
            assert observation['p_net_x'].shape == (p_net.num_nodes, dimensions['p_net_x_dim'])
            assert observation['v_net_x'].shape == (v_net.num_nodes, dimensions['v_net_x_dim'])
            assert observation['p_net_edge_attr'].shape[-1] == dimensions['p_net_edge_dim']
            assert np.all(np.isfinite(observation['p_net_x']))
    config.reliability_features.names = list(FEATURE_NAMES)
    base = constructor.construct(p_net, v_net, solution, 0)
    config.reliability.enabled = False
    disabled = PNetVNetFeatureConstructor(p_net, v_net, config).construct(p_net, v_net, solution, 0)
    assert feature_dimensions(config) == (0, 0, 0)
    for name in ('p_net_x', 'v_net_x'):
        assert disabled[name].shape[-1] < base[name].shape[-1]
    config.reliability_features.enabled = False
    original = PNetVNetFeatureConstructor(p_net, v_net, config).construct(p_net, v_net, solution, 0)
    for key in original:
        np.testing.assert_array_equal(disabled[key], original[key])


def test_feature_values_and_ordering(tmp_path):
    config = config_for_smoke(tmp_path)
    config.reliability_features.names = list(FEATURE_NAMES)
    p_net, v_net = make_networks()
    ReliabilityAttributeProvider(config.reliability).apply(p_net, seed=17)
    v_net.graph['availability_requirement'] = .999
    solution = make_success_solution(v_net)
    obs = {'p_net_x': np.zeros((3, 1)), 'v_net_x': np.zeros((2, 1)),
           'p_net_edge_index': np.asarray([[0, 1, 1, 2], [1, 2, 0, 1]]),
           'p_net_edge_attr': np.zeros((4, 1))}
    provider = ReliabilityFeatureProvider(config)
    result = provider.extend(obs, p_net, v_net, solution)
    availability = provider.evaluator.availability(p_net, solution)
    node_a = p_net.nodes[0]['availability']
    np.testing.assert_allclose(result['p_net_x'][0, 1:],
                               [node_a, .999, node_a-.999, availability, -np.log(node_a),
                                -np.log(.999)+np.log(availability), 0.,
                                1.+np.log(availability)/-np.log(.999)], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(result['v_net_x'][0, 1:], [.999, availability, -np.log(.999)+np.log(availability),
                                                      1.+np.log(availability)/-np.log(.999)])


def test_disabled_generators_preserve_original_stream(tmp_path):
    config = config_for_smoke(tmp_path)
    enhanced_p = PhysicalNetwork.from_setting(config.p_net_setting, seed=17)
    enhanced = VirtualNetworkRequestSimulator.from_setting(config.v_sim_setting, seed=17)
    enhanced.renew(seed=17)
    p_setting = OmegaConf.to_container(config.p_net_setting, resolve=True)
    v_setting = OmegaConf.to_container(config.v_sim_setting, resolve=True)
    p_setting.pop('reliability')
    v_setting.pop('sla')
    original_p = PhysicalNetwork.from_setting(p_setting, seed=17)
    original = VirtualNetworkRequestSimulator.from_setting(v_setting, seed=17)
    original.renew(seed=17)
    assert list(enhanced_p.edges) == list(original_p.edges)
    assert [enhanced_p.nodes[n]['cpu'] for n in enhanced_p] == [original_p.nodes[n]['cpu'] for n in original_p]
    for first, second in zip(enhanced.v_nets, original.v_nets):
        assert first.arrival_time == second.arrival_time
        assert first.lifetime == second.lifetime
        assert list(first.nodes(data=True)) == list(second.nodes(data=True))
        assert list(first.edges(data=True)) == list(second.edges(data=True))
    enhanced.renew(seed=17)
    assert [v.graph['availability_requirement'] for v in enhanced.v_nets] == [
        v.graph['availability_requirement'] for v in copy.deepcopy(enhanced).v_nets]


def test_checkpoint_roundtrip_and_incompatible_same_dimension_features(tmp_path):
    config = config_for_smoke(tmp_path)
    config.reliability_features.names = ['raw_availability']
    solver_cls = SolverRegistry.get('ppo_dual_gat+')
    solver = solver_cls(Mock(), Mock(), Mock(), Mock(), config)
    solver.save_model('test.pkl')
    path = str(Path(solver.model_dir) / 'test.pkl')
    solver.load_model(path)
    saved = {key: value.clone() for key, value in solver.policy.state_dict().items()}
    for parameter in solver.policy.parameters():
        parameter.data.zero_()
    solver.load_model(path)
    for key, value in solver.policy.state_dict().items():
        torch.testing.assert_close(value, saved[key])
    config.reliability_features.names = ['log_risk']
    with pytest.raises(RuntimeError, match='Incompatible reliability observation'):
        solver.load_model(path)


def test_metadata_persists_gml_and_workload_reload(tmp_path):
    config = config_for_smoke(tmp_path)
    p_net = PhysicalNetwork.from_setting(config.p_net_setting, seed=17)
    p_dir = str(tmp_path / 'p')
    p_net.save_dataset(p_dir)
    loaded = PhysicalNetwork.load_dataset(p_dir)
    assert loaded.graph['reliability_mode'] == 'node_and_link'
    for node in p_net:
        assert loaded.nodes[node]['availability'] == p_net.nodes[node]['availability']
    sim = VirtualNetworkRequestSimulator.from_setting(config.v_sim_setting, seed=17)
    sim.renew(seed=17)
    v_dir = str(tmp_path / 'v')
    sim.save_dataset(v_dir)
    restored = VirtualNetworkRequestSimulator.load_dataset(v_dir)
    assert [v.graph['availability_requirement'] for v in restored.v_nets] == [v.graph['availability_requirement'] for v in sim.v_nets]
    restored.renew(seed=17)
    assert [v.graph['availability_requirement'] for v in restored.v_nets] == [v.graph['availability_requirement'] for v in sim.v_nets]


def test_online_overlap_departure_release_and_recorder(tmp_path):
    config = make_config(tmp_path)
    with open_dict(config):
        config.reliability = {'enabled': True, 'mode': 'node_and_link'}
        config.joint_objective = {'name': 'multiplicative', 'alpha': .5, 'beta': .1}
        config.rl = {'reward_calculator': {'name': 'joint_terminal'}}
    p_net, first = make_networks()
    second = copy.deepcopy(first)
    second.set_graph_attrs_data({'id': 1, 'arrival_time': 2., 'lifetime': 1.})
    for v_net in (first, second):
        v_net.graph['availability_requirement'] = .999
    ReliabilityAttributeProvider({'mode': 'node_and_link', 'nodes': {'lambda_f': .00001, 'mu_r': .1},
                                  'links': {'lambda_f': .000005, 'mu_r': .1}}).apply(p_net, seed=17)
    simulator = VirtualNetworkRequestSimulator(v_nets=[first, second], events=[
        VirtualNetworkEvent(id=0, type=1, v_net_id=0, time=1.),
        VirtualNetworkEvent(id=1, type=1, v_net_id=1, time=2.),
        VirtualNetworkEvent(id=2, type=0, v_net_id=1, time=3.),
        VirtualNetworkEvent(id=3, type=0, v_net_id=0, time=5.)], v_sim_setting={})
    counter = Counter(NODE_ATTRS, LINK_ATTRS, {}, config)
    controller = Controller(NODE_ATTRS, LINK_ATTRS, [], config)
    recorder = Recorder(counter, config)
    env = SolutionStepEnvironment(p_net, simulator, controller, recorder, counter, Mock(), config)
    recorder.count_init_p_net_info(env.p_net)
    env.num_processed_v_nets = 0
    env.start_run_time = time.time()
    env.ready(0)
    next_instance, _, done, arrival = env.step(make_success_solution(first))
    assert not done
    assert next_instance['p_net'].nodes[0]['cpu'] == 8.
    assert next_instance['p_net'].edges[0, 1]['bw'] == 6.
    assert arrival['delivered_availability'] < 1
    assert arrival['availability_requirement'] == .999
    assert arrival['joint_utility'] <= arrival['economic_quality']
    _, _, done, _ = env.step(make_success_solution(second))
    assert done
    assert [env.p_net.nodes[n]['cpu'] for n in env.p_net] == [10., 10., 10.]
    assert [env.p_net.edges[e]['bw'] for e in env.p_net.edges] == [10., 10.]
    assert env.recorder.state['inservice_count'] == 0
    for record in recorder.memory:
        assert record['availability_requirement'] == .999
        assert record['delivered_availability'] == arrival['delivered_availability']
    summary = counter.summary_records(recorder.memory)
    assert summary['sla_accepted_count'] == 2
    assert 'sla_class_metrics_json' in summary


def test_provider_cache_identity_and_disable_cleanup(tmp_path):
    from virne.utils.dataset import get_p_net_dataset_dir_from_setting, get_v_nets_dataset_dir_from_setting
    config = config_for_smoke(tmp_path)
    first_p = get_p_net_dataset_dir_from_setting(config.p_net_setting, seed=17)
    first_v = get_v_nets_dataset_dir_from_setting(config.v_sim_setting, seed=17)
    config.reliability.nodes.lambda_f = .00002
    config.sla.probabilities = [.5, .5, 0., 0.]
    add_simulation_into_config(config)
    assert first_p != get_p_net_dataset_dir_from_setting(config.p_net_setting, seed=17)
    assert first_v != get_v_nets_dataset_dir_from_setting(config.v_sim_setting, seed=17)
    config.reliability.enabled = False
    config.sla.enabled = False
    add_simulation_into_config(config)
    assert 'reliability' not in config.p_net_setting
    assert 'sla' not in config.v_sim_setting
    assert '-reliability_' not in config.simulation.p_net_dataset_dir
    assert '-sla_' not in config.simulation.v_nets_dataset_dir


def test_incomplete_scientific_config_fails_early(tmp_path):
    from virne.reliability.config import validate_reliability_config
    config = config_for_smoke(tmp_path)
    validate_reliability_config(config)
    config.joint_objective.beta = None
    with pytest.raises(ValueError, match='beta'):
        validate_reliability_config(config)
    config.reliability.enabled = False
    config.rl.reward_calculator.name = 'fixed_intermediate'
    # Irrelevant thesis settings do not affect the disabled baseline.
    config.reliability.mode = 'invalid_unused_mode'
    config.reliability_features.names = list(FEATURE_NAMES)
    validate_reliability_config(config)
    assert feature_dimensions(config) == (0, 0, 0)
