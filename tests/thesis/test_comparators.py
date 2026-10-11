"""Comparator reconstruction contracts, candidate actions and shared evaluation."""

import math
from unittest.mock import Mock

from omegaconf import OmegaConf
import numpy as np
import pytest

from virne.core import Controller, Counter, Recorder
from virne.network import PhysicalNetwork, VirtualNetworkRequestSimulator
from virne.reliability.paper9 import Paper9Reward
from virne.reliability.evaluator import ReliabilityEvaluator
from virne.solver import SolverRegistry
from virne.solver.heuristic.reliability_rank import combine_rank
from virne.solver.learning.reinforcement_learning.cand_rl_solver import (
    CandRLInstanceEnv, CandRLObservationBuilder, PlacementExecutor,
)
from tests.thesis.test_integration import config_for_smoke


def cand_config(tmp_path):
    config = config_for_smoke(tmp_path)
    config.reliability_features.enabled = False
    config.rl.reward_calculator.name = 'joint_terminal'
    config.cand_rl = OmegaConf.create({
        'profile': 'thesis_joint', 'k': 3, 'distance_weight': 'hops',
        'source': 0, 'destination': 1, 'resource_names': ['cpu'],
        'capacity_attributes': ['max_cpu'], 'hidden_dims': [16, 16],
        'mask_padding': True, 'redundancy': 'none'})
    return config


def infrastructure(config):
    p_net = PhysicalNetwork.from_setting(config.p_net_setting, seed=17)
    simulator = VirtualNetworkRequestSimulator.from_setting(config.v_sim_setting, seed=17)
    simulator.renew(seed=17)
    attrs, links = config.v_sim_setting.node_attrs_setting, config.v_sim_setting.link_attrs_setting
    counter = Counter(attrs, links, {}, config)
    controller = Controller(attrs, links, {}, config)
    recorder = Recorder(counter, config)
    return p_net, simulator.v_nets[0], controller, recorder, counter


def test_cand_actions_train_and_inference_share_candidate_mask(tmp_path):
    config = cand_config(tmp_path)
    p_net, v_net, controller, recorder, counter = infrastructure(config)
    env = CandRLInstanceEnv(p_net, v_net, controller, recorder, counter, Mock(), config)
    observation, _ = env.reset()
    assert observation['candidate_x'].shape == (CandRLObservationBuilder.dimension(config.cand_rl),)
    np.testing.assert_array_equal(observation['action_mask'], env.generate_action_mask())
    assert len(env.generate_action_mask()) == 3
    action = int(np.flatnonzero(observation['action_mask'])[0])
    host = env.candidates[action]
    env.step(action)
    assert list(env.solution.node_slots.values())[0] == host
    solver = SolverRegistry.get('cand_rl')(controller, recorder, counter, Mock(), config)
    solver.eval()
    solution = solver.solve({'p_net': p_net, 'v_net': v_net})
    assert solution['result']
    assert len(solution.node_slots) == v_net.num_nodes
    assert 0 < ReliabilityEvaluator(config.reliability.mode).availability(p_net, solution) <= 1
    solver.save_model('cand.pkl')
    solver.load_model(str(tmp_path / 'ppo_dual_gat+' / 'unit-smoke' / 'models' / 'cand.pkl'))
    config.cand_rl.distance_weight = 'delay'
    with pytest.raises(RuntimeError, match='Incompatible Cand-RL'):
        solver.load_model(str(tmp_path / 'ppo_dual_gat+' / 'unit-smoke' / 'models' / 'cand.pkl'))


def test_cand_empty_candidates_safe_termination(tmp_path):
    config = cand_config(tmp_path)
    p_net, v_net, controller, recorder, counter = infrastructure(config)
    for node in p_net:
        p_net.nodes[node]['cpu'] = 0
    # Force positive demand; topology/resources retain same metadata.
    for node in v_net:
        v_net.nodes[node]['cpu'] = 1
    env = CandRLInstanceEnv(p_net, v_net, controller, recorder, counter, Mock(), config)
    obs, _ = env.reset()
    assert obs['action_mask'].tolist() == [True, False, False]
    assert env._no_feasible_action
    _, _, terminated, _, _ = env.step(0)
    assert terminated and not env.solution['result']
    assert not env.solution.node_slots
    with pytest.raises(ValueError):
        PlacementExecutor.node_for_action([], 0)


def test_paper9_printed_formula_and_blocked_profile(tmp_path):
    reward = Paper9Reward(rho=2, sigma=.3, nu=.4, varsigma=.5, c=10)
    operation = .3*2+.4*3+.5*4
    assert reward.terminal(accepted=True, availability=.9999, requirement=.999,
                           node_cost=2, vnf_cost=3, node_energy=4) == pytest.approx(
                               math.exp(-.0009)*2*(-operation)+10)
    assert reward.intermediate(valid=False) == -10
    assert reward.terminal(accepted=False, availability=0, requirement=.999,
                           node_cost=0, vnf_cost=0, node_energy=0) == -100
    with pytest.raises(ValueError):
        Paper9Reward(rho=None, sigma=1, nu=1, varsigma=1, c=1)
    config = cand_config(tmp_path)
    config.cand_rl.profile = 'paper9_sanity'
    with pytest.raises(NotImplementedError, match='BLOCKED'):
        SolverRegistry.get('cand_rl')(Mock(), Mock(), Mock(), Mock(), config)


def test_ra_combine_requires_explicit_selection():
    scores, availability = {0: 10, 1: 5}, {0: .99, 1: .999}
    assert combine_rank(scores, availability, 'product', normalization='max') == {0: .99, 1: .4995}
    assert combine_rank(scores, availability, 'weighted_sum', .5, normalization='max')[0] == .995
    with pytest.raises(ValueError):
        combine_rank(scores, availability, None)


@pytest.mark.parametrize('key', ['ra_nrm', 'ra_grc', 'ra_evne_a', 'ra_evne_original'])
def test_ra_heuristics_shared_solution_contract(tmp_path, key):
    config = config_for_smoke(tmp_path)
    config.solver.solver_name = key
    config.ra_heuristic.combine = 'product'
    config.ra_heuristic.normalization = 'max'
    config.ra_evne.cpu_attribute = 'cpu'
    config.ra_evne.bandwidth_attribute = 'bw'
    config.ra_evne.empty_distance = 1
    config.ra_evne.weibull_scale = 250000.
    config.ra_evne.weibull_shape = 1.5
    config.ra_evne.ages = [0.]*12
    config.ra_evne.failure_counts = [0]*12
    config.ra_evne.last_failure_positions = [0]*12
    p_net, v_net, controller, recorder, counter = infrastructure(config)
    solver = SolverRegistry.get(key)(controller, recorder, counter, Mock(), config)
    solution = solver.solve({'p_net': p_net, 'v_net': v_net})
    assert solution['result']
    assert set(solution.node_slots) == set(v_net.nodes)
    assert ReliabilityEvaluator('node_and_link').availability(p_net, solution) <= 1
