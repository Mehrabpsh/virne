"""Objective formulas and reward trajectories compared to original shaping."""

import copy
import math
import unittest
from types import SimpleNamespace

import networkx as nx
from omegaconf import OmegaConf

from virne.reliability.evaluator import SLAShortfallCalculator
from virne.reliability.objective import make_objective
from virne.solver.learning.rl_core import thesis_reward
from virne.solver.learning.rl_core.reward_calculator import RewardCalculatorRegistry


class ObjectiveTests(unittest.TestCase):
    def test_formulas_and_monotonicity(self):
        q, alpha, beta = .6, .3, 2.
        for shortfall in (0., 1., 2.):
            result = SLAShortfallCalculator.calculate(.999, .999-.001*shortfall, beta)
            expected = {'weighted_sum': alpha*q+(1-alpha)*math.exp(-beta*shortfall),
                        'additive': q-beta*shortfall,
                        'multiplicative': q*math.exp(-beta*shortfall),
                        'geometric': q**alpha*math.exp(-beta*shortfall)**(1-alpha)}
            for name, value in expected.items():
                objective = make_objective({'name': name, 'alpha': alpha, 'beta': beta})
                self.assertAlmostEqual(objective(q, result), value)
                zero = SLAShortfallCalculator.calculate(.999, .999, beta)
                self.assertLessEqual(objective(q, result), objective(q, zero)+1e-12)

    def test_no_violation_and_geometric_boundary(self):
        result = SLAShortfallCalculator.calculate(.999, .999, 1)
        for name in ('additive', 'multiplicative'):
            self.assertEqual(make_objective({'name': name, 'beta': 1})(.6, result), .6)
        violated = SLAShortfallCalculator.calculate(.9999, 0, 2)
        self.assertAlmostEqual(make_objective({'name': 'geometric', 'alpha': 1., 'beta': 2})(.6, violated), .6)

    def test_parameters_required(self):
        for config in ({}, {'name': 'additive'}, {'name': 'geometric', 'beta': 1},
                       {'name': 'weighted_sum', 'beta': 1, 'alpha': 1.1}):
            with self.assertRaises(ValueError):
                make_objective(config)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create({
            'reliability': {'enabled': True, 'mode': 'node_only'},
            'joint_objective': {'name': 'additive', 'alpha': .5, 'beta': .01},
            'rl': {'reward_calculator': {'name': 'joint_terminal', 'intermediate_reward': .1}}})
        self.p_net = nx.path_graph(3)
        nx.set_node_attributes(self.p_net, .999, 'availability')
        self.v_net = nx.Graph()
        self.v_net.add_nodes_from(range(3))
        self.v_net.num_nodes = 3
        self.v_net.graph['availability_requirement'] = .999

    def test_joint_strategies_match_originals_with_utility_substitution(self):
        strategies = {'joint_terminal': 'vanilla', 'joint_fixed': 'fixed_intermediate',
                      'joint_adaptive': 'adaptive_intermediate', 'joint_gradual': 'gradual_intermediate'}
        for name, original in strategies.items():
            calculator = RewardCalculatorRegistry.get(name)(self.config)
            baseline = RewardCalculatorRegistry.get(original)(self.config)
            for stage in ('partial', 'complete', 'failure'):
                solution = {'node_slots': {0: 0, 1: 1}, 'link_paths': {},
                            'result': stage == 'complete', 'place_result': stage != 'failure',
                            'route_result': True, 'v_net_r2c_ratio': .7, 'v_net_reward': 0.}
                proxy = copy.deepcopy(solution)
                if stage == 'complete' or (stage == 'partial' and name == 'joint_gradual'):
                    _, utility = calculator.service.evaluate(self.p_net, self.v_net, solution)
                    proxy['v_net_r2c_ratio'] = utility
                expected = baseline.compute(self.p_net, self.v_net, proxy)
                self.assertAlmostEqual(calculator.compute(self.p_net, self.v_net, solution), expected)
                self.assertEqual(solution['v_net_r2c_ratio'], .7)
                self.assertEqual(solution['v_net_reward'], expected)

    def test_potential_explicitly_blocked(self):
        with self.assertRaisesRegex(NotImplementedError, 'Phi'):
            RewardCalculatorRegistry.get('potential_based')(self.config)


if __name__ == '__main__':
    unittest.main()
