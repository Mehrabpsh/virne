"""Analytic contracts from the 1405-07-16 integrated checkpoint."""
import copy
import math
import unittest

import networkx as nx
import numpy as np

from virne.reliability import (
    ReliabilityAttributeProvider, DiscreteRequestSLAModel, ReliabilityEvaluator,
    SLAShortfallCalculator, component_availability,
)


class ComponentTests(unittest.TestCase):
    def test_ctmc_and_monotonicity(self):
        value = component_availability(0.001, 0.1)
        self.assertAlmostEqual(value, 0.1 / 0.101)
        self.assertLess(component_availability(0.002, 0.1), value)
        self.assertGreater(component_availability(0.001, 0.2), value)
        self.assertAlmostEqual(component_availability(1e308, 1e308), 0.5)

    def test_invalid_rates(self):
        for rate in (-1.0, 0.0, math.nan, math.inf):
            for pair in ((rate, 0.1), (0.001, rate)):
                with self.assertRaises(ValueError):
                    component_availability(*pair)

    def test_provider_determinism_and_resource_preservation(self):
        config = {'mode': 'node_and_link', 'nodes': {
            'lambda_f': {'distribution': 'uniform', 'low': 0.001, 'high': 0.002},
            'mu_r': [0.1, 0.2, 0.3]}, 'links': {'lambda_f': 0.001, 'mu_r': 0.1}}
        first = nx.path_graph(3)
        nx.set_node_attributes(first, 10, 'cpu')
        second = copy.deepcopy(first)
        np.random.seed(17)
        before = np.random.get_state()
        provider = ReliabilityAttributeProvider(config)
        provider.apply(first, seed=17)
        provider.apply(second, seed=17)
        self.assertEqual(dict(first.nodes(data=True)), dict(second.nodes(data=True)))
        self.assertEqual(list(first.edges(data=True)), list(second.edges(data=True)))
        self.assertTrue(all(d['cpu'] == 10 for _, d in first.nodes(data=True)))
        self.assertTrue(np.array_equal(before[1], np.random.get_state()[1]))
        self.assertEqual(first.nodes[0], copy.deepcopy(first).nodes[0])

    def test_provider_requires_explicit_rates(self):
        with self.assertRaises(ValueError):
            ReliabilityAttributeProvider({}).apply(nx.path_graph(2))


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.p_net = nx.path_graph(3)
        nx.set_node_attributes(self.p_net, {0: .99, 1: .98, 2: .97}, 'availability')
        nx.set_edge_attributes(self.p_net, .96, 'availability')
        self.solution = {'node_slots': {0: 0, 1: 0, 2: 2},
                         'link_paths': {(0, 2): [(0, 1), (1, 2)],
                                        (1, 2): [(2, 1), (1, 0)]}}

    def test_unique_nodes_and_transit_node(self):
        evaluator = ReliabilityEvaluator('node_only')
        self.assertAlmostEqual(evaluator.availability(self.p_net, self.solution), .99 * .98 * .97)
        self.assertAlmostEqual(evaluator.availability(self.p_net, {'node_slots': {0: 0, 1: 0}}), .99)

    def test_unique_undirected_links(self):
        self.assertAlmostEqual(ReliabilityEvaluator('node_and_link').availability(
            self.p_net, self.solution), .99 * .98 * .97 * .96**2)

    def test_node_only_ignores_link_attributes(self):
        for _, _, data in self.p_net.edges(data=True):
            data.clear()
        self.assertAlmostEqual(ReliabilityEvaluator().availability(self.p_net, self.solution), .99*.98*.97)
        with self.assertRaises(KeyError):
            ReliabilityEvaluator('node_and_link').availability(self.p_net, self.solution)

    def test_invalid_embedding_rejected(self):
        with self.assertRaises(ValueError):
            ReliabilityEvaluator().availability(self.p_net, {'node_slots': {0: 99}})
        with self.assertRaises(ValueError):
            ReliabilityEvaluator().availability(self.p_net, {'link_paths': {0: [(0, 2)]}})


class SLATests(unittest.TestCase):
    def test_shortfall_compliance(self):
        calculate = SLAShortfallCalculator.calculate
        result = calculate(.999, .998, 2)
        self.assertAlmostEqual(result.sla_shortfall, 1)
        self.assertAlmostEqual(result.sla_compliance, math.exp(-2))
        for availability in (.999, 1):
            result = calculate(.999, availability, 2)
            self.assertEqual(result.sla_shortfall, 0)
            self.assertEqual(result.sla_compliance, 1)
            self.assertTrue(result.sla_satisfied)
        self.assertGreater(calculate(.999, .998, 2).sla_compliance,
                           calculate(.999, .997, 2).sla_compliance)

    def test_safe_boundary_validation(self):
        for required, delivered, beta in ((1, .99, 1), (0, .99, 1), (.99, 1.1, 1),
                                           (.99, -.1, 1), (.99, .98, 0), (.99, .98, math.nan)):
            with self.assertRaises(ValueError):
                SLAShortfallCalculator.calculate(required, delivered, beta)
        result = SLAShortfallCalculator.calculate(np.nextafter(1., 0.), 0., 1.)
        self.assertTrue(math.isfinite(result.sla_shortfall))

    def test_assignment_deterministic(self):
        config = {'probabilities': [.25]*4}
        first, second = [nx.Graph() for _ in range(20)], [nx.Graph() for _ in range(20)]
        model = DiscreteRequestSLAModel(config)
        model.assign(first, seed=17)
        model.assign(second, seed=17)
        self.assertEqual([g.graph for g in first], [g.graph for g in second])
        self.assertEqual(len({g.graph['availability_requirement'] for g in first}), 4)
        with self.assertRaises(ValueError):
            DiscreteRequestSLAModel({}).assign(first)


if __name__ == '__main__':
    unittest.main()
