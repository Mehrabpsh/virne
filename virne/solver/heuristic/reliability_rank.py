"""Thesis heuristic adaptations and a provenance-labeled RA-EVNE ranking port.

RA-EVNE equations (12)–(18): Su et al., Advanced Engineering Sciences,
50(2), 126–132 (2018). Original uses Weibull age/environment reliability;
-A substitutes the thesis shared analytic component availability only.
"""

import math

import networkx as nx

from virne.reliability.evaluator import ReliabilityEvaluator
from virne.solver.base_solver import SolverRegistry
from virne.solver.heuristic.node_rank import BaseNodeRankSolver
from virne.solver.rank.node_rank import NRMNodeRank, GRCNodeRank


def combine_rank(scores: dict, availability: dict, name: str, alpha=None, normalization=None) -> dict:
    """Explicit validation candidates, never a silently selected final combine."""
    if normalization not in ('max', 'none'):
        raise ValueError('RA combine normalization must be explicitly max or none')
    maximum = max(scores.values(), default=0.)
    normalized = ({node: score / maximum if maximum > 0 else 0. for node, score in scores.items()}
                  if normalization == 'max' else dict(scores))
    if name == 'product':
        return {node: normalized[node] * availability[node] for node in scores}
    if name == 'weighted_sum':
        if alpha is None or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('RA weighted_sum requires explicit alpha in [0, 1]')
        return {node: alpha*normalized[node]+(1-alpha)*availability[node] for node in scores}
    raise ValueError('RA-NRM/RA-GRC combine must be explicitly product or weighted_sum')


class ReliabilityRankSolver(BaseNodeRankSolver):
    base_rank_cls = NRMNodeRank

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)
        if not config.get('reliability', {}).get('enabled', False):
            raise ValueError('RA heuristic requires reliability.enabled=true')
        self.base_rank = self.base_rank_cls()
        self.settings = config.ra_heuristic
        combine_rank({}, {}, self.settings.combine, self.settings.alpha, self.settings.normalization)
        self.evaluator = ReliabilityEvaluator(config.reliability.mode)
        self.link_rank = None

    def node_mapping(self, v_net, p_net, solution):
        virtual_order = list(self.base_rank(v_net))
        base_scores = self.base_rank(p_net, sort=False)
        availability = {node: self.evaluator.availability(p_net, {'node_slots': {0: node}})
                        for node in p_net.nodes}
        combined = combine_rank(base_scores, availability, self.settings.combine, self.settings.alpha, self.settings.normalization)
        physical_order = sorted(combined, key=combined.get, reverse=True)
        return self.controller.node_mapper.node_mapping(
            v_net, p_net, sorted_v_nodes=virtual_order, sorted_p_nodes=physical_order,
            solution=solution, reusable=self.reusable, matching_mathod=self.matching_mathod)


@SolverRegistry.register(solver_name='ra_nrm', solver_type='node_ranking')
class RANRMSolver(ReliabilityRankSolver):
    """Thesis adaptation of Virne NRM, not a published original algorithm."""


@SolverRegistry.register(solver_name='ra_grc', solver_type='node_ranking')
class RAGRCSolver(ReliabilityRankSolver):
    """Thesis adaptation of Virne GRC, not a published original algorithm."""
    base_rank_cls = GRCNodeRank


class RAEVNEBaseSolver(BaseNodeRankSolver):
    use_thesis_availability = False

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)
        self.settings = config.ra_evne
        self.link_rank = None
        self.evaluator = ReliabilityEvaluator(config.reliability.mode)
        initial = self.settings.empty_distance
        if initial is None or not math.isfinite(initial) or initial <= 0:
            raise ValueError('RA-EVNE needs an explicit first-placement LS convention: ra_evne.empty_distance > 0')
        if not self.settings.cpu_attribute or not self.settings.bandwidth_attribute:
            raise ValueError('RA-EVNE requires explicit CPU and bandwidth attribute names')
        if not self.use_thesis_availability:
            for name in ('weibull_scale', 'weibull_shape'):
                value = self.settings.get(name)
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'RA-EVNE-Original requires {name} > 0')
            for name in ('ages', 'failure_counts', 'last_failure_positions'):
                if self.settings.get(name) is None:
                    raise ValueError(f'RA-EVNE-Original requires explicit {name} per physical node')

    def _importance(self, network, node, distances) -> float:
        cpu = float(network.nodes[node][self.settings.cpu_attribute])
        bandwidth = sum(float(network.edges[node, neighbor][self.settings.bandwidth_attribute])
                        for neighbor in network.neighbors(node))
        lengths = distances[node]
        # Disconnected components cannot have finite global distance-sum centrality.
        total = sum(lengths.values())
        centrality = 1/total if len(lengths) == len(network) and total > 0 else 0.
        return (cpu + bandwidth) * centrality

    def _original_reliability(self, p_net) -> dict:
        settings = self.settings
        arrays = [settings.ages, settings.failure_counts, settings.last_failure_positions]
        if any(len(values) != len(p_net) for values in arrays):
            raise ValueError('RA-EVNE age/failure/position arrays must match physical node count')
        failed_count = sum(count > 0 for count in settings.failure_counts)
        result = {}
        for node, age, count, position in zip(p_net.nodes, *arrays):
            if not math.isfinite(age) or age < 0 or int(count) != count or count < 0:
                raise ValueError('RA-EVNE requires nonnegative finite age and integer failure count')
            if count > 0 and (int(position) != position or not 1 <= position <= failed_count):
                raise ValueError('Failure positions must be in 1..number of failed nodes')
            environment = position/(failed_count*count) if count > 0 else 1.
            result[node] = math.exp(-(age/settings.weibull_scale)**settings.weibull_shape) * environment
        return result

    def node_mapping(self, v_net, p_net, solution):
        virtual_distances = dict(nx.all_pairs_shortest_path_length(v_net))
        virtual_order = sorted(v_net.nodes, reverse=True,
                               key=lambda node: self._importance(v_net, node, virtual_distances))
        physical_distances = dict(nx.all_pairs_shortest_path_length(p_net))
        reliability = ({node: self.evaluator.availability(p_net, {'node_slots': {0: node}})
                        for node in p_net.nodes} if self.use_thesis_availability
                       else self._original_reliability(p_net))
        for v_node in virtual_order:
            selected = list(solution.node_slots.values())
            candidates = self.controller.find_candidate_nodes(v_net, p_net, v_node, filter=selected)
            scores = {}
            for node in candidates:
                if any(host not in physical_distances[node] for host in selected):
                    continue
                distance = sum(physical_distances[node][host] for host in selected) if selected else self.settings.empty_distance
                scores[node] = self._importance(p_net, node, physical_distances) * reliability[node] / distance
            if not scores:
                return False
            host = max(scores, key=scores.get)
            placed, _ = self.controller.node_mapper.place(v_net, p_net, v_node, host, solution)
            if not placed:
                return False
        return True


@SolverRegistry.register(solver_name='ra_evne_original', solver_type='node_ranking')
class RAEVNEOriginalSolver(RAEVNEBaseSolver):
    """Published ranking equations with explicit reconstructed LS initialization.

    Static age/failure-history inputs do not simulate realized failures. This is
    a ranking port on Virne's feasibility/routing, not a full paper reproduction.
    """


@SolverRegistry.register(solver_name='ra_evne_a', solver_type='node_ranking')
class RAEVNEAvailabilitySolver(RAEVNEBaseSolver):
    """Thesis adaptation of the same importance ranking using analytic A_n."""
    use_thesis_availability = True
