"""One shared analytic availability evaluator for policy, rewards and metrics."""

import math
from dataclasses import dataclass
from typing import Protocol

from virne.reliability.model import AVAILABILITY, DiscreteRequestSLAModel


@dataclass(frozen=True)
class ReliabilityResult:
    availability_requirement: float
    delivered_availability: float
    sla_shortfall: float
    sla_compliance: float
    sla_satisfied: bool


def validate_availability(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f'Component availability must be in (0, 1]: {value}')
    return value


class AvailabilityBackend(Protocol):
    def availability(self, p_net, solution, mode: str) -> float: ...


class AnalyticBackend:
    """Independent failure domains, each used physical component counted once.

    Used nodes include routing transit nodes: paths depend on those physical
    nodes as well as on the VNF hosts. Undirected reverse edges share a domain.
    """

    def used_components(self, p_net, solution):
        nodes = set(solution.get('node_slots', {}).values())
        edges = set()
        for path in solution.get('link_paths', {}).values():
            for source, target in path:
                if not p_net.has_edge(source, target):
                    raise ValueError(f'Embedding references absent physical link {(source, target)}')
                nodes.update((source, target))
                edges.add((source, target) if p_net.is_directed() else frozenset((source, target)))
        for node in nodes:
            if node not in p_net:
                raise ValueError(f'Embedding references absent physical node {node}')
        return nodes, edges

    def log_availability(self, p_net, solution, mode: str) -> float:
        if mode not in ('node_only', 'node_and_link'):
            raise ValueError(f'Unsupported reliability mode: {mode}')
        nodes, edges = self.used_components(p_net, solution)
        # Iterate graph order for deterministic summation, even for mixed ID types.
        terms = [math.log(validate_availability(p_net.nodes[n][AVAILABILITY]))
                 for n in p_net.nodes if n in nodes]
        if mode == 'node_and_link':
            terms.extend(math.log(validate_availability(data[AVAILABILITY]))
                         for u, v, data in p_net.edges(data=True)
                         if ((u, v) if p_net.is_directed() else frozenset((u, v))) in edges)
        return math.fsum(terms)

    def availability(self, p_net, solution, mode: str) -> float:
        return math.exp(self.log_availability(p_net, solution, mode))


class SLAShortfallCalculator:
    @staticmethod
    def calculate(availability_requirement: float, delivered_availability: float,
                  beta: float) -> ReliabilityResult:
        required, delivered = float(availability_requirement), float(delivered_availability)
        if not math.isfinite(required) or not 0 < required < 1:
            raise ValueError('A_req must be finite and strictly between 0 and 1')
        if not math.isfinite(delivered) or not 0 <= delivered <= 1:
            raise ValueError('A_emb must be finite and in [0, 1]')
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError('beta must be finite and strictly positive')
        shortfall = max(0.0, (required - delivered) / (1.0 - required))
        return ReliabilityResult(required, delivered, shortfall,
                                 math.exp(-beta * shortfall), delivered >= required)


class ReliabilityEvaluator:
    def __init__(self, mode: str = 'node_only', backend: AvailabilityBackend | None = None):
        if mode not in ('node_only', 'node_and_link'):
            raise ValueError(f'Unsupported reliability mode: {mode}')
        self.mode = mode
        self.backend = backend or AnalyticBackend()

    def availability(self, p_net, solution) -> float:
        return self.backend.availability(p_net, solution, self.mode)

    def evaluate(self, p_net, v_net, solution, beta: float) -> ReliabilityResult:
        return SLAShortfallCalculator.calculate(DiscreteRequestSLAModel.requirement(v_net),
                                               self.availability(p_net, solution), beta)
