"""Frozen per-request economic/availability objectives, independent of shaping."""

import math
from abc import ABC, abstractmethod

from virne.reliability.evaluator import ReliabilityResult


class JointObjective(ABC):
    def __init__(self, beta: float, alpha: float | None = None):
        if beta is None or not math.isfinite(beta) or beta <= 0:
            raise ValueError('Explicit finite beta > 0 is required (validation or DEV value)')
        if alpha is not None and (not math.isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError('alpha must be in [0, 1]')
        self.beta, self.alpha = beta, alpha

    def __call__(self, q_i: float, result: ReliabilityResult) -> float:
        if not math.isfinite(q_i) or q_i < 0:
            raise ValueError('Economic R2C must be finite and nonnegative')
        if not math.isfinite(result.sla_shortfall) or result.sla_shortfall < 0:
            raise ValueError('SLA shortfall must be finite and nonnegative')
        utility = self.utility(q_i, result.sla_shortfall)
        if not math.isfinite(utility):
            raise ValueError('Joint utility overflowed; check explicit scientific parameters')
        return utility

    @abstractmethod
    def utility(self, q_i: float, shortfall: float) -> float: ...

    def require_alpha(self) -> float:
        if self.alpha is None:
            raise ValueError('This objective requires explicit alpha in [0, 1]')
        return self.alpha


class WeightedSumObjective(JointObjective):
    def utility(self, q_i: float, shortfall: float) -> float:
        alpha = self.require_alpha()
        return alpha * q_i + (1 - alpha) * math.exp(-self.beta * shortfall)


class AdditiveShortfallObjective(JointObjective):
    def utility(self, q_i: float, shortfall: float) -> float:
        return q_i - self.beta * shortfall


class MultiplicativeObjective(JointObjective):
    def utility(self, q_i: float, shortfall: float) -> float:
        return q_i * math.exp(-self.beta * shortfall)


class GeometricObjective(JointObjective):
    def utility(self, q_i: float, shortfall: float) -> float:
        alpha = self.require_alpha()
        if alpha == 0:
            return math.exp(-self.beta * shortfall)
        if q_i == 0:
            return 0.0
        return math.exp(alpha * math.log(q_i) - (1 - alpha) * self.beta * shortfall)


OBJECTIVES = {
    'weighted_sum': WeightedSumObjective,
    'additive': AdditiveShortfallObjective,
    'multiplicative': MultiplicativeObjective,
    'geometric': GeometricObjective,
}


def make_objective(config) -> JointObjective:
    name = config.get('name')
    if name not in OBJECTIVES:
        raise ValueError(f'Unknown or unspecified joint objective: {name}')
    objective = OBJECTIVES[name](config.get('beta'), config.get('alpha'))
    if name in ('weighted_sum', 'geometric'):
        objective.require_alpha()
    return objective
