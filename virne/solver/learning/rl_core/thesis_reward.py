"""Joint reward adapters delegate shaping/failure semantics to Virne originals."""

import copy
from typing import Protocol

from virne.reliability.service import ReliabilityService
from virne.solver.learning.rl_core.reward_calculator import (
    BaseRewardCalculator, RewardCalculatorRegistry, VanillaRewardCalculator,
    FixedWeightRewardCalculator, AdaptiveWeightRewardCalculator,
    GradualIntermediateRewardCalculator,
)


class JointRewardCalculator(BaseRewardCalculator):
    original_calculator = VanillaRewardCalculator
    use_partial_utility = False

    def __init__(self, config):
        super().__init__(config)
        self.service = ReliabilityService(config)
        self.original = self.original_calculator(config)

    def compute(self, p_net, v_net, solution) -> float:
        proxy = copy.copy(solution)
        if solution.get('result', False) or (
                self.use_partial_utility and solution.get('place_result', False)
                and solution.get('route_result', False)):
            _, utility = self.service.evaluate(p_net, v_net, solution)
            proxy['v_net_r2c_ratio'] = utility
        reward = self.original.compute(p_net, v_net, proxy)
        solution['v_net_reward'] = proxy['v_net_reward']
        self.v_net_reward += reward
        return reward


@RewardCalculatorRegistry.register('joint_terminal')
class JointTerminalRewardCalculator(JointRewardCalculator):
    """Zero intermediate/failure reward; completed embedding receives U_i."""


@RewardCalculatorRegistry.register('joint_fixed')
class JointFixedRewardCalculator(JointRewardCalculator):
    original_calculator = FixedWeightRewardCalculator


@RewardCalculatorRegistry.register('joint_adaptive')
class JointAdaptiveRewardCalculator(JointRewardCalculator):
    original_calculator = AdaptiveWeightRewardCalculator


@RewardCalculatorRegistry.register('joint_gradual')
class JointGradualRewardCalculator(JointRewardCalculator):
    original_calculator = GradualIntermediateRewardCalculator
    use_partial_utility = True


class Potential(Protocol):
    """Future potential contract: callable(p_net, v_net, solution) -> float.

    Requires a scientifically approved Phi and terminal-state convention before
    wiring gamma * Phi(next) - Phi(current) into the episode environment.
    """

    def __call__(self, p_net, v_net, solution) -> float: ...


@RewardCalculatorRegistry.register('potential_based')
class PendingPotentialRewardCalculator(BaseRewardCalculator):
    def __init__(self, config):
        raise NotImplementedError(
            'BLOCKED: checkpoint specifies gamma*Phi(next)-Phi(current), '
            'but does not freeze Phi or its terminal-state convention'
        )

    def compute(self, p_net, v_net, solution) -> float:
        raise NotImplementedError('Potential definition pending')
