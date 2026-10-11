"""Composition and annotation shared by solver-independent evaluation hooks."""

from dataclasses import asdict

from virne.reliability.evaluator import ReliabilityEvaluator
from virne.reliability.model import DiscreteRequestSLAModel
from virne.reliability.objective import make_objective


class ReliabilityService:
    def __init__(self, config):
        reliability = config.get('reliability') or {}
        if not reliability.get('enabled', False):
            raise ValueError('Reliability-aware execution requires reliability.enabled=true')
        if reliability.get('backend', 'analytic') != 'analytic':
            raise NotImplementedError('SPN backend is optional and not implemented')
        self.config = config
        self.evaluator = ReliabilityEvaluator(reliability.get('mode', 'node_only'))
        self.objective = make_objective(config.get('joint_objective') or {})

    def evaluate(self, p_net, v_net, solution):
        result = self.evaluator.evaluate(p_net, v_net, solution, self.objective.beta)
        utility = self.objective(float(solution.get('v_net_r2c_ratio', 0.0)), result)
        return result, utility

    def annotate(self, p_net, v_net, solution) -> None:
        """Failed attempts have no delivered service; omit conditional SLA scores."""
        objective = self.config.joint_objective
        fields = {
            'availability_requirement': DiscreteRequestSLAModel.requirement(v_net),
            'delivered_availability': None, 'sla_shortfall': None,
            'sla_compliance': None, 'sla_satisfied': False, 'joint_utility': None,
            'economic_quality': float(solution.get('v_net_r2c_ratio', 0.0)),
            'objective_strategy': objective.name,
            'reward_strategy': self.config.rl.reward_calculator.name,
            'reliability_mode': self.evaluator.mode,
            'objective_alpha': objective.alpha, 'objective_beta': objective.beta,
            'reliability_config_id': p_net.graph.get('reliability_config_id'),
        }
        if solution.get('result', False):
            result, utility = self.evaluate(p_net, v_net, solution)
            fields.update(asdict(result), joint_utility=utility)
        solution.update(fields)
