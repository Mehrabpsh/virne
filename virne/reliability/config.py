"""Fail early on incomplete scientific configuration; disabled mode is a no-op."""

from virne.reliability.features import feature_names, feature_signature
from virne.reliability.model import DiscreteRequestSLAModel
from virne.reliability.service import ReliabilityService


def validate_reliability_config(config) -> None:
    enabled = config.get('reliability', {}).get('enabled', False)
    reward = config.rl.reward_calculator.name
    if not enabled:
        if reward.startswith('joint_') or reward == 'potential_based':
            raise ValueError('Joint rewards require reliability.enabled=true')
        return
    if not config.get('sla', {}).get('enabled', False):
        raise ValueError('Reliability-aware recording/reward requires sla.enabled=true')
    ReliabilityService(config)
    DiscreteRequestSLAModel(config.sla)
    feature_signature(config)
    if feature_names(config) and config.solver.solver_name != 'ppo_dual_gat+':
        if config.solver.solver_name == 'cand_rl':
            raise ValueError('Cand-RL has its own observation; set reliability_features.enabled=false')
        # Heuristics never construct a policy observation; their measurements
        # still use the shared infrastructure. Other RL architectures need adapters.
        from virne.solver import SolverRegistry
        if SolverRegistry.get(config.solver.solver_name).type == 'r_learning':
            raise ValueError('Reliability features currently support ppo_dual_gat+ only')
