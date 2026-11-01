"""Explicit Cand-RL reconstruction for the shared Virne thesis comparison.

Recovered: distance-sum Cand filtering/order, candidate representation, PPO.
Reconstructed/configured: flattened MLP, K, resource channels, endpoints, padding
mask, per-request episodes and Virne graph placement/routing. No claim of exact
Paper-9 reproduction. See docs/thesis_comparator_provenance.md.
"""

import math

import networkx as nx
import numpy as np
import torch
from torch import nn
from gymnasium import spaces

from virne.reliability.evaluator import ReliabilityEvaluator
from virne.reliability.model import DiscreteRequestSLAModel
from virne.reliability.scaling import LogRiskScaler, remaining_budget_fraction
from virne.solver.base_solver import SolverRegistry
from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.instance_rl_environment import JointPRStepInstanceRLEnv
from virne.solver.learning.rl_core.rl_solver import PPOSolver
from virne.solver.learning.rl_core.policy_builder import OptimizerBuilder
from virne.solver.learning.rl_policy.base_policy import BaseActorCritic


class CandidateSelector:
    """Resource/type filtered top-K by d(n,current_source)+d(n,destination)."""

    def __init__(self, settings):
        if not isinstance(settings.k, int) or isinstance(settings.k, bool) or settings.k <= 0:
            raise ValueError('Cand-RL requires explicit integer candidate size K > 0')
        self.settings = settings
        self.k = int(settings.k)
        if settings.distance_weight not in ('hops', 'delay'):
            raise ValueError('Explicit Cand distance_weight must be hops or delay')
        if settings.source is None or settings.destination is None:
            raise ValueError('Cand-RL needs explicit substrate source/destination or a workload endpoint provider')

    def select(self, env) -> list:
        p_net, v_net = env.p_net, env.v_net
        current = env.selected_p_net_nodes[-1] if env.selected_p_net_nodes else v_net.graph.get('source', self.settings.source)
        destination = v_net.graph.get('destination', self.settings.destination)
        if current not in p_net or destination not in p_net:
            raise ValueError('Cand endpoint absent from substrate')
        current_v_node = env.curr_v_node_id
        requirements = {attr.name: max((v_net.edges[edge][attr.name]
                            for edge in v_net.edges(current_v_node)), default=0.)
                        for attr in v_net.get_link_attrs(['resource'])}
        graph = nx.Graph()
        graph.add_nodes_from(p_net.nodes)
        for source, target, data in p_net.edges(data=True):
            if all(data[name] >= demand for name, demand in requirements.items()):
                if self.settings.distance_weight == 'delay':
                    delay = data.get('delay')
                    if delay is None or not math.isfinite(delay) or delay < 0:
                        raise ValueError('Cand delay ranking requires finite nonnegative delay on every link')
                graph.add_edge(source, target, **data)
        weight = None if self.settings.distance_weight == 'hops' else 'delay'
        from_source = nx.single_source_dijkstra_path_length(graph, current, weight=weight)
        from_destination = nx.single_source_dijkstra_path_length(graph, destination, weight=weight)
        feasible = env.controller.find_candidate_nodes(v_net, p_net, current_v_node,
                                                        filter=env.selected_p_net_nodes)
        candidates = []
        for node in feasible:
            if node not in from_source or node not in from_destination:
                continue
            supported = p_net.nodes[node].get('supported_vnf_types')
            requested_type = v_net.nodes[current_v_node].get('vnf_type')
            if supported is not None and requested_type is not None and requested_type not in supported and 'all' not in supported:
                continue
            candidates.append(node)
        # Stable graph/controller iteration order is the documented tie convention.
        return sorted(candidates, key=lambda node: from_source[node]+from_destination[node])[:self.k]


class CandRLObservationBuilder:
    """K candidate rows: normalized resources + shared component A_n; demand + SLA."""

    def __init__(self, config):
        self.settings = config.cand_rl
        self.resources = list(self.settings.resource_names or [])
        self.capacities = list(self.settings.capacity_attributes or [])
        if not self.resources or len(self.resources) != len(self.capacities):
            raise ValueError('Cand-RL requires matching explicit resource_names/capacity_attributes')
        self.evaluator = ReliabilityEvaluator(config.reliability.mode)
        self.features = list(self.settings.get('observation_features', []))
        if len(set(self.features)) != len(self.features) or set(self.features) - {'normalized_log_risk', 'remaining_budget_fraction'}:
            raise ValueError('Cand observation_features must be unique normalized_log_risk / remaining_budget_fraction')
        self.scaler = LogRiskScaler(config.reliability.nodes) if 'normalized_log_risk' in self.features else None

    @staticmethod
    def dimension(settings) -> int:
        features = settings.get('observation_features', [])
        return (settings.k*(len(settings.resource_names)+1+int('normalized_log_risk' in features))
                +len(settings.resource_names)+1+int('remaining_budget_fraction' in features))

    def construct(self, env, candidates) -> np.ndarray:
        rows = np.zeros((self.settings.k, len(self.resources)+1+int(self.scaler is not None)), dtype=np.float32)
        for index, node in enumerate(candidates):
            for col, (resource, capacity) in enumerate(zip(self.resources, self.capacities)):
                initial = float(env.p_net.nodes[node][capacity])
                if not math.isfinite(initial) or initial <= 0:
                    raise ValueError('Cand normalization requires positive finite component capacities')
                rows[index, col] = env.p_net.nodes[node][resource]/initial
            availability = self.evaluator.availability(env.p_net, {'node_slots': {0: node}})
            rows[index, len(self.resources)] = availability
            if self.scaler is not None:
                rows[index, len(self.resources)+1] = self.scaler.risk(availability)
        demand = [float(env.v_net.nodes[env.curr_v_node_id][resource]) for resource in self.resources]
        required = DiscreteRequestSLAModel.requirement(env.v_net)
        context = [required]
        if 'remaining_budget_fraction' in self.features:
            log_partial = self.evaluator.backend.log_availability(env.p_net, env.solution, self.evaluator.mode)
            context.append(remaining_budget_fraction(required, log_partial))
        return np.concatenate((rows.reshape(-1), demand,
                               context)).astype(np.float32)


class PlacementExecutor:
    """Map candidate index to physical ID; shared Virne step owns transactions."""

    @staticmethod
    def node_for_action(candidates, action):
        action = int(action)
        if not 0 <= action < len(candidates):
            raise ValueError('Cand action selected an unavailable/padded candidate')
        return candidates[action]


class CandRLInstanceEnv(JointPRStepInstanceRLEnv):
    def __init__(self, p_net, v_net, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)
        self.selector = CandidateSelector(config.cand_rl)
        self.observation_builder = CandRLObservationBuilder(config)
        self.num_actions = self.selector.k
        self.action_space = spaces.Discrete(self.num_actions)
        self.candidates = []

    def generate_action_mask(self):
        self.candidates = self.selector.select(self)
        mask = np.zeros(self.num_actions, dtype=bool)
        self._no_feasible_action = not self.candidates
        if self.candidates:
            if self.config.cand_rl.mask_padding:
                mask[:len(self.candidates)] = True
            else:
                mask[:] = True
        else:
            mask[0] = True
        return mask

    def get_observation(self, *args, **kwargs):
        mask = self.generate_action_mask()
        return {'candidate_x': self.observation_builder.construct(self, self.candidates),
                'action_mask': mask}

    def action_to_p_node_id(self, action):
        return PlacementExecutor.node_for_action(self.candidates, action)

    def step(self, action):
        if not self._no_feasible_action and not 0 <= int(action) < len(self.candidates):
            self.solution['place_result'] = False
            self.solution['description'] = 'Invalid padded Cand action (terminate reconstruction)'
            self.counter.count_solution(self.v_net, self.solution)
            return self._make_step_result(self.compute_reward(), True, self.solution.to_dict())
        return super().step(action)


class CandRLHead(nn.Module):
    def __init__(self, input_dim, widths, output_dim):
        super().__init__()
        dimensions = [input_dim, *widths, output_dim]
        layers = []
        for source, destination in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(source, destination), nn.ReLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, observation):
        return self.network(observation['candidate_x'])


class CandRLPolicy(BaseActorCritic):
    def __init__(self, settings):
        super().__init__()
        widths = list(settings.hidden_dims or [])
        if not widths or any(not isinstance(width, int) or isinstance(width, bool) or width <= 0 for width in widths):
            raise ValueError('Cand-RL requires explicit positive hidden_dims reconstruction')
        dimension = CandRLObservationBuilder.dimension(settings)
        self.actor = CandRLHead(dimension, widths, settings.k)
        self.critic = CandRLHead(dimension, widths, 1)


def build_cand_policy(agent):
    policy = CandRLPolicy(agent.config.cand_rl).to(agent.device)
    return policy, OptimizerBuilder.build_optimizer(agent.config, policy)


def cand_observation_as_tensor(observation, device):
    batch = [observation] if isinstance(observation, dict) else observation
    return {'candidate_x': torch.as_tensor(np.asarray([item['candidate_x'] for item in batch]),
                                           dtype=torch.float32, device=device),
            'action_mask': torch.as_tensor(np.asarray([item['action_mask'] for item in batch]),
                                           dtype=torch.bool, device=device)}


@SolverRegistry.register(solver_name='cand_rl', solver_type='r_learning')
class CandRLSolver(InstanceAgent, PPOSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        CandidateSelector(config.cand_rl)
        CandRLObservationBuilder(config)
        if config.cand_rl.profile == 'paper9_sanity':
            raise NotImplementedError('BLOCKED Paper-9 sanity: redundancy/accounting, operational cost/energy '
                                      'metadata and reward constants/gate must be explicitly reconstructed; '
                                      'Paper9Reward formula is available independently')
        if config.cand_rl.profile != 'thesis_joint' or not config.rl.reward_calculator.name.startswith('joint_'):
            raise ValueError('Cand main comparison requires profile=thesis_joint and a joint reward')
        if config.cand_rl.mask_padding not in (True, False):
            raise ValueError('Explicit Cand mask_padding bool is required')
        if config.cand_rl.redundancy != 'none':
            raise NotImplementedError('Only explicitly selected no-redundancy Cand thesis reconstruction is implemented')
        InstanceAgent.__init__(self, CandRLInstanceEnv)
        PPOSolver.__init__(self, controller, recorder, counter, logger, config,
                           build_cand_policy, cand_observation_as_tensor, **kwargs)

    def get_additional_checkpoint_state(self):
        from omegaconf import OmegaConf
        signature = OmegaConf.to_container(self.config.cand_rl, resolve=True)
        if not signature.get('observation_features'):
            signature.pop('observation_features', None)  # legacy checkpoint compatibility
        elif 'normalized_log_risk' in signature['observation_features']:
            signature['node_risk_bounds'] = LogRiskScaler(self.config.reliability.nodes).signature()
        return {'cand_rl_signature': signature}

    def load_additional_checkpoint_state(self, checkpoint):
        if checkpoint.get('cand_rl_signature') != self.get_additional_checkpoint_state()['cand_rl_signature']:
            raise ValueError('Incompatible Cand-RL candidate/representation/profile configuration')
