"""Frozen feature candidates; dimensions and ordering share one provider."""

import math
import numpy as np

from virne.reliability.evaluator import ReliabilityEvaluator, validate_availability
from virne.reliability.model import AVAILABILITY, DiscreteRequestSLAModel


FEATURE_NAMES = ('raw_availability', 'sla_requirement', 'sla_margin',
                 'partial_availability', 'log_risk', 'remaining_budget')
COMPONENT_FEATURES = ('raw_availability', 'sla_margin', 'log_risk')
CONTEXT_FEATURES = ('sla_requirement', 'partial_availability', 'remaining_budget')


def feature_names(config) -> tuple[str, ...]:
    settings = config.get('reliability_features') or {}
    if not config.get('reliability', {}).get('enabled', False) or not settings.get('enabled', False):
        return ()
    names = tuple(settings.get('names', ()))
    if len(set(names)) != len(names) or any(name not in FEATURE_NAMES for name in names):
        raise ValueError(f'Feature names must be unique members of {FEATURE_NAMES}')
    return names


def feature_dimensions(config) -> tuple[int, int, int]:
    names = feature_names(config)
    link_enabled = config.get('reliability', {}).get('mode', 'node_only') == 'node_and_link'
    return (len(names), sum(name in CONTEXT_FEATURES for name in names),
            sum(name in COMPONENT_FEATURES for name in names) if link_enabled else 0)


def feature_signature(config) -> dict:
    names = feature_names(config)
    # With no features the original architecture remains compatible in either mode.
    return {'names': list(names), 'mode': config.reliability.mode if names else None,
            'version': 1}


class ReliabilityFeatureProvider:
    def __init__(self, config):
        self.names = feature_names(config)
        self.mode = config.get('reliability', {}).get('mode', 'node_only')
        self.evaluator = ReliabilityEvaluator(self.mode) if self.names else None

    def extend(self, obs, p_net, v_net, solution):
        if not self.names:
            return obs
        required = DiscreteRequestSLAModel.requirement(v_net)
        partial = self.evaluator.availability(p_net, solution)
        # Use the backend log product for budget stability even on underflow.
        log_partial = self.evaluator.backend.log_availability(p_net, solution, self.mode)
        context = {'sla_requirement': required, 'partial_availability': partial,
                   'remaining_budget': -math.log(required) + log_partial}

        def row(availability, names):
            values = dict(context)
            if availability is not None:
                availability = validate_availability(availability)
                values.update(raw_availability=availability,
                              sla_margin=availability-required, log_risk=-math.log(availability))
            return [values[name] for name in names]

        p_features = [row(p_net.nodes[n][AVAILABILITY], self.names) for n in p_net.nodes]
        obs['p_net_x'] = np.concatenate((obs['p_net_x'], np.asarray(p_features, dtype=np.float32)), axis=-1)
        context_names = [name for name in self.names if name in CONTEXT_FEATURES]
        v_features = np.tile(np.asarray(row(None, context_names), dtype=np.float32), (v_net.num_nodes, 1))
        obs['v_net_x'] = np.concatenate((obs['v_net_x'], v_features), axis=-1)
        if self.mode == 'node_and_link':
            edge_names = [name for name in self.names if name in COMPONENT_FEATURES]
            # ObservationHandler uses directed edges, including reverse directions.
            node_ids = list(p_net.nodes)
            edges = obs['p_net_edge_index'].T
            e_features = [row(p_net.edges[node_ids[int(u)], node_ids[int(v)]][AVAILABILITY], edge_names)
                          for u, v in edges]
            extras = np.asarray(e_features, dtype=np.float32).reshape(len(edges), len(edge_names))
            obs['p_net_edge_attr'] = np.concatenate((obs['p_net_edge_attr'], extras), axis=-1)
        return obs
