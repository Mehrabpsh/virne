"""Repairable component attributes; independent RNGs preserve baseline workloads."""

import math
from collections.abc import Mapping

import numpy as np


RATE_FAILURE = 'lambda_f'
RATE_REPAIR = 'mu_r'
AVAILABILITY = 'availability'
SLA_REQUIREMENT = 'availability_requirement'
SLA_LEVELS = (0.99, 0.999, 0.9995, 0.9999)


def categorical_rates(value):
    """Validate categorical support without drawing from any RNG."""
    values = np.asarray(value.get('values'), dtype=float)
    probabilities = np.asarray(value.get('probabilities'), dtype=float)
    if (values.ndim != 1 or not len(values) or len(set(values)) != len(values)
            or not np.all(np.isfinite(values) & (values > 0))):
        raise ValueError('Categorical rates require unique finite positive values')
    if (probabilities.shape != values.shape
            or not np.all(np.isfinite(probabilities) & (probabilities >= 0))
            or not math.isclose(float(probabilities.sum()), 1., rel_tol=1e-9)):
        raise ValueError('Categorical probabilities must match values and sum to one')
    if set(value) != {'distribution', 'values', 'probabilities'}:
        raise ValueError('Categorical rate keys are distribution, values, probabilities')
    return values, probabilities


def component_availability(lambda_f: float, mu_r: float) -> float:
    """Steady-state UP probability of an independent two-state CTMC."""
    if not all(math.isfinite(x) and x > 0 for x in (lambda_f, mu_r)):
        raise ValueError('lambda_f and mu_r must be finite and strictly positive')
    # Scaling avoids overflow when both input rates are large.
    scale = max(lambda_f, mu_r)
    return (mu_r / scale) / (lambda_f / scale + mu_r / scale)


class ReliabilityAttributeProvider:
    """Explicit heterogeneous rates or explicitly configured uniform sampling.

    No experimental distribution is selected by this provider. Arrays follow
    graph iteration order; scalar values are useful for deterministic fixtures.
    """

    def __init__(self, config: Mapping):
        self.config = config
        self.mode = config.get('mode', 'node_only')
        if self.mode not in ('node_only', 'node_and_link'):
            raise ValueError(f'Unsupported reliability mode: {self.mode}')
        if config.get('backend', 'analytic') != 'analytic':
            raise NotImplementedError('Only analytic reliability is implemented; SPN is optional')

    @staticmethod
    def _rates(spec: Mapping, name: str, size: int, rng) -> np.ndarray:
        value = spec.get(name)
        if value is None:
            raise ValueError(f'Explicit reliability {name} configuration is required')
        if isinstance(value, Mapping) or hasattr(value, 'items'):
            distribution = value.get('distribution')
            if distribution == 'categorical':
                values, probabilities = categorical_rates(value)
                return rng.choice(values, size=size, p=probabilities)
            if distribution != 'uniform':
                raise ValueError(f'Unsupported rate distribution: {distribution}')
            low, high = float(value['low']), float(value['high'])
            if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
                raise ValueError('Uniform rate bounds must be finite and 0 < low <= high')
            return rng.uniform(low, high, size)
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            return np.full(size, float(values))
        if values.shape != (size,):
            raise ValueError(f'{name} must be scalar or have {size} explicit values')
        return values

    def apply(self, p_net, seed: int | None = None) -> None:
        """Add metadata without touching resource attributes or global RNG state."""
        rng = np.random.default_rng(self.config.get('seed', seed))
        groups = [('nodes', list(p_net.nodes), p_net.nodes)]
        if self.mode == 'node_and_link':
            groups.append(('links', list(p_net.edges), p_net.edges))
        # Opt-in stream separation preserves every legacy scalar/uniform draw.
        categorical = any(
            hasattr((self.config.get(group) or {}).get(rate), 'get')
            and self.config[group][rate].get('distribution') == 'categorical'
            for group, _, _ in groups for rate in (RATE_FAILURE, RATE_REPAIR))
        root_seed = self.config.get('seed', seed)
        streams = np.random.SeedSequence(root_seed).spawn(4) if categorical else None
        # Validate all rates before writing any attributes.
        prepared = []
        for group_index, (name, components, view) in enumerate(groups):
            spec = self.config.get(name) or {}
            if categorical:
                # Stable assignments even if callers insert the same nodes in another order.
                def component_key(item):
                    if name == 'links' and not p_net.is_directed():
                        return tuple(sorted((type(node).__name__, repr(node)) for node in item))
                    return (type(item).__name__, repr(item))
                components = sorted(components, key=component_key)
            failures = self._rates(spec, RATE_FAILURE, len(components),
                                   np.random.default_rng(streams[2*group_index]) if streams else rng)
            repairs = self._rates(spec, RATE_REPAIR, len(components),
                                  np.random.default_rng(streams[2*group_index+1]) if streams else rng)
            for component, failure, repair in zip(components, failures, repairs):
                metadata = {}
                for rate_name, sampled in ((RATE_FAILURE, failure), (RATE_REPAIR, repair)):
                    rate_spec = spec.get(rate_name)
                    if hasattr(rate_spec, 'get') and rate_spec.get('distribution') == 'categorical':
                        support, _ = categorical_rates(rate_spec)
                        metadata[rate_name + '_class'] = int(np.flatnonzero(support == sampled)[0])
                if 'lambda_f_class' in metadata:
                    metadata['reliability_class'] = metadata['lambda_f_class']
                prepared.append((view[component], float(failure), float(repair),
                                 component_availability(float(failure), float(repair)), metadata))
        for data, failure, repair, availability, metadata in prepared:
            for key in ('lambda_f_class', 'mu_r_class', 'reliability_class'):
                data.pop(key, None)
            data.update(lambda_f=failure, mu_r=repair, availability=availability)
            data.update(metadata)
        p_net.graph['reliability_mode'] = self.mode
        p_net.graph['reliability_config_id'] = self.config.get('config_id') or 'explicit-unidentified'


class DiscreteRequestSLAModel:
    """Request graph metadata with caller-specified assignments/probabilities."""

    def __init__(self, config: Mapping):
        self.config = config
        self.levels = np.asarray(config.get('levels', SLA_LEVELS), dtype=float)
        if self.levels.ndim != 1 or not len(self.levels):
            raise ValueError('SLA levels must be a nonempty vector')
        if not np.all(np.isfinite(self.levels) & (self.levels > 0) & (self.levels < 1)):
            raise ValueError('SLA requirements must be finite and strictly between 0 and 1')

    def assign(self, v_nets, seed: int | None = None) -> None:
        assignment = self.config.get('assignment', 'discrete')
        if assignment == 'explicit':
            values = self.config.get('values')
            if values is None or len(values) != len(v_nets):
                raise ValueError('SLA explicit values must match the request count')
        elif assignment == 'discrete':
            probabilities = np.asarray(self.config.get('probabilities'), dtype=float)
            if (probabilities.shape != self.levels.shape
                    or not np.all(np.isfinite(probabilities) & (probabilities >= 0))
                    or not math.isclose(float(probabilities.sum()), 1.0)):
                raise ValueError('Explicit SLA probabilities must match levels and sum to one')
            rng = np.random.default_rng(self.config.get('seed', seed))
            values = rng.choice(self.levels, size=len(v_nets), p=probabilities)
        else:
            raise ValueError(f'Unsupported SLA assignment: {assignment}')
        if any(float(value) not in self.levels for value in values):
            raise ValueError('Each SLA assignment must belong to the configured levels')
        for v_net, value in zip(v_nets, values):
            v_net.graph[SLA_REQUIREMENT] = float(value)
            v_net.graph['sla_class'] = str(float(value))

    @staticmethod
    def requirement(v_net) -> float:
        value = float(v_net.graph[SLA_REQUIREMENT])
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('Request availability requirement must be strictly between 0 and 1')
        return value
