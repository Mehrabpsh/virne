"""Configured-support scaling; never infer bounds from a sampled substrate."""
import math
import numpy as np

from .model import categorical_rates, component_availability


def rate_support(spec):
    if hasattr(spec, 'get'):
        distribution = spec.get('distribution')
        if distribution == 'categorical':
            values, probabilities = categorical_rates(spec)
            return values[probabilities > 0].tolist()
        if distribution == 'uniform':
            values = [float(spec['low']), float(spec['high'])]
            if not 0 < values[0] <= values[1]:
                raise ValueError('Uniform rates require 0 < low <= high')
        else:
            raise ValueError(f'Unsupported rate support: {distribution}')
    else:
        values = np.asarray(spec, dtype=float).reshape(-1).tolist()
    if not values or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('Rate support must be finite and positive')
    return values


class LogRiskScaler:
    def __init__(self, component_settings):
        failures = rate_support(component_settings['lambda_f'])
        repairs = rate_support(component_settings['mu_r'])
        self.best = -math.log(component_availability(min(failures), max(repairs)))
        self.worst = -math.log(component_availability(max(failures), min(repairs)))

    def risk(self, availability):
        risk = -math.log(float(availability))
        if not math.isfinite(risk) or risk < self.best - 1e-12 or risk > self.worst + 1e-12:
            raise ValueError('Component availability outside configured risk support')
        if self.worst == self.best:
            return 0.0
        return min(1., max(0., (risk-self.best)/(self.worst-self.best)))

    def preference(self, availability):
        return 1. - self.risk(availability)

    def signature(self):
        return {'best_log_risk': self.best, 'worst_log_risk': self.worst}


def remaining_budget_fraction(required, log_partial):
    return 1. + log_partial / -math.log(required)
