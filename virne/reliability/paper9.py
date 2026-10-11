"""Printed Paper-9 reward equations; constants/cost semantics remain explicit.

Santos et al. (2022), FGCS 136, equations (12)–(14). This formula does
not claim a recovered experimental profile or repair its counterintuitive sign.
"""
import math


class Paper9Reward:
    def __init__(self, *, rho, sigma, nu, varsigma, c):
        values = (rho, sigma, nu, varsigma, c)
        if any(value is None or not math.isfinite(value) for value in values):
            raise ValueError('Paper-9 reward constants must all be explicitly supplied')
        self.rho, self.sigma, self.nu, self.varsigma, self.c = values

    def terminal(self, *, accepted: bool, availability: float, requirement: float,
                 node_cost: float, vnf_cost: float, node_energy: float) -> float:
        if not accepted:
            return -100.
        if not all(math.isfinite(value) for value in
                   (availability, requirement, node_cost, vnf_cost, node_energy)):
            raise ValueError('Paper-9 reward inputs must be finite')
        if not 0 <= availability <= 1 or not 0 < requirement < 1:
            raise ValueError('Invalid Paper-9 availability/requirement')
        difference = availability - requirement
        availability_impact = difference if difference < 0 else math.exp(-difference)
        operational_impact = self.sigma*node_cost+self.nu*vnf_cost+self.varsigma*node_energy
        return availability_impact*self.rho*(-operational_impact)+self.c

    @staticmethod
    def intermediate(*, valid: bool) -> float:
        return 0. if valid else -10.
