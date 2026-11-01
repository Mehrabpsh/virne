"""Optimistic bounds certify impossibility, never feasibility without a witness."""
import math
import networkx as nx
from .model import component_availability
from .scaling import rate_support


def configured_upper_bound(reliability, minimum_nodes):
    """Best conceivable connected request; independent of sampled instances."""
    def best(group):
        spec = reliability[group]
        return component_availability(min(rate_support(spec['lambda_f'])),
                                      max(rate_support(spec['mu_r'])))
    bound = best('nodes')**minimum_nodes
    if reliability['mode'] == 'node_and_link':
        bound *= best('links')**max(0,minimum_nodes-1)
    return bound


def availability_upper_bound(p_net, v_net, mode):
    n = len(v_net)
    if n > len(p_net):
        return 0.
    log_bound = sum(sorted((math.log(d['availability']) for _, d in p_net.nodes(data=True)), reverse=True)[:n])
    if mode == 'node_and_link':
        # Each connected virtual component needs a connected physical union.
        minimum_edges = n - nx.number_connected_components(v_net)
        if minimum_edges > p_net.number_of_edges():
            return 0.
        log_bound += sum(sorted((math.log(d['availability']) for _, _, d in p_net.edges(data=True)), reverse=True)[:minimum_edges])
    elif mode != 'node_only':
        raise ValueError('Unknown reliability mode')
    return math.exp(log_bound)


def classify_attainability(upper_bound, requirement, satisfied_witness=False):
    if upper_bound < requirement:
        if satisfied_witness:
            raise ValueError('Satisfying witness contradicts availability upper bound')
        return 'proven_impossible'
    return 'witnessed_attainable' if satisfied_witness else 'unknown'
