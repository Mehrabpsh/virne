"""Operational feasibility, chronological outages, reservations and frozen eval."""
import copy
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from tests.core.test_environment import make_environment, make_success_solution
from virne.core import Solution
from virne.reliability.live_failures import (
    LiveFailureProcess, OPERATIONAL, mapping_operational, operational_view,
)


def process_for(env, initial='all_up'):
    env.config.reliability = {'mode': 'node_and_link', 'nodes': {'lambda_f': .01, 'mu_r': .1},
                              'links': {'lambda_f': .01, 'mu_r': .1}}
    env.config.sla = {'levels': [.99], 'assignment': 'discrete', 'probabilities': [1.]}
    env.config.training = {'num_train_epochs': 0}
    env.config.system = {'if_offline_system': False}
    return LiveFailureProcess(env.p_net, env.v_net_simulator, env.config,
                              {'enabled': True, 'initial_state': initial}, seed=31)


def attach(env, process, events):
    process.events = sorted(events, key=lambda e: (e['time'], e['component']))
    env._live_failures = process
    env.ready(0)


def test_transit_host_outage_does_not_release_reserved_resources(tmp_path):
    env, request = make_environment(tmp_path)
    process = process_for(env)
    transit = process.node_index[1]
    attach(env, process, [{'time': 2., 'component': transit, 'up': False},
                          {'time': 4., 'component': transit, 'up': True}])
    _, _, done, _ = env.step(make_success_solution(request))
    assert done
    summary = json.loads((tmp_path / 'environment-audit/environment-audit/live_failure_summary.json').read_text())
    assert summary['total_service_downtime'] == 2.
    assert summary['observed_service_availability'] == .5
    assert summary['fulfilled_service_time_ratio'] == .5
    assert summary['observed_sla_satisfaction_rate'] == 0.
    assert summary['interrupted_service_fraction'] == 1.
    assert summary['component_failure_count'] == summary['component_repair_count'] == 1
    assert [env.p_net.nodes[n]['cpu'] for n in env.p_net] == [10., 10., 10.]
    assert all(env.p_net.edges[e]['bw'] == 10. for e in env.p_net.edges)
    assert env.recorder.state['inservice_count'] == 0


def test_overlapping_node_and_link_outages_count_union_not_sum(tmp_path):
    env, request = make_environment(tmp_path)
    p = process_for(env)
    host = p.node_index[0]
    edge = p.link_index[frozenset((1, 2))]
    p.events = [{'time': 2., 'component': host, 'up': False},
                {'time': 3., 'component': edge, 'up': False},
                {'time': 4., 'component': host, 'up': True},
                {'time': 5., 'component': edge, 'up': True}]
    p.advance(1.)
    p.arrival(request, make_success_solution(request))
    # Failure changes no node or link residual resources.
    before = copy.deepcopy(env.p_net)
    p.advance(3.5)
    assert env.p_net.nodes[0]['cpu'] == before.nodes[0]['cpu']
    assert env.p_net.edges[1, 2]['bw'] == before.edges[1, 2]['bw']
    p.advance(5.)
    p.departure(request)
    assert p.summary()['total_service_downtime'] == 3.
    assert p.services[0]['interruptions'] == 1


def test_departure_while_failed_releases_once_and_stops_downtime(tmp_path):
    env, request = make_environment(tmp_path)
    p = process_for(env)
    attach(env, p, [{'time': 3., 'component': p.node_index[0], 'up': False}])
    env.step(make_success_solution(request))
    assert p.summary()['total_service_downtime'] == 2.
    assert not env.p_net.nodes[0][OPERATIONAL]
    assert env.p_net.nodes[0]['cpu'] == 10.
    with pytest.raises(ValueError, match='Duplicate service departure'):
        p.departure(request)


def test_failed_host_is_masked_even_without_resource_checks_and_unsafe_placement(tmp_path):
    env, request = make_environment(tmp_path)
    p = process_for(env)
    attach(env, p, [{'time': 1., 'component': p.node_index[0], 'up': False}])
    candidates = env.controller.find_candidate_nodes(request, env.p_net, 0,
                                                     check_node_constraint=False, check_link_constraint=False)
    assert 0 not in candidates
    ok, _ = env.controller.node_mapper.place(request, env.p_net, 0, 0, Solution(request),
                                  if_allow_constraint_violation=True)
    assert not ok
    _, _, done, record = env.step(make_success_solution(request))
    assert done and not record['result']
    assert p.summary()['observed_service_availability'] is None
    assert p.summary()['fulfilled_service_time_ratio'] == 0.


@pytest.mark.parametrize('owner', ['node', 'link'])
def test_routing_excludes_failed_transit_domains(tmp_path, owner):
    env, request = make_environment(tmp_path)
    p = process_for(env)
    component = p.node_index[1] if owner == 'node' else p.link_index[frozenset((0, 1))]
    attach(env, p, [{'time': 1., 'component': component, 'up': False}])
    for method in ['first_shortest', 'k_shortest', 'all_shortest', 'bfs_shortest', 'available_k_shortest']:
        assert env.controller.topology_analyzer.find_shortest_paths(request, env.p_net, (0, 1), (0, 2), method=method) == []
    assert not mapping_operational(env.p_net, make_success_solution(request))
    # IDs and capacities stay intact despite the routing view excluding domains.
    assert list(env.p_net.nodes) == [0, 1, 2]
    assert list(operational_view(env.p_net).nodes) == ([0, 2] if owner == 'node' else [0, 1, 2])


def test_repair_restores_candidates_without_changing_ids_or_capacity(tmp_path):
    env, request = make_environment(tmp_path)
    p = process_for(env)
    p.events = [{'time': 1., 'component': p.node_index[0], 'up': False},
                {'time': 2., 'component': p.node_index[0], 'up': True}]
    p.advance(1.)
    assert not env.controller.constraint_checker.check_node_level_constraints(request, env.p_net, 0, 0)[0]
    p.advance(2.)
    assert env.controller.constraint_checker.check_node_level_constraints(request, env.p_net, 0, 0)[0]
    assert env.p_net.nodes[0]['cpu'] == 10.


def test_trajectories_do_not_depend_on_policy_rng_or_admission(tmp_path):
    env, _ = make_environment(tmp_path)
    np.random.seed(17)
    before = np.random.get_state()
    first = process_for(env)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    np.random.random(100)
    second = process_for(env)
    assert first.events == second.events
    assert first.trace_hash == second.trace_hash


def test_simultaneous_failure_repair_does_not_create_phantom_interruption(tmp_path):
    env, request = make_environment(tmp_path)
    p = process_for(env)
    p.events = [{'time': 2., 'component': p.node_index[0], 'up': False},
                {'time': 2., 'component': p.node_index[0], 'up': True}]
    p.advance(1.)
    p.arrival(request, make_success_solution(request))
    p.advance(5.)
    p.departure(request)
    assert p.summary()['observed_service_availability'] == 1.
    assert p.services[0]['interruptions'] == 0


def test_live_training_and_offline_mode_are_rejected(tmp_path):
    env, _ = make_environment(tmp_path)
    process_for(env)
    env.config.training.num_train_epochs = 1
    with pytest.raises(ValueError, match='evaluation-only'):
        LiveFailureProcess(env.p_net, env.v_net_simulator, env.config, {'enabled': True}, 31)
    env.config.training.num_train_epochs = 0
    env.config.system.if_offline_system = True
    with pytest.raises(ValueError, match='online'):
        LiveFailureProcess(env.p_net, env.v_net_simulator, env.config, {'enabled': True}, 31)


def test_invalid_settings_are_not_silently_ignored(tmp_path):
    env, _ = make_environment(tmp_path)
    process_for(env)
    for settings in [{'enabled': 'true'}, {'enabled': True, 'event_limit': 0},
                     {'enabled': True, 'initial_state': 'unknown'}, {'enabled': True, 'typo': 1}]:
        with pytest.raises(ValueError):
            LiveFailureProcess(env.p_net, env.v_net_simulator, env.config, settings, 31)


def test_failure_exactly_at_departure_is_not_a_service_interruption(tmp_path):
    env, request = make_environment(tmp_path)
    p = process_for(env)
    attach(env, p, [{'time': 5., 'component': p.node_index[0], 'up': False}])
    env.step(make_success_solution(request))
    assert p.summary()['total_service_downtime'] == 0.
    assert p.summary()['interrupted_service_fraction'] == 0.


def test_unsafe_bulk_mapping_rolls_back_when_a_later_host_is_down(tmp_path):
    env, request = make_environment(tmp_path)
    p = process_for(env)
    attach(env, p, [{'time': 1., 'component': p.node_index[1], 'up': False}])
    solution = Solution(request)
    assert not env.controller.node_mapper.node_mapping(request, env.p_net, [0,1], [0,1], solution,
                                                        if_allow_constraint_violation=True)
    assert not solution.node_slots
    assert env.p_net.nodes[0]['cpu'] == 10.


def test_direct_deployment_cannot_reserve_a_down_domain(tmp_path):
    env, request = make_environment(tmp_path)
    p = process_for(env)
    attach(env, p, [{'time': 1., 'component': p.node_index[1], 'up': False}])
    assert not env.controller.deploy(request, env.p_net, make_success_solution(request))
    assert all(env.p_net.edges[e]['bw'] == 10. for e in env.p_net.edges)


def test_event_limit_aborts_instead_of_truncating_the_experiment(tmp_path):
    env, _ = make_environment(tmp_path)
    process_for(env)
    env.config.reliability.nodes = {'lambda_f': 100., 'mu_r': 100.}
    with pytest.raises(ValueError, match='event_limit exceeded'):
        LiveFailureProcess(env.p_net, env.v_net_simulator, env.config,
                            {'enabled': True, 'event_limit': 1}, 31)


def test_stationary_initial_states_are_saved_and_reproducible(tmp_path):
    env, _ = make_environment(tmp_path)
    first = process_for(env, initial='stationary')
    second = process_for(env, initial='stationary')
    assert first.trace_hash == second.trace_hash
    assert all('initial_up' in component for component in first.components)
    for component in second.components:
        attrs = env.p_net.nodes[component['id']] if component['owner'] == 'node' else env.p_net.edges[component['id']]
        assert attrs[OPERATIONAL] == component['initial_up']
