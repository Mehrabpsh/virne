"""Chronological, evaluation-only component outages with retained reservations.

The resource ledger and physical IDs never change on failure/repair. This module
owns only operational state and service-time measurements, not policy updates.
"""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

OPERATIONAL = '_live_operational'
LIVE_METRICS = (
    'observed_service_availability', 'fulfilled_service_time_ratio',
    'observed_sla_satisfaction_rate', 'observed_sla_service_rate_all_arrivals',
    'total_service_downtime', 'interrupted_service_fraction',
    'service_downtime_p95', 'component_failure_count', 'component_repair_count',
    'requested_service_time', 'admitted_service_time', 'fulfilled_service_time',
)


def node_operational(graph, node):
    return node in graph and graph.nodes[node].get(OPERATIONAL, True)


def link_operational(graph, source, target):
    return (node_operational(graph, source) and node_operational(graph, target)
            and graph.has_edge(source, target)
            and graph.edges[source, target].get(OPERATIONAL, True))


def operational_view(graph):
    if not graph.graph.get('_live_failures_enabled', False):
        return graph
    return nx.subgraph_view(graph, filter_node=lambda n: node_operational(graph, n),
                            filter_edge=lambda u, v: link_operational(graph, u, v))


def mapping_operational(graph, solution):
    return (all(node_operational(graph, n) for n in solution['node_slots'].values())
            and all(link_operational(graph, u, v)
                    for path in solution['link_paths'].values() for u, v in path))


def validate_settings(settings):
    allowed = {'enabled', 'seed_offset', 'initial_state', 'event_limit'}
    if set(settings) - allowed:
        raise ValueError('Unknown live_failures settings: ' + str(set(settings) - allowed))
    if not isinstance(settings.get('enabled', False), bool):
        raise ValueError('live_failures.enabled must be boolean')
    for name, default, minimum in [('seed_offset', 1000003, 0), ('event_limit', 1000000, 1)]:
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError('live_failures.' + name + ' has an invalid integer value')
    if settings.get('initial_state', 'all_up') not in ('all_up', 'stationary'):
        raise ValueError('live_failures.initial_state must be all_up or stationary')


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False,
                      default=lambda x: x.item() if hasattr(x, 'item') else str(x))


class LiveFailureProcess:
    """Exogenous component trajectories, advanced before each workload event.

    Component transitions at time t precede workload departures/arrivals at t;
    workload ties retain Virne's existing departure-before-arrival ordering.
    Every service is measured through its original departure, including outages.
    """

    def __init__(self, p_net, simulator, config, settings, seed):
        validate_settings(settings)
        if config.get('training', {}).get('num_train_epochs', 0) != 0:
            raise ValueError('Live failures are evaluation-only; training must be disabled')
        if config.get('system', {}).get('if_offline_system', False):
            raise ValueError('Live failures require the online arrival/departure system')
        if config.get('solver', {}).get('shortest_method') == 'mcf':
            raise ValueError('Live failures do not support the MCF routing backend')
        if seed is None or isinstance(seed, bool) or int(seed) < 0:
            raise ValueError('Live failures require a nonnegative experiment seed')
        from virne.reliability.model import ReliabilityAttributeProvider, DiscreteRequestSLAModel
        world = copy.deepcopy(p_net)
        ReliabilityAttributeProvider(config.reliability).apply(world, seed=int(seed))
        # Disabled-awareness baselines receive no additional observation channels.
        requests = copy.deepcopy(simulator.v_nets)
        DiscreteRequestSLAModel(config.sla).assign(requests, seed=int(seed))
        self.requirements = {int(v.id): float(v.graph['availability_requirement']) for v in requests}
        self.p_net = p_net
        self.time = 0.0
        self.horizon = max(float(v.arrival_time + v.lifetime) for v in requests)
        self.settings = dict(settings)
        self.seed = int(seed)
        self.active = {}
        self.services = {}
        self.events = []
        self.event_index = 0
        self.components = []
        self.node_index, self.link_index = {}, {}
        p_net.graph['_live_failures_enabled'] = True
        for node in p_net:
            p_net.nodes[node][OPERATIONAL] = True
        for u, v in p_net.edges:
            p_net.edges[u, v][OPERATIONAL] = True
        ordered_nodes = sorted(world.nodes, key=lambda n: _json(n))
        ordered_links = sorted(world.edges, key=lambda e: _json(sorted(e, key=_json)))
        entries = [('node', n, world.nodes[n]) for n in ordered_nodes]
        if config.reliability.mode == 'node_and_link':
            entries += [('link', e, world.edges[e]) for e in ordered_links]
        limit = settings.get('event_limit', 1000000)
        for owner, identity, attrs in entries:
            index = len(self.components)
            rates = [float(attrs['lambda_f']), float(attrs['mu_r'])]
            if not all(math.isfinite(r) and r > 0 for r in rates):
                raise ValueError('Component failure/repair rates must be finite and positive')
            component = {'owner': owner, 'id': identity, 'lambda_f': rates[0], 'mu_r': rates[1]}
            self.components.append(component)
            if owner == 'node':
                self.node_index[identity] = index
            else:
                self.link_index[self._edge_key(*identity)] = index
            identity_seed = int.from_bytes(hashlib.sha256(_json([owner, identity]).encode()).digest()[:8], 'big')
            rng = np.random.default_rng(np.random.SeedSequence(
                [int(seed), settings.get('seed_offset', 1000003), identity_seed]))
            up = (settings.get('initial_state', 'all_up') == 'all_up'
                  or rng.random() < rates[1] / sum(rates))
            component['initial_up'] = bool(up)
            self._set_state(index, up)
            moment = 0.0
            while True:
                duration = float(rng.exponential(1.0 / rates[0 if up else 1]))
                next_time = moment + duration
                if not math.isfinite(next_time) or next_time > self.horizon:
                    break
                if next_time <= moment:
                    raise ValueError('Component event interval is below simulation-time resolution')
                moment = next_time
                up = not up
                if len(self.events) >= limit:
                    raise ValueError('live_failures.event_limit exceeded; trajectory was not truncated')
                self.events.append({'time': moment, 'component': index, 'up': bool(up)})
        self.events.sort(key=lambda e: (e['time'], e['component']))
        self.trace_hash = hashlib.sha256(_json({'components': self.components,
                                               'events': self.events, 'horizon': self.horizon}).encode()).hexdigest()

    def _edge_key(self, u, v):
        return (u, v) if self.p_net.is_directed() else frozenset((u, v))

    def _set_state(self, index, up):
        component = self.components[index]
        if component['owner'] == 'node':
            self.p_net.nodes[component['id']][OPERATIONAL] = bool(up)
        else:
            self.p_net.edges[component['id']][OPERATIONAL] = bool(up)

    def _service_up(self, service):
        return (all(node_operational(self.p_net, n) for n in service['_nodes'])
                and all(link_operational(self.p_net, u, v) for u, v in service['_links']))

    def _integrate(self, until):
        duration = until - self.time
        for service in self.active.values():
            if not self._service_up(service):
                service['downtime'] += duration
        self.time = until

    def advance(self, until):
        until = float(until)
        if not math.isfinite(until) or until < self.time or until > self.horizon:
            raise ValueError('Live failure clock must advance within the workload horizon')
        while self.event_index < len(self.events) and self.events[self.event_index]['time'] <= until:
            moment = self.events[self.event_index]['time']
            self._integrate(moment)
            before = {key: self._service_up(s) for key, s in self.active.items()}
            # Simultaneous transitions cause one service-level interruption at most.
            while self.event_index < len(self.events) and self.events[self.event_index]['time'] == moment:
                event = self.events[self.event_index]
                self._set_state(event['component'], event['up'])
                self.event_index += 1
            for key, s in self.active.items():
                if before[key] and not self._service_up(s) and moment < s['arrival_time'] + s['lifetime']:
                    s['interruptions'] += 1
        self._integrate(until)

    def arrival(self, v_net, solution):
        if not math.isclose(self.time, float(v_net.arrival_time), rel_tol=0, abs_tol=1e-12):
            raise ValueError('Arrival does not match the live simulation clock')
        key = int(v_net.id)
        if key in self.services:
            raise ValueError('Duplicate service arrival in live failure measurements')
        accepted = bool(solution['result'])
        if accepted and not mapping_operational(self.p_net, solution):
            raise ValueError('Accepted mapping uses a failed component')
        from virne.reliability.evaluator import AnalyticBackend
        nodes, _ = AnalyticBackend().used_components(self.p_net, solution) if accepted else (set(), set())
        links = {self._edge_key(u, v): (u, v) for path in solution['link_paths'].values() for u, v in path} if accepted else {}
        service = {'v_net_id': key, 'arrival_time': float(v_net.arrival_time),
                   'lifetime': float(v_net.lifetime), 'accepted': accepted,
                   'availability_requirement': self.requirements[key], 'downtime': 0.0,
                   'interruptions': 0, 'completed': False, '_nodes': nodes, '_links': list(links.values())}
        self.services[key] = service
        if accepted:
            self.active[key] = service

    def departure(self, v_net):
        key = int(v_net.id)
        if key not in self.services:
            raise ValueError('Departure has no preceding live-measured arrival')
        service = self.services[key]
        if not math.isclose(self.time, service['arrival_time'] + service['lifetime'], rel_tol=0, abs_tol=1e-12):
            raise ValueError('Departure does not match the original service lifetime')
        if service['completed']:
            raise ValueError('Duplicate service departure')
        self.active.pop(key, None)
        service['completed'] = True
        lifetime = service['lifetime']
        observed = 1.0 - service['downtime'] / lifetime if service['accepted'] else None
        service.update(observed_availability=observed,
                       observed_sla_satisfied=bool(observed is not None and observed >= service['availability_requirement']))

    def summary(self):
        if len(self.services) != len(self.requirements) or self.active or any(not s['completed'] for s in self.services.values()):
            raise ValueError('Cannot finalize live measurements before all original departures')
        accepted = [s for s in self.services.values() if s['accepted']]
        requested = math.fsum(s['lifetime'] for s in self.services.values())
        admitted = math.fsum(s['lifetime'] for s in accepted)
        downtime = math.fsum(s['downtime'] for s in accepted)
        compliant = sum(s['observed_sla_satisfied'] for s in accepted)
        processed = self.events[:self.event_index]
        return {
            'live_failures_enabled': True, 'failure_trace_sha256': self.trace_hash,
            'initial_state': self.settings.get('initial_state', 'all_up'),
            'failure_seed': self.seed, 'failure_seed_offset': self.settings.get('seed_offset', 1000003),
            'observation_horizon': self.horizon, 'recovery_policy': 'retain_reservations_wait_for_repair',
            'requested_service_time': requested, 'admitted_service_time': admitted,
            'fulfilled_service_time': admitted - downtime,
            'observed_service_availability': (admitted - downtime) / admitted if admitted else None,
            'fulfilled_service_time_ratio': (admitted - downtime) / requested if requested else None,
            'observed_sla_satisfaction_rate': compliant / len(accepted) if accepted else None,
            'observed_sla_service_rate_all_arrivals': compliant / len(self.services) if self.services else None,
            'total_service_downtime': downtime,
            'interrupted_service_fraction': sum(s['interruptions'] > 0 for s in accepted) / len(accepted) if accepted else None,
            'service_downtime_p95': float(np.quantile([s['downtime'] for s in accepted], .95)) if accepted else None,
            'component_failure_count': sum(not e['up'] for e in processed),
            'component_repair_count': sum(e['up'] for e in processed),
            'observed_class_metrics': self._class_metrics(),
        }

    def _class_metrics(self):
        result = {}
        for level in sorted(set(self.requirements.values())):
            group = [s for s in self.services.values() if s['availability_requirement'] == level]
            accepted = [s for s in group if s['accepted']]
            compliant = sum(s['observed_sla_satisfied'] for s in accepted)
            result[str(level)] = {'arrivals': len(group), 'accepted': len(accepted), 'compliant': compliant,
                                  'compliant_service_rate': compliant / len(group) if group else None,
                                  'conditional_satisfaction': compliant / len(accepted) if accepted else None}
        return result

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        (directory / 'live_failure_summary.json').write_text(_json(summary) + '\n')
        (directory / 'live_failure_trace.json').write_text(_json({
            'trace_sha256': self.trace_hash, 'components': self.components,
            'events': self.events, 'horizon': self.horizon}) + '\n')
        services = [{k: v for k, v in s.items() if not k.startswith('_')} for s in self.services.values()]
        with (directory / 'observed_services.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(services[0]) if services else ['v_net_id'])
            writer.writeheader()
            writer.writerows(services)
        return summary
