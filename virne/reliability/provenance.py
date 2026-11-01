"""Capture the actual initial exogenous inputs before any solver decision."""
import json
from pathlib import Path


def input_snapshot(pn, simulator):
    def graph(network):
        return {'nodes': [(n, dict(a)) for n, a in network.nodes(data=True)],
                'edges': [(u, v, dict(a)) for u, v, a in network.edges(data=True)]}
    return {'substrate': graph(pn), 'requests': [
        dict(graph(v), id=v.id, arrival_time=v.arrival_time, lifetime=v.lifetime,
             sla=v.graph.get('availability_requirement')) for v in simulator.v_nets]}


def capture_inputs(config, pn, simulator):
    audit = (config.get('logger') or {}).get('runtime_audit_path')
    if audit:
        target = Path(audit).with_name('runtime_inputs.json')
        target.write_text(json.dumps(input_snapshot(pn, simulator), sort_keys=True,
                                     allow_nan=False, default=lambda value: value.item()))
