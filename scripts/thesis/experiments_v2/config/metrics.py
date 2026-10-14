"""Derive supported numeric Counter fields from the current source, plus assessment fields."""
import ast
import inspect
import textwrap
from scripts.thesis.experiments.common import METRICS
from .schema import require


def supported():
    from virne.core.counter import Counter
    tree=ast.parse(textwrap.dedent(inspect.getsource(Counter.summary_records)))
    fields={node.slice.value for node in ast.walk(tree) if isinstance(node,ast.Subscript)
            and isinstance(node.value,ast.Name) and node.value.id=='summary_info'
            and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str)}
    fields.update(METRICS)
    fields.update(['sla_accepted_count','v_net_count','runtime'])
    fields.update(prefix+name for prefix in ('mean_','peak_') for name in ('p_net_node_resource_utilization','p_net_link_resource_utilization'))
    return fields


def validate_metrics(names):
    require(all(isinstance(n,str) for n in names),'Metric names must be strings')
    unknown=set(names)-supported()
    require(not unknown,f'Unsupported/non-numeric configured metrics: {sorted(unknown)}')
