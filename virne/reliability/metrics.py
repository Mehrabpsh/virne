"""Accepted-service SLA metrics with explicit all-arrival service denominator."""

import json


def summarize_reliability(records) -> dict:
    arrivals = records.loc[records['event_type'] == 1]
    accepted = arrivals.loc[arrivals['result'].map(lambda value: str(value).lower() == 'true')]
    summary = {
        'sla_accepted_count': int(len(accepted)),
        'sla_satisfaction_rate': float(accepted['sla_satisfied'].mean()) if len(accepted) else 0.0,
        'sla_service_rate_all_arrivals': float(accepted['sla_satisfied'].sum() / len(arrivals)) if len(arrivals) else 0.0,
        'mean_delivered_availability': float(accepted['delivered_availability'].mean()) if len(accepted) else None,
        'mean_sla_shortfall': float(accepted['sla_shortfall'].mean()) if len(accepted) else None,
        'median_sla_shortfall': float(accepted['sla_shortfall'].median()) if len(accepted) else None,
    }
    shortfalls = accepted['sla_shortfall'].dropna()
    summary['sla_shortfall_distribution_json'] = json.dumps({
        'zero': int((shortfalls == 0).sum()),
        'positive_to_one': int(((shortfalls > 0) & (shortfalls <= 1)).sum()),
        'above_one': int((shortfalls > 1).sum()),
        'quantiles': {str(q): float(shortfalls.quantile(q)) for q in (0, .25, .5, .75, 1)} if len(shortfalls) else {},
    }, sort_keys=True)
    strata = {}
    for level, group in arrivals.groupby('availability_requirement'):
        successes = group.loc[group['result'].map(lambda value: str(value).lower() == 'true')]
        strata[str(level)] = {
            'arrivals': len(group), 'accepted': len(successes),
            'acceptance_rate': len(successes) / len(group),
            'sla_satisfaction_rate': float(successes['sla_satisfied'].mean()) if len(successes) else None,
            'mean_shortfall': float(successes['sla_shortfall'].mean()) if len(successes) else None,
            'median_shortfall': float(successes['sla_shortfall'].median()) if len(successes) else None,
            'mean_delivered_availability': float(successes['delivered_availability'].mean()) if len(successes) else None,
            'mean_r2c': float(successes['v_net_r2c_ratio'].mean()) if len(successes) else None,
        }
    summary['sla_class_metrics_json'] = json.dumps(strata, sort_keys=True)
    return summary
