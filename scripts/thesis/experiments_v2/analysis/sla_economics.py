"""Common SLA-dependent payment assessment; never changes solvers or training rewards.

Reports can derive these metrics from a copied bundle without rewriting its raw
records, frozen configuration, or original aggregation.
"""
import datetime
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.thesis.experiments.common import write_json

PRIMARY = 'sla_adjusted_time_r2c_ratio'
PREDICTED = 'predicted_sla_adjusted_time_r2c_ratio'
OBSERVED = 'observed_sla_adjusted_time_r2c_ratio'
METRICS = (PRIMARY, PREDICTED, OBSERVED,
           'sla_adjusted_time_revenue', 'predicted_sla_adjusted_time_revenue',
           'observed_sla_adjusted_time_revenue', 'sla_nominal_time_revenue',
           'sla_reserved_time_cost')
OBSERVED_METRICS = (OBSERVED, 'observed_sla_adjusted_time_revenue')
METADATA = ('sla_economic_beta', 'sla_economic_availability_basis')
DESCRIPTION = (
    'SLA-adjusted Time R/C = sum(R_i L_i exp(-beta s_i)) / sum(C_i L_i), '
    'where s_i = max(0, (SLA_i - A_i)/(1 - SLA_i)). Meeting SLA receives full nominal payment. '
    'Resources remain reserved through the original lifetime, including outages. '
    'One assessment beta is shared by every method, independent of its training objective. '
    'The primary metric uses analytical availability in failure_free and observed uptime/lifetime '
    'in live_failures. Predicted assessment is also retained for live mappings. '
    'Analytical payment is a model-based score, not expected finite-lifetime payment under this '
    'nonlinear contract. Live-minus-free primary differences compare different assessment bases '
    'and must not be interpreted as the isolated economic effect of failures. '
    'This cost/lifetime-weighted ratio is not the mean training reward. '
    'No admissions or zero total cost give an undefined ratio, not zero.'
)


def payment_beta(resolved, override=None):
    reporting = resolved['experiment']['reporting']
    if override is not None:
        value, source = override, 'explicit_report_override'
    elif 'sla_economic_beta' in reporting:
        value, source = reporting['sla_economic_beta'], 'reporting.sla_economic_beta'
    else:
        value, source = reporting['assessment_beta'], 'legacy_reporting.assessment_beta'
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('SLA economic beta must be finite and strictly positive')
    return float(value), source


def job_directory(bundle, job):
    relative = job.get('relative_directory')
    if relative and (Path(bundle) / relative).is_dir():
        return Path(bundle) / relative
    return Path(job['run_directory'])


def _boolean(values, name):
    mapped = values.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
    if mapped.isna().any():
        raise ValueError(f'Invalid boolean {name}')
    return mapped.astype(bool)


def _finite(values, name):
    values = pd.to_numeric(values, errors='raise').astype(float)
    if not np.isfinite(values).all():
        raise ValueError(f'Nonfinite {name}')
    return values


def calculate(arrivals, beta, *, mode='failure_free', observed=None):
    """Return run totals and admitted-service audit rows using a common payment law."""
    payment_beta({'experiment': {'reporting': {'sla_economic_beta': beta}}})
    if mode not in ('failure_free', 'live_failures'):
        raise ValueError(f'Unknown SLA economic evaluation mode: {mode}')
    required = {'v_net_id', 'result', 'v_net_lifetime', 'v_net_revenue', 'v_net_cost',
                'availability_requirement', 'delivered_availability'}
    if required - set(arrivals):
        raise ValueError(f'Missing SLA economic inputs: {sorted(required - set(arrivals))}')
    if arrivals.v_net_id.isna().any() or arrivals.v_net_id.duplicated().any():
        raise ValueError('SLA economic arrivals require unique request IDs')
    accepted = _boolean(arrivals.result, 'result')
    services = arrivals.loc[accepted].copy().set_index('v_net_id')
    for name in ('v_net_lifetime', 'v_net_revenue', 'v_net_cost', 'availability_requirement', 'delivered_availability'):
        services[name] = _finite(services[name], name)
    if (services.v_net_lifetime <= 0).any() or (services[['v_net_revenue', 'v_net_cost']] < 0).any().any():
        raise ValueError('Invalid SLA economic lifetime/revenue/cost')
    if not services.availability_requirement.between(0, 1, inclusive='neither').all():
        raise ValueError('SLA requirements must be strictly between zero and one')
    if not services.delivered_availability.between(0, 1).all():
        raise ValueError('Predicted availability must be between zero and one')
    audit = services[['v_net_lifetime', 'availability_requirement', 'delivered_availability']].copy()
    audit['nominal_time_revenue'] = services.v_net_revenue * services.v_net_lifetime
    audit['reserved_time_cost'] = services.v_net_cost * services.v_net_lifetime
    _finite(audit.nominal_time_revenue, 'nominal time revenue')
    _finite(audit.reserved_time_cost, 'reserved time cost')

    def payment(availability, prefix):
        shortfall = ((services.availability_requirement - availability) /
                     (1 - services.availability_requirement)).clip(lower=0)
        audit[prefix + '_shortfall'] = shortfall
        audit[prefix + '_payment_factor'] = np.exp(-float(beta) * shortfall)
        audit[prefix + '_adjusted_time_revenue'] = audit.nominal_time_revenue * audit[prefix + '_payment_factor']
        return math.fsum(audit[prefix + '_adjusted_time_revenue'])

    predicted_revenue = payment(services.delivered_availability, 'predicted')
    observed_revenue = None
    if mode == 'live_failures':
        columns = {'v_net_id', 'accepted', 'lifetime', 'downtime', 'completed', 'observed_availability'}
        if observed is None or columns - set(observed):
            raise ValueError('Live SLA economic assessment requires complete observed service records')
        if observed.v_net_id.duplicated().any() or set(observed.v_net_id) != set(arrivals.v_net_id):
            raise ValueError('Observed service IDs must cover every arrival exactly once')
        obs = observed.set_index('v_net_id').reindex(arrivals.v_net_id)
        if not np.array_equal(_boolean(obs.accepted, 'accepted').to_numpy(), accepted.to_numpy()):
            raise ValueError('Observed admissions disagree with assessment arrivals')
        if not _boolean(obs.completed, 'completed').all():
            raise ValueError('SLA economic assessment requires all original departures')
        obs = observed.set_index('v_net_id').reindex(services.index)
        lifetime = _finite(obs.lifetime, 'observed lifetime')
        downtime = _finite(obs.downtime, 'downtime')
        if not np.allclose(lifetime, services.v_net_lifetime, rtol=1e-10, atol=1e-10):
            raise ValueError('Observed lifetime disagrees with original lifetime')
        if (downtime < 0).any() or (downtime > lifetime).any():
            raise ValueError('Downtime must be within the original lifetime')
        availability = 1 - downtime / lifetime
        if not np.allclose(_finite(obs.observed_availability, 'observed availability'), availability,
                           rtol=1e-10, atol=1e-12):
            raise ValueError('Observed availability disagrees with uptime/lifetime')
        if 'availability_requirement' in obs and not np.allclose(
                _finite(obs.availability_requirement, 'observed SLA'), services.availability_requirement,
                rtol=0, atol=1e-12):
            raise ValueError('Observed SLA disagrees with common assessment')
        audit['downtime'] = downtime
        audit['observed_availability'] = availability
        observed_revenue = payment(availability, 'observed')
    cost = math.fsum(audit.reserved_time_cost)
    nominal = math.fsum(audit.nominal_time_revenue)
    primary_revenue = observed_revenue if mode == 'live_failures' else predicted_revenue
    ratio = lambda revenue: revenue / cost if revenue is not None and cost > 0 else None
    values = {PRIMARY: ratio(primary_revenue), PREDICTED: ratio(predicted_revenue),
              OBSERVED: ratio(observed_revenue), 'sla_adjusted_time_revenue': primary_revenue,
              'predicted_sla_adjusted_time_revenue': predicted_revenue,
              'observed_sla_adjusted_time_revenue': observed_revenue,
              'sla_nominal_time_revenue': nominal, 'sla_reserved_time_cost': cost,
              'sla_economic_beta': float(beta),
              'sla_economic_availability_basis': 'observed' if mode == 'live_failures' else 'predicted'}
    return values, audit.reset_index()


def assess_job(bundle, job, beta):
    directory = job_directory(bundle, job)
    path = directory / 'assessment_arrivals.csv'
    columns = ['v_net_id', 'result', 'v_net_lifetime', 'v_net_revenue', 'v_net_cost',
               'availability_requirement', 'delivered_availability']
    arrivals = pd.read_csv(path, usecols=columns)
    mode = job.get('evaluation_mode')
    if not mode:
        import yaml
        config = yaml.safe_load((directory / 'config.yaml').read_text())
        mode = 'live_failures' if config.get('live_failures', {}).get('enabled', False) else 'failure_free'
    observed_path = directory / 'observed_services.csv'
    observed = pd.read_csv(observed_path) if mode == 'live_failures' else None
    values, services = calculate(arrivals, beta, mode=mode, observed=observed)
    hashes = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in ([path, observed_path] if mode == 'live_failures' else [path])}
    return values, services, hashes


def prepare_report(bundle, resolved, manifest, *, beta_override=None):
    """Derive report sidecars/in-memory tables; leave the frozen aggregation unchanged."""
    from .aggregate import tables, paired_differences, paired_modes
    bundle = Path(bundle)
    beta, beta_source = payment_beta(resolved, beta_override)
    raw_path = bundle / 'aggregated/results_long.csv'
    frame = pd.read_csv(raw_path) if raw_path.is_file() else pd.DataFrame()
    aggregate_path = bundle / 'aggregated/results_summary.csv'
    aggregate = pd.read_csv(aggregate_path) if aggregate_path.is_file() else pd.DataFrame()
    jobs = {job['run_id']: job for job in manifest if job['phase'] == 'eval'}
    completed, missing, inputs, audits = [], [], {}, []
    for index, row in frame.iterrows():
        job = jobs.get(row.run_id, row.to_dict())
        try:
            values, services, hashes = assess_job(bundle, job, beta)
        except FileNotFoundError as exc:
            missing.append({'run_id': row.run_id, 'reason': str(exc)})
            values = {key: None for key in METRICS}
            values.update(sla_economic_beta=beta, sla_economic_availability_basis='unavailable')
        else:
            # Independently check that the new payment denominator is the old ledger.
            for key, reference in [('sla_reserved_time_cost', 'total_time_cost'),
                                   ('sla_nominal_time_revenue', 'total_time_revenue')]:
                if reference in row and pd.notna(row[reference]) and not math.isclose(
                        values[key], row[reference], rel_tol=1e-10, abs_tol=1e-7):
                    raise ValueError(f'{row.run_id}: SLA economic totals disagree with {reference}')
            inputs[row.run_id] = hashes
            completed.append(row.run_id)
            for key in ('run_id', 'method', 'train_seed', 'eval_seed', 'evaluation_mode'):
                if key in row:services[key] = row[key]
            audits.append(services)
        for key, value in values.items():
            frame.loc[index, key] = value
    data = {'observations': frame, 'aggregate': aggregate, 'beta': beta,
            'completed': len(completed), 'missing': missing}
    out = bundle / 'reports'
    out.mkdir(exist_ok=True)
    if len(frame):
        active = list(dict.fromkeys([*resolved['experiment']['metrics'], PRIMARY]))
        # Missing legacy inputs stay explicit null, never inferred as full payment.
        for metric in METRICS:frame[metric] = pd.to_numeric(frame[metric], errors='raise')
        by_train, by_eval, aggregate = tables(frame, active)
        data.update(aggregate=aggregate, by_train=by_train, by_eval=by_eval,
                    paired=paired_differences(by_eval, active), paired_modes=paired_modes(frame, active))
        identity = [c for c in ('run_id', 'method', 'label', 'category', 'train_seed', 'eval_seed', 'evaluation_mode',
                              'acceptance_rate', 'long_term_r2c_ratio', 'long_term_time_r2c_ratio') if c in frame]
        frame[identity + list(METADATA) + list(METRICS)].to_csv(out / 'sla_economics_per_run.csv', index=False)
        columns = [c for c in aggregate if c in ('evaluation_mode', 'method', 'label', 'kind', 'run_count')
                   or any(c.startswith(metric + '_') for metric in METRICS)]
        aggregate[columns].to_csv(out / 'sla_economics_summary.csv', index=False)
        if audits:
            pd.concat(audits, ignore_index=True).to_csv(out / 'sla_economics_services.csv.gz', index=False, compression='gzip')
    write_json(out / 'sla_economics_provenance.json', {
        'version': 1, 'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'beta': beta, 'beta_source': beta_source, 'definition': DESCRIPTION,
        'source_experiment_commit': (bundle / 'source_commit.txt').read_text().strip() if (bundle / 'source_commit.txt').is_file() else None,
        'assessment_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_results_long_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path.is_file() else None,
        'evaluations_requested': len(frame), 'evaluations_computed': len(completed),
        'missing_inputs': missing, 'input_sha256': inputs,
        'new_simulations_launched': False, 'frozen_inputs_and_aggregates_modified': False})
    return data


def plot_report(bundle, resolved, data):
    if not resolved['experiment']['reporting']['plots'] or not data['completed']:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .aggregate import mode_groups
    out = Path(bundle) / 'plots'
    out.mkdir(exist_ok=True)
    for mode, group in mode_groups(data['aggregate']):
        if not group[PRIMARY + '_mean'].notna().any():continue
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(group))
        ax.bar(x, group[PRIMARY + '_mean'], color='#32658b')
        for column, offset, color, label in [('train_std', -.08, '#8c2d04', 'SD across model means'),
                                             ('eval_std', .08, '#202020', 'SD across scenario means')]:
            error = group[PRIMARY + '_' + column]
            valid = error.notna()
            if valid.any():ax.errorbar(x[valid] + offset, group.loc[valid, PRIMARY + '_mean'],
                                      yerr=error[valid], fmt='none', ecolor=color, capsize=3, label=label)
        ax.set_xticks(x, group.label, rotation=35, ha='right')
        ax.set_ylabel('SLA-adjusted Time R/C')
        basis = 'observed uptime / lifetime' if mode == 'live_failures' else 'analytical availability'
        ax.set_title(f'{mode or "Legacy"}: {basis}; common beta = {data["beta"]:.8g}')
        ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
        fig.tight_layout()
        for extension in resolved['experiment']['reporting']['formats']:
            fig.savefig(out / f'{mode + "__" if mode else ""}{PRIMARY}.{extension}', bbox_inches='tight', dpi=180)
        plt.close(fig)


def comparison_table(data):
    """Small reader-facing comparison, alongside the complete machine tables."""
    frame = data['aggregate']
    names = {'label': 'Method', 'long_term_time_r2c_ratio_mean': 'Raw Time R/C',
             PRIMARY + '_mean': 'SLA-adjusted Time R/C', PRIMARY + '_count': 'Measured runs'}
    columns = [c for c in ['evaluation_mode', *names] if c in frame]
    if PRIMARY + '_mean' not in columns:
        return pd.DataFrame()
    return frame[columns].rename(columns=names)
