"""Payment contract, complete live lifetimes, portable frozen-report assessment."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.thesis.experiments.common import ROOT
from scripts.thesis.experiments_v2.analysis.sla_economics import (
    PRIMARY, PREDICTED, OBSERVED, calculate, payment_beta, prepare_report)
from scripts.thesis.experiments_v2.analysis.aggregate import tables, mode_metrics
from scripts.thesis.experiments_v2.config.resolver import resolve
from scripts.thesis.experiments_v2.config.schema import ConfigError


def arrivals():
    return pd.DataFrame({
        'v_net_id': [1, 2, 3], 'result': [True, True, False],
        'v_net_lifetime': [10., 30., 50.], 'v_net_revenue': [2., 3., 0.],
        'v_net_cost': [4., 6., 0.], 'availability_requirement': [.9, .99, .99],
        'delivered_availability': [.95, .98, np.nan],
        # Deliberately wrong pre-existing per-method utility must not be reused.
        'assessment_compliance': [0., 0., np.nan]})


def observed():
    # Shuffled IDs: matches are by identity rather than CSV order.
    return pd.DataFrame({'v_net_id': [3, 2, 1], 'accepted': [False, True, True],
                         'lifetime': [50., 30., 10.], 'downtime': [0., .6, 0.],
                         'completed': [True, True, True],
                         'observed_availability': [np.nan, .98, 1.]})


def test_payment_is_full_above_sla_and_cost_lifetime_weighted_below():
    values, audit = calculate(arrivals(), .5)
    expected_revenue = 20 + 90 * math.exp(-.5)
    assert values['sla_reserved_time_cost'] == 220
    assert values['sla_nominal_time_revenue'] == 110
    assert values[PRIMARY] == pytest.approx(expected_revenue / 220)
    assert values[PREDICTED] == values[PRIMARY]
    assert values[OBSERVED] is None
    assert values['observed_sla_adjusted_time_revenue'] is None
    assert audit.v_net_id.tolist() == [1, 2]
    assert audit.predicted_payment_factor.tolist() == pytest.approx([1., math.exp(-.5)])
    # This must not accidentally become an unweighted mean per-service utility.
    assert values[PRIMARY] != pytest.approx((.5 + .5 * math.exp(-.5)) / 2)
    at_target = arrivals();at_target.loc[1, 'delivered_availability'] = .99
    assert calculate(at_target, .5)[0][PRIMARY] == pytest.approx(.5)


def test_live_uses_actual_uptime_without_releasing_reserved_cost():
    values, audit = calculate(arrivals(), .5, mode='live_failures', observed=observed())
    assert values[PRIMARY] == pytest.approx((20 + 90 * math.exp(-.5)) / 220)
    assert values[OBSERVED] == values[PRIMARY]
    assert values['sla_reserved_time_cost'] == 220
    assert audit.observed_availability.tolist() == pytest.approx([1., .98])
    # A worse analytical prediction must not leak into the realized payment.
    different = arrivals();different.loc[1, 'delivered_availability'] = .9
    changed, _ = calculate(different, .5, mode='live_failures', observed=observed())
    assert changed[OBSERVED] == values[OBSERVED]
    assert changed[PREDICTED] < values[PREDICTED]
    within_sla = observed();within_sla.loc[2, 'downtime'] = .5
    within_sla.loc[2, 'observed_availability'] = .95
    # A nonzero outage within the SLA allowance still earns full payment.
    assert calculate(arrivals(), .5, mode='live_failures', observed=within_sla)[0][OBSERVED] == values[OBSERVED]


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'admission', 'unfinished', 'lifetime', 'downtime', 'availability'])
def test_invalid_live_evidence_is_not_silently_scored(mutation):
    obs = observed()
    if mutation == 'missing':obs = obs.iloc[:2]
    elif mutation == 'duplicate':obs.loc[0, 'v_net_id'] = 2
    elif mutation == 'admission':obs.loc[0, 'accepted'] = True
    elif mutation == 'unfinished':obs.loc[0, 'completed'] = False
    elif mutation == 'lifetime':obs.loc[1, 'lifetime'] = 31
    elif mutation == 'downtime':obs.loc[1, 'downtime'] = 31
    elif mutation == 'availability':obs.loc[1, 'observed_availability'] = .99
    with pytest.raises(ValueError):calculate(arrivals(), .5, mode='live_failures', observed=obs)


@pytest.mark.parametrize('beta', [0, -1, float('nan'), float('inf'), True, None])
def test_invalid_common_beta(beta):
    with pytest.raises(ValueError):calculate(arrivals(), beta)


def test_zero_admissions_or_zero_cost_is_undefined_not_zero():
    a = arrivals();a['result'] = False
    values, audit = calculate(a, .5)
    assert values[PRIMARY] is None and values[PREDICTED] is None and audit.empty
    a = arrivals();a['v_net_cost'] = 0
    assert calculate(a, .5)[0][PRIMARY] is None


def test_common_parameter_and_metric_configuration():
    path = ROOT / 'scripts/thesis/experiments_v2/examples/local_dev.yaml'
    r = resolve(path, ['reporting.sla_economic_beta=0.23142476412881738',
                       'metrics=[acceptance_rate,sla_adjusted_time_r2c_ratio,observed_sla_adjusted_time_r2c_ratio]'],
                subset='original_ppo,nrm')[0]
    assert payment_beta(r)[0] == .23142476412881738
    assert r['selected_methods']['original_ppo']['virne_config']['joint_objective']['beta'] is None
    assert payment_beta(r, .7) == (.7, 'explicit_report_override')
    assert OBSERVED not in mode_metrics([PRIMARY, PREDICTED, OBSERVED], 'failure_free')
    with pytest.raises(ConfigError):resolve(path, ['reporting.sla_economic_beta=0'])


def fixture_bundle(tmp_path):
    bundle = tmp_path / 'bundle';bundle.mkdir();(bundle / 'aggregated').mkdir();(bundle / 'reports').mkdir()
    path = ROOT / 'scripts/thesis/experiments_v2/examples/local_dev.yaml'
    r = resolve(path, ['reporting.plots=false'], subset='nrm')[0]
    jobs, rows = [], []
    for mode in ['failure_free', 'live_failures']:
        directory = bundle / 'methods' / mode;directory.mkdir(parents=True)
        a = arrivals();a.to_csv(directory / 'assessment_arrivals.csv', index=False)
        if mode == 'live_failures':observed().to_csv(directory / 'observed_services.csv', index=False)
        run_id = 'eval_' + mode
        job = {'run_id': run_id, 'phase': 'eval', 'method': 'nrm', 'evaluation_mode': mode,
               'relative_directory': str(directory.relative_to(bundle)),
               'run_directory': '/nonexistent/server/path/' + run_id}
        jobs.append(job)
        rows.append({**job, 'label': 'NRM', 'category': 'heuristic', 'train_seed': np.nan, 'eval_seed': 100,
                     'total_time_cost': 220., 'total_time_revenue': 110., 'long_term_time_r2c_ratio': .5,
                     **{m: .5 for m in r['experiment']['metrics']}})
    frame = pd.DataFrame(rows);frame.to_csv(bundle / 'aggregated/results_long.csv', index=False)
    tables(frame, r['experiment']['metrics'])[2].to_csv(bundle / 'aggregated/results_summary.csv', index=False)
    (bundle / 'resolved_config.yaml').write_text(yaml.safe_dump(r))
    (bundle / 'execution_summary.yaml').write_text('jobs: []\n')
    (bundle / 'hardware.json').write_text('{}')
    (bundle / 'manifest.json').write_text(json.dumps(jobs))
    (bundle / 'source_commit.txt').write_text('frozen-experiment-source')
    (bundle / 'reports/validation_report.json').write_text(json.dumps({'passed': True, 'warnings': []}))
    return bundle, r, jobs


def test_report_backfill_is_portable_repeatable_and_does_not_launch_or_rewrite_inputs(tmp_path, monkeypatch):
    from scripts.thesis.experiments_v2.analysis.report import regenerate
    from scripts.thesis.experiments_v2.execution import launcher
    monkeypatch.setattr(launcher, 'launch', lambda *a, **k: pytest.fail('report must not launch jobs'))
    bundle, r, jobs = fixture_bundle(tmp_path)
    protected = list((bundle / 'aggregated').glob('*')) + list((bundle / 'methods').rglob('*.csv')) + [bundle / 'resolved_config.yaml']
    before = {p: p.read_bytes() for p in protected}
    for _ in range(2):
        html = regenerate(bundle, sla_economic_beta=.5).read_text()
        assert 'SLA-adjusted Time R/C' in html
        assert 'analytical payment' in html.lower() and 'observed uptime/lifetime' in html
        metrics = pd.read_csv(bundle / 'reports/sla_economics_per_run.csv')
        assert metrics[PRIMARY].tolist() == pytest.approx([(20 + 90 * math.exp(-.5)) / 220] * 2)
        assert metrics[OBSERVED].isna().tolist() == [True, False]
        provenance = json.loads((bundle / 'reports/sla_economics_provenance.json').read_text())
        assert provenance['beta'] == .5 and provenance['beta_source'] == 'explicit_report_override'
        assert provenance['evaluations_computed'] == 2 and not provenance['missing_inputs']
        assert all(p.read_bytes() == content for p, content in before.items())
        assert PRIMARY in (bundle / 'reports/experiment_report.md').read_text()
        report = json.loads((bundle / 'reports/experiment_report.json').read_text())
        assert report['sla_economics']['computed_evaluations'] == 2
        assert PRIMARY + '_mean' in report['results'][0]


def test_missing_records_remain_null_and_inconsistent_cost_is_rejected(tmp_path):
    bundle, r, jobs = fixture_bundle(tmp_path)
    live = bundle / jobs[1]['relative_directory'] / 'observed_services.csv'
    saved = live.read_bytes();live.unlink()
    data = prepare_report(bundle, r, jobs, beta_override=.5)
    assert data['completed'] == 1 and len(data['missing']) == 1
    assert pd.isna(data['observations'].loc[1, PRIMARY])
    live.write_bytes(saved)
    raw = pd.read_csv(bundle / 'aggregated/results_long.csv');raw.loc[0, 'total_time_cost'] = 219
    raw.to_csv(bundle / 'aggregated/results_long.csv', index=False)
    with pytest.raises(ValueError, match='totals disagree'):
        prepare_report(bundle, r, jobs, beta_override=.5)


def test_future_collection_adds_shared_payment_without_using_training_beta(tmp_path, monkeypatch):
    from scripts.thesis.experiments_v2.analysis import collect
    bundle, r, jobs = fixture_bundle(tmp_path)
    r['experiment']['reporting']['sla_economic_beta'] = .5
    monkeypatch.setattr(collect, 'collect_row', lambda row, *a, **k: {**row, 'beta': 999.})
    rows = collect.collect(bundle, r, jobs)
    assert len(rows) == 2
    assert all(row['sla_economic_beta'] == .5 for row in rows)
    assert all(row[PRIMARY] == pytest.approx((20 + 90 * math.exp(-.5)) / 220) for row in rows)
    frame = pd.DataFrame([{**row, 'label': 'NRM', 'category': 'heuristic', 'train_seed': np.nan,
                           'eval_seed': 100} for row in rows])
    result = tables(frame, [PRIMARY])[2]
    assert 'sla_economic_beta_mean' not in result
    assert result[PRIMARY + '_mean'].tolist() == pytest.approx([rows[0][PRIMARY], rows[1][PRIMARY]])
