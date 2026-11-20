"""Use the verified Counter/mapping assessment and full seeded-input fingerprints."""
from scripts.thesis.experiments.common import collect_row


def collect(bundle,resolved,manifest):
    beta=resolved['experiment']['reporting']['assessment_beta']
    from .sla_economics import assess_job, payment_beta
    economic_beta,_=payment_beta(resolved)
    from virne.reliability.live_failures import LIVE_METRICS
    rows=[]
    for row in manifest:
        if row['phase']!='eval':continue
        measured=collect_row(row,bundle,assessment_beta=beta,assessment_mode=resolved['resolved_environment']['reliability']['mode'])
        if row.get('evaluation_mode')=='failure_free':
            measured.update({metric:None for metric in LIVE_METRICS})
            measured.update(live_failures_enabled=False,failure_trace_sha256=None)
        measured.update(assess_job(bundle,row,economic_beta)[0])
        rows.append(measured)
    return rows
