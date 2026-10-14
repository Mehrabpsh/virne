"""Use the verified Counter/mapping assessment and full seeded-input fingerprints."""
from scripts.thesis.experiments.common import collect_row


def collect(bundle,resolved,manifest):
    beta=resolved['experiment']['reporting']['assessment_beta']
    return [collect_row(row,bundle,assessment_beta=beta,assessment_mode=resolved['resolved_environment']['reliability']['mode']) for row in manifest if row['phase']=='eval']
