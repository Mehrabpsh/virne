"""Self-contained human and machine reports with explicit statistical semantics."""
import json
from pathlib import Path
import pandas as pd
import yaml
from scripts.thesis.experiments.common import write_json

STATS='''For learned methods, average each metric across evaluation seeds within each trained model,
then average those model means. train_std is sample SD (ddof=1) across model means.
eval_std is sample SD across evaluation-scenario means after averaging models within each scenario.
For heuristics, mean and eval_std use evaluation seeds; train_std is null.
A single valid observation has undefined SD (null/NaN), not zero. Missing conditional SLA means
remain null and valid counts are reported. These crossed observations are dependent; neither SD
is a confidence interval. Paired differences average models within each evaluation scenario before
subtracting methods. All long-format observations remain available.'''


def run(bundle,resolved,summary,manifest,validation):
    bundle=Path(bundle);out=bundle/'reports'
    frame=pd.read_csv(bundle/'aggregated/results_summary.csv')
    metrics=resolved['experiment']['metrics']
    table='| Method | '+' | '.join(m+' mean' for m in metrics)+' |\n| --- | '+' | '.join('---' for _ in metrics)+' |\n'
    for _,row in frame.iterrows():table+='| '+str(row.label)+' | '+' | '.join('NaN' if pd.isna(row[m+'_mean']) else f'{row[m+"_mean"]:.6g}' for m in metrics)+' |\n'
    plots=sorted((bundle/'plots').glob('*')) if (bundle/'plots').exists() else []
    hardware=json.loads((bundle/'hardware.json').read_text())
    text=('# Thesis Experiment Framework v2 — preliminary pipeline validation\n\n'
          '**Engineering validation only. Objective, reward, beta, comparator settings and scientific budgets remain unfrozen.**\n\n'
          f'Source commit: `{(bundle/"source_commit.txt").read_text().strip()}`. Source status, tracked patch and experiment-source snapshots are preserved.\n\n'
          '## Hardware and execution\n\n```yaml\n'+yaml.safe_dump(hardware,sort_keys=False)+'```\n\n'
          f'Jobs: {len(manifest)} successful, 0 failed. Calibration executed by this runner: **no**.\n\n'
          '## Frozen experiment\n\n[Resolved config](../resolved_config.yaml), [original input](../input_config.yaml), '
          '[CLI overrides](../cli_overrides.yaml), [raw Virne overrides](../raw_virne_overrides.yaml), '
          '[execution summary](../execution_summary.yaml), [manifest](../manifest.csv).\n\n'
          'Protocol:\n\n```yaml\n'+yaml.safe_dump(resolved['experiment']['protocol'],sort_keys=False)+'```\n\n'
          'Active method settings:\n\n```yaml\n'+yaml.safe_dump(resolved['experiment']['methods'],sort_keys=False)+'```\n\n'
          '## Calibration provenance\n\n```yaml\n'+yaml.safe_dump({'result':resolved['experiment'].get('calibration_result'),
              'suggestions':resolved['experiment'].get('calibration_suggestions'),'overridden_values':resolved['calibrated_overrides']},sort_keys=False)+'```\n\n'
          '## Metrics\n\n'+table+'\n'+STATS+'\n\n'
          'Virne Counter definitions are retained, including average R2C over all arrivals. SLA means use accepted services; '
          'SLA service rate uses all arrivals. Baseline reliability/SLA remain disabled in the solver; recorded mappings '
          'are assessed post hoc in the common world using the frozen evaluator. The configurable assessment_beta is '
          'used only by that API; only availability/shortfall/satisfaction are retained, so it does not change baseline rewards or define an objective.\n\n'
          '## Fairness and validation\n\n'
          'Every enabled method resolves to the same protected environment. Full regenerated topology/resource/reliability/VNR/time/SLA '
          'snapshots are compared per evaluation seed after all overrides. Record IDs, times, sizes and resource-demand totals '
          'are checked against regeneration. This proves deterministic seeded generation; it is not runtime instrumentation of every input attribute. '
          'Final departures restore resources. Learned evaluations must reload their expected checkpoint and contain no training/optimizer telemetry.\n\n'
          '[Structural validation](validation_report.md).\n\n'
          '## Warnings\n\n'+''.join('- '+v+'\n' for v in validation['warnings'])+'\n'
          'Cand-RL remains a thesis_joint reconstruction with custom Cand MLP actor/critic and redundancy=none. RA parameters and static '
          'Weibull/history assumptions are provisional; no exact Paper-9 reproduction or scientific superiority is claimed. '
          'One training epoch cannot establish convergence.\n\n'
          '## Figures\n\n'+''.join(f'- [{p.name}](../plots/{p.name})\n' for p in plots if p.suffix in ('.png','.pdf')))
    if resolved['experiment']['reporting']['generate_markdown_report']:(out/'experiment_report.md').write_text(text)
    write_json(out/'experiment_report.json',{'hardware':hardware,'execution_summary':summary,'resolved_config':resolved,
               'manifest':manifest,'validation':validation,'statistics_semantics':STATS,'results':frame.to_dict('records')})
