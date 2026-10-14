"""Human-readable report backed by manifest, source and validation artifacts."""
import argparse
import json
from pathlib import Path
import pandas as pd
from common import METRICS, read_yaml, write_json


def run(bundle):
    bundle=Path(bundle)
    p=read_yaml(bundle/'active_profile.yaml');point=read_yaml(bundle/'calibration/recommended_config.yaml')
    hardware=json.loads((bundle/'hardware.json').read_text())
    validation=json.loads((bundle/'validation_report.json').read_text())
    manifest=json.loads((bundle/'pipeline_manifest.json').read_text())
    summary=pd.read_csv(bundle/'aggregated/results_summary.csv')
    fields=['label']+[k+'_mean' for k in METRICS]
    # Avoid an optional tabulate dependency.
    table='| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n'
    for _,r in summary.iterrows():
        table+='| '+' | '.join(f'{r[k]:.6g}' if isinstance(r[k],float) else str(r[k]) for k in fields)+' |\n'
    plots=sorted((bundle/'plots').glob('*.png'))
    text=('# Preliminary experiment pipeline report\n\n'
          '**These are preliminary pipeline-validation results, not final scientific comparisons.** '
          'Objective/reward selection, beta calibration, comparator parameter tuning, workload and scientific budgets remain future work.\n\n'
          f'Source commit: `{(bundle/"source_commit.txt").read_text().strip()}`\n\n'
          f'Hardware/device and provenance: `{hardware}`\n\n'
          f'Profile: `{p["name"]}`; quick: `{p.get("quick",False)}`; epochs: `{p["epochs"]}`; '
          f'training seeds: `{p["train_seeds"]}`; held-out test seeds: `{p["test_seeds"]}`; requests: `{p["requests"]}`.\n\n'
          f'Calibration: `{point}`. See [calibration report](calibration_report.md).\n\n'
          f'Jobs: {len(manifest)} successful training/evaluation jobs, 0 failed. Calibration jobs are listed separately in calibration_summary.csv. '
          'A failed subprocess stops the pipeline and is recorded in status.tsv; COMPLETE.txt is written only after all required outputs pass.\n\n'
          'All evaluation uses OnlineSystem arrival/accept/departure semantics. Input generation resets the seed separately for PN and VNR; '
          'full topology, resource, arrival, lifetime and SLA snapshots are hashed and compared across every method per test seed. '
          'Records are checked against regenerated request IDs/times/sizes. Baseline reliability/features/reward behavior remain disabled/original; '
          'its SLA assessment is derived from the recorded node_slots/link_paths with the frozen analytic evaluator. '
          'The same post-hoc calculation is checked against enabled-method recorder metrics.\n\n'
          'Cand-RL is a reconstructed candidate-based comparator, profile thesis_joint, custom Cand-specific MLP actor/critic with Virne PPO, redundancy=none. '
          'RA-EVNE-original is a ranking equation reconstruction with explicit static Weibull history, not an exact reproduction.\n\n'
          f'Active comparator values (provisional, not scientifically tuned): `{p["comparators"]}`.\n\n'
          'Resolved environment/resource/CTMC/SLA settings: [resolved configuration](../resolved_experiment_config.yaml). '
          'Every run also includes its full config.yaml and launch command.\n\n'
          'Executed methods: '+', '.join(summary.label)+'.\n\n'+table+'\n'
          'Metrics use Virne Counter definitions, including avg_r2c_ratio over all arrivals. SLA means use accepted services; service rate uses all arrivals. '
          'Missing values remain null/NaN. Summary std uses ddof=1; single-observation std is NaN. '
          'Training/test replicates share streams; the descriptive std is not an independence claim.\n\n'
          'Warnings:\n\n'+''.join('- '+w+'\n' for w in validation['warnings'])+'\n'
          'Figures:\n\n'+''.join(f'- [{v.stem}](../plots/{v.name}) ([PDF](../plots/{v.stem}.pdf))\n' for v in plots)+'\n'
          'Convergence: LOCAL_DEV has one epoch and cannot establish convergence. Training telemetry is plotted only when epoch logs provide sufficient data; '
          'plots/training_telemetry.json records availability. No absent training rewards are invented.\n')
    (bundle/'reports/experiment_report.md').write_text(text)
    (bundle/'experiment_report.md').write_text(text.replace('(calibration_report.md)','(reports/calibration_report.md)').replace('(../','('))
    write_json(bundle/'experiment_report.json',{'hardware':hardware,'profile':p,'calibration':point,'validation':validation,'summary':summary.to_dict('records'),'manifest':manifest})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('bundle',type=Path);args=a.parse_args();run(args.bundle)
