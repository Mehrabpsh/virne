#!/usr/bin/env python3
"""Execute a concrete experiment. Calibration is never invoked by this tool."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import argparse
import json
import time
import pandas as pd
import yaml
from scripts.thesis.experiments.common import command, write_json, write_yaml
from scripts.thesis.experiments_v2.config.resolver import resolve
from scripts.thesis.experiments_v2.config.schema import ConfigError
from scripts.thesis.experiments_v2.execution.planner import jobs, summary
from scripts.thesis.experiments_v2.execution.launcher import launch
from scripts.thesis.experiments_v2.execution.runtime import freeze, install_signals, new_bundle, stage


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--set',action='append',default=[]);p.add_argument('--virne-set',action='append',default=[])
    p.add_argument('--methods');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--summary-output',type=Path,help='Optionally save dry-run summary; existing files refused')
    p.add_argument('--run-tests',action='store_true')
    return p


def execute(resolved,original,plan,output=None,run_tests=False):
    # Import plotting only after preparing a writable matplotlib cache.
    bundle=new_bundle(resolved,output);install_signals();manifest=[];start=time.monotonic()
    def checkpoint():
        write_json(bundle/'manifest.json',manifest)
        write_json(bundle/'pipeline_manifest.json',manifest)  # compatibility for shared telemetry renderer
        pd.DataFrame(manifest).to_csv(bundle/'manifest.csv',index=False)
    try:
        freeze(bundle,resolved,original,plan);checkpoint()
        if run_tests:
            stage(bundle,'tests',lambda:command(bundle,'pytest',['bash','scripts/thesis/test.sh','tests','scripts/thesis/experiments/test_pipeline.py','scripts/thesis/experiments_v2/tests','-q']))
        # Save every exact planned per-run configuration before expensive execution.
        from scripts.thesis.experiments_v2.execution.planner import final_config
        planned_configs=bundle/'planned_configs';planned_configs.mkdir()
        for job in plan['jobs']:write_yaml(planned_configs/(job['run_id']+'.yaml'),final_config(resolved,job,bundle,plan['hardware']['selected_device']))
        for phase in ('train','eval'):
            def run_phase():
                for job in plan['jobs']:
                    if job['phase']!=phase:continue
                    manifest.append(launch(bundle,resolved,job,plan['hardware']['selected_device']));checkpoint()
            stage(bundle,'training' if phase=='train' else 'online_evaluation',run_phase)
        from scripts.thesis.experiments_v2.analysis import collect, aggregate, validate, plot, report
        rows=stage(bundle,'collection',lambda:collect.collect(bundle,resolved,manifest))
        stage(bundle,'aggregation',lambda:aggregate.run(bundle,resolved,rows))
        validation=stage(bundle,'validation',lambda:validate.run(bundle,resolved,manifest))
        if resolved['experiment']['reporting']['plots']:stage(bundle,'plots',lambda:plot.run(bundle,resolved))
        stage(bundle,'report',lambda:report.run(bundle,resolved,plan,manifest,validation))
        reporting=resolved['experiment']['reporting']
        required=['reports/experiment_report.json','reports/validation_report.json','aggregated/results_long.csv',
                  'aggregated/results_by_train_seed.csv','aggregated/results_summary.csv','aggregated/results_summary.json']
        if reporting['generate_markdown_report']:required.append('reports/experiment_report.md')
        if reporting['plots']:
            for metric in resolved['experiment']['metrics']+['sla_class_breakdown']:
                required.extend(f'plots/{metric}.{fmt}' for fmt in reporting['formats'])
        for file in required:
            if not (bundle/file).is_file():raise FileNotFoundError(bundle/file)
        write_json(bundle/'completion.json',{'passed':True,'jobs':len(manifest),'runtime_seconds':time.monotonic()-start,'calibration_executed':False})
        (bundle/'COMPLETE.txt').write_text(f'All required structural stages succeeded. Jobs: {len(manifest)}. Runtime: {time.monotonic()-start:.3f}s.\nPreliminary pipeline validation; scientific parameters remain unfrozen.\n')
        print(f'COMPLETE: {bundle}',flush=True)
        return bundle
    except BaseException as exc:
        checkpoint();write_json(bundle/'FAILED.json',{'error':str(exc),'runtime_seconds':time.monotonic()-start})
        raise


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    try:
        resolved,original=resolve(args.config,args.set,args.virne_set,args.methods)
        plan=summary(resolved)
    except (ConfigError,KeyError,TypeError) as exc:p.error(str(exc))
    for warning in plan['warnings']:print('WARNING: '+warning,file=sys.stderr,flush=True)
    print(yaml.safe_dump(plan,sort_keys=False),flush=True)
    if args.summary_output:
        with args.summary_output.open('x') as f:yaml.safe_dump(plan,f,sort_keys=False)
    if args.dry_run:return plan
    return execute(resolved,original,plan,args.output,args.run_tests)

if __name__=='__main__':main()
