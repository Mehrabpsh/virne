"""Canonical sequential experiment orchestration; no changes to frozen solvers."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import socket
import signal
import subprocess
import sys
import time

from common import ROOT, METHODS, command, launch, read_yaml, write_json, write_yaml


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--profile',default='local_dev',help='local_dev, gpu_server, or a YAML path')
    a.add_argument('--output',type=Path)
    a.add_argument('--quick',action='store_true',help='Keep every method and epochs=1; shrink requests/calibration seeds')
    a.add_argument('--run-tests',action='store_true')
    a.add_argument('--dry-run',action='store_true',help='Print profile/job plan without creating outputs')
    args=a.parse_args()
    path=Path(args.profile)
    if not path.is_file():path=Path(__file__).parent/'profiles'/f'{args.profile.lower()}.yaml'
    p=read_yaml(path)
    p['quick']=args.quick
    if args.quick:
        if p['name']!='LOCAL_DEV':a.error('--quick is only valid for LOCAL_DEV')
        p.update(requests=4,epochs=1)
        p['calibration'].update(requests=8,seeds=[11])
    if p['name']=='LOCAL_DEV' and p['epochs']!=1:a.error('LOCAL_DEV requires EPOCHS=1')
    if not set(p['train_seeds']).isdisjoint(p['test_seeds']):a.error('Training/test seeds must be held out')
    if len(set(p['train_seeds']))!=len(p['train_seeds']) or len(set(p['test_seeds']))!=len(p['test_seeds']):a.error('Duplicate seeds')
    if args.dry_run:
        print(json.dumps({'profile':p,'methods':METHODS,'training_jobs':3*len(p['train_seeds']),
                          'evaluation_jobs':len(p['test_seeds'])*(8+3*len(p['train_seeds'])),
                          'calibration_jobs':2*len(p['calibration']['seeds'])*len(p['calibration']['arrival_rates'])*len(p['calibration']['failure_rates']),
                          'stages':['preflight','tests (optional)','calibration','freeze','train','online evaluation','aggregate','validate','plot','report','complete']},indent=2))
        return
    timestamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    bundle=(args.output or ROOT/'results/thesis_experiments'/f'{timestamp}_{p["name"].lower()}').resolve()
    bundle.mkdir(parents=True,exist_ok=False)
    for d in ('logs','reports'): (bundle/d).mkdir()
    print(f'Results directory: {bundle}',flush=True)
    (bundle/'status.tsv').write_text('step\tstatus\truntime_seconds\n')
    os.environ['PYTHON']=sys.executable
    os.environ['OMP_NUM_THREADS']=str(p['threads']);os.environ['MKL_NUM_THREADS']=str(p['threads'])
    os.environ['OPENBLAS_NUM_THREADS']=str(p['threads'])
    os.environ['MPLCONFIGDIR']=str(bundle/'matplotlib_cache')
    os.environ['PYTHONPATH']=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')
    (bundle/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+''.join('export '+k+'='+shlex.quote(os.environ[k])+'\n' for k in ('PYTHON','PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','MPLCONFIGDIR')))
    start=time.monotonic()
    manifest=[]
    def checkpoint():
        write_json(bundle/'pipeline_manifest.json',manifest)
        import pandas as pd
        pd.DataFrame(manifest).to_csv(bundle/'pipeline_manifest.csv',index=False)
    def stage(name,fn):
        print(f'Stage: {name}',flush=True);t=time.monotonic()
        standalone={'freeze':'build_experiment_config','aggregate':'aggregate_results',
                    'validate':'validate_results','plots':'plot_results','report':'generate_report'}
        if name in standalone:
            with (bundle/'commands.sh').open('a') as f:
                f.write(shlex.join([sys.executable,'scripts/thesis/experiments/'+standalone[name]+'.py',str(bundle)])+'\n')
        try:result=fn()
        except Exception:
            with (bundle/'status.tsv').open('a') as f:f.write(f'{name}\tfailed\t{time.monotonic()-t:.3f}\n')
            raise
        with (bundle/'status.tsv').open('a') as f:f.write(f'{name}\tpassed\t{time.monotonic()-t:.3f}\n')
        return result
    def preflight():
        import torch
        for f in ('main.py','scripts/thesis/run.sh','scripts/thesis/test.sh','virne/configs/main.yaml','virne/configs/thesis.yaml','virne/configs/p_net_setting/default.yaml','virne/configs/v_sim_setting/default.yaml'):
            if not (ROOT/f).is_file():raise FileNotFoundError(f)
        def git(*cmd):return subprocess.check_output(['git',*cmd],cwd=ROOT,text=True)
        (bundle/'source_commit.txt').write_text(git('rev-parse','HEAD'))
        (bundle/'source_status.txt').write_text(git('status','--short'))
        (bundle/'tracked_changes.patch').write_text(git('diff','--binary','HEAD'))
        # Preserve untracked experiment source too, since LOCAL_DEV permits dirty trees.
        import shutil
        shutil.copytree(Path(__file__).parent,bundle/'pipeline_source',ignore=shutil.ignore_patterns('__pycache__'))
        p['selected_device']='cuda' if p['use_cuda'] and torch.cuda.is_available() else 'cpu'
        hardware={'timestamp_utc':timestamp,'branch':git('branch','--show-current').strip(),'commit':git('rev-parse','HEAD').strip(),
                  'dirty':bool(git('status','--porcelain')),'python_path':sys.executable,'python_version':sys.version,'pytorch_version':torch.__version__,
                  'cuda_available':torch.cuda.is_available(),'selected_device':p['selected_device'],'hostname':socket.gethostname(),
                  'command_line':shlex.join(sys.argv),'profile':p['name'],'threads':p['threads'],
                  'cpu':__import__('platform').processor(),'cuda_device':torch.cuda.get_device_name(0) if p['selected_device']=='cuda' else None}
        write_json(bundle/'hardware.json',hardware);write_yaml(bundle/'active_profile.yaml',p)
        print(json.dumps(hardware,indent=2),flush=True)
        import hydra, pandas, torch_geometric, matplotlib  # dependency preflight
    def interrupted(signum,frame):
        raise RuntimeError(f'Pipeline interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        stage('preflight',preflight)
        if args.run_tests or p['run_tests']:stage('tests',lambda:command(bundle,'pytest',['bash','scripts/thesis/test.sh','tests','scripts/thesis/experiments/test_pipeline.py','-q']))
        import calibrate, build_experiment_config, aggregate_results, validate_results, plot_results, generate_report
        point=stage('calibration',lambda:calibrate.run(bundle,p))
        stage('freeze',lambda:build_experiment_config.run(bundle,p,point))
        def train():
            for method in ('original_ppo','reliability_ppo','cand_rl'):
                for seed in p['train_seeds']:
                    row=launch(bundle,p,point,method,seed,'train',train_seed=seed)
                    if not Path(row['model_path']).is_file():raise FileNotFoundError(row['model_path'])
                    manifest.append(row);checkpoint()
        stage('training',train)
        def evaluate():
            for method,(_,_,category) in METHODS.items():
                for train_seed in (p['train_seeds'] if category=='learned' else [None]):
                    model=next((r['model_path'] for r in manifest if r['phase']=='train' and r['method']==method and r['train_seed']==train_seed),None)
                    for seed in p['test_seeds']:
                        manifest.append(launch(bundle,p,point,method,seed,'eval',train_seed,model));checkpoint()
        stage('online_evaluation',evaluate)
        stage('aggregate',lambda:aggregate_results.run(bundle))
        stage('validate',lambda:validate_results.run(bundle))
        stage('plots',lambda:plot_results.run(bundle))
        stage('report',lambda:generate_report.run(bundle))
        required=['resolved_experiment_config.yaml','pipeline_manifest.csv','pipeline_manifest.json','aggregated/results_long.csv',
                  'aggregated/results_summary.csv','aggregated/results_summary.json','validation_report.json','reports/experiment_report.md',
                  'experiment_report.json','commands.sh','status.tsv']
        for f in required:
            if not (bundle/f).is_file():raise FileNotFoundError(bundle/f)
        from common import METRICS
        for key in METRICS+['sla_class_breakdown']:
            for ext in ('png','pdf'):
                if not (bundle/'plots'/f'{key}.{ext}').is_file():raise FileNotFoundError(key+'.'+ext)
        seconds=time.monotonic()-start
        (bundle/'COMPLETE.txt').write_text(f'All structural stages succeeded in {seconds:.3f} seconds.\nPreliminary pipeline validation; not scientific conclusions.\n')
        print(f'COMPLETE: {bundle} ({seconds:.1f}s)',flush=True)
    except Exception as exc:
        checkpoint()
        write_json(bundle/'FAILED.json',{'error':str(exc),'elapsed_seconds':time.monotonic()-start})
        print(f'FAILED: {bundle}: {exc}',file=sys.stderr,flush=True)
        raise

if __name__=='__main__':main()
