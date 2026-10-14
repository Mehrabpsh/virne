"""Shared provenance, journaling and subprocess environment for the two tools."""
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
import signal
from scripts.thesis.experiments.common import ROOT, write_json, write_yaml


def new_bundle(resolved,output=None,suffix=''):
    exp=resolved['experiment']
    timestamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root=Path(exp['runtime']['results_root'])
    if not root.is_absolute():root=ROOT/root
    bundle=(Path(output) if output else root/f'{exp["experiment"]["name"]}_{timestamp}{suffix}').resolve()
    bundle.mkdir(parents=True,exist_ok=False)
    for folder in ('logs','reports'):(bundle/folder).mkdir()
    print(f'Results directory: {bundle}',flush=True)
    threads=str(exp['runtime']['threads'])
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]=threads
    os.environ['PYTHON']=sys.executable
    os.environ['PYTHONPATH']=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')
    os.environ['MPLCONFIGDIR']=str(bundle/'matplotlib_cache')
    # torch may already be imported by dry-run/preflight capability resolution.
    import torch
    torch.set_num_threads(exp['runtime']['threads'])
    lines='#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'
    lines+=''.join('export '+key+'='+shlex.quote(os.environ[key])+'\n' for key in ('PYTHON','PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','MPLCONFIGDIR'))
    (bundle/'commands.sh').write_text(lines)
    (bundle/'status.tsv').write_text('step\tstatus\truntime_seconds\n')
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
    (bundle/'source_commit.txt').write_text(git('rev-parse','HEAD'))
    (bundle/'source_status.txt').write_text(git('status','--short'))
    (bundle/'tracked_changes.patch').write_text(git('diff','--binary','HEAD'))
    for name in ('experiments','experiments_v2'):
        shutil.copytree(ROOT/'scripts/thesis'/name,bundle/'framework_source'/name,ignore=shutil.ignore_patterns('__pycache__'))
    return bundle


def freeze(bundle,resolved,original,summary):
    bundle=Path(bundle)
    (bundle/'input_config.yaml').write_text(original)
    write_yaml(bundle/'cli_overrides.yaml',resolved['cli_overrides'])
    write_yaml(bundle/'raw_virne_overrides.yaml',resolved['raw_virne_overrides'])
    write_yaml(bundle/'resolved_config.yaml',resolved)
    write_yaml(bundle/'execution_summary.yaml',summary)
    write_json(bundle/'hardware.json',summary['hardware'])
    write_json(bundle/'source_metadata.json',{'command_line':shlex.join(sys.argv),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()})


def stage(bundle,name,function):
    print('Stage: '+name,flush=True);start=time.monotonic()
    try:result=function()
    except BaseException:
        with (Path(bundle)/'status.tsv').open('a') as f:f.write(f'{name}\tfailed\t{time.monotonic()-start:.3f}\n')
        raise
    with (Path(bundle)/'status.tsv').open('a') as f:f.write(f'{name}\tpassed\t{time.monotonic()-start:.3f}\n')
    return result


def install_signals():
    def interrupted(signum,frame):raise RuntimeError(f'Execution interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
