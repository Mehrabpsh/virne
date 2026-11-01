"""Reuse the verified Bash launcher; override every composed value explicitly."""
import json
import os
from pathlib import Path
import sys
from scripts.thesis.experiments.common import command, override, write_yaml
from .planner import final_config, system_mode


def flatten(data,prefix=''):
    for key,value in data.items():
        path=prefix+'.'+key if prefix else key
        # Hydra's base rate fields are null scalars: replace them atomically,
        # rather than emitting nonexistent dotted distribution/value subkeys.
        rate_mapping = path in {f'reliability.{group}.{rate}'
            for group in ('nodes','links') for rate in ('lambda_f','mu_r')}
        if isinstance(value,dict) and value and not rate_mapping:
            yield from flatten(value,path)
        else:
            yield path,value


def hydra_value(value):
    if isinstance(value,dict):return "{"+",".join(str(k)+":"+hydra_value(v) for k,v in value.items())+"}"
    if isinstance(value,list):return "["+",".join(hydra_value(v) for v in value)+"]"
    return json.dumps(value,allow_nan=False)


def launch(bundle,resolved,job,device):
    method=resolved['selected_methods'][job['method']]
    config=final_config(resolved,job,bundle,device)
    directory=Path(bundle)/job['relative_directory']
    env={'PYTHON':sys.executable,'SOLVER':job['solver'],'RUN_ID':job['run_id'],
         'OUTPUT_ROOT':config['experiment']['save_root_dir'],'PROFILE':'configured',
         'SEED':str(job['seed']),'EPOCHS':str(config['training']['num_train_epochs']),
         'OFFLINE':str(config['system']['if_offline_system']).lower(),'MODEL_PATH':config['solver']['pretrained_model_path']}
    excluded={'experiment.run_id','experiment.save_root_dir'}
    arguments=[k+'='+hydra_value(v) for k,v in flatten(config) if k not in excluded]
    print('  '+job['run_id'],flush=True)
    command(bundle,job['run_id'],['bash','scripts/thesis/run.sh',*arguments],env)
    # Fail at the first mismatched job, as well as in bundle validation.
    from omegaconf import OmegaConf
    from ..analysis.validate import mode_errors
    audit=json.loads((directory/'runtime_system.json').read_text())
    errors=mode_errors(resolved,job,OmegaConf.load(directory/'config.yaml'),audit)
    if errors:raise ValueError('; '.join(errors))
    model=config['solver']['pretrained_model_path'] or (str(directory/'models/model.pkl') if job['phase']=='train' else '')
    result={**job,'category':method['capabilities']['kind'],'label':method['label'],'model_path':model,
            'run_directory':str(directory.resolve()),'completed':True,
            'system':system_mode(resolved,job['phase']),'tensorboard_directory':config['logger']['tensorboard_dir']}
    if job['phase']=='train' and not Path(model).is_file():raise FileNotFoundError(f'Training checkpoint missing: {model}')
    return result
