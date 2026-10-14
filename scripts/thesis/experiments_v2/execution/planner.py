"""Pure job-matrix construction and device-aware execution summaries."""
import copy
import platform
import socket
import sys
from pathlib import Path
from scripts.thesis.experiments.common import ROOT


def hardware(requested):
    import torch
    available=torch.cuda.is_available()
    selected='cuda' if requested in ('auto','cuda') and available else 'cpu'
    return {'hostname':socket.gethostname(),'python_path':sys.executable,'python_version':sys.version,
            'pytorch_version':str(torch.__version__),'cuda_available':available,'requested_device':requested,
            'selected_device':selected,'cuda_device':torch.cuda.get_device_name(0) if selected=='cuda' else None,
            'platform':platform.platform(),'warning':'CUDA requested but unavailable; falling back to CPU' if requested=='cuda' and not available else None}


def jobs(resolved):
    exp=resolved['experiment'];protocol=exp['protocol'];result=[]
    for method,settings in resolved['selected_methods'].items():
        solver=settings['solver'];learned=settings['capabilities']['kind']=='learned'
        trains=protocol['training']['seeds'] if learned else [None]
        for train in trains:
            if learned:result.append(job(method,solver,'train',train,train))
            for evaluation in protocol['evaluation']['seeds']:result.append(job(method,solver,'eval',evaluation,train))
    # All training precedes evaluation, while keeping method/config identities independent.
    return [r for r in result if r['phase']=='train']+[r for r in result if r['phase']=='eval']


def job(method,solver,phase,seed,train_seed):
    run_id=f'{phase}__{method}__train_{train_seed if train_seed is not None else "none"}__seed_{seed}'
    section=Path('methods')/method/phase
    if phase=='eval' and train_seed is not None:section/=f'train_{train_seed}'
    # Preserve the existing launcher and normal Virne <solver>/<run_id> leaves.
    relative=section/solver/run_id
    return {'run_id':run_id,'method':method,'solver':solver,'phase':phase,'seed':seed,
            'train_seed':train_seed,'eval_seed':seed if phase=='eval' else None,
            'test_seed':seed if phase=='eval' else None,'relative_directory':str(relative),
            'relative_output_root':str(section)}


def summary(resolved,device=None):
    planned=jobs(resolved);exp=resolved['experiment'];device=device or hardware(exp['runtime']['device'])
    train=sum(j['phase']=='train' for j in planned)
    learned_eval=sum(j['phase']=='eval' and j['train_seed'] is not None for j in planned)
    return {'experiment':exp['experiment'],'environment':resolved['resolved_environment'],
            'environment_config_hash':resolved['environment_config_hash'],
            'methods':{n:{'solver':m['solver'],'label':m['label'],**m['capabilities']} for n,m in resolved['selected_methods'].items()},
            'protocol':exp['protocol'],'hardware':device,'calibrated_values':exp.get('calibration_result',{}).get('selected',{}),
            'calibrated_overrides':resolved['calibrated_overrides'],'warnings':resolved['warnings']+([device['warning']] if device['warning'] else []),
            'job_counts':{'training':train,'learned_evaluation':learned_eval,'heuristic_evaluation':len(planned)-train-learned_eval,'total':len(planned)},
            'jobs':planned,'calibration_executed':False}


def final_config(resolved,job,bundle,selected_device):
    config=copy.deepcopy(resolved['selected_methods'][job['method']]['virne_config'])
    root=(Path(bundle)/job['relative_output_root']).resolve()
    config['experiment'].update(seed=job['seed'],run_id=job['run_id'],save_root_dir=str(root))
    config['system']['if_offline_system']=job['phase']=='train'
    config['training']['num_train_epochs']=resolved['experiment']['protocol']['training']['epochs'] if job['phase']=='train' else 0
    config['training']['use_cuda']=selected_device=='cuda'
    config['training']['seed']=job['seed'] if job['phase']=='train' else None
    config['solver']['pretrained_model_path']=''
    if job['phase']=='eval' and job['train_seed'] is not None:
        training=next(r for r in jobs(resolved) if r['phase']=='train' and r['method']==job['method'] and r['train_seed']==job['train_seed'])
        config['solver']['pretrained_model_path']=str((Path(bundle)/training['relative_directory']/'models/model.pkl').resolve())
    return config
