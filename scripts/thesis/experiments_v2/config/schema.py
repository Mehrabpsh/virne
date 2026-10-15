"""Strict experiment schema and dotted-path primitives (Python 3.10 compatible)."""
import copy
import math
import re
import yaml


class ConfigError(ValueError):
    pass


def dealias(value):
    """Break YAML anchor identity so a method override cannot mutate other methods."""
    if isinstance(value,dict):return {k:dealias(v) for k,v in value.items()}
    if isinstance(value,list):return [dealias(v) for v in value]
    return copy.deepcopy(value)


def require(condition, message):
    if not condition:
        raise ConfigError(message)


def keys(value, allowed, path):
    require(isinstance(value, dict), f'{path} must be a mapping')
    unknown=set(value)-set(allowed)
    require(not unknown, f'Unknown {path} settings: {sorted(unknown)}')


def get_path(data, path):
    value=data
    for key in path.split('.'):
        try:
            value=value[int(key)] if isinstance(value, list) else value[key]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ConfigError(f'Unknown configuration path: {path}') from exc
    return value


def set_path(data, path, value):
    parts=path.split('.')
    parent=get_path(data,'.'.join(parts[:-1])) if len(parts)>1 else data
    key=parts[-1]
    if isinstance(parent,list):
        require(key.isdigit() and int(key)<len(parent), f'Unknown configuration path: {path}')
        parent[int(key)]=copy.deepcopy(value)
    else:
        require(isinstance(parent,dict) and key in parent,f'Unknown configuration path: {path}')
        parent[key]=copy.deepcopy(value)


def parse_assignments(items):
    parsed=[]
    for item in items:
        require('=' in item, f'Expected KEY=VALUE: {item}')
        key,value=item.split('=',1)
        require(bool(key) and not key.startswith(('+','~')), f'Invalid override path: {key}')
        try:
            value=yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ConfigError(f'Invalid YAML override: {item}') from exc
        parsed.append((key,value))
    return parsed


def positive(value, name, integer=False):
    require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>0,
            f'{name} must be a concrete positive number')
    if integer:
        require(isinstance(value,int), f'{name} must be an integer')


def seed_list(values, name):
    require(isinstance(values,list) and bool(values), f'{name} must be a nonempty seed list')
    require(all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in values),f'{name}: seeds must be nonnegative integers')
    require(len(set(values))==len(values), f'{name}: duplicate seeds')


def validate(data):
    keys(data, ['experiment','runtime','environment','calibration','protocol','learning_defaults','methods',
                'metrics','reporting','calibration_result','calibration_suggestions'], 'experiment schema')
    for name in ('experiment','runtime','environment','protocol','learning_defaults','methods','metrics','reporting'):
        require(name in data, f'Missing required section: {name}')
    keys(data['experiment'],['name','description'],'experiment')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',str(data['experiment'].get('name',''))) is not None,'Invalid experiment.name')
    keys(data['runtime'],['device','threads','results_root','tensorboard'],'runtime')
    tb=data['runtime'].get('tensorboard',{'enabled':False,'flush_secs':10,'request_interval':1})
    keys(tb,['enabled','flush_secs','request_interval'],'runtime.tensorboard')
    require(isinstance(tb['enabled'],bool),'runtime.tensorboard.enabled must be boolean')
    positive(tb['flush_secs'],'runtime.tensorboard.flush_secs',True)
    positive(tb['request_interval'],'runtime.tensorboard.request_interval',True)
    if tb['enabled']:
        import importlib.util
        require(importlib.util.find_spec('tensorboard') is not None,'TensorBoard enabled but tensorboard is not installed')
    require(data['runtime'].get('device') in ('auto','cpu','cuda'),'runtime.device must be auto/cpu/cuda')
    positive(data['runtime'].get('threads'),'runtime.threads',True)
    require(isinstance(data['runtime'].get('results_root'),str),'runtime.results_root is required')
    keys(data['environment'],['substrate','workload','resources','reliability','sla'],'environment')
    for name in ('substrate','workload','resources','reliability','sla'):
        require(name in data['environment'],f'Missing environment.{name}')
    keys(data['protocol'],['training','evaluation'],'protocol')
    keys(data['protocol']['training'],['seeds','epochs','online'],'protocol.training')
    keys(data['protocol']['evaluation'],['seeds','online'],'protocol.evaluation')
    train=data['protocol']['training'];evaluation=data['protocol']['evaluation']
    seed_list(train['seeds'],'training');seed_list(evaluation['seeds'],'evaluation')
    require(set(train['seeds']).isdisjoint(evaluation['seeds']),'Training/evaluation seeds must be disjoint')
    positive(train['epochs'],'protocol.training.epochs',True)
    require(isinstance(train.get('online',False),bool),'protocol.training.online must be boolean')
    require(isinstance(evaluation.get('online'),bool),'protocol.evaluation.online must be boolean')
    require(evaluation['online'], 'OfflineSystem evaluation is not supported cleanly: Virne resets and scales substrate resources by 0.75 between requests while retaining the arrival/departure ledger. Use protocol.evaluation.online=true.')
    keys(data['learning_defaults'],['batch_size','target_steps','repeat_times','network','learning_rate'],'learning_defaults')
    require(isinstance(data['methods'],dict) and bool(data['methods']),'methods must be a nonempty mapping')
    for name,method in data['methods'].items():
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',name) is not None,f'Unsafe method name: {name}')
        keys(method,['enabled','label','solver','kind','awareness','objective','reward','reliability_features',
                     'candidate','network','endpoints','profile','redundancy','ra_heuristic','ra_evne','history','overrides'],f'methods.{name}')
        require(isinstance(method.get('enabled'),bool),f'{name}.enabled must be explicit boolean')
        require(isinstance(method.get('solver'),str),f'{name}.solver is required')
        require(method.get('kind','auto') in ('auto','learned','heuristic'),f'{name}: invalid kind')
    require(isinstance(data['metrics'],list) and bool(data['metrics']) and len(set(data['metrics']))==len(data['metrics']),'metrics must be a nonempty unique list')
    keys(data['reporting'],['plots','formats','generate_markdown_report','assessment_beta'],'reporting')
    for flag in ('plots','generate_markdown_report'):
        require(isinstance(data['reporting'].get(flag),bool),f'reporting.{flag} must be boolean')
    formats=data['reporting'].get('formats')
    require(isinstance(formats,list) and bool(formats) and set(formats)<= {'png','pdf'},'reporting.formats must contain png/pdf')
    positive(data['reporting'].get('assessment_beta'),'reporting.assessment_beta')
    calibration=data.get('calibration',{})
    keys(calibration,['methods','seeds','requests','search_space','criteria','beta_suggestion','adopt_beta_for'],'calibration')
    if calibration:
        seed_list(calibration['seeds'],'calibration')
        require(set(calibration['seeds']).isdisjoint(train['seeds']+evaluation['seeds']),'Calibration seeds must be disjoint from training/evaluation seeds')
    return data
