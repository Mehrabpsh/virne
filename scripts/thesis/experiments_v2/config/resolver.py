"""Load immutable input, apply typed overrides, translate and validate every method."""
import copy
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import yaml
from .schema import dealias, ConfigError, get_path, parse_assignments, require, set_path, validate
from .adapters import OWNED, common_environment, environment_contract, learning, method_config, put, under, validate_virne
from ..execution.capabilities import inspect_solver
from scripts.thesis.experiments.common import ROOT, digest


def resolve(path, sets=(), virne_sets=(), subset=None, *, document=None):
    path=Path(path).resolve()
    original=path.read_text()
    try:data=yaml.safe_load(original)
    except yaml.YAMLError as exc:raise ConfigError(f'Invalid experiment YAML: {path}: {exc}') from exc
    data=dealias(document if document is not None else data)
    # Materialize compatible defaults before dotted overrides are applied.
    protocol=data.get('protocol',{}) if isinstance(data,dict) else {}
    training=protocol.get('training',{}) if isinstance(protocol,dict) else {}
    if isinstance(training,dict):training.setdefault('online',False)
    runtime=data.get('runtime',{}) if isinstance(data,dict) else {}
    if isinstance(runtime,dict):runtime.setdefault('tensorboard',{})
    if isinstance(runtime,dict) and isinstance(runtime['tensorboard'],dict):
        for key,value in {'enabled':False,'flush_secs':10,'request_interval':1}.items():
            runtime['tensorboard'].setdefault(key,value)
    overrides=parse_assignments(sets);raw=parse_assignments(virne_sets)
    for key,value in overrides:set_path(data,key,value)
    validate(data)
    from .metrics import validate_metrics
    validate_metrics(data['metrics'])
    selected=[name for name,m in data['methods'].items() if m['enabled']]
    if subset is not None:
        requested=subset.split(',') if isinstance(subset,str) else list(subset)
        require(len(set(requested))==len(requested),'Duplicate method subset names')
        require(set(requested)<=set(data['methods']),f'Unknown method subset: {set(requested)-set(data["methods"])}')
        require(all(data['methods'][m]['enabled'] for m in requested),'Cannot select disabled methods; use --set methods.NAME.enabled=true')
        selected=requested
    require(bool(selected),'No methods selected')
    for key,_ in raw:
        require(not under(key,OWNED),f'Raw override attempts runner-owned field: {key}')
        require(key not in ('reliability.enabled','sla.enabled'),'Use method awareness flags; raw overrides cannot alter the verified baseline execution switch')
    with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'virne/configs')):
        base=OmegaConf.to_container(compose(config_name='main'),resolve=True)
    common=common_environment(base,data['environment'])
    learning(common,data['learning_defaults'])
    expected=copy.deepcopy(common)
    for key,value in raw:put(expected,key,value)
    expected_contract=environment_contract(expected)
    method_common=copy.deepcopy(common)
    for key,value in raw:
        if under(key,('p_net_setting','v_sim_setting','reliability','sla')):put(method_common,key,value)
    methods={};warnings=[]
    for name in selected:
        spec=data['methods'][name]
        cap=inspect_solver(spec['solver'],spec.get('kind','auto'))
        config=method_config(method_common,spec)
        for key,value in raw:put(config,key,value)
        require(environment_contract(config)==expected_contract,f'{name}: resolved protected environment differs across methods')
        # Deterministic lifecycle controls override Virne defaults and are runner-owned.
        config['training'].update(num_workers=1,if_use_random_training_seed=False,distributed_training=False,
                                  save_interval=1,eval_interval=1)
        config['experiment'].update(num_simulations=1,if_load_p_net=False,if_load_v_nets=False,if_save_p_net=False,if_save_v_nets=False,if_save_config=True)
        config['system'].update(if_changeable_v_nets=False,if_time_window=False,renew_v_net_simulator=False)
        config['recorder'].update(if_save_records=True,if_temp_save_records=True)
        config['logger'].update(backends=['console','file'],level='INFO' if cap['kind']=='learned' else 'WARNING')
        if data['runtime']['tensorboard']['enabled']:
            config['logger']['backends'].append('tensorboard')
        validate_virne(config)
        methods[name]={'label':spec.get('label',name),'solver':spec['solver'],'capabilities':cap,'virne_config':config}
        if cap['warning']:warnings.append(f'{name}: {cap["warning"]}')
    calibrated=data.get('calibration_result',{}).get('selected',{})
    changes=[]
    for key,old in calibrated.items():
        value=get_path(data,key)
        if value!=old:changes.append({'path':key,'calibrated':old,'runtime':value,'source':'experiment override'})
    # Compare raw global environment changes against the calibrated experiment translation.
    calibration_paths=[]
    if raw and calibrated:
        before_contract=environment_contract(common)
        if expected_contract!=before_contract:
            for key,value in raw:
                if under(key,('p_net_setting','v_sim_setting','reliability','sla')):
                    changes.append({'path':key,'calibrated':get_path(common,key),'runtime':value,'source':'raw global Virne override; calibrated environment changed'})
    if changes:warnings.append('CALIBRATED PARAMETER OVERRIDDEN: '+str(changes))
    result={'schema_version':2,'source_config':str(path),'experiment':data,'resolved_environment':expected_contract,
            'environment_config_hash':digest(expected_contract),'selected_methods':methods,'warnings':warnings,
            'calibrated_overrides':changes,'cli_overrides':[{'path':k,'value':v} for k,v in overrides],
            'raw_virne_overrides':[{'path':k,'value':v} for k,v in raw]}
    return result,original
