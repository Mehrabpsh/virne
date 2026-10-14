"""Translate experiment semantics into existing, validated Virne configuration keys."""
import copy
import math
import numpy as np
from omegaconf import OmegaConf
from .schema import ConfigError, get_path, keys, positive, require, set_path

PROTECTED=('p_net_setting','v_sim_setting','reliability','sla')
OWNED=('experiment','simulation','hydra','system','solver.solver_name','solver.pretrained_model_path',
       'solver.pretrained_subsolver_model_path','training.num_train_epochs','training.seed','training.use_cuda',
       'training.num_workers','training.if_use_random_training_seed','training.distributed_training',
       'training.save_interval','training.eval_interval','training.model_dir_name','recorder','logger')


def under(path, prefixes):
    return any(path==p or path.startswith(p+'.') or p.startswith(path+'.') for p in prefixes)


def put(config, path, value):
    current=get_path(config,path)
    if isinstance(current,dict) and isinstance(value,dict):
        for key,item in value.items():put(config,path+'.'+key,item)
    else:set_path(config,path,value)


def rate(value, name):
    if isinstance(value,dict):
        if value.get('distribution')=='fixed':
            keys(value,['distribution','value'],name);positive(value.get('value'),name)
            return float(value['value'])
        keys(value,['distribution','low','high'],name)
        require(value.get('distribution')=='uniform',f'{name}: only scalar, fixed or uniform rates are supported; scale pseudo-keys are not supported')
        positive(value.get('low'),name+'.low');positive(value.get('high'),name+'.high')
        require(value['low']<=value['high'],f'{name}: low exceeds high')
        return copy.deepcopy(value)
    positive(value,name)
    return float(value)


def common_environment(base, environment):
    config=copy.deepcopy(base)
    keys(environment['substrate'],['topology'],'environment.substrate')
    topology=environment['substrate']['topology']
    for key,value in topology.items():put(config,'p_net_setting.topology.'+key,value)
    positive(config['p_net_setting']['topology']['num_nodes'],'substrate num_nodes',True)
    work=environment['workload']
    keys(work,['num_requests','topology','vnr_size','arrival_rate','lifetime'],'environment.workload')
    positive(work.get('num_requests'),'num_requests',True)
    put(config,'v_sim_setting.num_v_nets',work['num_requests'])
    for key,value in work.get('topology',{}).items():put(config,'v_sim_setting.topology.'+key,value)
    for section in ('vnr_size','lifetime'):
        require(section in work,f'workload.{section} is required')
        target='v_net_size' if section=='vnr_size' else 'lifetime'
        for key,value in work[section].items():put(config,'v_sim_setting.'+target+'.'+key,value)
    positive(work.get('arrival_rate'),'arrival_rate')
    put(config,'v_sim_setting.arrival_rate.rate',work['arrival_rate'])
    for owner,group in [('physical_node',('p_net_setting','node_attrs_setting')),('physical_link',('p_net_setting','link_attrs_setting')),
                        ('virtual_node',('v_sim_setting','node_attrs_setting')),('virtual_link',('v_sim_setting','link_attrs_setting'))]:
        resources=environment['resources']
        keys(resources,['physical_node','physical_link','virtual_node','virtual_link'],'environment.resources')
        for name,spec in resources.get(owner,{}).items():
            attrs=config[group[0]][group[1]]
            matches=[i for i,a in enumerate(attrs) if a['name']==name and a['type']=='resource']
            require(len(matches)==1,f'Unknown {owner} resource {name!r}')
            for key,value in spec.items():put(config,f'{group[0]}.{group[1]}.{matches[0]}.{key}',value)
    rel=environment['reliability']
    keys(rel,['mode','backend','config_id','nodes','links'],'environment.reliability')
    require(rel.get('mode') in ('node_only','node_and_link'),'Invalid reliability mode')
    require(rel.get('backend','analytic')=='analytic','Only the frozen analytic reliability backend is supported')
    for key in ('mode','backend','config_id'):
        if key in rel:config['reliability'][key]=rel[key]
    for group in ('nodes','links'):
        keys(rel[group],['lambda_f','mu_r'],f'reliability.{group}')
        for key in ('lambda_f','mu_r'):config['reliability'][group][key]=rate(rel[group].get(key),f'{group}.{key}')
    sla=environment['sla']
    keys(sla,['levels','probabilities','assignment','values'],'environment.sla')
    config['sla'].update(copy.deepcopy(sla))
    config['reliability']['enabled']=True;config['sla']['enabled']=True
    return config


def learning(config, defaults):
    mapping={'batch_size':'training.batch_size','target_steps':'rl.target_steps','repeat_times':'rl.repeat_times'}
    for key,target in mapping.items():
        if key in defaults:positive(defaults[key],key,True);put(config,target,defaults[key])
    for key,value in defaults.get('network',{}).items():put(config,'nn.'+key,value)
    for key,value in defaults.get('learning_rate',{}).items():
        positive(value,'rl.learning_rate.'+key);put(config,'rl.learning_rate.'+key,value)


def method_config(common, method):
    config=copy.deepcopy(common)
    awareness=method.get('awareness',{'reliability':False,'sla':False})
    keys(awareness,['reliability','sla'],'method awareness')
    require(all(isinstance(awareness.get(k),bool) for k in ('reliability','sla')),'Awareness flags must be explicit booleans')
    require(awareness['reliability']==awareness['sla'],'Frozen execution supports both reliability/SLA enabled, or both disabled with post-hoc assessment')
    config['reliability']['enabled']=awareness['reliability'];config['sla']['enabled']=awareness['sla']
    config['reliability_features']={'enabled':bool(method.get('reliability_features',[])),'names':method.get('reliability_features',[])}
    objective=method.get('objective',{'name':'original'})
    keys(objective,['name','alpha','beta'],'method objective')
    if objective.get('name')=='original':
        require(not awareness['reliability'],'Enabled reliability requires an explicit joint objective for the frozen recorder')
        config['joint_objective']={'name':None,'alpha':None,'beta':None}
    else:
        config['joint_objective']={'name':objective.get('name'),'alpha':objective.get('alpha'),'beta':objective.get('beta')}
    reward=method.get('reward',{'name':'fixed_intermediate'})
    keys(reward,['name','intermediate_reward'],'method reward')
    config['rl']['reward_calculator'].update(reward)
    solver=method['solver'];config['solver']['solver_name']=solver
    if solver=='cand_rl':
        for key,value in method.get('candidate',{}).items():put(config,'cand_rl.'+key,value)
        for key,value in method.get('endpoints',{}).items():put(config,'cand_rl.'+key,value)
        for key,value in method.get('network',{}).items():put(config,'cand_rl.'+key,value)
        for key in ('profile','redundancy'):
            if key in method:put(config,'cand_rl.'+key,method[key])
    else:
        require(not any(k in method for k in ('candidate','endpoints','profile','redundancy')),f'{solver}: Cand structured settings require cand_rl')
        for key,value in method.get('network',{}).items():put(config,'nn.'+key,value)
    for group in ('ra_heuristic','ra_evne'):
        for key,value in method.get(group,{}).items():put(config,group+'.'+key,value)
    if 'history' in method:
        keys(method['history'],['initialization'],'history')
        require(method['history']['initialization']=='zero','Only explicit static zero history initialization is supported')
        for key in ('ages','failure_counts','last_failure_positions'):
            config['ra_evne'][key]=[0]*config['p_net_setting']['topology']['num_nodes']
    for path,value in method.get('overrides',{}).items():
        require(not under(path,PROTECTED),f'Method override modifies protected environment: {path}; use environment or global --virne-set')
        require(not under(path,OWNED),f'Method override modifies runner-owned field: {path}')
        put(config,path,value)
    return config


def environment_contract(config):
    contract={key:copy.deepcopy(config[key]) for key in PROTECTED}
    # Disabled baseline execution is deliberate; the assessment world still exists.
    contract['reliability']['enabled']=True;contract['sla']['enabled']=True
    return contract


def validate_virne(config):
    from virne.reliability.config import validate_reliability_config
    from virne.reliability.model import DiscreteRequestSLAModel
    cfg=OmegaConf.create(config)
    try:
        validate_reliability_config(cfg)
        from virne.solver.learning.rl_core import thesis_reward
        from virne.solver.learning.rl_core.reward_calculator import RewardCalculatorRegistry
        RewardCalculatorRegistry.get(cfg.rl.reward_calculator.name)(cfg)
        # Validate common SLA probabilities even for a disabled-awareness baseline.
        if cfg.sla.assignment=='explicit':
            model=DiscreteRequestSLAModel(cfg.sla)
            require(cfg.sla.values is not None and len(cfg.sla.values)==cfg.v_sim_setting.num_v_nets,'Explicit SLA values must match request count')
            require(all(float(v) in model.levels for v in cfg.sla.values),'Explicit SLA values must belong to configured levels')
        else:DiscreteRequestSLAModel(cfg.sla).assign([],seed=0)
        from virne.utils.dataset import resolve_poisson_arrival_setting
        resolve_poisson_arrival_setting(cfg.v_sim_setting.arrival_rate)
        for group in ('p_net_setting','v_sim_setting'):
            topology=config[group]['topology']
            require(topology['type'] in ('path','star','waxman','random'),'Unsupported v2 topology: '+str(topology['type']))
            if topology['type']=='waxman':
                positive(topology.get('wm_alpha',.5),'wm_alpha')
                require(0<topology.get('wm_beta',.2)<=1,'wm_beta must be in (0,1]')
            if topology['type']=='random':require(0<topology.get('random_prob',.5)<=1,'random_prob must be in (0,1]')
        require(cfg.v_sim_setting.v_net_size.distribution=='uniform' and cfg.v_sim_setting.v_net_size.dtype=='int','v2 VNR size uses uniform integer generation')
        require(cfg.v_sim_setting.lifetime.distribution=='exponential','v2 lifetime uses the existing exponential process')
        for section in ('node_attrs_setting','link_attrs_setting'):
            physical={a['name'] for a in config['p_net_setting'][section] if a['type']=='resource'}
            virtual={a['name'] for a in config['v_sim_setting'][section] if a['type']=='resource'}
            require(virtual<=physical,f'Virtual resource names absent from physical resources: {virtual-physical}')
        for path in ('training.batch_size','rl.target_steps','rl.repeat_times','nn.embedding_dim','nn.num_gnn_layers'):
            positive(get_path(config,path),path,True)
        require(0<=cfg.rl.gamma<=1,'rl.gamma must be in [0,1]')
        for group in ('nodes','links'):
            for key in ('lambda_f','mu_r'):
                value=config['reliability'][group][key]
                require(rate(value,f'reliability.{group}.{key}')==value,'Direct Virne fixed-rate mappings are unsupported; use a scalar or the experiment-level fixed adapter')
        for group in ('p_net_setting','v_sim_setting'):
            for section in ('node_attrs_setting','link_attrs_setting'):
                for attr in config[group][section]:
                    if attr.get('type')=='resource':
                        require(attr['distribution']=='uniform','Only uniform resource distributions are exposed in v2')
                        require(0<=attr['low']<=attr['high'],f'Invalid resource bounds: {group}.{attr["name"]}')
        size=config['v_sim_setting']['v_net_size'];positive(size['low'],'vnr size low',True);positive(size['high'],'vnr size high',True)
        require(size['low']<=size['high'],'vnr_size.low exceeds high')
        positive(config['v_sim_setting']['num_v_nets'],'num_v_nets',True)
        positive(config['p_net_setting']['topology']['num_nodes'],'num_nodes',True)
        positive(config['v_sim_setting']['arrival_rate']['rate'],'arrival rate')
        positive(config['v_sim_setting']['lifetime']['scale'],'lifetime scale')
        if config['solver']['solver_name']=='cand_rl':
            from virne.solver.learning.reinforcement_learning.cand_rl_solver import CandidateSelector, CandRLObservationBuilder
            CandidateSelector(cfg.cand_rl);CandRLObservationBuilder(cfg)
            require(cfg.cand_rl.profile=='thesis_joint' and cfg.cand_rl.redundancy=='none','Cand requires thesis_joint and redundancy=none')
            require(cfg.rl.reward_calculator.name.startswith('joint_'),'Cand requires an explicit joint reward')
            require(isinstance(cfg.cand_rl.mask_padding,bool),'Cand mask_padding must be boolean')
            positive(cfg.cand_rl.k,'cand k',True)
            pn_names={a['name'] for a in config['p_net_setting']['node_attrs_setting']}
            vn_names={a['name'] for a in config['v_sim_setting']['node_attrs_setting']}
            require(set(cfg.cand_rl.resource_names)<=pn_names & vn_names and set(cfg.cand_rl.capacity_attributes)<=pn_names,'Cand resource/capacity attributes absent from environment')
            require(bool(cfg.cand_rl.hidden_dims) and all(isinstance(v,int) and v>0 for v in cfg.cand_rl.hidden_dims),'Cand hidden_dims required')
            require(all(isinstance(v,int) and not isinstance(v,bool) for v in (cfg.cand_rl.source,cfg.cand_rl.destination)),'Cand endpoints must be physical integer node IDs')
            require(0<=cfg.cand_rl.source<config['p_net_setting']['topology']['num_nodes'] and 0<=cfg.cand_rl.destination<config['p_net_setting']['topology']['num_nodes'],'Cand endpoints outside substrate')
        if config['solver']['solver_name'] in ('ra_nrm','ra_grc'):
            from virne.solver.heuristic.reliability_rank import combine_rank
            require(config['reliability']['enabled'],'RA ranking requires reliability awareness')
            combine_rank({}, {},cfg.ra_heuristic.combine,cfg.ra_heuristic.alpha,cfg.ra_heuristic.normalization)
        if config['solver']['solver_name'] in ('ra_evne_a','ra_evne_original'):
            positive(cfg.ra_evne.empty_distance,'ra_evne.empty_distance')
            for group in ('p_net_setting','v_sim_setting'):
                require(cfg.ra_evne.cpu_attribute in {a['name'] for a in config[group]['node_attrs_setting']} and cfg.ra_evne.bandwidth_attribute in {a['name'] for a in config[group]['link_attrs_setting']},'RA-EVNE resource attributes absent from environment')
            if config['solver']['solver_name']=='ra_evne_original':
                positive(cfg.ra_evne.weibull_scale,'weibull_scale');positive(cfg.ra_evne.weibull_shape,'weibull_shape')
                for key in ('ages','failure_counts','last_failure_positions'):
                    values=getattr(cfg.ra_evne,key)
                    require(values is not None and len(values)==cfg.p_net_setting.topology.num_nodes,f'RA-EVNE {key} must match PN size')
    except (ValueError,TypeError,KeyError,AttributeError,NotImplementedError) as exc:
        raise ConfigError(f'Incomplete/invalid Virne config for {config["solver"]["solver_name"]}: {exc}') from exc
