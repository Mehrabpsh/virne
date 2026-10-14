"""Pipeline-only adapters around the frozen launcher, generator and metrics."""
import ast
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
METHODS = {
    'original_ppo': ('ppo_dual_gat+', 'PPO-DualGAT', 'learned'),
    'reliability_ppo': ('ppo_dual_gat+', 'Reliability-aware PPO-DualGAT', 'learned'),
    'cand_rl': ('cand_rl', 'Cand-RL', 'learned'),
    'ffd': ('ffd_rank', 'FFD', 'heuristic'),
    'nrm': ('nrm_rank', 'NRM', 'heuristic'),
    'grc': ('grc_rank', 'GRC', 'heuristic'),
    'rw': ('rw_rank', 'RW', 'heuristic'),
    'ra_nrm': ('ra_nrm', 'RA-NRM', 'adaptation'),
    'ra_grc': ('ra_grc', 'RA-GRC', 'adaptation'),
    'ra_evne_a': ('ra_evne_a', 'RA-EVNE-A', 'adaptation'),
    'ra_evne_original': ('ra_evne_original', 'RA-EVNE-original', 'reconstruction'),
}
METRICS = ['acceptance_rate', 'avg_r2c_ratio', 'long_term_r2c_ratio',
           'long_term_time_r2c_ratio', 'sla_satisfaction_rate',
           'sla_service_rate_all_arrivals', 'mean_delivered_availability',
           'mean_sla_shortfall', 'median_sla_shortfall', 'max_inservice_count']

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def write_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(clean(value), sort_keys=False))


def override(key, value):
    return key+'='+json.dumps(value, separators=(',', ':'))


def settings(p, point, baseline=False):
    d = {
        'p_net_setting.topology.num_nodes': p['pn_nodes'],
        'v_sim_setting.num_v_nets': p['requests'],
        'v_sim_setting.v_net_size.low': p['vn_size'][0],
        'v_sim_setting.v_net_size.high': p['vn_size'][1],
        'v_sim_setting.arrival_rate.rate': point['arrival_rate'],
        'v_sim_setting.lifetime.scale': p['lifetime_scale'],
        'reliability.enabled': not baseline, 'reliability.mode': 'node_and_link',
        'reliability.config_id': 'PIPELINE_PROVISIONAL',
        'reliability.nodes.lambda_f': point['lambda_f'], 'reliability.nodes.mu_r': p['mu_r'],
        'reliability.links.lambda_f': point['lambda_f']*p['link_failure_multiplier'],
        'reliability.links.mu_r': p['mu_r'],
        'sla.enabled': not baseline, 'sla.levels': [0.99,0.999,0.9995,0.9999],
        'sla.probabilities': [0.25]*4,
        'joint_objective.name': None if baseline else p['objective'],
        'joint_objective.beta': None if baseline else point.get('beta',1.0),
        'reliability_features.enabled': False,
        'reliability_features.names': [],
        'rl.reward_calculator.name': 'fixed_intermediate' if baseline else p['reward'],
        'training.use_cuda': bool(p['selected_device']=='cuda'),
        'experiment.if_load_p_net': False, 'experiment.if_load_v_nets': False,
    }
    d.update(p['resource_overrides'])
    d.update(p['learning_overrides'])
    return d


def command(bundle, label, args, env=None):
    """Journal before execution and persist failures; never skip failed jobs."""
    bundle=Path(bundle)
    with (bundle/'commands.sh').open('a') as f:
        prefix = ['env']+[k+'='+str(v) for k,v in (env or {}).items()] if env else []
        f.write(shlex.join(prefix+list(map(str,args)))+'\n')
    log=bundle/'logs'/f'{label}.log'
    start=time.monotonic()
    with log.open('w') as stream:
        result=subprocess.run(args,cwd=ROOT,env={**os.environ,**(env or {})},stdout=stream,stderr=subprocess.STDOUT)
    with (bundle/'status.tsv').open('a') as f:
        f.write(f'{label}\t'+('passed' if result.returncode==0 else 'failed')+f'\t{time.monotonic()-start:.3f}\n')
    if result.returncode:
        raise RuntimeError(f'{label} failed ({result.returncode}); {log}\n'+log.read_text()[-6000:])


def launch(bundle, p, point, method, seed, phase, train_seed=None, model=None, root=None):
    solver, label, category=METHODS[method]
    run_id=f'{point["point_id"]+"__" if phase=="calibration" else ""}{phase}__{method}__train_{train_seed if train_seed is not None else "none"}__seed_{seed}'
    output=Path(root or (Path(bundle)/'runs'/method))
    d=settings(p,point,baseline=method=='original_ppo')
    if method=='reliability_ppo':
        d.update({'reliability_features.enabled':True,'reliability_features.names':p['features']})
    if phase=='train':d['logger.level']='INFO'
    c=p['comparators']
    if method in ('ra_nrm','ra_grc','cand_rl'):
        d.update(c[method])
    if method.startswith('ra_evne'):
        d.update(c['evne'])
    if method=='ra_evne_original':
        d.update({k:v for k,v in c['evne_original'].items() if k!='history_value'})
        for key in ('ages','failure_counts','last_failure_positions'):
            d['ra_evne.'+key]=[c['evne_original']['history_value']]*p['pn_nodes']
    env={'SOLVER':solver,'RUN_ID':run_id,'OUTPUT_ROOT':str(output),'PROFILE':'configured',
         'PYTHON':os.environ['PYTHON'],'SEED':str(seed),'EPOCHS':str(p['epochs'] if phase=='train' else 0),
         'OFFLINE':'true' if phase=='train' else 'false','MODEL_PATH':str(model or '')}
    print(f'  {run_id}',flush=True)
    command(bundle,run_id,['bash','scripts/thesis/run.sh']+[override(k,v) for k,v in d.items()],env)
    run_dir=output/solver/run_id
    row={'run_id':run_id,'method':method,'label':label,'category':category,'phase':phase,
         'train_seed':train_seed,'test_seed':seed if phase!='train' else None,
         'seed':seed,'run_directory':str(run_dir),'model_path':str(model or (run_dir/'models/model.pkl' if phase=='train' else '')),
         'completed':True}
    return row


def generated(config_path, assessment=True):
    """Regenerate the exact seeded inputs; optional post-hoc reliability metadata."""
    from virne.network import PhysicalNetwork, VirtualNetworkRequestSimulator
    from virne.utils.config import add_simulation_into_config
    cfg=OmegaConf.load(config_path)
    add_simulation_into_config(cfg)
    pn=PhysicalNetwork.from_setting(cfg.p_net_setting,seed=cfg.experiment.seed)
    sim=VirtualNetworkRequestSimulator.from_setting(cfg.v_sim_setting,seed=cfg.experiment.seed)
    sim.renew(seed=cfg.experiment.seed)
    if assessment and not cfg.reliability.enabled:
        from virne.reliability.model import ReliabilityAttributeProvider, DiscreteRequestSLAModel
        ReliabilityAttributeProvider(cfg.reliability).apply(pn,seed=cfg.experiment.seed)
        DiscreteRequestSLAModel(cfg.sla).assign(sim.v_nets,seed=cfg.experiment.seed)
    return cfg,pn,sim


def input_snapshot(pn,sim):
    """Include full resource demands, topology, time and assigned SLA, in graph order."""
    def graph(g):
        return {'nodes':[(n,dict(a)) for n,a in g.nodes(data=True)],
                'edges':[(u,v,dict(a)) for u,v,a in g.edges(data=True)]}
    substrate=graph(pn)
    requests=[dict(graph(v),id=v.id,arrival_time=v.arrival_time,lifetime=v.lifetime,
                   sla=v.graph.get('availability_requirement')) for v in sim.v_nets]
    return {'substrate':substrate,'requests':requests}


def digest(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def collect_row(row, bundle, *, assessment_beta=None, assessment_mode=None):
    from virne.core.counter import Counter
    from virne.reliability.evaluator import ReliabilityEvaluator, SLAShortfallCalculator
    from virne.reliability.metrics import summarize_reliability
    directory=Path(row['run_directory'])
    paths=sorted(p for p in (directory/'records').glob('*.csv') if not p.name.startswith('temp-'))
    if not paths:
        raise FileNotFoundError(f'No final records: {directory}')
    frame=pd.read_csv(paths[-1])
    metrics=Counter.summary_records(frame)
    cfg,pn,sim=generated(directory/'config.yaml')
    snapshot=input_snapshot(pn,sim)
    arrivals=frame.loc[frame.event_type==1].copy()
    for key in ('delivered_availability','sla_shortfall'):
        if key not in arrivals: arrivals[key]=np.nan
    expected={v.id:v for v in sim.v_nets}
    counter=Counter(cfg.v_sim_setting.node_attrs_setting,cfg.v_sim_setting.link_attrs_setting,cfg.v_sim_setting.get('graph_attrs_setting',{}),cfg)
    if len(arrivals)!=len(expected) or set(arrivals.v_net_id)!=set(expected):
        raise ValueError('Generated request IDs/count differ from records')
    for _,r in arrivals.iterrows():
        v=expected[r.v_net_id]
        for key,value in [('v_net_arrival_time',v.arrival_time),('v_net_lifetime',v.lifetime),
                          ('v_net_num_nodes',v.num_nodes),('v_net_num_egdes',v.num_links),
                          ('v_net_node_demand',counter.calculate_v_net_node_resource(v)),
                          ('v_net_link_demand',counter.calculate_sum_link_resource(v))]:
            if not np.isclose(r[key],value,rtol=1e-10,atol=1e-10):
                raise ValueError(f'Request stream mismatch {row["run_id"]}: {key}')
    # Derive a common assessment for the original baseline from recorded mappings.
    # For enabled methods also compare every metric against the frozen recorder.
    evaluator=ReliabilityEvaluator(assessment_mode or 'node_and_link')
    if assessment_beta is None:
        assessment_beta=cfg.joint_objective.beta
    if assessment_beta is None:
        assessment_beta=read_yaml(Path(bundle)/'calibration/recommended_config.yaml')['beta']
    for idx,r in arrivals.iterrows():
        req=expected[r.v_net_id].graph['availability_requirement']
        arrivals.loc[idx,'availability_requirement']=req
        success=str(r.result).lower()=='true'
        if success:
            solution={k:ast.literal_eval(r[k]) for k in ('node_slots','link_paths')}
            res=SLAShortfallCalculator.calculate(req,evaluator.availability(pn,solution),assessment_beta)
            for key,value in [('delivered_availability',res.delivered_availability),('sla_shortfall',res.sla_shortfall),('sla_satisfied',res.sla_satisfied)]:
                if key in frame and not np.isclose(float(r[key]),float(value),rtol=1e-7,atol=1e-10):
                    raise ValueError(f'Recorded {key} disagrees with shared evaluator')
                arrivals.loc[idx,key]=value
        else:
            arrivals.loc[idx,'sla_satisfied']=False
    reliability=summarize_reliability(arrivals)
    metrics.update(reliability)
    for key in ('p_net_node_resource_utilization','p_net_link_resource_utilization'):
        metrics['mean_'+key]=frame[key].mean()
        metrics['peak_'+key]=frame[key].max()
    accepted=arrivals.loc[arrivals.result.astype(str).str.lower()=='true']
    metrics['delivered_availability_distribution_json']=json.dumps(clean(accepted.delivered_availability.describe().to_dict())) if 'delivered_availability' in accepted else '{}'
    metrics['v_net_count']=len(arrivals)
    metrics.update(row)
    metrics.update(records=str(paths[-1]),runtime=float((directory/'runtime_seconds.txt').read_text()),
                   reliability_enabled=bool(cfg.reliability.enabled),reliability_mode=cfg.reliability.mode,
                   objective=cfg.joint_objective.name,reward=cfg.rl.reward_calculator.name,
                   beta=cfg.joint_objective.beta,topology=cfg.p_net_setting.topology.type,
                   assessment_mode=evaluator.mode,assessment_provenance='frozen evaluator applied to recorded mappings',
                   input_hash=digest(snapshot),source_commit=(directory/'source_commit.txt').read_text().strip(),
                   model_reload_observed='Loaded pretrained model from '+row['model_path'] in (directory/'stdout.log').read_text() if row['phase']=='eval' and row['category']=='learned' else None)
    fpdir=Path(bundle)/'input_fingerprints'
    fpdir.mkdir(exist_ok=True)
    write_json(fpdir/(row['run_id']+'.json'),snapshot)
    # Per-arrival derived data retain D/A distributions and training reward if present.
    arrivals.to_csv(directory/'assessment_arrivals.csv',index=False)
    return metrics
