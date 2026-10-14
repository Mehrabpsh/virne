#!/usr/bin/env python3
"""Explicit, CPU-only environment calibration. Never trains learned methods."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import argparse
import copy
import itertools
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
import yaml
from scripts.thesis.experiments.common import clean, collect_row, digest, write_json, write_yaml
from scripts.thesis.experiments.calibrate import score
from scripts.thesis.experiments.shared_checks import evaluation_errors, metric_errors
from scripts.thesis.experiments.common import METRICS
from scripts.thesis.experiments_v2.config.resolver import resolve
from scripts.thesis.experiments_v2.config.schema import dealias, ConfigError, get_path, keys, positive, require, set_path
from scripts.thesis.experiments_v2.execution.launcher import launch
from scripts.thesis.experiments_v2.execution.planner import hardware, job
from scripts.thesis.experiments_v2.execution.runtime import freeze, install_signals, new_bundle, stage


def plan_calibration(path,sets):
    original=path.read_text();data=dealias(yaml.safe_load(original))
    from scripts.thesis.experiments_v2.config.schema import parse_assignments
    for key,value in parse_assignments(sets):set_path(data,key,value)
    calibration=data.get('calibration',{})
    require(bool(calibration.get('search_space')),'Calibration requires an explicit nonempty search_space')
    positive(calibration.get('requests'),'calibration.requests',True)
    names=calibration.get('methods')
    require(isinstance(names,list) and bool(names) and len(set(names))==len(names),'Calibration methods must be a unique nonempty list')
    resolved,_=resolve(path,sets,subset=names)
    require(all(m['capabilities']['kind']=='heuristic' for m in resolved['selected_methods'].values()),'Calibration cannot train or evaluate learned methods')
    space=calibration['search_space']
    for key,values in space.items():
        require(key.startswith('environment.'),f'Calibration searches only environment paths: {key}')
        get_path(data,key)
        require(isinstance(values,list) and bool(values),f'Empty calibration candidates for {key}')
    criteria=calibration['criteria']
    keys(criteria,['acceptance_rate','sla_satisfaction_rate','minimum_peak_utilization','require_all_sla_classes','require_monotonic_sla_difficulty','monotonic_tolerance'],'calibration.criteria')
    for name in ('acceptance_rate','sla_satisfaction_rate'):
        require(0<=criteria[name]['min']<criteria[name]['max']<=1,f'Invalid calibration interval {name}')
    require(0<=criteria['minimum_peak_utilization']<=1,'Invalid utilization threshold')
    require(criteria.get('monotonic_tolerance',0)>=0,'monotonic_tolerance must be nonnegative')
    suggestions=calibration.get('beta_suggestion',{})
    keys(suggestions,['enabled','target_compliance'],'beta_suggestion')
    if suggestions.get('enabled'):require(0<suggestions['target_compliance']<1,'beta suggestion target must be in (0,1)')
    candidates=[]
    for index,values in enumerate(itertools.product(*space.values())):
        selected=dict(zip(space,values));candidate=copy.deepcopy(data)
        for key,value in selected.items():set_path(candidate,key,value)
        candidate['environment']['workload']['num_requests']=calibration['requests']
        r,_=resolve(path,subset=names,document=candidate)
        candidates.append({'point_id':f'grid_{index:03d}','selected':selected,'resolved':r})
    return resolved,original,data,candidates


def select(points,criteria):
    rules={'acceptance_range':[criteria['acceptance_rate']['min'],criteria['acceptance_rate']['max']],
           'satisfaction_range':[criteria['sla_satisfaction_rate']['min'],criteria['sla_satisfaction_rate']['max']],
           'minimum_peak_utilization':criteria['minimum_peak_utilization']}
    for point in points:
        point['selection_loss']=score(point,rules)
        point['qualifies']=point['selection_loss']==0 and (not criteria['require_all_sla_classes'] or point['class_coverage']==point['expected_class_count'])
    return min(points,key=lambda p:(not p['qualifies'],p['selection_loss'],p['point_id']))


def execute(path,sets,output,results=None):
    resolved,original,data,candidates=plan_calibration(path,sets)
    require(not output.exists(),'Refusing to overwrite calibrated YAML: '+str(output))
    bundle=new_bundle(resolved,results,'_calibration');install_signals();start=time.monotonic();rows=[];manifest=[]
    calibration=data['calibration'];warnings=[]
    plan={'tool':'calibration','hardware':hardware('cpu'),'methods':calibration['methods'],'seeds':calibration['seeds'],
          'search_space':calibration['search_space'],'job_count':len(candidates)*len(calibration['methods'])*len(calibration['seeds']),
          'learned_methods_used':False}
    freeze(bundle,resolved,original,plan)
    try:
        points=[]
        for candidate in candidates:
            point_rows=[]
            r=candidate['resolved'];point=candidate['point_id']
            print('Calibration point '+point+': '+str(candidate['selected']),flush=True)
            for seed,method in itertools.product(calibration['seeds'],calibration['methods']):
                j=job(method,r['selected_methods'][method]['solver'],'calibration',seed,None)
                old=j['run_id'];j['run_id']=point+'__'+old
                j['relative_directory']=j['relative_directory'].replace(old,j['run_id'])
                j['relative_output_root']=str(Path('raw_runs')/point/j['relative_output_root'])
                j['relative_directory']=str(Path('raw_runs')/point/j['relative_directory'])
                row=launch(bundle,r,j,'cpu');manifest.append(row)
                write_json(bundle/'manifest.json',manifest);pd.DataFrame(manifest).to_csv(bundle/'manifest.csv',index=False)
                metric=collect_row(row,bundle,assessment_beta=data['reporting']['assessment_beta'],assessment_mode=r['resolved_environment']['reliability']['mode'])
                metric['point_id']=point;point_rows.append(metric);rows.append(metric)
            frame=pd.DataFrame(point_rows)
            errors=evaluation_errors(frame,calibration['requests'])+metric_errors(frame,METRICS)
            for seed,g in frame.groupby('seed'):
                if g.input_hash.nunique()!=1:errors.append(f'{point}: calibration inputs differ between methods for seed {seed}')
            require(not errors,'Calibration structural errors: '+str(errors))
            class_counts={}
            for row in point_rows:
                for level,values in json.loads(row['sla_class_metrics_json']).items():
                    total,satisfied=class_counts.get(level,(0,0))
                    class_counts[level]=(total+values['accepted'],satisfied+values['accepted']*(values['sla_satisfaction_rate'] or 0))
            rates=[s/n for _,(n,s) in sorted(class_counts.items(),key=lambda x:float(x[0])) if n]
            tolerance=calibration['criteria'].get('monotonic_tolerance',0)
            penalty=sum(max(0,b-a-tolerance) for a,b in zip(rates,rates[1:])) if calibration['criteria']['require_monotonic_sla_difficulty'] else 0
            points.append({'point_id':point,'selected':candidate['selected'],'acceptance_rate':frame.acceptance_rate.mean(),
                'sla_satisfaction_rate':frame.sla_satisfaction_rate.mean(),
                'peak_utilization':frame[['peak_p_net_node_resource_utilization','peak_p_net_link_resource_utilization']].max(axis=1).mean(),
                'max_inservice_count':frame.max_inservice_count.mean(),'class_order_penalty':penalty,'class_coverage':len(rates),
                'expected_class_count':len(data['environment']['sla']['levels'])})
        chosen=select(points,calibration['criteria'])
        if not chosen['qualifies']:warnings.append('PROVISIONAL: no point met all target ranges/accepted-class coverage; least-degenerate point selected by declared loss and grid ID.')
        suggestion=None
        if calibration.get('beta_suggestion',{}).get('enabled'):
            positive_d=[]
            for row in rows:
                if row['point_id']==chosen['point_id']:
                    raw=pd.read_csv(Path(row['run_directory'])/'assessment_arrivals.csv')
                    positive_d.extend(raw.loc[raw.sla_shortfall>0,'sla_shortfall'].tolist())
            if positive_d:
                median=float(np.median(positive_d))
                suggestion={'value':-math.log(calibration['beta_suggestion']['target_compliance'])/median,
                            'positive_D_median':median,'target_compliance':calibration['beta_suggestion']['target_compliance'],
                            'formula':'-log(target_compliance) / median_positive_D','scientifically_frozen':False}
            else:warnings.append('No positive accepted-service D; beta-scale suggestion unavailable.')
        calibrated=copy.deepcopy(data)
        for key,value in chosen['selected'].items():set_path(calibrated,key,value)
        calibrated['calibration_suggestions']={'multiplicative_beta_scale':suggestion}
        adopted=[]
        for method in calibration.get('adopt_beta_for',[]):
            require(suggestion is not None,'Explicit beta adoption requested but no suggestion is available')
            target=f'methods.{method}.objective.beta';previous=get_path(calibrated,target)
            set_path(calibrated,target,suggestion['value']);adopted.append({'method':method,'previous':previous,'value':suggestion['value']})
        calibrated['calibration_result']={'source_config':str(path.resolve()),'source_config_sha256':hashlib.sha256(original.encode()).hexdigest(),'resolved_input_sha256':digest(data),
            'source_commit':(bundle/'source_commit.txt').read_text().strip(),'searched_paths':list(calibration['search_space']),
            'candidate_values':calibration['search_space'],'methods':calibration['methods'],'seeds':calibration['seeds'],
            'calibration_request_budget':calibration['requests'],'selected':chosen['selected'],'selection_metrics':chosen,
            'selection_rule':'Prefer qualifying points, then minimum normalized interval-distance + load-shortfall + SLA-order penalty, then grid ID; qualification additionally requires accepted SLA-class coverage if configured.',
            'criteria':calibration['criteria'],'warnings':warnings,'explicit_beta_adoptions':adopted,'report_directory':str(bundle)}
        calibrated=clean(calibrated)
        for warning in warnings:print('WARNING: '+warning,file=sys.stderr,flush=True)
        pd.DataFrame(rows).to_csv(bundle/'calibration_summary.csv',index=False)
        write_json(bundle/'calibration_summary.json',{'runs':rows,'points':points,'selected':chosen,'warnings':warnings,'suggestions':suggestion})
        write_yaml(bundle/'calibrated_config.yaml',calibrated)
        text=('# Explicit environment calibration — preliminary\n\nNo learned methods were trained or evaluated.\n\n'
              'Loss and selection rule:\n\n```yaml\n'+yaml.safe_dump(calibrated['calibration_result'],sort_keys=False)+'```\n\n'
              'Beta is a method parameter. Suggestion/adoption provenance:\n\n```yaml\n'+yaml.safe_dump({'suggestion':suggestion,'explicit_adoptions':adopted},sort_keys=False)+'```\n\n'
              'Calibration request count is a measurement budget; only explicitly searched environment paths are changed in the output experiment. '
              'No final scientific parameters are frozen.\n')
        (bundle/'reports/calibration_report.md').write_text(text)
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as f:yaml.safe_dump(calibrated,f,sort_keys=False)
        (bundle/'COMPLETE.txt').write_text(f'Calibration structurally complete. Jobs: {len(manifest)}. Output: {output.resolve()}. Runtime: {time.monotonic()-start:.3f}s.\n')
        print(f'COMPLETE: {bundle}\nCalibrated config: {output.resolve()}',flush=True)
        return bundle
    except BaseException as exc:
        write_json(bundle/'FAILED.json',{'error':str(exc),'runtime_seconds':time.monotonic()-start});raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--results',type=Path);p.add_argument('--set',action='append',default=[])
    p.add_argument('--dry-run',action='store_true');args=p.parse_args(argv)
    try:
        resolved,original,data,candidates=plan_calibration(args.config,args.set)
        if args.dry_run:
            plan={'tool':'calibration','hardware':hardware('cpu'),'methods':data['calibration']['methods'],'seeds':data['calibration']['seeds'],
                  'points':[{'point_id':c['point_id'],'selected':c['selected']} for c in candidates],
                  'job_count':len(candidates)*len(data['calibration']['methods'])*len(data['calibration']['seeds']),'learned_methods_used':False}
            print(yaml.safe_dump(plan,sort_keys=False));return plan
        return execute(args.config,args.set,args.output or args.config.with_name(args.config.stem+'.calibrated.yaml'),args.results)
    except (ConfigError,KeyError,TypeError) as exc:p.error('Invalid calibration configuration: '+str(exc))

if __name__=='__main__':main()
