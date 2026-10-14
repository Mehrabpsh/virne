"""Hard structural checks separated from preliminary scientific warnings."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import METHODS, METRICS, read_yaml, write_json
from scripts.thesis.experiments.shared_checks import evaluation_errors, metric_errors


def _run(bundle):
    bundle=Path(bundle)
    manifest=json.loads((bundle/'pipeline_manifest.json').read_text())
    p=read_yaml(bundle/'active_profile.yaml')
    frame=pd.read_csv(bundle/'aggregated/results_long.csv')
    errors=[];warnings=[]
    def check(condition,message):
        if not condition: errors.append(message)
    check(len(frame)==len(p['test_seeds'])*(8+3*len(p['train_seeds'])),'Wrong number of evaluation jobs')
    check(len(manifest)==len(frame)+3*len(p['train_seeds']),'Wrong number of training jobs')
    check(len({r['run_id'] for r in manifest})==len(manifest),'Duplicate run identifiers')
    expected=set()
    for m,(_,_,category) in METHODS.items():
        for ts in (p['train_seeds'] if category=='learned' else [None]):
            for seed in p['test_seeds']: expected.add((m,ts,seed))
    actual={(r.method,None if pd.isna(r.train_seed) else int(r.train_seed),int(r.test_seed)) for _,r in frame.iterrows()}
    check(actual==expected,'Missing/unexpected method x train seed x test seed combinations')
    check(all(r['completed'] for r in manifest),'Manifest contains incomplete jobs')
    commit=(bundle/'source_commit.txt').read_text().strip()
    for r in manifest:
        d=Path(r['run_directory'])
        for f in ('config.yaml','hydra_config.yaml','source_commit.txt','stdout.log','runtime_seconds.txt'):
            check((d/f).is_file(),f'{r["run_id"]}: missing {f}')
        check((d/'source_commit.txt').read_text().strip()==commit,f'{r["run_id"]}: source commit mismatch')
        from omegaconf import OmegaConf
        config=OmegaConf.load(d/'config.yaml')
        check(config.training.num_train_epochs==(p['epochs'] if r['phase']=='train' else 0),f'{r["run_id"]}: epoch budget mismatch')
        if r['phase']=='eval':check(not config.system.if_offline_system,f'{r["run_id"]}: evaluation must use OnlineSystem')
        if r['category']=='learned': check(Path(r['model_path']).is_file(),f'{r["run_id"]}: model missing')
    errors.extend(evaluation_errors(frame,p['requests']))
    errors.extend(metric_errors(frame,METRICS))
    for seed,g in frame.groupby('test_seed'):
        check(g.input_hash.nunique()==1,f'Input topology/resources/request/SLA mismatch for test seed {seed}')
    for key in ('sla_shortfall','delivered_availability','availability_requirement'):
        for r in manifest:
            if r['phase']!='eval':continue
            a=pd.read_csv(Path(r['run_directory'])/'assessment_arrivals.csv')
            accepted=a.loc[a.result.astype(str).str.lower().eq('true')]
            vals=a[key] if key=='availability_requirement' else accepted[key]
            check(np.isfinite(vals).all(),f'{r["run_id"]}: invalid raw {key}')
            if key!='sla_shortfall':check(vals.between(0,1).all(),f'{r["run_id"]}: raw probability outside range')
    summary=pd.read_csv(bundle/'aggregated/results_summary.csv')
    for key in METRICS:
        check((np.isfinite(summary[key+'_mean']) | summary[key+'_count'].eq(0)).all(),f'Summary invalid: {key}')
    point=read_yaml(bundle/'calibration/recommended_config.yaml')
    if point['provisional']:warnings.append('Calibration provisional: no grid point met every desired interval and SLA class coverage.')
    if (frame.acceptance_rate.isin([0,1])).any():warnings.append('Some acceptance rates degenerate in small streams; increase request budgets for scientific work.')
    warnings.extend(['All figures/results are preliminary pipeline validation.','Objective, reward, beta scaling, comparator settings and budgets remain scientifically unfrozen.',
                     'Homogeneous scalar CTMC rates and zero Weibull history are explicit development assumptions.',
                     'Baseline SLA metrics are derived from recorded mappings; reliability stayed disabled in the solver.',
                     'Seed std combines training and test seeds; it is descriptive and not a confidence interval.'])
    result={'passed':not errors,'hard_errors':errors,'warnings':warnings,'evaluation_runs':len(frame),'training_runs':len(manifest)-len(frame),'shared_inputs_verified':not any('mismatch for test seed' in e for e in errors)}
    write_json(bundle/'validation_report.json',result)
    text='# Pipeline validation\n\n'+json.dumps(result,indent=2)+'\n'
    (bundle/'validation_report.md').write_text(text);(bundle/'reports/validation_report.md').write_text(text)
    if errors:raise ValueError('Structural validation failed: '+'; '.join(errors))
    return result

def run(bundle):
    bundle=Path(bundle)
    (bundle/'reports').mkdir(exist_ok=True)
    for file in ('validation_report.json','validation_report.md','reports/validation_report.md'):
        (bundle/file).unlink(missing_ok=True)
    try:
        return _run(bundle)
    except Exception as exc:
        bundle=Path(bundle)
        if not (bundle/'validation_report.json').exists():
            result={'passed':False,'hard_errors':[str(exc)],'warnings':[]}
            write_json(bundle/'validation_report.json',result)
            text='# Pipeline validation FAILED\n\n'+str(exc)+'\n'
            (bundle/'validation_report.md').write_text(text)
            (bundle/'reports').mkdir(exist_ok=True)
            (bundle/'reports/validation_report.md').write_text(text)
        raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('bundle',type=Path);args=a.parse_args();run(args.bundle)
