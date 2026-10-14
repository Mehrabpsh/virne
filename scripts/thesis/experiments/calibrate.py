"""CPU heuristic calibration and transparent least-degenerate selection."""
import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
if __package__:
    from .common import collect_row, launch, read_yaml, write_json, write_yaml
else:
    from common import collect_row, launch, read_yaml, write_json, write_yaml


def score(point, rules):
    """Equal-weight distance to requested intervals, plus load/ordering penalties."""
    def distance(value, interval):
        lo,hi=interval
        return max(lo-value,0,value-hi)/(hi-lo)
    loss=distance(point['acceptance_rate'],rules['acceptance_range'])
    loss+=distance(point['sla_satisfaction_rate'],rules['satisfaction_range'])
    loss+=max(0,rules['minimum_peak_utilization']-point['peak_utilization'])
    loss+=point['class_order_penalty']
    return loss


def run(bundle, p):
    out=Path(bundle)/'calibration'
    out.mkdir()
    cp={**p,'requests':p['calibration']['requests'],'selected_device':'cpu'}
    rows=[]
    points=[]
    for index,(arrival,rate) in enumerate(itertools.product(p['calibration']['arrival_rates'],p['calibration']['failure_rates'])):
        point={'point_id':f'grid_{index:02d}','arrival_rate':arrival,'lambda_f':rate,'beta':p['calibration']['reference_beta']}
        print(f'Calibration {point}',flush=True)
        group=[]
        for seed,method in itertools.product(p['calibration']['seeds'],('nrm','grc')):
            row=launch(bundle,cp,point,method,seed,'calibration',root=out/'raw_runs'/point['point_id'])
            # Grid point prefixes disambiguate IDs in calibration tables and journals.
            metric=collect_row(row,bundle)
            metric.update(point)
            rows.append(metric)
            group.append(metric)
        frame=pd.DataFrame(group)
        class_totals={}
        for r in group:
            import json
            for level,v in json.loads(r['sla_class_metrics_json']).items():
                a,b=class_totals.get(level,(0,0))
                class_totals[level]=(a+v['accepted'], b+(v['sla_satisfaction_rate'] or 0)*v['accepted'])
        rates=[b/a for _,(a,b) in sorted(class_totals.items(),key=lambda x:float(x[0])) if a]
        order_penalty=sum(max(0,b-a) for a,b in zip(rates,rates[1:]))
        summary={**point,'acceptance_rate':frame.acceptance_rate.mean(),
                 'sla_satisfaction_rate':frame.sla_satisfaction_rate.mean(),
                 'peak_utilization':frame[['peak_p_net_node_resource_utilization','peak_p_net_link_resource_utilization']].max(axis=1).mean(),
                 'class_order_penalty':order_penalty,'class_coverage':len(rates),
                 'max_inservice_count':frame.max_inservice_count.mean()}
        summary['selection_loss']=score(summary,p['calibration'])
        summary['qualifies']=summary['selection_loss']==0 and len(rates)==4
        points.append(summary)
    # Stable point ID resolves ties; no PPO score enters calibration.
    selected=min(points,key=lambda x:(not x['qualifies'],x['selection_loss'],x['point_id']))
    positives=[]
    for r in rows:
        if r['point_id']==selected['point_id']:
            f=pd.read_csv(Path(r['run_directory'])/'assessment_arrivals.csv')
            positives.extend(f.loc[f.sla_shortfall>0,'sla_shortfall'].tolist())
    if not positives:
        raise ValueError('Cannot calibrate beta: selected point has no positive shortfalls; expand grid')
    median=float(np.median(positives))
    selected['beta']=-math.log(p['calibration']['beta_target_compliance'])/median
    selected['beta_calibration']={'positive_shortfall_median':median,'target_compliance':p['calibration']['beta_target_compliance'],'formula':'-log(target_compliance) / median_positive_D'}
    selected['provisional']=not selected['qualifies']
    pd.DataFrame(rows).to_csv(out/'calibration_summary.csv',index=False)
    write_json(out/'calibration_summary.json',{'runs':rows,'points':points,'selected':selected})
    write_yaml(out/'recommended_config.yaml',selected)
    text=('# Preliminary heuristic calibration\n\nNo PPO training or proposed-method outcomes enter selection.\n\n'
          'Selection: minimize equal-weight normalized distance of acceptance and satisfaction to configured intervals, '
          'plus max(0, minimum peak utilization - peak utilization), plus sum of increases in satisfaction between ascending SLA classes. '
          'Require all four accepted SLA classes. Prefer qualifying points, then lowest loss, then grid ID.\n\n'
          f'Rules: `{p["calibration"]}`\n\nSelected: `{selected}`\n\n'
          'Beta: -log(target compliance) / median positive accepted-service shortfall at selected point. '
          'This temporary scale calibration is not a final objective choice.\n\n'
          +('PROVISIONAL: no point meets all desired ranges/coverage; selected least-degenerate point.\n\n' if selected['provisional'] else 'Configured target ranges met; scientific calibration remains preliminary.\n\n')
          +pd.DataFrame(points).to_string(index=False)+'\n')
    (out/'calibration_report.md').write_text(text)
    (Path(bundle)/'reports'/'calibration_report.md').write_text(text)
    return selected

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('bundle',type=Path); args=a.parse_args()
    run(args.bundle,read_yaml(args.bundle/'active_profile.yaml'))
