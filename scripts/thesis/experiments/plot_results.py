"""Preliminary matplotlib figures, PNG and PDF, without seaborn."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
if __package__:
    from .common import METRICS, METHODS
else:
    from common import METRICS, METHODS


def run(bundle):
    bundle=Path(bundle);out=bundle/'plots';out.mkdir(exist_ok=True)
    frame=pd.read_csv(bundle/'aggregated/results_long.csv')
    plt.rcParams.update({'font.size':10,'savefig.dpi':180,'pdf.fonttype':42})
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig)
    for key in METRICS:
        group=frame.groupby('method',sort=False)[key].agg(['mean','std','count'])
        fig,ax=plt.subplots(figsize=(10,6))
        x=np.arange(len(group));valid=group['mean'].notna()
        ax.bar(x[valid],group['mean'][valid],yerr=group['std'][valid].fillna(0),capsize=3,color='#32658b')
        ax.set_xticks(x,[METHODS[m][1] for m in group.index],rotation=40,ha='right')
        ax.set_ylabel(key.replace('_',' ').capitalize());ax.set_title('Preliminary pipeline validation')
        ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
        save(fig,key)
    rows=[]
    for _,r in frame.iterrows():
        for level,metrics in json.loads(r.sla_class_metrics_json).items():
            rows.append({'method':r.method,'level':float(level),**metrics})
    classes=pd.DataFrame(rows)
    classes.to_csv(out/'sla_class_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,6))
    for m,g in classes.groupby('method',sort=False):
        stats=g.groupby('level').sla_satisfaction_rate.agg(['mean','std'])
        ax.errorbar([str(v) for v in stats.index],stats['mean'],yerr=stats['std'].fillna(0),marker='o',capsize=2,label=METHODS[m][1])
    ax.set(xlabel='Requested SLA availability',ylabel='Satisfaction among accepted requests',title='Preliminary pipeline validation: SLA classes')
    ax.legend(fontsize=8,bbox_to_anchor=(1.01,1));ax.grid(alpha=.25);save(fig,'sla_class_breakdown')
    training_plots(bundle,METHODS,save)


def training_plots(bundle,methods,save):
    """Shared training telemetry renderer; save callback controls output formats."""
    bundle=Path(bundle)
    out=bundle/'plots'
    manifest=json.loads((bundle/'pipeline_manifest.json').read_text())
    training=[]
    for r in manifest:
        if r['phase']!='train':continue
        directory=Path(r['run_directory'])
        # The frozen file logger writes train_epochs and loss telemetry to running.log.
        log=(directory/'logs/running.log')
        entries=[]
        if log.exists():
            import re
            pattern=r'epoch\s+(\d+), success_count\s+(\d+), r2c ([\d.]+)'
            entries=[(int(a),int(b),float(c)) for a,b,c in re.findall(pattern,log.read_text())]
        if entries:
            fig,ax=plt.subplots(figsize=(7,4))
            ax.plot([v[0] for v in entries],[v[2] for v in entries],marker='o')
            ax.set(xlabel='Training epoch',ylabel='Recorded training R2C',title='Preliminary training telemetry: '+methods[r['method']][1])
            save(fig,'training_'+r['run_id'])
        update_points=0
        telemetry=directory/'logs/training_info.csv'
        if telemetry.exists():
            data=pd.read_csv(telemetry)
            for column,name in [('value/reward','reward'),('loss/loss','loss')]:
                if column not in data: continue
                values=pd.to_numeric(data[column],errors='coerce').dropna()
                if values.empty: continue
                update_points=max(update_points,len(values))
                fig,ax=plt.subplots(figsize=(7,4))
                ax.plot(values.index,values,marker='o',markersize=3)
                ax.set(xlabel='Logged PPO update',ylabel=column,
                       title='Preliminary training '+name+': '+methods[r['method']][1])
                save(fig,'training_'+name+'_'+r['run_id'])
        training.append({'run_id':r['run_id'],'available_epoch_points':len(entries),
                         'available_update_points':update_points,
                         'convergence_claim':False})
    (out/'training_telemetry.json').write_text(json.dumps(training,indent=2)+'\n')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('bundle',type=Path);args=a.parse_args();run(args.bundle)
