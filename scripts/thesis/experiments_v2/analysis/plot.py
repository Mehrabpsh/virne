"""Configured metric figures with explicitly distinct training/scenario error bars."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.thesis.experiments.plot_results import training_plots
from .aggregate import tables


def run(bundle,resolved):
    bundle=Path(bundle);out=bundle/'plots';out.mkdir(exist_ok=True)
    exp=resolved['experiment'];formats=exp['reporting']['formats']
    frame=pd.read_csv(bundle/'aggregated/results_long.csv')
    summary=pd.read_csv(bundle/'aggregated/results_summary.csv')
    plt.rcParams.update({'font.size':10,'savefig.dpi':180,'pdf.fonttype':42})
    def save(fig,name):
        fig.tight_layout()
        for extension in formats:fig.savefig(out/f'{name}.{extension}',bbox_inches='tight')
        plt.close(fig)
    def bars(stats,metric,name):
        fig,ax=plt.subplots(figsize=(11,6));x=np.arange(len(stats))
        mean=stats[metric+'_mean'];valid=mean.notna()
        ax.bar(x[valid],mean[valid],color='#32658b',alpha=.8)
        # Keep two separate SDs rather than inventing independent pooled replicates.
        for column,offset,color,label in [('train_std',-.09,'#8c2d04','SD across trained-model means'),
                                           ('eval_std',.09,'#202020','SD across scenario means')]:
            error=stats[metric+'_'+column]
            mask=valid & error.notna()
            if mask.any():ax.errorbar(x[mask]+offset,mean[mask],yerr=error[mask],fmt='none',ecolor=color,capsize=3,label=label)
        ax.set_xticks(x,stats.label,rotation=40,ha='right');ax.set_ylabel(metric.replace('_',' '))
        ax.set_title('Preliminary pipeline validation');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
        save(fig,name)
    for metric in exp['metrics']:bars(summary,metric,metric)
    rows=[]
    for _,r in frame.iterrows():
        for level,values in json.loads(r.sla_class_metrics_json).items():
            rows.append({'method':r.method,'label':r.label,'category':r.category,'train_seed':r.train_seed,
                         'eval_seed':r.eval_seed,'level':float(level),**values})
    classes=pd.DataFrame(rows);classes.to_csv(out/'sla_class_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,6))
    if len(classes):
        for method,group in classes.groupby('method',sort=False):
            rates=group.groupby(['level','eval_seed']).sla_satisfaction_rate.mean().groupby('level').agg(['mean','std'])
            ax.errorbar([str(v) for v in rates.index],rates['mean'],yerr=rates['std'].fillna(0),marker='o',capsize=2,label=group.iloc[0].label)
    ax.set(xlabel='SLA requirement',ylabel='Accepted-service satisfaction',title='Preliminary SLA classes; scenario SD after averaging training seeds')
    ax.legend(fontsize=8,bbox_to_anchor=(1.01,1));ax.grid(alpha=.2);save(fig,'sla_class_breakdown')
    methods={name:(m['solver'],m['label'],m['capabilities']['kind']) for name,m in resolved['selected_methods'].items()}
    training_plots(bundle,methods,save)
