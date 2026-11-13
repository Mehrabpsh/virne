"""Crossed-seed summaries, with distinct model and scenario variability."""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.thesis.experiments.common import write_json


def mode_groups(frame):
    """Legacy single-mode files keep their schema; new modes never pool."""
    return frame.groupby('evaluation_mode',sort=False) if 'evaluation_mode' in frame else [(None,frame)]


def mode_metrics(metrics,mode):
    from virne.reliability.live_failures import LIVE_METRICS
    return [m for m in metrics if mode!='failure_free' or m not in LIVE_METRICS]


def tables(frame,metrics):
    outputs=[[],[],[]]
    for mode,group in mode_groups(frame):
        for target,table in zip(outputs,_tables(group,metrics)):
            if mode is not None:table.insert(0,'evaluation_mode',mode)
            target.append(table)
    return tuple(pd.concat(values,ignore_index=True) for values in outputs)


def _tables(frame,metrics):
    metadata={'train_seed','eval_seed','test_seed','seed','run_id','method','label','category','phase','solver',
              'relative_directory','relative_output_root','run_directory','model_path','records','completed',
              'reliability_enabled','reliability_mode','objective','reward','beta','topology','assessment_mode',
              'assessment_provenance','input_hash','source_commit','model_reload_observed','evaluation_mode',
              'live_failures_enabled','failure_trace_sha256','runtime_inputs_verified'}
    numeric=[]
    frame=frame.copy()
    for column in frame.columns:
        if column in metadata:continue
        try:values=pd.to_numeric(frame[column],errors='raise')
        except (ValueError,TypeError):continue
        if pd.api.types.is_bool_dtype(values):continue
        frame[column]=values;numeric.append(column)
    required=[m for m in metrics if m not in numeric]
    if required:raise ValueError(f'Configured metrics absent/non-numeric: {required}')
    by_train=[];by_eval=[];summary=[]
    for method,group in frame.groupby('method',sort=False):
        learned=group.iloc[0]['category']=='learned'
        per_train=group.groupby('train_seed')[numeric].mean() if learned else pd.DataFrame()
        per_eval=group.groupby('eval_seed')[numeric].mean()
        for seed,row in per_train.iterrows():
            by_train.append({'method':method,'label':group.iloc[0]['label'],'train_seed':int(seed),
                             'evaluation_scenarios':int(group.loc[group.train_seed==seed,'eval_seed'].nunique()),**row.to_dict()})
        for seed,row in per_eval.iterrows():
            by_eval.append({'method':method,'label':group.iloc[0]['label'],'eval_seed':int(seed),**row.to_dict()})
        result={'method':method,'label':group.iloc[0]['label'],'kind':group.iloc[0]['category'],
                'run_count':len(group),'training_seed_count':len(per_train),'evaluation_seed_count':len(per_eval)}
        for metric in numeric:
            values=per_train[metric] if learned else per_eval[metric]
            result.update({metric+'_mean':values.mean(),metric+'_train_std':per_train[metric].std(ddof=1) if learned else np.nan,
                           metric+'_eval_std':per_eval[metric].std(ddof=1),metric+'_count':int(group[metric].count()),
                           metric+'_train_mean_count':int(values.count()) if learned else 0,
                           metric+'_eval_mean_count':int(per_eval[metric].count())})
        summary.append(result)
    return pd.DataFrame(by_train,columns=['method','label','train_seed','evaluation_scenarios',*numeric]),pd.DataFrame(by_eval),pd.DataFrame(summary)


def paired_differences(by_eval,metrics):
    outputs=[]
    for mode,group in mode_groups(by_eval):
        result=_paired_differences(group,metrics)
        if mode is not None:result.insert(0,'evaluation_mode',mode)
        outputs.append(result)
    return pd.concat(outputs,ignore_index=True)


def _paired_differences(by_eval,metrics):
    rows=[]
    for left,right in itertools.combinations(by_eval.method.unique(),2):
        a=by_eval.loc[by_eval.method==left].set_index('eval_seed')
        b=by_eval.loc[by_eval.method==right].set_index('eval_seed')
        for seed in a.index.intersection(b.index):
            rows.append({'method_a':left,'method_b':right,'eval_seed':int(seed),**{m:a.loc[seed,m]-b.loc[seed,m] for m in metrics}})
    return pd.DataFrame(rows,columns=['method_a','method_b','eval_seed',*metrics])


def paired_modes(frame,metrics):
    """Within the same model/scenario: live minus failure-free, without pooling."""
    keys=['method','train_seed','eval_seed']
    columns=keys+['difference_direction',*metrics]
    if 'evaluation_mode' not in frame:return pd.DataFrame(columns=columns)
    free=frame.loc[frame.evaluation_mode.eq('failure_free')]
    live=frame.loc[frame.evaluation_mode.eq('live_failures')]
    if live.empty:return pd.DataFrame(columns=columns)
    # Pandas matches null train_seed values for heuristic pairs as intended.
    paired=live.merge(free,on=keys,suffixes=('_live','_free'),validate='one_to_one')
    if len(paired)!=len(live):raise ValueError('Live evaluation lacks a failure-free pair')
    result=paired[keys].copy();result['difference_direction']='live_failures_minus_failure_free'
    for metric in metrics:
        result[metric]=paired[metric+'_live']-paired[metric+'_free']
    return result[columns]


def run(bundle,resolved,rows):
    out=Path(bundle)/'aggregated';out.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'results_long.csv',index=False)
    by_train,by_eval,summary=tables(frame,resolved['experiment']['metrics'])
    by_train.to_csv(out/'results_by_train_seed.csv',index=False)
    by_eval.to_csv(out/'results_by_eval_seed.csv',index=False)
    summary.to_csv(out/'results_summary.csv',index=False)
    write_json(out/'results_summary.json',summary.to_dict('records'))
    paired_differences(by_eval,resolved['experiment']['metrics']).to_csv(out/'paired_differences_by_eval_seed.csv',index=False)
    paired_modes(frame,resolved['experiment']['metrics']).to_csv(out/'paired_evaluation_modes.csv',index=False)
    return frame
