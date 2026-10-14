"""Shared structural checks used by v1 and v2; no solver behavior changes."""
import numpy as np
import pandas as pd
from pathlib import Path


def evaluation_errors(frame, requests):
    errors=[]
    def check(condition,message):
        if not condition:errors.append(message)
    for _,r in frame.iterrows():
        check(Path(r.records).is_file(),f'{r.run_id}: records missing')
        raw=pd.read_csv(r.records);arr=raw.loc[raw.event_type==1]
        successes=arr.result.astype(str).str.lower().eq('true').sum()
        check(len(raw)==2*requests and raw.event_id.is_unique,f'{r.run_id}: missing/duplicate arrival or departure events')
        check(len(arr)==requests,f'{r.run_id}: request count mismatch')
        check(successes==r.success_count==r.sla_accepted_count,f'{r.run_id}: accepted counts inconsistent')
        check(np.isclose(r.acceptance_rate,successes/len(arr)),f'{r.run_id}: acceptance denominator')
        check(np.isclose(r.sla_service_rate_all_arrivals,r.acceptance_rate*r.sla_satisfaction_rate),f'{r.run_id}: SLA denominator')
        check((raw.inservice_count>=0).all(),f'{r.run_id}: negative in-service count')
        # OnlineSystem processes through final departures; verify resources released.
        check(raw.iloc[-1].inservice_count==0,f'{r.run_id}: final departures did not release all services')
        for key in ('p_net_node_resource_utilization','p_net_link_resource_utilization'):
            check(np.isclose(raw.iloc[-1][key],0,atol=1e-9),f'{r.run_id}: resources not fully released ({key})')
        if r.category=='learned':check(str(r.model_reload_observed).lower()=='true',f'{r.run_id}: no successful checkpoint reload evidence')
    return errors


def metric_errors(frame, metric_names):
    errors=[]
    def check(condition,message):
        if not condition:errors.append(message)
    for key in metric_names:
        values=pd.to_numeric(frame[key],errors='coerce')
        nullable=key in ('mean_delivered_availability','mean_sla_shortfall','median_sla_shortfall')
        allowed=frame.sla_accepted_count.eq(0) if nullable else pd.Series(False,index=frame.index)
        check((np.isfinite(values)|allowed).all(),f'Nonfinite required metric: {key}')
        if 'rate' in key or key=='mean_delivered_availability':
            check((values.dropna().between(0,1)).all(),f'Probability outside [0,1]: {key}')
        if 'shortfall' in key:check((values.dropna()>=0).all(),f'Negative shortfall: {key}')
    return errors
