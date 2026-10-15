"""Matrix/artifact/aggregation checks plus shared v1 record/accounting validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scripts.thesis.experiments.common import METRICS, write_json
from scripts.thesis.experiments.shared_checks import evaluation_errors, metric_errors
from ..execution.planner import final_config, jobs, system_mode, tensorboard_path
from .aggregate import tables, paired_differences


def same_frame(actual,expected,name):
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True),check_dtype=False,rtol=1e-8,atol=1e-10)
    except AssertionError as exc:raise ValueError(f'{name}: stored aggregation differs from raw observations: {exc}') from exc


def fairness_errors(frame):
    return [f'Exogenous input fingerprint mismatch for evaluation seed {seed}' for seed,g in frame.groupby('eval_seed') if g.input_hash.nunique()!=1]


def mode_errors(resolved,job,config,audit):
    expected=system_mode(resolved,job['phase'])
    errors=[]
    if bool(config.system.if_offline_system)!=(expected=='OfflineSystem') or config.system.if_changeable_v_nets or config.system.if_time_window:
        errors.append(f'{job["run_id"]}: requested {expected} differs from final config')
    if audit.get('system')!=expected:
        errors.append(f'{job["run_id"]}: requested {expected}, actual runtime {audit.get("system")}')
    if job['phase']=='train' and not audit.get('training_completed'):
        errors.append(f'{job["run_id"]}: runtime did not complete solver.learn')
    if job['phase']=='eval' and (audit.get('training_requested') or audit.get('optimizer_updates',0)):
        errors.append(f'{job["run_id"]}: runtime trained during evaluation')
    return errors


def run(bundle,resolved,manifest):
    bundle=Path(bundle);errors=[]
    def check(condition,message):
        if not condition:errors.append(message)
    try:
        expected=jobs(resolved);expected_ids={j['run_id'] for j in expected}
        check(len(manifest)==len(expected),'Incomplete planned job matrix')
        check({r['run_id'] for r in manifest}==expected_ids,'Missing/unexpected job IDs')
        check(len({r['run_id'] for r in manifest})==len(manifest),'Duplicate job IDs')
        check(all(r.get('completed') for r in manifest),'Incomplete manifest entries')
        frame=pd.read_csv(bundle/'aggregated/results_long.csv')
        eval_jobs={j['run_id'] for j in expected if j['phase']=='eval'}
        check(set(frame.run_id)==eval_jobs and len(frame)==len(eval_jobs),'Aggregation does not contain exactly the evaluation matrix')
        commit=(bundle/'source_commit.txt').read_text().strip()
        for r in manifest:
            directory=Path(r['run_directory'])
            check(directory.resolve()==(bundle/r['relative_directory']).resolve(),f'{r["run_id"]}: wrong result layout')
            for file in ('config.yaml','hydra_config.yaml','command.sh','stdout.log','runtime_seconds.txt','source_commit.txt'):
                check((directory/file).is_file(),f'{r["run_id"]}: missing {file}')
            config=OmegaConf.load(directory/'config.yaml')
            audit_path=directory/'runtime_system.json'
            check(audit_path.is_file(),f'{r["run_id"]}: missing runtime mode evidence')
            if audit_path.is_file():errors.extend(mode_errors(resolved,r,config,json.loads(audit_path.read_text())))
            if resolved['experiment']['runtime'].get('tensorboard',{}).get('enabled'):
                tb=tensorboard_path(bundle,r)
                check(config.logger.tensorboard_dir==str(tb),f'{r["run_id"]}: wrong TensorBoard route')
                check(bool(list(tb.glob('events.out.tfevents.*'))),f'{r["run_id"]}: missing TensorBoard event files')
            check((directory/'source_commit.txt').read_text().strip()==commit,f'{r["run_id"]}: source mismatch')
            check(config.experiment.seed==r['seed'],f'{r["run_id"]}: seed mismatch')
            check(config.training.num_train_epochs==(resolved['experiment']['protocol']['training']['epochs'] if r['phase']=='train' else 0),f'{r["run_id"]}: epoch mismatch')
            if r['category']=='learned':check(Path(r['model_path']).is_file(),f'{r["run_id"]}: checkpoint missing')
            if r['phase']=='eval':
                check(not config.system.if_offline_system and not config.system.if_changeable_v_nets and not config.system.if_time_window,f'{r["run_id"]}: not OnlineSystem')
                log=(directory/'stdout.log').read_text()
                check('Start to learn singly' not in log and 'Pretrain '+r['solver']+' for ' not in log and 'Save model to ' not in log,f'{r["run_id"]}: training observed during evaluation')
                check(not (directory/'logs/training_info.csv').exists(),f'{r["run_id"]}: optimizer update telemetry exists during evaluation')
                if r['category']=='learned':
                    check(config.solver.pretrained_model_path==r['model_path'],f'{r["run_id"]}: wrong loaded checkpoint path')
                    check(not (directory/'models/model.pkl').exists(),f'{r["run_id"]}: evaluation unexpectedly saved model')
        request_count=resolved['resolved_environment']['v_sim_setting']['num_v_nets']
        errors.extend(evaluation_errors(frame,request_count));errors.extend(metric_errors(frame,METRICS))
        errors.extend(fairness_errors(frame))
        job_map={j['run_id']:j for j in expected}
        for _,row in frame.iterrows():
            j=job_map[row.run_id]
            check(row.method==j['method'] and row.eval_seed==j['eval_seed'],f'{row.run_id}: wrong method/evaluation seed in aggregation')
            check((pd.isna(row.train_seed) and j['train_seed'] is None) or row.train_seed==j['train_seed'],f'{row.run_id}: wrong training seed in aggregation')
        for _,r in frame.iterrows():
            arrivals=pd.read_csv(Path(r.run_directory)/'assessment_arrivals.csv')
            accepted=arrivals.loc[arrivals.result.astype(str).str.lower().eq('true')]
            for key in ('availability_requirement','delivered_availability','sla_shortfall'):
                values=arrivals[key] if key=='availability_requirement' else accepted[key]
                check(np.isfinite(values).all(),f'{r.run_id}: invalid raw {key}')
                check((values>=0).all() if key=='sla_shortfall' else values.between(0,1).all(),f'{r.run_id}: raw {key} outside range')
        by_train,by_eval,summary=tables(frame,resolved['experiment']['metrics'])
        for name,table in [('results_by_train_seed',by_train),('results_by_eval_seed',by_eval),('results_summary',summary),
                           ('paired_differences_by_eval_seed',paired_differences(by_eval,resolved['experiment']['metrics']))]:
            same_frame(pd.read_csv(bundle/'aggregated'/f'{name}.csv'),table,name)
        for file in ('input_config.yaml','cli_overrides.yaml','raw_virne_overrides.yaml','resolved_config.yaml','execution_summary.yaml','source_commit.txt','source_status.txt','hardware.json','commands.sh','manifest.json','manifest.csv'):
            check((bundle/file).is_file(),f'Missing provenance: {file}')
        warnings=list(resolved['warnings'])+['Preliminary pipeline-validation results; scientific parameters remain unfrozen.',
                  'Post-hoc baseline assessment preserves disabled reliability/SLA execution and original observations/reward.',
                  'Crossed training/evaluation observations are dependent; train_std and eval_std describe distinct variation, not confidence intervals.']
        if (frame.acceptance_rate.isin([0,1])).any():warnings.append('Degenerate acceptance in some small evaluation streams.')
        result={'passed':not errors,'hard_errors':errors,'warnings':warnings,'jobs':len(manifest),'evaluations':len(frame),
                'shared_inputs_verified':not any('fingerprint mismatch' in error for error in errors)}
    except Exception as exc:
        result={'passed':False,'hard_errors':errors+[str(exc)],'warnings':resolved.get('warnings',[])}
        write_json(bundle/'reports/validation_report.json',result)
        (bundle/'reports/validation_report.md').write_text('# Validation FAILED\n\n'+json.dumps(result,indent=2)+'\n')
        raise
    write_json(bundle/'reports/validation_report.json',result)
    (bundle/'reports/validation_report.md').write_text('# Structural validation\n\n'+json.dumps(result,indent=2)+'\n')
    if errors:raise ValueError('Structural validation failed: '+'; '.join(errors))
    return result
