"""Self-contained human and machine reports with explicit statistical semantics."""
import json
import html
import datetime
import argparse
from pathlib import Path
import pandas as pd
import yaml
from scripts.thesis.experiments.common import write_json

STATS='''For learned methods, average each metric across evaluation seeds within each trained model,
then average those model means. train_std is sample SD (ddof=1) across model means.
eval_std is sample SD across evaluation-scenario means after averaging models within each scenario.
For heuristics, mean and eval_std use evaluation seeds; train_std is null.
A single valid observation has undefined SD (null/NaN), not zero. Missing conditional SLA means
remain null and valid counts are reported. These crossed observations are dependent; neither SD
is a confidence interval. Paired differences average models within each evaluation scenario before
subtracting methods. All long-format observations remain available.'''


def run(bundle,resolved,summary,manifest,validation):
    bundle=Path(bundle);out=bundle/'reports'
    render_html(bundle,resolved,summary,manifest,validation)
    frame=pd.read_csv(bundle/'aggregated/results_summary.csv')
    metrics=resolved['experiment']['metrics']
    table='| Method | '+' | '.join(m+' mean' for m in metrics)+' |\n| --- | '+' | '.join('---' for _ in metrics)+' |\n'
    for _,row in frame.iterrows():table+='| '+str(row.label)+' | '+' | '.join('NaN' if pd.isna(row[m+'_mean']) else f'{row[m+"_mean"]:.6g}' for m in metrics)+' |\n'
    plots=sorted((bundle/'plots').glob('*')) if (bundle/'plots').exists() else []
    hardware=json.loads((bundle/'hardware.json').read_text())
    text=('# Thesis Experiment Framework v2 — preliminary pipeline validation\n\n'
          '**Engineering validation only. Objective, reward, beta, comparator settings and scientific budgets remain unfrozen.**\n\n'
          f'Source commit: `{(bundle/"source_commit.txt").read_text().strip()}`. Source status, tracked patch and experiment-source snapshots are preserved.\n\n'
          '## Hardware and execution\n\n```yaml\n'+yaml.safe_dump(hardware,sort_keys=False)+'```\n\n'
          f'Jobs: {len(manifest)} successful, 0 failed. Calibration executed by this runner: **no**.\n\n'
          '## Frozen experiment\n\n[Resolved config](../resolved_config.yaml), [original input](../input_config.yaml), '
          '[CLI overrides](../cli_overrides.yaml), [raw Virne overrides](../raw_virne_overrides.yaml), '
          '[execution summary](../execution_summary.yaml), [manifest](../manifest.csv).\n\n'
          'Protocol:\n\n```yaml\n'+yaml.safe_dump(resolved['experiment']['protocol'],sort_keys=False)+'```\n\n'
          'Active method settings:\n\n```yaml\n'+yaml.safe_dump(resolved['experiment']['methods'],sort_keys=False)+'```\n\n'
          '## Calibration provenance\n\n```yaml\n'+yaml.safe_dump({'result':resolved['experiment'].get('calibration_result'),
              'suggestions':resolved['experiment'].get('calibration_suggestions'),'overridden_values':resolved['calibrated_overrides']},sort_keys=False)+'```\n\n'
          '## Metrics\n\n'+table+'\n'+STATS+'\n\n'
          'Virne Counter definitions are retained, including average R2C over all arrivals. SLA means use accepted services; '
          'SLA service rate uses all arrivals. Baseline reliability/SLA remain disabled in the solver; recorded mappings '
          'are assessed post hoc in the common world using the frozen evaluator. The configurable assessment_beta is '
          'used only by that API; only availability/shortfall/satisfaction are retained, so it does not change baseline rewards or define an objective.\n\n'
          '## Fairness and validation\n\n'
          'Every enabled method resolves to the same protected environment. Full regenerated topology/resource/reliability/VNR/time/SLA '
          'snapshots are compared per evaluation seed after all overrides. Record IDs, times, sizes and resource-demand totals '
          'are checked against regeneration. This proves deterministic seeded generation; it is not runtime instrumentation of every input attribute. '
          'Final departures restore resources. Learned evaluations must reload their expected checkpoint and contain no training/optimizer telemetry.\n\n'
          '[Structural validation](validation_report.md).\n\n'
          '## Warnings\n\n'+''.join('- '+v+'\n' for v in validation['warnings'])+'\n'
          'Cand-RL remains a thesis_joint reconstruction with custom Cand MLP actor/critic and redundancy=none. RA parameters and static '
          'Weibull/history assumptions are provisional; no exact Paper-9 reproduction or scientific superiority is claimed. '
          'One training epoch cannot establish convergence.\n\n'
          '## Figures\n\n'+''.join(f'- [{p.name}](../plots/{p.name})\n' for p in plots if p.suffix in ('.png','.pdf')))
    if resolved['experiment']['reporting']['generate_markdown_report']:(out/'experiment_report.md').write_text(text)
    write_json(out/'experiment_report.json',{'hardware':hardware,'execution_summary':summary,'resolved_config':resolved,
               'manifest':manifest,'validation':validation,'statistics_semantics':STATS,'results':frame.to_dict('records')})


def read_json(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def read_csv(path):
    if not path.is_file() or not path.stat().st_size:return pd.DataFrame()
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def render_html(bundle,resolved,summary,manifest,validation=None):
    """Render existing artifacts only; never launch jobs or reaggregate metrics."""
    from ..execution.planner import system_mode
    bundle=Path(bundle).resolve();out=bundle/'reports';out.mkdir(exist_ok=True)
    esc=lambda value:html.escape(str(value),quote=True)
    def table(frame):
        if frame.empty:return '<p class="muted">No observations available.</p>'
        return '<div class="scroll">'+frame.to_html(index=False,escape=True,na_rep='—',float_format=lambda x:f'{x:.6g}')+'</div>'
    def dump(value):return '<pre>'+esc(yaml.safe_dump(value,sort_keys=False))+'</pre>'
    def link(path):
        import os
        relative=os.path.relpath(path,out)
        from urllib.parse import quote
        return '<a href="'+quote(relative,safe='/')+'">'+esc(os.path.relpath(path,bundle))+'</a>'
    validation=validation or read_json(out/'validation_report.json',{'passed':False,'warnings':['Validation unavailable'],'hard_errors':[]})
    aggregate=read_csv(bundle/'aggregated/results_summary.csv')
    planned=summary.get('jobs',[]);by_id={r['run_id']:r for r in manifest}
    failed=read_json(bundle/'FAILED.json',{})
    statuses=[]
    for job in planned:
        entry=by_id.get(job['run_id'])
        status='completed' if entry and entry.get('completed') else 'skipped / pending'
        if status!='completed' and failed.get('failed_run_id')==job['run_id']:
            status='failed / interrupted'
        statuses.append({'method':job['method'],'phase':job['phase'],'run_id':job['run_id'],'status':status})
    status_frame=pd.DataFrame(statuses)
    meta=read_json(bundle/'source_metadata.json',{})
    completion=read_json(bundle/'completion.json',{})
    generated=datetime.datetime.now(datetime.timezone.utc).isoformat()
    overview={'experiment':resolved['experiment']['experiment'],'bundle':str(bundle),
        'git_commit':(bundle/'source_commit.txt').read_text().strip() if (bundle/'source_commit.txt').exists() else 'unavailable',
        'start_utc':meta.get('timestamp_utc','unavailable'),'end_utc':completion.get('end_utc','not recorded'),
        'report_generated_utc':generated,'hardware':summary.get('hardware',{}),
        'methods':list(resolved['selected_methods']),'protocol':resolved['experiment']['protocol'],
        'Training system':system_mode(resolved,'train'),'Evaluation system':system_mode(resolved,'eval'),
        'calibration_executed':False}
    sections=[]
    def section(number,title,body):sections.append((str(number),title,body))
    section(1,'Experiment Overview',dump(overview))
    counts={'planned':len(planned),'completed':sum(r.get('completed',False) for r in manifest),
            'failed_or_interrupted':sum(s['status']=='failed / interrupted' for s in statuses),
            'pipeline_failed':bool(failed),'skipped_or_pending':sum(s['status']=='skipped / pending' for s in statuses)}
    method_status=status_frame.groupby(['method','status']).size().rename('jobs').reset_index() if not status_frame.empty else pd.DataFrame()
    section(2,'Job Status',dump(counts)+table(method_status)+table(status_frame)+(dump(failed) if failed else ''))
    configs={}
    for name,method in resolved['selected_methods'].items():
        config=method['virne_config']
        configs[name]={'solver':method['solver'],'specification':resolved['experiment']['methods'][name],
            'resolved_learning':config.get('training'), 'resolved_rl':config.get('rl'),
            'joint_objective':config.get('joint_objective'),'reward':config.get('reward')}
    section(3,'Configuration Summary','<h3>Common protected environment</h3>'+dump(resolved['resolved_environment'])+
            '<h3>Method settings and learning configuration</h3>'+dump(configs))
    tb=bundle/'tensorboard'
    diagnostics=[];training_links=[]
    for entry in manifest:
        if entry['phase']!='train':continue
        directory=Path(entry['run_directory'])
        audit=read_json(directory/'runtime_system.json',{})
        diagnostics.append({'method':entry['method'],'run_id':entry['run_id'],**audit})
        telemetry=directory/'logs/training_info.csv'
        if telemetry.exists():training_links.append(link(telemetry))
    figures=sorted((bundle/'plots').glob('*.png'))
    training_figs=[p for p in figures if any(k in p.name for k in ('train','loss','reward','return','entropy','kl'))]
    def images(paths):return ''.join('<figure><a href="../plots/'+esc(p.name)+'"><img loading="lazy" src="../plots/'+esc(p.name)+'" alt="'+esc(p.stem)+'"></a><figcaption>'+esc(p.stem)+'</figcaption></figure>' for p in paths)
    tb_command='python -m tensorboard.main --logdir '+__import__('shlex').quote(str(tb))+' --host 127.0.0.1 --port 6006'
    section(4,'Training Diagnostics',table(pd.DataFrame(diagnostics))+'<p>Native Virne optimizer telemetry: '+', '.join(training_links)+'</p>'+
        '<p>TensorBoard is a live visualization layer. CSV/JSON artifacts remain authoritative. Root: '+
        (link(tb) if tb.is_dir() else 'unavailable (disabled or legacy bundle)')+'</p><pre>'+esc(tb_command)+'</pre>'+images(training_figs)+
        '<p>Training curves describe the recorded budget; a tiny smoke run does not establish convergence.</p>')
    def metric_table(tokens):
        columns=[c for c in aggregate if c in ('method','label','category') or any(token in c for token in tokens)]
        return table(aggregate[columns]) if columns else table(pd.DataFrame())
    def finalized_runs(tokens):
        observations=read_csv(bundle/'aggregated/results_long.csv')
        columns=[c for c in observations if c in ('method','train_seed','eval_seed') or
                 (pd.api.types.is_numeric_dtype(observations[c]) and any(token in c for token in tokens))]
        return '<h3>Finalized per-run observations</h3>'+table(observations[columns]) if columns else ''
    section(5,'Economic Evaluation',metric_table(['acceptance','r2c','revenue','cost'])+
        finalized_runs(['revenue','cost'])+images([p for p in figures if any(k in p.name for k in ('acceptance','r2c'))]))
    section(6,'Reliability / SLA Evaluation',metric_table(['availability','sla','compliance'])+images([p for p in figures if any(k in p.name for k in ('availability','sla'))]))
    section(7,'System Metrics',metric_table(['utilization','inservice','consumption','resource'])+
        finalized_runs(['utilization','inservice','consumption','resource'])+
        '<p>Only finalized metrics present in the existing aggregation are shown. Unconfigured metrics are not inferred.</p>')
    section(8,'Aggregated Statistics','<p>'+esc(STATS)+'</p>'+table(aggregate)+
        '<h3>Across training seeds</h3>'+table(read_csv(bundle/'aggregated/results_by_train_seed.csv'))+
        '<h3>Across evaluation seeds</h3>'+table(read_csv(bundle/'aggregated/results_by_eval_seed.csv'))+
        '<h3>Paired differences by evaluation seed</h3>'+table(read_csv(bundle/'aggregated/paired_differences_by_eval_seed.csv')))
    section(9,'Validation / Fairness Checks',dump(validation)+
        '<p>Environment fingerprint, scenario fairness, checkpoint reload, resource release and aggregation checks are taken from the existing validator.</p>')
    artifacts=[p for p in bundle.rglob('*') if p.is_file() and ('framework_source' not in p.parts and 'matplotlib_cache' not in p.parts)
               and (p.suffix in ('.csv','.json','.yaml','.pkl','.png','.pdf','.patch','.sh','.tsv') or p.name in ('source_commit.txt','source_status.txt','COMPLETE.txt') or p.name.startswith('events.out.tfevents'))]
    section(10,'Artifact Index','<ul>'+''.join('<li>'+link(p)+'</li>' for p in sorted(artifacts))+'</ul>')
    template=(Path(__file__).parent/'templates/experiment.html').read_text()
    from string import Template
    output=Template(template).substitute(title=esc(resolved['experiment']['experiment']['name']),
        navigation=''.join('<a href="#section-'+i+'">'+i+'. '+esc(title)+'</a>' for i,title,_ in sections),
        sections=''.join('<section id="section-'+i+'"><h2>'+i+'. '+esc(title)+'</h2>'+body+'</section>' for i,title,body in sections))
    (out/'experiment_report.html').write_text(output)
    return out/'experiment_report.html'


def regenerate(bundle):
    """Read a frozen v2 bundle; do not resolve input again or execute any jobs."""
    bundle=Path(bundle).resolve()
    resolved=yaml.safe_load((bundle/'resolved_config.yaml').read_text())
    summary=yaml.safe_load((bundle/'execution_summary.yaml').read_text())
    manifest=read_json(bundle/'manifest.json',[])
    validation=read_json(bundle/'reports/validation_report.json',{'passed':False,'warnings':['Validation unavailable'],'hard_errors':[]})
    if (bundle/'aggregated/results_summary.csv').is_file():run(bundle,resolved,summary,manifest,validation)
    else:render_html(bundle,resolved,summary,manifest,validation)
    return bundle/'reports/experiment_report.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Regenerate reports from an existing frozen experiments_v2 bundle without running jobs.')
    parser.add_argument('--bundle',required=True,type=Path)
    print(regenerate(parser.parse_args().bundle))
