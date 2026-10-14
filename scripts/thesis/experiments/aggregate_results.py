"""Keep every available counter metric and summarize numeric columns across seeds."""
import argparse
import json
from pathlib import Path
import pandas as pd
from common import collect_row, write_json


def run(bundle):
    bundle=Path(bundle)
    manifest=json.loads((bundle/'pipeline_manifest.json').read_text())
    rows=[collect_row(r,bundle) for r in manifest if r['phase']=='eval']
    frame=pd.DataFrame(rows)
    out=bundle/'aggregated';out.mkdir(exist_ok=True)
    frame.to_csv(out/'results_long.csv',index=False)
    numeric=[c for c in frame.select_dtypes(include='number').columns if c not in ('train_seed','test_seed','seed')]
    summary=frame.groupby(['method','label'],sort=False)[numeric].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index()
    summary.to_csv(out/'results_summary.csv',index=False)
    write_json(out/'results_summary.json',summary.to_dict('records'))
    return frame

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('bundle',type=Path);args=a.parse_args();run(args.bundle)
