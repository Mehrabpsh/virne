#!/usr/bin/env python3
"""Compare baseline records exactly, excluding run identity and wall-clock fields."""
import argparse
import json
from pathlib import Path

import pandas as pd


def latest_records(run_dir: Path):
    paths = sorted((run_dir / 'records').glob('*.csv'))
    paths = [path for path in paths if not path.name.startswith('temp-')]
    if not paths:
        raise FileNotFoundError(f'No saved records in {run_dir}')
    return pd.read_csv(paths[-1])


def compare(first: Path, second: Path) -> dict:
    before, after = latest_records(first), latest_records(second)
    excluded = [name for name in before if name in ('run_id', 'start_run_time')
                or 'running_time' in name or 'clock_time' in name]
    columns = [name for name in before if name not in excluded]
    pd.testing.assert_frame_equal(before[columns], after[columns], check_exact=True)
    return {'passed': True, 'rows': len(before), 'columns_compared': columns,
            'excluded': excluded, 'before': str(first), 'after': str(second)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.dumps(compare(args.before, args.after), indent=2)
    print(report)
    if args.output:
        args.output.write_text(report+'\n')
