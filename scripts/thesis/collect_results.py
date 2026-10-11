#!/usr/bin/env python3
"""Collect per-run CSV summaries plus scientific configuration into JSON/CSV."""
import argparse
import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from virne.core.counter import Counter


def collect(root: Path) -> list[dict]:
    rows = []
    for config_path in sorted(root.glob('*/*/config.yaml')):
        config = OmegaConf.load(config_path)
        run_dir = config_path.parent
        record_paths = sorted(path for path in (run_dir / 'records').glob('*.csv')
                              if not path.name.startswith('temp-'))
        if not record_paths:
            continue
        row = Counter.summary_csv(str(record_paths[-1]))
        row.update(run_id=config.experiment.run_id, solver=config.solver.solver_name,
                   seed=config.experiment.seed, offline=config.system.if_offline_system,
                   train_epochs=config.training.num_train_epochs,
                   reliability_enabled=OmegaConf.select(config, 'reliability.enabled', default=False),
                   reliability_mode=OmegaConf.select(config, 'reliability.mode'),
                   reliability_config_id=OmegaConf.select(config, 'reliability.config_id'),
                   objective=OmegaConf.select(config, 'joint_objective.name'),
                   alpha=OmegaConf.select(config, 'joint_objective.alpha'),
                   beta=OmegaConf.select(config, 'joint_objective.beta'),
                   reward=config.rl.reward_calculator.name,
                   features_json=json.dumps(OmegaConf.to_container(config.reliability_features)),
                   records=str(record_paths[-1].resolve()), run_directory=str(run_dir.resolve()),
                   completed=(run_dir / 'runtime_seconds.txt').is_file())
        if (run_dir / 'runtime_seconds.txt').is_file():
            row['wall_runtime_seconds'] = int((run_dir / 'runtime_seconds.txt').read_text())
        rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output-prefix', type=Path)
    args = parser.parse_args()
    prefix = args.output_prefix or args.root / 'run_manifest'
    frame = pd.DataFrame(collect(args.root))
    frame.to_csv(str(prefix)+'.csv', index=False)
    frame.to_json(str(prefix)+'.json', orient='records', indent=2)
    print(f'Collected {len(frame)} runs into {prefix}.csv/.json')
