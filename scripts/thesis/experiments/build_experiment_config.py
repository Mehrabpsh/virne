"""Freeze pipeline settings and fully composed Virne configuration templates."""
import argparse
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from common import ROOT, read_yaml, settings, override, write_yaml


def run(bundle,p,point):
    with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'virne/configs')):
        enabled=compose(config_name='main',overrides=[override(k,v) for k,v in settings(p,point).items()])
        baseline=compose(config_name='main',overrides=[override(k,v) for k,v in settings(p,point,True).items()])
    result={'purpose':'Preliminary pipeline validation; not scientifically frozen',
            'profile':p,'selected_calibration':point,
            'common_virne_config':OmegaConf.to_container(enabled,resolve=True),
            'original_baseline_virne_config':OmegaConf.to_container(baseline,resolve=True),
            'comparator_architecture':'Cand-specific MLP actor/critic with Virne PPO; redundancy none',
            'method_overrides':'Actual final per-job configs stored in runs/**/config.yaml; comparators and features in profile',
            'stream_mechanism':'Seed-reset regeneration, full input hashes and record arrival checks; baseline SLA assessed post hoc'}
    write_yaml(Path(bundle)/'resolved_experiment_config.yaml',result)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('bundle',type=Path);args=a.parse_args()
    run(args.bundle,read_yaml(args.bundle/'active_profile.yaml'),read_yaml(args.bundle/'calibration/recommended_config.yaml'))
