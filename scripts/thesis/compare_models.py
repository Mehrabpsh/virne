#!/usr/bin/env python3
"""Check exact policy tensor equality for two locally generated checkpoints."""
import argparse
import json
from pathlib import Path

import torch


def compare(first: Path, second: Path) -> dict:
    left = torch.load(first, map_location='cpu', weights_only=False)['policy']
    right = torch.load(second, map_location='cpu', weights_only=False)['policy']
    if left.keys() != right.keys():
        raise AssertionError('Checkpoint policy keys differ')
    different = [key for key in left if not torch.equal(left[key], right[key])]
    if different:
        raise AssertionError(f'Policy tensors differ: {different}')
    return {'passed': True, 'policy_tensors_compared': len(left),
            'first': str(first), 'second': str(second)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.dumps(compare(args.first, args.second), indent=2)
    args.output.write_text(report + '\n')
    print(report)
