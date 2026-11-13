"""Exercise full UniBrain and all four policies with two model updates."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unibrain_agent.policy_checkpoint import inspect_policy
from unibrain_agent.replacement_training import parser, train


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--source', type=Path, required=True)
    cli.add_argument('--data', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--dataset', choices=('abide', 'adhd'), required=True)
    cli.add_argument('--fold', type=int, default=1)
    cli.add_argument('--gpu', type=int, default=0)
    cli.add_argument('--threads', type=int, default=2)
    args = cli.parse_args()
    options = ['--method', 'original', '--setting', 'original', '--paper-original',
               '--coordination', 'taskforce', '--taskforce-profile', 'paper', '--coordination-batch-size', '1',
               '--epochs', '2', '--max-steps', '2', '--eval-limit', '2', '--verify-gradients', '--checkpoint-mode', 'resume']
    for key, value in vars(args).items():
        options += ['--' + key, str(value)]
    train(parser().parse_args(options))
    evidence = inspect_policy(args.output / 'last.pt', args.output / 'report.json')
    evidence['scope'] = 'Two full-model updates and one replay update; not convergence or policy effectiveness'
    evidence['replay_batch_override'] = {'acceptance': 1, 'paper_default': 64}
    (args.output / 'policy_check.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
