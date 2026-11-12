"""Review a saved segmentation audit using the configured Agent provider."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unibrain_agent.agents import Agent
from unibrain_agent.training_diagnosis import diagnostic_context, qualify, review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('configured', 'rules'), default='configured')
    parser.add_argument('--qualify', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    agent = Agent(args.backend, args.output, lambda *a, **kw: None)
    result = {}
    if args.qualify:
        result['qualification'] = qualify(agent)
    context = diagnostic_context(json.loads(args.report.read_text(encoding='utf-8')))
    result['review'] = review(agent, context)
    result['calls'] = agent.calls
    (args.output / 'review.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'provider': agent.calls[-1]['backend'], 'issues': result['review']['issues'],
                      'qualification_passed': result.get('qualification', {}).get('passed')}))


if __name__ == '__main__':
    main()
