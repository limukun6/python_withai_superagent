"""Read progress without reading any validation scores or sending requests."""
import argparse
import json
from pathlib import Path


def safe_read(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matrix', type=Path)
    args = parser.parse_args()
    completed = safe_read(args.matrix / 'checkpoint.json', [])
    current = []
    for path in sorted(args.matrix.glob('*/events.jsonl')):
        if (path.parent / 'run_outcome.json').exists():
            continue
        events = []
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        counts = safe_read(path.parent / 'classifier_fits.json', {})
        current.append(dict(path=path.parent.name,
            completed_classifier_fits=counts.get('completed_classifier_fits', 0),
            requests=sum(row.get('kind') == 'llm_request' for row in events),
            responses=sum(row.get('kind') == 'llm_response' for row in events),
            last_event=events[-1]['kind'] if events else None))
    print(json.dumps(dict(frozen_paths=len(completed), frozen=[row['task']+'_'+row['arm'] for row in completed],
        completed_fit_counts=[row['outcome']['classifier_fit_accounting']['completed_classifier_fits'] for row in completed],
        active=current, batch_completion=safe_read(args.matrix / 'completion.json'))))
