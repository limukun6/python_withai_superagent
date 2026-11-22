"""Read progress without reading any validation scores or sending requests."""
import argparse
import json
from pathlib import Path


def completion_failure(outcome):
    if outcome.get('requested_planner_completed'):
        return None
    if outcome.get('error'):
        return dict(stage='run', error_type=outcome.get('error_type'), error=outcome['error'])
    final = outcome.get('finalization') or {}
    if final.get('refit_failure'):
        return dict(stage='final_refit', **final['refit_failure'])
    if final.get('validation_failure'):
        return dict(stage='validation', detail=final['validation_failure'])
    return dict(stage='completion', error='Requested planner did not complete; inspect preserved outcome')


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
    frozen = [row for row in completed if row['outcome'].get('requested_planner_completed')]
    partial = [dict(path=row['task']+'_'+row['arm'], model_available=row['outcome'].get('model_available'),
                    failure=completion_failure(row['outcome'])) for row in completed if not row['outcome'].get('requested_planner_completed')]
    print(json.dumps(dict(frozen_paths=len(frozen), frozen=[row['task']+'_'+row['arm'] for row in frozen],
        partial_paths=partial, observed_path_outcomes=len(completed),
        completed_fit_counts=[row['outcome']['classifier_fit_accounting']['completed_classifier_fits'] for row in frozen],
        active=current, batch_completion=safe_read(args.matrix / 'completion.json'))))
