"""Archive completed SMS C1 using macro-F1, with accuracy as a separate diagnostic."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import joblib
from sklearn.metrics import f1_score, confusion_matrix
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from superagent.pricing import resolve_pricing

PAIRS = [('L1','L0'), ('R1','R0'), ('E1','E0'), ('E0','R0'), ('E1','R1'),
         ('R0','L0'), ('R1','L1'), ('E0','L0'), ('E1','L1')]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'holdout', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    completion = read(a.run / 'completion.json')
    rows = read(a.run / 'results.json')
    if completion['status'] != 'completed' or {r['arm'] for r in rows} != {'L0','L1','R0','R1','E0','E1'}:
        raise ValueError('Require all six completed and scored models')
    truth = [str(y) for y in joblib.load(a.holdout)['y']]
    a.output.mkdir(parents=True, exist_ok=False)
    predictions = {}
    arms = {}
    for row in rows:
        arm = row['arm']
        directory = Path(row['run_dir'])
        dest = a.output / arm
        dest.mkdir()
        for name in ('run_manifest.json', 'actions.json', 'history.json', 'classifier_fits.json', 'final_selection.json'):
            shutil.copyfile(directory / name, dest / name)
        shutil.copyfile(a.run / (arm+'_verification.json'), dest / 'verification.json')
        with (a.run / 'predictions' / arm / 'rows.csv').open(encoding='utf-8', newline='') as f:
            records = list(csv.DictReader(f))
        if [int(r['row_id']) for r in records] != list(range(len(truth))):
            raise ValueError('Prediction row order differs')
        predictions[arm] = [r['prediction'] for r in records]
        manifest = read(directory / 'run_manifest.json')
        labels = [str(v) for v in manifest['task_contract']['metric']['labels']]
        if manifest['task_contract']['metric']['name'] != 'macro_f1':
            raise ValueError('This study requires macro-F1, not pooled accuracy')
        accuracy = sum(x == y for x, y in zip(predictions[arm], truth, strict=True)) / len(truth)
        observed_score = float(f1_score(truth, predictions[arm], labels=labels, average='macro', zero_division=0))
        if abs(observed_score-row['evaluation']['score']) > 1e-12 or not row['evaluation']['reload_identical']:
            raise ValueError('Prediction score/reload differs')
        usage = []
        usage_file = directory / 'llm_usage.jsonl'
        if usage_file.exists():
            usage = [json.loads(line) for line in usage_file.read_text(encoding='utf-8').splitlines()
                     if json.loads(line).get('event') == 'llm_call']
        inp = sum(r['input_tokens'] for r in usage)
        out = sum(r['output_tokens'] for r in usage)
        cost = resolve_pricing('DeepSeek-V4.1-Flash').course_cost(inp, out)
        outcome = row['outcome']
        if len(usage) != outcome['new_api_requests'] or abs(cost-outcome['known_course_cost_rmb']) > 1e-9:
            raise ValueError('Usage/cost differs from official outcome')
        history = read(directory / 'history.json')
        proposed = [r for r in history if r.get('proposal')]
        evidence = dict(proposals=len(proposed),
            configuration_statuses=dict(Counter(r.get('configuration_fact_check', {}).get('status', 'absent') for r in proposed)),
            hypothesis_statuses=dict(Counter(r.get('hypothesis_check', {}).get('status', 'absent') for r in proposed)),
            purposes=dict(Counter(r['proposal'].get('purpose', 'unstated') for r in proposed)),
            numeric_expectations=sum(r['proposal'].get('expected_change') is not None for r in proposed),
            prose_counts={name: sum(r.get('explanation_audit', {}).get('prose_claims', {}).get(name, 0) for r in proposed)
                          for name in ('claims_checked','confirmed','contradicted','unverifiable','informational')},
            scope='Machine flags from bounded phrase checker, not independent semantic annotations or error incidence')
        arms[arm] = dict(score=observed_score, primary_metric='macro_f1', accuracy=accuracy,
            confusion_labels=labels, confusion_matrix=confusion_matrix(truth, predictions[arm], labels=labels).tolist(),
            rows=len(truth), selected=outcome['selected_candidate'],
            new_fits=row['new_classifier_fits'], reused_fits=row.get('reused_classifier_fits', 0),
            process_seconds=row['process_seconds'], reference_process_seconds=row.get('reference_process_seconds'),
            new_API_requests=len(usage), input_tokens=inp, output_tokens=out, known_estimated_rmb=cost,
            prediction_sha256=sha(a.run / 'predictions' / arm / 'rows.csv'), reload_identical=True,
            executed_candidates=[r['candidate'] for r in history], evidence=evidence)
        save(dest / 'measured_usage_summary.json', dict(requests=len(usage), input_tokens=inp,
            output_tokens=out, known_estimated_rmb=cost))
    contrasts = []
    for left, right in PAIRS:
        lp, rp = predictions[left], predictions[right]
        contrasts.append(dict(left=left, right=right, delta=arms[left]['score']-arms[right]['score'],
            prediction_disagreements=sum(x != z for x, z in zip(lp, rp, strict=True)),
            left_correct_only=sum(x == y and z != y for x, z, y in zip(lp, rp, truth, strict=True)),
            right_correct_only=sum(z == y and x != y for x, z, y in zip(lp, rp, truth, strict=True))))
    save(a.output / 'summary.json', dict(arms=arms, contrasts=contrasts, completion=completion,
        new_input_tokens=sum(r['input_tokens'] for r in arms.values()),
        new_output_tokens=sum(r['output_tokens'] for r in arms.values()),
        known_incremental_estimated_rmb=sum(r['known_estimated_rmb'] for r in arms.values()),
        scope='One known SMS development holdout; all six paths new; macro-F1 primary; not independent H',
        archiving_classifier_fits=0, archiving_API_requests=0))
    for name in ('protocol.json', 'completion.json', 'physical_fits.jsonl', 'all_models_frozen.json'):
        shutil.copyfile(a.run / name, a.output / name)
    print({arm: {k: r[k] for k in ('score','new_fits','reused_fits','new_API_requests','known_estimated_rmb')}
           for arm, r in arms.items()})
