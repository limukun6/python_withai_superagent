"""Read-only archive of the completed object-image catalogue comparison; no fits or API."""
import argparse
from pathlib import Path
import shutil
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'data', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    completion = read(a.run / 'completion.json')
    rows = read(a.run / 'results.json')
    if completion['status'] != 'completed' or len(rows) != 3:
        raise ValueError('Require three completed and scored paths')
    if sum(r['actual_classifier_fits'] for r in rows) != completion['actual_classifier_fits']:
        raise ValueError('Fit accounting differs')
    a.output.mkdir(parents=True, exist_ok=False)
    for name in ('protocol.json', 'completion.json', 'physical_fits.jsonl',
                 'all_models_frozen.json', 'results.json', 'contrasts.json'):
        shutil.copyfile(a.run / name, a.output / name)
    prep = a.output / 'data_registration'
    prep.mkdir()
    for name in ('protocol.json', 'completion.json', 'registration.json'):
        shutil.copyfile(a.data / name, prep / name)
    summary = []
    for row in rows:
        folder = a.output / row['arm']
        folder.mkdir()
        for name in ('run_manifest.json', 'run_outcome.json', 'actions.json',
                     'final_selection.json', 'classifier_fits.json', 'verification.json',
                     'primary_split.json', 'confirmation_1_split.json', 'confirmation_2_split.json'):
            source = a.run / row['arm'] / name
            if source.exists():
                shutil.copyfile(source, folder / name)
        if not row['evaluation']['reload_identical']:
            raise ValueError('Prediction reload was not verified')
        summary.append(dict(arm=row['arm'], candidates=row['candidates'],
            score=row['evaluation']['score'], heldout_rows=row['evaluation']['rows'],
            fits=row['actual_classifier_fits'], process_seconds=row['process_seconds'],
            selected=row['selection']['single_candidate'],
            prediction_sha256=sha(a.run / row['arm'] / 'predictions/rows.csv'),
            selected_at_trial=row['selection']['selected_experiment'],
            reload_identical=True))
    save(a.output / 'summary.json', dict(arms=summary,
        contrasts=read(a.run / 'contrasts.json'), completion=completion,
        scope='Known B development flower cohort; not unseen H; catalogue availability comparison, not LLM effect',
        archiving_classifier_fits=0, archiving_API_requests=0))
    print(summary)
