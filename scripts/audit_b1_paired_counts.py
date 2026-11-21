"""Reconcile a completed two-arm run from existing frozen CSVs and independent score receipts."""
import argparse
from pathlib import Path

from scripts import run_b1_ab_acceptance as fixture
from scripts.run_b1_06_mrpc import paired_counts


def reconcile(run, output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = fixture.read(run / 'protocol.json')
    original = fixture.read(run / 'results.json')
    filename = 'results.json'
    fixture.write(output / 'protocol.json', dict(source_batch=protocol['batch'], original_results_sha256=fixture.sha((run / filename).read_bytes()),
                  frozen_models_sha256=fixture.sha((run / 'all_models_frozen.json').read_bytes()),
                  scope='CSV label serialization and paired counts only', classifier_fits_allowed=0, api_allowed=0))
    try:
        arms = {'B1-06': ('lexical', 'extended'), 'B1-07': ('fixed', 'adaptive')}[protocol['batch']]
        frozen = fixture.read(run / 'all_models_frozen.json')
        for path in original['paths']:
            name = path['name']
            if (path['model_sha256'] != frozen[name] or path['delivery']['csv_sha256'] != fixture.sha((run / name / 'pred_0/rows.csv').read_bytes())
                    or protocol['tasks'][name]['holdout_sha256'] != fixture.sha((run / name / 'holdout.joblib').read_bytes())
                    or protocol['tasks'][name]['holdout_sha256'] != fixture.sha((run / str(protocol['tasks'][name]['outer_seed']) / 'holdout.joblib').read_bytes())):
                raise ValueError('Stored model, prediction, or held row identity changed')
        new = {seed: paired_counts(run, int(seed), arms) for seed in original['paired_counts']}
        changed = new != original['paired_counts']
        fixture.write(output / 'results.json', {**original, 'paired_counts': new})
        fixture.write(output / 'completion.json', dict(status='reconciled', paired_summary_changed=changed,
                      independent_frozen_scores_verified=True, classifier_fits=0, predictions_regenerated=0, api_requests=0))
    except Exception as error:
        fixture.write(output / 'completion.json', dict(status='failed', error=str(error), classifier_fits=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('new_output', type=Path)
    args = parser.parse_args()
    reconcile(args.run, args.new_output)
