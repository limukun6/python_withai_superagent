"""Replicate frozen relation predictions using the two catalogue scopes of the official CLI."""
import argparse
import os
from pathlib import Path
import shutil
from unittest.mock import patch

from scripts import run_b1_ab_acceptance as rec, run_b1_05_pair_pilot as pilot
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from tools.ab_registered_rows import registered_partitioner


REFERENCES = {'baseline': 'br_02_relation', 'full': 'br_03_ordered_features'}


def run(prepared, output):
    output.mkdir(parents=True, exist_ok=False)
    rec.offline(output)
    os.environ.update(SUPERAGENT_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    archive = rec.ROOT / 'research'
    registration = rec.read(archive / 'br_01_relation_groups/results.json')
    train = prepared / 'optimizer/train.zip'
    rows = prepared / 'optimizer/inner.json'
    features = prepared / 'delivery/features.zip'
    source = rec.digest(ab_run.sources())
    plan = rec.ROOT / 'docs/batches/BA19_01_native_relation_plan.md'
    rec.write(output / 'protocol.json', dict(batch='BA19-01', source_sha256=source,
        runner_sha256=ab_run.sha(Path(__file__)), plan_sha256=ab_run.sha(plan), order=list(REFERENCES),
        train_sha256=registration['train_sha256'], split_sha256=registration['signed_split_sha256'],
        features_sha256=registration['features_sha256'], seeds=[707,808,909], per_path_fit_cap=11,
        per_path_seconds=1200, worker_seconds=300, model_api_limit=0))
    try:
        files = [(train,'train_sha256'), (rows,'signed_split_sha256'), (features,'features_sha256')]
        if any(ab_run.sha(path) != registration[key] for path,key in files):
            raise ValueError('Registered training, split or unlabelled prediction packet differs')
        task = load_development_task(train)
        if task.X_val is not None or task.y_val is not None:
            raise ValueError('Optimizer received validation data')
        spec = TaskSpec.from_task(task).record()
        partition = registered_partitioner(task, rows, registration['signed_split_sha256'])
        registered = output / 'registered'
        registered.mkdir()
        with patch.object(ab_run, 'grouped_partition', partition):
            _, splits = ab_run.freeze_partitions(task, registered, 707, MetricSpec.for_task(task))
        results = {}
        for scope, reference in REFERENCES.items():
            catalog = ab_run.registered_capabilities(spec, len(task.y_train), scope)
            expected_ids = list(registration['menu'])
            expected_digest = rec.digest(registration['menu'])
            if scope == 'full':
                old = rec.read(archive / reference / 'preflight.json')
                expected_ids, expected_digest = old['candidate_ids'], old['catalog_sha256']
            if list(catalog) != expected_ids or rec.digest(catalog) != expected_digest:
                raise ValueError('Native scope differs from the registered catalogue: ' + scope)
            case = output / scope
            case.mkdir()
            shutil.copy2(train, case / 'train.zip')
            shutil.copytree(registered, case / 'registered')
            anchor = next(iter(catalog))
            before = len(rec.trace(output))
            launcher = ('-m','tools.ab_registered_run','--split-manifest',rows,'--split-sha256',
                        registration['signed_split_sha256'],'--','run','--catalog-scope',scope)
            entry = dict(name=scope, files={'train.zip':registration['train_sha256']},
                         splits=splits, catalog_sha256=expected_digest)
            trained = pilot.train_one(case, entry, source, launcher=launcher)
            manifest = rec.read(case / 'run/run_manifest.json')
            if (manifest['catalog_scope'] != scope or list(manifest['capabilities']) != expected_ids
                    or trained['actions'][0]['candidate_id'] != anchor
                    or len(rec.trace(output)) - before != trained['fits']):
                raise ValueError('CLI catalogue, first choice or observed fit count differs')
            results[scope] = trained
        rec.write(output / 'all_models_frozen.json', {s:r['model_sha256'] for s,r in results.items()})
        for scope, reference in REFERENCES.items():
            case = output / scope
            official, relocated = case/'run', case/'relocated'
            selection = rec.read(official / 'final_selection.json')
            relocated.mkdir()
            for name in ('task_spec.json','final_selection.json',selection['model_file']):
                shutil.copy2(official/name, relocated/name)
            digests = []
            for number, model in enumerate((official,relocated)):
                csv = case / f'export_{number}/rows.csv'
                pilot.child(['-m','superagent','predict',features,'--run-dir',model,'--output',csv],
                            case / f'predict_{number}.log', pilot.TRIAL)
                read_export(csv, registration['outer']['held_rows'])
                digests.append(ab_run.sha(csv))
            prior = rec.read(archive / reference / 'results.json')
            results[scope]['delivery'] = dict(csv_sha256=digests, reference_csv_sha256=prior['two_identical_prediction_exports'],
                rows=registration['outer']['held_rows'], relocated_identical=digests[0]==digests[1],
                reproduced_reference=digests[0]==prior['two_identical_prediction_exports'])
        rec.write(output / 'results.json', results)
        if any(not row['delivery'][condition] for row in results.values()
               for condition in ('relocated_identical','reproduced_reference')):
            raise ValueError('Frozen prediction comparison did not reproduce')
        if source != rec.digest(ab_run.sources()):
            raise ValueError('Production code changed during the comparison')
        rec.write(output / 'completion.json',dict(status='reproduced', full_autonomous_runs=2,
            actual_classifier_fits=len(rec.trace(output)), model_api_requests=0, course_rmb=0,
            freshly_scored_labels=0, fresh_prediction_processes=4))
    except Exception as error:
        rec.write(output/'completion.json', dict(status='failed', error=str(error),
            actual_classifier_fits=len(rec.trace(output)), model_api_requests=0))
        raise


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for option in ('prepared','output'):
        cli.add_argument('--'+option, required=True, type=Path)
    args = cli.parse_args()
    run(args.prepared.resolve(), args.output.resolve())
