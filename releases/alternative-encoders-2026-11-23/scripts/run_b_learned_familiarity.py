"""Matched split experiment for a supervised head with one extra training-only feature."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, read, save
from superagent import ab_run
from superagent.ab_methods import PREFERENCE_CATALOG_ENV, TEXT_DEEP, TEXT_FAMILIARITY
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.external_assets import model_resources, verify_resources
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups

SEED = 29111
SPLITS = ('primary', 'confirmation_1')
ARMS = {'original': TEXT_DEEP, 'familiarity': TEXT_FAMILIARITY}
FULL_PACKET = '53ab4dee7794fafa303d4a3dabdc26d9fac98250e5a5b918b16bf369542d8158'


def worker(payload, arm, out, env, caps):
    out.mkdir(exist_ok=True)
    admission, ledger, physical = [out / n for n in ('admission.json', 'fits.json', 'physical.jsonl')]
    file = out / 'worker.joblib'
    name = ARMS[arm]
    if not admission.exists():
        save(admission, dict(maximum_classifier_fits=1, ledger_path=str(ledger)))
        args = [sys.executable, '-m', 'superagent.experiment_worker', str(payload), name, str(file),
                '--fit-admission', str(admission)]
        with (out / 'worker.log').open('x') as log:
            subprocess.run(args, cwd=ROOT, timeout=360, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(env, AB_B1_FIT_LOG=str(physical), AB_B1_PER_WORKER_FIT_LIMIT='1'))
    entries = [json.loads(line) for line in physical.read_text().splitlines()]
    account = read(ledger)
    check(len(entries) == account['attempted_classifier_fits'] == account['completed_classifier_fits'] == 1
          and account['status'] == 'completed' and entries[0]['candidate'] == account['candidate'] == name,
          'Physical fit counts or worker identity differ')
    artifact = joblib.load(file)
    facts = artifact['metrics']
    config = facts['resolved_configuration']
    check(config['id'] == name and config['params'] == caps[name]['params'] and facts['classifier_fits'] == 1,
          'Method differs from the registered candidate')
    usage = facts['training_metadata']['usage']
    check(usage['api_calls'] == usage['api_rmb'] == 0, 'Encoder incurred API usage')
    receipt = dict(method=name, payload=ab_run.sha(payload), model=ab_run.sha(file),
                   classifier_fits=1, fit_ledger=ab_run.sha(ledger), physical_log=ab_run.sha(physical),
                   training_seconds=facts['training_seconds'], encoder_tokens=usage['input_tokens'])
    if arm == 'familiarity':
        pipeline = artifact['model']
        features = pipeline.named_steps['features']
        receipt.update(familiarity_rms=features.familiarity_scale_,
                       familiarity_coefficient=float(pipeline.named_steps['classifier'].coef_[0, -1]),
                       training_titles=len(features.corpus_.title_digests_), vocabulary=len(features.corpus_.vocabulary_))
    frozen_save(out / 'receipt.json', receipt)
    return artifact, receipt


def differences(new, old, labels, groups):
    wins, reference = new == labels, old == labels
    return dict(rows=len(labels), original_correct=int(reference.sum()), familiarity_correct=int(wins.sum()),
                gained=int((wins & ~reference).sum()), lost=int((reference & ~wins).sum()),
                delta=paired(wins.astype(int) - reference, groups))


def evaluate_reserve(inputs, historical, out, env, caps):
    prior = historical / 'deep-e5'
    packet = prior / 'refit_data.joblib'
    check(ab_run.sha(packet) == FULL_PACKET, 'Full training set differs')
    artifact, receipt = worker(packet, 'familiarity', out / 'full', env, caps)
    check(artifact['metrics']['training_rows'] == 6608, 'Unexpected full-training row count')
    folder = out / 'full'
    model = artifact['model']
    path = folder / 'selected_model.joblib'
    if not path.exists():
        shutil.copyfile(prior / 'task_spec.json', folder / 'task_spec.json')
        joblib.dump(dict(kind='pair_text', candidate=TEXT_FAMILIARITY, model=model), path)
    resources, _ = model_resources(model)
    verify_resources(resources)
    check([r['name'] for r in resources] == ['minilm_l12'], 'Unexpected downstream encoder')
    frozen_save(folder / 'final_selection.json', dict(selected_experiment=TEXT_FAMILIARITY, family=TEXT_FAMILIARITY,
        model_file=path.name, model_format='candidate_bundle_v1', model_sha256=ab_run.sha(path),
        task_spec_sha256=ab_run.sha(prior / 'task_spec.json'), training_rows=6608,
        test_accessed=False, refit=True, choice_source='matched feature study', resources=resources))
    original = read(historical / 'exports_frozen.json')['exports']['deep-e5']
    new = verified_export(folder, inputs, env, out)
    frozen_save(out / 'exports_frozen.json', dict(original=original, familiarity=new, refit=receipt['model']))
    expected = read(historical / 'protocol.json')['identities']['packets']
    check(all(ab_run.sha(inputs / n) == expected[n] for n in ('features.zip', 'scoring_only.json', 'groups_only.json')),
          'Reserve packets changed')
    y = np.asarray(read(inputs / 'scoring_only.json')['labels'])
    g = np.asarray(read(inputs / 'groups_only.json'))
    base = classes(historical / 'deep-e5_export_0/rows.csv', original, len(y))
    check(len(y) == len(g) == 1693 and int((base == y).sum()) == 1224, 'Historical baseline differs')
    return differences(classes(out / 'full_export_0/rows.csv', new, len(y)), base, y, g)


def run(workspace, inputs, historical, out):
    out.mkdir(parents=True, exist_ok=True)
    env = dict(offline_environment(workspace), **{PREFERENCE_CATALOG_ENV: 'deep-familiarity'})
    reference = read(historical / 'protocol.json')
    packet = inputs / 'anonymous.zip'
    check(ab_run.sha(packet) == reference['identities']['packets']['anonymous.zip'], 'Training packet changed')
    task = load_development_task(packet)
    spec, metric = TaskSpec.from_task(task).record(), MetricSpec.for_task(task)
    check(len(task.y_train) == 6608 and metric.name == 'accuracy', 'Unexpected task contract')
    groups = sample_groups(task, 'shared_text')
    with patch.dict(os.environ, env):
        caps = ab_run.registered_capabilities(spec, len(groups))
        with patch.dict(os.environ, {PREFERENCE_CATALOG_ENV: 'deep'}):
            old = ab_run.registered_capabilities(spec, len(groups))
        check([(n, c) for n, c in caps.items() if n != TEXT_FAMILIARITY] == list(old.items())
              and TEXT_DEEP in old and TEXT_FAMILIARITY in caps, 'Catalog changed beyond the declared feature')
        plan = ROOT / 'docs/batches/B_learned_familiarity_1123_zh.md'
        frozen_save(out / 'protocol.json', dict(seed=SEED, training=ab_run.sha(packet), plan=ab_run.sha(plan),
            runner=ab_run.sha(Path(__file__)), production=ab_run.sources(), arms=ARMS, inner_splits=list(SPLITS),
            max_new_fits=5, model_api_cap=0, minimum_pooled_gain=.002, group_resamples=5000))
        partitions = out / 'partitions'
        if not partitions.exists():
            partitions.mkdir()
            ab_run.freeze_partitions(task, partitions, SEED, metric)
        reports, all_delta, all_groups = {}, [], []
        for index, split in enumerate(SPLITS):
            payload = partitions / (split + '.joblib')
            seal = read(partitions / (split + '_split.json'))
            held = np.asarray(seal['search_indices'])
            y, g = task.y_train[held], groups[held]
            check(seal['seed'] == SEED + 101 * index and seal['row_ids_sha256'] == fingerprint(held)
                  and seal['label_sha256'] == fingerprint(y) and seal['evaluation_payload_sha256'] == ab_run.sha(payload)
                  and seal['search_rows_sharing_text'] == 0 and not set(groups[seal['train_indices']]).intersection(g),
                  'Unexpected split or shared title in training and evaluation')
            rows, receipts = {}, {}
            for arm in ARMS:
                artifact, receipts[arm] = worker(payload, arm, out / (split + '_' + arm), env, caps)
                hard = np.asarray(artifact['search_predictions'])
                pred = artifact['prediction_record']
                check(pred['evaluation_payload_sha256'] == seal['evaluation_payload_sha256'] and
                      pred['label_sha256'] == seal['label_sha256'] and pred['prediction_sha256'] == fingerprint(hard)
                      and pred['rows'] == len(held) and np.array_equal(hard, pred['predictions']), 'Mismatched row evidence')
                rows[arm] = hard
            report = differences(rows['familiarity'], rows['original'], y, g)
            reports[split] = dict(**report, row_order_sha256=fingerprint(held), workers=receipts)
            all_delta.extend((rows['familiarity'] == y).astype(int) - (rows['original'] == y))
            all_groups.extend(g)
            print(split, report, flush=True)
        aggregate = paired(np.asarray(all_delta), np.asarray(all_groups))
        passed = all(r['gained'] > r['lost'] for r in reports.values()) and aggregate['net_pp'] >= .2
        frozen_save(out / 'inner_results.json', dict(partitions=reports, pooled_observations=len(all_delta),
            pooled_delta=aggregate, new_fits=4, passed=passed))
        reserve = evaluate_reserve(inputs, historical, out, env, caps) if passed else None
        result = dict(seed=SEED, partitions=reports, pooled_observations=len(all_delta),
            pooled_delta=aggregate, qualified_for_full_refit=passed, development_reserve=reserve,
            actual_new_supervised_fits=4 + int(passed), model_api_calls=0, model_api_rmb=0,
            course_designated_validation_used=False, formal_test_used=False, autonomous_selection_run=False)
        frozen_save(out / 'results.json', result)
        frozen_save(out / 'completion.json', dict(status='complete', results=ab_run.sha(out / 'results.json'),
            protocol=ab_run.sha(out / 'protocol.json'), physical_fits=sum(len(f.read_text().splitlines())
            for f in out.glob('*/physical.jsonl')), course_designated_validation_used=False, formal_test_used=False))
        print('completion', read(out / 'completion.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('workspace', 'inputs', 'historical', 'output'):
        p.add_argument('--' + k, type=Path, required=True)
    a = p.parse_args()
    run(*(getattr(a, k).resolve() for k in ('workspace', 'inputs', 'historical', 'output')))
