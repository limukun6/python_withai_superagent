"""Verify the shipped default against the two original course archives. No model, route or task hint goes to solve."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, archive, env, read, require, save, sha, title_rows
from scripts.verify_anonymous_planner import execute
from scripts.verify_unified_raw import MEMBERS, feature_package


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    packages = {}
    for tag in MEMBERS:
        folder = args.output / tag
        folder.mkdir()
        feature_package(args.course, tag, folder / 'features.zip')
        packages[tag] = dict(raw=sha(archive(args.course, tag)), features=sha(folder / 'features.zip'))
    save(args.output / 'protocol.json', dict(entry='unified_agent solve, no optional flags, no stdin',
        agent=source_identity(ROOT), runner=sha(Path(__file__)), packages=packages, fits_per_case=10,
        model_requests=0, scoring='Seal predictions of both tasks before consulting the designated validation labels.'))


def run(args):
    tag = args.case
    folder = args.output / tag
    package = archive(args.course, tag)
    spec = read(args.output / 'protocol.json')
    features = folder / 'features.zip'
    require(spec['agent'] == source_identity(ROOT) and spec['runner'] == sha(Path(__file__)), 'Source changed after registration')
    require(spec['packages'][tag] == dict(raw=sha(package), features=sha(features)), 'Input changed')
    fits = folder / 'physical.jsonl'
    access = folder / 'opened.jsonl'
    settings = env(args.weights, fits)
    for name in ('SUPERAGENT_PLANNER_KEY_FILE', 'SUPERAGENT_PLANNER_LEDGER', 'SUPERAGENT_TASK_BUDGET_ID',
                 'SUPERAGENT_RESEARCH_REFLECTION_OUTBOX'):
        settings.pop(name, None)
    settings.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT / 'scripts/unified_acceptance', ROOT))),
                    UNIFIED_RAW_INPUT=str(package), UNIFIED_RAW_MEMBERS=json.dumps(MEMBERS[tag]), UNIFIED_RAW_LOG=str(access))
    result = folder / 'delivery/predictions.csv'
    execute(['solve', package, '--run-dir', folder / 'run', '--prediction-input', features, '--output', result],
            folder / 'solve.log', settings)
    agent = folder / 'run/agent'
    outcome = read(agent / 'run_outcome.json')
    selected = read(agent / 'final_selection.json')
    decisions = read(agent / 'decisions.json')
    counts = len(fits.read_text().splitlines())
    opened = sorted(set(json.loads(line) for line in access.read_text().splitlines()))
    require(outcome['status'] == 'completed' and outcome['attempted_classifier_fits'] == counts <= spec['fits_per_case']
            and outcome['new_api_requests'] == 0 and not outcome['formal_test_accessed'], 'Search or budget mismatch')
    require(opened == sorted(MEMBERS[tag]) and decisions[-1]['choice']['action'] == 'stop' and selected['refit'] and
            selected['model_sha256'] == sha(agent / selected['model_file']), 'Unverified read, stopping action or refit')
    replay = folder / 'replay/predictions.csv'
    execute(['predict', features, '--run-dir', folder / 'run', '--output', replay], folder / 'reload.log', settings)
    require(sha(replay) == sha(result) and counts == len(fits.read_text().splitlines()), 'Reload changed predictions or fitted a model')
    save(folder / 'sealed.json', dict(model_sha256=selected['model_sha256'], prediction_sha256=sha(result),
        members=decisions[-1]['choice']['members'], training_rows=selected['training_rows'], fits=counts,
        actions=[d['choice']['action'] for d in decisions], opened_training_members=opened,
        official_test_access=False, llm_calls=outcome['new_api_requests'], independent_reload=True))
    print(tag, counts, 'fits, predictions frozen', flush=True)


def score(args):
    seals = {tag: read(args.output / tag / 'sealed.json') for tag in MEMBERS}
    save(args.output / 'all_predictions_frozen.json', {tag: r['prediction_sha256'] for tag, r in seals.items()})
    result = {}
    for tag, receipt in seals.items():
        package = archive(args.course, tag)
        if tag == 'table':
            with zipfile.ZipFile(package) as raw:
                truth = np.load(io.BytesIO(raw.read('in_vehicle_coupon_recommendation/y_val.npy')), allow_pickle=False)
        else:
            truth = np.array([row[2] for row in title_rows(package, 'val', labels=True)])
        output = args.output / tag / 'delivery/predictions.csv'
        require(sha(output) == receipt['prediction_sha256'], 'Prediction changed after sealing')
        with output.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        require([r['row_id'] for r in rows] == list(map(str, range(len(truth)))), 'Row order differs')
        values = np.asarray([r['prediction'] for r in rows], dtype=truth.dtype)
        correct = int(np.sum(values == truth))
        result[tag] = dict(receipt, correct=correct, rows=len(truth), accuracy=correct/len(truth))
    save(args.output / 'verified.json', dict(agent=read(args.output / 'protocol.json')['agent'], cases=result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train', 'score'))
    parser.add_argument('--case', choices=tuple(MEMBERS))
    for name in ('output', 'weights', 'course'):
        parser.add_argument('--' + name, type=lambda s: Path(s).resolve(), required=True)
    args = parser.parse_args()
    if args.action == 'train' and not args.case:
        parser.error('Choose a case for train')
    {'prepare': prepare, 'train': run, 'score': score}[args.action](args)
