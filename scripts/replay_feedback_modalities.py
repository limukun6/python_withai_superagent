"""Compare feedback-local with frozen predictions of the previous release on a public development panel."""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from scripts.verify_anonymous_planner import execute


def prepare(args):
    previous = read(args.source/'protocol.json')
    plan = ROOT/'research/reliable_delivery_1127/modality_protocol.json'
    expected = read(plan)
    require(sha(args.source/'protocol.json') == expected['parent_protocol_sha256'], 'Different public panel')
    destinations = {}
    for case, spec in previous['cases'].items():
        folder = args.source/case
        require(all(sha(folder/name) == value for name, value in spec['packets_sha256'].items()), 'Input changed')
        destinations[case] = read(folder/'sealed.json')['predictions']
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'protocol.json', dict(agent=source_identity(ROOT), script=sha(Path(__file__)),
         plan=sha(plan), parent_protocol=expected['parent_protocol_sha256'], frozen_parent=destinations))


def run(args, protocol):
    spec = read(args.source/'protocol.json')['cases'][args.case]
    original = args.source/args.case
    dest = args.output/args.case
    dest.mkdir()
    fits = dest/'physical.jsonl'
    features = original/spec['inputs']['original']
    settings = env(args.weights, fits)
    for name in ('SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_TASK_BUDGET_ID',
                 'SUPERAGENT_RESEARCH_REFLECTION_OUTBOX','REFLECTION_STUDY_LIVE','AB_B1_PER_WORKER_FIT_LIMIT',
                 'SUPERAGENT_LESSON_STORE','SUPERAGENT_LESSONS_READONLY'):
        settings.pop(name, None)
    settings.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/reflection_study_observer', ROOT))),
                    PROPOSAL_STUDY_OUTER_LABELS=str(args.source/'evaluation'),
                    PROPOSAL_STUDY_OUTER_FEATURES=str(features))
    require(all(sha(original/name) == value for name, value in spec['packets_sha256'].items()), 'Input changed')
    run_dir = dest/'run'
    primary = dest/'original'/'predictions.csv'
    execute(['solve',original/'train.zip','--prediction-input',features,'--run-dir',run_dir,
             '--output',primary,'--task-analysis','feedback-local'], dest/'solve.log', settings)
    agent = run_dir/'agent'
    outcome = read(agent/'run_outcome.json')
    physical = len(fits.read_text().splitlines())
    require(physical == outcome['attempted_classifier_fits'] <= 10 and outcome['new_api_requests'] == 0 and
            outcome['status'] == 'completed' and not outcome['formal_test_accessed'], 'Completion or budget error')
    first = read(run_dir/'first_delivery'/'prediction_manifest.json')
    require(first['prediction_count'] == spec['heldout'] and first['delivery_status'] == 'partial', 'Missing first output')
    predictions = {}
    for view, filename in spec['inputs'].items():
        path = dest/view/'predictions.csv'
        if view != 'original':
            execute(['predict',original/filename,'--run-dir',run_dir,'--output',path], dest/(view+'.log'), settings)
        replay = dest/(view+'-replay')/'predictions.csv'
        execute(['predict',original/filename,'--run-dir',run_dir,'--output',replay], dest/(view+'-replay.log'), settings)
        require(sha(path) == sha(replay), 'Fresh-process predictions changed')
        predictions[view] = sha(path)
    require(physical == len(fits.read_text().splitlines()), 'Prediction unexpectedly fit a model')
    opening = read(agent/'task_opening.json')
    save(dest/'sealed.json', dict(fits=physical,model_api_requests=0,completed=True,first_rows=first['prediction_count'],
         predictions=predictions,reload_exact=True,reading_check=opening.get('check'),reading_verdict=opening['check_status'],
         disposition=opening['result'],changed_experiment=opening['changed_experiment'],
         members=read(agent/'decisions.json')[-1]['choice']['members']))


def read_labels(path, truth):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require([r['row_id'] for r in rows] == list(map(str,range(len(truth)))), 'Prediction row order differs')
    return np.asarray([r['prediction'] for r in rows], dtype=truth.dtype)


def score(args, protocol):
    previous = read(args.source/'protocol.json')
    sealed = {case: read(args.output/case/'sealed.json') for case in previous['cases']}
    save(args.output/'predictions-frozen.json', {case:r['predictions'] for case,r in sealed.items()})
    for case, info in sealed.items():
        spec = previous['cases'][case]
        truth_file = args.source/'evaluation'/(case+'.npy')
        require(sha(truth_file) == spec['truth_sha256'], 'Held-out labels changed')
        truth = np.load(truth_file, allow_pickle=False)
        metric = accuracy_score if spec['metric'] == 'accuracy' else lambda y,p: f1_score(y,p,average='macro')
        info.update(metric=spec['metric'], views={})
        for view, digest in info['predictions'].items():
            old = args.source/case/(view+'.csv')
            new = args.output/case/view/'predictions.csv'
            require(sha(old) == protocol['frozen_parent'][case][view] and sha(new) == digest, 'Frozen output changed')
            old_labels, new_labels = read_labels(old,truth), read_labels(new,truth)
            info['views'][view] = dict(rows=len(truth), parent=float(metric(truth,old_labels)),
                current=float(metric(truth,new_labels)), same_labels=int(sum(old_labels == new_labels)), identical=sha(old)==digest)
    save(args.output/'results.json', sealed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('prepare','run','score'))
    for key in ('source','output','weights'):
        parser.add_argument('--'+key, type=lambda p:Path(p).resolve(), required=True)
    parser.add_argument('--case', choices=('t01','t02','t03'))
    args = parser.parse_args()
    if args.step == 'prepare':
        prepare(args)
    else:
        protocol = read(args.output/'protocol.json')
        require(protocol['agent'] == source_identity(ROOT) and protocol['script'] == sha(Path(__file__)) and
                protocol['parent_protocol'] == sha(args.source/'protocol.json') and
                protocol['plan'] == sha(ROOT/'research/reliable_delivery_1127/modality_protocol.json'),
                'Runtime, evidence, or study changed')
        if args.step == 'score':
            score(args, protocol)
        elif args.case:
            run(args, protocol)
        else:
            parser.error('run requires a case')
