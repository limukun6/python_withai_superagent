"""Test the pair that narrowly missed the existing opening gate, on disjoint training groups."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from scripts.run_b_shared_full_task2 import check, read, seal
from superagent.ab_complement import MIN_GAIN, canonical_probabilities
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.ab_run import sha, sources
from superagent.data import load_development_task
from superagent.models import predict_with_model, search_candidates
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups, take

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'docs/batches/B_near_threshold_plan_20261123_zh.md'


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = read(args.original / 'protocol.json')
    frozen = read(args.original / 'models_frozen.json')
    check(protocol['production'] == sources() and sha(args.original/'anonymous.zip') == protocol['training']
          and frozen['protocol'] == sha(args.original/'protocol.json'), 'Parent experiment differs')
    run_dir = args.original / 'agent'
    opening = read(run_dir/'primary_split.json')
    decision = read(run_dir/'selection.json')['confirmation']['b_scheduler_review']
    anchor, partner = decision['proposed_pair']
    rows = decision['primary_rows']
    best_pair = max(item['correct'] for item in decision['opening_combinations']
                    if set(item['members']) == {anchor, partner})
    gain = best_pair - decision['opening_correct'][anchor]
    check(anchor == decision['anchor'] == frozen['selection'] and decision['confirmation_rows'] == 0
          and not frozen['components'] and 0 < gain == math.ceil(MIN_GAIN*rows)-1,
          'The prespecified one-row shortfall was not present')
    task = load_development_task(args.original/'anonymous.zip')
    groups = sample_groups(task, 'shared_text')
    excluded = set(groups[opening['search_indices']])
    eligible = list(opening['train_indices'])
    check(not excluded.intersection(groups[eligible]) and sorted(eligible+opening['search_indices']) ==
          list(range(len(groups))), 'Parent split includes overlapping title groups or omits rows')
    salt = 'b-borderline-20261123-v1/'
    buckets = [int(hashlib.sha256((salt+g).encode()).hexdigest(), 16) % 8 for g in groups]
    train = [i for i in eligible if buckets[i] >= 2]
    panels = [[i for i in eligible if buckets[i] == n] for n in (0, 1)]
    sets = [set(groups[ix]) for ix in (train, *panels)]
    check(not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]) and
          all(len(ids) > 100 and set(task.y_train[ids]) == {0, 1} for ids in (train, *panels)),
          'Training and shadow groups must be disjoint and have both classes')
    protocol_file = args.out/'protocol.json'
    seal(protocol_file, dict(plan=sha(PLAN), runner=sha(Path(__file__)), source=protocol['shared_commit'],
         source_run_protocol=sha(args.original/'protocol.json'), source_frozen=sha(args.original/'models_frozen.json'),
         selection_review=sha(run_dir/'selection.json'), original_training=protocol['training'],
         rows_excluded=rows, anchor=anchor, partner=partner, opening_gain=gain, gain_gate=MIN_GAIN,
         training_rows=len(train), training_ids_sha256=fingerprint(train), panels=[dict(rows=len(ids),
         row_ids_sha256=fingerprint(ids)) for ids in panels], group_overlaps=0, classifier_fit_cap=2, model_api_cap=0))
    data = args.out/'training.joblib'
    joblib.dump(dict(kind='pair_text', pair_semantics='preference', X_train=take(task.X_train, train),
                     y_train=task.y_train[train], X_search=take(task.X_train, train[:1]), mode='refit'), data)
    os.environ.update(SUPERAGENT_DEEP_MINILM_WEIGHTS=str(args.weights/'all-MiniLM-L12-v2'),
                      SUPERAGENT_MINILM_WEIGHTS=str(args.weights/'all-MiniLM-L6-v2'),
                      SUPERAGENT_E5_SMALL_WEIGHTS=str(args.weights/'e5-small-v2'), TOKENIZERS_PARALLELISM='false',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    parent_menu = read(run_dir/'run_manifest.json')
    os.environ[PREFERENCE_CATALOG_ENV] = parent_menu['preference_catalog']
    available = {c.family: c.params for c in search_candidates('pair_text', 'all')}
    check(all(available.get(name) == parent_menu['capabilities'][name]['params'] for name in (anchor, partner)),
          'The worker must see the same two candidate configurations as the parent')
    if hasattr(sys, 'addaudithook'):
        def offline(event, args):
            if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
                raise PermissionError('Shadow study is offline')
        sys.addaudithook(offline)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/ab_acceptance_guard', ROOT))),
               AB_B1_OFFLINE='1', AB_B1_FIT_LOG=str(args.out/'physical.jsonl'), AB_B1_PER_WORKER_FIT_LIMIT='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
    models = []
    for n, name in enumerate((anchor, partner)):
        fit = args.out/f'fit_{n}.json'
        admission = args.out/f'admission_{n}.json'
        seal(admission, dict(maximum_classifier_fits=1, ledger_path=str(fit)))
        artifact = args.out/f'worker_{n}.joblib'
        with (args.out/f'worker_{n}.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'superagent.experiment_worker', str(data), name, str(artifact),
                            '--fit-admission', str(admission)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=300, check=True)
        receipt = read(fit)
        check(receipt['attempted_classifier_fits'] == receipt['completed_classifier_fits'] == 1
              and receipt['candidate'] == name and receipt['status'] == 'completed', 'Worker count differs')
        models.append(joblib.load(artifact)['model'])
    observed = [json.loads(line)['candidate'] for line in (args.out/'physical.jsonl').read_text().splitlines()]
    check(observed == [anchor, partner], 'Independent physical counts differ')
    prediction_tables = []
    for ids in panels:
        x = take(task.X_train, ids)
        hard = predict_with_model('pair_text', models[0], x)
        p = np.mean([canonical_probabilities(model, x)[1][:,1] for model in models], axis=0)
        tie = np.asarray([left < right for left, right in x])
        vote = ((p>.5) | ((p == .5) & tie)).astype(int)
        prediction_tables.append((hard, vote))
    seal(args.out/'frozen.json', dict(protocol=sha(protocol_file), models=[sha(args.out/f'worker_{i}.joblib')
         for i in (0,1)], predictions=[[fingerprint(col) for col in preds] for preds in prediction_tables]))
    findings, grouped = [], []
    for ids, (hard, vote) in zip(panels, prediction_tables, strict=True):
        y = task.y_train[ids]
        a, b = hard == y, vote == y
        group_cells = {}
        for g, one, two in zip(groups[ids], a, b, strict=True):
            group_cells.setdefault(g, Counter())[int(one)+2*int(two)] += 1
        grouped.append([[v[k] for k in range(4)] for _,v in sorted(group_cells.items())])
        findings.append(dict(rows=len(y), groups=len(group_cells), anchor=int(a.sum()), pair=int(b.sum()),
                             gained=int((~a&b).sum()), lost=int((a&~b).sum()), net=int(b.sum()-a.sum())))
    seal(args.out/'grouped.json', grouped)
    net = sum(item['net'] for item in findings)
    total = sum(item['rows'] for item in findings)
    result = dict(panels=findings, combined_rows=total, combined_gain=net, gain_percentage_points=100*net/total,
                  meets_gate=all(item['net']>0 for item in findings) and net/total >= MIN_GAIN, physical_fits=2,
                  model_api_calls=0, formal_test_opened=False, designated_validation_opened=False)
    seal(args.out/'results.json', result)
    seal(args.out/'completion.json', {k:sha(args.out/(k+'.json')) for k in ('protocol','frozen','grouped','results')})
    print(json.dumps(result))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for flag in ('original','weights','out'):
        cli.add_argument('--'+flag, type=lambda x:Path(x).resolve(), required=True)
    run(cli.parse_args())
