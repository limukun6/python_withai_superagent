"""Evaluate two opening decisions on held title groups using eight existing fitted heads."""
import argparse
from hashlib import sha256
from itertools import combinations
import os
from pathlib import Path
import sys
from unittest.mock import patch

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.check_b_complement_stability import choose
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read
from superagent import ab_run
from superagent.ab_complement import canonical_probabilities
from superagent.ab_methods import TEXT_DEEP
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups

PAIRS = list(combinations(range(4), 2))


def read_fold(directory, global_groups):
    info = read(directory / 'primary_split.json')
    file = directory / 'primary.joblib'
    data = joblib.load(file)
    held = np.asarray(info['search_indices'])
    groups = global_groups[held]
    truth = data['y_search']
    check(ab_run.sha(file) == info['evaluation_payload_sha256'] and info['label_sha256'] == fingerprint(truth)
          and info['row_ids_sha256'] == fingerprint(held) and not set(global_groups[info['train_indices']]).intersection(groups),
          'Evaluation rows, payload or group boundary changed')
    rows = data['X_search']
    hard, probabilities, names, models = [], [], [], []
    for entry in read(directory / 'history.json')[:4]:
        name = entry['candidate']
        path = directory / f"trial_{entry['trial']}.joblib"
        artifact = joblib.load(path)
        rec = artifact['prediction_record']
        check(entry['split_id'] == 'primary' and entry['evaluation_payload_sha256'] == rec['evaluation_payload_sha256']
              == info['evaluation_payload_sha256'] and entry['row_ids_sha256'] == info['row_ids_sha256']
              and entry['label_sha256'] == rec['label_sha256'] == info['label_sha256']
              and artifact['metrics']['resolved_configuration']['id'] == name, 'Checkpoint is from another run')
        preds = artifact['search_predictions']
        check(fingerprint(preds) == rec['prediction_sha256'], 'Checkpoint hard predictions changed')
        classes, prob = canonical_probabilities(artifact['model'], rows)
        check(classes.tolist() == [0, 1] and np.isfinite(prob).all(), 'Invalid probabilities')
        if 'search_scores' in artifact:
            np.testing.assert_array_equal(prob, artifact['search_scores'])
        hard.append(preds)
        probabilities.append(prob[:, 1])
        names.append(name)
        models.append(ab_run.sha(path))
    check(len(set(names)) == 4 and TEXT_DEEP in names, 'Missing shared opening member')
    return dict(truth=truth, groups=groups, ids=held, names=names, hard=np.column_stack(hard),
                prob=np.column_stack(probabilities), tie=np.array([a < b for a, b in rows]),
                models=models, split=info['row_ids_sha256'])


def classify(prob, tie):
    return ((prob > .5) | ((prob == .5) & tie)).astype(int)


def choose_weights(fold, rows):
    y, probs, tie = (fold[k][rows] for k in ('truth', 'prob', 'tie'))
    _, g = np.unique(fold['groups'][rows], return_inverse=True)
    hits = list((fold['hard'][rows] == y[:, None]).T)
    hits.extend(classify((probs[:, a] + probs[:, b]) / 2, tie) == y for a, b in PAIRS)
    hits.append(np.ones(len(rows), dtype=int))
    counts = np.column_stack([np.bincount(g, weights=h) for h in hits]).astype(int)
    leaders, pairs = choose(counts.sum(axis=0, keepdims=True), PAIRS)
    baseline = list(pairs[0]) or [int(leaders[0])]
    weights = np.zeros(4)
    rng = np.random.default_rng(77891)
    for _ in range(100):
        index = rng.integers(0, len(counts), size=(50, len(counts)))
        anchors, picks = choose(counts[index].sum(axis=1), PAIRS)
        for anchor, pick in zip(anchors, picks):
            if pick:
                weights[list(pick)] += .5
            else:
                weights[anchor] += 1
    check(weights.sum() == 5000, 'Each bootstrap draw must contribute one vote')
    top = sorted(range(4), key=lambda i: -weights[i])[:3]
    kept = {i: float(weights[i] / weights[top].sum()) for i in top if weights[i] > 0}
    return baseline, kept


def predictions(source, src, target, dest):
    check(source['names'] == target['names'] and not set(source['groups'][src]).intersection(target['groups'][dest]),
          'Source selection and target scoring share groups or differ in inventory')
    standard, weights = choose_weights(source, src)
    probs, hard, ties = (target[k][dest] for k in ('prob', 'hard', 'tie'))
    raw = hard[:, standard[0]] if len(standard) == 1 else classify(probs[:, standard].mean(axis=1), ties)
    idx = list(weights)
    proposed = hard[:, idx[0]] if len(idx) == 1 else classify(probs[:, idx] @ list(weights.values()), ties)
    return dict(ids=target['ids'][dest].tolist(), original=raw.tolist(), proposed=proposed.tolist(),
                deep=hard[:, target['names'].index(TEXT_DEEP)].tolist(),
                original_choice=[source['names'][i] for i in standard],
                weighted_choice={source['names'][i]: w for i, w in weights.items()},
                selection_rows=len(src), scoring_rows=len(dest), scoring_groups=len(set(target['groups'][dest])))


def alternating_groups(groups):
    ordered = sorted(set(groups), key=lambda g: sha256(('opening-1123-v1/' + str(g)).encode()).digest())
    chosen = set(ordered[::2])
    return np.array([g in chosen for g in groups])


def score(results, truth, groups):
    directions, pooled_delta, pooled_groups = {}, [], []
    for name, segments in results.items():
        ids = np.array([i for part in segments for i in part['ids']])
        check(len(set(ids)) == len(ids), 'Evaluation row appears twice')
        y, g = truth[ids], groups[ids]
        correct = {n: np.array([v for part in segments for v in part[n]]) == y
                   for n in ('original', 'proposed', 'deep')}
        new, old, deep = (correct[k] for k in ('proposed', 'original', 'deep'))
        delta = new.astype(int) - old
        directions[name] = dict(rows=len(ids), groups=len(set(g)), correct={n: int(x.sum()) for n, x in correct.items()},
            gained=int((new & ~old).sum()), lost=int((~new & old).sum()), delta=paired(delta, g),
            versus_fixed_deep=paired(new.astype(int) - deep, g),
            choices=[{k: v for k, v in part.items() if k not in ('ids','original','proposed','deep')} for part in segments])
        check(not set(pooled_groups).intersection(g), 'Evaluation group appears in both directions')
        pooled_delta.extend(delta)
        pooled_groups.extend(g)
    joint = paired(np.asarray(pooled_delta), np.asarray(pooled_groups))
    passed = all(r['gained'] > r['lost'] for r in directions.values()) and joint['net_pp'] >= .2
    return dict(directions=directions, pooled=joint, total_rows=len(pooled_delta), gate=passed)


def run(inputs, prior, recent, out):
    out.mkdir(parents=True, exist_ok=True)
    packet = inputs / 'anonymous.zip'
    expected = read(prior / 'protocol.json')['identities']['packets']['anonymous.zip']
    check(ab_run.sha(packet) == expected == read(recent / 'protocol.json')['packets']['anonymous.zip'], 'Wrong input')
    frozen_save(out / 'protocol.json', dict(plan=ab_run.sha(ROOT / 'docs/batches/B_cross_seed_portfolio_1123_zh.md'),
        runner=ab_run.sha(Path(__file__)), training_packet=expected, model_api_cap=0, fresh_fit_cap=0))
    task = load_development_task(packet)
    groups = sample_groups(task, 'shared_text')
    folds = {'8123': read_fold(prior / 'deep-e5', groups), '18643': read_fold(recent / 'autonomous', groups)}
    sets = {k: set(f['groups']) for k, f in folds.items()}
    cross, within = {}, {}
    for s, t in (('8123', '18643'), ('18643', '8123')):
        source, target = folds[s], folds[t]
        held = np.flatnonzero([g not in sets[s] for g in target['groups']])
        cross[t] = [predictions(source, np.arange(len(source['groups'])), target, held)]
        first = alternating_groups(target['groups'][held])
        a, b = held[first], held[~first]
        within[t] = [predictions(target, a, target, b), predictions(target, b, target, a)]
    frozen_save(out / 'frozen_predictions.json', dict(cross=cross, within=within))
    frozen_save(out / 'prediction_seal.json', dict(predictions=ab_run.sha(out / 'frozen_predictions.json'),
        folds={k: dict(row_ids=f['split'], models=f['models'], names=f['names']) for k, f in folds.items()}))
    comparative = {name: score(parts, task.y_train, groups) for name, parts in (('cross_seed', cross), ('within_fold', within))}
    result = dict(**comparative, qualifies_for_native_run=all(a['gate'] for a in comparative.values()),
        actual_new_fits=0, model_api_requests=0, official_validation=False, formal_test=False,
        prediction_seal=ab_run.sha(out / 'prediction_seal.json'))
    frozen_save(out / 'results.json', result)
    frozen_save(out / 'completion.json', dict(status='complete', results=ab_run.sha(out / 'results.json'),
        frozen_models_reused=8, actual_new_fits=0, model_api_requests=0))
    print({name: [(k, v['correct']) for k, v in a['directions'].items()] for name, a in comparative.items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('workspace', 'inputs', 'prior', 'recent', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    args = p.parse_args()
    with patch.dict(os.environ, offline_environment(args.workspace.resolve())):
        run(*(getattr(args, k).resolve() for k in ('inputs', 'prior', 'recent', 'output')))
