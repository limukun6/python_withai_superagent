"""Audition categorical representations against a delivered model on matched training evidence."""
from importlib.util import find_spec
from itertools import combinations
import json
import re
import time

import joblib
import numpy as np

from .ab_run import sha
from .cell_encoders import PREFIX, SCRATCH, capability, resources
from .experiment_policy import identity
from .models import SoftVoteModel, _clean_categories, predict_with_model, probability_with_model
from .splits import take
from .task_constraints import permitted


def score(metric, labels, matrix):
    classes = np.asarray(metric.labels)
    hard = classes[matrix.argmax(1)]
    truth = labels[:, None] == classes
    return dict(score=float(metric.evaluate(labels, hard, matrix, classes)['score']),
                brier=float(np.mean(np.sum((matrix-truth)**2, axis=1))))


def contest(metric, truth, incumbent, trials):
    rows = []
    def record(names, share, prediction):
        rows.append(dict(encoders=list(names), share=share, **score(metric, truth, prediction)))
    record((), 0., incumbent)
    for count in (1, 2):
        for names in combinations(sorted(trials), count):
            side = np.mean([trials[n] for n in names], axis=0)
            for share in (.125, .25, .5, 1.):
                record(names, share, (1-share)*incumbent + share*side)
    def rank(entry):
        complexity = len(entry['encoders']) + int(entry['share'] < 1.)
        return (-entry['score'], entry['brier'], complexity, entry['share'], entry['encoders'])
    return sorted(rows, key=rank)


def passes(panels):
    if len(panels) != 2:
        return False
    weights = [f['rows'] for f in panels]
    gains = [f['challenger']['score']-f['incumbent']['score'] for f in panels]
    costs = [f['challenger']['brier']-f['incumbent']['brier'] for f in panels]
    return bool(min(gains) > 0 and np.average(gains, weights=weights) >= .003
                and np.average(costs, weights=weights) <= 0.)


def after_delivery(task, spec, metric, selection, choice, observed, payloads, executor, directory, log, timeout):
    from .autonomous_run import planner_observation
    original = selection['model_sha256']
    report = dict(incumbent=original, scope='training partitions only', screened={}, decisions=[], confirmations=[])

    def retain(reason):
        report.update(action='retain', reason=reason, exported_model=original)
        log.write_json('cell_encoder_review.json', report)
        return selection, choice

    folds = ('primary', 'confirmation_1', 'confirmation_2')
    members = choice['members']
    if (task.kind != 'categorical_tabular' or find_spec('catboost') is None or not 200 <= len(task.y_train) <= 25000 or
            not 2 <= task.X_train.shape[1] <= 64 or
            (spec.get('evaluation_protocol') or {}).get('kind', 'grouped_stratified') != 'grouped_stratified' or
            'linear_only' in spec.get('method_restrictions', ())):
        return retain('No admissible categorical representation review for this task')
    if any(any(n not in observed.get(f, {}) or observed[f][n]['probabilities'] is None for n in members) for f in folds):
        return retain('The incumbent needs class-aligned probabilities on three training panels')

    def measure(name, fold, path, destination, payload_hash, seed=None, cached=False):
        if sha(path) != payload_hash:
            raise ValueError('Changed training partition')
        if cached:
            bundle = joblib.load(destination)
            facts = bundle['metrics']
        else:
            _, facts = executor.worker(path, name, destination, timeout)
            bundle = joblib.load(destination)
        if seed is not None and facts['resolved_configuration'].get('training_random_seed') != seed:
            raise ValueError('Confirmation used another random seed')
        matched = dict(payloads[fold], sha256=payload_hash)
        return planner_observation(bundle, facts, matched, metric)['probabilities']

    try:
        if sha(directory/selection['model_file']) != original:
            raise ValueError('Incumbent identity changed')
        partitions = [json.loads((directory/(f+'_split.json')).read_text())['held'] for f in folds]
        seen = set()
        for f, indices in zip(folds, partitions, strict=True):
            if seen.intersection(indices) or len(set(indices)) != len(payloads[f]['labels']) or len(indices) < 100:
                return retain('Insufficient disjoint training-only confirmation rows')
            seen.update(indices)
            for name in members:
                observation = observed[f][name]
                if observation['row_key'] != payloads[f]['sha256'] or not np.array_equal(observation['labels'], payloads[f]['labels']):
                    raise ValueError('Incumbent feedback is not tied to these ordered training rows')
                if len(members) == 1 and not np.array_equal(np.asarray(metric.labels)[observation['probabilities'].argmax(1)],
                                                            observation['predictions']):
                    return retain('A probability vote would change the incumbent decision rule')
        primary = payloads['primary']
        if sha(primary['path']) != primary['sha256']:
            raise ValueError('Changed fitting packet')
        x = _clean_categories(joblib.load(primary['path'])['X_train'])
        levels = [np.unique(col) for col in x.T]
        if sum(map(len, levels)) > 2048:
            return retain('The training vocabulary exceeds the cell representation budget')
        words = sum(bool(re.search(r'[^\W\d_]{2,}', str(value))) for value in set(np.concatenate(levels)))
        assets = resources()
        report['available'] = {k: v['name'] for k, v in assets.items()}
        report['partitions'] = {f: dict(hash=payloads[f]['sha256'], rows=len(payloads[f]['labels'])) for f in folds}
        seed_info = selection.get('seed_replication', {})
        seeds, repeated = seed_info.get('seeds', []), seed_info.get('members_repeated', [])
        # Preserve room to reconstruct the deployed incumbent and confirm/refit even a two-encoder nominee.
        reserve = 2*len(seeds)*len(repeated) + 6
        eligible = [k for k in sorted(assets, key=lambda k: (assets[k]['bytes'], k))
                    if (k == SCRATCH or words >= 3) and permitted(capability(k), executor.restrictions)]
        def free():
            return executor.limit - executor.accounting()['attempted_classifier_fits']
        if not eligible or free() <= reserve or executor.accounting()['unknown_failed_attempts']:
            return retain('Insufficient supported options or remaining accounted fits')
        report['eligible'] = eligible
        executor.caps.update({PREFIX+k: capability(k) for k in eligible})
        original_primary = np.mean([observed['primary'][name]['probabilities'] for name in members], axis=0)
        catalog = {}
        for key in eligible:
            if free() <= reserve or time.monotonic() - executor.start >= executor.wall or executor.accounting()['unknown_failed_attempts']:
                break
            try:
                dest = directory/('cell_primary_'+key+'.joblib')
                data = measure(PREFIX+key, 'primary', primary['path'], dest, primary['sha256'])
                catalog[key] = data
                report['screened'][key] = score(metric, primary['labels'], data)
            except RuntimeError as error:
                report['screened'][key] = dict(error=str(error)[-500:])
        ranking = contest(metric, primary['labels'], original_primary, catalog)
        winner = ranking[0]
        report.update(primary_incumbent=score(metric, primary['labels'], original_primary),
                      primary_incumbent_seeds='original fit of each selected member', nominee=winner, alternatives=len(ranking))
        report['decisions'].append('Lock signature and mixing weight before either confirmation score')
        log.write_json('cell_encoder_review.json', dict(report, action='locked'))
        keys, share = winner['encoders'], winner['share']
        if not keys:
            return retain('Primary feedback preferred the incumbent')
        for fold in folds[1:]:
            packet = payloads[fold]
            per_member = [[observed[fold][n]['probabilities']] for n in members]
            for seed in seeds:
                # Repeat the exact deployed estimator, including its seeds, on the same held-out rows.
                path = directory/f'cell_{fold}_seed_{seed}.joblib'
                joblib.dump(dict(joblib.load(packet['path']), supervised_random_seed=seed), path)
                digest = sha(path)
                for i, name in enumerate(members):
                    if name not in repeated:
                        continue
                    pilot = directory/f'seed_pilot_{i}.joblib'
                    reusable = (fold == seed_info.get('pilot_split') and seed == seeds[0] and pilot.is_file()
                                and (directory/'seed_pilot_data.joblib').is_file() and sha(directory/'seed_pilot_data.joblib') == digest)
                    dest = pilot if reusable else directory/f'cell_{fold}_reference_{i}_{seed}.joblib'
                    per_member[i].append(measure(name, fold, path, dest, digest, seed, reusable))
            prior = np.mean([np.mean(rows, axis=0) for rows in per_member], axis=0)
            proposals = [measure(PREFIX+k, fold, packet['path'], directory/('cell_'+fold+'_'+k+'.joblib'), packet['sha256'])
                         for k in keys]
            blended = (1-share)*prior + share*np.mean(proposals, axis=0)
            report['confirmations'].append(dict(fold=fold, rows=len(prior), incumbent=score(metric, packet['labels'], prior),
                                                challenger=score(metric, packet['labels'], blended)))
            log.write_json('cell_encoder_review.json', dict(report, action='confirm'))
        if not passes(report['confirmations']):
            return retain('The locked proposal did not pass both paired feedback checks')
        models, weights, identifiers = [], [], []
        if share < 1:
            models.append(joblib.load(directory/selection['model_file'])['model'])
            weights.append(1-share)
            identifiers.append(original)
        for k in keys:
            model, _ = executor.worker(directory/'refit_data.joblib', PREFIX+k, directory/('cell_full_'+k+'.joblib'), timeout)
            models.append(model)
            weights.append(share/len(keys))
            identifiers.append(PREFIX+k)
        model = models[0] if len(models)==1 else SoftVoteModel(task.kind, models, np.asarray(metric.labels), tuple(weights))
        family = identifiers[0] if len(models)==1 else 'cell_encoder_probability_vote'
        path = directory/'cell_selected_model.joblib'
        joblib.dump(dict(kind=task.kind, candidate=family, model=model), path)
        probe = take(task.X_train, np.linspace(0, len(task.y_train)-1, min(16, len(task.y_train)), dtype=int))
        restored = joblib.load(path)['model']
        np.testing.assert_array_equal(predict_with_model(task.kind, model, probe), predict_with_model(task.kind, restored, probe))
        a, b = [probability_with_model(task.kind, m, probe) for m in (model, restored)]
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1], rtol=1e-7, atol=1e-8)
        scores = {p['fold']:p['challenger']['score'] for p in report['confirmations']}
        recipe = dict(identities=identifiers, weights=weights, incumbent_members=members if share<1 else [])
        selected = dict(selection, family=family, selected_experiment='cells_'+identity(recipe)[:12], model_file=path.name,
                        ensemble_components=identifiers if len(identifiers)>1 else [], single_candidate=identifiers[0],
                        model_sha256=sha(path), cell_encoder_recipe=recipe, selection_metric='two matched training confirmations',
                        selection_score=float(np.average(list(scores.values()), weights=[p['rows'] for p in report['confirmations']])),
                        fold_scores=scores)
        selected.pop('seed_replication', None)
        log.write_json('cell_encoder_review.json', dict(report, action='adopt', recipe=recipe, exported_model=selected['model_sha256']))
        log.write_json('final_selection.json', selected)
        return selected, dict(choice, members=(members if share<1 else [])+[PREFIX+k for k in keys])
    except Exception as error:
        return retain(f'Optional extension could not complete: {type(error).__name__}: {error}')
