"""One matched, budgeted challenge after a deliverable model has already been saved."""
import joblib
import numpy as np


def review_stop(task, spec, metric, menu, stop, incumbent, executor, directory, log, timeout, observed, payloads):
    from .autonomous_run import freeze_model, planner_observation

    spent = executor.accounting()['attempted_classifier_fits']
    remaining = executor.limit - spent
    confirmations = sorted(fold for fold in stop['evidence']['scores'] if fold.startswith('confirmation_'))
    challenger = None
    for action in menu:
        if action['action'] == 'explore' and action['split'] == 'primary':
            name = action['members'][0]
            cost = executor.caps[name]
            if 2 * cost['classifier_fits'] + cost['refit_classifier_fits'] <= remaining:
                challenger = name
                break
    log.write_json('incumbent_selection.json', incumbent)
    decision = dict(incumbent=stop['members'], incumbent_model=incumbent['model_sha256'], candidate=challenger,
                    fits_at_stop=spent, allowance_left=remaining, comparisons=[], status='pending',
                    final_members=stop['members'], delivered_model=incumbent['model_sha256'])
    if challenger is None or not confirmations:
        decision.update(status='insufficient_room_or_evidence', additional_fits=0)
        log.write_json('stop_review.json', decision)
        return incumbent, stop

    folds = ['primary', confirmations[0]]
    decision['folds'] = folds
    log.write_json('stop_review.json', decision)
    scores = {}
    selected, final = incumbent, stop
    try:
        cap = executor.caps[challenger]
        for index, fold in enumerate(folds):
            data = payloads[fold]
            reserve = cap['refit_classifier_fits'] + (1 - index) * cap['classifier_fits']
            artifact = directory / ('challenge_' + fold + '.joblib')
            _, facts = executor.worker(data['path'], challenger, artifact, timeout, reserve=reserve)
            measured = planner_observation(joblib.load(artifact), facts, data, metric)
            earlier = [observed[fold][m] for m in stop['members']]
            if any(row['row_key'] != measured['row_key'] for row in earlier):
                raise ValueError('Challenger and incumbent used different evaluation rows')
            if len(earlier) == 1:
                prediction = earlier[0]['predictions']
            else:
                probabilities = np.mean([row['probabilities'] for row in earlier], axis=0)
                prediction = np.asarray(metric.labels)[probabilities.argmax(axis=1)]
            base_score = stop['evidence']['scores'][fold]
            observed[fold][challenger] = measured
            before, after = prediction == data['labels'], measured['predictions'] == data['labels']
            scores[fold] = measured['score']
            decision['comparisons'].append(dict(fold=fold, rows=len(before), incumbent=base_score,
                challenger=measured['score'], delta=measured['score']-base_score,
                gained=int(np.sum(~before & after)), lost=int(np.sum(before & ~after))))
            log.write_json('stop_review.json', decision)
        deltas = [row['delta'] for row in decision['comparisons']]
        if min(deltas) > 0 and float(np.mean(deltas)) >= .003:
            proposal = dict(action='stop', members=[challenger], split=None, utility=float(np.mean(list(scores.values()))),
                            evidence={'scores': scores}, reason='Higher declared metric on both matched training folds.')
            # Write the new full-data model to separate files. The incumbent pointer remains valid on failure.
            selected = freeze_model(task, spec, metric, proposal, executor, directory, log, timeout, challenger=True)
            selected.update(selection_evidence='stop_review.json', incumbent_model_sha256=incumbent['model_sha256'])
            final = proposal
            decision.update(status='accepted', final_members=[challenger], delivered_model=selected['model_sha256'])
        else:
            decision['status'] = 'retained'
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as error:
        decision.update(status='incumbent_retained_after_error', reason=f'{type(error).__name__}: {error}'[:500])
        selected, final = incumbent, stop

    decision['additional_fits'] = executor.accounting()['attempted_classifier_fits'] - spent
    log.write_json('stop_review.json', decision)
    if executor.accounting()['unknown_failed_attempts']:
        raise RuntimeError('Unknown physical fit usage during the optional challenge')
    if decision['status'] == 'accepted':
        log.write_json('selection.json', dict(choice=final, scope='matched training folds', evidence='stop_review.json'))
        log.write_json('final_selection.json', selected)
    return selected, final
