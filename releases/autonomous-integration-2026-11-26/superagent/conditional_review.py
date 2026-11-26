"""Ask for one decision-changing measurement after a model is already deliverable."""
import math

import joblib
import numpy as np

from .experiment_policy import identity


PROMPT = (
    'Decide whether measuring a real disagreement is worthwhile. The incumbent is already saved. '
    'All scores shown are measured on training folds. Each option lost one comparison and won another. '
    'The physical cost includes any missing reference measurements and a possible deployment refit. '
    'Past exploration counts are only context. Return one JSON object. To keep the incumbent use '
    '{"decision":"keep","reason":"..."} and no measurement is performed. To authorize a probe use '
    '{"decision":"measure","option":0,"cited_folds":["primary","confirmation_1"],'
    '"if_pass":"refit_challenger","if_fail":"deliver_incumbent","minimum_delta":null,'
    '"reason":"...","opportunity_cost":"..."}. Choose the actual zero-based option and cite its two fold names. '
    'The pass rule requires a strictly positive new delta and at least the displayed weighted mean gain over '
    'all three folds. minimum_delta may be null or a stricter gain on the third fold, in score units (not percent). '
    'Explain which uncertainty could alter the decision and what the measurement displaces. Prefer keeping '
    'the incumbent when both outcomes would warrant the same action. No raw examples or evaluation outcomes are available.'
)
MEAN_GAIN = .003
PASS, FAIL = 'refit_challenger', 'deliver_incumbent'


def group_observation(bank, members, metric):
    if any(name not in bank for name in members):
        return None
    rows = [bank[name] for name in members]
    if len({r['row_key'] for r in rows}) != 1:
        raise ValueError('Reference predictions were measured on different rows')
    if len(rows)==1:
        return rows[0]
    if any(r['probabilities'] is None for r in rows):
        return None
    matrix = np.mean([r['probabilities'] for r in rows],axis=0)
    prediction = np.asarray(metric.labels)[matrix.argmax(axis=1)]
    value = metric.evaluate(rows[0]['labels'],prediction,matrix,metric.labels)['score']
    return dict(rows[0],predictions=prediction,probabilities=matrix,score=value)


def compare(fold, base, other):
    if base['row_key']!=other['row_key'] or not np.array_equal(base['labels'],other['labels']):
        raise ValueError('Challenger and incumbent evidence do not match')
    before = base['predictions']==base['labels']
    after = other['predictions']==base['labels']
    return dict(fold=fold,rows=len(before),incumbent=base['score'],challenger=other['score'],
                delta=other['score']-base['score'],gained=int(sum(after & ~before)),lost=int(sum(before & ~after)))


def feasible_probes(incumbent, observed, folds, caps, metric, remaining, *, scheduled=None):
    names = ['primary', *sorted(f for f in folds if f.startswith('confirmation_'))]
    if len(names)<3 or any(f not in observed for f in names[:3]):
        return [], 'third_training_fold_unavailable'
    first, second, third = names[:3]
    if scheduled is not None and (scheduled['action']!='confirm' or scheduled['split']!=third):
        return [], 'next_experiment_is_not_a_third_fold_confirmation'
    held = [set(map(int,folds[f][1])) for f in names[:3]]
    if any(not ids for ids in held) or any(held[i]&held[j] for i,j in ((0,1),(0,2),(1,2))):
        return [], 'evaluation_rows_not_disjoint'
    base = [group_observation(observed[f],incumbent,metric) for f in names[:3]]
    if any(o is None for o in base[:2]):
        return [], 'incumbent_lacks_two_matched_scores'
    options = []
    conflict = False
    for name in sorted(observed[first].keys() & observed[second].keys()):
        if list(incumbent)==[name] or name not in caps:
            continue
        prior = [compare(f,reference,observed[f][name]) for f,reference in zip((first,second),base)]
        if min(r['delta'] for r in prior)>=0 or max(r['delta'] for r in prior)<=0:
            continue
        conflict = True
        missing = [n for n in dict.fromkeys([*incumbent,name]) if n not in observed[third]]
        if not missing or any(n not in caps for n in missing):
            continue
        if scheduled is not None and missing!=scheduled['members']:
            continue
        measurement = sum(caps[n]['classifier_fits'] for n in missing)
        refit = caps[name]['refit_classifier_fits'] if scheduled is None else scheduled['evidence']['reserved_refits']
        if measurement+refit>remaining:
            continue
        n = len(held[2])
        required = max(0.,(MEAN_GAIN*(sum(r['rows'] for r in prior)+n)-sum(r['rows']*r['delta'] for r in prior))/n)
        maximum = (observed[third][name]['score'] if name in observed[third] else 1.) - (base[2]['score'] if base[2] is not None else 0.)
        if maximum<=0 or required>maximum+1e-12:
            continue
        options.append(dict(challenger=name,estimator=caps[name]['estimator'],representation=caps[name]['representation'],
            comparisons=prior,third_fold=third,third_rows=n,missing=missing,measurement_fits=measurement,
            refit_fits=refit,required_delta=required,maximum_possible_delta=maximum))
        if scheduled is not None:
            options[-1].update(incumbent_known_third_score=base[2]['score'] if base[2] is not None else None,
                               challenger_known_third_score=observed[third].get(name,{}).get('score'))
    return options, ('eligible' if options else 'no_affordable_new_decisive_measurement' if conflict else 'no_rank_reversal')


def validate(answer, packet, *, pass_action=PASS, fail_action=FAIL):
    if not isinstance(answer,dict) or not isinstance(answer.get('reason'),str) or not answer['reason'].strip():
        raise ValueError('A review requires a reason')
    if answer.get('decision')=='keep':
        return dict(decision='keep',reason=answer['reason'])
    i=answer.get('option')
    if answer.get('decision')!='measure' or type(i) is not int or not 0<=i<len(packet['options']):
        raise ValueError('The selected probe is not available')
    option=packet['options'][i]
    if (answer.get('if_pass'),answer.get('if_fail'))!=(pass_action,fail_action):
        raise ValueError('The two outcomes must lead to distinct admissible decisions')
    refs=answer.get('cited_folds')
    if not isinstance(refs,list) or sorted(refs)!=sorted(r['fold'] for r in option['comparisons']):
        raise ValueError('The conflicting training folds are not both cited')
    cost=answer.get('opportunity_cost')
    if not isinstance(cost,str) or not cost.strip() or 'minimum_delta' not in answer:
        raise ValueError('Missing cost or threshold')
    threshold=answer['minimum_delta']
    if threshold is not None and (type(threshold) not in (float,int) or not math.isfinite(threshold)
            or not 0<=threshold<=option['maximum_possible_delta']+1e-12):
        raise ValueError('Unattainable or invalid decision threshold')
    return dict(decision='measure',option=i,cited_folds=refs,if_pass=pass_action,if_fail=fail_action,
                minimum_delta=max(option['required_delta'],threshold or 0.),reason=answer['reason'],opportunity_cost=cost)


def evaluate_probe(probe, latest, minimum_delta):
    comparisons=[*probe['comparisons'],latest]
    weighted=float(np.average([r['delta'] for r in comparisons],weights=[r['rows'] for r in comparisons]))
    passed=latest['delta']>0 and latest['delta']+1e-12>=minimum_delta and weighted+1e-12>=MEAN_GAIN
    return weighted, bool(passed)


def review_delivery(task,spec,metric,selection,choice,executor,directory,log,timeout,observed,payloads,folds,flash,memory):
    from .autonomous_run import freeze_model, planner_observation
    remaining=executor.limit-executor.accounting()['attempted_classifier_fits']
    record=dict(incumbent=choice['members'],incumbent_sha256=selection['model_sha256'],remaining_fits=remaining,
                executed=FAIL,status='skipped')
    try:
        options,reason=feasible_probes(choice['members'],observed,folds,executor.caps,metric,remaining)
        record.update(reason=reason,feasible_probes=len(options))
        if options:
            packet=dict(metric=metric.name,incumbent=choice['members'],options=options,remaining_fits=remaining,
                        minimum_weighted_gain=MEAN_GAIN,historical_explorations=memory)
            record['input_digest']=identity(packet)
            log.write_json('conditional_review_input.json',packet)
            if flash is None:
                raise ValueError('The conditional reviewer is unavailable')
            answer=flash.ask(dict(instruction=PROMPT,evidence=packet),log,'feedback_reflection',cost_ceiling=.12)
            plan=validate(answer,packet)
            record.update(plan=plan,status='declined' if plan['decision']=='keep' else 'plan_frozen')
            log.write_json('conditional_review.json',record)
            if plan['decision']=='measure':
                probe=options[plan['option']]
                name,fold=probe['challenger'],probe['third_fold']
                bank=dict(observed[fold])
                data=payloads[fold]
                reserve=probe['refit_fits']+probe['measurement_fits']
                for index,member in enumerate(probe['missing']):
                    reserve-=executor.caps[member]['classifier_fits']
                    artifact=directory/f'conditional_fold_{index}.joblib'
                    _,facts=executor.worker(data['path'],member,artifact,timeout,reserve=reserve)
                    bank[member]=planner_observation(joblib.load(artifact),facts,data,metric)
                reference=group_observation(bank,choice['members'],metric)
                if reference is None:
                    raise ValueError('The saved ensemble cannot be evaluated on this fold')
                latest=compare(fold,reference,bank[name])
                comparisons=[*probe['comparisons'],latest]
                weighted,passed=evaluate_probe(probe,latest,plan['minimum_delta'])
                record.update(third=latest,weighted_delta=weighted,passed=passed,status='measured')
                log.write_json('conditional_review.json',record)
                if passed:
                    scores={r['fold']:r['challenger'] for r in comparisons}
                    proposal=dict(action='stop',members=[name],split=None,utility=float(np.mean(list(scores.values()))),
                                  evidence={'scores':scores},reason='Precommitted conditional review passed on matched training folds.')
                    new=freeze_model(task,spec,metric,proposal,executor,directory,log,timeout,challenger=True,
                                     challenger_prefix='conditional_')
                    new.update(selection_evidence='conditional_review.json',incumbent_model_sha256=selection['model_sha256'])
                    log.write_json('selection.json',dict(choice=proposal,scope='matched training folds',evidence='conditional_review.json'))
                    log.write_json('final_selection.json',new)
                    selection,choice=new,proposal
                    record.update(executed=PASS,status='adopted')
    except (ValueError,TypeError,KeyError,RuntimeError,OSError,AssertionError) as error:
        record.update(status='retained_after_error',reason=f'{type(error).__name__}: {error}'[:360])
    record['delivered_sha256']=selection['model_sha256']
    log.write_json('conditional_review.json',record)
    if executor.accounting()['unknown_failed_attempts']:
        raise RuntimeError('Unknown fit count during conditional review')
    return selection,choice
