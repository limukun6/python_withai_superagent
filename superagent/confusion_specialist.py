"""Design and measure a two-class refinement of a frozen classifier from internal prediction errors."""
from dataclasses import dataclass
import math

import joblib
import numpy as np

from .conditional_review import MEAN_GAIN, compare
from .experiment_policy import descriptor, fit_prior, identity, similarity
from .splits import take


def top_pairs(probabilities):
    order = np.argsort(-probabilities, axis=1, kind='stable')[:, :2]
    return np.sort(order, axis=1)


def aligned(kind, model, inputs, labels):
    from .models import probability_with_model
    classes, raw = probability_with_model(kind, model, inputs)
    classes = np.asarray(classes).tolist()
    raw = np.asarray(raw, dtype=float)
    if (len(classes) != len(labels) or set(classes) != set(labels)
            or raw.shape != (len(inputs), len(labels)) or not np.isfinite(raw).all()
            or np.any(raw < 0) or not np.allclose(raw.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError('Refinement needs probabilities aligned to the fitted class domain')
    return raw[:, [classes.index(k) for k in labels]]


def apply_rule(kind, expert, inputs, scores, predictions, labels, pair):
    """Only the two leading class scores may change; keep their total probability mass."""
    out = scores.copy()
    values = None if predictions is None else np.asarray(predictions).copy()
    mask = np.all(top_pairs(scores) == pair, axis=1)
    rows = np.flatnonzero(mask)
    if len(rows):
        names = [labels[k] for k in pair]
        conditional = aligned(kind, expert, take(inputs, rows), names)
        columns = np.ix_(rows, pair)
        mass = scores[columns].sum(axis=1, keepdims=True)
        out[columns] = conditional * mass
        if values is not None:
            values[rows] = np.asarray(names)[np.argmax(conditional, axis=1)]
    return values, out, mask


def apply_hard_rule(kind, expert, inputs, predictions, labels, pair):
    from .models import predict_with_model
    values = np.asarray(predictions).copy()
    names = [labels[k] for k in pair]
    mask = np.isin(values,names)
    rows = np.flatnonzero(mask)
    if len(rows):
        result = predict_with_model(kind,expert,take(inputs,rows))
        if result.shape != (len(rows),) or not np.isin(result,names).all():
            raise ValueError('Specialist predicted outside the fitted class pair')
        values[rows] = result
    return values,mask


@dataclass
class ClassPairModel:
    kind: str
    base: object
    expert: object
    classes_: np.ndarray
    pair: tuple
    gate: str = 'top_two'

    def verified_legacy_id(self, selection):
        """Recover the unique ID of the first producer's signed, fully described model."""
        detail = selection.get('refinement', {})
        members = selection.get('ensemble_components')
        if (self.gate != 'top_two' or selection.get('family') != 'class_pair_refinement' or not selection.get('model_sha256')
                or not selection.get('task_spec_sha256') or not isinstance(members,list) or not members
                or not isinstance(detail,dict) or not isinstance(detail.get('method'),str)):
            return None
        labels = self.classes_.tolist()
        if detail.get('classes') != [labels[k] for k in self.pair]:
            return None
        route = 'pair_refinement_' + identity([members,list(self.pair),detail['method'],labels])[:16]
        return route if route == selection.get('selected_experiment') else None

    def predict(self, inputs):
        from .models import predict_with_model
        pred = predict_with_model(self.kind, self.base, inputs)
        if self.gate == 'predicted_member':
            return apply_hard_rule(self.kind,self.expert,inputs,pred,self.classes_,self.pair)[0]
        p = aligned(self.kind, self.base, inputs, self.classes_.tolist())
        return apply_rule(self.kind, self.expert, inputs, p, pred, self.classes_, self.pair)[0]

    def predict_proba(self, inputs):
        if self.gate == 'predicted_member':
            raise ValueError('This frozen hard-label model has no probability output')
        p = aligned(self.kind, self.base, inputs, self.classes_.tolist())
        return apply_rule(self.kind, self.expert, inputs, p, None, self.classes_, self.pair)[1]


def propose_pair(row, metric):
    """Rank recurring, observable pairs by their maximum possible primary-metric improvement."""
    if len(metric.labels) <= 2 or (row['probabilities'] is None and metric.requires_scores):
        return None
    labels = np.asarray(metric.labels)
    scores, truth, pred = row['probabilities'], row['labels'], row['predictions']
    if scores is None:
        index = {label:i for i,label in enumerate(metric.labels)}
        possibilities = sorted({tuple(sorted((index[a],index[b]))) for a,b in zip(truth,pred) if a!=b})
        pairs = None
    else:
        pairs = top_pairs(scores)
        possibilities = np.unique(pairs,axis=0)
    eligible = []
    for option in possibilities:
        pair = np.asarray(option)
        mask = np.isin(pred,labels[pair]) if pairs is None else np.all(pairs == pair, axis=1)
        errors = mask & np.isin(truth, labels[pair]) & (truth != pred)
        count = int(errors.sum())
        if count < 2 or mask.sum() < max(5, math.ceil(math.sqrt(len(truth))/2)):
            continue
        optimistic = pred.copy()
        optimistic[errors] = truth[errors]
        possible = scores
        if metric.requires_scores:
            possible = scores.copy()
            ids = np.flatnonzero(errors)
            # Order the two scores in favor of the known class only to estimate opportunity.
            # These oracle values never reach the fitted specialist or its acceptance test.
            values = np.sort(scores[np.ix_(ids,pair)], axis=1)
            possible[np.ix_(ids,pair)] = np.where((truth[ids] == labels[pair[0]])[:,None], values[:,::-1], values)
        gain = metric.evaluate(truth, optimistic, possible, list(metric.labels))['score'] - row['score']
        if gain > 0:
            eligible.append(dict(pair=pair.tolist(), gate='predicted_member' if pairs is None else 'top_two',
                                 recoverable_errors=count, primary_rows=int(mask.sum()),
                                 opportunity_bound=float(gain)))
    return min(eligible, key=lambda r: (-r['opportunity_bound'], -r['recoverable_errors'], r['pair']), default=None)


def choose_expert(caps, measured, pair, metric, profile, blocked, *, hard=False):
    options = {name:cap for name,cap in caps.items() if name not in blocked and (hard or cap.get('supports_probabilities'))
               and cap['classifier_fits'] == cap['refit_classifier_fits'] == 1}
    direct = {}
    names = np.asarray(metric.labels)[pair]
    for name in sorted(options.keys() & measured.keys()):
        row = measured[name]
        prob = row['probabilities']
        if prob is None and not hard:
            continue
        candidate_labels = row['predictions'] if prob is None else names[np.argmax(prob[:,pair],axis=1)]
        correct = [int(np.count_nonzero((row['labels'] == label) & (candidate_labels == label))) for label in names]
        counts = [int(np.count_nonzero(row['labels'] == label)) for label in names]
        direct[name] = float(np.mean([(hit+1)/(n+2) for hit,n in zip(correct,counts)]))
    if not direct:
        return None
    desc = {n:descriptor(c) for n,c in options.items()}
    sample_size = len(next(iter(measured.values()))['labels'])
    quality = {}
    for name,cap in options.items():
        if name in direct:
            estimate = direct[name]
        else:
            weights = {n:similarity(desc[name],desc[n])**2 for n in direct}
            estimate = (0.5 + sum(w*direct[n] for n,w in weights.items())) / (1+sum(weights.values()))
        quality[name] = estimate + fit_prior(cap,profile)/math.sqrt(sample_size)
    best = min(quality, key=lambda n:(-quality[n], identity(n)))
    return dict(method=best, eligible_methods=len(options), observed_methods=len(direct),
                direct_pair_skill=direct.get(best), forecast_score=quality[best])


def design_experiment(opening, choice, incumbent, bases, names, observed, profile, executor, timeout, blocked, local):
    """Return None to preserve the ordinary probe, or return the terminal decision and its immutable model."""
    from .ab_run import sha
    directory = opening.log.run_dir
    task,metric = opening.task,opening.metric
    remaining = executor.limit - executor.accounting()['attempted_classifier_fits']
    design = dict(version=1, status='not_applicable', evidence_scope='primary training holdout only')
    def abstain(reason):
        design['reason'] = reason
        opening.save()
        return None
    opening.record['designed_method'] = design
    if 'linear_only' in task.spec.get('method_restrictions', ()):
        return abstain('Piecewise decisions are outside the declared linear-only methods')
    from .output_recipe import needs_scores
    if remaining < 3:
        return abstain('Three single fits cannot be reserved')
    hard = any(bases[f]['probabilities'] is None for f in names)
    if hard and (metric.requires_scores or needs_scores(task.spec.get('output_recipe'))):
        return abstain('The scoring or submission contract requires probabilities')
    primary = dict(bases['primary'],probabilities=None) if hard else bases['primary']
    pair_info = propose_pair(primary, metric)
    if pair_info is None:
        return abstain('No recurring top-two confusion with measurable opportunity')
    pair = pair_info['pair']
    expert_info = choose_expert(executor.caps, observed['primary'], pair, metric, profile, blocked, hard=hard)
    if not expert_info:
        return abstain('No affordable permitted probabilistic learner with supporting evidence')
    expert = expert_info['method']
    labels = np.asarray(metric.labels)[pair]
    def train_indices(ids):
        return ids[np.isin(task.y_train[ids], labels)]
    primary_fit = train_indices(opening.folds['primary'][0])
    if min(np.count_nonzero(task.y_train[primary_fit] == k) for k in labels) < 5:
        return abstain('Too few primary fitting examples of one proposed class')
    signature = [choice['members'], pair, expert, list(metric.labels)]
    if hard:
        signature.append('predicted_member')
    route = 'pair_refinement_' + identity(signature)[:16]
    design.update(**pair_info, **expert_info, classes=labels.tolist(), method_id=route, status='frozen_before_probe',
                  incumbent_model_sha256=incumbent['model_sha256'])
    opening.record.update(local_experiment=local, tested_candidate=route, changed_experiment=True,
                          confirmation=names, comparisons=[], result='retained')
    opening.save()
    spent = executor.accounting()['attempted_classifier_fits']

    def fit(ids, tag, reserve):
        subset = train_indices(np.asarray(ids))
        if min(np.count_nonzero(task.y_train[subset] == k) for k in labels) < 2:
            raise ValueError('A confirmation training fold lacks two usable classes')
        payload = directory / ('pair_training_'+tag+'.joblib')
        joblib.dump(dict(kind=task.kind, mode='refit', X_train=take(task.X_train, subset), y_train=task.y_train[subset],
                         X_search=take(task.X_train, subset[:1]), metric_spec=metric.record(),
                         pair_semantics=task.spec.get('input_schema',{}).get('pair_semantics')), payload)
        model,facts = executor.worker(payload, expert, directory/('pair_expert_'+tag+'.joblib'), timeout, reserve=reserve)
        if facts['training_rows'] != len(subset):
            raise ValueError('Specialist was fitted on unexpected rows')
        design.setdefault('physical_training_rows',{})[tag] = len(subset)
        return model

    try:
        for i,fold in enumerate(names):
            fit_ids,held,_ = opening.folds[fold]
            model = fit(fit_ids,fold,2-i)
            before = bases[fold]
            features = take(task.X_train,held)
            if hard:
                pred,mask = apply_hard_rule(task.kind,model,features,before['predictions'],metric.labels,pair)
                prob = None
            else:
                pred,prob,mask = apply_rule(task.kind,model,features,before['probabilities'],
                                            before['predictions'],metric.labels,pair)
            score = metric.evaluate(before['labels'],pred,prob,list(metric.labels))['score']
            row = compare(fold,before,dict(before,predictions=pred,probabilities=prob,score=score))
            row.update(routed_rows=int(sum(mask)), outside_changes=int(sum((pred!=before['predictions'])&~mask)))
            opening.record['comparisons'].append(row)
            if row['delta'] <= 1e-12:
                design.update(status='rejected',reason='No positive whole-fold gain on '+fold)
                return incumbent,choice
        rows = opening.record['comparisons']
        if np.mean([r['delta'] for r in rows]) < MEAN_GAIN:
            design.update(status='rejected',reason='Matched gain below the registered adoption threshold')
            return incumbent,choice
        source = directory/incumbent['model_file']
        if sha(source) != incumbent['model_sha256']:
            raise ValueError('Incumbent model changed after delivery')
        base = joblib.load(source)['model']
        expert_model = fit(np.arange(len(task.y_train)), 'refit', 0)
        wrapped = ClassPairModel(task.kind,base,expert_model,np.asarray(metric.labels),tuple(pair),gate=pair_info['gate'])
        destination = directory/'designed_model.joblib'
        temporary = directory/'designed_model.tmp'
        joblib.dump(dict(kind=task.kind,candidate=route,model=wrapped),temporary)
        # Cross-process export will use the same pickle and immutable selection pointer.
        verify = take(task.X_train,np.linspace(0,len(task.y_train)-1,min(10,len(task.y_train)),dtype=int))
        restored = joblib.load(temporary)['model']
        if not np.array_equal(wrapped.predict(verify), restored.predict(verify)):
            raise ValueError('Restored specialist changed predictions')
        temporary.replace(destination)
        scores = {r['fold']:r['challenger'] for r in rows}
        value = float(np.mean(list(scores.values())))
        updated = dict(incumbent,selected_experiment=route,family=route,composition='class_pair_refinement',model_file=destination.name,
                       model_sha256=sha(destination),single_candidate=None,ensemble_components=list(choice['members']),
                       refinement=dict(method=expert,classes=labels.tolist(),gate=pair_info['gate'],refit_rows=design['physical_training_rows']['refit']),
                       incumbent_model_sha256=incumbent['model_sha256'],selection_evidence='task_opening.json',
                       fold_scores=scores,selection_score=value)
        stop = dict(action='stop',members=[route],split=None,utility=value,evidence=dict(scores=scores),
                    reason='The specialized model improved the task metric on both matched training folds.')
        opening.log.write_json('opening_final_selection.json',updated)
        opening.log.write_json('selection.json',dict(choice=stop,scope='two matched training folds',evidence='task_opening.json'))
        opening.log.write_json('final_selection.json',updated)
        opening.record.update(result='adopted',winning_members=[route])
        design['status']='adopted'
        return updated,stop
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        design.update(status='failed',reason=str(error)[:300])
        if executor.accounting()['unknown_failed_attempts']:
            raise
        return incumbent,choice
    finally:
        opening.record['additional_fits']=executor.accounting()['attempted_classifier_fits']-spent
        opening.save()
