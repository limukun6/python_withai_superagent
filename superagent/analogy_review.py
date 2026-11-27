"""Allow a retained preference model to recruit a supervised nearest-example adviser."""
import os

import joblib
import numpy as np

from .ab_run import sha
from .preference_analogy import WeightedProbabilityVote, derived_candidate
from .models import SoftVoteModel
from .experiment_policy import identity
from .finalist_race import paired_error
from .splits import sample_groups
from .task_constraints import permitted


SHARES=(0.,.25,.5,1.)


def can_review(task,metric,folds,static,limit):
    from .autonomous_run import declared_semantics
    if (task.kind!='pair_text' or declared_semantics(task)!='preference' or metric.name!='accuracy'
            or set(metric.labels)!={0,1} or limit<16 or len(folds)<3):
        return False
    # Only reserve opportunities when the ordinary menu already offers usable encoders and
    # each confirmation provides different held rows under the task's requested split.
    first=[set(folds[name][1]) for name in ('primary','confirmation_1','confirmation_2') if name in folds]
    if len(first)!=3 or any(first[i]&first[j] for i in range(3) for j in range(i)):
        return False
    groups=sample_groups(task,'shared_text')
    if any(set(groups[fit]) & set(groups[held]) for fit,held,_ in folds.values()):
        return False
    return any(cap.get('resource')=='local_frozen_encoder_cpu' and cap['params'].get('encoder')
               for cap in static.values())


def blend_counts(anchor,neighbour,truth,shares,*,classes=(0,1)):
    if anchor['row_key']!=neighbour['row_key'] or not np.array_equal(anchor['labels'],truth) or not np.array_equal(neighbour['labels'],truth):
        raise ValueError('Blend requires two aligned, previously withheld sets of predictions')
    matrices=[np.asarray(obs['probabilities']) for obs in (anchor,neighbour)]
    if any(a.ndim!=2 or a.shape!=(len(truth),2) or not np.isfinite(a).all() for a in matrices):
        raise ValueError('Blend has no calibrated binary probability evidence')
    labels=np.asarray(classes)
    if labels.shape!=(2,) or set(labels)!={0,1}:
        raise ValueError('Blend requires the ordered binary preference labels')
    return [int(np.sum(labels[np.argmax((1-w)*matrices[0]+w*matrices[1],axis=1)]==truth)) for w in shares]


def nominate(anchor,neighbour,truth,*,classes=(0,1)):
    scores=blend_counts(anchor,neighbour,truth,SHARES,classes=classes)
    position=1+int(np.argmax(scores[1:]))
    weight=SHARES[position]
    base=np.asarray(anchor['probabilities'])
    alt=(1-weight)*base+weight*np.asarray(neighbour['probabilities'])
    labels=np.asarray(classes)
    before,after=labels[base.argmax(axis=1)]==truth,labels[alt.argmax(axis=1)]==truth
    changed=dict(rows=len(truth),gained=int((after&~before).sum()),lost=int((before&~after).sum()))
    gain=scores[position]-scores[0]
    uncertainty=paired_error(changed)
    if gain>0:
        reason='primary_improvement'
    elif changed['gained']>0 and -gain<=len(truth)*uncertainty:
        reason='paired_near_tie'
    else:
        reason='outside_sampling_uncertainty'
        weight=0.
    return dict(weight=weight,reason=reason,scores=scores,primary_gain=gain,
                paired=changed,paired_standard_error=uncertainty)


def promoted(primary_gain,confirmations,*,primary_rows=None):
    if len(confirmations)!=2:
        return False
    gain=sum(v['gain'] for v in confirmations)
    rows=sum(v['rows'] for v in confirmations)
    if primary_gain>0:
        return all(v['gain']>=0 for v in confirmations) and gain>=.003*rows
    return (primary_rows is not None and all(v['gain']>0 for v in confirmations)
            and gain+primary_gain>=.003*(rows+primary_rows))


def incumbent_parts(members,model):
    if len(members)==1:
        return [model],[1.]
    if not isinstance(model,SoftVoteModel) or len(set(members))!=len(members) or len(model.models)!=len(members):
        raise ValueError('Cannot link ensemble constituents to their frozen fitted models')
    shares=getattr(model,'weights',np.full(len(members),1./len(members)))
    if len(shares)!=len(members) or min(shares)<0 or not np.isclose(sum(shares),1.):
        raise ValueError('Invalid incumbent weights')
    return model.models,list(map(float,shares))


def representation_source(members,models,primary,caps):
    options=[]
    for m,model in zip(members,models):
        record=primary.get(m)
        if record is None or record.get('probabilities') is None:
            continue
        candidate=derived_candidate(model)
        if candidate:
            options.append((m,candidate,float(record['score']),identity(caps[m]['params'])))
    if not options:
        return None,[]
    ordered=sorted(options,key=lambda x:(-x[2],x[3],identity(x[1].params)))
    return ordered[0], [dict(member=n,adviser=c.family,measured_primary=s,execution_key=k) for n,c,s,k in ordered]


def incumbent_evidence(records,shares,metric):
    first=records[0]
    if any(v['row_key']!=first['row_key'] or not np.array_equal(v['labels'],first['labels']) for v in records):
        raise ValueError('Mixed incumbent has unmatched validation rows')
    probabilities=np.average([r['probabilities'] for r in records],axis=0,weights=shares)
    predictions=np.asarray(metric.labels)[probabilities.argmax(axis=1)]
    return dict(first,probabilities=probabilities,predictions=predictions,score=float(np.mean(predictions==first['labels'])))


def review(task,metric,stop,incumbent,executor,directory,log,timeout,observed,payloads):
    from .autonomous_run import planner_observation
    anchor_names=stop['members']
    spent=executor.accounting()['attempted_classifier_fits']
    report=dict(incumbent_members=anchor_names,incumbent_model=incumbent['model_sha256'],before_fits=spent,
                permitted_weights=list(SHARES),allocations=[],all_decisions_training_only=True)
    result=incumbent,stop

    def record(status,**values):
        report.update(status=status,**values)
        report['extra_fits']=executor.accounting()['attempted_classifier_fits']-spent
        log.write_json('analogy_review.json',report)

    def measure(name,fold,reserve):
        if name in observed[fold]:
            return observed[fold][name]
        source=payloads[fold]
        artifact=directory/f'analogy_{fold}_{name}.joblib'
        _,facts=executor.worker(source['path'],name,artifact,timeout,reserve=reserve)
        evidence=planner_observation(joblib.load(artifact),facts,source,metric)
        observed[fold][name]=evidence
        return evidence

    try:
        path=directory/incumbent['model_file']
        if sha(path)!=incumbent['model_sha256']:
            raise ValueError('Frozen incumbent identity changed')
        anchor_model=joblib.load(path)['model']
        anchor_models,anchor_shares=incumbent_parts(anchor_names,anchor_model)
        elected,options=representation_source(anchor_names,anchor_models,observed['primary'],executor.caps)
        if elected is None:
            record('retained_without_compatible_representation')
            return result
        source_name,candidate,_,_=elected
        report.update(incumbent_weights=anchor_shares,source_member=source_name,source_candidates=options)
        name=candidate.family
        cap=dict(id=name,kind='pair_text',params=candidate.params,route='preference_analogy',estimator='preference_analogy',
                 resource='local_frozen_encoder_cpu',origin='b_port',classifier_fits=1,refit_classifier_fits=1,
                 supports_probabilities=True)
        if not permitted(cap,executor.restrictions):
            record('retained_task_restrictions')
            return result
        wanted=('primary','confirmation_1','confirmation_2')
        def anchor_cost(fold):
            return sum(executor.caps[m]['classifier_fits'] for m in anchor_names if m not in observed[fold])
        needed=1+sum(anchor_cost(f)+1 for f in wanted)
        if needed>executor.limit-spent:
            record('retained_insufficient_budget',required=needed)
            return result
        executor.caps[name]=cap
        record('evaluating',candidate=name,derived_configuration=candidate.params,remaining_fits=executor.limit-spent)
        primary=incumbent_evidence([observed['primary'][m] for m in anchor_names],anchor_shares,metric)
        first=measure(name,'primary',needed-1)
        proposal=nominate(primary,first,payloads['primary']['labels'],classes=metric.labels)
        share,gain,scores=proposal['weight'],proposal['primary_gain'],proposal['scores']
        measured_scores={'primary':scores[SHARES.index(share)]/len(first['labels'])}
        report.update(primary=dict(row_key=primary['row_key'],rows=len(first['labels']),scores=scores),locked_weight=share,
                      transparent_nomination=proposal)
        record('weight_locked',already_observed_folds={f:sorted(observed[f]) for f in wanted})
        if share==0.:
            record('retained_primary')
            return result
        for j,fold in enumerate(wanted[1:]):
            pending=list(wanted[2+j:])
            reserve=1+sum(anchor_cost(f)+1 for f in pending)
            anchors=[]
            for k,member in enumerate(anchor_names):
                later=sum(executor.caps[m]['classifier_fits'] for m in anchor_names[k+1:] if m not in observed[fold])
                anchors.append(measure(member,fold,reserve+1+later))
            base=incumbent_evidence(anchors,anchor_shares,metric)
            near=measure(name,fold,reserve)
            truth=payloads[fold]['labels']
            base_count,vote_count=blend_counts(base,near,truth,[0.,share],classes=metric.labels)
            report['allocations'].append(dict(fold=fold,row_key=base['row_key'],rows=len(truth),
                                              incumbent=base_count,candidate=vote_count,gain=vote_count-base_count))
            measured_scores[fold]=vote_count/len(truth)
            record('confirming')
        if not promoted(gain,report['allocations'],primary_rows=len(first['labels'])):
            record('retained_confirmation')
            return result
        model,_=executor.worker(directory/'refit_data.joblib',name,directory/'analogy_full.joblib',timeout)
        models=[*anchor_models,model] if share<1 else [model]
        members=[*anchor_names,name] if share<1 else [name]
        weights=[*(float(w)*(1-share) for w in anchor_shares),share] if share<1 else [1.]
        final=WeightedProbabilityVote(task.kind,models,metric.labels,weights) if share<1 else model
        family='feedback_weighted_vote' if len(models)>1 else name
        destination=directory/'analogy_selected_model.joblib'
        temp=directory/'analogy_selected_model.tmp'
        joblib.dump(dict(kind=task.kind,candidate=family,model=final),temp)
        probe=[task.X_train[i] for i in np.linspace(0,len(task.y_train)-1,min(12,len(task.y_train)),dtype=int)]
        recovered=joblib.load(temp)['model']
        if not np.array_equal(recovered.predict(probe),final.predict(probe)):
            raise ValueError('Combined model failed local reload')
        os.replace(temp,destination)
        score=sum(measured_scores[f]*len(payloads[f]['labels']) for f in wanted)/sum(len(payloads[f]['labels']) for f in wanted)
        proposal=dict(action='stop',members=members,split=None,utility=score,reason='An independently confirmed neighbour weight improved the declared metric.',
                      evidence=dict(scores=measured_scores,weights=weights,selection_evidence='analogy_review.json'))
        selected=dict(incumbent,selected_experiment='stop_'+identity(dict(members=members,weights=weights))[:12],
                      family=family,ensemble_components=members if len(members)>1 else [],single_candidate=members[0],
                      selection_metric='matched primary and confirmation training folds',selection_score=score,fold_scores=measured_scores,
                      weights=weights,incumbent_model_sha256=incumbent['model_sha256'],selection_evidence='analogy_review.json',
                      model_file=destination.name,model_sha256=sha(destination))
        log.write_json('analogy_incumbent.json',incumbent)
        log.write_json('analogy_selection.json',dict(choice=proposal,scope='training feedback only'))
        record('accepted',final_members=members,weights=weights,model_sha256=selected['model_sha256'])
        # Predictions follow this last pointer. The intact incumbent is available throughout review.
        log.write_json('final_selection.json',selected)
        log.write_json('selection.json',dict(choice=proposal,scope='frozen from matched training feedback before external evaluation'))
        return selected,proposal
    except (ValueError,RuntimeError,OSError,KeyError,AssertionError) as error:
        record('retained_after_failed_opportunity',error=f'{type(error).__name__}: {error}'[:400])
        if executor.accounting()['unknown_failed_attempts']:
            raise
        return result
