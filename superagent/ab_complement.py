"""Choose a two-member preference vote only after matched row-level confirmation."""
from copy import deepcopy

import numpy as np

from .ab_scheduler import CONFIRMATIONS, PRIMARY, _action, next_action, selection_evidence
from .metrics import MetricSpec
from .models import SoftVoteModel, probability_with_model
from .score_evidence import checked_score_record


MIN_GAIN=.002
OPENING=4


def eligible(spec, capabilities, metric, limit):
    return (spec['kind']=='pair_text' and spec['input_schema'].get('pair_semantics')=='preference'
            and metric.name=='accuracy' and set(metric.labels)=={0,1} and limit>=8
            and len(capabilities)>=OPENING and all(c['supports_probabilities'] and
            c['classifier_fits']==c['refit_classifier_fits']==1 for c in capabilities.values()))


def canonical_probabilities(model, rows):
    ordered=[(a,b) if a<=b else (b,a) for a,b in rows]
    classes,p=probability_with_model('pair_text',model,ordered)
    if set(classes.tolist())!={0,1}:
        raise ValueError('Preference probability evidence requires classes 0 and 1')
    p=p[:,classes.tolist().index(1)]
    first=np.asarray([a<b for a,b in rows],dtype=bool)
    equal=np.asarray([a==b for a,b in rows],dtype=bool)
    positive=np.where(equal,.5,np.where(first,p,1-p))
    return np.array([0,1]),np.column_stack([1-positive,positive])


class PreferenceProbabilityVote(SoftVoteModel):
    def predict_proba(self, rows):
        return np.mean([canonical_probabilities(model,rows)[1] for model in self.models],axis=0)

    def predict(self, rows):
        values=self.predict_proba(rows)[:,1]
        return ((values>.5)|((values==.5)&np.asarray([a<b for a,b in rows],dtype=bool))).astype(int)


def split_columns(state, trusted, split, metric):
    local=trusted[split]
    records={}
    for item in state['prediction_evidence']:
        if item['split_id']!=split:
            continue
        name=item['candidate_id']
        matrix,classes=checked_score_record(item.get('score_record'),item,metric,name,local['labels'])
        records[name]=(np.asarray(item['predictions']),matrix[:,classes.index(1)],item['trial'])
    return np.asarray(local['labels']),np.asarray(local['tie_label_one'],dtype=bool),records


def average_hits(a,b,first,y):
    p=(a[1]+b[1])/2
    return int(((((p>.5)|((p==.5)&first)).astype(int))==y).sum())


def review(state, trusted):
    # The existing verifier binds the hard predictions, metric, payload, trusted labels and original row order.
    selection_evidence(state,trusted_partitions=trusted)
    if len(state['prediction_evidence'])!=len(state['history']):
        raise ValueError('A completed opening observation lacks prediction evidence')
    metric=MetricSpec.from_payload({'metric_spec':state['task']['metric']})
    y,first,opening=split_columns(state,trusted,PRIMARY,metric)
    if len(opening)!=OPENING:
        raise ValueError('Four matched opening observations required')
    hits={n:int((v[0]==y).sum()) for n,v in opening.items()}
    anchor=max(hits,key=hits.get)
    mixed={n:average_hits(opening[anchor],v,first,y) for n,v in opening.items() if n!=anchor}
    partner=max(mixed,key=mixed.get)
    basis=dict(anchor=anchor,partner=None,members=[anchor],opening_correct=hits,opening_combinations=mixed,
               primary_rows=len(y),confirmation_rows=0,evidence_trials=[v[2] for v in opening.values()],
               primary_single_accuracy=hits[anchor]/len(y),portfolio_score=hits[anchor]/len(y),promoted=False)
    if (mixed[partner]-hits[anchor])/len(y)<MIN_GAIN:
        return basis
    y2,first2,second=split_columns(state,trusted,CONFIRMATIONS[0],metric)
    basis['partner']=partner
    basis['pending']=[n for n in (anchor,partner) if n not in second]
    basis['evidence_trials'].extend(v[2] for v in second.values())
    if basis['pending']:
        return basis
    n2=len(y2)
    single={n:int((second[n][0]==y2).sum()) for n in (anchor,partner)}
    vote=average_hits(second[anchor],second[partner],first2,y2)
    pooled=mixed[partner]+vote-max(hits[n]+single[n] for n in (anchor,partner))
    passed=(mixed[partner]>max(hits[n] for n in (anchor,partner)) and vote>max(single.values())
            and pooled/(len(y)+n2)>=MIN_GAIN)
    basis.update(confirmation_correct=single,confirmation_combination=vote,confirmation_rows=n2,
                 pooled_gain=pooled,promoted=passed)
    if passed:
        basis.update(members=[anchor,partner],portfolio_score=(mixed[partner]+vote)/(len(y)+n2))
    return basis


def action(state, capabilities, trusted):
    if len(state['history'])<OPENING:
        projected=deepcopy(state)
        projected['registered_confirmation_splits']=[]
        proposed=next_action(projected,capabilities=capabilities,trusted_partitions={PRIMARY:trusted[PRIMARY]})
        if proposed['action']!='explore':
            raise ValueError('Unable to finish the four eligible opening trials: '+proposed['reason'])
        return proposed
    evidence=review(state,trusted)
    refs=evidence['evidence_trials']
    pending=evidence.get('pending',[])
    if pending:
        if len(pending)+2>state['budget']['remaining_fit_allowance']:
            raise ValueError('Matched confirmation and final refits exceed remaining allowance')
        return _action('confirm','Confirm leading model and the partner that corrected the most primary errors together.',
                       refs,pending[0],CONFIRMATIONS[0],1)
    return _action('stop','Matched evidence supports the registered vote.' if evidence['promoted'] else
                   'Retain the primary leader after the registered comparison.',refs)
