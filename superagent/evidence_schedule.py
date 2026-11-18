"""Local, accuracy-only search with an explicit decision between exploration and replication."""
import math
import numpy as np
from .local_search import LocalSearchPolicy
from .evidence_agent import route_of
from .models import search_candidates

MARGIN=.002
SEARCH_LIMIT=10
def catalog_for(kind):
    return [dict(id=c.family,params=c.params,route=route_of(c.family)) for c in search_candidates(kind,'expanded')]

def duel(left,right):
    old,new=np.asarray(left),np.asarray(right)
    if old.dtype!=bool or new.dtype!=bool or old.ndim!=1 or old.shape!=new.shape or not len(old):
        raise ValueError('Expected aligned correctness masks on one split')
    fixed=int((new&~old).sum());lost=int((old&~new).sum())
    return dict(n=len(old),fixed=fixed,lost=lost,delta=(fixed-lost)/len(old))

def schedule(catalog,measure,record,seed=707):
    policy=LocalSearchPolicy('interleaved_routes',catalog,seed)
    history=[];seen={};attempts=0;quiet=0;incumbent=None
    def observe(candidate,split,why):
        nonlocal attempts
        key=(candidate,split)
        if key not in seen:
            if attempts>=SEARCH_LIMIT:raise RuntimeError('Fit limit reached')
            attempts+=1
            record(dict(event='request',number=attempts,candidate=candidate,split=split,reason=why))
            mask=np.asarray(measure(candidate,split))
            if mask.dtype!=bool or mask.ndim!=1 or not len(mask):raise ValueError('Invalid observation')
            seen[key]=mask
            record(dict(event='observed',number=attempts,candidate=candidate,split=split,accuracy=float(mask.mean())))
        return seen[key]
    def can_compare(pair,split):
        cost=sum((name,split) not in seen for name in pair)
        return attempts+cost+max(0,4-len(history))<=SEARCH_LIMIT
    while attempts<SEARCH_LIMIT and len(history)<min(len(catalog),6):
        candidate,reason=policy.choose(history)
        mask=observe(candidate,0,reason)
        history.append(dict(candidate=candidate,route=route_of(candidate),status='completed',search_accuracy=float(mask.mean())))
        if incumbent is None:
            incumbent=candidate;continue
        pair=(incumbent,candidate)
        primary=duel(seen[incumbent,0],mask)
        net=primary['fixed']-primary['lost'];changes=primary['fixed']+primary['lost']
        plausible=changes>0 and (primary['delta']>=MARGIN or abs(net)<=math.sqrt(changes))
        record(dict(event='screen',reference=incumbent,candidate=candidate,plausible=plausible,**primary))
        if plausible and can_compare(pair,1):
            check=duel(*(observe(name,1,'paired_recheck_on_another_partition') for name in pair))
            deltas=[primary['delta'],check['delta']]
            agree=(deltas[0]>0)==(deltas[1]>0)
            if not agree and can_compare(pair,2):
                extra=duel(*(observe(name,2,'resolve_direction_disagreement') for name in pair))
                deltas.append(extra['delta'])
            promote=(sum(value>0 for value in deltas)>=2 and sum(deltas)/len(deltas)>=MARGIN)
            record(dict(event='duel',reference=incumbent,candidate=candidate,deltas=deltas,promoted=promote,
                        first_gain_failed_recheck=deltas[0]>0 and deltas[1]<=0))
            if promote:incumbent=candidate
            quiet=0 if promote or (not agree and len(deltas)==2) else quiet+1
        else:
            if plausible:
                record(dict(event='unresolved',candidate=candidate,reason='not_enough_budget_for_paired_recheck'))
            quiet=0 if plausible else quiet+1
        if len(history)>=4 and quiet>=2:
            reason='two_uncompetitive_or_rejected_challenges';break
        if len(history)>=4 and SEARCH_LIMIT-attempts<2:
            reason='no_room_for_new_candidate_and_recheck';break
    else:reason='search_or_catalog_limit'
    record(dict(event='stop',selected=incumbent,reason=reason,search_fits=attempts,distinct_candidates=len(history)))
    return incumbent
