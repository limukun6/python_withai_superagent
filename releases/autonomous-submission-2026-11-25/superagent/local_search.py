"""Search rules see candidate metadata and observed trials only."""
from copy import deepcopy
import math
import random
import statistics
from .metrics import score_of,completed

POLICIES=('random','balanced_blind','greedy_routes','protected_routes','interleaved_routes')


class LocalSearchPolicy:
    def __init__(self,name,catalog,seed):
        if name not in POLICIES:raise ValueError('Unknown local policy')
        self.name=name
        self.catalog=deepcopy(catalog)
        allowed={'id','route','params'}
        if any(set(row)!=allowed for row in catalog):
            raise ValueError('Catalog may contain metadata only, never unobserved scores')
        self.rng=random.Random(seed)
        self.coordinates={}
        for route in sorted({r['route'] for r in catalog}):
            members=[r for r in catalog if r['route']==route]
            numeric=sorted({k for r in members for k,v in r['params'].items()
                            if type(v) in (int,float) and v>0})
            ranges={k:(min(math.log(float(r['params'][k])) for r in members),
                       max(math.log(float(r['params'][k])) for r in members)) for k in numeric}
            for row in members:
                self.coordinates[row['id']]=tuple(
                    (math.log(float(row['params'][k]))-lo)/(hi-lo) if hi>lo else .5
                    for k,(lo,hi) in ranges.items())

    def _tie(self,items):return self.rng.choice(sorted(items))

    def add_candidates(self,items):
        if set(r['id'] for r in items)&set(r['id'] for r in self.catalog):raise ValueError('Candidate IDs cannot be registered twice')
        rebuilt=LocalSearchPolicy(self.name,[*self.catalog,*items],0)
        self.catalog,self.coordinates=rebuilt.catalog,rebuilt.coordinates
        # Keep this policy's existing RNG state and observed-route history.

    def _central(self,items):
        distances={r['id']:sum((x-.5)**2 for x in self.coordinates[r['id']]) for r in items}
        minimum=min(distances.values())
        return self._tie([key for key,value in distances.items() if abs(value-minimum)<1e-12])

    def choose(self,history):
        # A fresh snapshot prevents a consumer from mutating logged history.
        history=deepcopy(history)
        tried={r['candidate'] for r in history}
        available=[r for r in self.catalog if r['id'] not in tried]
        if not available:raise ValueError('Candidate pool exhausted')
        if not history:return self.catalog[0]['id'],'shared_anchor'
        if self.name=='random':return self._tie([r['id'] for r in available]),'uniform_without_replacement'
        routes=sorted({r['route'] for r in available})
        observed={route:[r for r in history if r['route']==route] for route in routes}
        unseen=[route for route in routes if not observed[route]]
        interleaved=self.name=='interleaved_routes'
        usable_seen=False if self.name=='balanced_blind' else any(any(completed(r) for r in observed[route]) for route in routes)
        explore_now=not interleaved or len(history)%2==1 or not usable_seen
        if unseen and explore_now:
            route=self._tie(unseen)
            return self._central([r for r in available if r['route']==route]),(
                'interleaved_explore_route' if interleaved else 'cover_untried_route')
        if self.name=='balanced_blind':
            minimum=min(len(observed[route]) for route in routes)
            route=self._tie([route for route in routes if len(observed[route])==minimum])
            return self._central([r for r in available if r['route']==route]),'balance_without_scores'
        priorities={}
        if interleaved:
            routes=[route for route in routes if observed[route]]
        for route in routes:
            successes=[score_of(r) for r in observed[route] if completed(r)]
            best=max(successes) if successes else -1.
            bonus=.02*math.sqrt(math.log(1+len(history))/len(observed[route])) if self.name in ('protected_routes','interleaved_routes') else 0.
            priorities[route]=best+bonus
        maximum=max(priorities.values())
        route=self._tie([route for route in routes if abs(priorities[route]-maximum)<1e-12])
        members=[r for r in available if r['route']==route]
        successful=[r for r in observed[route] if completed(r)]
        if not successful:return self._central(members),'route_retry_after_failed_configuration'
        incumbent=min(successful,key=lambda r:(-score_of(r),r['candidate']))
        center=self.coordinates[incumbent['candidate']]
        distances={r['id']:sum((a-b)**2 for a,b in zip(self.coordinates[r['id']],center)) for r in members}
        minimum=min(distances.values())
        candidate=self._tie([key for key,value in distances.items() if abs(value-minimum)<1e-12])
        reason=('interleaved_feedback_priority' if interleaved else
                'best_observed_route' if self.name=='greedy_routes' else 'best_plus_exploration_priority')
        return candidate,reason


def confirmation_shortlist(history,top_k=3):
    return [r['candidate'] for r in sorted((r for r in history if completed(r)),
            key=lambda r:(-score_of(r),r['candidate']))[:top_k]]


def select_confirmed(shortlist,scores,seeds):
    eligible=[]
    for candidate in shortlist:
        values=[scores.get((candidate,seed)) for seed in seeds]
        if all(value is not None for value in values):
            eligible.append((candidate,statistics.mean(values)))
    return min(eligible,key=lambda row:(-row[1],row[0])) if eligible else (None,None)
