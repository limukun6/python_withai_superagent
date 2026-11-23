"""Optional, train-history-only estimates for preserving finalization time."""
import math
import time

from .confirmation import shortlist


class PlanningTimeExhausted(RuntimeError):
    """No request was started because the assigned planning window expired."""


class RuntimeBudget:
    def __init__(self,start,total,trial_timeout,repeats,top_k,reuse_primary,refit,mode='shared',clock=None):
        if mode not in ('shared','reserve'):raise ValueError('Unknown time policy')
        if any(not math.isfinite(v) or v<=0 for v in (total,trial_timeout)):
            raise ValueError('Time budgets must be positive and finite')
        self.start,self.total,self.trial_timeout=start,total,trial_timeout
        self.repeats,self.top_k,self.reuse_primary,self.refit=repeats,top_k,reuse_primary,refit
        self.mode=mode;self.clock=clock or time.monotonic
        self.margin=min(2.,.05*total) if mode=='reserve' else 0.
        self.planner_observations=[]

    def remaining(self):return max(0.,self.total-(self.clock()-self.start))

    def fit_estimate(self,row):
        value=row.get('worker_wall_seconds',row.get('training_seconds',1.))
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:value=1.
        # The worker duration includes Python startup and loading/serializing artifacts.
        return 1.5*max(min(1.,self.trial_timeout),float(value))

    def tail(self,history,prospective=False):
        selected=shortlist(history,self.top_k)
        costs=[self.fit_estimate(row) for row in selected]
        next_cost=max(costs,default=1.5*min(1.,self.trial_timeout))
        if prospective:
            # The unmeasured candidate may enter the shortlist. Keep the most costly
            # possible members, rather than assuming the new candidate will lose.
            costs=sorted([*costs,next_cost],reverse=True)[:self.top_k]
        extra=self.repeats-int(self.reuse_primary) if self.repeats>1 else 0
        confirmation=extra*sum(costs)
        refit=2*max(costs,default=0.) if self.refit else 0.
        return {'confirmation_reserve_seconds':confirmation,'refit_reserve_seconds':refit,
            'output_margin_seconds':self.margin,'next_fit_estimate_seconds':next_cost,
            'total_reserve_seconds':confirmation+refit+self.margin}

    def search_window(self,history,needs_planner=False):
        remaining=self.remaining()
        if self.mode=='shared':
            return {'allowed':remaining>0,'fit_timeout_seconds':min(self.trial_timeout,remaining),
                'planner_deadline':None,'remaining_seconds':remaining,'reason':'shared_remaining_time'}
        tail=self.tail(history,prospective=True)
        # Before any successful observation, allocate an initial attempt but still
        # preserve proportional room for its repeats and a potentially larger refit.
        if not shortlist(history,self.top_k):
            extra=self.repeats-int(self.reuse_primary) if self.repeats>1 else 0
            fit_cap=max(0.,(remaining-self.margin)/(1+extra+(2 if self.refit else 0)))
            return {**tail,'allowed':fit_cap>=min(1.,self.trial_timeout),'fit_timeout_seconds':min(self.trial_timeout,fit_cap),
                'planner_deadline':self.clock()+fit_cap/2 if needs_planner else None,
                'remaining_seconds':remaining,'reason':'initial_observation_with_proportional_reserve'}
        free=remaining-tail['total_reserve_seconds']
        planner_estimate=max(2.,1.5*max(self.planner_observations,default=5.)) if needs_planner else 0.
        allowed=free>=tail['next_fit_estimate_seconds']+planner_estimate
        return {**tail,'allowed':allowed,'fit_timeout_seconds':min(self.trial_timeout,max(0.,free)),
            'planner_deadline':self.clock()+max(0.,free-tail['next_fit_estimate_seconds']) if needs_planner else None,
            'planner_estimate_seconds':planner_estimate,'remaining_seconds':remaining,
            'reason':'room_for_search_and_finalization' if allowed else 'reserve_time_for_confirmation_and_refit'}

    def observe_planner(self,elapsed):
        if math.isfinite(elapsed) and elapsed>=0:self.planner_observations.append(elapsed)

    def confirmation_limit(self,history):
        if self.mode=='shared':return self.total
        tail=self.tail(history)
        return max(0.,self.total-tail['refit_reserve_seconds']-self.margin)

    def description(self):
        return {'mode':self.mode,'duration_basis':'observed worker wall time, including process startup',
            'fit_safety_multiplier':1.5,'full_refit_multiplier':2.,'output_margin_seconds':self.margin,
            'hard_total_wall_clock_guarantee':False,
            'limitations':'Unseen configurations and larger refits may take longer; HTTP timeout is per connect/read phase, not a hard total deadline'}
