"""Versioned boundary records for task, decision, execution and memory."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Protocol

from .metrics import MetricSpec,score_of,metric_of

VERSION=1


def snapshot(value):return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))


@dataclass(frozen=True)
class TaskSpec:
    kind: str
    metric: MetricSpec
    input_schema: dict
    goal_resolution: dict | None = None

    @classmethod
    def from_task(cls,task):
        allowed={'categorical_tabular','numeric_tabular','mixed_tabular','text_tabular','single_text','pair_text','image'}
        if task.kind not in allowed:raise ValueError('Unsupported task kind')
        from .task_goal import resolve_goal
        metric=MetricSpec.for_task(task)
        return cls(task.kind,metric,snapshot(task.spec.get('input_schema',{})),snapshot(resolve_goal(task,metric)))

    def record(self):return {'contract_version':VERSION,'objective':'classification','kind':self.kind,
        'metric':self.metric.record(),'input_schema':snapshot(self.input_schema),
        'selection_data':'training-derived partitions only','formal_test_available_to_agent':False,
        'goal_resolution':snapshot(self.goal_resolution)}


@dataclass(frozen=True)
class ExperimentProposal:
    trial: int
    candidate: str
    params: dict
    reason: str
    hypothesis: str | None
    evidence_trials: tuple
    expected_change: dict | None
    route_plan: dict | None
    raw_proposal: dict | None

    @classmethod
    def from_decision(cls,trial,configuration,reason,proposal,history):
        if type(trial) is not int or trial!=len(history)+1:raise ValueError('Proposal must be the next trial')
        if [r['trial'] for r in history]!=list(range(1,trial)):raise ValueError('History must be ordered and contiguous')
        if not isinstance(configuration,dict) or set(configuration)!={'id','params'}:
            raise ValueError('Proposal configuration needs id and params')
        if not isinstance(configuration['id'],str) or not isinstance(configuration['params'],dict):
            raise ValueError('Malformed configuration')
        if configuration['id'] in {r['candidate'] for r in history}:raise ValueError('Cannot repeat a tried candidate')
        if not isinstance(reason,str) or not reason.strip():raise ValueError('Decision needs a reason')
        raw=snapshot(proposal) if proposal is not None else None
        refs=raw.get('evidence_trials',[]) if raw else []
        if raw and raw.get('candidate')!=configuration['id']:raise ValueError('Proposal and configuration disagree')
        prior={r['trial'] for r in history}
        if not isinstance(refs,list) or any(type(v) is not int for v in refs) or len(set(refs))!=len(refs) or not set(refs)<=prior:
            raise ValueError('Proposal references must be distinct earlier trials')
        expected=raw.get('expected_change') if raw else None
        if raw and raw.get('slice_expectation') is not None:
            from .targeted_feedback import validate_slice_expectation
            validate_slice_expectation(raw['slice_expectation'], history)
        if raw and raw.get('configuration_claim') is not None:
            from .targeted_feedback import candidate_facts
            kind = raw.get('task_kind')
            facts = candidate_facts(kind, configuration) or {}
            claim = raw['configuration_claim']
            if (not isinstance(claim, dict) or set(claim) != {'classifier'}
                    or not isinstance(claim['classifier'], str) or not claim['classifier'].strip()):
                raise ValueError('Configuration claim must name exactly one classifier')
            registered = facts.get('classifier')
            # An unknown fact cannot contradict a claim; the fitted estimator is
            # compared with the claim after the trial instead.
            if registered is not None and claim['classifier'] != registered:
                raise ValueError('Configuration claim disagrees with registered classifier')
        if expected is not None:
            from .hypothesis_agent import METRICS
            if not isinstance(expected,dict) or set(expected)!={'metric','reference_trial','direction','min_delta'}:
                raise ValueError('Malformed or future numerical expectation')
            if type(expected['reference_trial']) is not int or expected['reference_trial'] not in prior or expected['metric'] not in METRICS:
                raise ValueError('Expectation needs a supported training-internal metric and prior trial')
            if expected['direction'] not in ('increase','decrease') or type(expected['min_delta']) not in (int,float) or not math.isfinite(expected['min_delta']) or expected['min_delta']<0:
                raise ValueError('Invalid expectation direction or threshold')
        return cls(trial,configuration['id'],snapshot(configuration['params']),reason.strip(),
            raw.get('hypothesis') if raw else None,tuple(refs),snapshot(expected),
            snapshot(raw.get('route_plan')) if raw else None,raw)

    def record(self):return {'contract_version':VERSION,'trial':self.trial,'candidate':self.candidate,
        'params':snapshot(self.params),'reason':self.reason,'hypothesis':self.hypothesis,
        'evidence_trials':list(self.evidence_trials),'expected_change':snapshot(self.expected_change),
        'route_plan':snapshot(self.route_plan),'source':'agent' if self.raw_proposal else 'controller_or_local_rule',
        'raw_proposal':snapshot(self.raw_proposal)}

    def validate_available(self,available,history):
        configuration=next((c for c in available if c['id']==self.candidate),None)
        if configuration is None or snapshot(configuration['params'])!=self.params:
            raise ValueError('Decision selected unavailable or modified configuration')
        checked=type(self).from_decision(self.trial,configuration,self.reason,self.raw_proposal,history)
        if checked.record()!=self.record():raise ValueError('Typed proposal fields disagree with its original decision')


@dataclass(frozen=True)
class ExperimentResult:
    payload: dict

    @classmethod
    def from_trial(cls,row,proposal,metric_spec=None):
        if row['trial']!=proposal.trial or row['candidate']!=proposal.candidate or row['params']!=proposal.params:
            raise ValueError('Execution result does not match its predeclared proposal')
        if row['status'] not in ('completed','failed','timeout'):raise ValueError('Unknown experiment status')
        score=score_of(row) if row['status']=='completed' else None
        if row['status']=='completed' and score is None:raise ValueError('Completed result needs a finite primary score')
        if metric_spec is not None and row.get('primary_metric','accuracy')!=metric_spec.name:
            raise ValueError('Worker result uses a different task metric')
        identity=metric_spec.check_identity(row.get('metric_spec')) if metric_spec is not None else {'status':'unknown','reason':'task_metric_unavailable'}
        selected={key:row[key] for key in ('primary_metric','search_score','train_score','search_accuracy','train_accuracy',
            'score_gap','generalization_gap','training_seconds','classifier_fits','error','hypothesis_check',
            'estimator','configuration_fact_check','metric_spec') if key in row}
        return cls(snapshot({'contract_version':VERSION,'trial':row['trial'],'candidate':row['candidate'],
            'params':row['params'],'status':row['status'],'measurements':selected,
            'score_available':score is not None,'selection_score':score,'metric_identity':identity}))

    def record(self):return snapshot(self.payload)


@dataclass(frozen=True)
class DecisionState:
    payload: dict

    @classmethod
    def build(cls,profile,history,available,limit,feedback_visible=True,task_contract=None):
        from .hypothesis_agent import PROFILE_FIELDS
        declared=task_contract.metric.name if task_contract is not None else profile.get('primary_metric')
        name=metric_of(history,declared or 'accuracy')
        if declared is not None and name!=declared:raise ValueError('Task metric disagrees with history')
        if task_contract is not None and profile.get('primary_metric',name)!=name:
            raise ValueError('Profile metric disagrees with TaskSpec')
        public_profile={k:snapshot(profile[k]) for k in PROFILE_FIELDS if k in profile}
        public_profile['primary_metric']=name
        allowed=('trial','candidate','params','status','primary_metric','train_score','search_score','train_accuracy',
            'search_accuracy','score_gap','generalization_gap','training_seconds','class_recall','hypothesis_check',
            'paired_error_comparison','explanation_audit','proposal','selected_iterations','classifier_fits','error_slices',
            'estimator','configuration_fact_check','metric_spec','metric_identity')
        clean=[{k:snapshot(r[k]) for k in allowed if k in r} for r in history]
        if not feedback_visible:
            clean=[{k:r[k] for k in ('trial','candidate','params','proposal') if k in r} for r in clean]
        from .hypothesis_memory import build_memory
        payload={'contract_version':VERSION,'next_trial':len(history)+1,'feedback_visible':feedback_visible,
            'profile':public_profile,
            'history':clean,'available_candidates':snapshot(available),
            'trial_budget':{'maximum':limit,'attempted':len(history),'remaining':max(0,limit-len(history))},
            'memory':build_memory(history,available,name) if feedback_visible else None,
            'task_spec':task_contract.record() if task_contract is not None else None}
        return cls(snapshot(payload))

    def record(self):return snapshot(self.payload)


class DecisionPolicy(Protocol):
    def propose(self,state: DecisionState) -> ExperimentProposal: ...


class ExistingPolicyAdapter:
    """Keep existing planners behind the same interface as future architectures."""
    def __init__(self,policy):self.policy=policy

    def propose(self,state):
        record=state.record()
        chosen,reason=self.policy.choose(record['profile'],record['history'],record['available_candidates'])
        configuration=next(c for c in record['available_candidates'] if c['id']==chosen)
        return ExperimentProposal.from_decision(record['next_trial'],configuration,reason,
            getattr(self.policy,'last_proposal',None),record['history'])
