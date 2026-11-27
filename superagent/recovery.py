"""One bounded recovery journal across intake and method execution.

The controller supplies context; it never selects a target or ranks algorithms.
Only structural diagnostics belong here, never private row/pixel values.
"""
from copy import deepcopy
import hashlib
import json
import time


class RecoveryController:
    def __init__(self, log=None, *, client=None, fit_limit=None, deadline=None):
        self.log, self.client, self.deadline = log, client, deadline
        self.fit_limit = fit_limit
        self.state = dict(version='recovery-state-v1',stage='task_adaptation',status='running',
            goal=None,observed_evidence=[],unresolved_fields=[],failed_actions=[],actions=[],
            installed_capabilities=['inspect_header','inspect_array','inspect_image',
                                    'repair_adapter','revise_interpretation','alternate_method'],
            available_artifacts=[],attempted_classifier_fits=0)

    def context(self):
        state = deepcopy(self.state)
        client = self.client
        state['remaining_requests'] = max(0,client.max_requests-client.request_count) if client and client.max_requests is not None else 0
        state['remaining_fits'] = max(0,self.fit_limit-state['attempted_classifier_fits']) if self.fit_limit is not None else None
        state['remaining_seconds'] = max(0,self.deadline-time.monotonic()) if self.deadline else None
        return state

    def save(self):
        if self.log: self.log.write_json('recovery_state.json',self.context())

    def advance(self, stage, *, goal=None, accounting=None, artifacts=None, status='running'):
        self.state.update(stage=stage,status=status)
        if goal is not None:self.state['goal']=deepcopy(goal)
        if accounting is not None:self.state['attempted_classifier_fits']=accounting['attempted_classifier_fits']
        if artifacts is not None:self.state['available_artifacts']=list(artifacts)
        if stage not in ('task_adaptation','contract_probe'):self.state['unresolved_fields']=[]
        self.save()

    def failure(self, stage, diagnostic, *, action=None):
        # Proposals may contain misspelled fields but no row values. Keep their
        # identity rather than copying arbitrary caller content into the journal.
        signature=hashlib.sha256(json.dumps(dict(stage=stage,action=action,diagnostic=diagnostic),
            sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
        repeated=any(r['signature']==signature and r['evidence_count']==len(self.state['observed_evidence'])
                     for r in self.state['failed_actions'])
        record=dict(stage=stage,diagnostic=deepcopy(diagnostic),action_sha256=hashlib.sha256(
            json.dumps(action,sort_keys=True,default=str).encode()).hexdigest(),signature=signature,
            evidence_count=len(self.state['observed_evidence']),repeated_without_new_evidence=repeated)
        self.state['failed_actions'].append(record)
        self.state['stage']=stage
        self.state['unresolved_fields']=list(diagnostic.get('uncertainty_fields',[]))
        if self.log:self.log.event('recovery_failure',**record)
        self.save()
        return repeated

    def observed(self, action, observation):
        self.state['actions'].append(dict(action=action,outcome='new_structure_observation'))
        self.state['observed_evidence'].append(deepcopy(observation))
        self.save()

    def accepted(self, action, *, stage):
        self.state['actions'].append(dict(action=action,outcome='verified',stage=stage))
        self.advance(stage)

    def exhausted(self, reason):
        self.state.update(status='unresolved',stop_reason=reason)
        self.save()
