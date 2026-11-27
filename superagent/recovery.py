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
            installed_capabilities=['inspect_header','inspect_array','inspect_image','inspect_batch',
                                    'repair_adapter','revise_interpretation','normalize_task_rules','alternate_method','report_unresolved'],
            available_artifacts=[],attempted_classifier_fits=0)

    def context(self):
        state = deepcopy(self.state)
        # The complete journal stays on disk; only two structural previews reach
        # each stateless model request. Old hashes/diagnostics remain available.
        for record in state['failed_actions'][:-2]:record.pop('action_summary',None)
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
        from .action_context import summarize_action,CHECKS
        signature=hashlib.sha256(json.dumps(dict(stage=stage,action=action,diagnostic=diagnostic),
            sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
        repeated=any(r['signature']==signature and r['evidence_count']==len(self.state['observed_evidence'])
                     for r in self.state['failed_actions'])
        record=dict(stage=stage,diagnostic=deepcopy(diagnostic),action_sha256=hashlib.sha256(
            json.dumps(action,sort_keys=True,default=str).encode()).hexdigest(),signature=signature,
            evidence_count=len(self.state['observed_evidence']),repeated_without_new_evidence=repeated)
        record.update(action_summary=summarize_action(action),
                      expected_check=CHECKS.get(stage,'Verify the declared operation using installed capabilities.'))
        fields=sorted(diagnostic.get('uncertainty_fields',[]))
        evidence_sha=self.evidence_identity()
        record.update(unresolved_fields=fields,evidence_sha256=evidence_sha,
            repeated_unresolved_without_new_evidence=bool(fields) and any(
                r.get('unresolved_fields')==fields and r.get('evidence_sha256')==evidence_sha
                for r in self.state['failed_actions']))
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

    def evidence_identity(self):
        keys=('member','source','codec','collection','delimiter','fields','format','arrays','shape','dtype','width','height','mode')
        facts={json.dumps({k:o[k] for k in keys if k in o},sort_keys=True,ensure_ascii=False)
               for o in self.state['observed_evidence']}
        return hashlib.sha256(json.dumps(sorted(facts),ensure_ascii=False).encode()).hexdigest()

    def report_unresolved(self, report):
        self.state['unresolved_report']=deepcopy(report)
        self.state['unresolved_fields']=[q['field'] for q in report['questions']]
        self.state['actions'].append(dict(action='report_unresolved',outcome='model_report_not_verified_completion'))
        self.exhausted('Model reported that supplied evidence cannot define the remaining task fields')

    def accepted(self, action, *, stage):
        self.state['actions'].append(dict(action=action,outcome='verified',stage=stage))
        self.advance(stage)

    def exhausted(self, reason):
        self.state.update(status='unresolved',stop_reason=reason)
        self.save()
