"""One decision interface for raw-history and checked-memory experiments."""
from copy import deepcopy
from .contracts import ExperimentProposal
from .targeted_feedback import candidate_facts


def decision_context(state, checked):
    record = state.record()
    history = deepcopy(record['history'])
    for row in history:
        for field in ('hypothesis_check', 'paired_error_comparison', 'explanation_audit','configuration_fact_check'):
            row.pop(field, None)
        # Wall clock is machine noise, not a property of the candidate. Two arms
        # that ran the same trials differ only in the fourth decimal here, and a
        # measured replay showed that this alone flipped the next action.
        row.pop('training_seconds', None)
    kind = record['task_spec']['kind']
    context = dict(task_spec=record['task_spec'], history=history,
        available_candidates=record['available_candidates'], trial_budget=record['trial_budget'],
        candidate_facts={c['id']:candidate_facts(kind,c) for c in record['available_candidates']},
        instruction='Select exactly one available candidate to maximize the task primary metric. '
        'Return JSON with candidate, task_kind, configuration_claim={classifier: registered classifier}, '
        'reason, review, hypothesis, evidence_trials, expected_change, slice_expectation. '
        'Keep reason, review and hypothesis each under 60 words. '
        'expected_change must be {metric:"search_score",reference_trial: an earlier completed trial, '
        'direction:"increase",min_delta:0.001}. '
        'slice_expectation may be null or {slice_id: an observed slice,reference_trial: its prior trial,min_delta:0.001}. '
        'Slices measure accuracy even if the task primary metric differs. '
        'Use actual observations and configuration facts. Unmet hypotheses remain unverified interpretations. '
        'A failed configuration does not invalidate its entire algorithm family. '
        'Review the latest completed measurement and explain what the next experiment tests. '
        'Actual estimator observations describe what ran; declarations and classifier names are not measurements. '
        'A repeated family is allowed with an evidence-based parameter change. No extra fits or API calls.')
    if checked:
        context['checks'] = [{'trial':r['trial'], 'hypothesis_check':r.get('hypothesis_check'),
                             'configuration_fact_check':r.get('configuration_fact_check'),
                             'explanation_check':{
                                 'warnings':deepcopy((r.get('explanation_audit') or {}).get('warnings',[])),
                                 'freeform_prose_fully_verified':False}}
                             for r in record['history']]
        context['memory'] = record['memory']
    return context


class CheckedHistoryPolicy:
    stop_on_error = True

    def __init__(self, client, logger, checked):
        self.client, self.logger, self.checked = client, logger, checked

    def propose(self, state):
        context = decision_context(state, self.checked)
        answer = self.client.ask(context,self.logger,'checked_history' if self.checked else 'raw_history')
        if answer.get('task_kind') != context['task_spec']['kind']:
            raise ValueError('Planner changed task kind')
        for field in ('reason','review','hypothesis'):
            if not isinstance(answer.get(field), str) or not answer[field].strip():
                raise ValueError(f'Missing planner field: {field}')
        if answer.get('configuration_claim') is None or answer.get('expected_change') is None:
            raise ValueError('Planner omitted a configuration claim or numerical expectation')
        config = next((c for c in context['available_candidates'] if c['id']==answer.get('candidate')),None)
        if config is None:
            raise ValueError('Planner selected an unavailable candidate')
        return ExperimentProposal.from_decision(state.record()['next_trial'],config,
            answer['reason'],answer,state.record()['history'])
