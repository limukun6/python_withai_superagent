"""Deterministic evidence reports and limited, explicitly heuristic prose checks."""
from copy import deepcopy
import re

from .evidence_agent import latest_result,measured_facts

TIE=re.compile(r'\b(?:only tied|was unchanged|is unchanged|same accuracy as|exactly tied|identical accuracy)\b',re.I)
SIGNIFICANCE=re.compile(r'\b(?:is|was|are|were)\s+(?:not\s+)?(?:statistically\s+)?significant\b|\bno\s+(?:statistically\s+)?significant\s+difference\b',re.I)


def comparison_report(history):
    result=latest_result(history)
    reading=result['reading']
    if not history:text='No measured experiment is available yet.'
    elif reading=='execution_failure':text=f"Trial {result['latest_trial']} failed to execute; it has no measured primary score."
    elif result['latest_search_score'] is None:text=f"Trial {result['latest_trial']} has no usable primary-score measurement."
    elif result['best_prior_search_score'] is None:
        text=f"Trial {result['latest_trial']} internal {result['primary_metric']} score is {100*result['latest_search_score']:.6f}%; no earlier completed score is available."
    else:
        text=(f"Trial {result['latest_trial']} internal {result['primary_metric']} score is {100*result['latest_search_score']:.6f}%; "
              f"best earlier trial {result['best_prior_trial']} is {100*result['best_prior_search_score']:.6f}%; "
              f"difference {100*result['delta_vs_prior_best']:+.6f} percentage points ({reading}).")
    return {**result,'text':text,'statistical_test_performed':False,
            'uncertainty':'These exact score comparisons do not establish statistical significance or a causal explanation.'}


def audit_explanation(proposal,history,candidate=None):
    from .configuration_facts import proposal_change
    from .proposal_consistency import direction_diagnostics
    comparison=comparison_report(history)
    facts={row['id']:row for row in measured_facts(history)}
    cited=proposal.get('assessment',{}).get('fact_ids',[])
    if any(fid not in facts for fid in cited):raise ValueError('Decision explanation cites an unavailable measured fact')
    fields={key:proposal.get(key,'') for key in ('reason','review')}
    fields.update({f'assessment.{key}':proposal.get('assessment',{}).get(key,'') for key in ('interpretation','uncertainty')})
    warnings=[]
    for field,text in fields.items():
        if not isinstance(text,str):continue
        match=TIE.search(text)
        if match and comparison['reading'] in ('improved','lower'):
            warnings.append({'code':'possible_exact_tie_contradiction','field':field,'matched_text':match.group(),
                'comparison_trial':comparison['latest_trial'],'reference_trial':comparison['best_prior_trial'],
                'delta_vs_prior_best':comparison['delta_vs_prior_best'],
                'review_note':'Definite equality wording may contradict the supplied comparison; inspect the intended reference.'})
        match=SIGNIFICANCE.search(text)
        if match:
            warnings.append({'code':'unsupported_significance_wording','field':field,'matched_text':match.group(),
                'review_note':'No statistical significance test is supplied by this evaluator.'})
    change=proposal_change(proposal,history,candidate)
    consistency=direction_diagnostics(proposal,change)
    warnings.extend(consistency['warnings'])
    return {'executor_comparison':comparison,'cited_measured_facts':[deepcopy(facts[fid]) for fid in cited],
        'configuration_change':change,'direction_consistency':consistency,
        'selected_candidate':proposal['candidate'],'future_hypothesis':{'source':'LLM, unverified future hypothesis','text':proposal.get('hypothesis'),
            'expected_change':deepcopy(proposal.get('expected_change'))},
        'original_prose':{'source':'LLM, not certified as measured evidence','fields':deepcopy(fields)},
        'warnings':warnings,'freeform_prose_fully_verified':False,
        'check_scope':'Narrow English phrase checks only; warnings may need review and absence of warnings does not certify prose.',
        'changes_candidate':False,'triggers_api_retry':False}


def proposed_review_context(proposal,history,candidate):
    """Derived only from the already available history and original proposal."""
    audit=audit_explanation(proposal,history,candidate)
    return {'source':'executor-derived proposal diagnostics; no new experiment results',
        'configuration_change':deepcopy(audit['configuration_change']),
        'warnings':[{'id':f'proposal_warning_{i}',**deepcopy(row)} for i,row in enumerate(audit['warnings'],1)],
        'direction_check_applicable':audit['direction_consistency']['applicable'],
        'interpretation':'Warnings are heuristic leads, not proven errors. Explain intended alternatives or correct inconsistent prose; a reasonable candidate may stay unchanged. No warnings does not certify the explanation.',
        'candidate_veto':False,'freeform_prose_fully_verified':False}


def visible_warning_context(audit):
    """Send generated codes and facts, without recycling the flagged prose."""
    return {'codes':sorted({w['code'] for w in audit['warnings']}),
        'executor_comparison':deepcopy(audit['executor_comparison']),
        'configuration_change':deepcopy(audit.get('configuration_change',{'available':False})),
        'instruction':'This review describes evidence before the previous proposal; latest_result describes the current measurement. Use the explicit trial IDs and treat generated explanations as hypotheses, not additional measurements.'}
