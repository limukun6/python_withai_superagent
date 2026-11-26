"""Narrow direction reminders, never a semantic judge or candidate veto.

Only a single C change within the same declared route has an unambiguous
configuration direction here. The classifier's accuracy response is unknown.
English phrase context is heuristic; an empty warning list is not certification.
"""
import re


DIRECTION = re.compile(r'\b(stronger|more|increased|weaker|less|reduced)\s+regulari[sz]ation\b', re.I)
CONTRAST = re.compile(r'\b(?:alternative|counterhypothesis|counter-hypothesis|opposite hypothesis|both|rather than|versus|vs\.?|instead of|although|unlike|whether)\b', re.I)
PAST = re.compile(r'\b(?:previous|prior|earlier|baseline|reference)\b', re.I)
NEGATION_BEFORE = re.compile(r"\b(?:not|never|avoid|reject|don't|cannot|can't)\b[^.!?;]{0,32}$", re.I)
NEGATION_AFTER = re.compile(r"^\s+(?:(?:may|might|would|will|does|is|can)\s+)?(?:not|never|isn't|doesn't)\b", re.I)
CLAUSE = re.compile(r'(?<=[.!?;])\s+|\s+(?:but|however|whereas)\s+', re.I)


def direction_diagnostics(proposal, change):
    result={'scope':'same declared route, single changed C; narrow English phrases only',
            'applicable':False,'expected_regularization_direction':None,'mentions':[],
            'warnings':[],'freeform_semantics_verified':False}
    changed=change.get('changed_parameters',[])
    if not change.get('single_declared_parameter_change') or len(changed)!=1 or changed[0]['parameter']!='C':
        result['reason_not_checked']='No unambiguous single-C comparison within one declared route'
        return result
    effect=(changed[0].get('parameter_effect') or {}).get('declared_effect')
    if effect not in ('stronger_regularization','weaker_regularization'):
        result['reason_not_checked']='C direction unavailable'
        return result
    expected=effect.split('_')[0];result.update(applicable=True,expected_regularization_direction=expected)
    fields={name:proposal.get(name,'') for name in ('reason','hypothesis')}
    fields['assessment.interpretation']=(proposal.get('assessment') or {}).get('interpretation','')
    for field,text in fields.items():
        if not isinstance(text,str):continue
        for clause in CLAUSE.split(text):
            for match in DIRECTION.finditer(clause):
                direction='stronger' if match.group(1).lower() in ('stronger','more','increased') else 'weaker'
                # These cues exclude a claim from this narrow check, not from human review.
                cue=('explicit_comparison_or_alternative' if CONTRAST.search(clause) else
                     'negation' if NEGATION_BEFORE.search(clause[:match.start()]) or NEGATION_AFTER.search(clause[match.end():]) else
                     'historical_or_reference_context' if PAST.search(clause) else None)
                item={'field':field,'matched_text':match.group(),'clause':clause.strip()[:500],
                      'mentioned_direction':direction,'context_excluded_by_heuristic':cue}
                result['mentions'].append(item)
                if direction!=expected and cue is None:
                    result['warnings'].append({'code':'possible_regularization_direction_rationale_mismatch',
                        **item,'expected_direction':expected,'reference_trial':change['reference_trial'],
                        'selected_candidate':change['selected_candidate'],
                        'review_note':'The selected C change has the opposite direction to this rationale. Explain any intended counterhypothesis or repair the prose; this heuristic does not prove the candidate is wrong.'})
    return result
