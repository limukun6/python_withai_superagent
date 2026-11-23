"""Source-backed, narrow C semantics diagnostics; never an action veto or prose certificate."""
from functools import lru_cache
import hashlib
import math
import re

NUMBER=r'(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?'
VALUE=re.compile(r'\bC\s*=\s*('+NUMBER+r')',re.I)
RANGE=re.compile(r'from\s+C\s*=\s*('+NUMBER+r')\s+to\s+C\s*=\s*('+NUMBER+r')',re.I)
STRENGTH=re.compile(r'\b(increasing|decreasing|higher|lower|stronger|weaker|less|more)\s+(?:logistic\s+)?regulari[sz]ation\s+strength\b',re.I)
RELATION=re.compile(r'\b(increasing|decreasing)\s+C\s+(weakens|strengthens)\s+regulari[sz]ation\b',re.I)
INVERSE=re.compile(r'\bC\s+is\s+(inverse|directly proportional to)\s+regulari[sz]ation\s+strength\b',re.I)
EXCLUDED=re.compile(r"\b(?:not|never|whether|unlike|previous|baseline|rather than|instead of|counterhypothesis|different representation)\b|n['’]t\b",re.I)
MENTION=re.compile(r'\bC\b|regulari[sz]',re.I)
FAMILIES={'logistic':'LogisticRegression','linear_svm':'LinearSVC','rbf_svm':'SVC'}


@lru_cache(maxsize=3)
def definition(family):
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC,SVC
    cls={'logistic':LogisticRegression,'linear_svm':LinearSVC,'rbf_svm':SVC}[family]
    text=cls.__doc__;start=text.index('C : float');excerpt=text[start:].split('\n\n',1)[0].strip()
    if 'regularization' not in excerpt.lower() or 'inverse' not in excerpt.lower():
        raise ValueError('Runtime parameter documentation no longer supports the registered relation')
    return dict(estimator_class=cls.__name__,package='scikit-learn',version=sklearn.__version__,
                parameter='C',relation='inverse_regularization_strength',source='runtime_official_class_documentation',
                excerpt=excerpt,excerpt_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),
                accuracy_effect_verified=False)


def positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0


def check_parameter_semantics(fields,candidate,history,selected_facts,kind=None):
    """Candidate params are declarations; reference estimator identity is observed when present.

    Only exact definition sentences and explicitly bound same-route C changes are
    supported. Absolute "low/high regularization" with a lone C has no scale-
    independent cutoff. It remains unknown, including previously flagged prose.
    """
    from .configuration_facts import choose_reference, _route
    facts=selected_facts or {};family=facts.get('classifier');params=(candidate or {}).get('params',{})
    applicable=(family in FAMILIES and facts.get('source') in
                ('registered_training_function','validated_recipe_parameters','shared_ab_catalogue')
                and positive(params.get('C')) and params.get('penalty','l2') not in (None,'none'))
    doc=definition(family) if applicable else None
    reference,rule=choose_reference(history,candidate) if candidate else (None,'missing_candidate')
    bound=(applicable and reference is not None and reference.get('status')=='completed'
           and positive(reference.get('params',{}).get('C'))
           and reference.get('params',{}).get('penalty','l2') not in (None,'none')
           and _route(candidate) is not None and _route(candidate)==_route(reference)
           and reference.get('estimator',{}).get('family')==family)
    items=[]
    for field,text in fields.items():
        if not isinstance(text,str):continue
        # Decimal points are not sentence boundaries.
        for clause in re.split(r'(?<=[.!?;])\s+|\n+',text):
            if not MENTION.search(clause):continue
            item=dict(field=field,span=clause,status='unknown',reason='Unsupported or ambiguous parameter statement',
                      candidate_id=(candidate or {}).get('id'),parameter='C',candidate_parameter_value=params.get('C'),
                      candidate_value_source='declared_candidate_configuration',reference_trial=None)
            if not applicable:item['reason']='No supported, penalized estimator configuration with positive C'
            elif EXCLUDED.search(clause):item['reason']='Historical, alternative or negated scope is outside the narrow grammar'
            else:
                relation=RELATION.search(clause);inverse=INVERSE.search(clause)
                interval=RANGE.search(clause);strength=STRENGTH.search(clause)
                if relation:
                    expected='weakens' if relation[1].lower()=='increasing' else 'strengthens'
                    item.update(status='confirmed' if relation[2].lower()==expected else 'contradicted',
                                reason='Definition of C within the registered estimator',definition=doc)
                elif inverse:
                    item.update(status='confirmed' if inverse[1].lower()=='inverse' else 'contradicted',
                                reason='Definition of C within the registered estimator',definition=doc)
                elif strength and bound:
                    before=reference['params']['C'];after=params['C']
                    numbers=[float(m[1]) for m in VALUE.finditer(clause)]
                    pair=[float(interval[1]),float(interval[2])] if interval else None
                    linked=(pair==[before,after] if pair else len(numbers)==1 and numbers[0]==after)
                    if linked and before!=after:
                        direction='weaker' if after>before else 'stronger'
                        claimed='stronger' if strength[1].lower() in ('increasing','higher','stronger','more') else 'weaker'
                        item.update(status='confirmed' if direction==claimed else 'contradicted',
                            reason='Same-route numeric C change has inverse penalty-coefficient direction',definition=doc,
                            reference_trial=reference['trial'],reference_rule=rule,
                            reference_candidate=reference['candidate'],reference_parameter_value=before,
                            reference_parameter_source='recorded_trial_configuration',
                            reference_estimator_source=reference['estimator'].get('source'),
                            expected_penalty_direction=direction,claimed_penalty_direction=claimed)
            items.append(item)
    warnings=[dict(code='parameter_semantics_contradiction',**row) for row in items if row['status']=='contradicted']
    return dict(scope='Supported explicit English C definitions and same-route bound changes only',
                applicable=bool(applicable),definition=doc,items=items,warnings=warnings,
                freeform_prose_fully_verified=False,changes_candidate=False,triggers_api_retry=False)
