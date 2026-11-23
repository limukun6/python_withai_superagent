"""Check the machine-checkable claims inside free-form proposal prose.

Prose cannot be fully verified, and this module does not pretend otherwise. It
extracts the three claim types that can be checked without interpretation:

    classifier family   "a character n-gram SVM" when the candidate is logistic
    trial reference     "trial 7" when only two trials exist
    quoted percentage   "71.35%" when no measurement produced that number

Only a narrowly bound measurement can certify a quoted percentage. Other
number matches, uncertain statements and unsupported references stay unknown.
"""
import math
import re

# (pattern, canonical family, families it is compatible with)
CLASSIFIER_TERMS = (
    (re.compile(r'\blogistic\s+regression\b', re.I), 'logistic', ('logistic',)),
    (re.compile(r'\blogistic\b', re.I), 'logistic', ('logistic',)),
    (re.compile(r'\blinear\s+svm\b', re.I), 'linear_svm', ('linear_svm',)),
    (re.compile(r'\bsupport\s+vector\w*\b', re.I), 'svm', ('linear_svm', 'rbf_svm')),
    (re.compile(r'\bsvm\b', re.I), 'svm', ('linear_svm', 'rbf_svm')),
    (re.compile(r'\bcatboost\b', re.I), 'catboost', ('catboost',)),
    (re.compile(r'\bgradient\s+boost\w*\b', re.I), 'gradient_boosting',
     ('catboost', 'gradient_boosting')),
    (re.compile(r'\bextra[\s-]?trees?\b', re.I), 'extra_trees', ('extra_trees',)),
    (re.compile(r'\brandom\s+forest\b', re.I), 'random_forest', ('random_forest',)),
    (re.compile(r'\bna[iï]ve\s+bayes\b', re.I), 'naive_bayes', ('naive_bayes',)),
    (re.compile(r'\brbf\b', re.I), 'rbf_svm', ('rbf_svm',)),
)
TRIAL = re.compile(r'\btrials?\s+(\d+)\b', re.I)
TRIAL_WORD = re.compile(r'\btrials?\b', re.I)
PERCENT = re.compile(r'(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%')
MEASUREMENTS = ('search_accuracy', 'train_accuracy', 'search_score', 'train_score')
# A classifier word near one of these describes another experiment, not the
# selected candidate: "a stronger probe than the logistic baseline".
COMPARISON = re.compile(
    r'\b(?:than|over|above|below|beyond|surpass\w*|beat|versus|vs\.?|relative\s+to|'
    r'compared\s+(?:with|to)|instead\s+of|'
    r'rather\s+than|unlike|baseline|incumbent|previous)\b', re.I)
SENTENCE_SPLIT = re.compile(r'(?<=[.!?;])\s+|\n+')
# Candidate identifiers look like char_C1.0 or word_char_stats_C0.3.
ID_LIKE = re.compile(r'\b[a-z][a-z_]*(?:_[A-Za-z0-9.]+)+\b', re.I)
CURRENT_SUBJECT = re.compile(r'\b(?:(?:current|selected|proposed|next)\s+(?:model|classifier|candidate|configuration)|'
                            r'this\s+(?:model|classifier|candidate)|(?:this|it)\s+is|the\s+classifier\s+is)\b', re.I)
TEMPORAL_REFERENCE = re.compile(r'\b(?:after|following)\b', re.I)
# Only a nearby noun phrase is handled; "after observing scores, SVM is better
# than baseline" must not make the current SVM mention historical.
HISTORICAL_BASELINE_TAIL = re.compile(
    r'^\s+(?:(?:regression|model|classifier|C\d+(?:\.\d+)?)\s+){0,2}'
    r'(?:baseline|incumbent)\b', re.I)


def _family_of(kind, identifier, params=None):
    from .targeted_feedback import candidate_facts
    try:
        facts = candidate_facts(kind, {'id': identifier, 'params': params or {}})
    except (ValueError, KeyError, TypeError):
        return None
    return (facts or {}).get('classifier')


def scope_families(text, facts, history, kind):
    """Classifier families this sentence may legitimately be talking about."""
    families = set()
    if (facts or {}).get('classifier'):
        families.add(facts['classifier'])
    if not kind:
        return families
    for match in ID_LIKE.finditer(text):
        family = _family_of(kind, match.group())
        if family:
            families.add(family)
    for match in TRIAL.finditer(text):
        row = next((r for r in history if r.get('trial') == int(match.group(1))), None)
        if row:
            family = _family_of(kind, row.get('candidate'), row.get('params'))
            if family:
                families.add(family)
    return families


def sentence_around(text, start):
    """The sentence containing an offset, used to decide what a mention refers to."""
    boundaries = [(match.start(), match.end()) for match in SENTENCE_SPLIT.finditer(text)]
    begin, end = 0, len(text)
    for clause_start, clause_end in boundaries:
        if clause_start <= start:
            begin = clause_end
        else:
            end = clause_start
            break
    return text[begin:end], text[begin:start]


def classifier_claims(text):
    """Longest match wins: "logistic regression" does not also count as "logistic".

    A mention that sits in a comparison clause ("than the logistic baseline")
    describes a different experiment, so it is reported as a mention rather than
    scored against the selected candidate.
    """
    found = []
    for pattern, family, compatible in CLASSIFIER_TERMS:
        for match in pattern.finditer(text):
            found.append((match.start(), match.end(), match.group(), family, compatible))
    found.sort(key=lambda row: (row[0], -(row[1] - row[0])))
    kept = []
    for start, end, span, family, compatible in found:
        if any(inner_start <= start and end <= inner_end and (inner_start, inner_end) != (start, end)
               for inner_start, inner_end, *_ in kept):
            continue
        kept.append((start, end, span, family, compatible))
    claims = []
    for start, end, span, family, compatible in kept:
        sentence, before = sentence_around(text, start)
        # An explicit current subject starts a new reference scope. Earlier
        # "baseline" or "trial" words must not exempt a later current claim.
        subjects = list(CURRENT_SUBJECT.finditer(before))
        reference_prefix = before[subjects[-1].end():] if subjects else before
        after = sentence[len(before) + end - start:]
        # "than the logistic baseline" and "trial 1 logistic C0.05" both name a
        # different experiment. A temporal baseline noun phrase does too.
        elsewhere = bool(COMPARISON.search(reference_prefix) or TRIAL_WORD.search(reference_prefix)
            or (TEMPORAL_REFERENCE.search(reference_prefix) and HISTORICAL_BASELINE_TAIL.match(after)))
        claims.append({'kind': 'classifier', 'span': span, 'claim': family,
                       'compatible_with': compatible, 'sentence': sentence,
                       'in_comparison': elsewhere})
    return claims


def trial_claims(text):
    return [{'kind': 'trial_reference', 'span': match.group(), 'claim': int(match.group(1))}
            for match in TRIAL.finditer(text)]


def percentage_claims(text):
    return [{'kind': 'percentage', 'span': match.group(), 'claim': float(match.group(1))}
            for match in PERCENT.finditer(text)]


def measured_percentages(history):
    values = []
    for row in history:
        for key in MEASUREMENTS:
            value = row.get(key)
            if type(value) in (int, float) and math.isfinite(value):
                values.append(100.0 * value)
    return values


# Only this restricted declarative grammar is certified as a measurement.
# Complex references, multiple percentages and future predictions stay unknown.
BOUND_MEASUREMENT = re.compile(
    r'^trials?\s+(?P<trial>\d+)\s+(?P<split>train|search)\s+'
    r'(?P<metric>accuracy|macro_f1|balanced_accuracy|roc_auc|log_loss)\s+'
    r'(?:was|is|=|:)\s*(?P<value>\d{1,3}(?:\.\d+)?)\s*%[.!]?$', re.I)
NEGATION = re.compile(r"\b(?:not|never|no|without)\b|n['’]t\b",re.I)
UNCERTAIN = re.compile(r'\b(?:will|would|may|might|could|expect\w*|predict\w*|hope\w*|if)\b',re.I)


def percentage_evidence(text, position, history, tolerance):
    from .measurement_claims import check_measurement_claim
    sentence, _ = sentence_around(text,position)
    match=BOUND_MEASUREMENT.fullmatch(sentence.strip())
    if match is None:
        return {'status':'unverifiable','observed':None,'reason':'unbound_or_unsupported_statement'}
    return check_measurement_claim(dict(trial=int(match['trial']),split=match['split'].lower(),
        metric=match['metric'].lower(),value=float(match['value'])/100.),history,tolerance=tolerance/100.)


def check_prose(fields, facts, history, kind=None, tolerance=0.01):
    """Score every extractable claim in the given prose fields."""
    trials = {r.get("trial") for r in history}
    items = []
    for field, text in fields.items():
        if not isinstance(text, str) or not text.strip():
            continue
        for claim in classifier_claims(text):
            if claim['in_comparison']:
                items.append({'field': field, **claim, 'observed': None, 'status': 'informational'})
                continue
            scope = scope_families(claim['sentence'], facts, history, kind)
            historical_review = field in ('review','assessment.interpretation')
            bound_reference = bool(TRIAL.search(claim['sentence']) or any(
                row.get('candidate') and row['candidate'] in claim['sentence'] for row in history))
            unbound_review = historical_review and not CURRENT_SUBJECT.search(claim['sentence']) and not bound_reference
            if unbound_review or NEGATION.search(claim['sentence']) or UNCERTAIN.search(claim['sentence']) or len(scope)>1:
                status, observed = 'unverifiable', sorted(scope) if scope else None
            elif not scope:
                status, observed = 'unverifiable', None
            elif scope & set(claim['compatible_with']):
                status, observed = 'confirmed', sorted(scope)
            else:
                status, observed = 'contradicted', sorted(scope)
            items.append({'field': field, **claim, 'observed': observed, 'status': status,
                          'reference_resolution':'unbound_review_subject' if unbound_review else 'current_or_locally_scoped'})
        for match in TRIAL.finditer(text):
            claim={'kind':'trial_reference','span':match.group(),'claim':int(match.group(1))}
            sentence,_=sentence_around(text,match.start())
            if UNCERTAIN.search(sentence):
                items.append({'field':field,**claim,'observed':None,'status':'unverifiable'})
            elif claim['claim'] not in trials:
                items.append({'field': field, **claim, 'observed': sorted(trials), 'status': 'contradicted'})
            else:
                items.append({'field': field, **claim, 'observed': sorted(trials), 'status': 'confirmed'})
        for match in PERCENT.finditer(text):
            evidence=percentage_evidence(text,match.start(),history,tolerance)
            if 'claim' in evidence:
                evidence['measurement_claim']=evidence.pop('claim')
            items.append({'field':field,'kind':'percentage','span':match.group(),
                'claim':float(match.group(1)),**evidence})
    counts = {status: sum(item['status'] == status for item in items)
              for status in ('confirmed', 'contradicted', 'unverifiable', 'informational')}
    scored = len(items) - counts['informational']
    return {'claims_checked': scored, **counts, 'items': items,
            'unique_contradictions': sorted({item['span'] for item in items
                                             if item['status'] == 'contradicted'}),
            'scope': 'Classifier keywords, existence of trial identities and narrowly bound measurements only; unbound percentages are not confirmed. '
                     'classifier words in a comparison clause describe another experiment and '
                     'are counted as informational, not scored'}
