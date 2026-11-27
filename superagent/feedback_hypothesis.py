"""Propose a falsifiable sample-group check from a measured training trial."""
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import sqlite3


from .opening_evidence import subset

MODES = ('opening-feedback-local', 'opening-feedback-flash')
PROMPT = (
    'Read the anonymous task facts and the actual first training experiment in first_trial. '
    'Return exactly goal, metric, citations, uncertainty, check, hypothesis, if_false. '
    'Use the exact metric. Select keep, one existing check key, or {"all_of":[key1,key2]} to propose '
    'a new joint group of examples. Only class or slice keys can be joined. Cite first_trial, the '
    'source fact for each key, and a doc fact if supplied. State what later training results would '
    'refute the hypothesis. A 0.05 excess error on separate training folds is the threshold. '
    'Previous anonymous lessons are leads only. Do not change the task, budget or scoring rule. '
    'Keep the JSON concise. The deployable prediction is already protected.'
)


class LessonStore:
    """Only de-identified check keys and internal verdicts cross runs. A task cannot vote twice."""
    def __init__(self, directory, payload_digest, kind, metric):
        self.path = Path(directory).resolve()
        self.scope = (payload_digest, kind, metric)
        self.read_only = os.environ.get('SUPERAGENT_LESSONS_READONLY') == '1'

    def connection(self):
        return sqlite3.connect(self.path.as_uri()+'?mode=ro' if self.read_only else str(self.path),
                               uri=self.read_only,timeout=2)

    def hints(self, checks):
        if not self.path.exists():
            return []
        with self.connection() as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='internal_checks'").fetchone():
                return []
            result = db.execute('SELECT terms, verdict, COUNT(*) FROM internal_checks WHERE sample != ? '
                'AND modality=? AND metric=? GROUP BY terms, verdict ORDER BY terms, verdict',self.scope)
            hints = {}
            for encoded, verdict, count in result:
                terms = json.loads(encoded)
                if isinstance(terms,list) and 1<=len(terms)<=2 and all(isinstance(k,str) and k in checks for k in terms):
                    hints.setdefault(tuple(terms), Counter())[verdict] = count
        return [dict(terms=list(terms),outcomes=dict(counts)) for terms,counts in hints.items()]

    def record(self,terms,verdict):
        if self.read_only:
            return False
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS internal_checks (sample TEXT, modality TEXT, metric TEXT, '
                       'terms TEXT, verdict TEXT, PRIMARY KEY(sample,modality,metric,terms))')
            insert=db.execute('INSERT OR IGNORE INTO internal_checks VALUES (?,?,?,?,?)',
                       (*self.scope,json.dumps(sorted(terms)),verdict))
        return insert.rowcount==1


def joint_check(keys, checks):
    if (not isinstance(keys,list) or len(keys)!=2 or any(not isinstance(k,str) or k not in checks for k in keys)
            or len(set(keys))!=2 or any(checks[k]['kind'] not in ('slice','class') for k in keys)):
        raise ValueError('A joint hypothesis needs two distinct existing sample-group checks')
    terms=sorted(keys)
    parts=[checks[k] for k in terms]
    if parts[0]['source']==parts[1]['source']:
        raise ValueError('Joined checks must use distinct signals')
    return '&'.join(terms), dict(kind='all_of',terms=terms,parts=parts,source='first_trial')


def start_feedback(opening, trial, observed):
    from .task_opening import error_gap
    checks=opening.brief['checks']
    row=next(iter(observed['primary'].values()))
    held=opening.folds['primary'][1]
    signals={k:subset(opening.task,opening.metric,held,c) for k,c in checks.items() if c['kind'] in ('slice','class')}
    gaps={k:error_gap(row,m) for k,m in signals.items()}
    stats=dict(trial=trial,partition='primary_training_holdout',rows=len(held),
               score=row['score'],groups=gaps)
    opening.brief['facts']['first_trial']=stats
    hints=[]
    if os.environ.get('SUPERAGENT_LESSON_STORE'):
        try:
            primary=json.loads((opening.log.run_dir/'primary_split.json').read_text(encoding='utf-8'))
            opening.lesson_store=LessonStore(os.environ['SUPERAGENT_LESSON_STORE'],primary['payload_sha256'],
                                            opening.task.kind,opening.metric.name)
            hints=opening.lesson_store.hints(checks)
        except (OSError,sqlite3.Error,ValueError) as error:
            opening.record['lesson_store_issue']=type(error).__name__
    opening.brief['facts']['prior_checks']=hints
    opening.record.update(dispatch_after_trials=trial,prior_compatible_checks=len(hints))
    opening.log.write_json('opening_brief.json',opening.brief)
    ranked=sorted((k for k,v in gaps.items() if v['supported'] is not None),
                  key=lambda k:(-gaps[k]['gap'],k))[:4]
    options={tuple([k]):v for k,v in gaps.items()}
    previous={tuple(h['terms']) for h in hints if any(v>0 for name,v in h['outcomes'].items()
                                                      if name in ('deferred','supported'))}
    pairs=set(itertools.combinations(ranked,2)) | {t for t in previous if len(t)==2}
    for pair in pairs:
        a,b=pair
        if a in signals and b in signals and checks[a]['source']!=checks[b]['source']:
            options[tuple(sorted(pair))]=error_gap(row,signals[a]&signals[b])
    eligible=[key for key,gap in options.items() if gap['supported'] is True]
    terms=min(eligible,key=lambda k:(-options[k]['gap'], k not in previous,len(k),k)) if eligible else ()
    check=terms[0] if len(terms)==1 else {'all_of':list(terms)} if terms else 'keep'
    cited=['first_trial','contract',*(checks[k]['source'] for k in terms)]
    doc=next((k for k in opening.brief['facts'] if k.startswith('doc_')),None)
    if doc: cited.append(doc)
    return dict(goal='Classify the stated target using the declared metric.',metric=opening.metric.name,
        citations=list(dict.fromkeys(cited)),check=check,
        uncertainty='Does the primary training error pattern recur on separate training rows?',
        hypothesis=('The group defined by '+', '.join(terms)+' has at least five percentage points more errors.'
                    if terms else 'No measured group met the predeclared excess-error threshold.'),
        if_false='Retain the current model when another training fold fails to reproduce this difference.')


def validate_feedback(answer,brief):
    from .task_opening import validate
    if not isinstance(answer,dict) or not isinstance(answer.get('citations'),list) or 'first_trial' not in answer['citations']:
        raise ValueError('Feedback hypotheses must cite an actual completed training trial')
    checks=brief['checks']
    check=answer.get('check')
    if isinstance(check,str) and checks.get(check,{}).get('kind')=='all_of':
        check={'all_of':checks[check]['terms']}
    if isinstance(check,dict):
        if set(check)!={'all_of'}:
            raise ValueError('Unknown sample-group operation')
        key, definition=joint_check(check['all_of'],checks)
        if not all(part['source'] in answer['citations'] for part in definition['parts']):
            raise ValueError('A joint hypothesis must cite both observed signals')
        answer=validate(dict(answer,check=key),dict(brief,checks={**checks,key:definition}))
        checks[key]=definition
        return answer
    return validate(answer,brief)


def measure_late_answer(opening):
    """A late proposal can become a training lesson, without rewriting an earlier model decision."""
    from .task_opening import error_gap
    bases=getattr(opening,'decision_bases',None)
    if not bases or opening.record.get('interpretation_status')!='valid' or opening.record.get('reply_used'):
        return
    key=opening.answer['check']
    definition=opening.brief['checks'].get(key)
    if not definition or definition['kind'] not in ('class','slice','all_of') or key==opening.record.get('check'):
        return
    details={fold:error_gap(row,subset(opening.task,opening.metric,opening.folds[fold][1],definition))
             for fold,row in bases.items() if row is not None}
    if not details:
        return
    states=[row['supported'] for row in details.values()]
    verdict='deferred'
    if len(states)>=2 and all(s is True for s in states):
        verdict='supported'
    elif len(states)>=2 and all(s is False for s in states):
        verdict='refuted'
    opening.record['late_hypothesis']=dict(check=key,terms=definition.get('terms',[key]),
        disposition=verdict,observations=details,used_for_model=False)
    store=getattr(opening,'lesson_store',None)
    if store:
        try:
            opening.record['late_hypothesis']['stored']=store.record(definition.get('terms',[key]),verdict)
        except (OSError,sqlite3.Error) as error:
            opening.record['lesson_store_issue']=type(error).__name__


def save_disposition(opening):
    key=opening.record.get('check')
    definition=opening.brief['checks'].get(key)
    if not definition:
        return
    verdict=opening.record['check_status']
    measured=opening.record.get('diagnosis',{})
    if not measured or definition['kind'] not in ('class','slice','all_of'):
        return
    if verdict in ('uncorroborated','mixed','unmeasurable'):
        disposition='deferred'
    elif verdict in ('refuted','supported'):
        disposition=verdict
    else:
        return
    terms=definition.get('terms',[key])
    opening.record['training_lesson']=dict(terms=terms,disposition=disposition,
        measured_folds=sorted(measured),model_selection_evidence='current task training folds')
    store=getattr(opening,'lesson_store',None)
    if store is not None:
        try:
            opening.record['training_lesson']['stored']=store.record(terms,disposition)
        except (OSError,sqlite3.Error) as error:
            opening.record['lesson_store_issue']=type(error).__name__
