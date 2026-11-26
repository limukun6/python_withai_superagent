"""One evidence-grounded review of the next action, with an explicit way to abstain."""
import json
import math
from pathlib import Path

from .experiment_policy import identity


PROMPT = (
    'Review the default next action at index 0. Use only this training evidence. Compare the strongest '
    'case for continuing the default with any counterevidence. A different experiment may resolve uncertainty. '
    'Keeping the plan and abstaining are both legitimate. The archive counts are past observations, not promises. '
    'Output JSON: verdict (keep, insufficient, or experiment), option (0 for keep/insufficient, otherwise the '
    'index of a different feasible explore/confirm action), supporting_trials and contrary_trials (lists of '
    'actual trial numbers, empty when absent), reason, refute_if (a testable condition), and minimum_score '
    '(0..1 for an experiment, null otherwise). A changed option needs a supporting trial and a specific hypothesis. '
    'minimum_score is the score on the specified internal fold needed to support that hypothesis. Be concise.'
)


def load_memory(path):
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict) or set(value) != {'near', 'different'}:
        raise ValueError('Invalid historical exploration summary')
    for row in value.values():
        if (not isinstance(row, dict) or set(row) != {'improved','tied','worse','tasks'}
                or any(type(n) is not int or n < 0 for n in row.values())
                or row['tasks'] > row['improved'] + row['tied'] + row['worse']):
            raise ValueError('Invalid historical outcome counts')
    return value


def eligible(history, menu):
    return (sum(row['status'] == 'completed' for row in history) >= 3 and len(menu) > 1
            and any(a['action'] == 'explore' for a in menu))


def evidence_packet(menu, caps, profile, history, metric, remaining, memory):
    names = sorted({m for a in menu for m in a['members'] + a['evidence'].get('target_members', [])}
                   | {r['candidate'] for r in history})
    ids = {name: i for i, name in enumerate(names)}
    methods = []
    for name in names:
        cap = caps[name]
        methods.append([name, cap['estimator'], cap['representation'], cap['classifier_fits']])
    trials = []
    for row in history:
        trials.append(dict(trial=row['trial'], method=ids[row['candidate']], fold=row['split_id'], status=row['status'],
            train=row.get('train_score'), score=row.get('search_score'), class_recall=row.get('class_recall'),
            paired_errors=row.get('paired_errors')))
    options = []
    for a in menu:
        e = a['evidence']
        options.append([a['action'], [ids[m] for m in a['members']], a['split'], round(a['utility'], 7),
                        e.get('novelty'), e.get('expected'), e.get('reserved_refits',e.get('refit_fits')),
                        [ids[m] for m in e.get('target_members', [])]])
    stats = profile['statistics']
    return dict(metric=metric, task={k:stats[k] for k in ('task_kind','class_count','train_rows','feature_shape',
                'feature_count') if k in stats}, minority_fraction=profile['minority_class_fraction'],
                remaining_fits=remaining, earlier_primary_explorations=memory,
                archive_definition='Each trial versus the best earlier score on the same primary rows; different means novelty >= 0.5.',
                method_columns=['id','estimator','representation','physical_fits'], methods=methods, trials=trials,
                option_columns=['action','method_indexes','fold','priority','novelty','expected','refit_reserve','target_indexes'],
                options=options)


def unchanged(reason, verdict='keep', refs=()):
    return dict(verdict=verdict, option=0, supporting_trials=list(refs), contrary_trials=[],
                reason=reason, refute_if='', minimum_score=None)


def local_review(packet):
    rows = [t for t in packet['trials'] if t['status']=='completed' and t['fold']=='primary']
    if not rows:
        return unchanged('No measured primary method available for comparison.', 'insufficient')
    best = max(rows, key=lambda r:r['score'])
    measured = {packet['methods'][r['method']][1] for r in rows}
    spread = max(r['score'] for r in rows) - min(r['score'] for r in rows)
    tolerance = max(1 / math.sqrt(packet['task']['train_rows']), spread / 2)
    default = packet['options'][0]
    totals = packet['earlier_primary_explorations'].get('different', {})
    if totals and totals['improved'] == 0:
        return unchanged('Past measured alternatives have not yet supplied positive evidence.', 'insufficient', [best['trial']])
    for index, (kind, members, _, priority, novelty, _, reserve, _) in enumerate(packet['options']):
        method = packet['methods'][members[0]]
        if (kind != 'explore' or method[1] in measured or novelty < .5 or default[3]-priority > tolerance
                or method[3]+reserve > packet['remaining_fits']):
            continue
        if index == 0:
            return unchanged('The default already tests a competitive unmeasured estimator.', refs=[best['trial']])
        if default[0] == 'explore' and packet['methods'][default[1][0]][1] not in measured:
            return unchanged('The default already broadens the observed estimator families.', refs=[best['trial']])
        return dict(verdict='experiment', option=index, supporting_trials=[best['trial']],
                    contrary_trials=[r['trial'] for r in rows if r['trial']!=best['trial']],
                    reason='An unmeasured estimator has a priority within the observed uncertainty of the default.',
                    refute_if='Its measured primary score falls below the best previous primary score.',
                    minimum_score=best['score'])
    return unchanged('No unmeasured estimator cleared the cost and uncertainty checks.', refs=[best['trial']])


def validate(answer, packet):
    fields = {'verdict','option','supporting_trials','contrary_trials','reason','refute_if','minimum_score'}
    if not isinstance(answer, dict) or not fields.issubset(answer):
        raise ValueError('Incomplete reflection')
    v, index = answer['verdict'], answer['option']
    if (v not in ('keep','insufficient','experiment') or type(index) is not int
            or not 0 <= index < len(packet['options']) or (index != 0) != (v == 'experiment')):
        raise ValueError('Verdict does not identify an admissible action')
    seen = {r['trial'] for r in packet['trials']}
    for key in ('supporting_trials','contrary_trials'):
        refs = answer[key]
        if not isinstance(refs,list) or any(type(i) is not int or i not in seen for i in refs):
            raise ValueError('Reflection cites an unavailable trial')
    if not isinstance(answer['reason'], str) or not answer['reason'].strip() or not isinstance(answer['refute_if'],str):
        raise ValueError('Reflection lacks an explanation')
    threshold = answer['minimum_score']
    if v == 'experiment':
        if (packet['options'][index][0] not in ('explore','confirm') or not answer['supporting_trials']
                or not answer['refute_if'].strip() or type(threshold) not in (int,float) or not 0 <= threshold <= 1):
            raise ValueError('Experiment lacks a checkable prediction')
    elif threshold is not None:
        raise ValueError('No new experiment was proposed for this threshold')
    return {k:answer[k] for k in sorted(fields)}


def decide(mode, packet, flash, log):
    digest = identity(packet)
    log.write_json('reflection_input.json',packet)
    state = 'valid'
    try:
        if mode == 'flash':
            if flash is None:
                raise RuntimeError('Reflection client unavailable')
            answer = flash.ask(dict(instruction=PROMPT, evidence=packet), log, 'feedback_reflection', cost_ceiling=.12)
        elif mode == 'local':
            answer = local_review(packet)
        else:
            answer = unchanged('Unmodified control action.')
        result = validate(answer,packet)
    except (ValueError,RuntimeError,TimeoutError,OSError) as exc:
        state = 'fallback'
        result = unchanged(f'{type(exc).__name__}: {str(exc)[:240]}','insufficient')
    record = dict(mode=mode, input_digest=digest, status=state, decision=result)
    log.write_json('reflection.json',record)
    return record
