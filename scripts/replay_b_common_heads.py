"""Replay both local selection rules using the same three frozen, single-fit heads."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit_b_local_component_feedback import audit
from superagent.ab_scheduler import next_action, selection_evidence
from superagent.prediction_evidence import fingerprint


IDS = dict(word='word_C0.05', base='minilm_text_with_lexical_features',
           deep='minilm_l12_text_with_lexical_features')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def same(left, right):
    if left != right:
        raise ValueError('Frozen inputs do not agree')


def run(directory):
    original, new = directory/'original', directory/'components'
    state = read(original/'state_1.json')
    evidence = read(new/'inner_evidence.json')
    src = new/'training_payload.joblib'
    data = joblib.load(src)
    held = evidence['held_row_ids']
    primary = read(original/'primary_split.json')
    same(held, primary['search_indices'])
    same(held, list(data['held_ids']))
    same(primary['train_indices'], list(data['fit_ids']))
    same(digest(original/'primary.joblib'), primary['evaluation_payload_sha256'])
    real_payload = joblib.load(original/'primary.joblib')
    same([tuple(data['rows'][i]) for i in held], list(map(tuple,real_payload['X_search'])))
    labels = np.asarray(data['labels'])[held]
    np.testing.assert_array_equal(labels, real_payload['y_search'])
    groups = np.asarray(data['groups'])[held]
    truth = labels.tolist()
    same(fingerprint(truth), primary['label_sha256'])
    trusted = {'primary': dict(row_ids=held, labels=truth, groups=groups.tolist(),
                               evaluation_payload_sha256=primary['evaluation_payload_sha256'])}
    bank = {IDS[name]: evidence['observed_predictions'][name] for name in IDS}
    caps = {v['id']:v for v in state['available_candidates'] if v['id'] in (IDS['word'],IDS['base'])}
    caps[IDS['deep']] = dict(caps[IDS['base']], id=IDS['deep'], route='minilm_l12_lexical',
                              params=dict(encoder='sentence-transformers/all-MiniLM-L12-v2',
                                          C=.3, representation='word_char_minilm', swap_augmentation=True))
    actions = []
    while True:
        seen = {v['candidate'] for v in state['history']}
        state['available_candidates'] = [caps[n] for n in IDS.values() if n not in seen]
        action = next_action(state, trusted_partitions=trusted, capabilities=caps)
        actions.append(action)
        if action['action'] == 'stop':
            old_choice = selection_evidence(state, trusted_partitions=trusted)['incumbent']
            break
        if action['split_id'] != 'primary':
            old_choice = None
            break
        name = action['candidate_id']
        if name in seen or len(actions) > len(IDS):
            raise ValueError('Unexpected repeat')
        pred = bank[name]
        score = float(np.mean(np.array(pred)==labels))
        state['history'].append(dict(trial=len(actions), candidate=name, params=caps[name]['params'],
            split_id='primary', status='completed', primary_metric='accuracy', metric_spec=state['task']['metric'],
            search_score=score, search_accuracy=score, classifier_fits=1))
        state['prediction_evidence'].append(dict(trial=len(actions), candidate_id=name, split_id='primary',
            row_id_scope='positions within this run training input', row_ids=held, predictions=pred,
            prediction_sha256=fingerprint(pred), label_sha256=primary['label_sha256'],
            evaluation_payload_sha256=primary['evaluation_payload_sha256']))
        budget = state['budget']
        budget.update(attempted_classifier_fits=len(actions), completed_classifier_fits=len(actions),
                      remaining_fit_allowance=budget['fit_limit']-len(actions))
    probe = np.array([int(g[:2],16)<128 for g in groups])
    np.testing.assert_array_equal(probe, evidence['probe'])
    same(set(groups[probe]).isdisjoint(groups[~probe]), True)
    joint = int(np.count_nonzero((np.asarray(bank[IDS['word']])[probe]!=labels[probe]) &
                                 (np.asarray(bank[IDS['base']])[probe]!=labels[probe])))
    threshold = max(8, math.ceil(.05*int(probe.sum())))
    # Keep the production worker's two-fit reserve at the deep expansion gate.
    names = ['word','base']
    if joint >= threshold and 2+1+2 <= state['budget']['fit_limit']:
        names.append('deep')
    scores = {name:int(np.count_nonzero(np.asarray(bank[IDS[name]])==labels)) for name in names}
    new_choice = max(names, key=scores.get)  # All cost one; insertion order is the production tie-break.
    full_audit = audit(new)
    same(full_audit['original_chosen'], read(new/'worker_result.json')['selected'])
    for k,v in scores.items():
        same(evidence['scores'][k]['total'],v)
    prior = read(directory/'results.json')
    genuine = read(original/'prediction_evidence.json')
    old_differences = {k:int(np.count_nonzero(np.array(next(x['predictions'] for x in genuine
        if x['candidate_id']==IDS[k] and x['split_id']=='primary'))!=np.array(bank[IDS[k]]))) for k in ('word','base')}
    shared = None
    if old_choice == IDS[new_choice] and prior['runs']['components']['selected']=='component_'+new_choice:
        same(digest(new/'selected_model.joblib'),prior['runs']['components']['model'])
        shared = dict(model_sha256=prior['runs']['components']['model'],rows=prior['rows'],
                      correct=prior['correct']['components'], provenance='same prior frozen artifact; no new training or export')
    return dict(kind='same three heads / frozen action replay', inputs={str(p.relative_to(directory)):digest(p)
        for p in (src,new/'inner_evidence.json',original/'primary.joblib',original/'state_1.json',directory/'results.json')},
        source={p.name:digest(p) for p in (Path(__file__),REPO/'superagent/ab_scheduler.py',
                REPO/'superagent/ab_comparison.py',REPO/'scripts/b_component_worker.py')},
        menu=list(IDS.values()), primary_rows=len(held), row_fingerprint=fingerprint(held),
        scores=scores, historical_first_two_prediction_differences=old_differences,
        original_scheduler=dict(actions=actions,selected=old_choice),
        component_policy=dict(replayed=True, selected=IDS[new_choice], common_probe_errors=joint,
            expansion_threshold=threshold, observed=names, probe_rows=int(probe.sum()),
            confirmation_rows=int((~probe).sum()), eight_arm_audit_winner=full_audit['original_chosen']),
        reused_delivery=shared, actual_new_fits=0, model_api_requests=0)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    result = run(args.run)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2)
        handle.write('\n')
