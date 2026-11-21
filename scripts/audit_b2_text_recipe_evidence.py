"""Replay B2 choices and remeasure their real model evidence without fitting."""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import math
import os
from pathlib import Path
from unittest.mock import patch

import joblib
from sklearn.linear_model import LogisticRegression

from scripts import run_b1_ab_acceptance as record
from scripts.replay_b1_observed_actions import key, trusted_input
from superagent import ab_run, ab_text_recipes
from superagent.ab_scheduler import next_action
from superagent.metrics import MetricSpec
from superagent.models import predict_with_model
from superagent.prediction_evidence import fingerprint as vector_hash
from superagent.scheduler_bridge import fingerprint


PUBLISHED = 'research/b2_bounded_text_composition'


def choose(state, caps, trusted, enabled):
    proofs = []
    def compose(eligible, verified, menu):
        proposal = ab_text_recipes.propose(eligible, verified, menu)
        if proposal:
            proofs.append(proposal)
        return proposal
    action = next_action(state, capabilities=caps, trusted_partitions=trusted,
                         on_local_recipe=compose if enabled else None)
    return action, proofs


def remeasure(run, trial, history, evidence):
    row = next(r for r in history if r['trial']==trial)
    prediction = next(r for r in evidence if r['trial']==trial)
    split = row['split_id']
    if split!='primary' or row['status']!='completed':
        raise ValueError('Composition reference must be a completed primary observation')
    path = run/f'trial_{trial}.joblib'
    trained = joblib.load(path)
    payload = joblib.load(run/(split+'.joblib'))
    metric = MetricSpec.from_payload(payload)
    if (trained['metrics']['resolved_configuration']['id']!=row['candidate']
            or ab_run.sha(run/(split+'.joblib'))!=prediction['evaluation_payload_sha256']):
        raise ValueError('Wrong fitted candidate or split payload')
    scores = {}
    for side in ('train','search'):
        pred = predict_with_model(payload['kind'],trained['model'],payload['X_'+side])
        actual = metric.evaluate(payload['y_'+side],pred)
        stored = row[side+'_metrics']
        if not math.isclose(actual['score'], stored['score'], abs_tol=1e-12, rel_tol=0):
            raise ValueError('A referenced primary score differs after model reload')
        scores[side] = actual['score']
        if side=='search' and vector_hash(pred)!=prediction['prediction_sha256']:
            raise ValueError('Reloaded model changed the archived row-level predictions')
    return dict(trial=trial,candidate=row['candidate'],model_sha256=ab_run.sha(path),
                payload_sha256=prediction['evaluation_payload_sha256'],
                search_prediction_sha256=prediction['prediction_sha256'],scores=scores)


def audit_case(folder, entry, source, data, policy):
    run = folder/'run'
    caps,trusted = trusted_input(folder,entry,source)
    archived = record.read(run/'actions.json')
    if archived!=data['actions']:
        raise ValueError('Actions differ from published experiment')
    rows = []
    verified_trials = set()
    for step, actual in enumerate(archived,1):
        state = record.read(run/f'state_{step}.json')
        receipt = record.read(run/'bounded_recipes'/f'step_{step}.json')
        before = fingerprint(state)
        off,_ = choose(state,caps,trusted,False)
        on,proofs = choose(state,caps,trusted,True)
        archived_proofs = proofs if policy=='composed' else []
        if (receipt!=dict(step=step,state_sha256=before,policy=policy,action=actual,proposals=archived_proofs)
                or (on if policy=='composed' else off)!=actual or fingerprint(state)!=before):
            raise ValueError('Original six-field action, proof or state hash did not reproduce')
        item = dict(step=step,state_sha256=before,actual=key(actual),off=key(off),on=key(on),
                    changed=key(off)!=key(on),ablations=[])
        for proof in archived_proofs:
            for trial in proof['evidence_trials']:
                verified_trials.add(trial)
                old = state['prediction_evidence']
                reduced = dict(state,prediction_evidence=[r for r in old if r['trial']!=trial])
                if len(reduced['prediction_evidence'])!=len(old)-1:
                    raise ValueError('Evidence must refer to a unique completed trial')
                changed, remaining = choose(reduced,caps,trusted,True)
                if any(trial in p['evidence_trials'] for p in remaining):
                    raise ValueError('Composition used a trial without row-level evidence')
                damaged = deepcopy(state)
                row = next(r for r in damaged['prediction_evidence'] if r['trial']==trial)
                domain = tuple(state['task']['metric']['labels'])
                row['predictions'][0] = next(label for label in domain if label!=row['predictions'][0])
                refused,bad_proofs = choose(damaged,caps,trusted,True)
                if refused['action']!='stop' or 'reconciliation' not in refused['reason'] or bad_proofs:
                    raise ValueError('Inconsistent row prediction was not rejected')
                item['ablations'].append(dict(removed_reference=trial,next_action=key(changed),
                    next_action_changed=key(changed)!=key(on),proposal_survived=bool(remaining),
                    remaining_references=[p['evidence_trials'] for p in remaining],corruption_stopped=True))
        rows.append(item)
    observations = record.read(run/'history.json')
    predictions = record.read(run/'prediction_evidence.json')
    measured = [remeasure(run,t,observations,predictions) for t in sorted(verified_trials)]
    return dict(decisions=rows,model_rechecks=measured)


def audit(args):
    source_dir,out = args.previous.resolve(),args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()))
    os.environ.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
    published = record.ROOT/PUBLISHED
    for name in ('preflight','protocol','results','completion','all_six_models_frozen'):
        if ab_run.sha(source_dir/(name+'.json'))!=ab_run.sha(published/(name+'.json')):
            raise ValueError('Prior experiment differs from its committed data')
    proto, pre, result = (record.read(published/(name+'.json')) for name in ('protocol','preflight','results'))
    if record.digest(ab_run.sources())!=proto['source']:
        raise ValueError('Production source differs from the actual six runs')
    record.write(out/'protocol.json',dict(batch='B2-02',source=proto['source'],parent_results_sha256=ab_run.sha(published/'results.json'),
        plan_sha256=ab_run.sha(record.ROOT/'docs/batches/B2_02_replay_plan.md'),driver_sha256=ab_run.sha(Path(__file__)),
        paths=list(result['paths']),training_budget=0,model_api_budget=0,outer_metrics_budget=0))
    try:
        paths = {}
        with ExitStack() as stack:
            stack.enter_context(patch.object(LogisticRegression,'fit',side_effect=AssertionError('Reload cannot train')))
            for case,data in result['paths'].items():
                task,policy = case.split('_')
                registration = pre[task]
                menu = 'base' if policy=='original' else 'expanded'
                entry = dict(files={'train.zip':registration['train_sha256']},catalog_sha256=registration[menu+'_sha256'],
                             splits=registration['splits'])
                paths[case] = audit_case(source_dir/case,entry,proto['source'],data,policy)
                record.write(out/'checkpoint.json',paths)
        summaries = {}
        for name,case in paths.items():
            rows = case['decisions']
            ablations = [item for r in rows for item in r['ablations']]
            summaries[name] = dict(states=len(rows),same_state_changes=sum(r['changed'] for r in rows),
                removals=len(ablations),changed_after_removal=sum(a['next_action_changed'] for a in ablations),
                proposals_gone=sum(not a['proposal_survived'] for a in ablations),
                corruptions_rejected=sum(a['corruption_stopped'] for a in ablations),reloaded_models=len(case['model_rechecks']))
        if record.trace(out):
            raise ValueError('Unexpected supervised fitting')
        record.write(out/'results.json',dict(summary=summaries,paths=paths))
        totals = Counter()
        for row in summaries.values():
            totals.update(row)
        record.write(out/'completion.json',dict(status='completed',totals=dict(totals),new_training=0,new_model_api_requests=0,
            new_course_rmb=0,new_outer_grades=0,official_validation_test_or_H=False))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',reason=str(error),new_training=len(record.trace(out))))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous','weights','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    audit(parser.parse_args())
