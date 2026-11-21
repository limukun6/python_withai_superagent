"""Score a preregistered inference rule using only models already paid for on each path."""
import argparse
from contextlib import ExitStack
import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

from catboost import CatBoostClassifier
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts import run_b1_ab_acceptance as record
from scripts.replay_b1_observed_actions import trusted_input
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.ab_scheduler import selection_evidence
from superagent.data import load_development_task, load_prediction_features, load_training_task
from superagent.metrics import score_of
from superagent.models import probability_with_model
from superagent.scheduler_bridge import fingerprint
from superagent.splits import take
from tools.ab_registered_rows import content_digest


def pick(case, entry, sources, expected):
    run = case/'run'
    caps,trusted = trusted_input(case,entry,sources['representative'])
    history = record.read(run/'history.json')
    state = record.read(run/f'state_{len(history)+1}.json')
    selection = record.read(case/'relocated/final_selection.json')
    if (state['history']!=history or selection_evidence(state,trusted_partitions=trusted)['incumbent']!=selection['family']
            or ab_run.sha(case/'relocated'/selection['model_file'])!=expected or selection['model_sha256']!=expected):
        raise ValueError('Signed within-path winner or its training evidence changed')
    used = record.read(run/'classifier_fits.json')
    if not 1<used['attempted_classifier_fits']==used['completed_classifier_fits']<=11 or used['unknown_failed_attempts']:
        raise ValueError('Unverified path fit accounting')
    winner = selection['family']
    signed = {r['trial']:r for r in state['prediction_evidence']}
    ranked = sorted((r for r in history if r['trial'] in signed and r['status']=='completed' and r['split_id']=='primary'
                     and caps[r['candidate']]['route']!=caps[winner]['route'] and caps[r['candidate']]['supports_probabilities']),
                    key=lambda r:(-score_of(r),r['trial']))
    alternative = None
    if ranked:
        row = ranked[0]
        path = run/f"trial_{row['trial']}.joblib"
        if joblib.load(path)['metrics']['resolved_configuration']['id']!=row['candidate']:
            raise ValueError('Frozen trial does not implement the selected challenger')
        alternative = dict(candidate=row['candidate'],trial=row['trial'],route=caps[row['candidate']]['route'],
                           score=score_of(row),model_sha256=ab_run.sha(path),row_evidence_sha256=fingerprint(signed[row['trial']]))
    return dict(winner=winner,winner_file=selection['model_file'],winner_route=caps[winner]['route'],winner_sha256=expected,
                challenger=alternative,original_fit_attempts=used['attempted_classifier_fits'],
                alternative_trials_ranked=[r['trial'] for r in ranked],basis_state_sha256=fingerprint(state))


def predictions(kind, model, features):
    classes, values = probability_with_model(kind,model,features)
    classes = np.asarray(classes).reshape(-1)
    p = np.asarray(values)
    if (p.shape!=(len(features),2) or set(classes.tolist())!={0,1} or not np.isfinite(p).all()
            or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1.,rtol=0,atol=1e-8)):
        raise ValueError('Invalid binary class probabilities')
    return classes[p.argmax(axis=1)],p.max(axis=1)


def score(case,packet,entry,choice,task_source,prior):
    raw_ids = gzip.decompress((packet/'outer_ids.json.gz').read_bytes())
    if record.sha(raw_ids)!=entry['outer_ids_sha256'] or ab_run.sha(packet/'features.zip')!=entry['features_sha256']:
        raise ValueError('Held row IDs or inputs changed')
    ids = json.loads(raw_ids)['held']
    truth = task_source.y_train[ids]
    task = load_development_task(packet/'train.zip')
    features = load_prediction_features(packet/'features.zip',task.kind,task.spec)
    if content_digest(features,truth)!=entry['held_content_sha256']:
        raise ValueError('Held rows or labels changed')
    winner_path = case/'relocated'/choice['winner_file']
    if ab_run.sha(winner_path)!=choice['winner_sha256']:
        raise ValueError('Winner changed after freeze')
    primary,confidence = predictions(task.kind,joblib.load(winner_path)['model'],features)
    low = np.flatnonzero(confidence<.60)
    frozen_export = case/'export_0/rows.csv'
    if ab_run.sha(frozen_export)!=prior['csv_sha256'] or not np.array_equal(primary,np.asarray(read_export(frozen_export,len(ids)),dtype=truth.dtype)):
        raise ValueError('Frozen single-model predictions changed')
    changed = primary.copy()
    alternative = choice['challenger']
    challenger_all = None
    if alternative:
        path = case/'run'/f"trial_{alternative['trial']}.joblib"
        if ab_run.sha(path)!=alternative['model_sha256']:
            raise ValueError('Challenger changed after freeze')
        model = joblib.load(path)['model']
        challenger_all,full_confidence = predictions(task.kind,model,features)
        if len(low):
            challenger,confidence2 = predictions(task.kind,model,take(features,low))
            if not np.array_equal(challenger,challenger_all[low]) or not np.array_equal(confidence2>=.8,full_confidence[low]>=.8):
                raise ValueError('Sparse and dense challenger execution disagree on a decision')
            redirect = (challenger!=primary[low]) & (confidence2>=.80)
            changed[low[redirect]] = challenger[redirect]
    right = primary==truth
    switched = changed!=primary
    repairs = int((~right & (changed==truth)).sum())
    damage = int((right & (changed!=truth)).sum())
    if int(right.sum())!=prior['correct'] or int((changed==truth).sum())-int(right.sum())!=repairs-damage:
        raise ValueError('Paired performance accounting changed')
    return dict(rows=len(ids),original_correct=int(right.sum()),challenger_only_correct=int((challenger_all==truth).sum()) if alternative else None,
        gated_correct=int((changed==truth).sum()),extra_challenger_prediction_rows=len(low) if alternative else 0,
        switched=int(switched.sum()),repaired=repairs,harmed=damage,accuracy_delta_pp=100*(repairs-damage)/len(ids),
        gated_predictions_sha256=fingerprint(changed.tolist()),original_predictions_sha256=fingerprint(primary.tolist()),
        sparse_predictions_match_dense=True)


def study(args):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()))
    os.environ.pop('SUPERAGENT_MINILM_CACHE_DIR', None)
    src = record.ROOT/'research'
    setup = {'coupon':(args.coupons.resolve(),args.coupon_source.resolve(),'b_route_coupons'),
             'headlines':(args.headlines.resolve(),args.headline_source.resolve(),'b_route_titles')}
    proto = dict(batch='B-RELEASED-CONFIDENCE-02',plan_sha256=ab_run.sha(record.ROOT/'docs/batches/B_RELEASED_CONFIDENCE_02_plan.md'),
                 driver_sha256=ab_run.sha(Path(__file__)),thresholds=dict(winner_below=.6,challenger_at_least=.8),
                 source_comparisons={k:ab_run.sha(src/r/'results.json') for k,(_,_,r) in setup.items()},
                 seeds=[45101,45102],new_training_limit=0,model_api_limit=0)
    record.write(out/'protocol.json',proto)
    try:
        selections,inputs,grades = {},{},{}
        with ExitStack() as guard:
            for cls in (LogisticRegression,CatBoostClassifier):
                guard.enter_context(patch.object(cls,'fit',side_effect=AssertionError('Frozen predictor cannot train')))
            for name,(origin,staged,collection) in setup.items():
                headlines = name=='headlines'
                frozen = record.read(src/collection/'all_models_frozen.json')
                protocol = record.read(src/collection/'protocol.json')
                packet_root = args.headline_packets.resolve() if headlines else origin
                expected_pre = protocol['registered_inputs_sha256'] if headlines else ab_run.sha(src/collection/'preflight.json')
                if ab_run.sha(packet_root/'preflight.json')!=expected_pre:
                    raise ValueError('Trusted registration changed')
                pre = record.read(packet_root/'preflight.json')
                for seed in ('45101','45102'):
                    key = name+'_'+seed
                    packet = packet_root/seed
                    case = origin/(seed if headlines else seed+'_representative')
                    training = pre['entries'][seed+'_original' if headlines else seed]
                    entry = dict(training,**{k:pre['tasks'][seed][k] for k in ('outer_ids_sha256','held_content_sha256')}) if headlines else training
                    expected = frozen[seed]['representative'] if headlines else frozen[seed+'_representative']
                    selections[key] = pick(case,training,protocol['sources'],expected)
                    inputs[key] = (case,packet,entry)
            record.write(out/'all_models_and_choices_frozen.json',selections)
            if record.read(out/'all_models_and_choices_frozen.json')!=selections:
                raise ValueError('Frozen choices changed')
            registered = record.read(src/'ba19_released_delivery/preflight.json')
            for name,(origin,staged,collection) in setup.items():
                if ab_run.sha(staged)!=registered[name]['source_staged_sha256']:
                    raise ValueError('Expected training-only original archive')
                task = load_training_task(staged)
                baseline = record.read(src/collection/'results.json')['pairs']
                for seed in ('45101','45102'):
                    key = name+'_'+seed
                    grades[key] = score(*inputs[key],selections[key],task,baseline[seed]['grade']['representative'])
        record.write(out/'results.json',grades)
        record.write(out/'completion.json',dict(status='completed',paths=len(grades),new_fits=0,new_model_api_requests=0,
                    new_course_rmb=0,official_validation_test_or_H=False))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',reason=str(error),new_fits=len(record.trace(out))))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('coupons','headlines','coupon-source','headline-source','headline-packets','weights','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    study(parser.parse_args())
