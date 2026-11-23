"""Freeze two domain decisions from training records, execute the routed encoders, then score."""
import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_encoder_cascade import choose, positive_labels, predict
from scripts.b_frozen_head_frontier import paired
from scripts.package_b_grouped_titles import masks
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read
from scripts.run_b_pair_surface_checks import pairs
from scripts.run_b_shared_title_weighting import group_ids
from superagent.ab_methods import TEXT_DEEP, TEXT_FAMILIARITY, TRANSFER_METHODS
from superagent.ab_run import sha
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.score_evidence import checked_score_record

PLAN=Path(__file__).resolve().parents[1]/'docs/batches/B_encoder_cascade_1123_zh.md'
E5=TRANSFER_METHODS['deep-e5'][1]


def candidates(title_checkpoint, film_e5, film_l12):
    return {
        'titles':[(TEXT_FAMILIARITY,title_checkpoint),(E5,title_checkpoint)],
        'films':[(E5,film_e5),(TEXT_DEEP,film_l12)],
    }


def artifact(folder, name):
    selection=read(folder/'final_selection.json')
    file=folder/'selected_model.joblib'
    check(selection['model_sha256']==sha(file), 'Changed fitted model')
    members=selection['ensemble_components'] or [selection['family']]
    check(name in members and selection['refit'] and not selection['test_accessed'],'Unbound production component')
    return dict(model=sha(file),selection=sha(folder/'final_selection.json'),position=members.index(name),
                is_ensemble=bool(selection['ensemble_components']))


def load_models(arms, bound):
    models={}
    for name,folder in arms:
        ref=bound[name]
        check(artifact(folder,name)==ref,'Model registration differs')
        obj=joblib.load(folder/'selected_model.joblib')['model']
        model=obj.models[ref['position']] if ref['is_ensemble'] else obj
        feature=model.steps[0][1]
        expected={E5:'AlternativeSentenceFeatures',TEXT_DEEP:'DeepPreferenceFeatures',TEXT_FAMILIARITY:'FamiliarityPreferenceFeatures'}
        check(type(feature).__name__==expected[name] and (name!=E5 or feature.variant=='e5'),'Incorrect member order')
        models[name]=model
    return models


def training_fold(source, part, names, training, groups, omit):
    split=read(source/(part+'_split.json'))
    idx=np.asarray(split['search_indices'])
    rows=[training.X_train[i] for i in idx]
    y=np.asarray(training.y_train)[idx]
    tie=np.array([l<r for l,r in rows],dtype=bool)
    payload=sha(source/(part+'.joblib'))
    check(payload==split['evaluation_payload_sha256'] and fingerprint(idx)==split['row_ids_sha256'] and
          fingerprint(y)==split['label_sha256'], 'Fold differs from training input')
    check(split['search_rows_sharing_text']==0,'Text shared across fit and evaluation')
    records=[r for r in read(source/'prediction_evidence.json') if r['split_id']==part and r['candidate_id'] in names]
    check(len(records)==len(names)==len({r['candidate_id'] for r in records}), 'Missing probability evidence')
    keep=~np.isin(groups[idx],list(omit))
    check(keep.any(),'Empty fresh fold')
    scores={}
    for row in records:
        name=row['candidate_id']
        check(np.array_equal(row['row_ids'],idx) and row['evaluation_payload_sha256']==payload and
              row['prediction_sha256']==fingerprint(row['predictions']), 'Evidence row order differs')
        matrix,classes=checked_score_record(row['score_record'],row,MetricSpec('accuracy',(0,1)),name,y)
        p=matrix[:,classes.index(1)]
        np.testing.assert_array_equal(positive_labels(p,tie),row['predictions'])
        scores[name]=p[keep]
    detail=dict(all_rows=len(idx),used_rows=int(keep.sum()),used_groups=len(set(groups[idx][keep])),
                used_row_order=fingerprint(idx[keep]),payload=payload)
    return dict(scores=scores,tie_left=tie[keep],labels=y[keep]),detail,set(groups[idx])


def register(inputs, sources, evidence):
    setup={}
    for task, folder in inputs.items():
        local=read(folder/'protocol.json')
        packet=folder/'anonymous.zip'
        archive=sha(packet)
        feat=local['feature_sha'] if task=='titles' else local['features']
        check(sha(folder/'features.zip')==feat and archive==(local['training_sha'] if task=='titles' else local['train_packet']),
              'Research packet differs')
        for _,d in sources[task]:
            check(read(d/'run_manifest.json')['task_archive_sha256']==archive, 'Reference uses different training data')
        source=evidence[task]
        check(read(source/'run_manifest.json')['task_archive_sha256']==archive,'Decision source training differs')
        setup[task]=dict(training=archive,features=feat,scorer=local['locked_scores'] if task=='titles' else local['scoring'],
                         groups=local.get('locked_groups'), model_refs={n:artifact(d,n) for n,d in sources[task]},
                         history=sha(source/'history.json'),evidence=sha(source/'prediction_evidence.json'))
    return setup


def bind_decision(inputs, names, source):
    task=load_development_task(inputs/'anonymous.zip')
    check(task.X_val is None,'Training-only input required')
    groups=group_ids(task.X_train)
    primary,primary_ref,used=training_fold(source,'primary',names,task,groups,set())
    confirm,confirm_ref,_=training_fold(source,'confirmation_1',names,task,groups,used)
    return dict(policy=choose(primary,confirm),primary_source=primary_ref,fresh_confirmation_source=confirm_ref)


def export(folder, arms, identity, features, decision):
    folder.mkdir(exist_ok=True)
    check(sha(features)==identity['features'],'Outer features changed')
    rows=pairs(features)
    model=load_models(arms,identity['model_refs'])
    pred,base,needed,usage=predict(model,rows,decision,decision['threshold'])
    full,first,all_needed,dual_usage=predict(model,rows,decision,1.)
    np.testing.assert_array_equal(first,base)
    check(all_needed.all() and usage['second']['rows']==needed.sum() and usage['first']['rows']==len(rows),
          'Incorrect physical routing')
    check(np.array_equal(pred[~needed],base[~needed]) and np.array_equal(pred[needed],full[needed]),
          'Physical routing changed the hard prediction')
    file=folder/'predictions.npz'
    if not file.exists():
        with file.open('xb') as f:
            np.savez_compressed(f,selected=pred,first=base,full=full,lower_confidence=needed)
    with np.load(file) as data:
        for name,value in [('selected',pred),('first',base),('full',full),('lower_confidence',needed)]:
            np.testing.assert_array_equal(data[name],value)
    return dict(file=sha(file),selected=fingerprint(pred),always_dual=fingerprint(full),always_first=fingerprint(base),
                rows=len(rows),selected_usage=usage,control_usage=dual_usage)


def score(folder, inputs, identity, exp):
    scorer=inputs/'scoring_only.json'
    check(sha(scorer)==identity['scorer'] and sha(folder/'predictions.npz')==exp['file'],'Evaluation evidence differs')
    record=read(scorer)
    y=np.array(record['labels'])
    if identity['groups']:
        f=inputs/'groups_only.json'
        check(sha(f)==identity['groups'],'Outer group identities differ')
        groups=np.array(read(f))
    else:
        groups=np.array(record['components'])
    with np.load(folder/'predictions.npz') as data:
        predictions={k:data[k] for k in ('selected','full','first')}
    hits={k:pred==y for k,pred in predictions.items()}
    comparisons={}
    for baseline in ('first','full'):
        delta=hits['selected'].astype(int)-hits[baseline].astype(int)
        comparisons[baseline]=dict(gained=int((delta>0).sum()),lost=int((delta<0).sum()),delta=paired(delta,groups))
    public=masks(list(predictions),predictions,y,groups,np.repeat('outer',len(y)))
    frozen_save(folder/'grouped_correctness.json',public)
    savings=exp['control_usage']['second']['input_tokens']-exp['selected_usage']['second']['input_tokens']
    return dict(rows=len(y),correct={k:int(v.sum()) for k,v in hits.items()},comparisons=comparisons,
                second_rows_skipped=exp['selected_usage']['skips'],second_tokens_saved=savings,
                supports_followup=exp['selected_usage']['skips']>=.1*len(y) and
                comparisons['full']['delta']['net_pp']>=-.5)


def run(a):
    inputs={'titles':a.title_inputs,'films':a.film_inputs}
    arms=candidates(a.title_model,a.film_e5,a.film_l12)
    history={'titles':a.title_model,'films':a.film_history}
    a.out.mkdir(parents=True,exist_ok=True)
    refs=register(inputs,arms,history)
    frozen_save(a.out/'protocol.json',dict(plan=sha(PLAN),runner=sha(Path(__file__)),policy=sha(Path(__file__).with_name('b_encoder_cascade.py')),
                tasks=refs,fit_cap=0,model_api_cap=0))
    decisions={k:bind_decision(f,{name for name,_ in arms[k]},history[k]) for k,f in inputs.items()}
    frozen_save(a.out/'decisions.json',decisions)
    os.environ.update(offline_environment(a.workspace))
    sys.addaudithook(offline)
    exports={k:export(a.out/k,arms[k],refs[k],f/'features.zip',decisions[k]['policy']) for k,f in inputs.items()}
    frozen_save(a.out/'exports.json',exports)
    results={k:score(a.out/k,f,refs[k],exports[k]) for k,f in inputs.items()}
    result=dict(tasks=results,eligible=all(r['supports_followup'] for r in results.values()),physical_fits=0,model_api_calls=0,
                new_rmb=0,course_validation_or_test=False)
    frozen_save(a.out/'results.json',result)
    frozen_save(a.out/'completion.json',dict(results=sha(a.out/'results.json'),protocol=sha(a.out/'protocol.json'),
                decisions=sha(a.out/'decisions.json'),exports=sha(a.out/'exports.json'),fits=0,model_api_calls=0,
                groups={k:sha(a.out/k/'grouped_correctness.json') for k in inputs}))
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','title-inputs','film-inputs','title-model','film-e5','film-l12','film-history','out'):
        parser.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    run(parser.parse_args())
