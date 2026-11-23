"""Repeat the frozen early-exit rule with BGE and MiniLM-L6 as new candidates."""
import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_encoder_cascade import choose, evaluate, positive_labels, predict
from scripts.b_frozen_head_frontier import paired
from scripts.package_b_grouped_titles import masks
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.replay_b_encoder_agreement import predictions as original_predictions
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_cascade import bind_decision
from scripts.run_b_encoder_vote_followup import classes
from scripts.run_b_local_component_pair import check, read, save
from scripts.run_b_pair_surface_checks import pairs
from scripts.run_b_shared_title_weighting import group_ids
from superagent.ab_methods import TEXT, TEXT_DEEP, TRANSFER_METHODS
from superagent.ab_run import sha
from superagent.ab_sentence import fit_sentence
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint

PLAN=Path(__file__).resolve().parents[1]/'docs/batches/B_encoder_cascade_replication_1123_zh.md'
ORDER_AUDIT=PLAN.with_name('B_encoder_cascade_probability_audit_1123_zh.md')
BGE=TRANSFER_METHODS['deep-bge'][1]


def title_weights(a):
    original=a.studies/'alternative_encoders_01'
    follow=a.studies/'encoder_vote_followup_01'
    saved=a.studies/'replayed_vote_materialized_01'
    def trial(arm, part, name):
        folder=original/arm
        row,=[r for r in read(folder/'history.json') if r['candidate']==name and r['split_id']==part]
        return folder/('trial_'+str(row['trial'])+'.joblib')
    files={'primary':{TEXT_DEEP:trial('deep-bge','primary',TEXT_DEEP),BGE:trial('deep-bge','primary',BGE)},
           'confirmation_1':{TEXT_DEEP:trial('deep-e5','confirmation_1',TEXT_DEEP),BGE:follow/'deep-bge/worker.joblib'}}
    check(sha(files['confirmation_1'][BGE])==read(follow/'fits.json')['deep-bge']['model'], 'BGE confirmation changed')
    full={TEXT_DEEP:original/'deep-bge/selected_model.joblib',BGE:saved/'bge_full/selected_model.joblib'}
    check(sha(full[TEXT_DEEP])==read(original/'deep-bge_receipt.json')['model'] and
          sha(full[BGE])==read(saved/'models_frozen.json')['bge_full'],'Unbound full title models')
    return original, saved, files, full


def reconstruct_titles(original, inputs, files):
    task=load_development_task(inputs/'anonymous.zip')
    groups=group_ids(task.X_train)
    folds,details={},{}
    seen=set()
    for part,bound in files.items():
        directory=original/'deep-bge'
        source=directory/(part+'.joblib')
        frozen=read(directory/(part+'_split.json'))
        ids=np.array(frozen['search_indices'])
        rows=[task.X_train[i] for i in ids]
        y=np.asarray(task.y_train)[ids]
        tie=np.array([l<r for l,r in rows])
        payload=sha(source)
        registered=read(original/'protocol.json')['splits'][part]
        check(fingerprint(ids)==registered['row_ids_sha256']==frozen['row_ids_sha256'] and
              fingerprint(y)==registered['label_sha256'] and payload==registered['evaluation_payload_sha256'] and
              frozen['search_rows_sharing_text']==0,'Wrong reconstruction split')
        keep=~np.isin(groups[ids],list(seen))
        scores={}
        order_audit={}
        for name,worker in bound.items():
            loaded=joblib.load(worker)
            ref=loaded['prediction_record']
            historical,_=original_predictions(loaded['model'],rows)
            np.testing.assert_array_equal(historical,loaded['search_predictions'])
            p,_=evaluate(loaded['model'],rows)
            hard=positive_labels(p,tie)
            check(ref['evaluation_payload_sha256']==payload and ref['label_sha256']==fingerprint(y) and
                  ref['prediction_sha256']==fingerprint(historical),'Worker rows or labels changed')
            changed=hard!=historical
            order_audit[name]=dict(raw=fingerprint(historical),canonical=fingerprint(hard),changed=int(changed.sum()),
                max_distance_from_half=float(np.max(np.abs(p[changed]-.5))) if changed.any() else 0.)
            scores[name]=p[keep]
        folds[part]=dict(scores=scores,labels=y[keep],tie_left=tie[keep])
        details[part]=dict(all_rows=len(ids),used_rows=int(keep.sum()),used_groups=len(set(groups[ids][keep])),
                           used_row_order=fingerprint(ids[keep]),payload=payload,order_audit=order_audit)
        seen.update(groups[ids])
    return dict(policy=choose(folds['primary'],folds['confirmation_1']),primary_source=details['primary'],
                fresh_confirmation_source=details['confirmation_1'])


def full_l6(folder, task):
    model=folder/'model.joblib'
    start,done=folder/'fit_started.json',folder/'fit_completed.json'
    folder.mkdir(exist_ok=True)
    if not done.exists():
        check(not start.exists() and not model.exists(),'Unresolved physical fit')
        save(start,dict(candidate=TEXT,rows=len(task.X_train),fit_cap=1))
        fitted,_=fit_sentence(task.X_train,task.y_train,task.X_train[:1],'pair_text')
        with model.open('xb') as f:
            joblib.dump(fitted,f)
        save(done,dict(fits=1,model=sha(model),start=sha(start)))
    check(read(done)['model']==sha(model) and read(done)['start']==sha(start),'Different L6 model')
    return joblib.load(model)


def export(folder, models, decision, rows, reference_name, reference_file, reference_seal):
    folder.mkdir(exist_ok=True)
    routed,single,need,used=predict(models,rows,decision,decision['threshold'])
    full,first,_,always=predict(models,rows,decision,1.)
    np.testing.assert_array_equal(single,first)
    np.testing.assert_array_equal(routed,np.where(need,full,first))
    reference=classes(reference_file,reference_seal,len(rows))
    if reference_name=='full':
        anchor=full
    else:
        p,_=evaluate(models[reference_name],rows)
        anchor=positive_labels(p,np.array([l<r for l,r in rows]))
    np.testing.assert_array_equal(reference,anchor)
    output=folder/'predictions.npz'
    if not output.exists():
        with output.open('xb') as stream:
            np.savez_compressed(stream,selected=routed,full=full,first=single,reference=anchor)
    with np.load(output) as stored:
        for key,value in [('selected',routed),('full',full),('first',single),('reference',anchor)]:
            np.testing.assert_array_equal(stored[key],value)
    return dict(file=sha(output),selected_usage=used,control_usage=always,reference=reference_seal,
                reference_name=reference_name,reference_rows=len(rows))


def score(folder, exp, scoring, scoring_sha, groups_file, groups_sha):
    check(sha(scoring)==scoring_sha and sha(folder/'predictions.npz')==exp['file'],'Held rows changed')
    truth=read(scoring)
    y=np.array(truth['labels'])
    if groups_file:
        check(sha(groups_file)==groups_sha,'Held title groups changed')
        groups=np.array(read(groups_file))
    else:
        groups=np.array(truth['groups'])
    with np.load(folder/'predictions.npz') as z:
        predictions={k:z[k] for k in ('selected','full','first','reference')}
    hits={k:p==y for k,p in predictions.items()}
    comparisons={}
    for name in ('full','first','reference'):
        d=hits['selected'].astype(int)-hits[name].astype(int)
        comparisons[name]=dict(gained=int((d>0).sum()),lost=int((d<0).sum()),delta=paired(d,groups))
    frozen_save(folder/'grouped_correctness.json',masks(list(predictions),predictions,y,groups,['outer']*len(y)))
    saved=exp['selected_usage']['skips']
    return dict(rows=len(y),correct={k:int(v.sum()) for k,v in hits.items()},comparisons=comparisons,
                second_rows_skipped=saved,second_tokens_saved=exp['control_usage']['second']['input_tokens']-
                exp['selected_usage']['second']['input_tokens'],supports_followup=saved>=.1*len(y) and
                comparisons['full']['delta']['net_pp']>=-.5)


def run(a):
    p0=read(a.initial/'protocol.json')
    e0=read(a.initial/'completion.json')
    check(e0['results']==sha(a.initial/'results.json') and e0['protocol']==sha(a.initial/'protocol.json') and
          read(a.initial/'results.json')['eligible'] and p0['policy']==sha(Path(__file__).with_name('b_encoder_cascade.py')),
          'Initial gate or policy changed')
    original, saved, files, full=title_weights(a)
    tp, mp=read(a.titles/'protocol.json'),read(a.films/'protocol.json')
    run=a.films/'new'
    movie_model=run/'selected_model.joblib'
    signed=read(a.films/'decisions.json')['new']
    check(signed['selected']==[TEXT_DEEP] and signed['model']==sha(movie_model) and signed['fits']==10 and
          signed['physical']==sha(a.films/'new.physical.jsonl') and mp['outer_rows']==276 and
          mp['held_films']==95 and mp['outer_shared_films']==mp['outer_shared_texts']==0,'Movie split or model differs')
    check(sha(a.titles/'anonymous.zip')==tp['training_sha']==read(original/'deep-bge/run_manifest.json')['task_archive_sha256']
          and sha(a.titles/'features.zip')==tp['feature_sha'] and sha(a.films/'anonymous.zip')==mp['training'] and
          sha(a.films/'features.zip')==mp['features'],'Packet mismatch')
    a.out.mkdir(parents=True,exist_ok=True)
    frozen_save(a.out/'protocol.json',dict(plan=sha(PLAN),order_audit=sha(ORDER_AUDIT),runner=sha(Path(__file__)),policy=p0['policy'],
        initial=sha(a.initial/'completion.json'),titles={part:{n:sha(f) for n,f in ff.items()} for part,ff in files.items()},
        title_full={n:sha(f) for n,f in full.items()},title_packet=tp['training_sha'],title_features=tp['feature_sha'],
        film_packet=mp['training'],film_features=mp['features'],film_checkpoint=signed,film_history=sha(run/'prediction_evidence.json'),
        fit_cap=1,model_api_cap=0))
    os.environ.update(offline_environment(a.workspace))
    sys.addaudithook(offline)
    decisions={'titles':reconstruct_titles(original,a.titles,files),
               'films':bind_decision(a.films,{TEXT,TEXT_DEEP},run)}
    frozen_save(a.out/'decisions.json',decisions)
    title_models={n:joblib.load(file)['model'] for n,file in full.items()}
    film_models={TEXT_DEEP:joblib.load(movie_model)['model'],TEXT:full_l6(a.out/'film_l6',load_development_task(a.films/'anonymous.zip'))}
    exports={}
    for k,models,rows,reference,file,digest in [
        ('titles',title_models,pairs(a.titles/'features.zip'),'full',saved/'bge_vote_export_0/rows.csv',read(saved/'exports_frozen.json')['bge_vote']),
        ('films',film_models,pairs(a.films/'features.zip'),TEXT_DEEP,a.films/'new_export_0/rows.csv',read(a.films/'exports.json')['new'])]:
        exports[k]=export(a.out/k,models,decisions[k]['policy'],rows,reference,file,digest)
    frozen_save(a.out/'exports.json',exports)
    tasks={
        'titles':score(a.out/'titles',exports['titles'],a.titles/'scoring_only.json',tp['locked_scores'],a.titles/'groups_only.json',tp['locked_groups']),
        'films':score(a.out/'films',exports['films'],a.films/'scoring_only.json',mp['scoring'],None,None)}
    result=dict(tasks=tasks,eligible=all(v['supports_followup'] for v in tasks.values()),physical_fits=1,
        model_api_calls=0,new_rmb=0,course_validation_or_test=False)
    frozen_save(a.out/'results.json',result)
    frozen_save(a.out/'completion.json',dict(fits=1,model_api_calls=0,fit=sha(a.out/'film_l6/fit_completed.json'),
        **{k:sha(a.out/(k+'.json')) for k in ('results','protocol','decisions','exports')},
        groups={k:sha(a.out/k/'grouped_correctness.json') for k in tasks}))
    print(json.dumps({k:dict(correct=v['correct'],skips=v['second_rows_skipped']) for k,v in tasks.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','initial','studies','titles','films','out'):
        parser.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    run(parser.parse_args())
