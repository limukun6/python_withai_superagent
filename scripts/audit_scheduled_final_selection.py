"""Compare the interim checkpoint, selected family before refitting, and actual delivery without fitting."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from scripts.run_conditional_review_study import get_predictions
from scripts.run_scheduled_review_study import FRESH, reload_pointer
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from scripts.verify_conditional_study_receipts import score_from_counts
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.models import SoftVoteModel, predict_with_model
from superagent.predict import create_predictions


def freeze(root):
    records={}
    for case in FRESH:
        folder=root/case
        agent=folder/'control/run/agent'
        output=root/'final_audit'/case
        query=folder/'input.zip'
        interim=reload_pointer(output/'interim',agent,'verified_selection.json',query)
        selection=read(agent/'final_selection.json')
        decision=read(agent/'selection.json')['choice']
        require(selection['refit'] and (selection['ensemble_components'] or [selection['single_candidate']])==decision['members'],
                'Archived selection does not identify the final members')
        history=read(agent/'history.json')
        task=load_training_task(folder/'train.zip',read_declarations=True)
        metric=MetricSpec.for_task(task)
        models=[]
        primary=agent/'primary.joblib'
        require(sha(primary)==read(agent/'primary_split.json')['payload_sha256'],'Primary data changed')
        data=joblib.load(primary)
        for name in decision['members']:
            row=next(r for r in history if r['candidate']==name and r['split_id']=='primary' and r['status']=='completed')
            trial=joblib.load(agent/f"trial_{row['trial']}.joblib")
            require(trial['prediction_record']['evaluation_payload_sha256']==sha(primary)
                    and trial['metrics']['resolved_configuration']['id']==name,'Incompatible archived component')
            models.append(trial['model'])
        model=models[0] if len(models)==1 else SoftVoteModel(task.kind,models,np.asarray(metric.labels))
        actual=metric.evaluate(data['y_search'],predict_with_model(task.kind,model,data['X_search']))['score']
        require(abs(actual-decision['evidence']['scores']['primary'])<1e-9,'Rebuilt pre-refit selection does not match the decision')
        work=output/'selected_primary'
        work.mkdir(parents=True,exist_ok=True)
        if not (work/'task_spec.json').exists():
            os.link(agent/'task_spec.json',work/'task_spec.json')
            artifact=work/'selected_primary.joblib'
            joblib.dump(dict(kind=task.kind,candidate=selection['family'],model=model),artifact)
            record=dict(selection,model_file=artifact.name,model_sha256=sha(artifact),refit=False,delivery_status='partial',
                        training_rows=len(data['y_train']),partial_reason='Reconstruction using the original primary-fitted members.')
            save(work/'final_selection.json',record)
        selected=work/'predictions.csv'
        create_predictions(str(query),work,selected)
        final=folder/'control/delivery/predictions.csv'
        require(sha(final)==read(folder/'control/sealed.json')['prediction'],'Changed final output')
        records[case]=dict(primary_rows=len(data['y_train']),full_rows=len(task.y_train),members=decision['members'],
                           checkpoint=read(agent/'verified_selection.json')['single_candidate'],
                           outputs={k:dict(file=str(p.relative_to(root)),digest=sha(p)) for k,p in
                                    [('checkpoint',interim),('selected_primary',selected),('delivered',final)]})
    save(root/'final_audit/frozen.json',records)


def paired(y,old,new):
    change=(new==y).astype(int)-(old==y).astype(int)
    rng=np.random.default_rng(261176)
    draws=[rng.choice(change[y==label],size=(5000,int(sum(y==label))),replace=True).mean(axis=1) for label in np.unique(y)]
    ci=np.quantile(np.mean(draws,axis=0),[.025,.975])
    return dict(net_balanced_accuracy=score_from_counts(y,new,'balanced_accuracy')-score_from_counts(y,old,'balanced_accuracy'),
                changed=int(np.sum(old!=new)),corrected=int(sum(change>0)),spoiled=int(sum(change<0)),paired_bootstrap_95=ci.tolist())


def evaluate(root,output):
    summary={}
    for case,record in read(root/'final_audit/frozen.json').items():
        truth=root/'evaluation'/(case+'.npy')
        require(sha(truth)==read(root/'protocol.json')['inputs'][case]['truth_sha'],'Changed development labels')
        y=np.load(truth,allow_pickle=False)
        pred={k:get_predictions(root/r['file'],y,r['digest']) for k,r in record['outputs'].items()}
        agent=root/case/'control/run/agent'
        checkpoint={r['split_id']:r['search_score'] for r in read(agent/'history.json')
                    if r['status']=='completed' and r['candidate']==record['checkpoint']}
        scores=read(agent/'selection.json')['choice']['evidence']['scores']
        matched={fold:dict(checkpoint=checkpoint[fold],selected_family=value,delta=value-checkpoint[fold],
                           rows=len(read(agent/(fold+'_split.json'))['held']))
                 for fold,value in scores.items() if fold in checkpoint}
        summary[case]=dict(checkpoint=record['checkpoint'],selected_members=record['members'],
                           matched_training_folds=matched,
                           training_rows=[record['primary_rows'],record['full_rows']],
                           scores={k:score_from_counts(y,v,'balanced_accuracy') for k,v in pred.items()},
                           selection=paired(y,pred['checkpoint'],pred['selected_primary']),
                           refit=paired(y,pred['selected_primary'],pred['delivered']),
                           total=paired(y,pred['checkpoint'],pred['delivered']))
    save(output,dict(status='Retrospective decomposition on the two already graded fresh tasks. Informational only.',
                     method='The identical final method is rebuilt from its primary-fitted components, then compared with full refitting. Paired class-stratified bootstrap, 5000 replicates, seed 261176.',
                     tasks=summary,new_fits=0,new_api_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=lambda x:Path(x).resolve())
    p.add_argument('output',type=lambda x:Path(x).resolve())
    p.add_argument('--weights',type=lambda x:Path(x).resolve(),required=True)
    a=p.parse_args()
    if os.getenv('AB_B1_OFFLINE')=='1':
        freeze(a.root)
    else:
        fits=a.root/'unexpected_selected_replay_fits.jsonl'
        subprocess.run([sys.executable,'-m','scripts.audit_scheduled_final_selection',*sys.argv[1:]],cwd=ROOT,env=env(a.weights,fits),check=True)
        require(not fits.exists() or not fits.read_text().strip(),'Unexpected fit during selection reconstruction')
        evaluate(a.root,a.output)
