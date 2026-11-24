"""Frozen goal variants through the official CLI; at most twelve supervised fits."""
import argparse
from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a122_current_agent_acceptance import environment,child,physical

ROOT=Path(__file__).resolve().parents[1]
CASES=[
 dict(id='accuracy',metric='accuracy',labels='binary',valid=True),
 dict(id='renamed',metric='accuracy',labels='renamed',valid=True),
 dict(id='macro',metric='macro_f1',labels='multi',valid=True),
 dict(id='f1_pos0',metric='binary_f1',positive=0,valid=True),
 dict(id='f1_pos1',metric='binary_f1',positive=1,valid=True),
 dict(id='auc_pos0',metric='roc_auc',positive=0,valid=True),
 dict(id='auc_default',metric='roc_auc',valid=True,positive_unknown=True),
 dict(id='metric_default',valid=True,metric_unknown=True),
 dict(id='invalid_positive',metric='binary_f1',positive=9,valid=False),
 dict(id='invalid_metric',metric='precision_magic',valid=False),
 dict(id='one_class',metric='accuracy',labels='one',valid=False),
 dict(id='preference',pair='preference',one='left',valid=True),
 dict(id='symmetric',pair='symmetric',one='equivalent',valid=True),
 dict(id='symmetric_wrong',pair='symmetric',one='left',valid=False),
 dict(id='preference_wrong',pair='preference',one='right',valid=False),
 dict(id='pair_undeclared',pair=None,one='left',valid=True,pair_unknown=True),
 dict(id='pair_invalid',pair='causal_magic',one='left',valid=False),
 dict(id='pair_renamed_columns',pair='preference',one='left',columns=['alpha_text','beta_text'],valid=True),
]


def packet(path,case,x,y,query=False):
    split='test' if query else 'train'
    manifest=dict(schema_version=1,objective='classification',modality='tabular',
                  splits={split:dict(format='npz',path='rows.npz')})
    if 'metric' in case:manifest['metric']=case['metric']
    if 'positive' in case:manifest['positive_label']=case['positive']
    if 'pair' in case:
        columns=case.get('columns',['left','right'])
        manifest.update(modality='pair_text',metric='accuracy',pair_label_one=case['one'],
                        columns=[dict(name=n,type='text') for n in columns],target='label')
        if case['pair'] is not None:manifest['pair_semantics']=case['pair']
        manifest['splits'][split]=dict(format='csv',path='pairs.csv')
        buf=io.StringIO();writer=csv.writer(buf);writer.writerow(columns+([] if query else ['label']))
        for i in range(40):writer.writerow([f'first unique paragraph {i}',f'second unique paragraph {i}']+([] if query else [i%2]))
        body=buf.getvalue().encode();member='pairs.csv'
    else:
        labels=np.where(y==0,'zulu','alpha') if case.get('labels')=='renamed' else (
               np.arange(len(y))%3 if case.get('labels')=='multi' else np.zeros(len(y),dtype=int) if case.get('labels')=='one' else y)
        buf=io.BytesIO();np.savez(buf,**({'X':x} if query else {'X':x,'y':labels}));body=buf.getvalue();member='rows.npz'
    with zipfile.ZipFile(path,'x') as z:
        z.writestr('manifest.json',json.dumps(manifest));z.writestr(member,body)
        if not query:z.writestr('sealed_test.bin',b'Never open during task understanding or development')


def preflight(out,inputs):
    from scripts import a38_predict_guard
    from superagent import ab_run
    from superagent.cli import build_parser
    records=[]
    class StopBeforeTraining(RuntimeError):pass
    for case in read(inputs/'cases.json'):
        reached=[];folder=out/case['id'];folder.mkdir()
        def stop(*args,**kw):reached.append(True);raise StopBeforeTraining('Registered observation before all fits')
        error=None
        try:
            args=build_parser().parse_args(['run',str(inputs/(case['id']+'.zip')),'--run-dir',str(folder/'run'),
                  '--planner','local','--scheduler','ab-v1','--portfolio','ab-v1','--fit-limit','3','--seed','707'])
            with patch.object(ab_run,'Executor',side_effect=stop),redirect_stdout(io.StringIO()):args.handler(args)
        except ValueError as e:error=str(e)
        accepted=bool(reached);passed=accepted==case['valid'];details={}
        manifest=folder/'run/run_manifest.json'
        if accepted:
            m=read(manifest);spec=m['task_contract'];fields=spec['goal_resolution']['fields'];metric=case.get('metric','accuracy')
            if 'pair' in case:metric='accuracy'
            passed &= spec['metric']['name']==metric
            if 'positive' in case:passed &= spec['metric']['labels'][-1]==case['positive'] and fields['positive_label']['status']=='confirmed'
            if case.get('positive_unknown'):passed &= fields['positive_label']['status']=='unknown'
            if case.get('metric_unknown'):passed &= fields['primary_metric']['status']=='unknown'
            if case.get('pair_unknown'):passed &= fields['pair_semantics']['status']=='unknown'
            if case.get('pair') in ('preference','symmetric'):
                expected='preserves' if case['pair']=='symmetric' else 'flips'
                passed &= all(v['swap_label_rule']==expected for v in m['capabilities'].values())
            if case.get('labels')=='renamed':passed &= set(spec['metric']['labels'])=={'alpha','zulu'}
            details=dict(task_contract=spec,candidate_count=len(m['capabilities']))
        elif manifest.is_file():error=read(folder/'run/run_outcome.json').get('error')
        records.append(dict(id=case['id'],expected_accepted=case['valid'],reached_executor_before_fits=accepted,
                            passed=bool(passed),error=error,**details))
    save(out/'results.json',dict(cases=records,passed=sum(r['passed'] for r in records),total=len(records)))
    return dict(status='passed' if all(r['passed'] for r in records) else 'findings',actual_classifier_fits=0,real_API_requests=0)


def prepare(out):
    rng=np.random.default_rng(13824);x=rng.normal(size=(160,3));y=(x[:,0]+.4*x[:,1]>.2).astype(int)
    query=rng.normal(size=(32,3));save(out/'cases.json',CASES)
    for case in CASES:packet(out/(case['id']+'.zip'),case,x,y)
    for name in ('accuracy','renamed','f1_pos0','auc_pos0'):
        packet(out/(name+'_query.zip'),next(c for c in CASES if c['id']==name),query,np.zeros(len(query)),True)
    from superagent.data import load_training_task
    from superagent.splits import grouped_partition
    from superagent.prediction_evidence import fingerprint
    from tools.ab_registered_rows import content_digest
    baseline=load_training_task(out/'accuracy.zip')
    partitions=[(seed,*grouped_partition(baseline,seed,'shared_text')[:2]) for seed in (707,808,909)]
    for name in ('accuracy','renamed','f1_pos0','auc_pos0'):
        task=load_training_task(out/(name+'.zip'));splits=[]
        for seed,fit,held in partitions:
            splits.append(dict(seed=seed,train_indices=fit.tolist(),search_indices=held.tolist(),
                               row_ids_sha256=fingerprint(held.tolist()),label_sha256=fingerprint(task.y_train[held])))
        save(out/(name+'_rows.json'),dict(version=1,kind=task.kind,grouping='exact_rows',
             training_content_sha256=content_digest(task.X_train,task.y_train),splits=splits))
    save(out/'inputs.json',{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.zip','.json') and p.name!='protocol.json'})
    checks=out/'preflight';checks.mkdir();return preflight(checks,out)


def execute(out,inputs):
    for n,h in read(inputs/'inputs.json').items():
        if sha(inputs/n)!=h:raise ValueError('Frozen case bytes changed: '+n)
    check=out/'preflight';check.mkdir();result=preflight(check,inputs)
    if result['status']!='passed':return result
    # Restore parent fit permission by launching supervised workers in separate processes only.
    env=environment(out);results=[]
    for name in ('accuracy','renamed','f1_pos0','auc_pos0'):
        folder=out/name;folder.mkdir();run=folder/'run';rows=inputs/(name+'_rows.json')
        local={**env,'AB_A29_RUN_NAME':name}
        command=[sys.executable,'-m','tools.ab_registered_run','--split-manifest',str(rows),'--split-sha256',sha(rows),'--',
            'run',str(inputs/(name+'.zip')),'--run-dir',str(run),'--planner','local','--scheduler','ab-v1',
            '--portfolio','ab-v1','--catalog-scope','full','--fit-limit','3','--seed','707','--trial-timeout','60','--time-limit','180']
        save(folder/'command.json',command);timing=child(command,folder/'train.log',local,240)
        outcome=read(run/'run_outcome.json');selection=read(run/'final_selection.json');manifest=read(run/'run_manifest.json')
        fits=[r for r in physical(out) if r['run_name']==name];metric=manifest['task_contract']['metric']
        if not outcome['requested_planner_completed'] or outcome['new_api_requests'] or not 2<=len(fits)<=3:
            raise ValueError('Incomplete bounded goal path: '+name)
        if outcome['classifier_fit_accounting']['attempted_classifier_fits']!=len(fits):raise ValueError('Physical fits differ')
        for h in read(run/'history.json'):
            if h['metric_spec']!=metric or h['primary_metric']!=metric['name']:raise ValueError('Worker objective drift')
        query=inputs/(name+'_query.zip')
        child([sys.executable,'-m','scripts.a38_predict_guard',str(query),'--run-dir',str(run),'--output',str(folder/'native.csv')],folder/'native.log',local)
        child([sys.executable,'-m','tools.export_model_bundle','--run-dir',str(run),'--output',str(folder/'portable')],folder/'bundle.log',local)
        child([sys.executable,'-I',str(folder/'portable/bundle/predict.py'),'--features',str(query),'--output',str(folder/'isolated.csv')],folder/'isolated.log',local)
        if (folder/'native.csv').read_bytes()!=(folder/'isolated.csv').read_bytes():raise ValueError('Isolated predictions differ')
        prediction_manifest=read(folder/'prediction_manifest.json')
        if metric['name']=='roc_auc' and (prediction_manifest['score_class_order']!=[0,1] or metric['labels']!=[1,0]):
            raise ValueError('Declared positive-class order was not kept distinct from probability order')
        results.append(dict(case=name,fits=len(fits),training=timing,metric=metric,selected=selection['family'],
                            isolated_prediction_identical=True,rows=32,score_class_order=prediction_manifest['score_class_order']))
        save(out/'checkpoint.json',results);print(json.dumps(results[-1]),flush=True)
    def labels(name):
        with (out/name/'native.csv').open(encoding='utf-8',newline='') as f:return [r['prediction'] for r in csv.DictReader(f)]
    renamed=[{'zulu':'0','alpha':'1'}[p] for p in labels('renamed')]
    same=renamed==labels('accuracy')
    save(out/'results.json',dict(paths=results,label_rename_mapped_predictions_identical=same,
         same_registered_row_ids=True,query_labels_scored=False,scope='Goal metamorphic engineering, not real-task transfer'))
    return dict(status='passed' if same else 'findings',actual_classifier_fits=len(physical(out)),real_API_requests=0,
                A_cumulative_fits_after=1602+len(physical(out)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','execute'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs',type=Path);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();source=hashes(ROOT)
    os.environ.update(environment(out))
    save(out/'protocol.json',dict(batch='A138-'+a.phase,source_files=source,runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A138_A140_goal_and_feedback_plan_20261124_zh.md'),
         expected_cases=18,maximum_fits=12 if a.phase=='execute' else 0,maximum_API_requests=0,starting_A_fits=1602))
    try:
        end=prepare(out) if a.phase=='prepare' else execute(out,a.inputs.resolve())
        end.update(elapsed_seconds=time.monotonic()-started,source_unchanged=source==hashes(ROOT))
        if (out/'results.json').exists():end['results_sha256']=sha(out/'results.json')
        end['protocol_sha256']=sha(out/'protocol.json');save(out/'completion.json',end);print(json.dumps(end),flush=True)
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=len(physical(out)),
             real_API_requests=0,A_cumulative_fits_after=1602+len(physical(out)),elapsed_seconds=time.monotonic()-started));raise


if __name__=='__main__':main()
