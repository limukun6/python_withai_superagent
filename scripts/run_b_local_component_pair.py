"""Freeze, run once and score the two authorized local arms without an API."""
import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np

from superagent import ab_run
from superagent.cli import build_parser
from superagent.component_preferences import ComponentModel, sources, worker_env
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.splits import grouped_partition,sample_groups


ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/batches/B_local_components_pair_1123_zh.md'
ARCHIVE_SHA='56b5439ea3f77c7b933d9c0bc49705f3759b3a1d302da49adee80031904834e4'
MEMBER='Released_Task2-Headlines_CTR/train.csv'
SHAPE=dict(modality='pair_text',columns=[dict(name='option_a',type='text'),dict(name='option_b',type='text')],
           target='outcome',pair_semantics='preference',pair_label_one='left')


def check(v,message):
    if not v:
        raise ValueError(message)


def read(f):
    return json.loads(f.read_text())


def save(f,v):
    with f.open('x') as out:
        json.dump(v,out,indent=2)


def packets(rows,ids,train,target):
    split='train' if train else 'test'
    manifest=dict(schema_version=1,objective='classification',metric='accuracy',
                  splits={split:dict(format='csv',path='rows.csv')},**SHAPE)
    buf=io.StringIO(newline='')
    wr=csv.writer(buf)
    wr.writerow(['option_a','option_b']+(['outcome'] if train else []))
    for i in ids:
        wr.writerow(rows[i] if train else rows[i][:2])
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for k,v in [('manifest.json',json.dumps(manifest)),('rows.csv',buf.getvalue())]:
            item=zipfile.ZipInfo(k,date_time=(2020,1,1,0,0,0))
            item.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(item,v.encode())


def interfaces():
    return dict(source=ab_run.sources(),worker_and_legacy=sources(),runner=ab_run.sha(Path(__file__)),
                plan=ab_run.sha(PLAN),guard=ab_run.sha(ROOT/'scripts/ab_acceptance_guard/sitecustomize.py'))


def argv(out,arm):
    options=['run',str(out/'anonymous.zip'),'--scheduler','ab-v1','--portfolio','ab-v1',
          '--planner','local','--seed','707','--fit-limit','11','--time-limit','1200','--trial-timeout','300',
          '--run-dir',str(out/arm)]
    return options+(['--preference-components'] if arm=='components' else [])


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False)
    check(ab_run.sha(a.archive)==ARCHIVE_SHA,'Wrong released training source')
    with zipfile.ZipFile(a.archive) as z,z.open(MEMBER) as f:
        rows=[(j['headline_A'],j['headline_B'],int(j['label_pairwise'])) for j in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline=''))]
    ref=read(a.split)
    check(ab_run.sha(a.split)=='41eacfd2419f54f74ad0dfd74962a9ffc9cfc4f3f1bc48d2644c22df39fac0a2','Wrong old split reference')
    ids=sorted(i for part in ref['indices'] for i in part['outer'])
    reserved=sorted(set(range(len(rows)))-set(ids))
    check(len(set(ids))==len(ids)==6608 and len(reserved)==1693,'Wrong internal reserve membership')
    packets(rows,ids,True,a.out/'anonymous.zip')
    packets(rows,reserved,False,a.out/'features.zip')
    save(a.out/'scoring_only.json',dict(labels=[rows[i][2] for i in reserved],original_rows=reserved))
    task=load_development_task(a.out/'anonymous.zip')
    check(len(task.y_train)==6608 and task.X_val is None and task.spec['input_schema']['pair_semantics']=='preference','Wrong training contract')
    from types import SimpleNamespace
    all_groups=sample_groups(SimpleNamespace(kind='pair_text',X_train=[r[:2] for r in rows]),'shared_text')
    check(not set(all_groups[ids])&set(all_groups[reserved]),'Outer title groups overlap')
    save(a.out/'groups_only.json',list(all_groups[reserved]))
    with tempfile.TemporaryDirectory() as t:
        ab_run.freeze_partitions(task,Path(t),707,MetricSpec.for_task(task))
        primary=read(Path(t)/'primary_split.json')
        psha=ab_run.sha(Path(t)/'primary.joblib')
    caps=ab_run.registered_capabilities(TaskSpec.from_task(task).record(),len(ids),'full')
    check('minilm_text_with_lexical_features' in caps and len(caps)>=30,'Encoder or full catalog unavailable')
    for arm in ('original','components'):
        options=build_parser().parse_args(argv(a.out,arm))
        check(ab_run.validate_options(options)==11 and options.planner=='local','Wrong local budget')
    subprocess.run([sys.executable,str(ROOT/'scripts/b_component_worker.py'),'preflight'],check=True,env=worker_env())
    save(a.out/'protocol.json',dict(interfaces=interfaces(),training_sha=ab_run.sha(a.out/'anonymous.zip'),
         feature_sha=ab_run.sha(a.out/'features.zip'),locked_scores=ab_run.sha(a.out/'scoring_only.json'),
         locked_groups=ab_run.sha(a.out/'groups_only.json'),primary_sha=psha,
         primary_split_row_sha=primary['row_ids_sha256'],full_catalog_count=len(caps),
         archive_member=MEMBER,rows=len(ids),holdout=len(reserved),outer_title_groups_disjoint=True,
         order=['original','components'],allowed_actual_fits=[11,11],model_api_calls_allowed=0))


def command(cmd,log,env,timeout):
    start=time.monotonic()
    with log.open('x') as f:
        result=subprocess.run([sys.executable,'-m','superagent',*cmd],cwd=ROOT,
              env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    return result.returncode,time.monotonic()-start


def check_protocol(out):
    pr=read(out/'protocol.json')
    check(pr['interfaces']==interfaces() and pr['training_sha']==ab_run.sha(out/'anonymous.zip')
          and pr['feature_sha']==ab_run.sha(out/'features.zip'),'Source or inputs changed')
    return pr


def train_once(out):
    pr=check_protocol(out)
    save(out/'training_started.json',dict(pid=os.getpid(),order=pr['order']))
    runs={}
    for arm in pr['order']:
        check(not (out/arm).exists(),'Refusing to restart a path')
        env=dict(worker_env(),PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/ab_acceptance_guard',ROOT))),
                 AB_B1_FIT_LOG=str(out/(arm+'.physical.jsonl')),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',
                 MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        status,elapsed=command(argv(out,arm),out/(arm+'.log'),env,1260)
        folder=out/arm
        outcome=read(folder/'run_outcome.json') if (folder/'run_outcome.json').is_file() else {}
        if arm=='original':
            observed=(out/(arm+'.physical.jsonl'))
            count=len(observed.read_text().splitlines()) if observed.exists() else 0
            if status==0:
                account=read(folder/'classifier_fits.json')
                check(count==account['attempted_classifier_fits']==account['completed_classifier_fits'],'Original physical counters differ')
        else:
            events=[json.loads(v) for v in (folder/'component_fits.jsonl').read_text().splitlines()] if (folder/'component_fits.jsonl').is_file() else []
            count=sum(v['event']=='attempt' for v in events)
            if status==0:
                check(count==sum(v['event']=='physical_fit' for v in events)==outcome['classifier_fits'],'Component fit counters differ')
        check(count<=11,'Per-path fitting cap exceeded')
        runs[arm]=dict(exit=status,elapsed_seconds=elapsed,attempted=count,
                 completed_run=bool(status==0 and outcome.get('requested_planner_completed')),
                 selected=outcome.get('selected_candidate'),api_calls=outcome.get('new_api_requests'))
        if runs[arm]['completed_run']:
            check(outcome['new_api_requests']==0 and ab_run.sha(folder/'primary.joblib')==pr['primary_sha'],'Unequal split or API call')
            runs[arm]['model']=ab_run.sha(folder/'selected_model.joblib')
            runs[arm]['selection']=ab_run.sha(folder/'final_selection.json')
        save(out/(arm+'_receipt.json'),runs[arm])
        print(arm,runs[arm],flush=True)
    save(out/'models_frozen.json',runs)


def exports(out):
    pr=check_protocol(out)
    runs=read(out/'models_frozen.json')
    evidence={}
    for arm,r in runs.items():
        check(r['completed_run'] and ab_run.sha(out/arm/'selected_model.joblib')==r['model'],'Frozen path did not complete')
        paths=[]
        for i in range(2):
            directory=out/(arm+'_predict_'+str(i))
            directory.mkdir(exist_ok=False)
            dest=directory/'rows.csv'
            status,_=command(['predict',str(out/'features.zip'),'--run-dir',str(out/arm),'--output',str(dest)],
                    directory/'predict.log',worker_env(),600)
            check(status==0 and not read(directory/'prediction_manifest.json')['test_labels_read'],'Delivery failure')
            paths.append(ab_run.sha(dest))
        check(paths[0]==paths[1],'Independent predictions differ')
        evidence[arm]=dict(model=r['model'],predictions=paths[0],reloads=2)
    save(out/'both_predictions_frozen.json',dict(arms=evidence,features=pr['feature_sha']))


def score(out):
    pr=check_protocol(out)
    runs=read(out/'models_frozen.json')
    evidence=read(out/'both_predictions_frozen.json')['arms']
    check(ab_run.sha(out/'scoring_only.json')==pr['locked_scores'] and ab_run.sha(out/'groups_only.json')==pr['locked_groups'],'Scoring source changed')
    y=np.array(read(out/'scoring_only.json')['labels'])
    hits={}
    for arm in pr['order']:
        f=out/(arm+'_predict_0')/'rows.csv'
        check(ab_run.sha(f)==evidence[arm]['predictions'],'Unsealed predictions')
        with f.open() as fd:
            rows=list(csv.DictReader(fd))
        check([int(row['row_id']) for row in rows]==list(range(len(y))),'Mismatched export row order')
        hits[arm]=np.array([int(row['prediction']) for row in rows])==y
    a,b=hits['original'],hits['components']
    delta=b.astype(int)-a.astype(int)
    _,g=np.unique(read(out/'groups_only.json'),return_inverse=True)
    sizes=np.bincount(g)
    sums=np.bincount(g,weights=delta)
    rng=np.random.default_rng(20261123)
    draws=[]
    for _ in range(40):
        picks=rng.integers(len(sizes),size=(125,len(sizes)))
        draws.extend(100*sums[picks].sum(axis=1)/sizes[picks].sum(axis=1))
    save(out/'results.json',dict(rows=len(y),correct={arm:int(v.sum()) for arm,v in hits.items()},
         gained=int((b&~a).sum()),lost=int((~b&a).sum()),net_pp=100*float(delta.mean()),groups=len(sizes),
         paired_group_bootstrap_95_pp=np.quantile(draws,[.025,.975]).tolist(),runs=runs,
         actual_supervised_fits=sum(r['attempted'] for r in runs.values()),model_api_requests=0,new_model_api_rmb=0,
         official_validation_opened=False,official_test_opened=False,H_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','train','export','score'])
    p.add_argument('--archive',type=Path)
    p.add_argument('--split',type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    args.out=args.out.resolve()
    if args.stage=='prepare':
        prepare(args)
    elif args.stage=='train':
        train_once(args.out)
    elif args.stage=='export':
        exports(args.out)
    else:
        score(args.out)
