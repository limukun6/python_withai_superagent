"""Run the shared reference and optional close-single policy on one protected training-only holdout."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

from superagent.ab_run import sha
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import grouped_partition, sample_groups


ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/batches/B_autonomous_close_singles_plan_20261124_zh.md'
BASE='449b766eb4a29af2f4020092ab6b3eab425d25be'
SOURCE='6aac89257cef32aa0c498cfb0c9de4df73db4b91393551c5441ffa771016c393'
SEED=18643
OPTIONS=['--planner','local','--scheduler','ab-adaptive-complement','--portfolio','ab-v1',
         '--preference-catalog','deep-e5-familiarity','--cover-untried-encoder','--seed',str(SEED),
         '--fit-limit','10','--trial-timeout','300','--time-limit','1200']


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def seal(path, value):
    if path.exists():
        require(read(path)==value,'Sealed record differs: '+path.name)
    else:
        with path.open('x',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2)


def sources(tree):
    root=tree/'superagent'
    return {f.relative_to(root).as_posix():sha(f) for f in sorted(root.rglob('*.py'))}


def bucket(key,salt,modulo):
    return int(hashlib.sha256((salt+key).encode()).hexdigest(),16)%modulo


def packet(path, rows, labels=None):
    role='train' if labels is not None else 'test'
    schema=dict(schema_version=1,objective='classification',metric='accuracy',modality='pair_text',
                columns=[dict(name=n,type='text') for n in ('option_a','option_b')],target='outcome',
                pair_semantics='preference',pair_label_one='left',splits={role:dict(format='csv',path='rows.csv')})
    csv_data=io.StringIO(newline='')
    writer=csv.writer(csv_data)
    writer.writerow(['option_a','option_b']+(['outcome'] if labels is not None else []))
    if labels is None:
        writer.writerows(rows)
    else:
        writer.writerows((a,b,int(y)) for (a,b),y in zip(rows,labels,strict=True))
    with zipfile.ZipFile(path,'x') as z:
        for name,payload in (('manifest.json',json.dumps(schema)),('rows.csv',csv_data.getvalue())):
            entry=zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0))
            entry.compress_type,entry.create_system,entry.external_attr=zipfile.ZIP_DEFLATED,3,0o600<<16
            z.writestr(entry,payload.encode())


def prepare(a):
    require(sha(a.training)==SOURCE,'Wrong anonymous development-only source')
    control,changed=sources(a.control),sources(a.changed)
    previous=read(a.control/'research/a137_numeric_acceptance/run/run_manifest.json')['source_files']
    require(control==previous and set(control)<=set(changed) and
            {n for n in changed if changed[n]!=control.get(n)}=={'ab_close_singles.py','ab_run.py','cli.py'},
            'The treatment must differ only by the optional policy and native wiring')
    task=load_development_task(a.training)
    require(task.X_val is None and task.y_val is None and len(task.y_train)==8301,'Expected an 8301-row training-only packet')
    groups=sample_groups(task,'shared_text')
    remaining,opening,_=grouped_partition(task,SEED,'shared_text')
    require(fingerprint(opening)=='9f71a87253c396ae2c9ac705ceeed08c3a22a669c972c3b2693ba027c48ef0b0',
            'Earlier opening IDs differ')
    shadow=[int(i) for i in remaining if bucket(groups[i],'b-borderline-20261123-v1/',8) in (0,1)]
    first=set(groups[np.r_[opening,shadow]])
    earlier=[i for i,g in enumerate(groups) if g not in first and bucket(g,'b-fresh-singles-20261124-v1/',8) in (0,1)]
    old=first|set(groups[earlier])
    require((len(opening),len(shadow),len(earlier))==(1693,1637,1223),'Earlier development partitions changed')
    held=[i for i,g in enumerate(groups) if g not in old and bucket(g,'b-close-single-outer-20261124-v1/',4)==0]
    held_set=set(held)
    train=[i for i in range(len(groups)) if i not in held_set]
    require(not set(groups[held]) & (old|set(groups[train])) and 400<=len(held)<=1500,'Invalid group holdout')
    a.out.mkdir(parents=True,exist_ok=False)
    packet(a.out/'anonymous.zip',[task.X_train[i] for i in train],task.y_train[train])
    packet(a.out/'features.zip',[task.X_train[i] for i in held])
    with (a.out/'score_only.npz').open('xb') as stream:
        np.savez_compressed(stream,y=task.y_train[held],groups=groups[held],ids=np.asarray(held))
    seal(a.out/'protocol.json',dict(base=BASE,plan=sha(PLAN),runner=sha(Path(__file__)),source=SOURCE,
         sources=dict(control=control,proposal=changed),options=OPTIONS,fit_cap_each=10,model_api_cap=0,
         training_rows=len(train),held_rows=len(held),held_groups=len(set(groups[held])),
         prior_excluded_rows=len(opening)+len(shadow)+len(earlier),training_shared_groups=0,prior_shared_groups=0,
         train_ids=fingerprint(train),held_ids=fingerprint(held),training=sha(a.out/'anonymous.zip'),
         features=sha(a.out/'features.zip'),scoring=sha(a.out/'score_only.npz')))
    print(dict(train=len(train),held=len(held),groups=len(set(groups[held]))),flush=True)


def env(tree,weights,physical):
    return dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,(tree/'scripts/ab_acceptance_guard',tree))),
           SUPERAGENT_MINILM_WEIGHTS=str(weights/'all-MiniLM-L6-v2'),
           SUPERAGENT_DEEP_MINILM_WEIGHTS=str(weights/'all-MiniLM-L12-v2'),SUPERAGENT_E5_SMALL_WEIGHTS=str(weights/'e5-small-v2'),
           HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',AB_B1_OFFLINE='1',AB_B1_FIT_LOG=str(physical),
           AB_B1_PER_WORKER_FIT_LIMIT='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',
           TOKENIZERS_PARALLELISM='false')


def cli(tree,options,environment,log,seconds):
    with log.open('x') as stream:
        subprocess.run([sys.executable,'-m','superagent',*map(str,options)],cwd=tree,env=environment,
                       stdout=stream,stderr=subprocess.STDOUT,timeout=seconds,check=True)


def train_arms(a):
    protocol=read(a.out/'protocol.json')
    require(sha(a.out/'anonymous.zip')==protocol['training'] and sha(PLAN)==protocol['plan']
            and sha(Path(__file__))==protocol['runner'],'Registered study changed')
    summary={}
    menu=None
    for arm,tree in (('control',a.control),('proposal',a.changed)):
        root=a.out/arm
        folder=root/'agent'
        require(sources(tree)==protocol['sources'][arm],'Production sources changed')
        if not (folder/'run_outcome.json').is_file():
            root.mkdir(exist_ok=False)
            flags=OPTIONS+(['--confirm-close-singles'] if arm=='proposal' else [])
            cli(tree,['run',a.out/'anonymous.zip',*flags,'--run-dir',folder],env(tree,a.weights,root/'physical.jsonl'),
                root/'train.log',1260)
        outcome,manifest,fits,final,choice=[read(folder/f) for f in ('run_outcome.json','run_manifest.json',
                      'classifier_fits.json','final_selection.json','selection.json')]
        physical=[json.loads(v)['candidate'] for v in (root/'physical.jsonl').read_text().splitlines()]
        require(outcome['status']=='completed' and outcome['source_unchanged'] and outcome['requested_planner_completed']
                and outcome['full_refit_completed'] and not outcome['formal_test_accessed'] and outcome['new_api_requests']==0
                and manifest['source_files']==protocol['sources'][arm] and manifest['task_archive_sha256']==protocol['training']
                and manifest.get('confirm_close_singles',False)==(arm=='proposal') and manifest['fit_limit']==10
                and manifest['llm_request_limit']==0 and final['training_rows']==protocol['training_rows']
                and final['model_sha256']==sha(folder/final['model_file']), 'Autonomous path failed: '+arm)
        require(0<len(physical)<=10 and physical==[v['candidate'] for v in fits['workers']]
                and fits['attempted_classifier_fits']==fits['completed_classifier_fits']==len(physical)
                and fits['unknown_failed_attempts']==0,'Physical fit discrepancy')
        if menu is not None:
            require(menu==manifest['capabilities'],'Different menus across arms')
        menu=manifest['capabilities']
        history=read(folder/'history.json')
        summary[arm]=dict(model=final['model_sha256'],selected=final['family'],components=final['ensemble_components'],
             actions=read(folder/'actions.json'),physical=physical,actual_fits=len(physical),
             review=choice['confirmation']['b_scheduler_review'],
             observations=[dict(c=r['candidate'],split=r['split_id'],score=r['search_score'],
                           row_ids=r['row_ids_sha256'],payload=r['evaluation_payload_sha256']) for r in history])
        print(arm,'frozen',len(physical),flush=True)
    require(summary['control']['observations'][:4]==summary['proposal']['observations'][:4],
            'Initial observations or row partitions differ')
    seal(a.out/'models_frozen.json',dict(protocol=sha(a.out/'protocol.json'),arms=summary,catalog=list(menu)))


def score(a):
    protocol=read(a.out/'protocol.json')
    frozen=read(a.out/'models_frozen.json')
    require(frozen['protocol']==sha(a.out/'protocol.json') and sha(a.out/'features.zip')==protocol['features'],
            'Missing complete freeze or feature package changed')
    outputs={}
    for arm,tree in (('control',a.control),('proposal',a.changed)):
        folder=a.out/arm/'agent'
        require(sources(tree)==protocol['sources'][arm] and sha(folder/'selected_model.joblib')==frozen['arms'][arm]['model'],
                'Prediction source or model changed')
        hashes=[]
        for n in (0,1):
            target=a.out/arm/f'export_{n}'/'rows.csv'
            if not target.is_file():
                cli(tree,['predict',a.out/'features.zip','--run-dir',folder,'--output',target],
                    env(tree,a.weights,a.out/arm/'physical.jsonl'),a.out/arm/f'predict_{n}.log',300)
            manifest=read(target.parent/'prediction_manifest.json')
            require(not manifest['test_labels_read'] and manifest['prediction_count']==protocol['held_rows'],'Invalid export')
            hashes.append(sha(target))
        require(len(set(hashes))==1,'Fresh processes produced different CSV files')
        outputs[arm]=hashes[0]
    seal(a.out/'exports.json',dict(frozen=sha(a.out/'models_frozen.json'),features=protocol['features'],predictions=outputs))
    require(sha(a.out/'score_only.npz')==protocol['scoring'],'Scoring artifact changed')
    with np.load(a.out/'score_only.npz',allow_pickle=False) as data:
        y,groups,ids=[data[k] for k in ('y','groups','ids')]
    require(fingerprint(ids)==protocol['held_ids'],'Held row order differs')
    masks={}
    for arm in outputs:
        with (a.out/arm/'export_0/rows.csv').open(newline='') as f:
            reader=csv.DictReader(f)
            require(reader.fieldnames==['row_id','prediction'],'Export columns differ')
            rows=list(reader)
        require([r['row_id'] for r in rows]==list(map(str,range(len(y))))
                and all(r['prediction'] in ('0','1') for r in rows),'Unexpected row order or class')
        masks[arm]=np.asarray([int(r['prediction']) for r in rows])==y
    before,after=masks['control'],masks['proposal']
    counts=defaultdict(lambda:Counter())
    for group,l,r in zip(groups,before,after,strict=True):
        counts[group][int(l)+2*int(r)]+=1
    cells=np.asarray([[counts[g][i] for i in range(4)] for g in sorted(counts)])
    seal(a.out/'groups.json',cells.tolist())
    delta=cells[:,2]-cells[:,1]
    sizes=cells.sum(axis=1)
    rng=np.random.default_rng(62114)
    boot=[]
    for _ in range(100):
        ix=rng.integers(0,len(cells),size=(50,len(cells)))
        boot.extend(100*delta[ix].sum(axis=1)/sizes[ix].sum(axis=1))
    interval=np.quantile(boot,[.025,.975]).tolist()
    pp=100*float(delta.sum())/len(y)
    paths={arm:[(a['action'],a['candidate_id'],a['split_id']) for a in rec['actions']]
           for arm,rec in frozen['arms'].items()}
    changed=paths['control']!=paths['proposal']
    result=dict(rows=len(y),groups=len(cells),training_rows=protocol['training_rows'],catalog=len(frozen['catalog']),
          correct={arm:int(mask.sum()) for arm,mask in masks.items()},train_only=True,
          selected={arm:rec['selected'] for arm,rec in frozen['arms'].items()},
          fits={arm:rec['actual_fits'] for arm,rec in frozen['arms'].items()},gained=int(cells[:,2].sum()),
          lost=int(cells[:,1].sum()),delta_pp=pp,group_bootstrap_95_pp=interval,actions_differ=changed,
          gate=bool(changed and pp>=.5 and interval[0]>0),school_api_calls=0,new_rmb=0,
          designated_validation_opened=False,formal_test_opened=False)
    seal(a.out/'results.json',result)
    seal(a.out/'completion.json',{k:sha(a.out/(k+'.json')) for k in ('protocol','models_frozen','exports','groups','results')})
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','train','score','all'))
    for key in ('training','weights','control','changed','out'):
        parser.add_argument('--'+key,type=lambda p:Path(p).resolve(),required=True)
    args=parser.parse_args()
    if args.phase in ('prepare','all'):
        prepare(args)
    if args.phase in ('train','all'):
        train_arms(args)
    if args.phase in ('score','all'):
        score(args)
