"""Compare two frozen-recipe encoders on fresh title components outside three previous score panels."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from scripts.run_b_shared_full_task2 import check, read, seal
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.ab_run import sha, sources
from superagent.data import load_development_task
from superagent.models import predict_with_model, search_candidates
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups, take

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'docs/batches/B_fresh_single_encoders_plan_20261124_zh.md'


def bucket(group, salt):
    return int(hashlib.sha256((salt+group).encode()).hexdigest(),16)%8


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    parent,shadow,matched=[read(root/'protocol.json') for root in (a.parent,a.shadow,a.matched)]
    manifest=read(a.parent/'agent/run_manifest.json')
    check(sources()==manifest['source_files']==parent['production'] and
          matched['shadow_completion']==sha(a.shadow/'completion.json') and
          matched['parent_protocol']==sha(a.parent/'protocol.json') and
          sha(a.parent/'anonymous.zip')==parent['training'], 'Prior sources or picks differ')
    task=load_development_task(a.parent/'anonymous.zip')
    check(task.X_val is None and task.y_val is None,'Expected training only')
    groups=sample_groups(task,'shared_text')
    opening=read(a.parent/'agent/primary_split.json')
    old_salt='b-borderline-20261123-v1/'
    prior_panels=[[i for i in opening['train_indices'] if bucket(groups[i],old_salt)==n] for n in (0,1)]
    check([fingerprint(ids) for ids in prior_panels]==[p['row_ids_sha256'] for p in shadow['panels']],
          'Earlier evaluation groups changed')
    old_rows=opening['search_indices']+prior_panels[0]+prior_panels[1]
    old=set(groups[old_rows])
    tag=[-1 if g in old else bucket(g,'b-fresh-singles-20261124-v1/') for g in groups]
    panels=[[i for i,t in enumerate(tag) if t==n] for n in (0,1)]
    train=[i for i,t in enumerate(tag) if t not in (0,1)]
    new_groups=[set(groups[ids]) for ids in panels]
    check(not (new_groups[0]&new_groups[1] or (old|set(groups[train])) & (new_groups[0]|new_groups[1])) and
          all(len(ids)>100 and set(task.y_train[ids])=={0,1} for ids in (*panels,train)), 'Invalid group assignment')
    names=[shadow['anchor'],matched['candidate']]
    os.environ.update(SUPERAGENT_DEEP_MINILM_WEIGHTS=str(a.weights/'all-MiniLM-L12-v2'),
                      SUPERAGENT_MINILM_WEIGHTS=str(a.weights/'all-MiniLM-L6-v2'),
                      SUPERAGENT_E5_SMALL_WEIGHTS=str(a.weights/'e5-small-v2'),
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.environ[PREFERENCE_CATALOG_ENV]=manifest['preference_catalog']
    def offline(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            raise PermissionError('Encoder comparison is offline')
    sys.addaudithook(offline)
    catalog={c.family:c.params for c in search_candidates('pair_text','all')}
    check(all(catalog[n]==manifest['capabilities'][n]['params'] for n in names),'Candidate definition differs')
    seal(a.out/'protocol.json',dict(plan=sha(PLAN),runner=sha(Path(__file__)),shared=parent['shared_commit'],
         parent=sha(a.parent/'protocol.json'),previous_experiment=sha(a.matched/'completion.json'),
         names=names,parameters=[catalog[n] for n in names], excluded_evaluation_rows=len(old_rows),
         excluded_evaluation_groups=len(old),train_rows=len(train),train_ids=fingerprint(train),
         panels=[dict(rows=len(ids),groups=len(gs),ids=fingerprint(ids),prior_shared_groups=0,training_shared_groups=0)
                 for ids,gs in zip(panels,new_groups,strict=True)],fit_cap=2,model_api_cap=0))
    payload=a.out/'training.joblib'
    joblib.dump(dict(kind='pair_text',pair_semantics='preference',mode='refit',X_train=take(task.X_train,train),
                y_train=task.y_train[train],X_search=take(task.X_train,train[:1])),payload)
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/ab_acceptance_guard',ROOT))),
             AB_B1_OFFLINE='1',AB_B1_FIT_LOG=str(a.out/'physical.jsonl'),AB_B1_PER_WORKER_FIT_LIMIT='1',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
    files=[]
    for n,name in enumerate(names):
        admission=a.out/f'admission_{n}.json'
        receipt=a.out/f'fit_{n}.json'
        seal(admission,dict(maximum_classifier_fits=1,ledger_path=str(receipt)))
        artifact=a.out/f'worker_{n}.joblib'
        with (a.out/f'worker_{n}.log').open('x') as log:
            subprocess.run([sys.executable,'-m','superagent.experiment_worker',str(payload),name,str(artifact),
                            '--fit-admission',str(admission)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                           timeout=300,check=True)
        fit=read(receipt)
        check(fit['status']=='completed' and fit['attempted_classifier_fits']==fit['completed_classifier_fits']==1,
              'Physical fit did not complete')
        files.append(artifact)
    recorded=[json.loads(row)['candidate'] for row in (a.out/'physical.jsonl').read_text().splitlines()]
    check(recorded==names,'Observation and fit identities differ')
    seal(a.out/'models_frozen.json',dict(protocol=sha(a.out/'protocol.json'),models=[sha(f) for f in files],
         physical=sha(a.out/'physical.jsonl'),receipts=[sha(a.out/f'fit_{n}.json') for n in range(2)]))
    models=[joblib.load(f)['model'] for f in files]
    tables=[[predict_with_model('pair_text',m,take(task.X_train,ids)) for m in models] for ids in panels]
    seal(a.out/'exports.json',dict(frozen=sha(a.out/'models_frozen.json'),
         panel_predictions=[[fingerprint(col) for col in table] for table in tables]))
    summary, grouped=[], []
    for ids,(first,second) in zip(panels,tables,strict=True):
        y=task.y_train[ids]
        one,two=first==y,second==y
        counts=defaultdict(lambda:[0]*4)
        for g,b,c in zip(groups[ids],one,two,strict=True):
            counts[g][int(b)+2*int(c)]+=1
        grouped.append([counts[g] for g in sorted(counts)])
        summary.append(dict(rows=len(y),groups=len(counts),l12=int(one.sum()),e5=int(two.sum()),
                            gained=int((~one&two).sum()),lost=int((one&~two).sum()),net=int(two.sum()-one.sum())))
    cells=np.array(grouped[0]+grouped[1])
    d=cells[:,2]-cells[:,1]
    size=cells.sum(axis=1)
    rng=np.random.default_rng(60424)
    reps=[]
    for _ in range(100):
        draws=rng.integers(0,len(cells),size=(50,len(cells)))
        reps.extend(100*d[draws].sum(axis=1)/size[draws].sum(axis=1))
    interval=np.quantile(reps,[.025,.975]).tolist()
    pp=100*float(d.sum())/int(size.sum())
    ok=all(r['net']>0 for r in summary) and pp>=.5 and interval[0]>0
    result=dict(panels=summary,net=int(d.sum()),delta_pp=pp,group_bootstrap_95_pp=interval,pass_gate=ok,
                classifier_fits=2,model_api_calls=0,new_rmb=0,designated_validation_opened=False,formal_test_opened=False)
    seal(a.out/'grouped.json',grouped)
    seal(a.out/'results.json',result)
    seal(a.out/'completion.json',{k:sha(a.out/(k+'.json')) for k in ('protocol','models_frozen','exports','grouped','results')})
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','shadow','matched','weights','out'):
        parser.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    run(parser.parse_args())
