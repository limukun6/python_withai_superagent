"""Prepare grouped anonymous inputs, seal autonomous native runs, export twice, then score."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

from superagent.ab_run import sha, sources
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import grouped_partition, normalized_text, sample_groups


ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/batches/B_varied_autonomous_plan_20261124_zh.md'
BASE='9eb9b5a71f9ddce912550a28a17c12654dcd53a9'
FULL_SHA='6aac89257cef32aa0c498cfb0c9de4df73db4b91393551c5441ffa771016c393'
SMALL_SHA='f4e30ae46244196b927b2ea278c2d900ba2c53bcffa7b93f165e47aed1190bd2'
MOVIE_SHA='f664b56b946d83446e75831ee899c31cb4016eeb2a798d0832d676c2ed47bf73'
ARMS={'smoke':('fixture','varied',6),'reference':('titles','deep-e5-familiarity',11),
      'expanded':('titles','varied',11),'transfer':('movie','varied',11)}
SLOTS={'fixture':'k0','titles':'k1','movie':'k2'}


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def read(file):
    return json.loads(file.read_text(encoding='utf-8'))


def save(file,record):
    if file.exists():
        require(read(file)==record,'Sealed file differs: '+file.name)
    else:
        with file.open('x',encoding='utf-8') as f:
            json.dump(record,f,indent=2,ensure_ascii=False,allow_nan=False)


def package(path,rows,labels=None):
    has_y=labels is not None
    split='train' if has_y else 'test'
    manifest=dict(schema_version=1,objective='classification',metric='accuracy',modality='pair_text',
                  columns=[{'name':n,'type':'text'} for n in ('option_a','option_b')],target='outcome',
                  pair_semantics='preference',pair_label_one='left',splits={split:{'format':'csv','path':'rows.csv'}})
    table=io.StringIO(newline='')
    writer=csv.writer(table)
    writer.writerow(['option_a','option_b']+(['outcome'] if has_y else []))
    if has_y:
        writer.writerows((a,b,int(y)) for (a,b),y in zip(rows,labels,strict=True))
    else:
        writer.writerows(rows)
    with zipfile.ZipFile(path,'x') as z:
        for name,body in (('manifest.json',json.dumps(manifest)),('rows.csv',table.getvalue())):
            info=zipfile.ZipInfo(name,(2020,1,1,0,0,0))
            info.compress_type,info.external_attr,info.create_system=zipfile.ZIP_DEFLATED,0o600<<16,3
            z.writestr(info,body.encode())


def bucket(group,salt,n):
    return int(hashlib.sha256((salt+group).encode()).hexdigest(),16)%n


def title_split(large,small):
    groups=sample_groups(large,'shared_text')
    train,opening,_=grouped_partition(large,18643,'shared_text')
    seen=set(groups[opening])
    prior={'opening':len(opening)}
    for name,salt,mod,maximum in (
            ('shadow','b-borderline-20261123-v1/',8,2),('earlier','b-fresh-singles-20261124-v1/',8,2),
            ('cold','b-close-single-outer-20261124-v1/',4,1)):
        selected=[i for i in train if groups[i] not in seen and bucket(groups[i],salt,mod)<maximum]
        prior[name]=len(selected)
        seen.update(groups[selected])
    require(list(prior.values())==[1693,1637,1223,937],'Changed preceding title splits')
    second=grouped_partition(large,18744,'shared_text')[1]
    seen.update(groups[second])
    prior['full_confirmation']=len(second)
    sg=sample_groups(small,'shared_text')
    sg_seen=set()
    for seed,expected in ((18643,1295),(40531,1323)):
        ids=grouped_partition(small,seed,'shared_text')[1]
        require(len(ids)==expected,'Different historical small-packet split')
        sg_seen.update(sg[ids])
        prior['small_primary_'+str(seed)]=len(ids)
    old=sg_seen.copy()
    for k,expected in enumerate(('d066b2693d209c62f64f4781065ee5b09936cba73347b39579c31f4956207439',
                                 'e2f1a55ffccd111de478d1a225ad0a7bb9897c98146d14ba054a0821378395b7')):
        ids=[i for i,g in enumerate(sg) if g not in old and bucket(g,'title-weight-confirm-1123-v1:',5)==k]
        require(fingerprint(ids)==expected,'Changed historical 814/777 title partitions')
        sg_seen.update(sg[ids])
        prior['small_panel_'+str(k)]=len(ids)
    lookup={normalized_text(text):g for pair,g in zip(large.X_train,groups,strict=True) for text in pair}
    seen.update(lookup[normalized_text(s)] for pair,g in zip(small.X_train,sg,strict=True) if g in sg_seen for s in pair)
    held=[i for i,g in enumerate(groups) if g not in seen and bucket(g,'b-varied-models-20261124/v1/',3)<2]
    held_groups=set(groups[held])
    fit=[i for i,g in enumerate(groups) if g not in held_groups]
    require(held_groups.isdisjoint(seen) and held_groups.isdisjoint(groups[fit]) and len(held)+len(fit)==len(groups),
            'New development groups overlap')
    return fit,held,groups,prior


def fixture(folder):
    rows=[]
    outcomes=[]
    for i in range(180):
        name=('amber','ocean','silver','olive')[i%4]
        a=f'{name} report {i} has a full explanation'
        b=f'{name} update {i}?'
        reverse=bool((i*13)%7<3)
        rows.append((b,a) if reverse else (a,b))
        outcomes.append(int((i%4 in (0,2))!=reverse))
    package(folder/'train.zip',rows[:150],outcomes[:150])
    package(folder/'features.zip',rows[150:])


def prepare(a):
    require(sha(a.training)==FULL_SHA and sha(a.legacy)==SMALL_SHA,'Wrong training-only input')
    large,small=map(load_development_task,(a.training,a.legacy))
    require(large.y_val is None and small.y_val is None and (len(large.y_train),len(small.y_train))==(8301,6608),
            'Unexpected development-only inputs')
    fit,held,groups,prior=title_split(large,small)
    registration=read(a.movie/'protocol.json')
    require(sha(a.movie/'anonymous.zip')==registration['train_packet']==MOVIE_SHA and
            sha(a.movie/'features.zip')==registration['features'] and sha(a.movie/'scoring_only.json')==registration['scoring'] and
            registration['shared_outer_films']==registration['shared_outer_texts']==0,'Movie split receipt differs')
    a.out.mkdir(parents=True,exist_ok=False)
    inputs=a.out/'inputs'
    for name in SLOTS.values():
        (inputs/name).mkdir(parents=True)
    fixture(inputs/SLOTS['fixture'])
    folder=inputs/SLOTS['titles']
    package(folder/'train.zip',[large.X_train[i] for i in fit],large.y_train[fit])
    package(folder/'features.zip',[large.X_train[i] for i in held])
    with (folder/'score_only.npz').open('xb') as f:
        np.savez_compressed(f,labels=large.y_train[held],groups=groups[held],ids=np.asarray(held))
    for src,dst in (('anonymous.zip','train.zip'),('features.zip','features.zip'),('scoring_only.json','score_only.json')):
        shutil.copyfile(a.movie/src,inputs/SLOTS['movie']/dst)
    files={p.relative_to(inputs).as_posix():sha(p) for p in sorted(inputs.glob('*/*'))}
    save(a.out/'protocol.json',dict(base=BASE,source=sources(),plan=sha(PLAN),runner=sha(Path(__file__)),
         physical_guard=sha(ROOT/'scripts/ab_acceptance_guard/sitecustomize.py'),inputs=files,source_training=FULL_SHA,
         legacy_training=SMALL_SHA,prior_panels=prior,title_fit=len(fit),title_held=len(held),title_groups=len(set(groups[held])),
         shared_train_groups=0,shared_named_panel_groups=0,title_held_ids=fingerprint(held),title_fit_ids=fingerprint(fit),
         movie_fit=registration['train_rows'],movie_held=registration['held_rows'],arms=ARMS,seed=51803,
         fit_limit_total=sum(a[2] for a in ARMS.values()),planner_model_requests=0,course_validation_access=False,formal_test_access=False))
    print('Prepared',len(fit),'title fits rows,',len(held),'title holdout rows',flush=True)


def native_env(a,physical):
    return dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'scripts/ab_acceptance_guard'),str(ROOT)]),
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',AB_B1_OFFLINE='1',AB_B1_FIT_LOG=str(physical),
                AB_B1_PER_WORKER_FIT_LIMIT='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',
                TOKENIZERS_PARALLELISM='false',SUPERAGENT_MINILM_WEIGHTS=str(a.weights/'all-MiniLM-L6-v2'),
                SUPERAGENT_DEEP_MINILM_WEIGHTS=str(a.weights/'all-MiniLM-L12-v2'),SUPERAGENT_E5_SMALL_WEIGHTS=str(a.weights/'e5-small-v2'))


def cli(a,args,physical,log,timeout):
    with log.open('x') as stream:
        subprocess.run([sys.executable,'-m','superagent',*map(str,args)],cwd=ROOT,env=native_env(a,physical),
                       stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout)


def read_predictions(file):
    with file.open(newline='') as f:
        r=csv.DictReader(f)
        require(r.fieldnames==['row_id','prediction'],'Expected binary export')
        rows=list(r)
    require([r['row_id'] for r in rows]==list(map(str,range(len(rows)))) and all(r['prediction'] in ('0','1') for r in rows),
            'Export order or label differs')
    return np.array([int(r['prediction']) for r in rows])


def predict(a,name,n,expected,physical,model):
    task=SLOTS[ARMS[name][0]]
    arm=a.out/name
    source=a.out/'inputs'/task/'features.zip'
    require(sha(source)==read(a.out/'protocol.json')['inputs'][task+'/features.zip']
            and sha(arm/'agent/selected_model.joblib')==model,'Prediction inputs differ')
    file=arm/f'export_{n}'/'rows.csv'
    if not file.is_file():
        cli(a,['predict',source,'--run-dir',arm/'agent','--output',file],physical,arm/f'export_{n}.log',300)
    proof=read(file.parent/'prediction_manifest.json')
    require(proof['prediction_count']==expected and not proof['test_labels_read'] and len(read_predictions(file))==expected,
            'Independent native delivery failed')
    return sha(file)


def train_arm(a,name):
    task,catalog,limit=ARMS[name]
    protocol=read(a.out/'protocol.json')
    require(sources()==protocol['source'] and sha(PLAN)==protocol['plan'] and sha(Path(__file__))==protocol['runner'] and
            sha(ROOT/'scripts/ab_acceptance_guard/sitecustomize.py')==protocol['physical_guard'],'Study source changed')
    folder=a.out/name
    native=folder/'agent'
    slot=SLOTS[task]
    archive=a.out/'inputs'/slot/'train.zip'
    require(sha(archive)==protocol['inputs'][slot+'/train.zip'],'Training package changed')
    physical=folder/'physical.jsonl'
    if not (native/'run_outcome.json').exists():
        folder.mkdir(exist_ok=False)
        flags=['--planner','local','--scheduler','ab-v1','--portfolio','ab-v1','--preference-catalog',catalog,
               '--local-method-search','evidence','--task-profile','train','--fit-limit',str(limit),'--seed','51803',
               '--time-limit','1200','--trial-timeout','300','--run-dir',native]
        cli(a,['run',archive,*flags],physical,folder/'train.log',1260)
    outcome=read(native/'run_outcome.json')
    require(outcome.get('status')=='completed' and outcome['source_unchanged'] and
            outcome['requested_planner_completed'] and outcome['full_refit_completed'] and outcome['new_api_requests']==0 and
            not outcome['formal_test_accessed'],'Native run incomplete: '+name+' '+str(outcome.get('error','')))
    manifest=read(native/'run_manifest.json')
    require(manifest['source_files']==protocol['source'] and manifest['local_method_search']=='evidence' and
            manifest['preference_catalog']==catalog and manifest['task_archive_sha256']==sha(archive) and
            manifest['fit_limit']==limit and manifest['llm_request_limit']==0,'Native contract changed')
    final=read(native/'final_selection.json')
    history=read(native/'history.json')
    actions=read(native/'actions.json')
    workers=read(native/'classifier_fits.json')
    meter=[json.loads(line) for line in physical.read_text().splitlines()]
    require(0<len(meter)<=limit and workers['attempted_classifier_fits']==workers['completed_classifier_fits']==len(meter) and
            not workers['unknown_failed_attempts'] and [r['candidate'] for r in meter]==[r['candidate'] for r in workers['workers']] and
            all(r['classifier']==w['calls'][0]['classifier'] for r,w in zip(meter,workers['workers'],strict=True)),
            'Worker accounting mismatch')
    choices={str(i):read(native/f'method_choice_{i}.json') for i,action in enumerate(actions,1) if action['action']=='explore'}
    require(len(actions)==len(history)+1 and actions[-1]['action']=='stop' and
            all(choice['chosen']==actions[int(i)-1]['candidate_id'] for i,choice in choices.items()),'Unexplained model choice')
    n={'fixture':150,'titles':protocol['title_fit'],'movie':protocol['movie_fit']}[task]
    require(final['model_sha256']==sha(native/final['model_file']) and final['training_rows']==n,'Model refit differs')
    data=dict(protocol=sha(a.out/'protocol.json'),archive=sha(archive),catalog=list(manifest['capabilities']),
              capabilities=manifest['capabilities'],fit_limit=limit,source_count=len(manifest['source_files']),
              model=final['model_sha256'],selected=final['family'],refit_rows=n,physical=meter,actual_fits=len(meter),
              actions=actions,choices=choices,trials=[dict(id=v['candidate'],split=v['split_id'],score=v['search_score'],
              family=v['estimator']['family'],actual_class=v['estimator']['class'],rows=v['search_metrics']['rows'],
              payload=v['evaluation_payload_sha256'],row_ids=v['row_ids_sha256']) for v in history],
              school_api_calls=0,selection=read(native/'confirmation.json')['b_scheduler_review'])
    save(folder/'frozen.json',data)
    print(name,'sealed',final['family'],len(meter),'fits',flush=True)
    return data


def train(a):
    arm=train_arm(a,'smoke')
    physical=a.out/'smoke/physical.jsonl'
    exports=[predict(a,'smoke',n,30,physical,arm['model']) for n in (0,1)]
    require(exports[0]==exports[1],'Fixture independent predictions differ')
    save(a.out/'smoke/delivery.json',dict(exports=exports,rows=30,supervised_fits=arm['actual_fits']))
    real={name:train_arm(a,name) for name in ('reference','expanded','transfer')}
    reference=real['reference']['capabilities']
    expanded=real['expanded']['capabilities']
    require(set(reference)<set(expanded) and all(expanded[n]==r for n,r in reference.items()) and
            expanded==real['transfer']['capabilities'], 'The method sets or shared descriptors do not match')
    save(a.out/'models_frozen.json',dict(protocol=sha(a.out/'protocol.json'),arms={
         name:sha(a.out/name/'frozen.json') for name in ARMS},all_real_models_before_outer=True))


def evaluate(a):
    p=read(a.out/'protocol.json')
    sealed=read(a.out/'models_frozen.json')
    require(sealed['protocol']==sha(a.out/'protocol.json') and sources()==p['source'] and
            sha(Path(__file__))==p['runner'] and sha(PLAN)==p['plan'],'Missing frozen paths')
    outputs={}
    for name in ('reference','expanded','transfer'):
        folder=a.out/name
        require(sha(folder/'frozen.json')==sealed['arms'][name],'Frozen path changed')
        expected=p['movie_held'] if name=='transfer' else p['title_held']
        outputs[name]=[predict(a,name,i,expected,folder/'physical.jsonl',read(folder/'frozen.json')['model']) for i in (0,1)]
        require(outputs[name][0]==outputs[name][1],'Fresh prediction processes disagree')
    save(a.out/'exports.json',dict(frozen=sha(a.out/'models_frozen.json'),predictions=outputs))
    score_key=SLOTS['titles']+'/score_only.npz'
    scorefile=a.out/'inputs'/score_key
    require(sha(scorefile)==p['inputs'][score_key],'Title evaluation changed')
    with np.load(scorefile,allow_pickle=False) as f:
        y,groups=f['labels'],f['groups']
        require(fingerprint(f['ids'])==p['title_held_ids'],'Evaluation order changed')
    pred={n:read_predictions(a.out/n/'export_0/rows.csv') for n in outputs}
    before,after=pred['reference']==y,pred['expanded']==y
    _,ix=np.unique(groups,return_inverse=True)
    cells=np.stack([np.bincount(ix,weights=mask.astype(int)) for mask in
                    (~before&~after,before&~after,~before&after,before&after)],axis=1).astype(int)
    save(a.out/'title_groups.json',cells.tolist())
    diff,size=cells[:,2]-cells[:,1],cells.sum(axis=1)
    rng=np.random.default_rng(51803)
    vals=[]
    for _ in range(100):
        d=rng.integers(0,len(size),size=(50,len(size)))
        vals.extend(100*diff[d].sum(axis=1)/size[d].sum(axis=1))
    ci=np.quantile(vals,[.025,.975]).tolist()
    delta=100*int(diff.sum())/len(y)
    movie_key=SLOTS['movie']+'/score_only.json'
    movie_file=a.out/'inputs'/movie_key
    require(sha(movie_file)==p['inputs'][movie_key],'Movie evaluation changed')
    movie_y=np.asarray(read(movie_file)['labels'])
    records={n:read(a.out/n/'frozen.json') for n in ARMS}
    result=dict(exports=sha(a.out/'exports.json'),title_rows=len(y),title_groups=len(cells),
        correct={'reference':int(before.sum()),'expanded':int(after.sum())},gained=int(cells[:,2].sum()),
        lost=int(cells[:,1].sum()),delta_pp=delta,paired_group_95_pp=ci,promote=bool(delta>=.5 and ci[0]>0),
        movie_rows=len(movie_y),movie_correct=int(np.sum(pred['transfer']==movie_y)),movie_previously_used_development=True,
        selected={n:r['selected'] for n,r in records.items()},fits={n:r['actual_fits'] for n,r in records.items()},
        observed_families={n:sorted({q['family'] for q in r['trials']}) for n,r in records.items()},
        model_api_calls=0,new_rmb=0,designated_validation_read=False,formal_test_read=False)
    save(a.out/'results.json',result)
    save(a.out/'completion.json',{k:sha(a.out/(k+'.json')) for k in ('protocol','models_frozen','exports','title_groups','results')})
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','train','score','all'))
    for name in ('training','legacy','movie','weights','out'):
        parser.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    args=parser.parse_args()
    if args.phase in ('prepare','all'):
        prepare(args)
    if args.phase in ('train','all'):
        train(args)
    if args.phase in ('score','all'):
        evaluate(args)
