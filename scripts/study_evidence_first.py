"""Three-arm comparison on disjoint development packets of scheduling and generated class-pair methods."""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from scripts import compare_joint_feedback as joint
from scripts import study_feedback_design as initial
from scripts.run_planner_self_review import distinct, packet, source_identity
from scripts.unified_released_study import ROOT, read, require, save, sha

PLAN=ROOT/'research/evidence_first_1127/protocol.json'
ARMS=('feedback-local','feedback-confirm','feedback-design')
CASES=('v21','v83')


def frozen():
    return dict(agent=source_identity(ROOT),runner=sha(Path(__file__)),parent_runner=sha(Path(initial.__file__)),
                entry_runner=sha(Path(joint.__file__)),guard=sha(joint.GUARD),protocol=sha(PLAN))


def sample(ids,y,n,held,seed):
    subset,remainder=train_test_split(ids,train_size=n+held,stratify=y[ids],random_state=seed)
    train,outer=train_test_split(subset,train_size=n,stratify=y[subset],random_state=seed)
    return train,outer,remainder


def compare_ids(train,outer,registered):
    require([initial.fingerprint(train),initial.fingerprint(outer)]==[registered[k] for k in ('train_ids_sha256','held_ids_sha256')],
            'Previously registered membership did not match')


def prepare(a):
    study=a.root/'development'
    study.mkdir(parents=True)
    truthdir=study/'evaluation'
    truthdir.mkdir()
    settings=read(PLAN)
    old=read(a.previous/'development/registered.json')
    last=read(a.initial/'development/registered.json')
    before=read(initial.PLAN)['sampling']
    sampling=settings['sampling']
    cases={}
    for previous_case,x,y,old_code in initial.source_data(a.initial,a.previous):
        mapping=next(((k,v) for k,v in settings['panels'].items() if v[0]==previous_case),None)
        if not mapping:continue
        prefix,(_,modality,metric)=mapping
        ids=distinct(x,y)
        remaining=ids
        seen=set()
        if old_code:
            train,outer,remaining=sample(remaining,y,900,300,before['previous_joint_seed'])
            compare_ids(train,outer,old['cases'][old_code]);seen.update(train);seen.update(outer)
        for suffix,seed in (('a',before['pilot_seed']),('b',before['replication_seed'])):
            train,outer,remaining=sample(remaining,y,before['train'],before['outer'],seed)
            compare_ids(train,outer,last['cases'][previous_case+suffix]);seen.update(train);seen.update(outer)
        excluded=len(seen)
        for suffix,seed in (('a',sampling['first']),('b',sampling['second'])):
            code=prefix+suffix
            train,outer,remaining=sample(remaining,y,sampling['train'],sampling['outer'],seed)
            require(seen.isdisjoint(train) and seen.isdisjoint(outer) and set(train).isdisjoint(outer),'Split overlap')
            seen.update(train);seen.update(outer)
            folder=study/code
            folder.mkdir()
            packet(folder/'task.zip',modality,metric,x[train],y[train])
            packet(folder/'query.zip',modality,metric,x[outer])
            truth=truthdir/(code+'.npy')
            np.save(truth,y[outer],allow_pickle=False)
            cases[code]=dict(train_rows=len(train),held_rows=len(outer),metric=metric,modality=modality,
                excluded_earlier_registered_rows=excluded,train_ids_sha256=initial.fingerprint(train),
                held_ids_sha256=initial.fingerprint(outer),train_sha256=sha(folder/'task.zip'),
                query_sha256=sha(folder/'query.zip'),truth_sha256=sha(truth))
    require(set(cases)=={c+s for c in CASES for s in ('a','b')},'Incomplete study packets')
    save(study/'registered.json',dict(frozen=frozen(),cases=cases,earlier_registrations=[
         sha(a.previous/'development/registered.json'),sha(a.initial/'development/registered.json')]))


def run(a):
    if a.case.endswith('b'):
        require(a.case in read(a.root/'development/replications.json')['run'] and a.arm in ARMS[1:],'Ineligible replication')
    joint.run(a,expected_fingerprint=frozen(),additional_sealed_directory=(a.previous,a.initial))


def sealed(study,case):
    arms=ARMS if case.endswith('a') else ARMS[1:]
    result={}
    for arm in arms:
        folder=study/case/arm
        row=read(folder/'sealed.json')
        opening=read(folder/'run/agent/task_opening.json')
        result[arm]=dict(seal=row,design=opening.get('designed_method',{'status':'no_matched_core'}),
                         confirmations=opening.get('protected_confirmations',[]))
    control,design=[result[arm]['seal'] for arm in ARMS[1:]]
    require(control['core_actions']==design['core_actions'] and control['opportunity']==design['opportunity'],
            'Design and scheduling-only paths differed before treatment')
    return result


def gate(a):
    study=a.root/'development'
    trials={c:sealed(study,c+'a')['feedback-design']['design']['status'] for c in CASES}
    chosen=[c+'b' for c in CASES if trials[c]=='adopted'][:1]
    save(study/'replications.json',dict(pilot_status=trials,run=chosen,outer_opened=False))
    print('Independent replications:',chosen,flush=True)


def score(a):
    study=a.root/'development'
    reg=read(study/'registered.json')
    gate_record=read(study/'replications.json')
    selected=[*(c+'a' for c in CASES),*gate_record['run']]
    all_seals={c:sealed(study,c) for c in selected}
    results={}
    for case,arms in all_seals.items():
        meta=reg['cases'][case]
        truthfile=study/'evaluation'/(case+'.npy')
        require(sha(truthfile)==meta['truth_sha256'],'Truth changed')
        truth=np.load(truthfile,allow_pickle=False)
        arrays={}
        entries={}
        for arm,data in arms.items():
            f=study/case/arm
            seal=data['seal']
            dest=f/'delivery/predictions.csv'
            require(sha(dest)==sha(f/'reload/predictions.csv')==seal['output_sha'],'Changed export')
            with dest.open(newline='') as incoming:rows=list(csv.DictReader(incoming))
            require([r['row_id'] for r in rows]==list(map(str,range(len(truth)))),'Changed row order')
            raw=[r['prediction'] for r in rows]
            require(set(raw)<=set(map(str,np.unique(truth))),'Unexpected predicted label')
            pred=np.asarray(raw,dtype=truth.dtype)
            arrays[arm]=pred
            entries[arm]=dict(primary_score=joint.metrics(meta['metric'],truth,pred),correct=int(sum(truth==pred)),
                protected_confirmations=data['confirmations'],design=data['design'] if arm=='feedback-design' else None,
                **{k:seal[k] for k in ('core','tested','opportunity','remaining','result','comparisons','fits','requests',
                                      'course_cost_rmb','fresh_reload','seconds','model_sha','output_sha')})
        deltas={}
        for pair in (('feedback-local','feedback-confirm'),('feedback-confirm','feedback-design')):
            a0,a1=pair
            if a0 not in arrays:continue
            v0,v1=arrays[a0],arrays[a1]
            deltas[a0+' to '+a1]=dict(gained=int(sum((v0!=truth)&(v1==truth))),lost=int(sum((v0==truth)&(v1!=truth))),
                disagreements=int(sum(v0!=v1)),primary_gain=entries[a1]['primary_score']-entries[a0]['primary_score'])
        results[case]=dict(metric=meta['metric'],rows=len(truth),arms=entries,paired=deltas)
    save(study/'scored.json',dict(registration=reg,replication=gate_record,results=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','gate','score'))
    p.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    p.add_argument('--initial',type=lambda s:Path(s).resolve(),default=ROOT.parent.parent/'outputs/feedback-design-20261127')
    p.add_argument('--previous',type=lambda s:Path(s).resolve(),default=ROOT.parent.parent/'outputs/joint-feedback-20261127')
    p.add_argument('--weights',type=lambda s:Path(s).resolve(),default=ROOT.parent/'pretrained_weights')
    p.add_argument('--case',choices=tuple(c+s for c in CASES for s in ('a','b')))
    p.add_argument('--arm',choices=ARMS)
    args=p.parse_args()
    if args.action=='run' and (not args.case or not args.arm):p.error('run requires a case and arm')
    {'prepare':prepare,'run':run,'gate':gate,'score':score}[args.action](args)
