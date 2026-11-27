"""Prospectively compare general feedback with a confusion-driven candidate on new, disjoint records."""
import argparse
import csv
import hashlib
import io
from itertools import islice
from pathlib import Path
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split

from scripts import compare_joint_feedback as joint
from scripts.run_planner_self_review import distinct, packet, source_identity
from scripts.unified_released_study import ROOT, read, require, save, sha

PLAN=ROOT/'research/feedback_design_1127/protocol.json'
ARMS=('feedback-local','feedback-design')
CASES=('u31','u46','u72')


def frozen():
    return dict(agent=source_identity(ROOT),runner=sha(Path(__file__)),entry_runner=sha(Path(joint.__file__)),
                guard=sha(joint.GUARD),protocol=sha(PLAN))


def fingerprint(ids):
    return hashlib.sha256(np.asarray(ids,dtype='>u4').tobytes()).hexdigest()


def source_data(root,previous):
    with zipfile.ZipFile(root/'sources/letters.zip') as z, z.open('letter-recognition.data') as incoming:
        rows=list(islice(csv.reader(io.TextIOWrapper(incoming,encoding='ascii')),16000))
    require(len(rows)==16000 and all(len(r)==17 for r in rows),'Unexpected UCI training layout')
    yield 'u31',np.array([r[1:] for r in rows],dtype=float),np.array([r[0] for r in rows]),None
    for case,x,y in joint.raw_records(previous/'sources'):
        if case in ('d37','d58'):
            yield ('u46' if case=='d37' else 'u72'),x,y,case


def prepare(args):
    study=args.root/'development'
    study.mkdir()
    answers=study/'evaluation'
    answers.mkdir()
    protocol=read(PLAN)
    sampling=protocol['sampling']
    prior=read(args.previous/'development/registered.json')
    meta={}
    for case,x,y,old in source_data(args.root,args.previous):
        ids=distinct(x,y)
        if old:
            sampled,remainder=train_test_split(ids,train_size=1200,stratify=y[ids],random_state=sampling['previous_joint_seed'])
            used,held=train_test_split(sampled,train_size=900,stratify=y[sampled],random_state=sampling['previous_joint_seed'])
            require([fingerprint(used),fingerprint(held)]==[prior['cases'][old][k] for k in ('train_ids_sha256','held_ids_sha256')],
                    'Cannot verify earlier sample exclusion')
        else:
            remainder=ids
            sampled=[]
        seen=set(sampled)
        modality,metric=protocol['sources'][case][2:]
        for suffix,seed in (('a',sampling['pilot_seed']),('b',sampling['replication_seed'])):
            key=case+suffix
            subset,remainder=train_test_split(remainder,train_size=sampling['train']+sampling['outer'],
                                             stratify=y[remainder],random_state=seed)
            fit,held=train_test_split(subset,train_size=sampling['train'],stratify=y[subset],random_state=seed)
            require(seen.isdisjoint(subset) and set(fit).isdisjoint(held),'A previous row recurred')
            seen.update(subset)
            folder=study/key
            folder.mkdir()
            packet(folder/'task.zip',modality,metric,x[fit],y[fit])
            packet(folder/'query.zip',modality,metric,x[held])
            truth=answers/(key+'.npy')
            np.save(truth,y[held],allow_pickle=False)
            meta[key]=dict(source=case,modality=modality,metric=metric,train_rows=len(fit),held_rows=len(held),
                excluded_prior_rows=len(sampled),class_count=len(np.unique(y[fit])),train_sha256=sha(folder/'task.zip'),
                query_sha256=sha(folder/'query.zip'),truth_sha256=sha(truth),train_ids_sha256=fingerprint(fit),
                held_ids_sha256=fingerprint(held),source_training_population=len(y))
    hashes={n:sha(args.previous/'sources'/n) for n in ('train_5500.label','fashion-images.gz','fashion-labels.gz')}
    hashes['letters.zip']=sha(args.root/'sources/letters.zip')
    save(study/'registered.json',dict(frozen=frozen(),cases=meta,upstream_sha256=hashes))


def run(args):
    if args.case.endswith('b'):
        gate_record=read(args.root/'development/replications.json')
        require(args.case in gate_record['run'],'This source did not qualify for replication')
    joint.run(args,expected_fingerprint=frozen(),additional_sealed_directory=args.previous)


def sealed_directory(folder):
    return folder if (folder/'sealed.json').is_file() else folder/'export-recovery'


def paired(folder):
    control,design=[read(sealed_directory(folder/arm)/'sealed.json') for arm in ARMS]
    require(control['core_actions']==design['core_actions'] and control['core']==design['core'] and
            control['opportunity']==design['opportunity'],'Pre-intervention search differed')
    data=read(folder/ARMS[1]/'run/agent/task_opening.json')
    proposal=data.get('designed_method',dict(status='no_matched_core',reason=data['result']))
    return control,design,proposal


def gate(args):
    study=args.root/'development'
    decisions={}
    for case in CASES:
        _,_,record=paired(study/(case+'a'))
        decisions[case]=record['status']
    selected=[c+'b' for c in CASES if decisions[c]=='adopted'][:2]
    save(study/'replications.json',dict(pilot_training_decisions=decisions,run=selected,outer_labels_opened=False))
    print('Registered replication cases:',selected,flush=True)


def score(args):
    study=args.root/'development'
    decisions=read(study/'replications.json')
    included=[*(c+'a' for c in CASES),*decisions['run']]
    # A missing seal on any selected path blocks disclosure of all scores.
    seals={c:paired(study/c) for c in included}
    meta=read(study/'registered.json')
    results={}
    for case,(control,design,proposal) in seals.items():
        info=meta['cases'][case]
        truthfile=study/'evaluation'/(case+'.npy')
        require(sha(truthfile)==info['truth_sha256'],'Changed external truth')
        truth=np.load(truthfile,allow_pickle=False)
        rows={}
        ys=[]
        for arm,seal in zip(ARMS,(control,design)):
            directory=sealed_directory(study/case/arm)
            output=directory/'delivery/predictions.csv'
            require(sha(output)==sha(directory/'reload/predictions.csv')==seal['output_sha'],'Changed output or replay')
            with output.open(newline='') as handle:data=list(csv.DictReader(handle))
            require([r['row_id'] for r in data]==list(map(str,range(len(truth)))),'Rows do not match')
            raw=[r['prediction'] for r in data]
            require(set(raw)<=set(map(str,np.unique(truth))),'Unrecognized label')
            pred=np.asarray(raw,dtype=truth.dtype)
            ys.append(pred)
            rows[arm]=dict(primary_score=joint.metrics(info['metric'],truth,pred),correct=int(sum(truth==pred)),
                           export_recovery=seal.get('export_recovery'), **{k:seal[k] for k in ('fits','requests','course_cost_rmb','core','remaining','seconds','fresh_reload',
                                 'model_sha','output_sha','tested','result','comparisons','opportunity')})
        results[case]=dict(metric=info['metric'],rows=len(truth),arms=rows,design=proposal,
                           disagreements=int(sum(ys[0]!=ys[1])),gained=int(sum((ys[0]!=truth)&(ys[1]==truth))),
                           lost=int(sum((ys[0]==truth)&(ys[1]!=truth))),
                           primary_gain=rows[ARMS[1]]['primary_score']-rows[ARMS[0]]['primary_score'])
    save(study/'scored.json',dict(registration=meta,replication=decisions,results=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','gate','score'))
    p.add_argument('--root',required=True,type=lambda s:Path(s).resolve())
    p.add_argument('--previous',type=lambda s:Path(s).resolve(),default=ROOT.parent.parent/'outputs/joint-feedback-20261127')
    p.add_argument('--case',choices=tuple(c+s for c in CASES for s in ('a','b')))
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--weights',type=lambda s:Path(s).resolve(),default=ROOT.parent/'pretrained_weights')
    args=p.parse_args()
    if args.action=='run' and (not args.case or not args.arm):p.error('run requires case and arm')
    {'prepare':prepare,'run':run,'gate':gate,'score':score}[args.action](args)
