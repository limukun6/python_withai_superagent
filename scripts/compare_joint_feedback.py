"""Prepare, run, freeze, and independently score the registered three-task development comparison."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import time
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from scripts.run_planner_self_review import distinct, packet, source_identity
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from scripts.verify_anonymous_planner import execute

PLAN = ROOT / 'research/joint_feedback_1127/protocol.json'
GUARD = ROOT / 'scripts/joint_feedback_observer/sitecustomize.py'
CASES = ('d14', 'd37', 'd58')
ARMS = ('feedback-control', 'feedback-local')
CAMPUS = 'feedback-flash'
CONFIG = ('SUPERAGENT_PLANNER_KEY_FILE', 'SUPERAGENT_PLANNER_LEDGER', 'SUPERAGENT_TASK_BUDGET_ID',
          'SUPERAGENT_RESEARCH_REFLECTION_OUTBOX', 'SUPERAGENT_RESEARCH_PAIRED_REFLECTION', 'SUPERAGENT_LESSON_STORE')


def raw_records(sources):
    with zipfile.ZipFile(sources / 'chess.zip') as z:
        records = [r for r in csv.reader(io.StringIO(z.read('kr-vs-kp.data').decode('ascii'))) if r]
    require(all(len(r) == 37 for r in records), 'Invalid UCI record')
    chess = np.asarray(records)
    yield 'd14', chess[:, :-1].copy(), chess[:, -1]
    text, coarse = [], []
    for line in (sources / 'train_5500.label').read_bytes().decode('latin1').splitlines():
        label, question = line.split(' ', 1)
        main, _ = label.split(':', 1)
        text.append(question)
        coarse.append(main)
    yield 'd37', np.asarray(text), np.asarray(coarse)
    with gzip.open(sources/'fashion-images.gz', 'rb') as incoming:
        magic, n, h, w = struct.unpack('>4I', incoming.read(16))
        require((magic,n,h,w) == (2051,60000,28,28), 'Unexpected upstream training image dimensions')
        pixels = np.frombuffer(incoming.read(), dtype=np.uint8).reshape(n,h,w)
    with gzip.open(sources/'fashion-labels.gz', 'rb') as incoming:
        require(struct.unpack('>2I', incoming.read(8)) == (2049,n), 'Invalid training label header')
        labels = np.frombuffer(incoming.read(), dtype=np.uint8)
    require(len(labels) == n, 'Unequal upstream image and label counts')
    yield 'd58', pixels, labels


def frozen_runtime():
    return dict(runtime=source_identity(ROOT), runner=sha(Path(__file__)), guard=sha(GUARD), protocol=sha(PLAN))


def prepare(a):
    study = a.root/'development'
    require(not (study/'registered.json').exists(), 'Already registered')
    study.mkdir(exist_ok=True)
    answers = study/'evaluation'
    answers.mkdir(exist_ok=True)
    plan = read(PLAN)
    sampling = plan['sampling']
    seed = sampling['seed']
    metadata = {}
    for case, x, y in raw_records(a.root/'sources'):
        folder = study/case
        folder.mkdir(exist_ok=True)
        ids = distinct(x, y)
        subset, _ = train_test_split(ids, train_size=sampling['training']+sampling['evaluation'],
                                     stratify=y[ids], random_state=seed)
        train, held = train_test_split(subset, train_size=sampling['training'], stratify=y[subset], random_state=seed)
        modality, metric = plan['sources'][case][2:]
        packet(folder/'task.zip',modality,metric,x[train],y[train])
        packet(folder/'query.zip',modality,metric,x[held])
        np.save(answers/(case+'.npy'), y[held], allow_pickle=False)
        metadata[case] = dict(modality=modality,metric=metric,upstream_rows=len(y),unique_unambiguous_rows=len(ids),
            train_rows=len(train),held_rows=len(held), train_sha256=sha(folder/'task.zip'),query_sha256=sha(folder/'query.zip'),
            truth_sha256=sha(answers/(case+'.npy')), train_ids_sha256=hashlib.sha256(train.astype('>u4').tobytes()).hexdigest(),
            held_ids_sha256=hashlib.sha256(held.astype('>u4').tobytes()).hexdigest())
    save(study/'registered.json', dict(frozen=frozen_runtime(), cases=metadata,
         upstream_sha256={p.name:sha(p) for p in sorted((a.root/'sources').iterdir()) if p.is_file()}))


def run(a, *, expected_fingerprint=None, additional_sealed_directory=None):
    study = a.root/'development'
    arm, case = a.arm, a.case
    registration = read(study/'registered.json')
    require(registration['frozen'] == (expected_fingerprint or frozen_runtime()), 'Runner or agent changed after registration')
    files = study/case
    meta = registration['cases'][case]
    train, query = files/'task.zip', files/'query.zip'
    require(sha(train) == meta['train_sha256'] and sha(query) == meta['query_sha256'], 'Changed study input')
    if arm == CAMPUS:
        require(read(study/'campus_gate.json')['case'] == case and a.outbox and a.key_file,
                'Live case must be selected before any outer scores')
    folder = files/arm
    folder.mkdir()
    physical = folder/'physical.jsonl'
    settings = env(a.weights, physical)
    for key in CONFIG:
        settings.pop(key,None)
    # Worker-specific limits are computed by the production executor. The observer counts components independently.
    settings.pop('AB_B1_PER_WORKER_FIT_LIMIT',None)
    settings.update(PYTHONPATH=os.pathsep.join(map(str,(GUARD.parent,ROOT))),
        JOINT_OUTER_ANSWERS=str(study/'evaluation'), JOINT_UPSTREAM_SOURCES=str(a.root/'sources'),
        JOINT_PREDICTION_INPUT=str(query), REFLECTION_STUDY_LIVE='0', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if isinstance(additional_sealed_directory,(tuple,list)):
        settings['JOINT_ADDITIONAL_SEALED_DIRECTORIES'] = json.dumps([str(p) for p in additional_sealed_directory])
    elif additional_sealed_directory is not None:
        settings['JOINT_ADDITIONAL_SEALED_DIRECTORY'] = str(additional_sealed_directory)
    if arm == CAMPUS:
        settings.update(REFLECTION_STUDY_LIVE='1', SUPERAGENT_PLANNER_KEY_FILE=str(a.key_file),
                        SUPERAGENT_RESEARCH_REFLECTION_OUTBOX=str(a.outbox), SUPERAGENT_TASK_BUDGET_ID='joint-nov27-'+case)
    run_dir = folder/'run'
    output=folder/'delivery/predictions.csv'
    start=time.monotonic()
    execute(['solve',train,'--run-dir',run_dir,'--prediction-input',query,'--output',output,
             '--fit-limit','11','--time-limit','600','--task-interpretation','fallback','--task-analysis',arm],
            folder/'solve.log',settings)
    return seal_run(folder, run_dir, output, query, settings, meta, case, arm, time.monotonic()-start, folder)


def seal_run(folder,run_dir,output,query,settings,meta,case,arm,elapsed,evidence_dir,*,recovery=None):
    physical=folder/'physical.jsonl'
    agent=run_dir/'agent'
    outcome=read(agent/'run_outcome.json')
    selected=read(agent/'final_selection.json')
    opening=read(agent/'task_opening.json')
    receipt=read(output.parent/'prediction_manifest.json')
    first=read(run_dir/'first_delivery/prediction_manifest.json')
    fit_count=len(physical.read_text().splitlines()) if physical.exists() else 0
    require(fit_count == outcome['attempted_classifier_fits'] <= 11 and outcome['new_api_requests'] <= int(arm == CAMPUS),
            'Fit or model-request accounting mismatch')
    require(outcome['status'] == 'completed' and not outcome['formal_test_accessed'] and selected['refit']
            and selected['training_rows'] == meta['train_rows'] and sha(agent/selected['model_file']) == selected['model_sha256'],
            'Incomplete search or final refit')
    require(receipt['prediction_count'] == first['prediction_count'] == meta['held_rows']
            and receipt['prediction_sha256'] == sha(output) and receipt['delivery_phase'] == 'selected',
            'Incomplete first or final delivery')
    replay=evidence_dir/'reload/predictions.csv'
    execute(['predict',query,'--run-dir',run_dir,'--output',replay],evidence_dir/'reload.log',settings)
    require(sha(replay) == sha(output) and len(physical.read_text().splitlines()) == fit_count, 'Reload changed output or refit')
    decisions=read(agent/'decisions.json')
    seal=dict(case=case,arm=arm,seconds=round(elapsed,2),fits=fit_count,requests=outcome['new_api_requests'],
              course_cost_rmb=outcome['complete_run_course_cost_rmb'],unresolved_requests=outcome['unresolved_api_requests'],
              model=selected['selected_experiment'],model_sha=selected['model_sha256'],
              output_sha=sha(output),early_sha=first['prediction_sha256'],fresh_reload=True,
              core_actions=[(d['choice']['action'],d['choice']['split'],d['choice']['members']) for d in decisions],
              core=opening['incumbent_members'],opportunity=opening.get('opportunity'),check=opening.get('check'),
              verdict=opening['check_status'],local_candidate=opening.get('local_experiment'),tested=opening.get('tested_candidate'),
              changed=opening['changed_experiment'],remaining=opening['remaining_at_delivery'],result=opening['result'],
              comparisons=opening.get('comparisons',[]),interpretation=opening['interpretation_status'],
              source_at_decision=opening.get('hypothesis_source_at_decision'))
    if recovery is not None:
        seal['export_recovery'] = recovery
    save(evidence_dir/'sealed.json',seal)
    print(case,arm,fit_count,'fits',opening['check_status'],opening['result'],flush=True)


def gate(a):
    study=a.root/'development'
    cases=[]
    for case in CASES:
        control,local=[read(study/case/arm/'sealed.json') for arm in ARMS]
        require(control['opportunity'] == local['opportunity'] and control['core_actions'] == local['core_actions'],
                'Arms must share ordinary search and opportunity before the intervention')
        if local['verdict']=='supported' and local['tested'] and local['comparisons'] and local['opportunity']['reserved_fits']>0:
            cases.append(case)
    save(study/'campus_gate.json',dict(case=next(iter(cases),None),eligible=cases,outer_scores_opened=False))
    print('Campus eligibility:',cases,flush=True)


def metrics(metric, truth, values):
    if metric=='macro_f1': return float(f1_score(truth,values,average='macro',zero_division=0))
    if metric=='balanced_accuracy': return float(balanced_accuracy_score(truth,values))
    return float(accuracy_score(truth,values))


def score(a):
    study=a.root/'development'
    registration=read(study/'registered.json')
    gate_record=read(study/'campus_gate.json')
    live=gate_record['case']
    if live and not (study/live/CAMPUS/'sealed.json').exists():
        require((study/'campus_unavailable.json').exists(),'Selected live arm has no seal or documented external blocker')
    reports={}
    for case in CASES:
        meta=registration['cases'][case]
        arms=[*ARMS]+([CAMPUS] if case==live and (study/live/CAMPUS/'sealed.json').is_file() else [])
        truth_file=study/'evaluation'/(case+'.npy')
        require(sha(truth_file)==meta['truth_sha256'],'Truth changed')
        truth=np.load(truth_file,allow_pickle=False)
        details={}
        predictions={}
        for arm in arms:
            folder=study/case/arm
            seal=read(folder/'sealed.json')
            output=folder/'delivery/predictions.csv'
            require(sha(output)==seal['output_sha'],'Changed prediction')
            with output.open(newline='') as f:
                rows=list(csv.DictReader(f))
            require([row['row_id'] for row in rows] == list(map(str,range(len(truth)))),'Lost or permuted rows')
            yhat=np.asarray([row['prediction'] for row in rows],dtype=truth.dtype)
            predictions[arm]=yhat
            details[arm]=dict(seal,primary_score=metrics(meta['metric'],truth,yhat),correct=int(sum(truth==yhat)))
        ctrl=predictions[ARMS[0]]
        pairs={}
        for arm in arms[1:]:
            yhat=predictions[arm]
            pairs[arm]=dict(disagreements=int(sum(yhat!=ctrl)),gained=int(sum((yhat==truth)&(ctrl!=truth))),
                            lost=int(sum((yhat!=truth)&(ctrl==truth))),
                            primary_gain=details[arm]['primary_score']-details[ARMS[0]]['primary_score'])
        reports[case]=dict(metric=meta['metric'],rows=len(truth),arms=details,paired_vs_control=pairs)
    save(study/'scored.json',dict(registration=registration,cases=reports,campus_gate=gate_record))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','gate','score'))
    parser.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    parser.add_argument('--weights',type=lambda s:Path(s).resolve(),default=ROOT.parent/'pretrained_weights')
    parser.add_argument('--outbox',type=lambda s:Path(s).resolve())
    parser.add_argument('--key-file',type=lambda s:Path(s).resolve())
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--arm',choices=(*ARMS,CAMPUS))
    args=parser.parse_args()
    if args.action=='run' and (not args.case or not args.arm):parser.error('run requires case and arm')
    {'prepare':prepare,'run':run,'gate':gate,'score':score}[args.action](args)
