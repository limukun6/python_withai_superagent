"""Reused real raw proposals vs one checked proposal per frozen state, then fresh action audits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import joblib

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from scripts.run_a29_choice_matrix import physical
from superagent.ab_run import sources
from superagent.ab_flash import context_for, validate_proposal, preflight, MODEL, ENDPOINT
from superagent.benchmark import train_candidate
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.research_budget import IncrementalResearchAccount
from superagent.splits import grouped_partition, take, overlap_audit
from superagent.task_semantics import declared_semantics


ROOT=Path(__file__).resolve().parents[1]
SEEDS=(18331,18341)


def jsonhash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def remaining(args):
    return (datetime.fromisoformat(args.window_deadline)-datetime.now(timezone.utc)).total_seconds() if args.window_deadline else float('inf')


def prepare(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();states=[];entries=[]
    available={entry['task']:entry for entry in read(args.training_prepared/'readiness.json')}
    names=('coupon','preference') if args.task_set=='course' else ('sms','photo') if args.task_set=='multimodal' else ('coupon','preference','sms','photo')
    save(root/'protocol.json',dict(batch='A35-prepare',source_files=frozen,seeds=SEEDS,task_set=args.task_set,
        runner_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'docs/batches/A35_reused_raw_fixed_state_plan.md')))
    for name in names:
        parent=args.matrix if name in ('coupon','preference') else args.transfer
        if parent is None or read(parent/'completion.json')['status']!='completed':raise ValueError('Raw source study must be complete')
        if read(parent/'protocol.json')['source_files']!=frozen:raise ValueError('Raw study source differs')
        directory=parent/(name+'_R1_off');manifest=read(directory/'run_manifest.json');caps=manifest['capabilities']
        records=[json.loads(line) for line in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        requests=[row for row in records if row['kind']=='llm_request'][:3]
        proposals=[row for row in records if row['kind']=='ab_flash_proposal'][:3]
        if not 1<=len(requests)==len(proposals)<=3:raise ValueError('Cannot bind original requests/proposals')
        folder=root/name;folder.mkdir()
        for index,(request,proposal_event) in enumerate(zip(requests,proposals,strict=True)):
            original=json.loads(request['messages'][1]['content']);trial=len(original['history'])+1
            state=read(directory/f'state_{trial}.json');ids=tuple(row['id'] for row in original['available_candidates'])
            rebuilt=context_for(state,caps,ids,False)
            if rebuilt!=original:raise ValueError('Rebuilt raw prompt differs; cannot claim paired same history')
            raw=proposal_event['proposal']['raw_proposal'];validate_proposal(raw,state,caps,ids)
            checked=context_for(state,caps,ids,True)
            if {k:v for k,v in checked.items() if k not in ('checks','memory')}!=original:raise ValueError('Extra fields changed')
            stateid=f'{name}_{index+1}';path=folder/f'state_{index+1}.json'
            save(path,dict(state=state,capabilities=caps,eligible_ids=ids,checked_payload=checked,raw_answer=raw))
            states.append(dict(state_id=stateid,task=name,index=index+1,trial=trial,state_file=str(path),
                state_sha256=sha(path),original_messages_sha256=jsonhash(request['messages']),
                original_budget_request_id=request['budget_request_id'],raw_candidate=raw['candidate'],raw_answer_reused=True))
        entry=available[name]
        if sha(Path(entry['training_archive']))!=entry['training_archive_sha256']:raise ValueError('Train-only input changed')
        task=load_training_task(entry['training_archive']);metric=MetricSpec.for_task(task)
        entry={**entry,'audit_payloads':{}}
        previous={frozenset(read(args.training_prepared/name/(split+'_split.json'))['search_indices']) for split in entry['splits']}
        for seed in SEEDS:
            train,held,resource=grouped_partition(task,seed,'shared_text')
            if frozenset(held.tolist()) in previous:raise ValueError('Fresh action split repeats exposed rows')
            previous.add(frozenset(held.tolist()));payload=folder/f'audit_{seed}.joblib'
            joblib.dump(dict(kind=task.kind,X_train=take(task.X_train,train),y_train=task.y_train[train],
                X_search=take(task.X_train,held),y_search=task.y_train[held],metric_spec=metric.record(),pair_semantics=declared_semantics(task)),payload)
            entry['audit_payloads'][str(seed)]=dict(path=str(payload),sha256=sha(payload))
            save(folder/f'audit_{seed}.json',dict(seed=seed,train_indices=train.tolist(),search_indices=held.tolist(),
                row_ids_sha256=fingerprint(held.tolist()),payload_sha256=sha(payload),resource=resource,overlap=overlap_audit(task,train,held)))
        entries.append(entry)
    save(root/'states.json',states);save(root/'readiness.json',entries)
    save(root/'completion.json',dict(status='prepared',states=len(states),reused_R_answers=len(states),actual_classifier_fits=0,real_api_requests=0))
    print(json.dumps(dict(status='prepared',states=len(states),reused_R_answers=len(states))))


def ask(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();results=[];began=time.monotonic()
    protocol=dict(batch='A35-ask',source_files=frozen,runner_sha256=sha(Path(__file__)),execute_requested=args.execute,
        window_deadline=args.window_deadline,history_policy='incremental-research',course_budget_compliance='unverified_history')
    save(root/'protocol.json',protocol)
    try:
        if read(args.prepared/'protocol.json')['source_files']!=frozen:raise ValueError('Frozen state source differs')
        entries={entry['task']:entry for entry in read(args.prepared/'readiness.json')};states=read(args.prepared/'states.json')
        protocol.update(states=states,maximum_new_requests=len(states),reused_R_requests=len(states))
        ledger=None
        for entry in entries.values():
            options=SimpleNamespace(api_url=None,model=None,api_key_file=args.api_key_file,budget_ledger=args.ledger,
                task_budget_id=entry['task_budget_id'],llm_budget_rmb=.30,history_policy='incremental-research')
            ledger,_,_=preflight(options)
        protocol['project_usage_before']=ledger.project_summary();save(root/'protocol.json',protocol)
        if not args.execute:
            save(root/'completion.json',dict(status='validated_not_executed',real_api_requests=0,actual_classifier_fits=0));return
        for name,entry in entries.items():
            if remaining(args)<600:raise RuntimeError('Insufficient window to begin a new checked task block')
            folder=root/name;folder.mkdir();logger=RunLogger(folder);logger.initialize_usage_ledger()
            account=IncrementalResearchAccount(ledger,entry['task_budget_id'],str(folder))
            client=ExperimentLLM(ENDPOINT,str(args.api_key_file),MODEL,.30,max_requests=3,allowed_served_models={MODEL},account=account,streaming=True)
            client.set_deadline(time.monotonic()+min(600,remaining(args)))
            for item in [row for row in states if row['task']==name]:
                if sources()!=frozen or sha(Path(item['state_file']))!=item['state_sha256']:raise ValueError('Frozen state/code changed')
                document=read(Path(item['state_file']))
                answer=client.ask(document['checked_payload'],logger,item['state_id'])
                proposal=validate_proposal(answer,document['state'],document['capabilities'],tuple(document['eligible_ids']))
                results.append(dict(state_id=item['state_id'],task=name,raw_candidate=item['raw_candidate'],
                    checked_candidate=proposal.candidate,checked_proposal=proposal.record(),raw_answer_reused=True))
                save(root/'answers.json',results)
                print(json.dumps(dict(state_id=item['state_id'],status='answer_frozen',raw=item['raw_candidate'],checked=proposal.candidate)),flush=True)
        usage=[json.loads(line) for p in root.glob('*/llm_usage.jsonl') for line in p.read_text(encoding='utf-8').splitlines()]
        save(root/'completion.json',dict(status='completed',states=len(results),real_new_api_requests=sum(row['event']=='llm_call' for row in usage),
            reused_R_requests=len(states),known_new_course_cost_rmb=sum(row.get('course_cost_rmb') or 0 for row in usage),
            actual_classifier_fits=0,source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-began,
            project_usage_after=ledger.project_summary(),course_budget_compliance='unverified_history',valid_for_budgeted_final_report=False))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),automatic_retry=False,answers_frozen=len(results),actual_classifier_fits=0,
            elapsed_seconds=time.monotonic()-began));raise


def evaluate(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();start=time.monotonic();results=[]
    save(root/'protocol.json',dict(batch='A35-evaluate',source_files=frozen,seeds=SEEDS,runner_sha256=sha(Path(__file__)),real_api_requests=0))
    try:
        if read(args.answers/'completion.json')['status']!='completed':raise ValueError('All new checked answers must be frozen first')
        if read(args.prepared/'protocol.json')['source_files']!=frozen or read(args.answers/'protocol.json')['source_files']!=frozen:
            raise ValueError('Audit source differs')
        entries={entry['task']:entry for entry in read(args.prepared/'readiness.json')};answers=read(args.answers/'answers.json')
        unique=list(dict.fromkeys((row['task'],row[key]) for row in answers for key in ('raw_candidate','checked_candidate')))
        save(root/'work_order.json',[dict(task=t,candidate=c,seed=s) for t,c in unique for s in SEEDS])
        observer=ROOT/'scripts/a29_worker_observer/sitecustomize.py'
        os.environ.update(AB_A29_OBSERVE_FITS='1',AB_A29_FIT_LOG=str(root/'physical_fits.jsonl'),
            PYTHONPATH=os.pathsep.join([str(observer.parent),str(ROOT)]),SUPERAGENT_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for index,(name,candidate) in enumerate(unique):
            for seed in SEEDS:
                if remaining(args)<90:raise RuntimeError('Window too short for another audit worker')
                if sources()!=frozen:raise ValueError('Production source changed')
                payload=entries[name]['audit_payloads'][str(seed)]
                if sha(Path(payload['path']))!=payload['sha256']:raise ValueError('Audit payload changed')
                prefix=root/f'action_{index}_{seed}';admission=prefix.with_suffix('.admission.json');countfile=prefix.with_suffix('.fits.json')
                save(admission,dict(maximum_classifier_fits=1,ledger_path=str(countfile)))
                os.environ['AB_A29_RUN_NAME']=prefix.name
                artifact,facts=train_candidate(Path(payload['path']),candidate,prefix.with_suffix('.joblib'),60,fit_admission_path=admission)
                count=read(countfile) if countfile.exists() else None
                results.append(dict(task=name,candidate=candidate,seed=seed,facts=facts,fit_accounting=count));save(root/'results.json',results)
                if artifact is None or count is None or count['completed_classifier_fits']!=1 or count['status']!='completed':raise RuntimeError('Audit worker failed without retry')
                print(json.dumps(dict(task=name,candidate=candidate,seed=seed,score=facts['search_score'])),flush=True)
        comparisons=[]
        for row in answers:
            values={}
            for mode,key in [('R','raw_candidate'),('E','checked_candidate')]:
                scores=[next(item['facts']['search_score'] for item in results if item['task']==row['task'] and item['candidate']==row[key] and item['seed']==seed) for seed in SEEDS]
                values[mode]=dict(candidate=row[key],scores=scores,mean=sum(scores)/2)
            comparisons.append(dict(state_id=row['state_id'],task=row['task'],**values,checked_minus_raw=values['E']['mean']-values['R']['mean']))
        fits=len(physical(root))
        if fits!=len(results) or fits>4*len(answers):raise ValueError('Physical fit count differs')
        save(root/'comparison.json',comparisons)
        save(root/'completion.json',dict(status='completed',actual_classifier_fits=fits,logical_evaluations=4*len(answers),
            reused_evaluations=4*len(answers)-fits,real_api_requests=0,source_unchanged=sources()==frozen,
            elapsed_seconds=time.monotonic()-start,external_validation_opened=False))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),automatic_retry=False,observed_fit_entries=len(physical(root)),real_api_requests=0));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','ask','evaluate'])
    parser.add_argument('--output',type=Path,required=True)
    for name in ('matrix','transfer','training-prepared','prepared','answers','ledger','api-key-file'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--task-set',choices=['course','multimodal','all-four'],default='course');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--window-deadline');args=parser.parse_args()
    {'prepare':prepare,'ask':ask,'evaluate':evaluate}[args.phase](args)
