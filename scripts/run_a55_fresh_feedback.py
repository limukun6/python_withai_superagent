"""Fresh raw/checked decisions on identical calibrated states, then deduplicated action audits."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import joblib

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import physical
from scripts.campus_direct import campus_resolution,CAMPUS_IP
from superagent.ab_flash import context_for,validate_proposal,preflight,MODEL,ENDPOINT
from superagent.ab_run import sources
from superagent.benchmark import train_candidate
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.metrics import MetricSpec
from superagent.research_budget import IncrementalResearchAccount,research_preflight
from superagent.splits import grouped_partition,take,overlap_audit
from superagent.task_semantics import declared_semantics


ROOT=Path(__file__).resolve().parents[1];SEEDS=(20731,20741)


def remaining(args):
    deadline=datetime.fromisoformat(json.loads(args.window_config.read_text(encoding='utf-8-sig'))['deadline'])
    return (deadline-datetime.now(timezone.utc)).total_seconds()


def prepare(args,root,frozen):
    minimum=2700 if args.prefix_count==3 else 1200
    if remaining(args)<minimum:raise ValueError('Insufficient window for this predeclared prefix count')
    if read(args.study/'completion.json')['status']!='completed' or read(args.study/'protocol.json')['source_files']!=frozen:
        raise ValueError('Completed same-source study required')
    entries={e['task']:e for e in read(args.study/'protocol.json')['entries']};states=[]
    for name in ('coupon','preference'):
        d=args.study/(name+'_11');caps=read(d/'run_manifest.json')['capabilities']
        events=[json.loads(line) for line in (d/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        requests=[e for e in events if e['kind']=='llm_request'][:3]
        original=[r['proposal'] for r in read(d/'history.json') if r.get('proposal')][:3]
        if not requests or len(requests)!=len(original):raise ValueError('Cannot bind original proposal to request')
        folder=root/name;folder.mkdir()
        indices=range(len(requests)) if args.prefix_count==3 else [len(requests)-1]
        for index in indices:
            request,answer=requests[index],original[index]
            raw=json.loads(request['messages'][1]['content']);trial=len(raw['history'])+1;state=read(d/f'state_{trial}.json')
            ids=tuple(c['id'] for c in raw['available_candidates']);rebuilt=context_for(state,caps,ids,False,'calibrated-v2')
            if raw!=rebuilt:raise ValueError('Raw reconstruction differs')
            checked=context_for(state,caps,ids,True,'calibrated-v2')
            if raw!={k:v for k,v in checked.items() if k not in ('checks','memory')}:raise ValueError('Common inputs differ')
            path=folder/f'state_{index+1}.json';save(path,dict(state=state,capabilities=caps,eligible_ids=ids,R=raw,E=checked))
            states.append(dict(state_id=f'{name}_{index+1}',task=name,index=index,state_file=str(path),state_sha256=sha(path),
                original_candidate=answer['candidate'],original_request_id=request['budget_request_id'],source_request_ordinal=index+1))
        entry=entries[name];archive=Path(entry['training_archive'])
        if sha(archive)!=entry['training_archive_sha256']:raise ValueError('Training changed')
        task=load_training_task(archive);metric=MetricSpec.for_task(task);payloads={}
        seen={frozenset(read(args.study/'inputs'/name/(s+'_split.json'))['search_indices']) for s in entry['splits']}
        for seed in SEEDS:
            train,held,resource=grouped_partition(task,seed,'shared_text')
            if frozenset(held.tolist()) in seen:raise ValueError('Audit repeats exposed split')
            seen.add(frozenset(held.tolist()));path=folder/f'audit_{seed}.joblib'
            joblib.dump(dict(kind=task.kind,X_train=take(task.X_train,train),y_train=task.y_train[train],X_search=take(task.X_train,held),
                y_search=task.y_train[held],metric_spec=metric.record(),pair_semantics=declared_semantics(task)),path)
            payloads[str(seed)]=dict(path=str(path),sha256=sha(path))
            save(folder/f'audit_{seed}.json',dict(train_indices=train.tolist(),search_indices=held.tolist(),resource=resource,
                overlap=overlap_audit(task,train,held),payload_sha256=sha(path)))
        entries[name]={**entry,'audit_payloads':payloads}
    save(root/'states.json',states);save(root/'readiness.json',list(entries.values()))
    return dict(status='prepared',states=len(states),prefix_count=args.prefix_count,
        sampling_rule='first_up_to_three' if args.prefix_count==3 else 'last_available_request',actual_classifier_fits=0,real_api_requests=0)


def ask(args,root,frozen):
    if read(args.prepared/'protocol.json')['source_files']!=frozen:raise ValueError('Prepared source differs')
    entries={e['task']:e for e in read(args.prepared/'readiness.json')};states=read(args.prepared/'states.json');clients={};loggers={};answers=[]
    ledger=None
    for name,entry in entries.items():
        options=SimpleNamespace(api_url=None,model=None,api_key_file=args.api_key_file,budget_ledger=args.ledger,
            task_budget_id=entry['task_budget_id'],llm_budget_rmb=.30,history_policy='incremental-research')
        ledger,_,_=preflight(options);research_preflight(ledger,entry['task_budget_id'],.60)
    if not args.execute:
        return dict(status='validated_not_executed',states=len(states),real_api_requests=0,actual_classifier_fits=0)
    for name,entry in entries.items():
        for mode in ('R','E'):
            folder=root/(name+'_'+mode);folder.mkdir();logger=RunLogger(folder);logger.initialize_usage_ledger();loggers[name,mode]=logger
            account=IncrementalResearchAccount(ledger,entry['task_budget_id'],str(folder))
            client=ExperimentLLM(ENDPOINT,str(args.api_key_file),MODEL,.30,max_requests=3,allowed_served_models={MODEL},account=account,streaming=True)
            client.set_deadline(time.monotonic()+remaining(args));clients[name,mode]=client
    with campus_resolution(CAMPUS_IP):
        for position,item in enumerate(states):
            if remaining(args)<150:raise ValueError('No time for another paired question')
            if sources()!=frozen or sha(Path(item['state_file']))!=item['state_sha256']:raise ValueError('Frozen source/state changed')
            document=read(Path(item['state_file']));choices={}
            for mode in (('R','E') if position%2==0 else ('E','R')):
                answer=clients[item['task'],mode].ask(document[mode],loggers[item['task'],mode],'fixed_calibrated_'+mode)
                proposal=validate_proposal(answer,document['state'],document['capabilities'],tuple(document['eligible_ids']),'calibrated-v2')
                choices[mode]=dict(candidate=proposal.candidate,proposal=proposal.record())
                save(root/(item['state_id']+'_'+mode+'.json'),choices[mode])
            answers.append(dict(**item,**choices));save(root/'answers.json',answers)
            print(json.dumps(dict(state=item['state_id'],R=choices['R']['candidate'],E=choices['E']['candidate'])),flush=True)
    usage=[json.loads(line) for file in root.glob('*/llm_usage.jsonl') for line in file.read_text(encoding='utf-8').splitlines()]
    calls=[r for r in usage if r['event']=='llm_call']
    return dict(status='completed',paired_states=len(answers),real_api_requests=len(calls),actual_classifier_fits=0,
        known_new_course_cost_rmb=sum(r['course_cost_rmb'] for r in calls),input_tokens=sum(r['input_tokens'] for r in calls),
        output_tokens=sum(r['output_tokens'] for r in calls),project_usage_after=ledger.project_summary(),historical_answers_reused_as_new=False)


def evaluate(args,root,frozen):
    if (read(args.answers/'completion.json')['status']!='completed' or read(args.prepared/'protocol.json')['source_files']!=frozen
            or read(args.answers/'protocol.json')['source_files']!=frozen):
        raise ValueError('All new paired answers must freeze first on the same source')
    answers=read(args.answers/'answers.json');entries={e['task']:e for e in read(args.prepared/'readiness.json')}
    unique=list(dict.fromkeys((r['task'],r[m]['candidate']) for r in answers for m in ('R','E')));results=[]
    observer=ROOT/'scripts/a29_worker_observer/sitecustomize.py'
    os.environ.update(AB_A29_OBSERVE_FITS='1',AB_A29_FIT_LOG=str(root/'physical_fits.jsonl'),PYTHONPATH=os.pathsep.join([str(observer.parent),str(ROOT)]),
        SUPERAGENT_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    save(root/'work_order.json',[dict(task=t,candidate=c,seed=s) for t,c in unique for s in SEEDS])
    for index,(name,candidate) in enumerate(unique):
        for seed in SEEDS:
            if remaining(args)<90 or sources()!=frozen:raise ValueError('No time/source changed before worker')
            prefix=root/f'action_{index}_{seed}';admission=prefix.with_suffix('.admission.json');countfile=prefix.with_suffix('.fits.json')
            save(admission,dict(maximum_classifier_fits=1,ledger_path=str(countfile)));os.environ['AB_A29_RUN_NAME']=prefix.name
            payload=entries[name]['audit_payloads'][str(seed)]
            if sha(Path(payload['path']))!=payload['sha256']:raise ValueError('Payload changed')
            model,facts=train_candidate(Path(payload['path']),candidate,prefix.with_suffix('.joblib'),60,fit_admission_path=admission)
            count=read(countfile) if countfile.exists() else None
            results.append(dict(task=name,candidate=candidate,seed=seed,facts=facts,fit_accounting=count));save(root/'results.json',results)
            if model is None or count is None or count['status']!='completed' or count['completed_classifier_fits']!=1:raise ValueError('Audit failed without retry')
    comparisons=[]
    for item in answers:
        value={}
        for mode in ('R','E'):
            candidate=item[mode]['candidate'];scores=[next(r['facts']['search_score'] for r in results if r['task']==item['task'] and r['candidate']==candidate and r['seed']==seed) for seed in SEEDS]
            value[mode]=dict(candidate=candidate,scores=scores,mean=sum(scores)/2)
        comparisons.append(dict(state_id=item['state_id'],task=item['task'],**value,E_minus_R=value['E']['mean']-value['R']['mean'],
            original_R_candidate=item['original_candidate'],fresh_R_changed=value['R']['candidate']!=item['original_candidate']))
    save(root/'comparison.json',comparisons)
    if len(physical(root))!=len(results) or len(results)>4*len(answers):raise ValueError('Physical accounting differs')
    return dict(status='completed',paired_states=len(answers),actual_classifier_fits=len(results),logical_evaluations=4*len(answers),
        reused_evaluations=4*len(answers)-len(results),real_api_requests=0,external_validation_opened=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','ask','evaluate'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--window-config',type=Path,required=True)
    for name in ('study','prepared','answers','ledger','api-key-file'):p.add_argument('--'+name,type=Path)
    p.add_argument('--prefix-count',type=int,choices=[1,3],default=1);p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.phase!='prepare':args.prefix_count=read(args.prepared/'protocol.json')['prefix_count']
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();frozen=sources();save(root/'protocol.json',dict(batch='A55-'+args.phase,source_files=frozen,seeds=SEEDS,
        prefix_count=args.prefix_count,sampling_rule='first_up_to_three' if args.prefix_count==3 else 'last_available_request',
        runner_sha256=sha(Path(__file__)),explicit_process_dns=CAMPUS_IP,
        plan_sha256=sha(ROOT/'docs/batches/A55_fresh_feedback_plan.md'),automatic_retry=False,
        course_budget_compliance='unverified_history',valid_for_budgeted_final_report=False))
    try:
        result=globals()[args.phase](args,root,frozen);result.update(elapsed_seconds=time.monotonic()-start,source_unchanged=sources()==frozen)
        save(root/'completion.json',result);print(json.dumps(result))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),automatic_retry=False,observed_fit_entries=len(physical(root)),elapsed_seconds=time.monotonic()-start));raise
