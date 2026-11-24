"""Prepare real training-only states; collect bounded paired Flash decisions with a real ledger."""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent.parent if REPO.name=='integration' and REPO.parent.name=='runs' else REPO
sys.path.insert(0,str(REPO))
TASKS={'task1':'released_task1.zip','task2':'released_task2.zip','dry_bean':'dry_bean_numeric_tabular.zip','sms_spam':'sms_spam_single_text.zip'}
SCOPES={'task1':'task1','task2':'task2','dry_bean':'dry_bean','sms_spam':'sms_spam'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def sources():return {p.name:sha(p) for p in sorted((REPO/'superagent').glob('*.py'))}
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def prepare(out):
    from superagent.benchmark import run_one
    from superagent.contracts import DecisionState,TaskSpec
    from superagent.checked_history_policy import decision_context
    from superagent.models import search_candidates
    from tools.search_budget_study import outer_partition
    frozen=sources();start=time.monotonic();fits=0;states=[]
    protocol=dict(status='planned',tasks={k:{'archive':str(ROOT/'data'/v),'sha256':sha(ROOT/'data'/v),'scope':SCOPES[k]} for k,v in TASKS.items()},
        source_sha256=frozen,runner_sha256=sha(__file__),reserve_seed=41101,outer_seed=45101,search_seed=707,
        evaluation_seeds=[2707,3707],api_requests_max=24,prepare_classifier_fits_max=12,
        model='DeepSeek-V4.1-Flash',allowance_per_task_rmb=.30,
        scopes=SCOPES,scope_note='No dataset identity passed as a method-selection key',
        dependencies={n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn','joblib','catboost')})
    save(out/'protocol.json',protocol)
    unresolved=ROOT/'runs/flash_history_unresolved.json'
    gaps=read(unresolved) if unresolved.is_file() else []
    save(out/'accounting_preflight.json',dict(existing_unresolved_inventory=str(unresolved),unresolved_inventory_sha256=sha(unresolved) if gaps else None,
        unresolved_inventory_count=len(gaps),real_ledger_selected=False,ready_to_call=False,
        reason='Verified cumulative real task and shared history must be imported; fixture ledgers are ineligible'))
    try:
        for name,filename in TASKS.items():
            task,held_x,held_y,indices=outer_partition(ROOT/'data'/filename,45101)
            # Outer data never enters the search runner or decision state.
            save(out/(name+'_partition.json'),indices)
            summary=run_one(task,'local_interleaved_routes',707,out/name,None,max_trials=3,time_limit=1200,
                trial_timeout=90,anchor_first=True,confirmation_repeats=1,refit_full_train=False,
                portfolio='expanded',error_feedback=True,feature_cache=True)
            fits+=summary['classifier_fit_accounting']['completed_classifier_fits']
            if summary['failed_trials'] or summary['trials_attempted']!=3 or not summary['valid_policy_run']:
                raise RuntimeError(f'Incomplete history for {name}; see retained run artifacts')
            history=read(out/name/'trials.json');recorded=read(out/name/'decision_states.json')
            catalog=[dict(id=c.family,params=c.params) for c in search_candidates(task.kind,'expanded')]
            available=[c for c in catalog if c['id'] not in {r['candidate'] for r in history}]
            final=DecisionState.build(read(out/name/'data_profile.json'),history,available,4,True,TaskSpec.from_task(task)).record()
            for prefix,record in ((1,recorded[1]),(2,recorded[2]),(3,final)):
                # All prefixes have the same total decision budget.
                record['trial_budget']={'maximum':4,'attempted':prefix,'remaining':4-prefix}
                raw=decision_context(DecisionState(record),False);checked=decision_context(DecisionState(record),True)
                shared=deepcopy(checked);shared.pop('checks');shared.pop('memory')
                if raw!=shared or len(record['history'])!=prefix or record['next_trial']!=prefix+1:
                    raise ValueError('Unmatched decision state')
                filename=f'{name}_prefix{prefix}.json';save(out/filename,record)
                states.append(dict(task=name,prefix=prefix,file=filename,state_sha256=sha(out/filename),
                    raw_context_sha256=fingerprint(raw),checked_context_sha256=fingerprint(checked),primary_metric=record['task_spec']['metric']['name']))
                save(out/'states.json',states)
            print(f'{name}: 3 real search fits; metric={summary["primary_metric"]}; frozen 3 prefixes',flush=True)
        if sources()!=frozen:raise RuntimeError('Production source changed while preparing states')
        save(out/'completion.json',dict(status='prepared',states=len(states),classifier_fits=fits,api_requests=0,
            elapsed_seconds=time.monotonic()-start,source_unchanged=True,ready_for_ask_requires_verified_ledger=True))
    except Exception as error:
        save(out/'completion.json',dict(status='stopped',error=f'{type(error).__name__}: {error}',
            completed_fits_from_finished_tasks=fits,api_requests=0,elapsed_seconds=time.monotonic()-start))
        raise

def ask(args,out):
    from superagent.task_budget import TaskBudgetLedger,TaskBudgetAccount
    from superagent.experiment_llm import ExperimentLLM
    from superagent.checked_history_policy import CheckedHistoryPolicy,decision_context
    from superagent.contracts import DecisionState
    from superagent.events import RunLogger
    from superagent.cli import CAMPUS_CHAT_ENDPOINT,CAMPUS_FLASH_MODEL
    prepared=Path(args.prepared).resolve();protocol=read(prepared/'protocol.json');states=read(prepared/'states.json')
    if read(prepared/'completion.json')['status']!='prepared' or protocol['source_sha256']!=sources():
        raise ValueError('Prepared batch incomplete or production source changed; freeze a new preparation version')
    ledger=TaskBudgetLedger(args.ledger)
    # Do not use a mock/new-project ledger to bypass historical uncertainty.
    import sqlite3
    with sqlite3.connect(ledger.path) as db:
        origin=db.execute("SELECT value FROM metadata WHERE key='origin_sha256'").fetchone()[0]
    if origin=='explicit_new_project':raise ValueError('A2 requires verified historical import, not an empty or fixture ledger')
    before={scope:ledger.preflight(scope,.30) for scope in sorted(set(protocol['scopes'].values()))}
    save(out/'protocol.json',dict(prepared=str(prepared),prepared_protocol_sha256=sha(prepared/'protocol.json'),
        ledger=str(ledger.path),ledger_before=before,maximum_requests=24,source_sha256=sources()))
    logger=RunLogger(out);logger.initialize_usage_ledger();clients={};results=[];start=time.monotonic()
    try:
        for index,entry in enumerate(states):
            path=prepared/entry['file']
            if sha(path)!=entry['state_sha256']:raise ValueError('Frozen state changed')
            record=read(path);state=DecisionState(record);scope=protocol['scopes'][entry['task']]
            if scope not in clients:
                clients[scope]=ExperimentLLM(CAMPUS_CHAT_ENDPOINT,args.api_key_file,CAMPUS_FLASH_MODEL,.30,
                    max_requests=6,streaming=True,account=TaskBudgetAccount(ledger,scope,str(out)),
                    allowed_served_models={CAMPUS_FLASH_MODEL,'DSF'})
            for checked in ((False,True) if index%2==0 else (True,False)):
                if sources()!=protocol['source_sha256']:raise RuntimeError('Production source changed during calls')
                context=decision_context(state,checked)
                expected=entry['checked_context_sha256' if checked else 'raw_context_sha256']
                if fingerprint(context)!=expected:raise ValueError('Frozen context changed')
                logger.event('state_call',task=entry['task'],prefix=entry['prefix'],checked=checked,context_sha256=expected)
                proposal=CheckedHistoryPolicy(clients[scope],logger,checked).propose(state)
                results.append(dict(task=entry['task'],prefix=entry['prefix'],checked=checked,proposal=proposal.record()))
                save(out/'results.json',results)
                print(f'{entry["task"]} prefix{entry["prefix"]} {"checked" if checked else "raw"}: {proposal.candidate}',flush=True)
        status='complete'
    except Exception as error:
        status='stopped'
        detail=f'{type(error).__name__}: {error}'
        for client in clients.values():detail=detail.replace(client.key,'[redacted]')
        save(out/'failure.json',dict(error=detail,completed_proposals=len(results)))
        print(detail,flush=True)
    save(out/'completion.json',dict(status=status,api_requests=sum(c.request_count for c in clients.values()),
        valid_proposals=len(results),known_course_cost_rmb=sum(c.spent for c in clients.values()),
        elapsed_seconds=time.monotonic()-start,task_budget_after={s:ledger.snapshot(s) for s in before}))
    return int(status!='complete')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','ask'])
    parser.add_argument('--output',required=True);parser.add_argument('--prepared');parser.add_argument('--ledger')
    parser.add_argument('--api-key-file',default=str(ROOT/'api_key.txt'));args=parser.parse_args()
    if args.mode=='ask' and (not args.prepared or not args.ledger):parser.error('ask requires prepared and verified historical ledger')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if args.mode=='prepare':prepare(out);return 0
    try:return ask(args,out)
    except Exception as error:
        save(out/'completion.json',dict(status='blocked_before_request',error=f'{type(error).__name__}: {error}',api_requests=0))
        raise

if __name__=='__main__':raise SystemExit(main())
