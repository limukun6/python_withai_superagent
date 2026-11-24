"""Same-source four-task fixed-state feedback comparison using already-accounted local histories."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

import joblib
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts import run_a55_fresh_feedback as engine
from superagent.ab_flash import context_for
from superagent.ab_run import sources
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.splits import grouped_partition,take,overlap_audit
from superagent.task_semantics import declared_semantics

ROOT=Path(__file__).resolve().parents[1];SEEDS=(30311,30321)


def prepare(a,root,frozen):
    if read(a.study/'completion.json')['status']!='completed' or read(a.study/'protocol.json')['source_files']!=frozen:
        raise ValueError('A69 must first complete on the same production source')
    original=read(a.reference/'protocol.json')['entries'];entries=[];states=[]
    for name in ('coupon','preference','sms','bean'):
        directory=a.study/f'{name}_707_ab-v1';manifest=read(directory/'run_manifest.json');caps=manifest['capabilities']
        history=read(directory/'history.json');primary=[r for r in history if r['split_id']=='primary']
        if len(primary)<3 or any(r['split_id']!='primary' for r in history[:3]):raise ValueError('No common three-primary opening')
        folder=root/name;folder.mkdir()
        for prefix in (1,2,3):
            state=read(directory/f'state_{prefix+1}.json')
            if len(state['history'])!=prefix:raise ValueError('Opening prefix identity mismatch')
            state['budget'].update(llm_request_limit=3,used_llm_requests=0)
            ids=tuple(e['id'] for e in state['available_candidates'])
            raw=context_for(state,caps,ids,False,'calibrated-v2');checked=context_for(state,caps,ids,True,'calibrated-v2')
            if raw!={k:v for k,v in checked.items() if k not in ('checks','memory')}:raise ValueError('Common R/E context differs')
            path=folder/f'state_{prefix}.json';save(path,dict(state=state,capabilities=caps,eligible_ids=ids,R=raw,E=checked))
            next_primary=primary[prefix]['candidate'] if len(primary)>prefix else None
            states.append(dict(state_id=f'{name}_{prefix}',task=name,index=prefix-1,state_file=str(path),state_sha256=sha(path),
                original_candidate=next_primary,original_candidate_is_local_not_R=True))
        entry=original[name];archive=Path(entry['training_source'])
        if sha(archive)!=entry['training_source_sha256']:raise ValueError('Training source changed')
        task=load_training_task(archive);metric=MetricSpec.for_task(task);payloads={}
        used={frozenset(read(directory/(split+'_split.json'))['search_indices']) for split in ('primary','confirmation_1','confirmation_2')}
        for seed in SEEDS:
            train,held,resource=grouped_partition(task,seed,'shared_text')
            if frozenset(held.tolist()) in used:raise ValueError('New audit repeats an exposed split')
            used.add(frozenset(held.tolist()));path=folder/f'audit_{seed}.joblib'
            joblib.dump(dict(kind=task.kind,X_train=take(task.X_train,train),y_train=task.y_train[train],X_search=take(task.X_train,held),
                y_search=task.y_train[held],metric_spec=metric.record(),pair_semantics=declared_semantics(task)),path)
            save(folder/f'audit_{seed}.json',dict(train_indices=train.tolist(),search_indices=held.tolist(),overlap=overlap_audit(task,train,held)))
            payloads[str(seed)]=dict(path=str(path),sha256=sha(path))
        entries.append(dict(task=name,task_budget_id=entry.get('task_budget_id','dry_bean'),training_archive=str(archive),
            training_archive_sha256=sha(archive),audit_payloads=payloads))
    save(root/'states.json',states);save(root/'readiness.json',entries)
    return dict(status='prepared',states=12,tasks=4,actual_classifier_fits=0,real_api_requests=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','ask','evaluate'])
    for name in ('output','study','reference','prepared','answers','window-config','ledger','api-key-file'):p.add_argument('--'+name,type=Path)
    p.add_argument('--execute',action='store_true');a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    frozen=sources();start=time.monotonic();engine.SEEDS=SEEDS
    save(root/'protocol.json',dict(batch='A70-'+a.phase,source_files=frozen,runner_sha256=sha(Path(__file__)),
        reused_engine_sha256=sha(Path(engine.__file__)),plan_sha256=sha(ROOT/'docs/batches/A70_four_task_feedback_completion_plan.md'),
        seeds=SEEDS,maximum_requests=24 if a.phase=='ask' else 0,maximum_new_fits=48 if a.phase=='evaluate' else 0,
        course_budget_compliance='unverified_history',valid_for_budgeted_final_report=False))
    try:
        if a.phase=='prepare':result=prepare(a,root,frozen)
        elif a.phase=='ask':result=engine.ask(a,root,frozen)
        else:
            result=engine.evaluate(a,root,frozen)
            comparisons=read(root/'comparison.json')
            for r in comparisons:
                r['local_next_candidate']=r.pop('original_R_candidate');r['differs_from_local_next']=r.pop('fresh_R_changed')
            save(root/'comparison.json',comparisons)
        result.update(source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start)
        save(root/'completion.json',result);print(json.dumps(result))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),automatic_retry=False,
            observed_fit_entries=len(engine.physical(root)),elapsed_seconds=time.monotonic()-start));raise
