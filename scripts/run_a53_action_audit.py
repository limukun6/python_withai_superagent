"""Reuse four real first proposals on two fresh training-only splits; no API, deduplicated fits."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

import joblib

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import physical
from superagent.ab_run import sources
from superagent.benchmark import train_candidate
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.splits import grouped_partition,take,overlap_audit
from superagent.task_semantics import declared_semantics


SEEDS=(19521,19531)
ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','window-config','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();results=[]
    window=json.loads(args.window_config.read_text(encoding='utf-8-sig'));deadline=datetime.fromisoformat(window['deadline'])
    def remaining():return (deadline-datetime.now(timezone.utc)).total_seconds()
    save(root/'protocol.json',dict(batch='A53-action-audit',source_files=frozen,seeds=SEEDS,runner_sha256=sha(Path(__file__)),
        maximum_classifier_fits=16,real_api_requests=0,raw_answers_reused=True,external_validation_opened=False))
    try:
        if remaining()<1200:raise ValueError('Less than 20 minutes remain; no audit training started')
        if read(args.study/'completion.json')['status']!='completed' or read(args.study/'protocol.json')['source_files']!=frozen:
            raise ValueError('Completed same-source live study required')
        entries={r['task']:r for r in read(args.study/'protocol.json')['entries']};actions=[];work=[]
        for name in ('coupon','preference'):
            common=[]
            for arm in ('00','10','01','11'):
                directory=args.study/(name+'_'+arm);history=read(directory/'history.json')
                first=next(r for r in history if r.get('proposal'))
                events=[json.loads(line) for line in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
                request=next(r for r in events if r['kind']=='llm_request');payload=json.loads(request['messages'][1]['content'])
                common.append({k:v for k,v in payload.items() if k not in ('training_profile','context_version','instruction')})
                actions.append(dict(task=name,arm=arm,candidate=first['candidate'],answer_reused=True,request_id=request['budget_request_id']))
            if any(c!=common[0] for c in common):raise ValueError('First common history differs beyond treatments')
            entry=entries[name];archive=Path(entry['training_archive'])
            if sha(archive)!=entry['training_archive_sha256']:raise ValueError('Training bytes changed')
            task=load_training_task(archive);folder=root/name;folder.mkdir();metric=MetricSpec.for_task(task)
            seen={frozenset(read(args.study/'inputs'/name/(s+'_split.json'))['search_indices']) for s in entry['splits']}
            for seed in SEEDS:
                train,held,resource=grouped_partition(task,seed,'shared_text')
                if frozenset(held.tolist()) in seen:raise ValueError('Audit split repeats a registered split')
                seen.add(frozenset(held.tolist()));payload=folder/f'audit_{seed}.joblib'
                joblib.dump(dict(kind=task.kind,X_train=take(task.X_train,train),y_train=task.y_train[train],
                    X_search=take(task.X_train,held),y_search=task.y_train[held],metric_spec=metric.record(),pair_semantics=declared_semantics(task)),payload)
                save(folder/f'audit_{seed}.json',dict(seed=seed,train_indices=train.tolist(),search_indices=held.tolist(),resource=resource,
                    payload_sha256=sha(payload),overlap=overlap_audit(task,train,held)))
                candidates=list(dict.fromkeys(a['candidate'] for a in actions if a['task']==name))
                work.extend(dict(task=name,candidate=c,seed=seed,payload=str(payload),payload_sha256=sha(payload)) for c in candidates)
        save(root/'actions.json',actions);save(root/'work_order.json',work)
        observer=ROOT/'scripts/a29_worker_observer/sitecustomize.py'
        os.environ.update(AB_A29_OBSERVE_FITS='1',AB_A29_FIT_LOG=str(root/'physical_fits.jsonl'),
            PYTHONPATH=os.pathsep.join([str(observer.parent),str(ROOT)]),SUPERAGENT_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for index,item in enumerate(work):
            if remaining()<90 or sources()!=frozen:raise ValueError('Time/source changed; no further worker')
            prefix=root/f'action_{index}';admission=prefix.with_suffix('.admission.json');ledger=prefix.with_suffix('.fits.json')
            if sha(Path(item['payload']))!=item['payload_sha256']:raise ValueError('Payload changed')
            save(admission,dict(maximum_classifier_fits=1,ledger_path=str(ledger)));os.environ['AB_A29_RUN_NAME']=prefix.name
            model,facts=train_candidate(Path(item['payload']),item['candidate'],prefix.with_suffix('.joblib'),60,fit_admission_path=admission)
            counts=read(ledger) if ledger.exists() else None
            results.append(dict(**item,facts=facts,fit_accounting=counts));save(root/'results.json',results)
            if model is None or counts is None or counts['status']!='completed' or counts['completed_classifier_fits']!=1:raise ValueError('Audit worker failed without retry')
        comparisons=[]
        for action in actions:
            scores=[next(r['facts']['search_score'] for r in results if r['task']==action['task'] and r['candidate']==action['candidate'] and r['seed']==seed) for seed in SEEDS]
            comparisons.append(dict(**action,scores=scores,mean=sum(scores)/2))
        save(root/'comparison.json',comparisons)
        if len(physical(root))!=len(results):raise ValueError('Physical count mismatch')
        save(root/'completion.json',dict(status='completed',actual_classifier_fits=len(physical(root)),logical_evaluations=16,
            reused_evaluations=16-len(results),real_api_requests=0,source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,
            external_validation_opened=False))
        print(json.dumps(read(root/'completion.json')))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),observed_fit_entries=len(physical(root)),real_api_requests=0,automatic_retry=False));raise
