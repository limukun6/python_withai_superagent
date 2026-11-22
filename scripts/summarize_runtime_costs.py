"""Measured sequential timing decomposition, without treating every non-fit second as API overhead."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if read(args.study/'completion.json')['status']!='completed':raise ValueError('Study must finish before cost summary')
    records=read(args.study/'results.json');results=[]
    for row in records:
        arm=row.get('arm',row.get('scope'));directory=args.study/(row['task']+'_'+arm)
        history=read(directory/'history.json');caps=read(directory/'run_manifest.json')['capabilities'];outcome=row['outcome']
        usage=[json.loads(line) for line in (directory/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines()]
        routes=defaultdict(lambda:dict(completed_evaluations=0,worker_wall_seconds=0.,fit_training_seconds=0.))
        for trial in history:
            item=routes[caps[trial['candidate']]['route']];item['completed_evaluations']+=1
            item['worker_wall_seconds']+=trial.get('worker_wall_seconds',0);item['fit_training_seconds']+=trial.get('training_seconds',0)
        worker=sum(v['worker_wall_seconds'] for v in routes.values());api=sum(r.get('latency_seconds',0) for r in usage if r['event']=='llm_call')
        refit=outcome['finalization']['refit_training_seconds'];elapsed=outcome['elapsed_seconds']
        remainder=elapsed-worker-api-refit
        results.append(dict(task=row['task'],arm=arm,path_inner_seconds=elapsed,subprocess_seconds=row['process_seconds'],
            evaluation_worker_wall_seconds=worker,LLM_call_latency_seconds=api,final_refit_fit_training_seconds=refit,
            other_inner_seconds=remainder,other_includes='Refit worker startup, profiling, hashing, serialization, selection and logging; not pure API overhead',
            classifier_fits=outcome['classifier_fit_accounting']['attempted_classifier_fits'],routes=dict(routes),
            source='Recorded sequential wall timers; not CPU profiling, statistical attribution or counterfactual runtime'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError('Do not overwrite a timing audit')
    save(args.output,results)
    print(json.dumps([dict(task=r['task'],arm=r['arm'],path_seconds=r['path_inner_seconds'],api_seconds=r['LLM_call_latency_seconds'],worker_seconds=r['evaluation_worker_wall_seconds']) for r in results]))
