"""Real Flash pilot with explicit historical-usage policy; default is preflight only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha,launch
from superagent.ab_flash import preflight
from superagent.ab_run import sources,freeze_partitions
from superagent.ab_catalog import capabilities_for
from superagent.cli import build_parser
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);began=time.monotonic();frozen=sources()
    protocol=dict(batch='A16',source_files=frozen,runner_sha256=sha(Path(__file__)),
        history_policy=args.history_policy,course_budget_compliance='unverified_history' if args.history_policy=='incremental-research' else 'requires_verified_history',
        execute_requested=args.execute,maximum_classifier_fits=22,maximum_real_requests=6,
        per_run_allowance_rmb=.30,task_budget_id=args.task_budget_id,automatic_retries=False)
    save(root/'protocol.json',protocol)
    try:
        meta=next(t for t in read(args.prepared/'tasks.json') if t['name']=='sms')
        folder=args.prepared/'sms'
        for file,expected in meta['files'].items():
            if sha(folder/file)!=expected:raise ValueError('Prepared SMS bytes changed')
        commands=[]
        for mode in ('raw-history','checked-history'):
            cli=['run',str(folder/'train.zip'),'--scheduler','ab-v1','--portfolio','ab-v1','--planner',mode,
                '--api-key-file',str(args.api_key_file),'--budget-ledger',str(args.ledger),
                '--task-budget-id',args.task_budget_id,'--llm-budget-rmb','.30','--seed','707',
                '--history-policy',args.history_policy,
                '--run-dir',str(root/mode),'--fit-limit','11','--time-limit','600','--trial-timeout','60']
            parsed=build_parser().parse_args(cli)
            ledger,allowance,snapshot=preflight(parsed)
            if args.history_policy=='incremental-research':
                from superagent.research_budget import research_preflight
                research_preflight(ledger,args.task_budget_id,.60)
            else:
                ledger.preflight(args.task_budget_id,.60)
            commands.append([sys.executable,'-m','superagent',*cli])
        task=load_development_task(folder/'train.zip');metric=MetricSpec.for_task(task)
        partitions=root/'partitions';partitions.mkdir();_,allowed=freeze_partitions(task,partitions,707,metric)
        protocol.update(task=meta,budget_before=snapshot,capabilities=capabilities_for(TaskSpec.from_task(task).record(),
            train_rows=len(task.y_train)),splits=allowed,commands=commands)
        save(root/'protocol.json',protocol)
        if not args.execute:
            save(root/'completion.json',dict(status='validated_not_executed',real_api_requests=0,classifier_fits=0))
            return
        results=[]
        for mode,command in zip(('raw-history','checked-history'),commands):
            if sources()!=frozen:raise ValueError('Source changed after preflight')
            # launch is intentionally not used: it forces offline mode for local research batches.
            with (root/(mode+'.log')).open('w',encoding='utf-8') as file:
                completed=subprocess.run(command,cwd=Path(__file__).resolve().parents[1],
                    stdout=file,stderr=subprocess.STDOUT,timeout=660)
            value=read(root/mode/'run_outcome.json');results.append(value);save(root/'results.json',results)
            if completed.returncode or not value['requested_planner_completed']:
                raise RuntimeError('Live pilot stopped without retry; see run outcome and cumulative ledger')
            if value['new_api_requests']>3:raise ValueError('Per-arm request cap exceeded')
            manifest=read(root/mode/'run_manifest.json')
            if manifest['capabilities']!=protocol['capabilities']:raise ValueError('Candidate identity changed')
            for split in allowed:
                if read(root/mode/(split+'_split.json'))!=read(partitions/(split+'_split.json')):
                    raise ValueError('Actual split differs from preflight')
        save(root/'all_models_frozen.json',[read(root/mode/'final_selection.json') for mode in ('raw-history','checked-history')])
        # Scoring and delivery only after both arms have fully frozen models.
        for mode in ('raw-history','checked-history'):
            directory=root/mode;csv=directory/'predictions/rows.csv'
            launch([sys.executable,'-m','superagent','predict',str(folder/'features.zip'),
                '--run-dir',str(directory),'--output',str(csv)],root/(mode+'_predict.log'),60)
            launch([sys.executable,str(Path(__file__).parent/'a07_verify_export.py'),
                '--source-root',str(Path(__file__).resolve().parents[1]),'--run-dir',str(directory),
                '--holdout',str(folder/'holdout.joblib'),'--predictions',str(csv),
                '--output',str(directory/'verification.json')],root/(mode+'_verify.log'),60)
        save(root/'completion.json',dict(status='completed',real_api_requests=sum(r['new_api_requests'] for r in results),
            history_policy=args.history_policy,course_budget_compliance='unverified_history' if args.history_policy=='incremental-research' else 'verified_local_ledger',
            project_usage_after_run=ledger.project_summary(),
            classifier_fits=sum(r['classifier_fit_accounting']['attempted_classifier_fits'] for r in results),
            source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-began))
    except Exception as error:
        observed=[read(p) for p in root.glob('*/run_outcome.json')]
        save(root/'completion.json',dict(status='stopped' if args.execute else 'blocked_by_accounting_or_input',
            error=str(error),real_api_requests=sum(r['new_api_requests'] for r in observed) if observed else None if args.execute else 0,
            classifier_fits=sum(r['classifier_fit_accounting']['attempted_classifier_fits'] for r in observed) if observed else None if args.execute else 0,
            automatic_retry=False,history_policy=args.history_policy))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--api-key-file',type=Path,required=True);p.add_argument('--task-budget-id',default='sms_spam')
    p.add_argument('--history-policy',choices=['strict','incremental-research'],default='strict')
    p.add_argument('--execute',action='store_true');run(p.parse_args())
