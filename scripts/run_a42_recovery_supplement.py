"""Continue never-started matrix paths, retain failed paths, then score a labelled cohort."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from scripts.run_a29_choice_matrix import command, launch, physical, evaluation, settings
from scripts.status_a29_choice_matrix import completion_failure
from superagent.ab_run import sources
from superagent.ab_flash import preflight
from superagent.cli import build_parser
from superagent.research_budget import research_preflight


ROOT=Path(__file__).resolve().parents[1]


def inventory(parent):
    protocol=read(parent/'protocol.json');cohort=[];missing=[]
    if read(parent/'completion.json')['status']!='stopped':raise ValueError('Requires preserved stopped matrix')
    if (parent/'evaluation').exists():raise ValueError('Recovery amendment must be registered before external labels open')
    for task,arm in protocol['order']:
        directory=parent/(task+'_'+arm)
        if not directory.exists():missing.append((task,arm));continue
        if not (directory/'run_outcome.json').exists():raise ValueError('Unresolved started path cannot be restarted: '+directory.name)
        value=read(directory/'run_outcome.json')
        if not value['model_available']:raise ValueError('Failed path lacks a frozen partial model')
        selected=read(directory/'final_selection.json')
        if sha(directory/selected['model_file'])!=selected['model_sha256']:raise ValueError('Preserved model changed')
        cohort.append(dict(task=task,arm=arm,run_dir=str(directory),origin='original_A32',
            full_refit_comparison_eligible=bool(value['requested_planner_completed'] and value['full_refit_completed']),
            outcome_sha256=sha(directory/'run_outcome.json'),selection=selected))
    return protocol,cohort,missing


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();new=[]
    frozen=sources();parent,cohort,missing=inventory(args.original)
    if frozen!=parent['source_files'] or frozen!=read(args.prepared/'protocol.json')['source_files']:
        raise ValueError('Cannot resume on a different production version')
    if len(cohort)!=9 or len(missing)!=11 or sum(not row['full_refit_comparison_eligible'] for row in cohort)!=1:
        raise ValueError('This registered supplement requires the audited 8 complete / 1 partial / 11 unstarted state')
    deadline=datetime.fromisoformat(args.window_deadline)
    entries={row['task']:row for row in read(args.prepared/'readiness.json')}
    observer=ROOT/'scripts/a29_worker_observer/sitecustomize.py'
    immutable={str(file):sha(file) for file in (Path(__file__),ROOT/'scripts/run_a29_choice_matrix.py',observer)}
    protocol=dict(batch='A42-unstarted-path-supplement',source_files=frozen,original_protocol_sha256=sha(args.original/'protocol.json'),
        original_completion_sha256=sha(args.original/'completion.json'),runner_sha256=sha(Path(__file__)),script_hashes=immutable,
        plan_sha256=sha(ROOT/'docs/batches/A42_recovery_supplement_plan.md'),original_matrix_completed=False,
        original_matrix_modified=False,window_deadline=args.window_deadline,order=missing,
        maximum_new_classifier_fits=11*len(missing),maximum_new_requests=3*sum(settings(a)[0]!='local' for _,a in missing),
        historical_unknown_retained=True,automatic_retry=False,execute_requested=args.execute)
    save(root/'protocol.json',protocol)
    try:
        cmds=[];ledger=None
        for task,arm in missing:
            item=entries[task]
            if sha(Path(item['training_archive']))!=item['training_archive_sha256']:raise ValueError('Training archive changed')
            cmd=command(item,arm,root/(task+'_'+arm),args)
            if settings(arm)[0]!='local':
                ledger,_,_=preflight(build_parser().parse_args(cmd[3:]))
                allowance=.30*sum(t==task and settings(a)[0]!='local' for t,a in missing)
                research_preflight(ledger,item['task_budget_id'],allowance)
            cmds.append(dict(task=task,arm=arm,command=cmd))
        protocol.update(entries=list(entries.values()),commands=cmds,project_usage_before=ledger.project_summary())
        save(root/'protocol.json',protocol)
        if not args.execute:
            save(root/'completion.json',dict(status='validated_not_executed',new_paths=len(cmds),actual_classifier_fits=0,real_api_requests=0));return
        env={**os.environ,'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
            'TOKENIZERS_PARALLELISM':'false','AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),
            'PYTHONPATH':os.pathsep.join([str(observer.parent),str(ROOT)])}
        for item in cmds:
            if (deadline-datetime.now(timezone.utc)).total_seconds()<660:raise RuntimeError('No 660-second window for another path')
            if sources()!=frozen or any(sha(Path(p))!=h for p,h in immutable.items()):raise ValueError('Frozen source/runner changed')
            task,arm=item['task'],item['arm'];directory=root/(task+'_'+arm)
            code,seconds=launch(item['command'],root/(directory.name+'.log'),
                {**env,'AB_A29_RUN_NAME':directory.name,'SUPERAGENT_OFFLINE':'1' if settings(arm)[0]=='local' else '0'},660)
            outcome=read(directory/'run_outcome.json')
            new.append(dict(task=task,arm=arm,outcome=outcome,process_seconds=seconds));save(root/'checkpoint.json',new)
            if code or not outcome['requested_planner_completed']:
                raise RuntimeError('Supplement stopped without retry: '+directory.name+': '+str(completion_failure(outcome)))
            count=sum(row['run_name']==directory.name for row in physical(root));account=outcome['classifier_fit_accounting']
            if not(2<=count==account['attempted_classifier_fits']==account['completed_classifier_fits']<=11) or outcome['new_api_requests']>3:
                raise ValueError('Physical fit/request limits differ')
            manifest=read(directory/'run_manifest.json');entry=entries[task]
            if manifest['source_files']!=frozen or manifest['capabilities']!=entry['capabilities']:raise ValueError('Source/menu changed')
            for split in entry['splits']:
                if read(directory/(split+'_split.json'))!=read(args.prepared/task/(split+'_split.json')):raise ValueError('Split changed')
            selection=read(directory/'final_selection.json')
            if sha(directory/selection['model_file'])!=selection['model_sha256']:raise ValueError('Frozen model changed')
            cohort.append(dict(task=task,arm=arm,run_dir=str(directory),origin='new_A42',full_refit_comparison_eligible=True,
                outcome_sha256=sha(directory/'run_outcome.json'),selection=selection))
            save(root/'cohort.json',cohort)
            print(json.dumps(dict(task=task,arm=arm,status='model_frozen',fits=count,requests=outcome['new_api_requests'],seconds=seconds)),flush=True)
        order={tuple(pair):index for index,pair in enumerate(parent['order'])};cohort.sort(key=lambda row:order[row['task'],row['arm']])
        for row in cohort:
            d=Path(row['run_dir'])
            if sha(d/'run_outcome.json')!=row['outcome_sha256'] or sha(d/row['selection']['model_file'])!=row['selection']['model_sha256']:
                raise ValueError('Retained model/outcome changed before scoring')
        save(root/'cohort.json',cohort);save(root/'all_models_frozen.json',cohort)
        held={name:evaluation(entry,root,20) for name,entry in entries.items()}
        results=[]
        for row in cohort:
            d=Path(row['run_dir']);export=root/'exports'/(row['task']+'_'+row['arm']);export.mkdir(parents=True)
            prediction=export/'rows.csv';folder=held[row['task']]
            commands=[
                [sys.executable,'-m','scripts.a38_predict_guard',str(folder/'features.zip'),'--run-dir',str(d),'--output',str(prediction)],
                [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(d),
                 '--holdout',str(folder/'holdout.joblib'),'--predictions',str(prediction),'--output',str(export/'verification.json')]]
            for index,cmd in enumerate(commands):
                code,_=launch(cmd,export/f'export_{index}.log',{**env,'SUPERAGENT_OFFLINE':'1','AB_A29_RUN_NAME':'export_forbidden'},60)
                if code:raise RuntimeError('Frozen-model export failed: '+str(export))
            outcome=read(d/'run_outcome.json');history=read(d/'history.json');caps=read(d/'run_manifest.json')['capabilities']
            results.append(dict(**row,evaluation=read(export/'verification.json'),export_dir=str(export),
                classifier_fits=outcome['classifier_fit_accounting']['attempted_classifier_fits'],
                new_api_requests=outcome['new_api_requests'],known_course_cost_rmb=outcome['known_course_cost_rmb'],
                explored_routes=sorted({caps[r['candidate']]['route'] for r in history if r['status']=='completed' and r['split_id']=='primary'})))
            save(root/'results.json',results)
        fits=len(physical(root))
        if fits!=sum(row['outcome']['classifier_fit_accounting']['attempted_classifier_fits'] for row in new):raise ValueError('Unexpected fit during export')
        save(root/'completion.json',dict(status='completed',meaning='11 never-started paths and explicitly partial cohort measurement complete',
            original_matrix_completed=False,new_paths_completed=len(new),cohort_models=len(results),
            full_refit_comparison_eligible=sum(row['full_refit_comparison_eligible'] for row in results),partial_paths_retained=1,
            actual_new_classifier_fits=fits,real_new_api_requests=sum(row['outcome']['new_api_requests'] for row in new),
            known_new_course_cost_rmb=sum(row['outcome']['known_course_cost_rmb'] for row in new),project_usage_after=ledger.project_summary(),
            source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,automatic_retry=False,
            formal_test_opened=False,course_budget_compliance='unverified_history',valid_for_budgeted_final_report=False))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),automatic_retry=False,new_path_outcomes=len(new),
            observed_new_fit_entries=len(physical(root)),new_requests=sum(row['outcome']['new_api_requests'] for row in new),
            elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','prepared','ledger','api-key-file','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--window-deadline',required=True);parser.add_argument('--execute',action='store_true')
    run(parser.parse_args())
