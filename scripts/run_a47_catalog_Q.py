"""Same-source five-task catalogue study; deduplicate identical policies, freeze before scoring."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch,physical
from scripts.run_a34_transfer_coverage import evaluate_inputs
from superagent.ab_run import sources,registered_capabilities,freeze_partitions
from superagent.contracts import TaskSpec
from superagent.data import load_training_task
from superagent.metrics import MetricSpec


ROOT=Path(__file__).resolve().parents[1]


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();results=[]
    prepared={e['task']:e for e in read(args.prepared/'readiness.json')}
    original={e['task']:e for e in read(args.readiness/'readiness.json')}
    names=('coupon','preference','sms','photo','bean');entries={};order=[]
    observer=ROOT/'scripts/a29_worker_observer/sitecustomize.py'
    immutable={str(p):sha(p) for p in (Path(__file__),observer,ROOT/'scripts/run_a29_choice_matrix.py',ROOT/'scripts/run_a34_transfer_coverage.py')}
    deadline_text=json.loads(args.window_config.read_text(encoding='utf-8-sig'))['deadline'] if args.window_config else args.window_deadline
    save(root/'protocol.json',dict(batch='A47-five-task-catalog-Q',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A47_same_source_catalog_Q_plan.md'),script_hashes=immutable,
        maximum_classifier_fits=77,real_api_requests=0,execute_requested=args.execute,window_deadline=deadline_text))
    def offline(event,values):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('Network forbidden in A47')
    sys.addaudithook(offline)
    try:
        deadline=datetime.fromisoformat(deadline_text)
        if deadline.tzinfo is None:raise ValueError('Window deadline needs an explicit timezone')
        for name in names:
            entry=prepared.get(name)
            archive=Path(entry['training_archive']) if entry else Path(original[name]['source_archive'])
            digest=entry['training_archive_sha256'] if entry else original[name]['task_archive_sha256']
            if sha(archive)!=digest:raise ValueError('Training package changed')
            task=load_training_task(archive);spec=TaskSpec.from_task(task).record();caps={s:registered_capabilities(spec,len(task.y_train),s) for s in ('baseline','full')}
            same=caps['baseline']==caps['full']
            if same!=(name in ('preference','sms','bean')):raise ValueError('Registered deduplication pattern changed')
            folder=root/'inputs'/name;folder.mkdir(parents=True)
            _,splits=freeze_partitions(task,folder,707,MetricSpec.for_task(task))
            if entry:
                base=dict(entry)
            else:
                evaluation=args.existing_evaluation/name
                base=dict(task=name,evaluation_source=str(evaluation),evaluation_files_sha256={n:sha(evaluation/n) for n in ('features.zip','holdout.joblib')})
            entries[name]=dict(**base,task_spec_current=spec,current_capabilities=caps,training_source=str(archive),
                training_source_sha256=digest,registered_splits=splits,deduplicated_same_policy=same)
            scopes=['shared'] if same else ['full','baseline'] if name=='preference' else ['baseline','full']
            order.extend((name,scope) for scope in scopes)
        protocol=read(root/'protocol.json');protocol.update(entries=entries,order=order);save(root/'protocol.json',protocol)
        if not args.execute:
            save(root/'completion.json',dict(status='validated_not_executed',physical_paths=len(order),actual_classifier_fits=0,real_api_requests=0));return
        if (deadline-datetime.now(timezone.utc)).total_seconds()<1800:raise RuntimeError('Less than 30 minutes left; no training started')
        env={**os.environ,'AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),
            'PYTHONPATH':os.pathsep.join([str(observer.parent),str(ROOT)]),'SUPERAGENT_OFFLINE':'1',
            'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        for name,scope in order:
            if sources()!=frozen or any(sha(Path(p))!=h for p,h in immutable.items()):raise ValueError('Frozen source changed')
            if (deadline-datetime.now(timezone.utc)).total_seconds()<660:raise RuntimeError('No full path time remains')
            entry=entries[name];directory=root/(name+'_'+scope);actual_scope='full' if scope=='shared' else scope
            cmd=[sys.executable,'-m','superagent','run',entry['training_source'],'--run-dir',str(directory),
                '--scheduler','ab-v1','--portfolio','ab-v1','--catalog-scope',actual_scope,'--fit-limit','11',
                '--seed','707','--time-limit','600','--trial-timeout','60']
            code,seconds=launch(cmd,root/(directory.name+'.log'),{**env,'AB_A29_RUN_NAME':directory.name},660)
            value=read(directory/'run_outcome.json');results.append(dict(task=name,scope=scope,outcome=value,process_seconds=seconds));save(root/'checkpoint.json',results)
            if code or not value['requested_planner_completed']:raise RuntimeError('Path stopped without retry: '+directory.name)
            observed=sum(r['run_name']==directory.name for r in physical(root));counts=value['classifier_fit_accounting']
            if not(2<=observed==counts['attempted_classifier_fits']==counts['completed_classifier_fits']<=11):raise ValueError('Fit accounting differs')
            manifest=read(directory/'run_manifest.json')
            if manifest['source_files']!=frozen or manifest['capabilities']!=entry['current_capabilities'][actual_scope]:raise ValueError('Source/menu mismatch')
            for split in entry['registered_splits']:
                if read(directory/(split+'_split.json'))!=read(root/'inputs'/name/(split+'_split.json')):raise ValueError('Split mismatch')
            if any('training_profile' in read(p) for p in directory.glob('state_*.json')):raise ValueError('Default-off profile leaked into live state')
            selection=read(directory/'final_selection.json')
            if sha(directory/selection['model_file'])!=selection['model_sha256']:raise ValueError('Frozen model changed')
            results[-1].update(selection=selection,actual_classifier_fits=observed);save(root/'checkpoint.json',results)
            print(json.dumps(dict(task=name,scope=scope,status='model_frozen',fits=observed,seconds=seconds)),flush=True)
        save(root/'all_models_frozen.json',[dict(task=r['task'],scope=r['scope'],selection=r['selection']) for r in results])
        held={name:evaluate_inputs(entries[name],root,len(order)) for name in names}
        for row in results:
            directory=root/(row['task']+'_'+row['scope']);folder=held[row['task']];destination=directory/'predictions/rows.csv'
            commands=[
                [sys.executable,'-m','scripts.a38_predict_guard',str(folder/'features.zip'),'--run-dir',str(directory),'--output',str(destination)],
                [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(directory),
                    '--holdout',str(folder/'holdout.joblib'),'--predictions',str(destination),'--output',str(directory/'verification.json')]]
            for index,cmd in enumerate(commands):
                code,_=launch(cmd,root/(directory.name+f'_export_{index}.log'),{**env,'AB_A29_RUN_NAME':'inference_only'},60)
                if code:raise RuntimeError('Frozen export failed: '+directory.name)
            row['evaluation']=read(directory/'verification.json');save(root/'results.json',results)
        if len(physical(root))!=sum(r['actual_classifier_fits'] for r in results):raise ValueError('Unexpected fits during inference')
        save(root/'completion.json',dict(status='completed',tasks=5,physical_paths=len(order),logical_groups=10,
            deduplicated_tasks=['preference','sms','bean'],actual_classifier_fits=len(physical(root)),real_api_requests=0,
            source_unchanged=sources()==frozen,formal_test_opened=False,elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),observed_classifier_fit_entries=len(physical(root)),
            path_outcomes=len(results),real_api_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','readiness','existing-evaluation','output'):p.add_argument('--'+name,type=Path,required=True)
    window=p.add_mutually_exclusive_group(required=True)
    window.add_argument('--window-deadline');window.add_argument('--window-config',type=Path)
    p.add_argument('--execute',action='store_true')
    run(p.parse_args())
