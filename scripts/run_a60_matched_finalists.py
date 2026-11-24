"""A60 bounded local two-task decision/selection comparison, no real API requests."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch,physical
from scripts.run_a34_transfer_coverage import evaluate_inputs
from superagent.ab_run import sources,registered_capabilities
from superagent.contracts import TaskSpec
from superagent.data import load_training_task


ROOT=Path(__file__).resolve().parents[1]


def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();frozen=sources();results=[]
    order=[('coupon','ab-v1'),('coupon','ab-finalists'),('preference','ab-finalists'),('preference','ab-v1')]
    reference=read(args.reference/'protocol.json')
    plan=ROOT/'docs/batches/A60_matched_finalists_plan.md'
    protocol=dict(batch='A60',source_files=frozen,runner_sha256=sha(Path(__file__)),plan_sha256=sha(plan),
        reference_protocol_sha256=sha(args.reference/'protocol.json'),order=order,fit_limit_per_path=7,
        maximum_new_classifier_fits=28,starting_tracked_A_fits=970,planned_tracked_A_ceiling=998,
        real_api_requests=0,comparison_scope='Joint exploration allocation and final selection policy, two seen tasks, one split',
        old_ab_fixed_not_run=True,formal_test_or_H_opened=False)
    save(root/'protocol.json',protocol)
    def offline(event,values):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            raise PermissionError('Network forbidden in A60')
    sys.addaudithook(offline)
    try:
        if read(args.reference/'completion.json')['status']!='completed':raise ValueError('Completed A47 inputs required')
        entries={name:reference['entries'][name] for name in ('coupon','preference')}
        for name,entry in entries.items():
            archive=Path(entry['training_source'])
            if sha(archive)!=entry['training_source_sha256']:raise ValueError('Training package changed')
            task=load_training_task(archive)
            caps=registered_capabilities(TaskSpec.from_task(task).record(),len(task.y_train),'full')
            if caps!=entry['current_capabilities']['full']:raise ValueError('Existing catalogue changed')
        if not args.execute:
            save(root/'completion.json',dict(status='validated_not_executed',actual_classifier_fits=0,real_api_requests=0));return
        observer=ROOT/'scripts/a29_worker_observer/sitecustomize.py'
        env={**os.environ,'AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),
            'PYTHONPATH':os.pathsep.join([str(observer.parent),str(ROOT)]),'SUPERAGENT_OFFLINE':'1',
            'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        for name,scheduler in order:
            if sources()!=frozen or sha(Path(__file__))!=protocol['runner_sha256']:raise ValueError('Frozen code changed')
            directory=root/(name+'_'+scheduler);entry=entries[name]
            cmd=[sys.executable,'-m','superagent','run',entry['training_source'],'--run-dir',str(directory),
                '--scheduler',scheduler,'--portfolio','ab-v1','--fit-limit','7','--seed','707',
                '--time-limit','600','--trial-timeout','60','--partial-delivery','checkpoint']
            code,seconds=launch(cmd,root/(directory.name+'.log'),{**env,'AB_A29_RUN_NAME':directory.name},660)
            outcome=read(directory/'run_outcome.json')
            observed=sum(r['run_name']==directory.name for r in physical(root))
            row=dict(task=name,scheduler=scheduler,actual_classifier_fits=observed,seconds=seconds,outcome=outcome)
            results.append(row);save(root/'checkpoint.json',results)
            if code or not outcome['requested_planner_completed']:raise RuntimeError('No retry: '+directory.name)
            counts=outcome['classifier_fit_accounting']
            if not 2<=observed==counts['attempted_classifier_fits']==counts['completed_classifier_fits']<=7:
                raise ValueError('Physical classifier fits differ')
            if outcome['new_api_requests']!=0:raise ValueError('Unexpected API use')
            manifest=read(directory/'run_manifest.json')
            if manifest['source_files']!=frozen or manifest['capabilities']!=entry['current_capabilities']['full']:
                raise ValueError('Source/menu mismatch')
            for split in ('primary','confirmation_1','confirmation_2'):
                if read(directory/(split+'_split.json'))!=read(args.reference/'inputs'/name/(split+'_split.json')):
                    raise ValueError('Existing development partitions changed')
            selection=read(directory/'final_selection.json')
            if sha(directory/selection['model_file'])!=selection['model_sha256']:raise ValueError('Model hash changed')
            history=read(directory/'history.json');review=read(directory/'confirmation.json')
            row.update(selection=selection,confirmation=review,
                primary_candidates=[r['candidate'] for r in history if r['split_id']=='primary'],
                confirmation_fits=sum(r['split_id']!='primary' for r in history))
            save(root/'checkpoint.json',results)
            print(json.dumps(dict(task=name,scheduler=scheduler,status='frozen',fits=observed,seconds=seconds)),flush=True)
        save(root/'all_models_frozen.json',[dict(task=r['task'],scheduler=r['scheduler'],selection=r['selection']) for r in results])
        held={name:evaluate_inputs(entries[name],root,len(order)) for name in entries}
        for row in results:
            directory=root/(row['task']+'_'+row['scheduler']);folder=held[row['task']];pred=directory/'predictions/rows.csv'
            commands=[
                [sys.executable,'-m','scripts.a38_predict_guard',str(folder/'features.zip'),'--run-dir',str(directory),'--output',str(pred)],
                [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(directory),
                 '--holdout',str(folder/'holdout.joblib'),'--predictions',str(pred),'--output',str(directory/'verification.json')]]
            for index,cmd in enumerate(commands):
                code,_=launch(cmd,root/(directory.name+f'_export_{index}.log'),{**env,'AB_A29_RUN_NAME':'inference_only'},60)
                if code:raise RuntimeError('Frozen prediction export failed')
            row['evaluation']=read(directory/'verification.json');save(root/'results.json',results)
        if len(physical(root))!=sum(r['actual_classifier_fits'] for r in results):raise ValueError('Unexpected inference fit')
        save(root/'completion.json',dict(status='completed',physical_paths=len(results),actual_classifier_fits=len(physical(root)),
            real_api_requests=0,tracked_A_fits_after=970+len(physical(root)),source_unchanged=sources()==frozen,
            elapsed_seconds=time.monotonic()-start,formal_test_or_H_opened=False))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=len(physical(root)),
            real_api_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');main(p.parse_args())
