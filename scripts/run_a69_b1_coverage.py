"""A-owned B1 coverage using unchanged B methods and official fixed/dynamic scheduling."""
import argparse
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


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();start=time.monotonic();results=[]
    entries=read(a.reference/'protocol.json')['entries'];names=('coupon','preference','sms','bean')
    order=[(name,seed,arm) for i,name in enumerate(names) for j,seed in enumerate((707,1707))
           for arm in (('ab-fixed','ab-v1') if (i+j)%2==0 else ('ab-v1','ab-fixed'))]
    save(root/'protocol.json',dict(batch='A69-B1-coverage',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A69_B1_coverage_completion_plan.md'),order=order,maximum_classifier_fits=176,
        real_api_requests=0,starting_tracked_A_fits=1025,independent_outer_splits=False,
        comparison_scope='Four seen tasks, two internal seed sets, original registered outer development holdout'))
    def offline(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('A69 network forbidden')
    sys.addaudithook(offline)
    try:
        for name in names:
            entry=entries[name];path=Path(entry['training_source'])
            if sha(path)!=entry['training_source_sha256']:raise ValueError('Training archive changed')
            task=load_training_task(path);caps=registered_capabilities(TaskSpec.from_task(task).record(),len(task.y_train),'full')
            if caps!=entry['current_capabilities']['full']:raise ValueError('Registered methods changed')
        if not a.execute:
            save(root/'completion.json',dict(status='validated_not_executed',actual_classifier_fits=0,real_api_requests=0));return
        observer=ROOT/'scripts/a29_worker_observer';observer_hash=sha(observer/'sitecustomize.py')
        env={**os.environ,'AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),
            'PYTHONPATH':os.pathsep.join([str(observer),str(ROOT)]),'SUPERAGENT_OFFLINE':'1','PYTHONIOENCODING':'utf-8',
            'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        for name,seed,arm in order:
            if sources()!=frozen or sha(observer/'sitecustomize.py')!=observer_hash:raise ValueError('Frozen source/observer changed')
            label=f'{name}_{seed}_{arm}';directory=root/label;entry=entries[name]
            code,seconds=launch([sys.executable,'-m','superagent','run',entry['training_source'],'--run-dir',str(directory),
                '--scheduler',arm,'--portfolio','ab-v1','--fit-limit','11','--seed',str(seed),'--time-limit','600',
                '--trial-timeout','60','--partial-delivery','checkpoint'],root/(label+'.log'),{**env,'AB_A29_RUN_NAME':label},660)
            outcome=read(directory/'run_outcome.json');observed=sum(r['run_name']==label for r in physical(root))
            row=dict(task=name,seed=seed,scheduler=arm,label=label,seconds=seconds,actual_classifier_fits=observed,outcome=outcome)
            results.append(row);save(root/'checkpoint.json',results)
            if code or not outcome['requested_planner_completed']:raise RuntimeError('Path stopped without retry: '+label)
            counts=outcome['classifier_fit_accounting']
            if not 2<=observed==counts['attempted_classifier_fits']==counts['completed_classifier_fits']<=11 or outcome['new_api_requests']:
                raise ValueError('Fit/API accounting mismatch')
            manifest=read(directory/'run_manifest.json')
            if manifest['source_files']!=frozen or manifest['capabilities']!=entry['current_capabilities']['full']:raise ValueError('Source/menu mismatch')
            other=root/f"{name}_{seed}_{'ab-v1' if arm=='ab-fixed' else 'ab-fixed'}"
            if other.exists():
                for split in ('primary','confirmation_1','confirmation_2'):
                    if read(directory/(split+'_split.json'))!=read(other/(split+'_split.json')):raise ValueError('Paired partitions mismatch')
            selected=read(directory/'final_selection.json')
            if sha(directory/selected['model_file'])!=selected['model_sha256']:raise ValueError('Frozen model changed')
            history=read(directory/'history.json');review=read(directory/'confirmation.json')['b_scheduler_review']
            row.update(selection=selected,distinct_primary_candidates=len({r['candidate'] for r in history if r['split_id']=='primary'}),
                confirmation_fits=sum(r['split_id']!='primary' for r in history),promotions=len(review['promotions']))
            save(root/'checkpoint.json',results)
            print(__import__('json').dumps(dict(path=label,status='model_frozen',fits=observed,seconds=seconds)),flush=True)
        save(root/'all_models_frozen.json',[dict(label=r['label'],selection=r['selection']) for r in results])
        held={name:evaluate_inputs(entries[name],root,len(order)) for name in names}
        for row in results:
            directory=root/row['label'];folder=held[row['task']];pred=directory/'predictions/rows.csv'
            commands=[
                [sys.executable,'-m','scripts.a38_predict_guard',str(folder/'features.zip'),'--run-dir',str(directory),'--output',str(pred)],
                [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(directory),
                 '--holdout',str(folder/'holdout.joblib'),'--predictions',str(pred),'--output',str(directory/'verification.json')]]
            for index,cmd in enumerate(commands):
                code,_=launch(cmd,root/(row['label']+f'_export_{index}.log'),{**env,'AB_A29_RUN_NAME':'inference_only'},60)
                if code:raise ValueError('Frozen delivery failed: '+row['label'])
            row['evaluation']=read(directory/'verification.json');save(root/'results.json',results)
        if len(physical(root))!=sum(r['actual_classifier_fits'] for r in results):raise ValueError('Unexpected inference training')
        comparisons=[]
        for name in names:
            for seed in (707,1707):
                f=next(r for r in results if (r['task'],r['seed'],r['scheduler'])==(name,seed,'ab-fixed'))
                d=next(r for r in results if (r['task'],r['seed'],r['scheduler'])==(name,seed,'ab-v1'))
                comparisons.append(dict(task=name,seed=seed,metric=f['evaluation']['primary_metric'],fixed_score=f['evaluation']['score'],
                    dynamic_score=d['evaluation']['score'],dynamic_minus_fixed=d['evaluation']['score']-f['evaluation']['score'],
                    fixed_fits=f['actual_classifier_fits'],dynamic_fits=d['actual_classifier_fits'],fixed_seconds=f['seconds'],dynamic_seconds=d['seconds']))
        save(root/'comparison.json',comparisons)
        save(root/'completion.json',dict(status='completed',physical_paths=len(results),actual_classifier_fits=len(physical(root)),real_api_requests=0,
            tracked_A_fits_after=1025+len(physical(root)),source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,
            formal_test_or_H_opened=False,independent_outer_splits=False))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),completed_or_partial_paths=len(results),actual_classifier_fits=len(physical(root)),
            real_api_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');main(p.parse_args())
