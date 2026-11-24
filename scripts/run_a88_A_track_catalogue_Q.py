"""Frozen-source two-task A-track catalogue contrasts; score only after all four models freeze."""
import argparse
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
ORDER=[('coupon','baseline'),('photo','full'),('coupon','full'),('photo','baseline')]


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();results=[]
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(a.assets/'minilm'),
        SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS=str(a.assets/'mobilenet_v3_small-047dcff4.pth'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    immutable={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/a74_offline_guard/sitecustomize.py',
        ROOT/'scripts/a29_worker_observer/sitecustomize.py',ROOT/'scripts/run_a34_transfer_coverage.py',ROOT/'scripts/run_a29_choice_matrix.py']}
    save(root/'protocol.json',dict(batch='A88',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A88_A_track_catalogue_Q_plan.md'),script_hashes=immutable,
        order=ORDER,maximum_classifier_fits=44,starting_A_fits=1247,real_api_requests=0,
        trial_timeout=300,path_time_limit=1200,environment=read(a.assets/'environment.json'),assets=read(a.assets/'assets.json')))
    def offline(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('A88 offline')
    sys.addaudithook(offline)
    env={**os.environ,'A74_OFFLINE':'1','A74_SOURCE_ROOT':str(ROOT),'AB_A29_OBSERVE_FITS':'1',
        'AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),'PYTHONPATH':os.pathsep.join([str(ROOT/'scripts/a74_offline_guard'),str(ROOT)]),
        'SUPERAGENT_OFFLINE':'1','PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    try:
        if read(a.assets/'completion.json')['status']!='ready':raise ValueError('Assets unavailable')
        previous=read(a.reference/'protocol.json')['entries'];snapshot=read(a.snapshot/'results.json');entries={}
        for name in ('coupon','photo'):
            entry=previous[name];archive=Path(entry['training_source'])
            if sha(archive)!=entry['training_source_sha256']:raise ValueError('Training input changed')
            task=load_training_task(archive);spec=TaskSpec.from_task(task).record()
            caps={scope:registered_capabilities(spec,len(task.y_train),scope) for scope in ('baseline','full')}
            if any(caps[scope]!=snapshot[name][scope+'_capabilities'] for scope in caps):raise ValueError('Snapshot differs from executable menu')
            folder=root/'inputs'/name;folder.mkdir(parents=True);_,splits=freeze_partitions(task,folder,707,MetricSpec.for_task(task))
            entries[name]=dict(entry,task_spec_current=spec,current_capabilities=caps,registered_splits=splits)
        protocol=read(root/'protocol.json');protocol['entries']=entries;save(root/'protocol.json',protocol)
        for name,scope in ORDER:
            if sources()!=frozen or any(sha(ROOT/p)!=digest for p,digest in immutable.items()):raise ValueError('Frozen code changed')
            directory=root/(name+'_'+scope);active={**env,'AB_A29_RUN_NAME':directory.name}
            code,seconds=launch([sys.executable,'-m','superagent','run',entries[name]['training_source'],'--run-dir',str(directory),
                '--scheduler','ab-v1','--portfolio','ab-v1','--catalog-scope',scope,'--fit-limit','11','--seed','707',
                '--time-limit','1200','--trial-timeout','300','--partial-delivery','checkpoint'],root/(directory.name+'.log'),active,1260)
            if not (directory/'run_outcome.json').exists():raise RuntimeError('No official outcome; see '+directory.name+'.log')
            outcome=read(directory/'run_outcome.json');observed=sum(r['run_name']==directory.name for r in physical(root))
            results.append(dict(task=name,scope=scope,process_seconds=seconds,actual_classifier_fits=observed,outcome=outcome))
            save(root/'checkpoint.json',results)
            if code or not outcome['requested_planner_completed']:raise RuntimeError('Path stopped without retry: '+directory.name)
            counts=outcome['classifier_fit_accounting'];manifest=read(directory/'run_manifest.json')
            if not 2<=observed==counts['attempted_classifier_fits']==counts['completed_classifier_fits']<=11:raise ValueError('Fit counts differ')
            if manifest['source_files']!=frozen or manifest['capabilities']!=entries[name]['current_capabilities'][scope]:raise ValueError('Source/menu differs')
            for split in entries[name]['registered_splits']:
                if read(directory/(split+'_split.json'))!=read(root/'inputs'/name/(split+'_split.json')):raise ValueError('Actual split differs')
            selection=read(directory/'final_selection.json')
            if sha(directory/selection['model_file'])!=selection['model_sha256']:raise ValueError('Model changed')
            results[-1]['selection']=selection;save(root/'checkpoint.json',results)
            print(dict(task=name,scope=scope,status='model_frozen',fits=observed,seconds=seconds),flush=True)
        save(root/'all_models_frozen.json',[dict(task=r['task'],scope=r['scope'],selection=r['selection']) for r in results])
        held={name:evaluate_inputs(entries[name],root,4) for name in entries}
        for row in results:
            directory=root/(row['task']+'_'+row['scope']);folder=held[row['task']];destination=directory/'predictions/rows.csv'
            commands=[[sys.executable,'-m','scripts.a38_predict_guard',str(folder/'features.zip'),'--run-dir',str(directory),'--output',str(destination)],
                [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(directory),
                 '--holdout',str(folder/'holdout.joblib'),'--predictions',str(destination),'--output',str(directory/'verification.json')]]
            for i,command in enumerate(commands):
                code,_=launch(command,root/(directory.name+f'_export_{i}.log'),{**env,'AB_A29_RUN_NAME':'inference_only'},300)
                if code:raise RuntimeError('Post-freeze delivery/evaluation failed: '+directory.name)
            row['evaluation']=read(directory/'verification.json');save(root/'results.json',results)
        if len(physical(root))!=sum(r['actual_classifier_fits'] for r in results):raise ValueError('Inference fitted a classifier')
        save(root/'completion.json',dict(status='completed',physical_paths=4,tasks=2,actual_classifier_fits=len(physical(root)),
            A_cumulative_fits_after=1247+len(physical(root)),real_api_requests=0,source_unchanged=sources()==frozen,
            elapsed_seconds=time.monotonic()-start,formal_test_or_H_opened=False,not_complete_five_task_Q=True))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=len(physical(root)),
            A_cumulative_fits_after=1247+len(physical(root)),real_api_requests=0,automatic_retry=False,completed_paths=len(results),
            elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','assets','snapshot','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
