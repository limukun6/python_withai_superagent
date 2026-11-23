"""Three complete autonomous object-image catalogue arms, zero LLM requests, delayed scoring."""
import argparse
import csv
import os
from pathlib import Path
import sys
import time
import joblib
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch,physical
from superagent.ab_run import sources,registered_capabilities,freeze_partitions
from superagent.contracts import TaskSpec
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from tools.unattended_runtime import prevent_idle_sleep

ROOT=Path(__file__).resolve().parents[1]
ARMS=[('baseline','baseline',False,7),('full','full',True,9),('no_encoder','full',False,8)]


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();rows=[]
    save(root/'protocol.json',dict(batch='A96',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A96_flower_three_arm_Q_plan.md'),source_files=frozen,
        maximum_classifier_fits=33,starting_A_fits=1353,real_api_requests=0,arms=ARMS,registration=read(a.data/'registration.json')))
    try:
        registration=read(a.data/'registration.json')
        if read(a.data/'completion.json')['status']!='prepared' or sha(a.data/'train.zip')!=registration['training_package_sha256']:raise ValueError('Registered training input changed')
        task=load_training_task(a.data/'train.zip');spec=TaskSpec.from_task(task).record();inputs=root/'inputs';inputs.mkdir()
        _,splits=freeze_partitions(task,inputs,707,MetricSpec.for_task(task))
        for name,scope,encoder,expected in ARMS:
            if sources()!=frozen:raise ValueError('Source changed')
            os.environ.update(SUPERAGENT_MINILM_WEIGHTS='',SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS=str(a.assets/'mobilenet_v3_small-047dcff4.pth') if encoder else '')
            caps=registered_capabilities(spec,len(task.y_train),scope)
            if len(caps)!=expected:raise ValueError('Executable catalogue size differs')
            env={**os.environ,'A74_OFFLINE':'1','A74_SOURCE_ROOT':str(ROOT),'AB_A29_OBSERVE_FITS':'1',
                'AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),'AB_A29_RUN_NAME':name,
                'PYTHONPATH':os.pathsep.join([str(ROOT/'scripts/a74_offline_guard'),str(ROOT)]),'PYTHONIOENCODING':'utf-8',
                'SUPERAGENT_OFFLINE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
            directory=root/name;code,seconds=launch([sys.executable,'-m','superagent','run',str(a.data/'train.zip'),
                '--run-dir',str(directory),'--scheduler','ab-v1','--portfolio','ab-v1','--catalog-scope',scope,'--fit-limit','11',
                '--seed','707','--time-limit','1200','--trial-timeout','300','--partial-delivery','checkpoint'],root/(name+'.log'),env,1260)
            if code or not (directory/'run_outcome.json').exists():raise RuntimeError('Official path stopped: '+name)
            outcome=read(directory/'run_outcome.json');count=sum(r['run_name']==name for r in physical(root))
            if not outcome['requested_planner_completed'] or not 2<=count==outcome['classifier_fit_accounting']['completed_classifier_fits']<=11:raise ValueError('Path completion/fit count differs')
            manifest=read(directory/'run_manifest.json')
            if manifest['source_files']!=frozen or manifest['capabilities']!=caps:raise ValueError('Source/menu identity differs')
            for split in splits:
                if read(directory/(split+'_split.json'))!=read(inputs/(split+'_split.json')):raise ValueError('Actual split differs')
            selected=read(directory/'final_selection.json')
            if sha(directory/selected['model_file'])!=selected['model_sha256']:raise ValueError('Frozen model changed')
            rows.append(dict(arm=name,candidates=len(caps),actual_classifier_fits=count,process_seconds=seconds,selection=selected,
                executed_candidates=[r['candidate'] for r in read(directory/'history.json')]))
            save(root/'checkpoint.json',rows);print(dict(arm=name,status='frozen',fits=count,seconds=seconds),flush=True)
        save(root/'all_models_frozen.json',rows)
        if sha(a.data/'features.zip')!=registration['feature_package_sha256'] or sha(a.data/'holdout.joblib')!=registration['holdout_sha256']:raise ValueError('Development holdout changed')
        truth=[str(v) for v in joblib.load(a.data/'holdout.joblib')['y']];predictions={}
        for row in rows:
            name=row['arm'];directory=root/name;env={**os.environ,'SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':str(a.assets/'mobilenet_v3_small-047dcff4.pth'),'PYTHONIOENCODING':'utf-8'}
            destination=directory/'predictions/rows.csv'
            commands=[[sys.executable,'-m','scripts.a38_predict_guard',str(a.data/'features.zip'),'--run-dir',str(directory),'--output',str(destination)],
                [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(directory),
                 '--holdout',str(a.data/'holdout.joblib'),'--predictions',str(destination),'--output',str(directory/'verification.json')]]
            for i,command in enumerate(commands):
                code,_=launch(command,root/(name+f'_export{i}.log'),env,300)
                if code:raise RuntimeError('Frozen prediction failed: '+name)
            with destination.open(encoding='utf-8',newline='') as stream:records=list(csv.DictReader(stream))
            if [int(r['row_id']) for r in records]!=list(range(len(truth))):raise ValueError('Prediction row identity differs')
            predictions[name]=[r['prediction'] for r in records];row['evaluation']=read(directory/'verification.json');save(root/'results.json',rows)
        contrasts=[]
        for left,right in [('no_encoder','baseline'),('full','no_encoder'),('full','baseline')]:
            lp,rp=predictions[left],predictions[right]
            contrasts.append(dict(left=left,right=right,left_correct_only=sum(x==y and z!=y for x,z,y in zip(lp,rp,truth,strict=True)),
                right_correct_only=sum(z==y and x!=y for x,z,y in zip(lp,rp,truth,strict=True)),prediction_disagreements=sum(x!=z for x,z in zip(lp,rp,strict=True))))
        save(root/'contrasts.json',contrasts)
        if len(physical(root))!=sum(r['actual_classifier_fits'] for r in rows):raise ValueError('Unexpected inference fit')
        save(root/'completion.json',dict(status='completed',actual_classifier_fits=len(physical(root)),A_cumulative_fits_after=1353+len(physical(root)),
            real_api_requests=0,source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,formal_test_or_H_opened=False))
    except Exception as e:
        save(root/'completion.json',dict(status='stopped',error=str(e),actual_classifier_fits=len(physical(root)),
            A_cumulative_fits_after=1353+len(physical(root)),real_api_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','assets','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with prevent_idle_sleep():main(a)
