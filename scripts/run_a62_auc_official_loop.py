"""One three-fit real AUC CLI acceptance; same seen SMS data, explicit changed objective."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch,physical
from superagent.ab_run import sources


ROOT=Path(__file__).resolve().parents[1]


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources()
    save(root/'protocol.json',dict(batch='A62',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A62_auc_official_loop_plan.md'),maximum_classifier_fits=3,
        real_api_requests=0,starting_tracked_A_fits=996,planned_tracked_A_ceiling=999,
        same_seen_task_changed_goal=True,formal_test_or_H_opened=False))
    def offline(event,values):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('A62 network forbidden')
    sys.addaudithook(offline)
    try:
        entry=read(a.reference/'protocol.json')['entries']['sms'];original=Path(entry['training_source'])
        if sha(original)!=entry['training_source_sha256']:raise ValueError('SMS training source changed')
        archive=root/'auc_training.zip'
        with zipfile.ZipFile(original) as source,zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as dest:
            for name in source.namelist():
                if name not in ('train.csv','manifest.json'):raise ValueError('Expected SMS training-only members')
                content=source.read(name)
                if name=='manifest.json':
                    spec=json.loads(content);spec.update(metric='roc_auc',positive_label='spam')
                    if set(spec['splits'])!={'train'}:raise ValueError('Task contains nontraining split')
                    content=json.dumps(spec).encode()
                dest.writestr(name,content)
        protocol=read(root/'protocol.json');protocol.update(original_training_sha256=sha(original),auc_training_sha256=sha(archive))
        save(root/'protocol.json',protocol)
        if not a.execute:
            save(root/'completion.json',dict(status='validated_not_executed',actual_classifier_fits=0,real_api_requests=0));return
        observer=ROOT/'scripts/a29_worker_observer/sitecustomize.py';directory=root/'sms_auc'
        env={**os.environ,'AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),
            'PYTHONPATH':os.pathsep.join([str(observer.parent),str(ROOT)]),'SUPERAGENT_OFFLINE':'1',
            'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','AB_A29_RUN_NAME':'sms_auc'}
        cmd=[sys.executable,'-m','superagent','run',str(archive),'--run-dir',str(directory),
            '--scheduler','ab-v1','--portfolio','ab-v1','--fit-limit','3','--seed','707',
            '--time-limit','600','--trial-timeout','60','--partial-delivery','checkpoint']
        code,seconds=launch(cmd,root/'train.log',env,660);outcome=read(directory/'run_outcome.json')
        if code or not outcome['requested_planner_completed']:raise RuntimeError('AUC run failed, no retry: '+str(outcome.get('error')))
        if len(physical(root))!=3 or outcome['classifier_fit_accounting']['completed_classifier_fits']!=3:
            raise ValueError('Physical AUC fit counts differ')
        manifest=read(directory/'run_manifest.json')
        if manifest['source_files']!=frozen or outcome['primary_metric']!='roc_auc' or outcome['new_api_requests']!=0:
            raise ValueError('Source/metric/API identity differs')
        selection=read(directory/'final_selection.json');save(root/'model_frozen.json',selection)
        if sha(directory/selection['model_file'])!=selection['model_sha256']:raise ValueError('Frozen model changed')
        # Only now copy already registered outer development features and labels.
        folder=root/'evaluation';folder.mkdir()
        for name,digest in entry['evaluation_files_sha256'].items():
            original_file=Path(entry['evaluation_source'])/name
            if sha(original_file)!=digest:raise ValueError('Frozen development holdout changed')
            shutil.copy2(original_file,folder/name)
        pred=directory/'predictions/rows.csv'
        commands=[
            [sys.executable,'-m','scripts.a38_predict_guard',str(folder/'features.zip'),'--run-dir',str(directory),'--output',str(pred)],
            [sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(directory),
             '--holdout',str(folder/'holdout.joblib'),'--predictions',str(pred),'--output',str(directory/'verification.json')],
            [sys.executable,'-m','tools.export_b_handoff','--run-dir',str(directory),'--output',str(root/'b_terminal.json')]]
        for i,cmd in enumerate(commands):
            code,_=launch(cmd,root/f'export_{i}.log',{**env,'AB_A29_RUN_NAME':'inference_only'},60)
            if code:raise RuntimeError('AUC inference/interface verification failed')
        if len(physical(root))!=3:raise ValueError('Unexpected training during AUC export')
        save(root/'results.json',dict(outcome=outcome,selection=selection,evaluation=read(directory/'verification.json'),
            prediction_manifest=read(pred.parent/'prediction_manifest.json'),process_seconds=seconds))
        save(root/'completion.json',dict(status='completed',actual_classifier_fits=3,real_api_requests=0,
            tracked_A_fits_after=999,elapsed_seconds=time.monotonic()-start,source_unchanged=sources()==frozen,
            formal_test_or_H_opened=False,AB_AUC_end_to_end_enabled=True,
            scope='One seen binary task with changed explicit AUC objective; not unseen or superiority evidence'))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=len(physical(root)),
            real_api_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');main(p.parse_args())
