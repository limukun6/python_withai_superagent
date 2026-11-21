"""One real-photo development path through PNG decoding, AB scheduling and portable export."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np

from scripts.a_batch_common import save
from scripts import run_b1_ab_acceptance as fixture
from scripts.run_a10_real_task_pilot import launch,read,sha
from superagent.ab_run import sources,freeze_partitions
from superagent.ab_catalog import capabilities_for
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec


def run(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);fixture.offline(root)
    began=time.monotonic();frozen=sources();result={}
    try:
        registration=read(a.prepared/'registration.json')['photo_dev'];source=a.prepared/'photo_dev'
        for name,expected in registration['files'].items():
            if sha(source/name)!=expected:raise ValueError('Prepared photo task changed')
        task=load_development_task(source/'train.zip');metric=MetricSpec.for_task(task)
        caps=capabilities_for(TaskSpec.from_task(task).record(),train_rows=len(task.y_train))
        partitions=root/'partitions';partitions.mkdir();_,allowed=freeze_partitions(task,partitions,707,metric)
        # CSV task labels are strings; preserve the identical semantic IDs in the external scorer.
        held=joblib.load(source/'holdout.joblib');held['y']=np.asarray(held['y']).astype(task.y_train.dtype)
        scoring=root/'scoring_holdout.joblib';joblib.dump(held,scoring)
        save(root/'protocol.json',dict(batch='A13-photo',source_files=frozen,registration=registration,
            plan_sha256=sha(fixture.ROOT/'docs/batches/A13_photo_and_H_registration_plan.md'),
            runner_sha256=sha(Path(__file__)),capabilities=caps,splits=allowed,fit_limit=11,api_limit=0,
            scoring_label_conversion='CSV adapter casts class IDs to string; same IDs, no class remapping',
            scoring_holdout_sha256=sha(scoring)))
        directory=root/'official_run'
        command=[sys.executable,'-m','superagent','run',str(source/'train.zip'),'--run-dir',str(directory),
            '--scheduler','ab-v1','--portfolio','ab-v1','--fit-limit','11','--seed','707',
            '--time-limit','300','--trial-timeout','60']
        save(root/'command.json',command)
        training=launch(command,root/'train.log',330)
        outcome=read(directory/'run_outcome.json');result.update(outcome=outcome,train_process_seconds=training)
        counts=outcome['classifier_fit_accounting']
        if not outcome['requested_planner_completed'] or not 2<=counts['attempted_classifier_fits']==counts['completed_classifier_fits']<=11:
            raise ValueError('Photo path did not complete within actual-fit budget')
        for split in allowed:
            if read(directory/(split+'_split.json'))!=read(partitions/(split+'_split.json')):
                raise ValueError('Actual photo partition differs from pre-registration')
        selection=read(directory/'final_selection.json');save(root/'model_frozen.json',selection)
        relocated=root/'relocated';relocated.mkdir()
        for name in ('task_spec.json','final_selection.json',selection['model_file']):shutil.copy2(directory/name,relocated/name)
        csvs=[];export_seconds=0
        for index,run_dir in enumerate((directory,relocated)):
            target=run_dir/'predictions/rows.csv'
            export_seconds+=launch([sys.executable,'-m','superagent','predict',str(source/'features.zip'),
                '--run-dir',str(run_dir),'--output',str(target)],root/f'predict_{index}.log',60)
            csvs.append(target)
        if csvs[0].read_bytes()!=csvs[1].read_bytes():raise ValueError('Photo predictions differ after relocation')
        export_seconds+=launch([sys.executable,str(fixture.ROOT/'scripts/a07_verify_export.py'),
            '--source-root',str(fixture.ROOT),'--run-dir',str(directory),'--holdout',str(scoring),
            '--predictions',str(csvs[0]),'--output',str(root/'export_verification.json')],root/'verify.log',60)
        fits=len(fixture.trace(root))
        if fits!=counts['attempted_classifier_fits'] or sources()!=frozen:raise ValueError('Source/physical fit accounting differs')
        result.update(verification=read(root/'export_verification.json'),export_seconds=export_seconds,
            actions=read(directory/'actions.json'),relocated_prediction_identical=True)
        save(root/'results.json',result)
        save(root/'completion.json',dict(status='passed',actual_classifier_fits=fits,real_api_requests=0,
            source_unchanged=True,elapsed_seconds=time.monotonic()-began))
    except Exception as error:
        save(root/'results.json',result);save(root/'completion.json',dict(status='failed',error=str(error),
            actual_classifier_fits=len(fixture.trace(root)),real_api_requests=0,elapsed_seconds=time.monotonic()-began));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
