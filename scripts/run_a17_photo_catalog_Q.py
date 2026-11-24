"""Same-source photo catalogue Q control, with all models frozen before scoring."""
import argparse
import csv
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
from superagent.ab_run import sources,freeze_partitions,registered_capabilities
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);fixture.offline(root)
    began=time.monotonic();frozen=sources();results=[]
    try:
        registration=read(args.prepared/'registration.json')['photo_dev'];source=args.prepared/'photo_dev'
        for name,digest in registration['files'].items():
            if sha(source/name)!=digest:raise ValueError('Photo bytes changed: '+name)
        task=load_development_task(source/'train.zip');spec=TaskSpec.from_task(task).record();metric=MetricSpec.for_task(task)
        caps={scope:registered_capabilities(spec,len(task.y_train),scope) for scope in ('baseline','full')}
        if not caps['baseline'] or not set(caps['baseline'])<set(caps['full']):
            raise ValueError('Q requires a nonempty proper subset baseline')
        partitions=root/'partitions';partitions.mkdir();_,allowed=freeze_partitions(task,partitions,707,metric)
        duplicate_checks=[]
        for name in ('bean','sms'):
            data=load_development_task(args.generic_prepared/name/'train.zip');contract=TaskSpec.from_task(data).record()
            old,new=(registered_capabilities(contract,len(data.y_train),scope) for scope in ('baseline','full'))
            duplicate_checks.append(dict(task=name,catalogs_identical=old==new,
                status='deduplicated_identical_catalog' if old==new else 'B_submatrix_pending',
                added_ids=sorted(set(new)-set(old)),new_classifier_fits=0))
        protocol=dict(batch='A17-Q-photo',source_files=frozen,baseline='9b4fc18',
            plan_sha256=sha(fixture.ROOT/'docs/batches/A17_photo_catalog_Q_plan.md'),runner_sha256=sha(Path(__file__)),
            registration=registration,catalogs=caps,splits=allowed,task_contract=spec,
            fit_limit_per_path=11,batch_fit_limit=22,real_api_limit=0,duplicate_checks=duplicate_checks,
            validate_only=args.validate_only,selection='same B local scheduler and paired finalization in both scopes')
        save(root/'protocol.json',protocol)
        if args.validate_only:
            save(root/'completion.json',dict(status='validated',actual_classifier_fits=0,real_api_requests=0));return
        for scope in ('baseline','full'):
            if sources()!=frozen:raise ValueError('Production source changed')
            directory=root/scope
            command=[sys.executable,'-m','superagent','run',str(source/'train.zip'),'--run-dir',str(directory),
                '--planner','local','--scheduler','ab-v1','--portfolio','ab-v1','--catalog-scope',scope,
                '--fit-limit','11','--seed','707','--time-limit','300','--trial-timeout','60']
            save(root/(scope+'_command.json'),command)
            seconds=launch(command,root/(scope+'_train.log'),330)
            outcome=read(directory/'run_outcome.json');account=outcome['classifier_fit_accounting']
            if not (outcome['requested_planner_completed'] and outcome['new_api_requests']==0
                    and not account['unknown_failed_attempts']
                    and 2<=account['attempted_classifier_fits']==account['completed_classifier_fits']<=11):
                raise ValueError('Q path failed actual fit or completion contract')
            manifest=read(directory/'run_manifest.json')
            if manifest['source_files']!=frozen or manifest['capabilities']!=caps[scope]:
                raise ValueError('Q source/candidate identity differs from frozen protocol')
            for split in allowed:
                if read(directory/(split+'_split.json'))!=read(partitions/(split+'_split.json')):
                    raise ValueError('Q actual split differs')
            selection=read(directory/'final_selection.json')
            if sha(directory/selection['model_file'])!=selection['model_sha256']:raise ValueError('Model hash differs')
            results.append(dict(scope=scope,outcome=outcome,selection=selection,train_process_seconds=seconds,
                actions=read(directory/'actions.json')))
            save(root/'checkpoint.json',results)
            print(json.dumps(dict(scope=scope,status='model_frozen',fits=account['attempted_classifier_fits'])),flush=True)
        save(root/'all_models_frozen.json',[r['selection'] for r in results])
        held=joblib.load(source/'holdout.joblib');held['y']=np.asarray(held['y']).astype(task.y_train.dtype)
        scoring=root/'scoring_holdout.joblib';joblib.dump(held,scoring)
        vectors=[]
        for result in results:
            scope=result['scope'];directory=root/scope;selection=result['selection']
            relocated=directory/'relocated';relocated.mkdir()
            for name in ('task_spec.json','final_selection.json',selection['model_file']):shutil.copy2(directory/name,relocated/name)
            outputs=[];export_seconds=0
            for index,modeldir in enumerate((directory,relocated)):
                target=modeldir/'predictions/rows.csv'
                export_seconds+=launch([sys.executable,'-m','superagent','predict',str(source/'features.zip'),
                    '--run-dir',str(modeldir),'--output',str(target)],root/f'{scope}_predict_{index}.log',60)
                outputs.append(target)
            if outputs[0].read_bytes()!=outputs[1].read_bytes():raise ValueError('Relocated prediction differs')
            export_seconds+=launch([sys.executable,str(fixture.ROOT/'scripts/a07_verify_export.py'),
                '--source-root',str(fixture.ROOT),'--run-dir',str(directory),'--holdout',str(scoring),
                '--predictions',str(outputs[0]),'--output',str(directory/'verification.json')],root/(scope+'_verify.log'),60)
            with outputs[0].open(encoding='utf-8',newline='') as file:rows=list(csv.DictReader(file))
            vectors.append(np.asarray([r['prediction'] for r in rows]))
            result.update(verification=read(directory/'verification.json'),export_seconds=export_seconds,
                migrated_prediction_identical=True)
        before,after=vectors;labels=held['y'].astype(str)
        comparison=dict(score_delta=results[1]['verification']['score']-results[0]['verification']['score'],
            disagreements=int(np.sum(before!=after)),full_only_correct=int(np.sum((after==labels)&(before!=labels))),
            baseline_only_correct=int(np.sum((before==labels)&(after!=labels))),
            scope='One known development photo partition; catalogue effect, not LLM/scheduler efficacy')
        total=sum(r['outcome']['classifier_fit_accounting']['attempted_classifier_fits'] for r in results)
        if total!=len(fixture.trace(root)) or total>22 or sources()!=frozen:raise ValueError('Q final physical/source accounting differs')
        save(root/'results.json',dict(paths=results,comparison=comparison))
        save(root/'completion.json',dict(status='passed',actual_classifier_fits=total,real_api_requests=0,
            independent_observed_fits=len(fixture.trace(root)),source_unchanged=True,elapsed_seconds=time.monotonic()-began))
    except Exception as error:
        save(root/'results.json',dict(paths=results));save(root/'completion.json',dict(status='failed',error=str(error),
            actual_classifier_fits=len(fixture.trace(root)),real_api_requests=0,elapsed_seconds=time.monotonic()-began));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',required=True,type=Path)
    p.add_argument('--generic-prepared',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--validate-only',action='store_true');run(p.parse_args())
