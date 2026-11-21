"""Freeze five development task manifests and six-arm commands without training or API."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import zipfile

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources,registered_capabilities,freeze_partitions
from superagent.contracts import TaskSpec
from superagent.data import load_training_task
from superagent.metrics import MetricSpec


MODES=[('L0','local','ab-fixed'),('L1','local','ab-v1'),('R0','raw-history','ab-fixed'),
       ('E0','checked-history','ab-fixed'),('R1','raw-history','ab-v1'),('E1','checked-history','ab-v1')]
SCOPES={'coupon':'task1','preference':'task2','bean':'dry_bean','sms':'sms_spam','photo':'olivetti_faces'}


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);source=sources();entries=[]
    tasks=[('coupon',args.data/'released_task1.zip'),('preference',args.data/'released_task2.zip'),
        ('bean',args.generic_prepared/'bean/train.zip'),('sms',args.generic_prepared/'sms/train.zip'),
        ('photo',args.photo_prepared/'photo_dev/train.zip')]
    save(root/'protocol.json',dict(batch='A18-C1-readiness',source_files=source,
        runner_sha256=sha(Path(__file__)),actual_classifier_fits=0,real_api_requests=0,
        formal_test_opened=False,scope='Preparation, not executed C1 or final D5 freeze'))
    for name,path in tasks:
        opened=[];original=zipfile.ZipFile.read
        def training_only(archive,member,*a,**kw):
            value=member.filename if isinstance(member,zipfile.ZipInfo) else member
            lower=value.casefold()
            if any(x in lower for x in ('test','validation','val.csv','c_val','y_val')):
                raise ValueError('Readiness attempted to open nontraining data')
            opened.append(value);return original(archive,member,*a,**kw)
        taskdir=root/name;taskdir.mkdir()
        with patch.object(zipfile.ZipFile,'read',new=training_only):task=load_training_task(path)
        spec=TaskSpec.from_task(task).record();metric=MetricSpec.for_task(task)
        caps=registered_capabilities(spec,len(task.y_train),'full')
        gaps=[]
        if not caps:gaps.append('No legal executable candidates; clarify declared pair semantics/label direction or add matching methods')
        if spec['goal_resolution']['fields']['primary_metric']['status']!='confirmed':
            gaps.append('Primary metric is an executable default, not a verified declared objective')
        splits={}
        try:_,splits=freeze_partitions(task,taskdir,707,metric)
        except Exception as error:gaps.append('Partition preparation failed: '+str(error))
        commands=[]
        for arm,planner,scheduler in MODES:
            command=[sys.executable,'-m','superagent','run',str(path.resolve()),'--planner',planner,
                '--scheduler',scheduler,'--portfolio','ab-v1','--catalog-scope','full','--fit-limit','11',
                '--seed','707','--time-limit','300','--trial-timeout','60','--run-dir',str(root/'future_runs'/name/arm)]
            reasons=list(gaps)
            if planner!='local':
                command+=['--api-key-file','<API_KEY_FILE>','--budget-ledger','<VERIFIED_CUMULATIVE_LEDGER>',
                    '--task-budget-id',SCOPES[name],'--llm-budget-rmb','.30']
                reasons.append('11 historical unknown requests unreconciled; verified cumulative ledger required')
            commands.append(dict(arm=arm,planner=planner,scheduler=scheduler,command=command,
                status='prepared_not_executed' if not reasons else 'blocked',blocked_reasons=reasons))
        entry=dict(task=name,source_archive=str(path.resolve()),task_archive_sha256=sha(path),opened_members=opened,
            task_budget_id=SCOPES[name],
            task_description_source=task.spec.get('task_description_source'),
            rows=len(task.y_train),class_counts={str(k):v for k,v in Counter(task.y_train.tolist()).items()},
            task_spec=spec,capabilities=caps,splits=splits,commands=commands,task_gaps=gaps)
        entries.append(entry);save(taskdir/'readiness.json',entry)
    save(root/'readiness.json',entries)
    (root/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    gaps=sum(bool(t['task_gaps']) for t in entries)
    save(root/'completion.json',dict(status='prepared_with_gaps' if gaps else 'ready_local_only',tasks=len(entries),
        arms=6*len(entries),tasks_with_input_or_method_gaps=gaps,actual_classifier_fits=0,real_api_requests=0,
        source_unchanged=sources()==source,formal_experiments_completed=0))
    print(json.dumps(dict(tasks=[dict(name=e['task'],candidates=len(e['capabilities']),gaps=e['task_gaps']) for e in entries])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--generic-prepared',type=Path,required=True);p.add_argument('--photo-prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
