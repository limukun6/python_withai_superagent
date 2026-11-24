"""Training-only portable content/row/split identities using B's existing registered partition contract."""
import argparse
import os
from pathlib import Path
import time
import zipfile
import numpy as np
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','coupon-reference','assets','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(a.assets/'minilm'),
        SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS=str(a.assets/'mobilenet_v3_small-047dcff4.pth'))
    from scripts import a38_predict_guard
    from superagent.ab_run import sources,freeze_partitions,registered_capabilities
    from superagent.contracts import TaskSpec
    from superagent.data import load_training_task
    from superagent.metrics import MetricSpec
    from superagent.splits import effective_grouping
    from superagent.scheduler_bridge import fingerprint
    from tools.ab_registered_rows import content_digest,registered_partitioner
    frozen=sources();results={}
    save(out/'protocol.json',dict(batch='A94',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A94_portable_joint_inputs_plan.md'),source_files=frozen,
        actual_classifier_fits=0,real_api_requests=0,training_only=True,downstream_role_clarification='A93'))
    try:
        entries=read(a.reference/'protocol.json')['entries']
        for name in ('coupon','preference','bean'):
            entry=entries[name];src=Path(entry['training_source'])
            if sha(src)!=entry['training_source_sha256']:raise ValueError('Training package changed')
            task=load_training_task(src);folder=out/name;folder.mkdir();_,registered=freeze_partitions(task,folder,707,MetricSpec.for_task(task))
            records=[read(folder/(split+'_split.json')) for split in registered]
            manifest=dict(version=1,kind=task.kind,training_content_sha256=content_digest(task.X_train,task.y_train),
                grouping=effective_grouping(task.kind,'shared_text'),splits=records)
            save(folder/'registered_rows.json',manifest);digest=sha(folder/'registered_rows.json')
            replay=registered_partitioner(task,folder/'registered_rows.json',digest)
            for row in records:
                train,held,_=replay(task,row['seed'],'shared_text')
                if train.tolist()!=row['train_indices'] or held.tolist()!=row['search_indices']:raise ValueError('B replay changes actual rows')
                if name=='coupon':
                    original=read(a.coupon_reference/'inputs/coupon'/(row['id']+'_split.json'))
                    if any(original[key]!=row[key] for key in ('train_indices','search_indices','row_ids_sha256','label_sha256')):
                        raise ValueError('Active C1 Coupon partition identity differs')
            ordered=[content_digest([row],[label]) for row,label in zip(task.X_train,task.y_train,strict=True)]
            save(folder/'ordered_training_row_sha256.json',dict(version=1,algorithm='tools.ab_registered_rows.content_digest per one row including label',
                row_count=len(ordered),rows=ordered,source_content_sha256=manifest['training_content_sha256']))
            with zipfile.ZipFile(src) as archive:
                members={member:__import__('hashlib').sha256(archive.read(member)).hexdigest() for member in archive.namelist() if not member.endswith('/')}
            spec=TaskSpec.from_task(task).record();caps={scope:registered_capabilities(spec,len(task.y_train),scope) for scope in ('baseline','full')}
            results[name]=dict(task_spec=spec,training_rows=len(task.y_train),training_archive_sha256=sha(src),training_members_sha256=members,
                training_content_sha256=manifest['training_content_sha256'],registered_rows_sha256=digest,
                ordered_training_rows_sha256=sha(folder/'ordered_training_row_sha256.json'),
                task_budget_id=entry.get('task_budget_id','dry_bean' if name=='bean' else None),
                capabilities=caps,menu_sha256={scope:fingerprint(value) for scope,value in caps.items()},
                fit_limit=11,llm_request_limit=3,trial_timeout=300,path_time_limit=1200,registered_partition_replay_passed=True)
            save(out/'results.json',results)
        save(out/'completion.json',dict(status='passed',tasks=3,registered_splits_verified=9,actual_classifier_fits=0,real_api_requests=0,
            source_unchanged=sources()==frozen,formal_test_or_H_opened=False,elapsed_seconds=time.monotonic()-start))
        print({name:dict(rows=r['training_rows'],full_candidates=len(r['capabilities']['full']),task_budget_id=r['task_budget_id']) for name,r in results.items()})
    except Exception as error:
        save(out/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_api_requests=0,automatic_retry=False));raise
