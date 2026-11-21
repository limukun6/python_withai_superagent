"""Revalidate the two registered B blocks against the actual shared local and Flash entry options."""
import argparse
import os
from pathlib import Path
from unittest.mock import patch

from scripts import run_b1_ab_acceptance as fixture
from scripts.prepare_b_c1 import ARMS, ORDER_SALT, TASKS
from superagent import ab_run
from superagent.ab_catalog import capabilities_for
from superagent.ab_sentence import verified_directory
from superagent.cli import build_parser
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from tools.ab_registered_rows import registered_partitioner


def check(args):
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    fixture.offline(out)
    portable=fixture.read(fixture.ROOT/'research/bp_02_materialization/environment_1/results.json')
    initial=fixture.read(fixture.ROOT/'research/bc1_preparation/results.json')
    budget=dict(fits_per_arm=11,reserved_refit=1,wall_seconds=1200,trial_seconds=300,
                planned_paths=12,max_fits_across_paths=132,max_paid_requests_across_paths=24)
    source=fixture.digest(ab_run.sources())
    fixture.write(out/'protocol.json',dict(batch='BC1-02',joint_source=source,order_salt=ORDER_SALT,prior_reference=fixture.digest(initial),
        portable_reference=fixture.digest(portable),driver_sha256=fixture.sha(Path(__file__).read_bytes()),
        plan_sha256=fixture.sha((fixture.ROOT/'docs/batches/BC1_02_joint_plan.md').read_bytes()),planned_budget=budget,
        preparation_fit_limit=0,preparation_api_limit=0))
    tasks={}
    try:
        os.environ['SUPERAGENT_MINILM_WEIGHTS']=str(args.weights.resolve())
        if verified_directory() is None:
            raise ValueError('Reference paired-text directory requires the verified MiniLM resource')
        for name in TASKS:
            folder=args.portable/name
            registered=portable[name]
            for item,section in [('train.zip','optimizer'),('features.zip','delivery'),('inner_splits.json','optimizer')]:
                if fixture.sha((folder/section/item).read_bytes())!=registered['canonical_files'][item]:
                    raise ValueError('Portable input identity differs on '+name+'/'+item)
            file=folder/'optimizer/train.zip'
            data=load_development_task(file)
            if data.X_val is not None or data.y_val is not None or len(data.y_train)!=registered['training_rows']:
                raise ValueError('Unexpected training input')
            catalog=capabilities_for(TaskSpec.from_task(data).record(),train_rows=len(data.y_train))
            expected=initial['tasks'][name]
            if fixture.digest(catalog)!=expected['catalog_sha256'] or list(catalog)!=expected['candidate_ids']:
                raise ValueError('Shared catalogue differs from the registered B block')
            dest=out/name
            dest.mkdir()
            selector=registered_partitioner(data,folder/'optimizer/inner_splits.json',registered['canonical_files']['inner_splits.json'])
            with patch.object(ab_run,'grouped_partition',selector):
                _,parts=ab_run.freeze_partitions(data,dest,707,MetricSpec.for_task(data))
            for split,v in parts.items():
                if (any(v[k]!=expected['splits'][split][k] for k in ('id','seed','row_ids_sha256','label_sha256'))
                        or v['evaluation_payload_sha256']!=registered['splits'][split]['local_worker_sha256']):
                    raise ValueError('Frozen workers disagree on registered rows or local binary payload')
            arms=[]
            order=sorted(ARMS,key=lambda k:fixture.sha((ORDER_SALT+'/'+name+'/'+k).encode()))
            if order!=[b['arm'] for b in initial['scheduled_blocks'][name]]:
                raise ValueError('C1 execution order changed')
            for arm in order:
                planner,allocation=ARMS[arm]
                scheduler='ab-fixed' if allocation=='fixed' else 'ab-v1'
                command=['run',file.name,'--scheduler',scheduler,'--portfolio','ab-v1','--planner',planner,'--fit-limit','11',
                         '--seed','707','--time-limit','1200','--trial-timeout','300']
                flags=build_parser().parse_args(command)
                try:
                    limit=ab_run.validate_options(flags)
                    verdict=dict(option_support=True,actual_fit_limit=limit)
                except ValueError as error:
                    verdict=dict(option_support=False,reason=str(error))
                arms.append(dict(arm=arm,scheduler=scheduler,planner=planner,model_request_cap=0 if planner=='local' else 3,
                                anchor=next(iter(catalog)),validated_flags=command,**verdict))
            tasks[name]=dict(candidates=list(catalog),catalog_sha256=fixture.digest(catalog),inputs=registered['canonical_files'],
                             training_rows=len(data.y_train),held_rows=registered['held_rows'],splits=parts,arms=arms)
        fixture.write(out/'results.json',tasks)
        supported=sum(a['option_support'] for t in tasks.values() for a in t['arms'])
        fixture.write(out/'completion.json',dict(status='checked',scheduled=12,options_accepted=supported,
                      options_rejected=12-supported,executed_c1_paths=0,fits=0,model_api_requests=0,outer_scores=0))
    except Exception as error:
        fixture.write(out/'completion.json',dict(status='failed',reason=str(error),checked_tasks=list(tasks),fits=0,model_api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('portable','weights','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    check(parser.parse_args())
