"""One real anonymous official run to qualify the B coordinator and the transferred A25 source."""
import argparse
import os
from pathlib import Path

from scripts import run_b1_ab_acceptance as rec, run_b1_05_pair_pilot as pilot, run_b_c1_matrix as matrix
from superagent import ab_run
from superagent.contracts import TaskSpec
from superagent.metrics import MetricSpec


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    rec.offline(output)
    os.environ.update(SUPERAGENT_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    source=rec.digest(ab_run.sources())
    rec.write(output/'protocol.json',dict(batch='BA25-02',source=source,
        runner_sha256=ab_run.sha(Path(__file__)),
        plan_sha256=ab_run.sha(rec.ROOT/'docs/batches/BA25_02_receiver_plan.md'),
        seed=707,max_fits=11,model_api_cap=0,new_scores=0))
    try:
        task,query=rec.packages(output)
        catalog=ab_run.registered_capabilities(TaskSpec.from_task(task).record(),len(task.y_train),'full')
        registered=output/'registered'
        registered.mkdir()
        _,splits=ab_run.freeze_partitions(task,registered,707,MetricSpec.for_task(task))
        entry=dict(name='anonymous',anchor=next(iter(catalog)),files={'train.zip':ab_run.sha(output/'train.zip')},
            catalog_sha256=rec.digest(catalog),catalog_scope='full',history_policy='strict',splits=splits)
        trained=pilot.train_one(output,entry,source,launcher=('-m','superagent','run','--planner','local',
            '--catalog-scope','full','--history-policy','strict'))
        checked=matrix.verify_run(output/'run',source,entry,'L1')
        if checked['fits']!=len(rec.trace(output)) or checked['model_sha256']!=trained['model_sha256']:
            raise ValueError('Strict acceptance and independent physical counts disagree')
        delivery=rec.predict_twice(output/'run',query,output_dir=output,task_package=output/'test.zip')
        rec.write(output/'results.json',dict(actions=trained['actions'],source=source,strict_receipt=checked,
                delivery=delivery,action_counts=trained['action_counts']))
        rec.write(output/'completion.json',dict(status='passed',actual_fits=checked['fits'],model_api_requests=0,
            course_cost_rmb=0,fresh_prediction_processes=2,matching_prediction_rows=len(query),scores=0))
    except Exception as error:
        rec.write(output/'completion.json',dict(status='failed',error=str(error),actual_fits=len(rec.trace(output)),
            model_api_requests=0))
        raise


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    run(cli.parse_args().output.resolve())
