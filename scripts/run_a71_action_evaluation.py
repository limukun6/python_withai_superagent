"""Uniform 180-second evaluation of already sealed A70 actions, no API requests."""
import argparse
from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts import run_a55_fresh_feedback as engine
from superagent.ab_run import sources
from tools.unattended_runtime import prevent_idle_sleep


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','answers','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();start=time.monotonic()
    engine.SEEDS=(30311,30321);original=engine.train_candidate
    window=root/'window.json';save(window,dict(deadline=(datetime.now(timezone.utc)+timedelta(minutes=45)).isoformat()));a.window_config=window
    save(root/'protocol.json',dict(batch='A71',source_files=frozen,runner_sha256=sha(Path(__file__)),reused_engine_sha256=sha(Path(engine.__file__)),
        plan_sha256=sha(Path(__file__).resolve().parents[1]/'docs/batches/A71_frozen_action_evaluation_recovery_plan.md'),
        maximum_new_classifier_fits=20,real_api_requests=0,worker_timeout_seconds=180,original_A70_failure_retained=True,
        answers_sha256=sha(a.answers/'answers.json'),starting_tracked_A_fits=1194))
    def run_worker(payload,candidate,destination,old_timeout,**kwargs):
        if engine.remaining(a)<210:raise TimeoutError('Not enough time for another bounded evaluation')
        return original(payload,candidate,destination,180,**kwargs)
    engine.train_candidate=run_worker
    def offline(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('A71 no network or API allowed')
    sys.addaudithook(offline)
    try:
        with prevent_idle_sleep():result=engine.evaluate(a,root,frozen)
        if result['actual_classifier_fits']>20:raise ValueError('Unexpected unique-action fit count')
        comparisons=read(root/'comparison.json')
        for row in comparisons:
            row['local_next_candidate']=row.pop('original_R_candidate');flag=row.pop('fresh_R_changed')
            row['differs_from_local_next']=flag if row['local_next_candidate'] is not None else None
        save(root/'comparison.json',comparisons)
        result.update(elapsed_seconds=time.monotonic()-start,source_unchanged=sources()==frozen,
            worker_timeout_seconds=180,tracked_A_fits_after=1194+result['actual_classifier_fits'],no_new_answers_requested=True)
        save(root/'completion.json',result);print(result)
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=len(engine.physical(root)),
            real_api_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise
