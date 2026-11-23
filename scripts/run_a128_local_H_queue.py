"""Execute two registered local H1 paths serially, seal models, and never open outer labels."""
import argparse
import json
from pathlib import Path
import sys
import time
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a122_current_agent_acceptance import environment,child,physical

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();source=hashes(ROOT)
    save(a.output/'protocol.json',dict(batch='A128-H1-local',source_files=source,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A128_H1_local_snapshot_plan_20261123_zh.md'),
        snapshot_registration_sha256=sha(a.snapshot/'registration.json'),maximum_classifier_fits=22,
        real_API_requests=0,starting_A_fits=1538,outer_labels_accessed=False,run_order=['61411','61421']))
    results=[]
    def count():return sum(len(physical(a.output/seed)) for seed in ('61411','61421'))
    try:
        for seed in ('61411','61421'):
            target=a.output/seed
            command=[sys.executable,'-m','scripts.run_a123_H1_frozen','train','--prepared',str(a.snapshot),
                     '--outer',seed,'--arm','L0','--output',str(target)]
            process=child(command,a.output/(seed+'.log'),environment(a.output),1320)
            done=read(target/'completion.json');seal=read(target/'model_seal.json')
            if done['status']!='sealed' or done['real_API_requests'] or not seal['complete']:
                raise ValueError('H1 local path incomplete; no automatic retry')
            trace=physical(target)
            if len(trace)!=done['actual_classifier_fits'] or len(trace)>11:raise ValueError('Physical H1 count mismatch')
            results.append(dict(outer=seed,arm='L0',process=process,completion=done,model_seal=seal,
                actual_classifier_fits=len(trace),actual_outer_scores=None))
            save(a.output/'checkpoint.json',results);print(dict(outer=seed,status='sealed_unscored',fits=len(trace)),flush=True)
        if source!=hashes(ROOT) or count()>22:raise ValueError('Source/budget changed')
        save(a.output/'results.json',results)
        save(a.output/'completion.json',dict(status='local_paths_sealed_pending_R0',actual_classifier_fits=count(),
            real_API_requests=0,A_cumulative_fits_after=1538+count(),outer_scores_computed=0,source_unchanged=True,
            elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=count(),
            real_API_requests=0,A_cumulative_fits_after=1538+count(),automatic_retry=False));raise


if __name__=='__main__':main()
