"""Verify the merge preserves exact tested native source and the isolated old C1 source."""
import argparse
from pathlib import Path
import subprocess
import time
from scripts.a_batch_common import hashes, save
from scripts.run_a10_real_task_pilot import read, sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-C1',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    save(a.output/'protocol.json',dict(batch='A112-postmerge',runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A112_A114_native_and_H1_queue_plan.md'),
         actual_classifier_fits=0,real_API_requests=0))
    try:
        expected=read(ROOT/'research/a113_native_delivery/protocol.json')['source_files']
        if hashes(ROOT)!=expected: raise ValueError('Merged source differs from real trained/exported probe')
        if hashes(ROOT)!=read(ROOT/'research/a115_H_entry/protocol.json')['source_files']:
            raise ValueError('Merged source differs from H entry check')
        old={p.relative_to(a.old_C1/'superagent').as_posix():sha(p) for p in sorted((a.old_C1/'superagent').rglob('*.py'))}
        if old!=read(ROOT/'research/a89_coupon_C1/protocol.json')['source_files']:
            raise ValueError('Old C1 source checkout changed')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        save(a.output/'results.json',dict(merge_commit=commit,source_identical_to_real_native_delivery=True,
             source_identical_to_H_entry=True,isolated_old_C1_unchanged=True))
        save(a.output/'completion.json',dict(status='verified',actual_classifier_fits=0,real_API_requests=0,
             A_cumulative_fits_after=1505,elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0))
        raise


if __name__=='__main__': main()
