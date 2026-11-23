"""Verify published C1/D3 metadata in raw Git bytes; no model execution or API."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha

ROOT=Path(__file__).resolve().parents[1]
REV='db589a3'
PREFIXES=('research/b_c1_validation_1123/','research/b_integrated_offline_1123/')
EXACT={'research/b_c1_joint_delivery_1123/budget_handoff_receipt.json',
       'research/a70_independent_review/blinded_reviews.json','scripts/verify_b_c1_d3_handoff.py',
       'docs/batches/B_C1_D3_handoff_20261123_zh.md'}


def blob(path):
    return subprocess.check_output(['git','show',REV+':'+path],cwd=ROOT)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    commit=subprocess.check_output(['git','rev-parse',REV],cwd=ROOT,text=True).strip()
    save(a.output/'protocol.json',dict(batch='A117',provider_commit=commit,runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A117_B_C1_D3_receipt_plan.md'),actual_classifier_fits=0,real_API_requests=0))
    try:
        published=a.output/'published'
        paths=subprocess.check_output(['git','ls-tree','-r','--name-only',REV],cwd=ROOT,text=True).splitlines()
        exported={}
        for name in paths:
            if name in EXACT or name.startswith(PREFIXES):
                path=published/name
                if not path.resolve().is_relative_to(published.resolve()): raise ValueError('Unsafe artifact path')
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(blob(name));exported[name]=sha(path)
        result=subprocess.run([sys.executable,str(published/'scripts/verify_b_c1_d3_handoff.py')],cwd=published,
                              capture_output=True,text=True,encoding='utf-8',timeout=60)
        (a.output/'verification.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode: raise ValueError('Published receipt verifier failed; see log')
        verified=json.loads(result.stdout)
        original=read(ROOT/'research/a70_independent_review/blinded_reviews.json')
        their_original=read(published/'research/a70_independent_review/blinded_reviews.json')
        if original!=their_original: raise ValueError('B annotated a different original answer packet')
        c1=published/'research/b_c1_validation_1123'
        expected=read(ROOT/'research/a89_coupon_C1/protocol.json')['source_files']
        protocols={task:read(c1/task/'protocol.json') for task in ('preference','bean')}
        identities={task:{key:(value==expected) for key,value in row.items() if isinstance(value,dict) and 'ab_run.py' in value}
                    for task,row in protocols.items()}
        if any(not checks or not all(checks.values()) for checks in identities.values()):
            raise ValueError('C1 source identity does not match A original source')
        receipt=read(published/'research/b_c1_joint_delivery_1123/budget_handoff_receipt.json')
        save(a.output/'results.json',dict(provider_commit=commit,artifact_hashes=exported,verified=verified,
             C1_source_matches_A=identities,original_24_replies_match_A=True,
             resources=read(c1/'resource_summary.json'),project_after_reported_by_B=receipt['project_after'],
             actual_return_SQLite_received=False,paid_queue_remains_paused=True,
             case_and_annotation_semantic_adjudication_completed=False,
             limitation='Published receipts recomputed; no B model or raw prediction files locally rerun'))
        save(a.output/'completion.json',dict(status='receipts_verified',actual_classifier_fits=0,real_API_requests=0,
             A_cumulative_fits_after=1505,elapsed_seconds=time.monotonic()-start))
        print(result.stdout,flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0))
        raise


if __name__=='__main__': main()
