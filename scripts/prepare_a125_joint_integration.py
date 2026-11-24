"""Extract a pinned B release's existing A patches byte-for-byte; never apply or train automatically."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from scripts.a_batch_common import save,hashes

ROOT=Path(__file__).resolve().parents[1]
PROVIDER='5329503282245a5e8d60f42af46e0ae5f32bf187'
PREFIX='releases/alternative-encoders-2026-11-23/'
FILES=['docs/integration/A117_B_joint_encoder.patch','docs/integration/A117_B_single_conflict.patch',
       'SOURCE.json','docs/batches/B_joint_coverage_results_1123_zh.md','docs/batches/B_single_conflict_results_1123_zh.md']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    save(a.output/'protocol.json',dict(batch='A125-extract',provider=PROVIDER,source_files=hashes(ROOT),
         runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         plan_sha256=hashlib.sha256((ROOT/'docs/batches/A125_latest_B_integration_plan_20261123_zh.md').read_bytes()).hexdigest(),
         maximum_classifier_fits=0,real_API_requests=0))
    try:
        records=[]
        for name in FILES:
            raw=subprocess.check_output(['git','show',PROVIDER+':'+PREFIX+name],cwd=ROOT)
            target=a.output/Path(name).name;target.write_bytes(raw)
            record=dict(source=PREFIX+name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),local=str(target))
            if name.endswith('.patch'):
                record['changed_paths']=[line.split(' b/',1)[1] for line in raw.decode().splitlines() if line.startswith('diff --git ')]
            records.append(record)
        save(a.output/'results.json',records)
        save(a.output/'completion.json',dict(status='extracted_not_applied',actual_classifier_fits=0,real_API_requests=0,
             elapsed_seconds=time.monotonic()-start))
        print(json.dumps(records,ensure_ascii=False),flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0))
        raise


if __name__=='__main__':main()
