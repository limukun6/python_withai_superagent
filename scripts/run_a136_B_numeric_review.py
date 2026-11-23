"""Validate pinned B numeric/weak-label receipts without training, API calls or weights."""
import argparse
import io
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tarfile
import time

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a133_routing_verification import GUARD

ROOT=Path(__file__).resolve().parents[1]
PROVIDER='5bd6b145e63ca59af005aab6cf1e9cd24ff21931'
PREFIX='releases/bean-task2-2026-11-23'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();source=hashes(ROOT)
    owner=read(ROOT/'research/a133_a135_summary/current_ledger_owner.json')
    ledger_before=sha(Path(owner['ledger_path']))
    save(out/'protocol.json',dict(batch='A136',provider=PROVIDER,prefix=PREFIX,source_files=source,
         runner_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'docs/batches/A136_numeric_and_weak_label_review_plan_20261124_zh.md'),
         fit_cap=0,model_api_cap=0,public_network_cap=0,starting_A_fits=1599,ledger_sha256=ledger_before))
    try:
        raw=subprocess.check_output(['git','archive','--format=tar',PROVIDER,PREFIX],cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            for m in archive:
                name=PurePosixPath(m.name);target=out.joinpath(*name.parts)
                if (name.is_absolute() or '..' in name.parts or '\\' in m.name or
                    not (m.isfile() or m.isdir()) or not target.resolve().is_relative_to(out)):
                    raise ValueError('Unsafe Git archive path')
                if m.isdir():target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(m).read())
        release=out/PREFIX;verifier=release/'verify.py'
        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','SUPERAGENT_OFFLINE':'1',
             'PYTHONIOENCODING':'utf-8','PYTHONPATH':str(ROOT)}
        with (out/'verification.log').open('x',encoding='utf-8') as stream:
            p=subprocess.run([sys.executable,'-c',GUARD,str(verifier)],cwd=release,env=env,
                             stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        if p.returncode:raise RuntimeError('B verifier failed; see verification.log')
        studies={n:read(release/'evidence'/n/'results.json') for n in
                 ('geometry','kernel','adaptive','fixed','transfer','generations','transitive')}
        if source!=hashes(ROOT) or ledger_before!=sha(Path(owner['ledger_path'])):
            raise ValueError('Source or unique ledger changed')
        save(out/'results.json',dict(verifier_sha256=sha(verifier),verified=True,
             verifier_report=(out/'verification.log').read_text(encoding='utf-8').strip(),studies=studies,
             checksum_manifest_sha256=sha(release/'SHA256SUMS'),runtime_source_archive_sha256=sha(release/'runtime-source.zip'),
             production_changed=False,models_retrained=False,evidence_scope='B sealed metadata recomputation; not A performance replication'))
        save(out/'completion.json',dict(status='passed',actual_classifier_fits=0,real_API_requests=0,
             A_cumulative_fits_after=1599,source_unchanged=True,ledger_unchanged=True,
             elapsed_seconds=time.monotonic()-started,protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
        print(json.dumps(dict(status='passed',studies=7,actual_classifier_fits=0,real_API_requests=0)))
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=0,real_API_requests=0,
             elapsed_seconds=time.monotonic()-started));raise


if __name__=='__main__':main()
