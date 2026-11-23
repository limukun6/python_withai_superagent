"""Acquire the pinned public archive once, then rebuild only B's registered H2 training images."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import requests
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]
URL='https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1'
SIZE=94658721
EXPECTED='b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('after','assets','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    references=[ROOT/'scripts/prepare_b_h2_training_inputs.py',ROOT/'scripts/seal_b_h2.py',
        ROOT/'research/bh02_image_seal/source_ids.json.gz',ROOT/'research/bh_03_training_inputs/results.json']
    save(a.output/'protocol.json',dict(batch='A105',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A105_H2_training_inputs_plan.md'),
        reference_sha256={str(p.relative_to(ROOT)):sha(p) for p in references},
        source_url=URL,expected_bytes=SIZE,expected_sha256=EXPECTED,automatic_retry=False,
        actual_classifier_fits=0,real_API_requests=0,holdout_image_access_allowed=False))
    try:
        completion=read(a.after/'completion.json')
        if completion['status']!='completed':raise ValueError('Wait until SMS C1 completed; no concurrent preparation')
        partial=a.output/'EuroSAT_RGB.zip.partial';download_start=time.monotonic();count=0;digest=hashlib.sha256()
        with requests.Session() as session:
            session.proxies.update(http='http://127.0.0.1:7897',https='http://127.0.0.1:7897')
            with session.get(URL,stream=True,timeout=60) as response:
                response.raise_for_status()
                with partial.open('xb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        if time.monotonic()-download_start>300:raise TimeoutError('Public download window elapsed; no retry')
                        count+=len(chunk)
                        if count>SIZE:raise ValueError('Public archive exceeds pinned byte count')
                        digest.update(chunk);stream.write(chunk)
        if count!=SIZE or digest.hexdigest()!=EXPECTED:raise ValueError('Public archive byte identity differs')
        archive=a.output/'EuroSAT_RGB.zip';partial.rename(archive)
        print(dict(source_verified=True,bytes=count,download_seconds=time.monotonic()-download_start),flush=True)
        from scripts.seal_b_h2 import source_hashes
        if source_hashes(archive)!=EXPECTED:raise ValueError('Original author checksum differs')
        from scripts.prepare_b_h2_training_inputs import prepare
        prepare(archive,a.output/'training')
        expected=read(ROOT/'research/bh_03_training_inputs/results.json')
        actual=read(a.output/'training/results.json')
        if set(actual)!=set(expected):raise ValueError('Outer registrations differ')
        for seed,row in actual.items():
            for key in ('train_archive_sha256','inner_manifest_sha256'):
                if row[key]!=expected[seed][key]:raise ValueError('B original registered input identity differs: '+seed+'/'+key)
        os.environ['SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS']=str(a.assets/'mobilenet_v3_small-047dcff4.pth')
        from superagent.ab_run import registered_capabilities,sources
        from superagent.contracts import TaskSpec
        from superagent.data import load_training_task
        menus={}
        for seed in actual:
            task=load_training_task(a.output/'training'/seed/'train.zip');spec=TaskSpec.from_task(task).record()
            caps=registered_capabilities(spec,len(task.y_train),'full')
            if 'image_mobilenet_v3_small_linear' not in caps or len(caps)!=9:raise ValueError('Prepared image contract does not offer expected visual route')
            menus[seed]=dict(task_spec=spec,capabilities=caps,rows=len(task.y_train))
        save(a.output/'results.json',dict(registered_inputs=actual,menus=menus,source_files=sources(),
            original_source_sha256=EXPECTED,outer_holdout_image_members_opened=0,
            scope='Same pre-registered H2 training-region images only; no model performance or final strategy selection'))
        save(a.output/'completion.json',dict(status='prepared',training_photos_verified=5000,
            inner_partitions_verified=6,training_archives_match_B=True,inner_manifests_match_B=True,
            actual_classifier_fits=0,real_API_requests=0,model_predictions=0,holdout_image_members_opened=0,
            formal_H_strategy_frozen=False,elapsed_seconds=time.monotonic()-start))
        print('Prepared two identical registered training archives, six internal partitions and full9 image routes',flush=True)
    except Exception as e:
        save(a.output/'completion.json',dict(status='stopped',error=str(e),automatic_retry=False,
            actual_classifier_fits=0,real_API_requests=0,elapsed_seconds=time.monotonic()-start));raise
