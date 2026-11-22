"""Recover a public asset source at a newly pinned revision, retaining all original byte hashes."""
import argparse
import hashlib
from pathlib import Path
import shutil
import time
import requests
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_sentence import MODEL,REVISION,FILES
from superagent.ab_vision import WEIGHT_BYTES,WEIGHT_SHA256

DOWNLOAD_REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--environment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();records=[]
    save(root/'protocol.json',dict(batch='A82',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A82_encoder_revision_recovery_plan.md'),original_declared_revision=REVISION,
        download_revision=DOWNLOAD_REVISION,all_original_file_hashes_preserved=True,process_proxy='http://127.0.0.1:7897',
        original_environment=str(a.environment),actual_classifier_fits=0,real_api_requests=0,automatic_retry=False))
    try:
        shutil.copyfile(a.environment/'environment.json',root/'environment.json')
        todo=[(root/'minilm'/name,f'https://huggingface.co/{MODEL}/resolve/{DOWNLOAD_REVISION}/{name}',size,digest) for name,(size,digest) in FILES.items()]
        todo.append((root/'mobilenet_v3_small-047dcff4.pth','https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',WEIGHT_BYTES,WEIGHT_SHA256))
        with requests.Session() as session:
            session.proxies.update(http='http://127.0.0.1:7897',https='http://127.0.0.1:7897')
            for path,url,size,digest in todo:
                path.parent.mkdir(exist_ok=True,parents=True);partial=path.with_name(path.name+'.partial');checksum=hashlib.sha256();count=0
                with session.get(url,timeout=60,stream=True) as response:
                    response.raise_for_status()
                    with partial.open('xb') as stream:
                        for data in response.iter_content(1024*1024):
                            stream.write(data);checksum.update(data);count+=len(data)
                            if count>size:raise ValueError('Pinned file size exceeded: '+path.name)
                if count!=size or checksum.hexdigest()!=digest:raise ValueError('Original B byte identity differs: '+path.name)
                partial.rename(path);records.append(dict(file=str(path),source=url,bytes=count,sha256=digest,verified=True))
                save(root/'assets.json',records)
        save(root/'completion.json',dict(status='ready',verified_assets=len(records),original_B_byte_identity=True,
            environment_python=read(root/'environment.json')['executable'],actual_classifier_fits=0,real_api_requests=0,
            download_revision=DOWNLOAD_REVISION,elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_api_requests=0,
            automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise
