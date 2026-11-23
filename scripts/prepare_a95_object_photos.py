"""Prepare non-identity object classification images using B's published flower cohort, zero model calls."""
import argparse
import hashlib
import os
from pathlib import Path
import tarfile
import time
import requests
import joblib
import numpy as np
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha
from scripts.run_b_photo_pilot import SOURCE_HASH,cohort,image_packet
from superagent.data import load_training_task,load_prediction_features

ROOT=Path(__file__).resolve().parents[1]
URL='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    save(root/'protocol.json',dict(batch='A95',runner_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'docs/batches/A95_object_photo_preparation_plan.md'),
        shared_cohort_code_sha256=sha(ROOT/'scripts/run_b_photo_pilot.py'),source=URL,expected_bytes=228813984,expected_sha256=SOURCE_HASH,
        actual_classifier_fits=0,real_api_requests=0,automatic_retry=False,task='generic_flower_category_not_person_identity'))
    try:
        partial=root/'flower_photos.tgz.partial';count=0;digest=hashlib.sha256()
        with requests.Session() as s:
            s.proxies.update(http='http://127.0.0.1:7897',https='http://127.0.0.1:7897')
            with s.get(URL,stream=True,timeout=60) as response:
                response.raise_for_status()
                with partial.open('xb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        stream.write(chunk);digest.update(chunk);count+=len(chunk)
                        if count>228813984:raise ValueError('Source archive exceeds pinned size')
        if count!=228813984 or digest.hexdigest()!=SOURCE_HASH:raise ValueError('Official source byte identity differs')
        source=root/'flower_photos.tgz';partial.rename(source)
        with tarfile.open(source,'r:gz') as archive:license_text=archive.extractfile('flower_photos/LICENSE.txt').read().decode()
        selected=cohort(license_text);wanted={r['source'] for r in selected};images={}
        with tarfile.open(source,'r|gz') as archive:
            for member in archive:
                relative=member.name.removeprefix('flower_photos/')
                if relative in wanted:
                    if not member.isfile() or relative in images:raise ValueError('Invalid or duplicate photo member')
                    images[relative]=archive.extractfile(member).read()
        if set(images)!=wanted:raise ValueError('Missing selected photos')
        train=[r for r in selected if r['split']=='train'];held=[r for r in selected if r['split']=='test']
        image_packet(root/'train.zip','train',train,images);image_packet(root/'features.zip','test',held,images)
        task=load_training_task(root/'train.zip');query=load_prediction_features(root/'features.zip',task.kind,task.spec)
        if len(task.y_train)!=800 or query.shape!=(200,64,64,3):raise ValueError('Cohort dimensions changed')
        if len({hashlib.sha256(im.tobytes()).hexdigest() for im in [*task.X_train,*query]})!=1000:raise ValueError('Repeated decoded photo')
        joblib.dump(dict(X=query,y=np.asarray([r['label'] for r in held]),row_ids=list(range(len(held)))),root/'holdout.joblib')
        save(root/'registration.json',dict(training_rows=800,heldout_rows=200,classes=5,training_package_sha256=sha(root/'train.zip'),
            feature_package_sha256=sha(root/'features.zip'),holdout_sha256=sha(root/'holdout.joblib'),unique_creators=len({r['author'] for r in selected}),
            unique_photo_ids=len({r['photo'] for r in selected}),source_sha256=SOURCE_HASH,author_disjoint=True,
            evaluation_not_performed=True,task_kind=task.kind))
        save(root/'completion.json',dict(status='prepared',actual_classifier_fits=0,real_api_requests=0,elapsed_seconds=time.monotonic()-start,formal_test_or_H_opened=False))
    except Exception as e:
        save(root/'completion.json',dict(status='stopped',error=str(e),actual_classifier_fits=0,real_api_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise
