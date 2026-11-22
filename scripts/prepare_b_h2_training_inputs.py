"""Materialize only the pre-registered H2 training-region rows. No holdout image access or models."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import joblib
import numpy as np
from PIL import Image

from scripts import run_b1_ab_acceptance as fixture
from scripts.materialize_b_c1_rows import stored_zip
from scripts.seal_b_h2 import INNER, OUTER, source_hashes
from superagent import ab_run
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from tools.ab_registered_rows import content_digest, registered_partitioner


REF = fixture.ROOT/'research/bh02_image_seal'


def packet(src, members, names, path):
    if len(names)!=2500 or len(set(names))!=len(names):
        raise ValueError('Unregistered training set')
    table = io.StringIO(newline='')
    writer = csv.writer(table,lineterminator='\n')
    writer.writerow(['image','label'])
    entries, pixels = [], []
    for i,name in enumerate(names):
        label = name.rsplit('_',1)[0]
        short = f'photos/{i:04d}.jpg'
        image = src.read(members[name])
        with Image.open(io.BytesIO(image)) as photo:
            if photo.size!=(64,64) or photo.mode!='RGB' or photo.format!='JPEG':
                raise ValueError('Registered source image changed')
            pixels.append(fixture.sha(np.asarray(photo).tobytes()))
        entries.append((short,image))
        writer.writerow([short,label])
    manifest = dict(schema_version=1,modality='image',objective='classification',metric='accuracy',
                    image_preprocessing=dict(height=64,width=64,mode='RGB',resize='stretch_bilinear'),
                    target='label',columns=[dict(name='image',type='image_path')],splits=dict(train=dict(format='csv',path='observations.csv')))
    entries.extend([('manifest.json',json.dumps(manifest,sort_keys=True)),('observations.csv',table.getvalue()),
                    ('CREDITS.txt','EuroSAT by P. Helber, B. Bischke, A. Dengel and D. Borth. Contains modified Copernicus Sentinel data. https://zenodo.org/records/7711810 and https://github.com/phelber/EuroSAT#license\n')])
    stored_zip(path,entries)
    return pixels


def prepare_one(seed, archive, members, records, expected, output):
    rows=records[str(seed)]
    names=rows['outer_train_source_ids']
    directory=output/str(seed)
    directory.mkdir()
    file=directory/'train.zip'
    original=packet(archive,members,names,file)
    task=load_development_task(file)
    labels=[n.rsplit('_',1)[0] for n in names]
    if (task.kind!='image' or task.y_val is not None or task.X_val is not None or task.y_train.tolist()!=labels
            or [fixture.sha(img.tobytes()) for img in task.X_train]!=original):
        raise ValueError('Official JPEG import changed the registered pixels, order, or training labels')
    position={v:i for i,v in enumerate(names)}
    splits=[]
    for step,num in enumerate(INNER):
        split=('primary','confirmation_1','confirmation_2')[step]
        reg=rows['inner'][str(num)]
        if fixture.digest(reg)!=expected['partitions'][str(seed)]['inner_row_id_sha256'][str(num)]:
            raise ValueError('An inner registered partition differs')
        train=[position[x] for x in reg['train_source_ids']]
        held=[position[x] for x in reg['search_source_ids']]
        splits.append(dict(id=split,seed=num,train_indices=train,search_indices=held,
                           row_ids_sha256=fingerprint(held),label_sha256=fingerprint(task.y_train[held])))
    signed=directory/'inner_splits.json'
    fixture.write(signed,dict(version=1,kind=task.kind,training_content_sha256=content_digest(task.X_train,task.y_train),
                  grouping='exact_rows',splits=splits))
    signed_hash=fixture.sha(signed.read_bytes())
    selector=registered_partitioner(task,signed,signed_hash)
    engine=directory/'worker_inputs'
    engine.mkdir()
    with patch.object(ab_run,'grouped_partition',selector):
        trusted,allowed=ab_run.freeze_partitions(task,engine,INNER[0],MetricSpec.for_task(task))
    checks={}
    for part in splits:
        key=part['id']
        payload=engine/(key+'.joblib')
        worker=joblib.load(payload)
        a,b=part['train_indices'],part['search_indices']
        if (trusted[key]['row_ids']!=b or allowed[key]['evaluation_payload_sha256']!=fixture.sha(payload.read_bytes())
                or allowed[key]['row_ids_sha256']!=part['row_ids_sha256'] or allowed[key]['label_sha256']!=part['label_sha256']
                or not np.array_equal(task.X_train[a],worker['X_train']) or not np.array_equal(task.X_train[b],worker['X_search'])
                or not np.array_equal(task.y_train[a],worker['y_train']) or not np.array_equal(task.y_train[b],worker['y_search'])):
            raise ValueError('Worker contents differ from the original pre-registered training region split')
        checks[key]=dict(train_rows=len(a),internal_eval_rows=len(b),**allowed[key])
    return dict(train_rows=len(names),train_archive_sha256=fixture.sha(file.read_bytes()),
                inner_manifest_sha256=signed_hash,splits=checks), original


def prepare(source, output):
    output.mkdir(parents=True,exist_ok=False)
    fixture.offline(output)
    fixture.write(output/'protocol.json',dict(batch='BH-03',seal_protocol_sha256=fixture.sha((REF/'protocol.json').read_bytes()),
        plan_sha256=fixture.sha((fixture.ROOT/'docs/batches/BH_03_inputs_plan.md').read_bytes()),
        driver_sha256=fixture.sha(Path(__file__).read_bytes()),production_source=fixture.digest(ab_run.sources()),
        evaluated_models_allowed=0,holdout_images_allowed=0,api_allowed=0))
    try:
        ref=fixture.read(REF/'results.json')
        raw=gzip.decompress((REF/'source_ids.json.gz').read_bytes())
        if source_hashes(source)!=ref['archive_sha256'] or fixture.sha(raw)!=ref['source_ids_sha256']:
            raise ValueError('Source archive or the prior H2 seal differs')
        records=json.loads(raw)
        train={n for seed in OUTER for n in records[str(seed)]['outer_train_source_ids']}
        held={n for seed in OUTER for n in records[str(seed)]['outer_holdout_source_ids']}
        if len(train)!=5000 or len(held)!=1000 or train & held:
            raise ValueError('Training and held-region source lists do not retain the preregistered boundaries')
        runs, seen = {}, set()
        with zipfile.ZipFile(source) as archive:
            # Only training IDs are resolvable by packet(). The physical source hash above reads raw archive bytes.
            members={Path(m.filename).name:m for m in archive.infolist() if not m.filename.startswith('__MACOSX/') and Path(m.filename).name in train}
            if set(members)!=train:
                raise ValueError('Missing registered training JPEG')
            for seed in OUTER:
                result,pixels=prepare_one(seed,archive,members,records,ref,output)
                runs[str(seed)]=result
                seen.update(pixels)
        if len(seen)!=5000:
            raise ValueError('Duplicated pixels in sealed training sets')
        fixture.write(output/'results.json',runs)
        fixture.write(output/'completion.json',dict(status='training_inputs_ready',training_photos_verified=5000,partitions_verified=6,
            sealed_holdout_image_members_opened=0,spatial_test_members_opened=0,models_used=0,predictions=0,api_requests=0))
    except Exception as error:
        fixture.write(output/'completion.json',dict(status='failed',reason=str(error),models_used=0,api_requests=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    prepare(a.source,a.output)
