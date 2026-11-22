"""Register a real-photo development cohort and run two autonomous visual-resource arms."""
import argparse
import csv
import io
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import tarfile
from unittest.mock import patch
import zipfile

import joblib
import numpy as np

from scripts import run_b1_ab_acceptance as fixture
from scripts import run_b1_05_pair_pilot as pilot
from superagent.ab_run import sources
from superagent.ab_vision import NAME, WEIGHT_ENV, WEIGHT_SHA256, weights_file
from superagent.data import load_development_task, load_prediction_features


SOURCE_HASH = '4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c'
CLASSES = ('daisy','dandelion','roses','sunflowers','tulips')
ARMS = ('existing', 'vision')
CONTRACT = dict(width=64, height=64, mode='RGB', resize='stretch_bilinear')


def rank(salt, name):
    return fixture.sha(f'{salt}/{name}'.encode())


def cohort(license_text):
    pools = {c:[] for c in CLASSES}
    for line in license_text.splitlines():
        if '.jpg CC-BY ' not in line:
            continue
        relative, credit = line.split(' CC-BY ', 1)
        match = re.search(r'https?://(?:www\.)?flickr\.com/photos/([^/\s]+)',credit)
        photo = relative.split('/')[-1].split('_')[0]
        if not match or not photo.isdigit():
            raise ValueError('A source photograph lacks a parsed creator or photo ID')
        pools[relative.split('/')[0]].append((relative,match[1].casefold(),photo,credit))
    if sum(map(len,pools.values())) != 3670:
        raise ValueError('Source credit list is incomplete')
    queues = {c:iter(sorted(rows,key=lambda r:rank('BV01-photo',r[0]))) for c,rows in pools.items()}
    used_authors, used_photos, picks = set(), set(), []
    for i in range(200):
        for category in CLASSES:
            item = next((r for r in queues[category] if r[1] not in used_authors and r[2] not in used_photos), None)
            if item is None:
                raise ValueError('Source did not support the registered creator-disjoint sample')
            name, author, photo, credit = item
            used_authors.add(author)
            used_photos.add(photo)
            picks.append(dict(source=name, author=author, photo=photo, credit=credit, label='c'+str(CLASSES.index(category)),
                              split='train' if i < 160 else 'test'))
    return sorted(picks, key=lambda r:rank('BV01-order',r['source']))


def image_packet(path, split, rows, images):
    manifest = dict(schema_version=1, objective='classification', modality='image', metric='accuracy',
                    columns=[dict(name='image',type='image_path')], target='label', image_preprocessing=CONTRACT,
                    splits={split:dict(format='csv',path='observations.csv')})
    table = io.StringIO(newline='')
    writer = csv.writer(table, lineterminator='\n')
    writer.writerow(['image','label'] if split=='train' else ['image'])
    credits = ['Photographs licensed CC BY 2.0. https://creativecommons.org/licenses/by/2.0/',
               'TensorFlow Flowers. Individual attribution and source for each photograph follow.']
    with zipfile.ZipFile(path, 'x') as out:
        def put(name, value):
            info = zipfile.ZipInfo(name,(1980,1,1,0,0,0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            out.writestr(info, value)
        for i,r in enumerate(rows):
            short = f'images/i{i:04d}.jpg'
            writer.writerow([short,r['label']] if split=='train' else [short])
            put(short, images[r['source']])
            credits.append(short+' CC-BY '+r['credit'])
        put('observations.csv',table.getvalue())
        put('manifest.json',json.dumps(manifest,sort_keys=True))
        put('CREDITS.txt','\n'.join(credits)+'\n')


def prepare(root, source, checkpoint):
    if source.stat().st_size != 228813984 or fixture.sha(source.read_bytes()) != SOURCE_HASH:
        raise ValueError('Unexpected photograph source archive')
    common = root/'data'
    common.mkdir()
    with tarfile.open(source,'r:gz') as tar:
        license_text = tar.extractfile('flower_photos/LICENSE.txt').read().decode()
    selection = cohort(license_text)
    names = {r['source'] for r in selection}
    images = {}
    with tarfile.open(source, 'r|gz') as tar:
        for member in tar:
            relative = member.name.removeprefix('flower_photos/')
            if relative in names:
                if not member.isfile() or relative in images:
                    raise ValueError('Duplicate or invalid source member')
                images[relative] = tar.extractfile(member).read()
    if images.keys() != names:
        raise ValueError('Missing source photograph')
    for mode, dest in [('train','train.zip'),('test','features.zip')]:
        image_packet(common/dest, mode, [r for r in selection if r['split']==mode],images)
    task = load_development_task(common/'train.zip')
    held = [r for r in selection if r['split']=='test']
    query = load_prediction_features(common/'features.zip', task.kind, task.spec)
    if len({fixture.sha(img.tobytes()) for img in [*task.X_train,*query]}) != len(selection):
        raise ValueError('A decoded pixel image repeats')
    fit = np.asarray([i for i,r in enumerate(selection) if r['split']=='train'])
    out = np.asarray([i for i,r in enumerate(selection) if r['split']=='test'])
    labels = np.asarray([r['label'] for r in held])
    if (task.X_val is not None or task.y_val is not None or len(fit)!=800 or len(out)!=200 or
            task.y_train.tolist() != [selection[int(i)]['label'] for i in fit] or query.shape!=(200,64,64,3)):
        raise ValueError('Real image cohort or its assignment changed')
    joblib.dump(dict(X=query,y=labels,row_ids=out.tolist()),common/'holdout.joblib')
    ids_sha = pilot.identities_file(common/'source_ids.json.gz',[r['source'] for r in selection])
    meta = dict(source_sha256=SOURCE_HASH, license_sha256=fixture.sha(license_text.encode()), source_photographs=3670,
                unique_flickr_accounts=len({r['author'] for r in selection}), unique_photo_ids=len({r['photo'] for r in selection}),
                unique_decoded_images=len(selection), source_id_order_sha256=ids_sha, classes=list(CLASSES))
    entries = {}
    for arm in ARMS:
        p = root/arm
        p.mkdir()
        for file in ('train.zip','features.zip','holdout.joblib'):
            os.link(common/file, p/file)
        with patch.dict(os.environ, {WEIGHT_ENV:str(checkpoint) if arm=='vision' else ''}):
            if arm=='vision' and weights_file() is None:
                raise ValueError('Expected a verified local frozen visual encoder')
            entries[arm] = pilot.register_prepared_task(p,meta,'not_applicable',1000,fit,out,
                dict(grouping='one published Flickr account per photo'),dict(shared_accounts=0,shared_decoded_images=0))
    a,b = (entries[n] for n in ARMS)
    if ([*a['candidate_ids'],NAME] != b['candidate_ids'] or
            any(a[k] != b[k] for k in ('splits','row_ids_sha256','holdout_sha256','files'))):
        raise ValueError('Arms differ on another candidate or on evaluation inputs')
    return dict(batch='BV-01',source_sha256=fixture.digest(sources()),driver_sha256=fixture.sha(Path(__file__).read_bytes()),
        plan_sha256=fixture.sha((fixture.ROOT/'docs/batches/BV_01_photo_plan.md').read_bytes()),tasks=entries,
        runtime=pilot.environment(),vision_runtime={k:version(k) for k in ('torch','torchvision','Pillow')},
        weights_sha256=WEIGHT_SHA256, fit_limit_per_path=11, fit_limit_batch=22, wall_seconds=pilot.WALL,
        trial_seconds=pilot.TRIAL, inner_seed=pilot.INNER_SEED, model_api_limit=0, order=list(ARMS))


def run(args):
    root, source, weight = (p.resolve() for p in (args.output,args.source,args.weights))
    root.mkdir(parents=True,exist_ok=False)
    fixture.offline(root)
    os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    paths = {}
    try:
        protocol = prepare(root,source,weight)
        fixture.write(root/'protocol.json',protocol)
        if args.prepare_only:
            fixture.write(root/'completion.json',dict(status='prepared',fits=0,model_api_requests=0))
            return
        if not args.reference or protocol != fixture.read(args.reference/'protocol.json'):
            raise ValueError('Full run requires exact match with the pre-training protocol')
        for arm in ARMS:
            with patch.dict(os.environ,{WEIGHT_ENV:str(weight) if arm=='vision' else ''}):
                start = len(fixture.trace(root))
                result = pilot.train_one(root/arm,protocol['tasks'][arm],protocol['source_sha256'])
                if len(fixture.trace(root)) - start != result['fits']:
                    raise ValueError('Independent physical fit count disagrees')
                paths[arm] = result
                fixture.write(root/'results.json',dict(paths=paths))
        fixture.write(root/'all_models_frozen.json',{n:p['model_sha256'] for n,p in paths.items()})
        for arm in ARMS:
            with patch.dict(os.environ,{WEIGHT_ENV:str(weight) if arm=='vision' else ''}):
                paths[arm]['delivery'] = pilot.score_one(root/arm,protocol['tasks'][arm])
        predictions = []
        for arm in ARMS:
            with (root/arm/'pred_0/rows.csv').open(newline='',encoding='utf-8') as file:
                predictions.append(np.array([r['prediction'] for r in csv.DictReader(file)]))
        truth = joblib.load(root/'data/holdout.joblib')['y']
        correct = [p==truth for p in predictions]
        if any(int(p.sum()) != paths[arm]['delivery']['correct'] for arm,p in zip(ARMS,correct,strict=True)):
            raise ValueError('Paired correctness differs from independent export verification')
        base,extended = correct
        pair = dict(rows=len(truth), newly_correct=int(np.sum(extended & ~base)), newly_wrong=int(np.sum(base & ~extended)),
                    both_correct=int(np.sum(base & extended)), both_wrong=int(np.sum(~base & ~extended)))
        fixture.write(root/'results.json',dict(paths=paths,paired=pair))
        fixture.write(root/'completion.json',dict(status='passed',paths=2,fits=len(fixture.trace(root)),model_api_requests=0,
            api_rmb=0, all_models_frozen_before_scoring=True, course_validation_or_test_accessed=False, h1_h2_accessed=False))
    except Exception as error:
        fixture.write(root/'completion.json',dict(status='failed',reason=str(error),paths_returned=list(paths),
                      physical_fits_observed=len(fixture.trace(root)),model_api_requests=0,automatic_retry=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','weights','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    run(parser.parse_args())
