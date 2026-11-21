"""Check an optional pretrained visual representation on the shared, registered digit examples."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
from unittest.mock import patch
import zipfile

import joblib
import numpy as np
from sklearn.datasets import load_digits

from scripts import run_b1_ab_acceptance as rec, run_b1_05_pair_pilot as pilot
from superagent import ab_run
from superagent.ab_catalog import capabilities_for
from superagent.ab_vision import NAME, WEIGHT_ENV, WEIGHT_SHA256, weights_file
from superagent.contracts import TaskSpec
from superagent.data import TrainingTask, load_development_task, load_prediction_features
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import effective_grouping, grouped_partition
from tools.ab_registered_rows import content_digest, registered_partitioner


ROOT = rec.ROOT
A = ROOT/'research/a11_schedule_comparison'
ARMS = ('existing','vision')


def source():
    reference = next(t for t in rec.read(A/'protocol.json')['tasks'] if t['name']=='digits')
    data = load_digits()
    images = data.images.astype(np.float32)
    if rec.sha(images.tobytes()+data.target.tobytes())!=reference['task']['source']['content_sha256']:
        raise ValueError('Local digit data differs from the shared source')
    return reference,images,data.target


def rows_identity(fit,held,count):
    expected = rec.read(A/'digits_ab-v1/export_verification.json')['original_row_ids_sha256']
    if (fit!=sorted(set(fit)) or held!=sorted(set(held)) or sorted(fit+held)!=list(range(count)) or
            rec.sha(json.dumps(held).encode())!=expected):
        raise ValueError('Outer row IDs differ from the registered A11 study')
    return expected


def record_rows(out):
    _,images,labels = source()
    fit,held,_ = grouped_partition(TrainingTask('image',images,labels,{'metric':'accuracy'}),45101)
    train,held = fit.tolist(),held.tolist()
    rec.write(out/'rows.json',dict(train=train,held=held,held_sha256=rows_identity(train,held,len(images))))
    rec.write(out/'completion.json',dict(status='verified_rows',fit_rows=len(train),held_rows=len(held),fits=0,model_api_requests=0))


def packet(path, split, images, labels=None):
    from PIL import Image

    manifest = dict(schema_version=1,objective='classification',modality='image',metric='accuracy',target='label',
                    columns=[{'name':'image','type':'image_path'}],
                    image_preprocessing=dict(width=8,height=8,mode='L',resize='stretch_bilinear'),
                    splits={split:{'format':'csv','path':'rows.csv'}})
    buf=io.StringIO(newline='')
    writer=csv.writer(buf,lineterminator='\n')
    writer.writerow(['image','label'] if labels is not None else ['image'])
    with zipfile.ZipFile(path,'x') as archive:
        for i,image in enumerate(images):
            photo=f'images/d{i:04d}.png'
            stream=io.BytesIO()
            Image.fromarray(image).save(stream,format='PNG')
            pilot.write_member(archive,photo,stream.getvalue())
            writer.writerow([photo,str(labels[i])] if labels is not None else [photo])
        pilot.write_member(archive,'manifest.json',json.dumps(manifest,sort_keys=True).encode())
        pilot.write_member(archive,'rows.csv',buf.getvalue().encode())


def prepare(out,rows,weight):
    ref,images,labels=source()
    split=rec.read(rows)
    fit,held=split['train'],split['held']
    outer=rows_identity(fit,held,len(images))
    pixels=np.rint(images*(255/16)).astype(np.uint8)
    if not np.array_equal(images,np.rint(pixels.astype(float)*(16/255))):
        raise ValueError('Byte conversion lost a grey level')
    common=out/'shared'
    common.mkdir()
    packet(common/'train.zip','train',pixels[fit],labels[fit])
    packet(common/'features.zip','test',pixels[held])
    typed=load_development_task(common/'train.zip')
    query=load_prediction_features(common/'features.zip',typed.kind,typed.spec)
    if (typed.X_val is not None or typed.y_val is not None or not np.array_equal(typed.X_train,pixels[fit]) or
            not np.array_equal(query,pixels[held]) or not np.array_equal(typed.y_train,labels[fit].astype(str))):
        raise ValueError('Formal decoder changed rows, labels or source pixels')
    joblib.dump(dict(X=query,y=labels[held].astype(str),row_ids=held),common/'holdout.joblib')
    inner=[]
    for name,record in ref['frozen_partitions'].items():
        search=record['search_indices']
        if (fingerprint(search)!=record['row_ids_sha256'] or
                fingerprint(labels[fit][search])!=record['label_sha256']):
            raise ValueError('Registered inner source IDs or numeric label identities differ')
        inner.append(dict(id=name,seed=record['seed'],train_indices=record['train_indices'],search_indices=search,
                          row_ids_sha256=record['row_ids_sha256'],label_sha256=fingerprint(typed.y_train[search])))
    manifest=common/'inner.json'
    rec.write(manifest,dict(version=1,kind=typed.kind,grouping=effective_grouping(typed.kind,'shared_text'),
                           training_content_sha256=content_digest(typed.X_train,typed.y_train),splits=inner))
    inner_hash=ab_run.sha(manifest)
    select=registered_partitioner(typed,manifest,inner_hash)
    registered=common/'registered'
    registered.mkdir()
    with patch.object(ab_run,'grouped_partition',select):
        _,parts=ab_run.freeze_partitions(typed,registered,707,MetricSpec.for_task(typed))
    files={n:ab_run.sha(common/n) for n in ('train.zip','features.zip')}
    tasks,catalogs={},{}
    for name in ARMS:
        with patch.dict(os.environ,{WEIGHT_ENV:str(weight) if name=='vision' else ''}):
            if name=='vision' and weights_file() is None:
                raise ValueError('Verified offline visual checkpoint required')
            caps=capabilities_for(TaskSpec.from_task(typed).record(),train_rows=len(fit))
        folder=out/name
        folder.mkdir()
        for f in ('train.zip','features.zip','holdout.joblib'):
            os.link(common/f,folder/f)
        (folder/'registered').mkdir()
        for f in registered.glob('*_split.json'):
            os.link(f,folder/'registered'/f.name)
        tasks[name]=dict(name=name,semantics='not_applicable',files=files,splits=parts,
                         holdout_sha256=ab_run.sha(common/'holdout.joblib'),outer_rows=len(held),
                         candidate_ids=list(caps),catalog_sha256=rec.digest(caps))
        catalogs[name]=caps
    if list(catalogs['existing'])!=list(ref['capabilities']) or list(catalogs['vision'])!=[*catalogs['existing'],NAME]:
        raise ValueError('Comparison changed more than the registered visual option')
    if any(catalogs['vision'][n]!=v for n,v in catalogs['existing'].items()):
        raise ValueError('Shared method metadata changed between arms')
    return dict(batch='BV-03',source=rec.digest(ab_run.sources()),dataset_sha256=ref['task']['source']['content_sha256'],
                outer_row_sha256=outer,training_rows=len(fit),held_rows=len(held),inner_manifest=inner_hash,
                driver=ab_run.sha(Path(__file__)),plan=ab_run.sha(ROOT/'docs/batches/BV_03_gray_plan.md'),
                runtime=pilot.environment(),weights_sha256=WEIGHT_SHA256,tasks=tasks,fit_limit=22,model_api_limit=0)


def run(args):
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    try:
        if args.mode=='rows':
            record_rows(out)
            return
        weight=args.weights.resolve()
        protocol=prepare(out,args.rows,weight)
        rec.write(out/'protocol.json',protocol)
        if args.mode=='prepare':
            rec.write(out/'completion.json',dict(status='registered',fits=0,model_api_requests=0))
            return
        if rec.read(args.reference/'protocol.json')!=protocol:
            raise ValueError('Research inputs differ from preregistration')
        results={}
        launcher=('-m','tools.ab_registered_run','--split-manifest',out/'shared/inner.json',
                  '--split-sha256',protocol['inner_manifest'],'--','run')
        for name in ARMS:
            with patch.dict(os.environ,{WEIGHT_ENV:str(weight) if name=='vision' else ''}):
                before=len(rec.trace(out))
                result=pilot.train_one(out/name,protocol['tasks'][name],protocol['source'],launcher=launcher)
                if len(rec.trace(out))-before!=result['fits']:
                    raise ValueError('Physical fit accounting differs')
                history=rec.read(out/name/'run/history.json')
                result['primary_scores']={r['candidate']:r['search_score'] for r in history if r['split_id']=='primary'}
                results[name]=result
        rec.write(out/'all_models_frozen.json',{name:r['model_sha256'] for name,r in results.items()})
        preds=[]
        for name in ARMS:
            with patch.dict(os.environ,{WEIGHT_ENV:str(weight) if name=='vision' else ''}):
                results[name]['delivery']=pilot.score_one(out/name,protocol['tasks'][name])
            with (out/name/'pred_0/rows.csv').open(newline='') as f:
                preds.append(np.array([r['prediction'] for r in csv.DictReader(f)]))
        truth=joblib.load(out/'shared/holdout.joblib')['y']
        original,extended=[p==truth for p in preds]
        changes=dict(identical_predictions=int(np.sum(preds[0]==preds[1])),newly_correct=int(np.sum(extended & ~original)),
                     newly_wrong=int(np.sum(original & ~extended)),rows=len(truth))
        rec.write(out/'results.json',dict(paths=results,paired=changes))
        rec.write(out/'completion.json',dict(status='passed',paths=2,fits=len(rec.trace(out)),model_api_requests=0,
                  api_rmb=0,course_tests_accessed=False,h_scores=0))
    except Exception as error:
        rec.write(out/'completion.json',dict(status='failed',reason=str(error),actual_fits=len(rec.trace(out)),model_api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='mode',required=True)
    for mode in ('rows','prepare','run'):
        p=commands.add_parser(mode)
        p.add_argument('--output',type=Path,required=True)
        if mode!='rows':
            p.add_argument('--rows',type=Path,required=True)
            p.add_argument('--weights',type=Path,required=True)
        if mode=='run':
            p.add_argument('--reference',type=Path,required=True)
    run(parser.parse_args())
