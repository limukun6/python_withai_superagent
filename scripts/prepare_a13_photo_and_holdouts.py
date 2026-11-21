"""Download named public data once, register holdouts without any model fits."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import zipfile

import joblib
import numpy as np
from PIL import Image
import requests
from scipy.io import loadmat

from scripts.a_batch_common import save
from superagent.data import TrainingTask
from superagent.splits import grouped_partition,take,sample_groups,overlap_audit


def digest(data):return hashlib.sha256(data).hexdigest()


PROXY=None


def download(url,path,expected=None):
    if path.exists():raise FileExistsError('Do not overwrite a downloaded source')
    with requests.get(url,stream=True,timeout=60,proxies={'https':PROXY} if PROXY else None) as response:
        response.raise_for_status()
        with path.open('xb') as file:
            for chunk in response.iter_content(1024*1024):file.write(chunk)
    actual=digest(path.read_bytes())
    if expected and actual!=expected:raise ValueError('Public source checksum mismatch')
    return dict(url=url,file=path.name,bytes=path.stat().st_size,sha256=actual)


def png_package(path,images,labels,split,source):
    fields=['image','label'] if labels is not None else ['image']
    manifest=dict(schema_version=1,objective='classification',modality='image',metric='accuracy',
        columns=[dict(name='image',type='image_path')],target='label',source=source,
        image_preprocessing=dict(width=64,height=64,mode='L',resize='stretch_bilinear'),
        splits={split:dict(format='csv',path=split+'.csv')})
    rows=io.StringIO(newline='');writer=csv.writer(rows);writer.writerow(fields)
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for index,image in enumerate(images):
            name=f'images/{index:04d}.png';data=io.BytesIO();Image.fromarray(image).save(data,format='PNG')
            archive.writestr(name,data.getvalue())
            writer.writerow([name,int(labels[index])] if labels is not None else [name])
        archive.writestr(split+'.csv',rows.getvalue());archive.writestr('manifest.json',json.dumps(manifest))


def face_data(root):
    record=download('https://ndownloader.figshare.com/files/5976027',root/'olivettifaces.mat',
        'b612fb967f2dc77c9c62d3e1266e0c73d5fca46a4b8906c18e454d41af987794')
    faces=loadmat(root/'olivettifaces.mat')['faces'].T.copy().reshape(400,64,64).transpose(0,2,1)
    if faces.min()<0 or faces.max()>255:raise ValueError('Unexpected face pixel domain')
    images=faces.astype(np.uint8);labels=np.arange(400)//10
    task=TrainingTask('image',images,labels,{'metric':'accuracy'})
    train,held,resource=grouped_partition(task,45101)
    folder=root/'photo_dev';folder.mkdir()
    citation=dict(dataset='AT&T/Olivetti faces, sklearn 64x64 variant',
        attribution='AT&T Laboratories Cambridge',terms_url='https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html')
    png_package(folder/'train.zip',images[train],labels[train],'train',citation)
    png_package(folder/'features.zip',images[held],None,'test',citation)
    joblib.dump(dict(X=images[held],y=labels[held],row_ids=held.tolist()),folder/'holdout.joblib')
    save(folder/'partition.json',dict(train_rows=train.tolist(),held_rows=held.tolist(),resource=resource,
        audit=overlap_audit(task,train,held)))
    return dict(source=record,terms=citation,images=400,classes=40,shape=[64,64],
        train_rows=len(train),held_rows=len(held),content_sha256=digest(images.tobytes()+labels.tobytes()),
        files={p.name:digest(p.read_bytes()) for p in folder.iterdir()},
        role='development, not unseen; closed-set identity labels, not new-person recognition')


def pair_holdout(root):
    record=download('https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz',root/'paws_wiki_labeled_final.tar.gz')
    opened=[]
    with tarfile.open(root/'paws_wiki_labeled_final.tar.gz','r:gz') as archive:
        members=[m for m in archive.getmembers() if m.name.endswith('/train.tsv') or m.name=='train.tsv']
        if len(members)!=1:raise ValueError('Expected one official PAWS training member')
        raw=archive.extractfile(members[0]).read();opened.append(members[0].name)
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8')),delimiter='\t'))
    return register_pair_rows(root,rows,record,opened)


def register_pair_rows(root,rows,record,opened):
    x=[(r['sentence1'],r['sentence2']) for r in rows];y=np.asarray([int(r['label']) for r in rows])
    task=TrainingTask('pair_text',x,y,{'metric':'accuracy','input_schema':{'pair_semantics':'symmetric'}})
    groups=sample_groups(task,'shared_text');unique=np.unique(groups)
    np.random.default_rng(61401).shuffle(unique)
    chosen=[];count=0
    for group in unique:
        chosen.append(group);count+=int(np.sum(groups==group))
        if count>=2000:break
    selected=np.flatnonzero(np.isin(groups,chosen))
    task=TrainingTask('pair_text',take(x,selected),y[selected],task.spec)
    folder=root/'H1_paws';folder.mkdir()
    with (folder/'train.tsv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.writer(file,delimiter='\t');writer.writerow(['id','sentence1','sentence2','label'])
        for i in selected:writer.writerow([rows[i]['id'],*x[i],int(y[i])])
    splits={}
    for seed in (61411,61421):
        train,held,resource=grouped_partition(task,seed,'shared_text')
        splits[str(seed)]=dict(train=train.tolist(),held=held.tolist(),resource=resource,
            audit=overlap_audit(task,train,held))
    save(folder/'partitions.json',splits)
    return dict(source=record,opened_members=opened,official_dev_test_opened=False,
        selected_source_positions=selected.tolist(),selected_original_ids=[rows[i]['id'] for i in selected],
        rows=len(selected),label_counts={str(k):int(np.sum(task.y_train==k)) for k in np.unique(task.y_train)},
        semantics='symmetric; label1=paraphrase, label0=different meaning; swapping preserves labels',
        terms_url='https://github.com/google-research-datasets/paws/blob/master/LICENSE',
        terms='free use with Google acknowledgement appreciated; not a claimed CC licence',
        files={p.name:digest(p.read_bytes()) for p in folder.iterdir()},
        partitions=splits,model_evaluation_performed=False,score=None,
        readiness='data registered; symmetric task parser and B methods pending')


def image_holdout(root):
    record=download('https://archive.ics.uci.edu/static/public/486/rice+leaf+diseases.zip',root/'rice_leaf_diseases.zip')
    images=[];labels=[];names=[];hashes=[]
    with zipfile.ZipFile(root/'rice_leaf_diseases.zip') as archive:
        members=sorted(n for n in archive.namelist() if n.lower().endswith(('.jpg','.jpeg')) and '__MACOSX' not in n)
        classes=sorted({Path(n).parent.name for n in members})
        for name in members:
            raw=archive.read(name)
            with Image.open(io.BytesIO(raw)) as source:
                image=np.asarray(source.convert('RGB').resize((32,32),Image.Resampling.BILINEAR)).copy()
            images.append(image);labels.append(classes.index(Path(name).parent.name));names.append(name);hashes.append(digest(raw))
    if len(images)!=120 or len(classes)!=3:raise ValueError('Unexpected UCI Rice class/sample inventory')
    values=np.stack(images);targets=np.asarray(labels)
    task=TrainingTask('image',values,targets,{'metric':'accuracy'})
    splits={}
    for seed in (61411,61421):
        train,held,resource=grouped_partition(task,seed)
        if {hashes[i] for i in train}&{hashes[i] for i in held}:raise ValueError('Raw image duplicate crosses holdout')
        splits[str(seed)]=dict(train=train.tolist(),held=held.tolist(),resource=resource,audit=overlap_audit(task,train,held))
    folder=root/'H2_rice';folder.mkdir()
    np.savez_compressed(folder/'registered.npz',X=values,y=targets)
    save(folder/'partitions.json',splits)
    return dict(source=record,classes=classes,rows=len(values),shape=[32,32,3],
        label_counts={classes[k]:int(np.sum(targets==k)) for k in range(3)},
        members=names,raw_image_sha256=hashes,decoded_sha256=digest(values.tobytes()),
        files={p.name:digest(p.read_bytes()) for p in folder.iterdir()},partitions=splits,
        licence='CC BY 4.0',citation='Shah, J., Prajapati, H., & Dabhi, V. (2017). Rice Leaf Diseases. doi:10.24432/C5R013',
        source_page='https://archive.ics.uci.edu/dataset/486/rice%2Bleaf%2Bdiseases',
        model_evaluation_performed=False,score=None,
        boundary='Only exact duplicates grouped; no original plant/session IDs supplied, field independence unverified')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--dataset',choices=['all','photo_dev','H1','H2'],default='all')
    p.add_argument('--proxy',help='Explicit existing proxy for public dataset download only; never school API')
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();records={}
    global PROXY
    PROXY=a.proxy
    save(root/'protocol.json',dict(batch='A13-prepare',plan_sha256=digest((Path(__file__).resolve().parents[1]/
        'docs/batches/A13_photo_and_H_registration_plan.md').read_bytes()),runner_sha256=digest(Path(__file__).read_bytes()),
        actual_classifier_fits=0,real_api_requests=0,holdout_models_run=False))
    try:
        for name,prepare in [('photo_dev',face_data),('H1',pair_holdout),('H2',image_holdout)]:
            if a.dataset not in ('all',name):continue
            records[name]=prepare(root);save(root/'registration.json',records)
            print(json.dumps(dict(dataset=name,status='prepared',fits=0)),flush=True)
        save(root/'completion.json',dict(status='prepared',actual_classifier_fits=0,real_api_requests=0,
            holdout_scores_opened=False,elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(root/'registration.json',records);save(root/'completion.json',dict(status='failed',error=str(error),
            actual_classifier_fits=0,real_api_requests=0,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
