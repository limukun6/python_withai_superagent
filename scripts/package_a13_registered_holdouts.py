"""Package registered H tasks for the common CLI; never train or score them."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.contracts import TaskSpec
from superagent.data import load_training_task
from superagent.ab_catalog import capabilities_for


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--h1',required=True,type=Path)
    p.add_argument('--h2',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);results={}
    save(root/'protocol.json',dict(batch='A13-H-packaging',actual_classifier_fits=0,model_evaluation_performed=False,
        source_registrations=dict(H1=sha(a.h1/'registration.json'),H2=sha(a.h2/'registration.json'))))
    for key,parent,folder in [('H1',a.h1,'H1_paws'),('H2',a.h2,'H2_rice')]:
        registration=read(parent/'registration.json')[key]
        for file,expected in registration['files'].items():
            if sha(parent/folder/file)!=expected:raise ValueError('Registered data changed: '+key)
        path=root/(key+'_task.zip')
        if key=='H1':
            with (parent/folder/'train.tsv').open(encoding='utf-8',newline='') as file:
                rows=list(csv.DictReader(file,delimiter='\t'))
            buffer=io.StringIO(newline='');writer=csv.writer(buffer);writer.writerow(['sentence1','sentence2','label'])
            for row in rows:writer.writerow([row['sentence1'],row['sentence2'],row['label']])
            manifest=dict(schema_version=1,objective='classification',modality='pair_text',metric='accuracy',
                columns=[dict(name=n,type='text') for n in ('sentence1','sentence2')],target='label',
                pair_semantics='symmetric',pair_label_one='equivalent',
                splits={'train':dict(format='csv',path='train.csv')},
                source=dict(dataset='Google Research PAWS-Wiki labeled_final publisher train subset',
                            source_sha256=registration['source']['sha256']))
            with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('train.csv',buffer.getvalue());archive.writestr('manifest.json',json.dumps(manifest))
        else:
            manifest=dict(schema_version=1,objective='classification',modality='image',metric='accuracy',
                splits={'train':dict(format='npz',path='train.npz')},
                source=dict(dataset='UCI Rice Leaf Diseases',licence=registration['licence'],
                            citation=registration['citation'],source_sha256=registration['source']['sha256']),
                fixed_input_conversion='32x32 RGB bilinear; no learned preprocessing')
            with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('train.npz',(parent/folder/'registered.npz').read_bytes())
                archive.writestr('manifest.json',json.dumps(manifest))
        task=load_training_task(path);spec=TaskSpec.from_task(task).record()
        results[key]=dict(task_archive_sha256=sha(path),task_spec=spec,rows=len(task.y_train),
            initial_executable_candidates=list(capabilities_for(spec,train_rows=len(task.y_train))),
            model_evaluation_performed=False,score=None,partitions=registration['partitions'])
    save(root/'registration.json',results)
    save(root/'completion.json',dict(status='packaged_not_evaluated',actual_classifier_fits=0,real_api_requests=0,
        H1_requires_safe_symmetric_methods=not bool(results['H1']['initial_executable_candidates'])))
    print(json.dumps({k:dict(rows=v['rows'],kind=v['task_spec']['kind'],candidates=len(v['initial_executable_candidates'])) for k,v in results.items()}))
