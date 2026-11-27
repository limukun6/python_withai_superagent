"""A standard readable manifest does not encode the document's custom submission rules."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.sa7_multiturn_cases import array_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','cases','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'protocol.json',dict(batch='SA16-output-case-preparation',runner_sha256=sha(Path(__file__)),
        maximum_classifier_fits=0,maximum_real_API_requests=0,prediction_answers_opened=False))
    row=next(r for r in read(a.source/'registry.json') if r['id']=='N1')
    for key,hashed in [('package','package_sha256'),('query','query_sha256')]:
        if sha(a.source/row[key])!=row[hashed]:raise ValueError('Registered source changed')
    def array(archive,member,key):
        with zipfile.ZipFile(archive) as pack,np.load(io.BytesIO(pack.read(member)),allow_pickle=False) as data:return np.array(data[key])
    x=array(a.source/row['package'],'evidence.pack','sensors');y=array(a.source/row['package'],'evidence.pack','state').astype(str)
    q=array(a.source/row['query'],'future.pack','readings');columns=[dict(name=f'f{i}',type='numeric') for i in range(x.shape[1])]
    manifest=dict(schema_version=1,modality='tabular',target='class_code',metric='balanced_accuracy',columns=columns,
                  splits=dict(train=dict(format='csv',path='train.csv')))
    qmanifest={**manifest,'splits':dict(test=dict(format='csv',path='query.csv'))}
    def table(header,rows):
        stream=io.StringIO(newline='');writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows);return stream.getvalue()
    ids=[f'{i:06d}' for i in range(len(q))]
    output=dict(version='output-csv-v1',delimiter='|',encoding='utf-8-sig',header=True,columns=[
        dict(name='Case',kind='input_id',field='record_id'),
        dict(name='Result',kind='label',mapping={'0':'c0','1':'c1','2':'c2'}),
        dict(name='ScoreFor2',kind='probability',class_label='2')],order=dict(mode='input'))
    doc=('The manifest fixes the input columns, training target and balanced accuracy objective. '
         'The independent prediction package has query.csv with record_id and the same feature columns. '
         'Submission requirements are separate from the manifest: export pipe-separated UTF-8 CSV with BOM, '
         'exactly Case|Result|ScoreFor2. Case copies record_id unchanged, including leading zeros. '
         'Result maps training labels 0 to c0, 1 to c1, and 2 to c2. ScoreFor2 is the probability of training class 2. '
         'Preserve prediction input row order. Do not evaluate on prediction labels.')
    a.cases.mkdir(parents=True,exist_ok=False)
    members=[(a.cases/'A1.zip',dict(**{'README.md':doc,'manifest.json':json.dumps(manifest),
             'train.csv':table([c['name'] for c in columns]+['class_code'],[[*features,label] for features,label in zip(x,y)])})),
             (a.cases/'A1.query.zip',{'manifest.json':json.dumps(qmanifest),
             'query.csv':table(['record_id']+[c['name'] for c in columns],[[identifier,*features] for identifier,features in zip(ids,q)])})]
    for path,files in members:
        with zipfile.ZipFile(path,'w') as archive:
            for name,value in files.items():archive.writestr(name,value)
    expected=dict(kind='numeric_tabular',metric='balanced_accuracy',classes=np.unique(y).tolist(),training_rows=len(y),prediction_rows=len(q),
        training_features_sha256=array_fingerprint(x),training_labels_sha256=array_fingerprint(y),
        prediction_features_sha256=array_fingerprint(q),prediction_shape=list(q.shape),output=output)
    save(a.cases/'A1.oracle.json',dict(expected=expected,prediction_ids=ids))
    registry=[dict(id='A1',source='SA14 Iris development rows, standard manifest and new documented output',package='A1.zip',query='A1.query.zip',
        package_sha256=sha(a.cases/'A1.zip'),query_sha256=sha(a.cases/'A1.query.zip'),oracle_sha256=sha(a.cases/'A1.oracle.json'),
        training_rows=len(y),prediction_rows=len(q))]
    save(a.cases/'registry.json',registry);save(a.output/'registry.json',registry)
    from superagent.semantic_contract import resolve_training_task
    local=resolve_training_task(a.cases/'A1.zip',mode='fallback',prediction_package=a.cases/'A1.query.zip')
    result=dict(status='prepared',input_locally_readable=True,legacy_fallback_complete_output=local.spec.get('output_recipe')==output,
        actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,prediction_answers_opened=False,
        scope='existing public development rows, not independent task transfer')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
