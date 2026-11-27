"""Repackage already-used Iris development features to trigger the default .npy handoff."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts.run_sa5_semantic_panel import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','cases','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'protocol.json',dict(batch='SA15-case-preparation',runner_sha256=sha(Path(__file__)),
        maximum_classifier_fits=0,maximum_real_API_requests=0,prediction_answers_opened=False))
    row=next(r for r in read(a.source/'registry.json') if r['id']=='N1')
    for key,hashed in [('package','package_sha256'),('query','query_sha256')]:
        if sha(a.source/row[key])!=row[hashed]:raise ValueError('Registered SA14 source changed')
    def array(archive,member,key):
        with zipfile.ZipFile(archive) as pack,np.load(io.BytesIO(pack.read(member)),allow_pickle=False) as data:return np.array(data[key])
    x=array(a.source/row['package'],'evidence.pack','sensors');y=array(a.source/row['package'],'evidence.pack','state')
    q=array(a.source/row['query'],'future.pack','readings')
    def npy(value):
        stream=io.BytesIO();np.save(stream,value,allow_pickle=False);return stream.getvalue()
    a.cases.mkdir(parents=True,exist_ok=False)
    doc=('Classify the numeric observations. observations.npy contains the training feature matrix. decisions.npy contains '
         'its row-aligned integer target vector, with codes 0, 1, 2. The separate prediction package contains arrivals.npy '
         'with the same feature columns in the same order. The primary metric is balanced accuracy. '
         'Export semicolon-separated UTF-8 CSV with BOM and header Row;Decision. Row begins at 1 in prediction array order. '
         'Decision is the unchanged integer class code. No manifest is supplied; no prediction targets are available.')
    for path,members in [(a.cases/'D1.zip',{'README.md':doc,'observations.npy':npy(x),'decisions.npy':npy(y)}),
                         (a.cases/'D1.query.zip',{'arrivals.npy':npy(q)})]:
        with zipfile.ZipFile(path,'w') as archive:
            for name,value in members.items():archive.writestr(name,value)
    oracle=read(a.source/'N1.oracle.json');oracle['answer']['plan'].update(
        training=dict(format='npy',path='observations.npy',target_path='decisions.npy'),prediction=dict(format='npy',path='arrivals.npy'))
    save(a.cases/'D1.oracle.json',oracle)
    result=dict(status='prepared',source_registry_sha256=sha(a.source/'registry.json'),actual_classifier_fits=0,real_API_requests=0,
        course_cost_rmb=0,prediction_answers_opened=False,scope='same Iris development rows as SA14, new NPY packaging; not independent transfer')
    registry=[dict(row,id='D1',package='D1.zip',query='D1.query.zip',package_sha256=sha(a.cases/'D1.zip'),
                   query_sha256=sha(a.cases/'D1.query.zip'),oracle_sha256=sha(a.cases/'D1.oracle.json'))]
    save(a.cases/'registry.json',registry);save(a.output/'registry.json',registry)
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
