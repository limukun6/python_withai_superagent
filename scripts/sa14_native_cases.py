"""Public development repackaging, no manifest; answers stay outside task packages."""
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from sklearn.datasets import load_iris,load_digits

from scripts.run_sa5_semantic_panel import save,sha
from scripts.sa7_multiturn_cases import array_fingerprint
from scripts.sa8_real_delivery_cases import partition


def npz(**values):
    stream=io.BytesIO();np.savez_compressed(stream,**values);return stream.getvalue()


def output(delimiter=',',encoding='utf-8'):
    return dict(version='output-csv-v1',delimiter=delimiter,encoding=encoding,header=True,
        columns=[dict(name='Row',kind='row_index',start=1),dict(name='Decision',kind='label')],order=dict(mode='input'))


def build(destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=False);registry=[]
    def register(name,origin,train,query,answer,x,y,q,answers):
        for path,members in [(root/(name+'.zip'),train),(root/(name+'.query.zip'),query)]:
            with zipfile.ZipFile(path,'w') as archive:
                for member,value in members.items():archive.writestr(member,value)
        oracle=dict(answer=answer,expected=dict(kind='image' if answer['plan']['modality']=='image' else 'numeric_tabular',
            metric=answer['plan']['metric'],classes=np.unique(y).tolist(),training_rows=len(y),prediction_rows=len(q),
            training_features_sha256=array_fingerprint(x),training_labels_sha256=array_fingerprint(y),
            prediction_features_sha256=array_fingerprint(q),prediction_shape=list(q.shape),output=answer['plan']['output']))
        save(root/(name+'.oracle.json'),oracle);np.save(root/(name+'.development_answers.npy'),answers,allow_pickle=False)
        registry.append(dict(id=name,source=origin,package=name+'.zip',query=name+'.query.zip',
            package_sha256=sha(root/(name+'.zip')),query_sha256=sha(root/(name+'.query.zip')),
            oracle_sha256=sha(root/(name+'.oracle.json')),training_rows=len(y),prediction_rows=len(q),
            source_status='public development data; a new packaging is not independent task transfer evidence'))
    iris=load_iris();fit,held=partition(iris.data,iris.target,96,24)
    x,y,q=iris.data[fit],iris.target[fit],iris.data[held]
    refs=['doc:README.md:0']
    def proposal(plan):
        return dict(version='semantic-native-v1',plan=plan,evidence={k:refs for k in
            ('training','target','modality','metric','prediction','image_preprocessing','output')},uncertainties=[])
    a=proposal(dict(modality='tabular',training=dict(format='npz',path='evidence.pack',features_key='sensors',target_key='state'),
        prediction=dict(format='npz',path='future.pack',features_key='readings'),metric='balanced_accuracy',positive_label=None,
        image_preprocessing=None,output=output(';','utf-8-sig')))
    doc=('This is a numeric classification problem. evidence.pack is a NumPy NPZ container: sensors contains all numeric input '
         'columns; state is the target vector in the same row order, with integer class codes 0, 1, 2. '
         'The separate prediction package contains future.pack, an NPZ container whose readings array contains the corresponding '
         'input columns in their original order. Use balanced accuracy as the primary metric. '
         'Export a semicolon-delimited UTF-8 CSV with BOM. The header is Row;Decision. Row starts at 1 and increments in '
         'prediction array order. Decision contains unchanged integer class codes. Do not evaluate on prediction labels.')
    register('N1','sklearn.load_iris',{'README.md':doc,'evidence.pack':npz(sensors=x,state=y)},
             {'future.pack':npz(readings=q)},a,x,y,q,iris.target[held])
    digits=load_digits(n_class=3);fit,held=partition(digits.images,digits.target,90,30)
    all_pixels=np.rint(digits.images/16*255).astype(np.uint8);names=np.array(['amber','cobalt','jade'])
    train={'README.md':('Classify the bitmap images. Supervised images are under library/; the immediate subdirectory '
        'name is the target class, one of amber, cobalt, jade. Only library/ supplies training. The separate prediction '
        'package contains unlabeled images under arrivals/. Resize each decoded image to 10 by 10 pixels in RGB mode '
        'using bilinear stretch; use that same conversion at prediction. The primary metric is macro F1. '
        'Write UTF-8 comma-separated CSV with header Row,Decision. Row starts at 1 in ascending full prediction image '
        'path order. Decision is the unchanged class directory name. There are no original record IDs or sidecar labels.')}
    query={};expected_train=[];expected_labels=[];q=[]
    def encoded(index):
        image=Image.fromarray(all_pixels[index]);stream=io.BytesIO();image.save(stream,format='PNG')
        decoded=np.asarray(image.convert('RGB').resize((10,10),Image.Resampling.BILINEAR));return stream.getvalue(),decoded
    entries=[]
    for i,index in enumerate(fit):
        label=names[digits.target[index]];member=f'library/{label}/{i:04d}.png';body,decoded=encoded(index)
        train[member]=body;entries.append((member,decoded,label))
    for _,decoded,label in sorted(entries,key=lambda entry:entry[0]):expected_train.append(decoded);expected_labels.append(label)
    for i,index in enumerate(held):
        body,decoded=encoded(index);query[f'arrivals/{i:04d}.png']=body;q.append(decoded)
    a=proposal(dict(modality='image',training=dict(format='image_folder',prefix='library/'),
        prediction=dict(format='image_folder',prefix='arrivals/'),metric='macro_f1',positive_label=None,
        image_preprocessing=dict(width=10,height=10,mode='RGB',resize='stretch_bilinear'),output=output()))
    register('N2','sklearn.load_digits, previously used source family',train,query,a,np.stack(expected_train),np.array(expected_labels),
             np.stack(q),names[digits.target[held]])
    save(root/'registry.json',registry);return registry


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.output),ensure_ascii=False))
