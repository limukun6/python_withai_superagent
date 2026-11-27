"""Frozen Wine development recovery conditions; no answers in prediction packages."""
import csv
import io
from pathlib import Path
import zipfile

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from scripts.run_sa5_semantic_panel import save,sha
from scripts.sa7_multiturn_cases import array_fingerprint
from scripts.sa14_native_cases import npz,output


def table(header,rows):
    stream=io.StringIO(newline='');writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)
    return stream.getvalue()


def build(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    data=load_wine();fit,held=train_test_split(np.arange(len(data.target)),test_size=36,random_state=271119,stratify=data.target)
    x,y,q=data.data[fit],data.target[fit],data.data[held]
    names=[f'm{i:02d}' for i in range(x.shape[1])];renamed=[f'q{i:02d}' for i in range(x.shape[1])]
    records=table(names+['label'],[list(a)+[int(b)] for a,b in zip(x,y)])
    recipe=output();registry=[]
    common=('This is classification. The primary metric is balanced accuracy. '
        'Do not use prediction labels or evaluate external scores. Output UTF-8 comma-separated CSV with '
        'header Row,Decision. Row begins at 1 in original prediction row order. Decision is the unchanged class code. ')
    for name in ('R1','R2','R3','R4'):
        if name=='R1':
            doc=(common+'The supervised training file is train.csv. label is the target; all preceding 13 measurement '
                'columns are numeric predictors. A separate package contains query.csv. It contains the same 13 measurements '
                'in the same column order, but its exporter changed the field names. Inspect its actual header, then adapt '
                'the names to the corresponding training fields by their documented order. There are no IDs or split columns.')
            train={'README.md':doc,'train.csv':records};query={'query.csv':table(renamed,q)}
        elif name=='R2':
            doc=(common+'Training data is packed.npz, a NumPy NPZ container with exactly one two-dimensional numeric '
                'measurement matrix and one one-dimensional class vector. The matrix is the input, the vector is the target '
                'in the same row order. The internal key names are unknown; inspect the container metadata. '
                'The separate package has arrivals.npz containing exactly one two-dimensional measurement matrix with the '
                'same 13 columns in the same order. Inspect its key as well. No preprocessing or label remapping is requested.')
            train={'README.md':doc,'packed.npz':npz(cinder=x,opal=y)};query={'arrivals.npz':npz(pollen=q)}
        elif name=='R3':
            doc=(common+'Training file train.csv has 13 numeric measurement columns followed by label, the target. '
                'Use all measurement columns. Separate query.csv has the same feature names and order. No IDs or split columns.')
            train={'README.md':doc,'train.csv':records};query={'query.csv':table(names,q)}
        else:
            doc=(common+'Training container packed.npz has measurements (the 13-column numeric input), annotation_a '
                'and annotation_b (two different class annotations). Neither annotation is identified as the requested '
                'prediction target anywhere in the supplied material. The target definition is missing; neither is a '
                'default. Inspect the material if useful and explicitly retain target as unresolved rather than guessing '
                'from validation scores. The separate query.csv has 13 input measurement columns and no annotations.')
            train={'README.md':doc,'packed.npz':npz(measurements=x,annotation_a=y,annotation_b=(y+1)%3)}
            query={'query.csv':table(names,q)}
        for path,members in [(root/(name+'.zip'),train),(root/(name+'.query.zip'),query)]:
            with zipfile.ZipFile(path,'w') as archive:
                for member,body in members.items():archive.writestr(member,body)
        # CSV readers preserve categorical target codes as strings; numpy keeps ints.
        target=y if name=='R2' else y.astype(str)
        oracle=dict(recoverable=name!='R4',expected=None if name=='R4' else dict(kind='numeric_tabular',metric='balanced_accuracy',
            classes=np.unique(target).tolist(),training_rows=len(y),prediction_rows=len(q),
            training_features_sha256=array_fingerprint(x),training_labels_sha256=array_fingerprint(target),
            prediction_shape=list(q.shape),prediction_features_sha256=array_fingerprint(q),output=recipe),
            fault='first selected worker raises RuntimeError before launch; zero fits for that attempt' if name=='R3' else None)
        save(root/(name+'.oracle.json'),oracle)
        registry.append(dict(id=name,source='sklearn.load_wine public development data',package=name+'.zip',query=name+'.query.zip',
            package_sha256=sha(root/(name+'.zip')),query_sha256=sha(root/(name+'.query.zip')),oracle_sha256=sha(root/(name+'.oracle.json')),
            training_rows=len(y),prediction_rows=len(q),recoverable=name!='R4'))
    save(root/'registry.json',registry)
    save(root/'split.json',dict(fit=fit.tolist(),prediction=held.tolist(),prediction_answers_saved=False,
        evidence_boundary='one public source, four controlled development conditions, not four independent tasks'))
    return registry
