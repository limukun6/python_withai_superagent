"""One new public development source, two preregistered input conditions."""
import json
from pathlib import Path
import zipfile

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from scripts.run_sa5_semantic_panel import save,sha
from scripts.sa7_multiturn_cases import array_fingerprint
from scripts.sa14_native_cases import npz


def build(root):
    import io
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    data=load_breast_cancer();pool,held=train_test_split(np.arange(len(data.target)),test_size=40,
        random_state=271121,stratify=data.target)
    fit,_=train_test_split(pool,train_size=240,random_state=271122,stratify=data.target[pool])
    x,q=data.data[fit],data.data[held];y=np.where(data.target[fit]==1,'B','M')
    columns=[f'measure_{i:02d}' for i in range(30)];aliases=[f'read_{i:02d}' for i in range(30)]
    ids=[f'{5000+i:08d}' for i in range(len(q))]
    train_rows=[dict(batch_id=f'{1000+i:08d}',status_code=str(y[i]),**dict(zip(columns,map(float,row)))) for i,row in enumerate(x)]
    query_rows=[dict(specimen_key=ids[i],**dict(zip(reversed(aliases),map(float,reversed(row))))) for i,row in enumerate(q)]
    output1=dict(version='output-csv-v1',delimiter='\t',encoding='utf-8-sig',header=True,
        columns=[dict(name='Case',kind='input_id',field='specimen_key'),dict(name='Decision',kind='label'),
                 dict(name='BenignScore',kind='probability',class_label='B')],order=dict(mode='input'))
    doc1=('This is binary classification. Training file observations.jsonl contains one JSON object per line. '
        'status_code is the target, with class codes M and B; batch_id is a record identifier and must not be a predictor. '
        'Use all 30 numeric measurement fields in ascending numeric suffix order, measure_00 through measure_29. '
        'The primary metric is macro F1. A separate prediction package contains arrivals.jsonl. '
        'Prediction fields read_NN correspond to training measure_NN by the same two-digit suffix, irrespective of physical '
        'field order. The prediction exporter reverses the physical order of the measurement fields. '
        'The only non-measurement prediction field is its original identifier; inspect the actual header to establish its name. '
        'Preserve that identifier verbatim including leading zeros and preserve original prediction row order. '
        'Export UTF-8 with BOM, tab-separated, header Case, Decision, BenignScore (three tab-separated names). '
        'Case contains the original prediction identifier, Decision the unchanged M/B class, and BenignScore the '
        'probability of class B. No target values or class probabilities are supplied for prediction rows.')
    stream=io.BytesIO();np.save(stream,q,allow_pickle=False)
    output2=dict(version='output-csv-v1',delimiter=';',encoding='utf-8',header=True,
        columns=[dict(name='Index',kind='row_index',start=0),dict(name='Class',kind='label')],order=dict(mode='input'))
    doc2=('This is numeric binary classification. Training file archive.npz contains exactly one two-dimensional '
        'numeric measurement matrix and one one-dimensional class-code vector in matching row order. '
        'The matrix defines the input and the vector defines the target. Their key names are unknown; inspect metadata. '
        'Class codes are M and B. Use all 30 measurement columns in stored order. Primary metric is balanced accuracy. '
        'The separate prediction package has features.npy, one NumPy NPY matrix with those same columns and no labels. '
        'There are no original IDs, no pixel preprocessing and no class remapping. Export UTF-8 semicolon-separated '
        'CSV with header Index;Class. Index begins at zero and follows prediction matrix row order; Class is the unchanged M/B code.')
    registry=[]
    for name,train,query,recipe in [
        ('C1',{'README.md':doc1,'observations.jsonl':'\n'.join(json.dumps(r) for r in train_rows)+'\n'},
         {'arrivals.jsonl':'\n'.join(json.dumps(r) for r in query_rows)+'\n'},output1),
        ('C2',{'README.md':doc2,'archive.npz':npz(bedrock=x,lattice=y)}, {'features.npy':stream.getvalue()},output2)]:
        for path,members in [(root/(name+'.zip'),train),(root/(name+'.query.zip'),query)]:
            with zipfile.ZipFile(path,'w') as archive:
                for member,body in members.items():archive.writestr(member,body)
        oracle=dict(recoverable=True,prediction_ids=ids if name=='C1' else None,expected=dict(kind='numeric_tabular',
            metric='macro_f1' if name=='C1' else 'balanced_accuracy',classes=np.unique(y).tolist(),training_rows=len(y),prediction_rows=len(q),
            training_features_sha256=array_fingerprint(x),training_labels_sha256=array_fingerprint(y),prediction_shape=list(q.shape),
            prediction_features_sha256=array_fingerprint(q),output=recipe))
        save(root/(name+'.oracle.json'),oracle)
        registry.append(dict(id=name,source='sklearn.load_breast_cancer bundled public data; development',
            package=name+'.zip',query=name+'.query.zip',package_sha256=sha(root/(name+'.zip')),query_sha256=sha(root/(name+'.query.zip')),
            oracle_sha256=sha(root/(name+'.oracle.json')),training_rows=len(y),prediction_rows=len(q),recoverable=True))
    save(root/'registry.json',registry)
    save(root/'source.json',dict(source='sklearn.load_breast_cancer',fit_indices=fit.tolist(),prediction_indices=held.tolist(),
        source_X_sha256=array_fingerprint(data.data),source_y_sha256=array_fingerprint(data.target),prediction_answers_saved=False,
        evidence_boundary='one new source with two packaging conditions; not two independent tasks or a sealed transfer evaluation'))
    return registry
