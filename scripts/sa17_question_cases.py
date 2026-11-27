"""Controlled question-activation cases; no model answers or classification predictions are injected."""
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from scripts.run_sa5_semantic_panel import read,save,sha


def npy(value):
    stream=io.BytesIO();np.save(stream,value,allow_pickle=False);return stream.getvalue()


def npz(**values):
    stream=io.BytesIO();np.savez(stream,**values);return stream.getvalue()


def build(destination,source):
    root=Path(destination);root.mkdir(parents=True,exist_ok=False);source=Path(source)
    original=read(source/'registry.json')[0]
    for key,hashed in [('package','package_sha256'),('query','query_sha256')]:
        if sha(source/original[key])!=original[hashed]:raise ValueError('SA15 source package changed')
    with zipfile.ZipFile(source/original['package']) as archive:
        x=np.load(io.BytesIO(archive.read('observations.npy')),allow_pickle=False)
        y=np.load(io.BytesIO(archive.read('decisions.npy')),allow_pickle=False)
    with zipfile.ZipFile(source/original['query']) as archive:q=np.load(io.BytesIO(archive.read('arrivals.npy')),allow_pickle=False)
    registry=[]
    def register(name,task,query,condition,expected):
        for path,members in [(root/(name+'.zip'),task),(root/(name+'.query.zip'),query)]:
            with zipfile.ZipFile(path,'w') as archive:
                for member,value in members.items():archive.writestr(member,value)
        save(root/(name+'.condition.json'),dict(condition=condition,expected=expected))
        registry.append(dict(id=name,package=name+'.zip',query=name+'.query.zip',
            package_sha256=sha(root/(name+'.zip')),query_sha256=sha(root/(name+'.query.zip')),
            condition_sha256=sha(root/(name+'.condition.json'))))
    def condition(field,question,extra):
        return dict(field=field,question=question,evidence_refs=['doc:README.md:0'],
            operator_instruction='Controlled contract-interface test: include a field-scoped question for this field in uncertainties. '
            'Use the new semantic response version and exact uncertainty_schema. This is not a model answer or an algorithm choice. '
            'Infer all other task facts from the evidence workspace. '+extra)
    # U1 keeps the exact original task/query bytes. Only the operator condition is separate.
    shutil.copyfile(source/original['package'],root/'U1.zip');shutil.copyfile(source/original['query'],root/'U1.query.zip')
    save(root/'U1.condition.json',dict(condition=condition('positive_label','No positive class is specified.',
        'Return the complete supported native plan with positive_label null, retaining the question for executor applicability checking.'),
        expected=dict(accepted=True,field='positive_label',class_count=3,metric='balanced_accuracy',
                      resolution_reason='multiclass_metric_has_no_single_positive_class')))
    registry.append(dict(id='U1',package='U1.zip',query='U1.query.zip',package_sha256=sha(root/'U1.zip'),
                         query_sha256=sha(root/'U1.query.zip'),condition_sha256=sha(root/'U1.condition.json')))
    selected=np.isin(y,[0,1]);binary=x[selected],y[selected]
    doc=('This is binary numeric classification. observations.npy is the feature matrix; decisions.npy is its target vector. '
         'The arbitrary class codes are 0 and 1, and neither code is identified as the positive event. The requested metric is ROC-AUC. '
         'The separate prediction package contains arrivals.npy with the same feature order. '
         'Export UTF-8 comma CSV Row,Decision with a 1-based row index and unchanged class code. The positive-event meaning is unresolved; do not invent it.')
    register('U2',{'README.md':doc,'observations.npy':npy(binary[0]),'decisions.npy':npy(binary[1])},
        {'arrivals.npy':npy(q)},condition('positive_label','Which of the two arbitrary codes is the positive event?',
            'Return the otherwise supported native plan with positive_label null; the operator has not supplied the missing event meaning.'),
        dict(accepted=False,field='positive_label',class_count=2,metric='roc_auc',dependency_required=True))
    doc=('A numeric classification problem is stored in evidence.npz. predictors is the input matrix; outcome_a and outcome_b '
         'are different row-aligned outcome vectors. The intended target has not been selected, so do not choose either by convention. '
         'The primary criterion is balanced accuracy. The separate prediction package contains incoming.npz with predictors. '
         'Only the owner can identify the intended outcome vector. No task-specific solution or target preference is provided.')
    register('U3',{'README.md':doc,'evidence.npz':npz(predictors=x,outcome_a=y,outcome_b=np.roll(y,1))},
        {'incoming.npz':npz(predictors=q)},condition('target','Which outcome vector defines the intended prediction target?',
            'A null plan is appropriate because the core target is genuinely unspecified. Report the target question without choosing an outcome.'),
        dict(accepted=False,field='target',core_required=True,no_data_members_read=True))
    save(root/'registry.json',registry);return registry
