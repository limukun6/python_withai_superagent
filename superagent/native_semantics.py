"""Compile cited LLM choices into existing numeric-array/image-folder readers.

Only data descriptors are accepted. No code generation, task-name routing, or
label/metric defaults. Prediction replays the frozen descriptor without asking.
"""
from copy import deepcopy
from pathlib import PurePosixPath

import numpy as np

from .data import TrainingTask
from .format_sources import checked_name,package_files,roles
from .metrics import MetricSpec
from .schema import load_manifest_split,read_manifest
from .task_files import TrainingMembers,evaluation_path

VERSION='semantic-native-v1'
RECIPE_VERSION='semantic-native-input-v1'
INSTRUCTION=(
    'For numeric numpy arrays or class-folder images without manifest.json you may instead return '
    '{version:"semantic-native-v1",plan:{modality,training,prediction,metric,positive_label,image_preprocessing,output},'
    'evidence:{training:[],target:[],modality:[],metric:[],prediction:[],positive_label:[],image_preprocessing:[],output:[]},uncertainties:[]}. '
    'modality is tabular or image. training/prediction are existing reader descriptors, NOT code: '
    'npy training={format:"npy",path:<feature member>,target_path:<label member>}, prediction={format:"npy",path}; '
    'npz training={format:"npz",path,features_key,target_key}, prediction={format:"npz",path,features_key}; '
    'image_folder training/prediction={format:"image_folder",prefix:<directory ending slash>}, '
    'where each immediate training subdirectory is a class and prediction files are sorted by full path. '
    'For image_folder, image_preprocessing must be {width:<integer>,height:<integer>,mode:"RGB" or "L",resize:"stretch_bilinear"}; '
    'for arrays image_preprocessing must be null. prediction may be null only when no input is announced. '
    'metric must be explicit, positive_label may be null; numeric array labels retain numeric types. '
    'Every non-null declaration requires real document citations; target always requires evidence. '
    'Use native descriptors when the evidence describes arrays/folders; scalar-record plans cannot decode them. '
    'Do not reinterpret explicit manifests or select evaluation labels as features. '
    'The same output-csv-v1 schema applies; row_index is allowed only when the instructions request it. '
    'Original IDs from arbitrary sidecars/file names are not implemented by this reader: do not fabricate an ID field. '
    'Unresolved meaning must be stated in uncertainties, not filled by defaults.'
)


def require(condition,message):
    from .semantic_contract import ContractProbeError
    if not condition:raise ContractProbeError('Native contract: '+message)


def explicit_manifest(files):
    return any(PurePosixPath(n).name.casefold()=='manifest.json' for n in files.namelist())


def descriptor(value,*,training,modality):
    require(isinstance(value,dict),'reader descriptor must be an object')
    fmt=value.get('format')
    shapes={'npy':{'format','path','target_path'} if training else {'format','path'},
            'npz':{'format','path','features_key','target_key'} if training else {'format','path','features_key'},
            'image_folder':{'format','prefix'}}
    require(fmt in shapes and set(value)==shapes[fmt],'unsupported descriptor fields/format')
    for key in set(value)-{'format'}:
        require(isinstance(value[key],str) and value[key], 'descriptor fields must be nonempty strings')
    if fmt=='image_folder':
        require(modality=='image','folder reader requires image modality')
        prefix=value['prefix'];checked_name(prefix)
        require(prefix.endswith('/'),'image prefix must end with slash')
        require(not training or not evaluation_path(prefix,'train'),'training folder is an evaluation source')
        require(training or not roles(prefix)&{'validation'},'prediction folder is a validation source')
    else:
        path=checked_name(value['path'])
        require(not training or not evaluation_path(path,'train'),'training array is an evaluation source')
        require(training or not roles(path)&{'validation'},'prediction array is a validation source')
        from .semantic_contract import _not_label_vector
        _not_label_vector(path)
        if fmt=='npy' and training:
            target=checked_name(value['target_path'])
            require(path!=target and not evaluation_path(target,'train'),'invalid training label source')
        elif fmt=='npz':
            _not_label_vector(value['features_key'])
            require(not roles(value['features_key'])&{'validation','test'},'feature key names an evaluation split')
            if training:
                require(value['features_key']!=value['target_key'],'feature and target keys coincide')
                require(not roles(value['target_key'])&{'validation','test'},'target key names an evaluation split')
    return deepcopy(value)


def manifest_for(recipe,split):
    manifest=dict(schema_version=1,objective='classification',modality=recipe['modality'],
                  metric=recipe['metric'],_manifest_parent='.',_schema_origin='semantic_native_v1',
                  _metric_source='declared_task_readme',splits={split:deepcopy(recipe['training' if split=='train' else 'prediction'])})
    if recipe.get('positive_label') is not None:manifest['positive_label']=recipe['positive_label']
    if recipe['image_preprocessing'] is not None:manifest['image_preprocessing']=deepcopy(recipe['image_preprocessing'])
    return manifest


def compile_native(files,workspace,response,*,prediction_files=None):
    require(not explicit_manifest(files),'explicit manifest cannot be overridden')
    require(isinstance(response,dict) and set(response)=={'version','plan','evidence','uncertainties'} and response['version']==VERSION,
            'invalid response envelope')
    require(response['uncertainties']==[],'unresolved required semantics')
    plan=response['plan'];keys={'modality','training','prediction','metric','positive_label','image_preprocessing','output'}
    require(isinstance(plan,dict) and set(plan)==keys,'invalid plan fields')
    require(plan['modality'] in ('tabular','image') and isinstance(plan['metric'],str),'explicit modality/metric required')
    evidence=response['evidence'];known={d['id'] for d in workspace['documents']}
    require(isinstance(evidence,dict) and not set(evidence)-keys-{'target'},'invalid evidence fields')
    for key in {'training','target','modality','metric'} | {k for k in ('prediction','positive_label','image_preprocessing','output') if plan[k] is not None}:
        refs=evidence.get(key)
        require(isinstance(refs,list) and refs and all(isinstance(r,str) and r in known for r in refs),'missing/fabricated evidence for '+key)
    training=descriptor(plan['training'],training=True,modality=plan['modality'])
    query=descriptor(plan['prediction'],training=False,modality=plan['modality']) if plan['prediction'] is not None else None
    if prediction_files is not None:require(query is not None,'announced prediction input needs a descriptor')
    if query is not None and prediction_files is None and query['format']!='image_folder' and training['format']!='image_folder':
        require(query['path']!=training['path'] or query['format']==training['format']=='npz'
                and query['features_key'] not in (training['features_key'],training['target_key']),
                'shared array needs a distinct feature member/key')
    if training['format']=='image_folder':
        from .image_files import image_contract
        image_contract(plan['image_preprocessing'])
        require(query is None or query['format']=='image_folder','folder prediction must retain its image decoder')
    else:
        require(plan['image_preprocessing'] is None,'array input cannot silently apply a pixel transform')
        require(query is None or query['format'] in ('npy','npz'),'array prediction needs an array reader')
    recipe=dict(version=RECIPE_VERSION,**deepcopy({k:plan[k] for k in ('modality','metric','positive_label','image_preprocessing')}),
                training=training,prediction=query)
    # Restrict before decoding: even explicit LLM selections cannot read eval members.
    archive=TrainingMembers(files)
    kind,x,y,spec=load_manifest_split(archive,manifest_for(recipe,'train'),'train')
    require(len(np.unique(y))>=2,'training requires two observed classes')
    if plan['positive_label'] is not None and y.dtype.kind in 'US':spec['positive_label']=str(plan['positive_label'])
    spec.update(native_input_recipe=recipe,semantic_interpretation=dict(version=VERSION,evidence=deepcopy(evidence),
                claims_status='source_citations_checked; interpretation_requires_evaluation'))
    spec['goal_sources']['primary_metric']='declared_task_readme'
    # Preserve source meaning separately from inference-time columns.
    spec['semantic_target_source']=training.get('target_path',training.get('target_key','immediate_class_directory'))
    task=TrainingTask(kind,x,y,spec);metric=MetricSpec.for_task(task)
    if plan['output'] is not None:
        from .output_recipe import validate_recipe
        spec['output_recipe']=validate_recipe(plan['output'],list(metric.labels));spec['output_recipe_evidence']=deepcopy(evidence['output'])
    if query is not None:
        owner=prediction_files if prediction_files is not None else files
        _prediction(owner,task.kind,task.spec)
    return task


def _prediction(files,kind,spec):
    recipe=spec['native_input_recipe']
    require(recipe.get('version')==RECIPE_VERSION and recipe.get('prediction') is not None,'missing frozen prediction descriptor')
    query=descriptor(recipe['prediction'],training=False,modality=recipe['modality'])
    if explicit_manifest(files):
        declared=read_manifest(files)
        require(declared is not None and declared.get('modality')==recipe['modality'] and declared.get('splits',{}).get('test')==query,
                'prediction manifest conflicts with frozen source')
        require(declared.get('image_preprocessing')==recipe['image_preprocessing'],'prediction manifest conflicts with frozen decoder')
    if spec.get('image_decoder_version'):
        from importlib.metadata import version
        from .image_files import DECODER_POLICY
        require(spec.get('image_decoder_policy')==DECODER_POLICY and spec['image_decoder_version']==version('Pillow'),
                'frozen image decoder changed')
    detected,x,_,observed=load_manifest_split(files,manifest_for(recipe,'test'),'test',include_labels=False)
    require(detected==kind and observed.get('feature_shape')==spec.get('feature_shape'),'prediction kind/shape differs from frozen training')
    return x


def predict_native(package,kind,spec):
    with package_files(package) as files:return _prediction(files,kind,spec)
