"""Deliver a disclosed training class prior when features or a usable split cannot be established."""
import csv
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np

from .ab_run import sha
from .events import RunLogger, atomic_text
from .metrics import MetricSpec
from .native_schema import declared_member, description, descriptor, find_member, named_table
from .schema import _member, _validate_labels, read_manifest
from .table_input import declared_metric, heading, table, target_column
from .task_files import TABLE_FORMATS, open_package


INPUT_ERRORS = (ValueError, KeyError, FileNotFoundError, ImportError, zipfile.BadZipFile)


def training_member(archive,manifest,field='path'):
    desc,path=_member(archive,manifest,'train',field)
    tokens=re.split(r'[^a-z0-9]+',path.casefold())
    if set(tokens) & {'val','valid','validation','dev','test','testing','holdout','prediction'}:
        raise ValueError('The declared training source names an evaluation split')
    return desc,path


def prior_evidence(package, *, unresolved_metric=False):
    with open_package(package,'train',shared=True,for_partial_labels=unresolved_metric) as archive:
        manifest=read_manifest(archive, read_metric=True, for_partial_labels=unresolved_metric)
        expected_prediction=None
        if manifest:
            desc=manifest.get('splits',{}).get('train',{})
            fmt=desc.get('format')
            primary='unresolved' if unresolved_metric else manifest.get('metric','accuracy')
        else:
            prose=description(archive)
            primary='unresolved' if unresolved_metric else declared_metric(prose) or 'accuracy'
            formats=[fmt for fmt in TABLE_FORMATS if find_member(archive,['train.'+fmt])]
            named=None if getattr(archive,'shared_table',None) else named_table(prose,'train')
            expected_prediction=None if getattr(archive,'shared_table',None) else named_table(prose,'test')
            if named and named==expected_prediction:
                raise ValueError('Training and prediction files have the same declaration')
            if named:
                fmt=PurePosixPath(named).suffix[1:].lower()
                desc={'format':fmt,'path':declared_member(archive,named)}
            elif len(formats)==1:
                fmt=formats[0]
                desc=descriptor(archive,fmt,'train')
            elif not formats and find_member(archive,['y_train.npy']):
                fmt='npy'
                desc={'format':fmt,'target_path':find_member(archive,['y_train.npy'])}
            else:
                raise ValueError('No independent source identifies the training target')
            manifest={'_manifest_parent':'.','splits':{'train':desc}}
        target=manifest.get('target')
        if fmt in TABLE_FORMATS:
            _,path=training_member(archive,manifest)
            names=heading(archive,path,fmt)
            target=target if target is not None else target_column(description(archive),names)
            if target not in names:
                raise ValueError('Declared training target is missing')
            _,rows=table(archive,path,fmt,[target])
            labels=np.asarray([row[0] for row in rows])
        elif fmt=='npy':
            _,path=training_member(archive,manifest,'target_path')
            labels=np.load(io.BytesIO(archive.read(path)),allow_pickle=False)
        elif fmt=='npz':
            _,path=training_member(archive,manifest)
            with np.load(io.BytesIO(archive.read(path)),allow_pickle=False) as bundle:
                labels=np.array(bundle[desc.get('target_key','y')])
        else:
            raise ValueError('No independently readable label vector in this format')
        stats = class_prior(labels, primary, manifest.get('positive_label'))
        return dict(stats, format=fmt, target=target, training_label_member=path,
                    prediction_member=expected_prediction, shared_table=getattr(archive,'shared_table',None))


def class_prior(labels, primary, positive=None):
    labels=_validate_labels(labels,len(labels))
    if not len(labels):
        raise ValueError('Training label evidence is empty')
    values,counts=np.unique(labels,return_counts=True)
    from .metric_declarations import MetricDeclarationError, metric_name, resolved_name
    if primary != 'unresolved' and metric_name(primary) == 'f1':
        try:
            primary = resolved_name(primary, values.tolist(), positive)
        except MetricDeclarationError:
            # Every F1 averaging scheme accepts class predictions; no ranking or score is claimed.
            primary = 'unresolved_f1'
    if len(values)<2 and primary=='roc_auc':
        raise ValueError('A single observed class cannot define ROC-AUC probability columns')
    if positive is not None and labels.dtype.kind in 'US':
        positive=str(positive)
    if positive is not None and (len(values)!=2 or positive not in values.tolist()):
        raise ValueError('Declared positive class conflicts with observed training labels')
    if positive is not None:
        order=[i for i,c in enumerate(values) if c!=positive]+[values.tolist().index(positive)]
        values,counts=values[order],counts[order]
    if primary=='binary_f1' and len(values)!=2:
        raise ValueError('Binary F1 needs two observed classes')
    return dict(labels=values.tolist(), counts=counts.tolist(), prior=(counts/counts.sum()).tolist(),
                majority=values[int(np.argmax(counts))].item(), metric=primary, training_rows=len(labels))


def record_input_failure(package,directory,logger,error,phase,*,api_usage=None,resolved_task=None):
    from .format_intake import DECODE_ERRORS
    reason=type(error).__name__+': '+str(error)
    outcome=dict(requested_planner_completed=False,contract_unresolved=True,phase=phase,reason=reason,
                 attempted_classifier_fits=0,completed_classifier_fits=0,unknown_failed_attempts=0,
                 new_api_requests=0,known_course_cost_rmb=0,formal_test_accessed=False,
                 external_score=None)
    from .metric_declarations import MetricDeclarationError
    if isinstance(error,MetricDeclarationError):
        outcome.update(reason_code='metric_unresolved', selection_metric=None)
    if api_usage:
        outcome.update(api_usage)
    try:
        if getattr(error,'verified_prior',None) is not None:
            evidence = error.verified_prior
        elif resolved_task is not None and resolved_task.spec.get('input_recipe'):
            spec = resolved_task.spec
            stats = class_prior(resolved_task.y_train, spec['metric'], spec.get('positive_label'))
            evidence = dict(stats, format='adaptive_records', target=spec['input_recipe']['target'], frozen_spec=spec)
        elif isinstance(error,MetricDeclarationError) and resolved_task is not None:
            task_spec = dict(resolved_task.spec)
            stats = class_prior(resolved_task.y_train,task_spec['metric'],task_spec.get('positive_label'))
            evidence = dict(stats, format='frozen_features', model_kind=resolved_task.kind, frozen_spec=task_spec,
                            target=task_spec.get('input_schema',{}).get('target'))
        else:
            evidence=prior_evidence(package,unresolved_metric=isinstance(error,MetricDeclarationError))
    except DECODE_ERRORS as second:
        logger.write_json('run_outcome.json',dict(outcome,status='failed',delivery_status='unavailable',
            model_available=False,partial_reason=str(second)))
        return 2
    logger.write_json('partial_model.json',evidence)
    logger.write_json('task_spec.json',dict(task_kind='partial_classification',metric=evidence['metric'],
        labels=evidence['labels'],input_format=evidence['format'],target=evidence['target']))
    logger.write_json('final_selection.json',dict(selected_experiment='training_class_prior',
        family='training_class_prior',delivery_status='partial',contract_unresolved=True,test_accessed=False,
        model_format='class_prior_v1',model_file='partial_model.json',model_sha256=sha(directory/'partial_model.json'),
        task_spec_sha256=sha(directory/'task_spec.json'),refit=False,training_rows=evidence['training_rows']))
    logger.write_json('run_outcome.json',dict(outcome,status='partial',model_available=True,delivery_status='partial',
        method='training_class_prior',training_rows=evidence['training_rows']))
    logger.event('partial_prior_frozen',reason=reason,observed_labels=evidence['labels'])
    return 0


def input_count(package,prior):
    fmt=prior['format']
    if fmt == 'frozen_features':
        from .data import load_prediction_features
        return len(load_prediction_features(package,prior['model_kind'],prior['frozen_spec']))
    if fmt == 'adaptive_records':
        from .format_intake import prediction_rows
        records = prediction_rows(package, prior['frozen_spec'])
        features = {c['name'] for c in prior['frozen_spec']['input_schema']['columns']}
        if not features or any(not features.intersection(row) for row in records):
            raise ValueError('Prediction rows have no verified input field in common with training')
        return len(records)
    with open_package(package,'test',contract=prior.get('shared_table'),
                      for_partial_labels=prior['metric'].startswith('unresolved')) as archive:
        manifest=read_manifest(archive,for_partial_labels=prior['metric'].startswith('unresolved'))
        if manifest is None:
            if fmt=='npy':
                path=find_member(archive,['X_test.npy','C_test.npy'],required=True)
                desc={'format':'npy','path':path}
            elif prior.get('prediction_member'):
                path=declared_member(archive,prior['prediction_member'])
                desc={'format':PurePosixPath(path).suffix[1:].lower(),'path':path}
            else:
                desc=descriptor(archive,fmt,'test')
            manifest={'_manifest_parent':'.','splits':{'test':desc}}
        desc,path=_member(archive,manifest,'test')
        fmt=desc.get('format')
        # A training target member must never masquerade as a prediction table.
        tokens=set(re.split(r'[^a-z0-9]+',PurePosixPath(path).name.lower()))
        if path==prior['training_label_member'] or tokens & {'labels','target','truth'} or 'y' in tokens and 'test' in tokens:
            raise ValueError('Prediction source names a label vector')
        fmt=desc.get('format')
        if fmt in TABLE_FORMATS:
            headers=heading(archive,path,fmt)
            declared=manifest.get('target',prior['target'])
            features=[h for h in headers if h!=declared]
            if not features:
                raise ValueError('Prediction table has no feature column')
            _,rows=table(archive,path,fmt,features[:1])
            return sum(1 for _ in rows)
        if fmt=='npy':
            from .legacy_array import load_array
            x=load_array(archive.read(path))
        elif fmt=='npz':
            with np.load(io.BytesIO(archive.read(path)),allow_pickle=False) as bundle:
                x=bundle[desc.get('features_key','X')]
        else:
            raise ValueError('Unknown feature format for partial prediction')
        if x.ndim<2:
            raise ValueError('Expected a feature matrix with a row dimension')
        return len(x)


def predict(package,directory,output,selection):
    path=directory/selection['model_file']
    if sha(path)!=selection['model_sha256']:
        raise ValueError('Partial model was modified after selection')
    prior=json.loads(path.read_text(encoding='utf-8'))
    count=input_count(package,prior)
    if not count:
        raise ValueError('The number of prediction rows is zero')
    scores=prior['metric']=='roc_auc'
    if scores:
        MetricSpec('roc_auc',tuple(prior['labels']))
    output=Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with atomic_text(output,newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['row_id','prediction']+([f'score_{i}' for i in range(len(prior['prior']))] if scores else []))
        writer.writerows([i,prior['majority'],*(prior['prior'] if scores else [])] for i in range(count))
    receipt=dict(selected_experiment='training_class_prior',delivery_status='partial',
                 contract_unresolved=selection.get('contract_unresolved',True),partial_reason=selection.get('partial_reason'),refit=False,
                 prediction_count=count,test_labels_read=False,model_sha256=selection['model_sha256'],
                 score_class_order=prior['labels'] if scores else None,prediction_sha256=sha(output),external_score=None)
    RunLogger(output.parent).write_json('prediction_manifest.json',receipt)
    return output
