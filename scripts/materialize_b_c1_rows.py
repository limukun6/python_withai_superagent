"""Materialize and verify B's registered C1 datasets without environment-dependent resampling."""
import argparse
import gzip
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile

import joblib
import numpy as np

from scripts import run_b1_ab_acceptance as fixture, run_b1_05_pair_pilot as pilot
from scripts.run_bq_01_portfolios import pack
from superagent import ab_run
from superagent.contracts import TaskSpec
from superagent.data import TrainingTask, load_development_task, load_prediction_features, load_training_task
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import effective_grouping, sample_groups, take
from tools.ab_registered_rows import content_digest, registered_partitioner, validated_ids


REFERENCE = fixture.ROOT/'research/bc1_preparation'


def stored_zip(destination, entries):
    with zipfile.ZipFile(destination, 'w' if hasattr(destination,'write') else 'x') as archive:
        for name, data in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_STORED)


def canonical_npz(data):
    members = []
    with np.load(io.BytesIO(data), allow_pickle=False) as bundle:
        for name in bundle.files:
            array = bundle[name]
            array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder('<'), copy=False))
            out = io.BytesIO()
            np.lib.format.write_array(out, array, version=(1,0), allow_pickle=False)
            members.append((name+'.npy',out.getvalue()))
    output = io.BytesIO()
    stored_zip(output,members)
    return output.getvalue()


def packet(destination, task, indices, split):
    x = take(task.X_train, indices)
    y = task.y_train[indices] if split=='train' else None
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        raw = Path(tmp)/'packet.zip'
        if task.kind=='pair_text':
            pilot.pair_package(raw,x,y,'preference',split)
        else:
            pack(raw,task.kind,x,y,'accuracy',split)
        with zipfile.ZipFile(raw) as archive:
            entries = [(name,canonical_npz(archive.read(name)) if name.endswith('.npz') else archive.read(name))
                       for name in archive.namelist()]
        stored_zip(destination, entries)
        if split=='train' and TaskSpec.from_task(load_development_task(raw)) != TaskSpec.from_task(load_development_task(destination)):
            raise ValueError('Canonical container changed the training contract')


def load_original(name, file, expected):
    if name=='headlines':
        x,y,_,meta = pilot.pair_source(name,file)
        if meta['training_member_sha256'] != expected['training_member_sha256']:
            raise ValueError('Original headline training member changed')
        return TrainingTask('pair_text',x,y), dict(training_csv_sha256=meta['training_member_sha256'])
    with zipfile.ZipFile(file) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        member = manifest['splits']['train']
        if member['format']!='csv' or member['path']!='train.csv' or manifest['modality']!='tabular':
            raise ValueError('Expected the public numeric training CSV')
        csv_digest = fixture.sha(archive.read('train.csv'))
    task = load_training_task(file)
    source_digest = content_digest(task.X_train,task.y_train)
    audited = fixture.read(fixture.ROOT/'research/bp_01_split_transport/environment_1/results.json')['bean']
    if task.kind!='numeric_tabular' or source_digest!=audited['source_content_sha256']:
        raise ValueError('The supplied numeric training examples or their order differ')
    return task, dict(training_csv_sha256=csv_digest, csv_matches_b=csv_digest==expected['training_member_sha256'],
                      source_content_sha256=source_digest)


def check_rejections(task, manifest, digest):
    rotated = np.roll(np.arange(len(task.y_train)),1)
    shifted = TrainingTask(task.kind,take(task.X_train,rotated),task.y_train[rotated],task.spec)
    probes = ((shifted,digest,'Training content'),(task,'0'*64,'manifest differs'))
    for item, expected, reason in probes:
        try:
            registered_partitioner(item,manifest,expected)
        except ValueError as error:
            if reason not in str(error):
                raise
        else:
            raise ValueError('Invalid partition input unexpectedly passed')


def materialize(root, name, file, expected):
    task, source = load_original(name,file,expected['source'])
    id_bytes = gzip.decompress((REFERENCE/name/'row_ids.json.gz').read_bytes())
    if fixture.sha(id_bytes) != expected['row_ids_sha256']:
        raise ValueError('Source row identities differ from B preregistration')
    ids = json.loads(id_bytes)
    fit, held = ids['outer']['train'], ids['outer']['held']
    outer = dict(train_indices=fit,search_indices=held,row_ids_sha256=fingerprint(held),label_sha256=fingerprint(task.y_train[held]))
    train, hold = validated_ids(outer, task.y_train, sample_groups(task,'shared_text'))
    optimize, delivery = root/name/'optimizer', root/name/'delivery'
    optimize.mkdir(parents=True)
    delivery.mkdir()
    training, features = optimize/'train.zip', delivery/'features.zip'
    packet(training,task,train,'train')
    packet(features,task,hold,'test')
    typed = load_development_task(training)
    query = load_prediction_features(features,typed.kind,typed.spec)
    training_digest = content_digest(take(task.X_train,train),task.y_train[train])
    held_digest = content_digest(take(task.X_train,hold),task.y_train[hold])
    if (typed.X_val is not None or typed.y_val is not None or content_digest(typed.X_train,typed.y_train)!=training_digest or
            content_digest(query,task.y_train[hold])!=held_digest):
        raise ValueError('Materialized data or row order differs from the registered source')
    splits = []
    for split, row in expected['splits'].items():
        splits.append(dict(id=split,seed=row['seed'],row_ids_sha256=row['row_ids_sha256'],label_sha256=row['label_sha256'],
                           **ids['inner'][split]))
    manifest = optimize/'inner_splits.json'
    fixture.write(manifest,dict(version=1,kind=typed.kind,training_content_sha256=training_digest,
        grouping=effective_grouping(typed.kind,'shared_text'),splits=splits))
    digest = fixture.sha(manifest.read_bytes())
    split_source = registered_partitioner(typed,manifest,digest)
    worker_dir = root/name/'a_worker_inputs'
    worker_dir.mkdir()
    with patch.object(ab_run,'grouped_partition',split_source):
        trusted, allowed = ab_run.freeze_partitions(typed,worker_dir,707,MetricSpec.for_task(typed))
    checks = {}
    for split, reg in expected['splits'].items():
        payload = worker_dir/(split+'.joblib')
        workers = joblib.load(payload)
        fit_ids,held_ids = (ids['inner'][split][k] for k in ('train_indices','search_indices'))
        if (allowed[split]['evaluation_payload_sha256']!=fixture.sha(payload.read_bytes()) or
                trusted[split]['row_ids']!=held_ids or allowed[split]['row_ids_sha256']!=reg['row_ids_sha256'] or
                allowed[split]['label_sha256']!=reg['label_sha256']):
            raise ValueError('A worker file does not match local payload and registered row identities')
        logical = dict(training=content_digest(workers['X_train'],workers['y_train']),
                       evaluation=content_digest(workers['X_search'],workers['y_search']))
        if (logical['training']!=content_digest(take(typed.X_train,fit_ids),typed.y_train[fit_ids]) or
                logical['evaluation']!=content_digest(take(typed.X_train,held_ids),typed.y_train[held_ids])):
            raise ValueError('Worker payload examples differ from the source rows')
        checks[split] = dict(logical=logical, local_worker_sha256=allowed[split]['evaluation_payload_sha256'])
    check_rejections(typed,manifest,digest)
    return dict(source=source, canonical_files={'train.zip':fixture.sha(training.read_bytes()),'features.zip':fixture.sha(features.read_bytes()),
        'inner_splits.json':digest}, training_rows=len(train), held_rows=len(hold), training_content_sha256=training_digest,
        held_content_sha256=held_digest, rows_sha256=expected['row_ids_sha256'], splits=checks,
        changed_row_order_rejected=True, changed_manifest_digest_rejected=True)


def run(args):
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    fixture.offline(root)
    reference=fixture.read(REFERENCE/'results.json')['tasks']
    fixture.write(root/'protocol.json',dict(batch='BP-02',driver_sha256=fixture.sha(Path(__file__).read_bytes()),
        helper_sha256=fixture.sha((fixture.ROOT/'tools/ab_registered_rows.py').read_bytes()), environment=pilot.environment(),
        a_freeze_implementation_sha256=fixture.sha(Path(ab_run.__file__).read_bytes()), reference_sha256=fixture.sha((REFERENCE/'results.json').read_bytes()),
        plan_sha256=fixture.sha((fixture.ROOT/'docs/batches/BP_02_plan.md').read_bytes()), fits_allowed=0, model_api_allowed=0))
    results={}
    try:
        for task in ('headlines','bean'):
            results[task] = materialize(root,task,getattr(args,task),reference[task])
            fixture.write(root/'results.json',results)
        fixture.write(root/'completion.json',dict(status='passed', tasks=2, actual_worker_payloads=6, fits=0,
                      model_predictions=0, outer_model_scores=0, model_api_requests=0))
    except Exception as error:
        fixture.write(root/'completion.json',dict(status='failed', reason=str(error), fits=0, model_api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('headlines','bean','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
