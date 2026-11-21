"""Check public input and split portability across two Python environments, without fitting."""
import argparse
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import struct
from unittest.mock import patch
import zipfile

import numpy as np
from superagent.data import TrainingTask, load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import grouped_partition, overlap_audit, take


ROOT = Path(__file__).resolve().parents[1]
FILES = {'bean':'dry_bean_numeric_tabular.zip','sms':'sms_spam_single_text.zip'}
SEEDS = {'primary':707, 'confirmation_1':808, 'confirmation_2':909}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def creator_probe(path):
    # The OS byte is part of the standard 46-byte central directory header. Leave
    # the compressed bytes (including held-out members) and local headers alone.
    raw = bytearray(path.read_bytes())
    original = sha(raw)
    with zipfile.ZipFile(path) as archive:
        offset = archive.start_dir
        offsets, creators = [], []
        for info in archive.infolist():
            if raw[offset:offset+4] != b'PK\x01\x02' or raw[offset+5] != info.create_system:
                raise ValueError('Unexpected ZIP central directory layout')
            creators.append(info.create_system)
            offsets.append(offset+5)
            offset += 46 + sum(struct.unpack_from('<HHH', raw, offset+28))
    for pos in offsets:
        raw[pos] = 0
    return dict(original_sha256=original, creator_bytes=creators, with_windows_creator_sha256=sha(raw))


def portable_rows(rows, labels):
    stream = hashlib.sha256()
    for row, label in zip(rows, labels, strict=True):
        value = row.tolist() if isinstance(row,np.ndarray) else row
        scalar = label.item() if isinstance(label,np.generic) else label
        stream.update(json.dumps([value,scalar], ensure_ascii=False, allow_nan=False, separators=(',',':')).encode() + b'\n')
    return stream.hexdigest()


def audited_partition(task, fit, held):
    indices = list(fit)+list(held)
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(task.y_train))):
        raise ValueError('Registered row IDs must be disjoint and cover the entire parent input')
    if overlap_audit(task,fit,held)['search_duplicate_rows'] != 0:
        raise ValueError('Duplicate feature row occurs on both sides')
    labels = set(task.y_train.tolist())
    if set(task.y_train[fit].tolist()) != labels or set(task.y_train[held].tolist()) != labels:
        raise ValueError('One registered partition lost a training class')
    return dict(fit_rows=len(fit), held_rows=len(held), held_ids_sha256=fingerprint(held),
        fit_input_sha256=portable_rows(take(task.X_train,fit),task.y_train[fit]),
        held_input_sha256=portable_rows(take(task.X_train,held),task.y_train[held]))


def assess(path, name, a, b):
    packet = creator_probe(path)
    packet['a_sha256'] = a['task']['source']['package_sha256']
    packet['matches_a_after_creator_change'] = packet['with_windows_creator_sha256'] == packet['a_sha256']
    expected = b['tasks'][name+'_expanded']
    if packet['original_sha256'] != expected['source']['package_sha256']:
        raise ValueError('Input package differs from registered B bytes')
    opened = []
    real_read = zipfile.ZipFile.read
    def train_only(archive, member, *args, **kwargs):
        member_name = member.filename if isinstance(member,zipfile.ZipInfo) else member
        if member_name not in ('manifest.json','train.csv'):
            raise ValueError('Attempted to open a non-training member')
        opened.append(member_name)
        return real_read(archive, member, *args, **kwargs)
    with patch.object(zipfile.ZipFile,'read',train_only):
        task = load_training_task(path)
    frozen_bytes = gzip.decompress((ROOT/'research/bq_01_portfolios'/name/'row_ids.json.gz').read_bytes())
    if sha(frozen_bytes) != expected['row_ids_sha256']:
        raise ValueError('Portable row record differs from B preregistration')
    ids = json.loads(frozen_bytes)
    train, held, _ = grouped_partition(task,45101)
    a_export = read_json(ROOT/'research/a10_real_task_pilot'/name/'export_verification.json')
    fresh = dict(outer_held_rows=len(held), matches_b_outer_ids=held.tolist()==ids['outer']['held'],
                 matches_a_outer_ids=sha(json.dumps(held.tolist()).encode()) == a_export['original_row_ids_sha256'], inner={})
    subset = TrainingTask(task.kind,take(task.X_train,train),task.y_train[train],task.spec)
    for split, seed in SEEDS.items():
        _, inner, _ = grouped_partition(subset,seed)
        p = dict(row_ids_sha256=fingerprint(inner.tolist()), label_sha256=fingerprint(subset.y_train[inner]))
        fresh['inner'][split] = dict(**p, matches_b=all(expected['splits'][split][k]==v for k,v in p.items()),
                                     matches_a=all(a['splits'][split][k]==v for k,v in p.items()))
    fit, held = ids['outer']['train'], ids['outer']['held']
    imported = dict(outer=audited_partition(task,fit,held), inner={})
    registered = TrainingTask(task.kind,take(task.X_train,fit),task.y_train[fit],task.spec)
    for split in SEEDS:
        part = ids['inner'][split]
        imported['inner'][split] = audited_partition(registered,part['train_indices'],part['search_indices'])
    return dict(training_members_opened=opened, archive=packet, source_rows=len(task.y_train),
                source_content_sha256=portable_rows(task.X_train,task.y_train), fresh=fresh, imported=imported)


def run(directory, output):
    output.mkdir(parents=True, exist_ok=False)
    a = {t['name']:t for t in read_json(ROOT/'research/a10_real_task_pilot/protocol.json')['tasks']}
    b = read_json(ROOT/'research/bq_01_portfolios/protocol.json')
    save(output/'protocol.json',dict(batch='BP-01', python=platform.python_version(), system=platform.system(),
        versions={p:version(p) for p in ('scikit-learn','numpy','scipy','joblib')},
        source_sha256=sha(Path(__file__).read_bytes()), split_code_sha256=sha((ROOT/'superagent/splits.py').read_bytes()),
        plan_sha256=sha((ROOT/'docs/batches/BP_01_plan.md').read_bytes()), classifier_fit_limit=0, prediction_limit=0, api_limit=0))
    try:
        results = {name:assess(directory/file,name,a[name],b) for name,file in FILES.items()}
        save(output/'results.json',results)
        save(output/'completion.json',dict(status='checked', tasks=len(results), fits=0, model_predictions=0, outer_model_scores=0, api_requests=0))
    except Exception as error:
        save(output/'completion.json',dict(status='failed', error=str(error), fits=0, api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.packages,args.output)
