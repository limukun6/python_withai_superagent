"""Prepare SICK TRAIN directed entailment and measure dependence under registered and native partitions."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np

from scripts import run_b1_ab_acceptance as rec
from scripts.build_external_tasks import write_member
from superagent.ab_catalog import capabilities_for
from superagent.contracts import TaskSpec
from superagent.data import TrainingTask, load_development_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import grouped_partition, normalized_text, sample_groups
from tools.ab_registered_rows import content_digest, registered_partitioner


SOURCE_SHA = 'da272af924399fc2de4889ed55383e38822c10d16c4a949a89d597d09a0b7912'
SOURCE_MD5 = 'b03210036cc64a4cf1bc7a0525357001'
LABELS = {'A_entails_B':'entails','A_neutral_B':'neutral','A_contradicts_B':'contradicts'}
INNER = (('primary',707),('confirmation_1',808),('confirmation_2',909))


def read_train(path):
    raw = path.read_bytes()
    if rec.sha(raw)!=SOURCE_SHA or hashlib.md5(raw).hexdigest()!=SOURCE_MD5:
        raise ValueError('SICK publisher archive changed')
    records = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive, archive.open('SICK.txt') as file:
        names = file.readline().decode('utf-8-sig').rstrip('\r\n').split('\t')
        if names[-1]!='SemEval_set':
            raise ValueError('Split cannot be checked independently of other fields')
        fields = ['pair_ID','sentence_A','sentence_B','entailment_AB','sentence_A_original','sentence_B_original']
        positions = [names.index(k) for k in fields]
        for line in file:
            line = line.rstrip(b'\r\n')
            if line.rsplit(b'\t',1)[-1]!=b'TRAIN':
                continue
            cells = line.split(b'\t')
            key,a,b,label,origin_a,origin_b = (cells[i].decode('utf-8') for i in positions)
            records.append((int(key),a,b,LABELS[label],origin_a,origin_b))
    if len(records)!=4439 or len({r[0] for r in records})!=len(records):
        raise ValueError('Publisher training count or IDs differ')
    return records


def families(records):
    pairs = [(r[1],r[2]) for r in records]
    edges = list(pairs)
    for _,a,b,_,origin_a,origin_b in records:
        edges.extend((text,origin) for text,origin in ((a,origin_a),(b,origin_b)) if normalized_text(origin))
    return sample_groups(SimpleNamespace(kind='pair_text',X_train=edges),'shared_text')[:len(pairs)]


def pack(path, records, split):
    columns = ['left','right']
    stream = io.StringIO(newline='')
    writer = csv.writer(stream,lineterminator='\n')
    training = split=='train'
    writer.writerow(columns+(['target'] if training else []))
    for r in records:
        writer.writerow([r[1],r[2]]+([r[3]] if training else []))
    schema = dict(schema_version=1,objective='classification',modality='pair_text',metric='accuracy',
                  pair_semantics='relation',columns=[dict(name=k,type='text') for k in columns],
                  target='target',splits={split:dict(format='csv',path='input.csv')})
    with zipfile.ZipFile(path,'x') as archive:
        write_member(archive,'manifest.json',json.dumps(schema,sort_keys=True).encode())
        write_member(archive,'input.csv',stream.getvalue().encode())


def partition(groups, targets, seed):
    carrier = TrainingTask('single_text',groups.tolist(),targets)
    train,held,_ = grouped_partition(carrier,seed)
    return train,held


def overlap(records, groups, train, held):
    known = {normalized_text(text) for i in train for text in records[i][1:3]}
    known_pairs = {(records[i][1],records[i][2]) for i in train}
    seen = set(groups[train])
    return dict(train_rows=len(train),held_rows=len(held),train_families=len(seen),
                held_families=len(set(groups[held])),families_shared=len(seen.intersection(groups[held])),
                held_rows_in_training_families=int(np.isin(groups[held],list(seen)).sum()),
                held_rows_sharing_a_visible_sentence=sum(any(normalized_text(t) in known for t in records[i][1:3]) for i in held),
                repeated_ordered_pairs=sum((records[i][1],records[i][2]) in known_pairs for i in held))


def run(source, output):
    output.mkdir(parents=True,exist_ok=False)
    rec.offline(output)
    rec.write(output/'protocol.json',dict(batch='BR-01',dataset='Marelli et al. SICK',source_sha256=SOURCE_SHA,
        source_md5=SOURCE_MD5,publisher='https://zenodo.org/records/2787612',split='TRAIN',target='entailment_AB',
        inputs=['sentence_A','sentence_B'],group_only=['sentence_A_original','sentence_B_original'],
        outer_seed=45101,inner_seeds=dict(INNER),driver_sha256=rec.sha(Path(__file__).read_bytes()),
        plan_sha256=rec.sha((rec.ROOT/'docs/batches/BR_01_relation_plan.md').read_bytes()),fit_limit=0,api_limit=0))
    try:
        records = read_train(source)
        targets = np.array([r[3] for r in records])
        groups = families(records)
        train,held = partition(groups,targets,45101)
        outside = overlap(records,groups,train,held)
        train_rows,held_rows = ([records[i] for i in ids] for ids in (train,held))
        for name in ('optimizer','delivery'):
            (output/name).mkdir()
        train_zip, query = output/'optimizer/train.zip',output/'delivery/features.zip'
        pack(train_zip,train_rows,'train')
        pack(query,held_rows,'test')
        data = load_development_task(train_zip)
        contract = TaskSpec.from_task(data).record()
        menu = capabilities_for(contract,train_rows=len(train))
        if (data.kind!='text_tabular' or data.X_val is not None or data.X_train.columns!=('left','right')
                or data.X_train.text_columns!=(0,1) or not np.array_equal(data.y_train,targets[train])
                or contract['input_schema'].get('pair_semantics')!='relation' or not menu
                or any(c['swap_label_rule']!='not_applicable' for c in menu.values())):
            raise ValueError('Wrong ordered-relation task, feature columns or candidate rules')
        manifest = dict(version=1,kind=data.kind,grouping='exact_rows',
                        training_content_sha256=content_digest(data.X_train,data.y_train),splits=[])
        audits = {}
        for split,seed in INNER:
            fit,valid = partition(groups[train],targets[train],seed)
            native_fit,native_valid,_ = grouped_partition(data,seed,'shared_text')
            audits[split] = dict(native=overlap(train_rows,groups[train],native_fit,native_valid),
                                 registered=overlap(train_rows,groups[train],fit,valid))
            manifest['splits'].append(dict(id=split,seed=seed,train_indices=fit.tolist(),search_indices=valid.tolist(),
                row_ids_sha256=fingerprint(valid.tolist()),label_sha256=fingerprint(data.y_train[valid])))
        if any(item['families_shared'] for item in (outside,*(x['registered'] for x in audits.values()))):
            raise ValueError('A connected source family crosses a registered boundary')
        split_file = output/'optimizer/inner.json'
        rec.write(split_file,manifest)
        signed = rec.sha(split_file.read_bytes())
        checker = registered_partitioner(data,split_file,signed)
        for row in manifest['splits']:
            fit,valid,_ = checker(data,row['seed'],'shared_text')
            if fit.tolist()!=row['train_indices'] or valid.tolist()!=row['search_indices']:
                raise ValueError('Existing row-registration adapter changed the registered split')
        ids = {'train':[r[0] for r in train_rows],'held':[r[0] for r in held_rows]}
        raw_ids = json.dumps(ids,separators=(',',':')).encode()
        (output/'source_ids.json.gz').write_bytes(gzip.compress(raw_ids,mtime=0))
        result = dict(rows=len(records),source_families=len(set(groups)),outer=outside,inner=audits,
            missing_original_caption_cells=sum(not normalized_text(o) for r in records for o in r[4:6]),
            train_classes={k:int((data.y_train==k).sum()) for k in sorted(set(targets))},kind=data.kind,
            optimizer_columns=list(data.X_train.columns),menu=menu,signed_split_sha256=signed,
            train_sha256=rec.sha(train_zip.read_bytes()),features_sha256=rec.sha(query.read_bytes()),
            outer_source_ids_sha256=rec.sha(raw_ids),held_content_sha256=content_digest([r[1:3] for r in held_rows],targets[held]))
        rec.write(output/'results.json',result)
        rec.write(output/'completion.json',dict(status='prepared',source_training_rows=len(records),
                  registered_internal_splits=3,model_fits=0,model_predictions=0,model_api_requests=0,performance_scores=0))
    except Exception as error:
        rec.write(output/'completion.json',dict(status='failed',reason=str(error),model_fits=0,model_api_requests=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    run(args.source.resolve(),args.output.resolve())
