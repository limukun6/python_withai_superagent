"""Run the official original/expanded catalogues on sealed, original-format Released training subsets."""
import argparse
import csv
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch
import zipfile

import numpy as np

from scripts import run_b1_ab_acceptance as rec, run_b1_05_pair_pilot as pilot
from scripts.materialize_b_c1_rows import stored_zip
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.ab_sentence import verified_directory
from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_training_task, load_prediction_features
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import effective_grouping, sample_groups, take
from tools.ab_registered_rows import content_digest, registered_partitioner, validated_ids


TASKS = {'coupon':'coupon', 'headlines':'preference'}
ORDER = [('coupon','baseline'),('coupon','full'),('headlines','full'),('headlines','baseline')]
Q = rec.ROOT/'research/bq_01_portfolios'


def write_packet(path, name, x, y=None, readme=None):
    split = 'train' if y is not None else 'test'
    members = [readme] if readme else []
    if name == 'coupon':
        arrays = [('C',x),('y',y)] if y is not None else [('C',x)]
        for prefix, array in arrays:
            stream = io.BytesIO()
            np.save(stream,array,allow_pickle=array.dtype.hasobject)
            members.append((f'table/{prefix}_{split}.npy',stream.getvalue()))
    else:
        stream = io.StringIO(newline='')
        writer = csv.writer(stream,lineterminator='\n')
        writer.writerow(['headline_A','headline_B'] + (['label_pairwise'] if y is not None else []))
        writer.writerows(((a,b,y[i]) if y is not None else (a,b)) for i,(a,b) in enumerate(x))
        members.append((f'titles/{split}.csv',stream.getvalue().encode()))
    stored_zip(path,members)


def prepare(name, staged, folder, course, q):
    folder.mkdir()
    if ab_run.sha(staged)!=course['staged_sha256']:
        raise ValueError('Training-only original-package copy differs: '+name)
    original = load_training_task(staged)
    raw_ids = gzip.decompress((Q/name/'row_ids.json.gz').read_bytes())
    ref = q['tasks'][name+'_original']
    if rec.sha(raw_ids)!=ref['row_ids_sha256']:
        raise ValueError('Previously registered outer/inner row IDs changed')
    ids = json.loads(raw_ids)
    hold = ids['outer']['held']
    outer = dict(train_indices=ids['outer']['train'],search_indices=hold,
                 row_ids_sha256=fingerprint(hold),label_sha256=fingerprint(original.y_train[hold]))
    train, held = validated_ids(outer,original.y_train,sample_groups(original,'shared_text'))
    source = original.spec['task_description_source']
    with zipfile.ZipFile(staged) as archive:
        readme = archive.read(source['member'])
    if rec.sha(readme)!=source['sha256']:
        raise ValueError('Training README changed')
    write_packet(folder/'train.zip',name,take(original.X_train,train),original.y_train[train],(source['member'],readme))
    write_packet(folder/'features.zip',name,take(original.X_train,held))
    task = load_development_task(folder/'train.zip')
    contract = TaskSpec.from_task(task).record()
    features = load_prediction_features(folder/'features.zip',task.kind,task.spec)
    training = content_digest(take(original.X_train,train),original.y_train[train])
    if (task.spec.get('task_description_source')!=source or task.X_val is not None or task.y_val is not None
            or training!=content_digest(task.X_train,task.y_train)
            or content_digest(features,original.y_train[held])!=content_digest(take(original.X_train,held),original.y_train[held])
            or contract['goal_resolution']['fields']['primary_metric']['source']!='declared_task_readme'):
        raise ValueError('Task meaning, raw training values or prediction row order changed')
    inner = folder/'inner.json'
    splits = [dict(id=s,seed=row['seed'],row_ids_sha256=row['row_ids_sha256'],label_sha256=row['label_sha256'],
                   **ids['inner'][s]) for s,row in ref['splits'].items()]
    rec.write(inner,dict(version=1,kind=task.kind,grouping=effective_grouping(task.kind,'shared_text'),
                         training_content_sha256=training,splits=splits))
    registered = folder/'registered'
    registered.mkdir()
    partition = registered_partitioner(task,inner,ab_run.sha(inner))
    with patch.object(ab_run,'grouped_partition',partition):
        _, actual = ab_run.freeze_partitions(task,registered,707,MetricSpec.for_task(task))
    catalogs = {}
    for scope, previous in [('baseline','original'),('full','expanded')]:
        caps = ab_run.registered_capabilities(contract,len(task.y_train),scope)
        old = q['tasks'][name+'_'+previous]
        if list(caps)!=old['candidate_ids'] or rec.digest(caps)!=old['catalog_sha256']:
            raise ValueError('Compared portfolios changed beyond the original registration: '+name+' '+scope)
        catalogs[scope]=dict(ids=list(caps),sha256=rec.digest(caps))
    return dict(name=name,training_rows=len(train),held_rows=len(held),outer_ids_sha256=ref['row_ids_sha256'],
                raw_training_content=training,held_content=content_digest(features,original.y_train[held]),
                train_sha256=ab_run.sha(folder/'train.zip'),features_sha256=ab_run.sha(folder/'features.zip'),
                inner_sha256=ab_run.sha(inner),splits=actual,catalogs=catalogs,task_contract=contract,
                readme_sha256=source['sha256'],source_staged_sha256=course['staged_sha256'])


def export_frozen(case, data, expected_rows):
    official, relocated = case/'run', case/'relocated'
    selection = rec.read(official/'final_selection.json')
    relocated.mkdir()
    for name in ('task_spec.json','final_selection.json',selection['model_file']):
        shutil.copy2(official/name,relocated/name)
    digests = []
    for i, source in enumerate((official,relocated)):
        path = case/f'export_{i}/rows.csv'
        pilot.child(['-m','superagent','predict',data,'--run-dir',source,'--output',path],case/f'predict_{i}.log',300)
        read_export(path,expected_rows)
        digests.append(ab_run.sha(path))
    if digests[0]!=digests[1]:
        raise ValueError('Native and relocated predictions disagree')
    return digests[0]


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    rec.offline(root)
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()),SUPERAGENT_OFFLINE='1',
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    source = rec.digest(ab_run.sources())
    q = rec.read(Q/'protocol.json')
    audit = rec.read(rec.ROOT/'research/ba19_released_inputs/results.json')['course_inputs']
    plan = rec.ROOT/'docs/batches/BA19_03_released_delivery_plan.md'
    rec.write(root/'protocol.json',dict(batch='BA19-03',source=source,runner_sha256=ab_run.sha(Path(__file__)),
        plan_sha256=ab_run.sha(plan),Q_reference=ab_run.sha(Q/'protocol.json'),order=ORDER,seed=707,fit_limit=11,
        batch_fit_limit=44,wall=1200,worker=300,api_limit=0,environment=pilot.environment()))
    started = time.monotonic()
    paths = {}
    try:
        if verified_directory() is None:
            raise ValueError('The registered, local MiniLM is required for catalogue parity')
        packets = {n:prepare(n,args.staged/t/'train_only.zip',root/n,audit[t],q) for n,t in TASKS.items()}
        rec.write(root/'preflight.json',packets)
        for name,scope in ORDER:
            p, data = packets[name], root/name
            key = name+'_'+scope
            case = root/key
            case.mkdir()
            shutil.copy2(data/'train.zip',case/'train.zip')
            shutil.copytree(data/'registered',case/'registered')
            exe = ('-m','tools.ab_registered_run','--split-manifest',data/'inner.json',
                   '--split-sha256',p['inner_sha256'],'--','run','--catalog-scope',scope)
            entry = dict(name=key,files={'train.zip':p['train_sha256']},splits=p['splits'],
                         catalog_sha256=p['catalogs'][scope]['sha256'])
            before = len(rec.trace(root))
            result = pilot.train_one(case,entry,source,launcher=exe)
            receipt = rec.read(case/'run/run_manifest.json')
            if (receipt['catalog_scope']!=scope or list(receipt['capabilities'])!=p['catalogs'][scope]['ids']
                    or receipt['task_contract']!=p['task_contract'] or result['fits']!=len(rec.trace(root))-before):
                raise ValueError('Worker count, task declaration or actual menu differs')
            paths[key]=result
            rec.write(root/'checkpoint.json',paths)
        rec.write(root/'all_models_frozen.json',{k:v['model_sha256'] for k,v in paths.items()})
        for name,scope in ORDER:
            key = name+'_'+scope
            p = packets[name]
            paths[key]['csv_sha256']=export_frozen(root/key,root/name/'features.zip',p['held_rows'])
        comparisons = {}
        for name,t in TASKS.items():
            p=packets[name]
            original = load_training_task(args.staged/t/'train_only.zip')
            order=json.loads(gzip.decompress((Q/name/'row_ids.json.gz').read_bytes()))['outer']
            y=original.y_train[order['held']]
            feature_rows=take(original.X_train,order['held'])
            if content_digest(feature_rows,y)!=p['held_content']:
                raise ValueError('Grading inputs differ from the registered held rows')
            metric=MetricSpec.for_task(original)
            correct={}
            for scope in ('baseline','full'):
                key=name+'_'+scope
                csv_path=root/key/'export_0/rows.csv'
                if ab_run.sha(csv_path)!=paths[key]['csv_sha256']:
                    raise ValueError('Frozen predictions changed before grading')
                pred=np.asarray(read_export(csv_path,len(y)),dtype=y.dtype)
                correct[scope]=pred==y
                paths[key].update(correct=int(correct[scope].sum()),rows=len(y),scores=metric.evaluate(y,pred))
            a,b=correct['baseline'],correct['full']
            comparisons[name]=dict(full_only_correct=int((~a & b).sum()),baseline_only_correct=int((a & ~b).sum()),
                delta_accuracy_pp=100*(float(b.mean())-float(a.mean())))
        if rec.digest(ab_run.sources())!=source or sum(p['fits'] for p in paths.values())!=len(rec.trace(root)):
            raise ValueError('Production source or batch accounting changed')
        rec.write(root/'results.json',dict(paths=paths,comparisons=comparisons))
        rec.write(root/'completion.json',dict(status='passed',full_autonomous_runs=4,actual_fits=len(rec.trace(root)),
            fresh_prediction_processes=8,new_course_validation_or_test_scores=0,model_api_requests=0,course_rmb=0,
            wall_seconds=time.monotonic()-started))
    except Exception as error:
        rec.write(root/'completion.json',dict(status='failed',error=str(error),completed_paths=list(paths),
            actual_fits=len(rec.trace(root)),model_api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('staged','weights','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    run(parser.parse_args())
