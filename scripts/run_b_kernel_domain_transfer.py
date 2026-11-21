"""Freeze six independently trained text portfolios before grading held-out review domains."""
import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch
import zipfile

import numpy as np

from scripts import run_b1_ab_acceptance as record, run_b1_05_pair_pilot as pilot
from scripts.run_b_a19_released_delivery import export_frozen
from scripts.run_bq_01_portfolios import pack
from superagent import ab_run
from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_prediction_features
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import effective_grouping, grouped_partition, normalized_text
from tools.ab_registered_rows import content_digest, registered_partitioner


DATASET_URL = 'https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences'
ARCHIVE_SHA = 'afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343'
MEMBERS = {
    'amazon': ('amazon_cells_labelled.txt', '47003fc0a0d4840b00e96e715b6189bad09e7443a3da41c4cbe12ffc79f86ae3'),
    'imdb': ('imdb_labelled.txt', 'aef2e49e3da25714d61175e3a6e68eeef74a20a2f914318dc3be9947ea86512d'),
    'yelp': ('yelp_labelled.txt', 'c76468b7b5c6e56a0804d728345c5f84aa2142ddb214420f61cc9cfd4c00d2ea'),
}
ORDER = (('amazon','original'), ('amazon','extended'), ('imdb','extended'), ('imdb','original'),
         ('yelp','original'), ('yelp','extended'))
PLAN = 'docs/batches/B_TEXT_KERNEL_05_domain_transfer_plan.md'


def read_original(path):
    if ab_run.sha(path) != ARCHIVE_SHA:
        raise ValueError('Public source archive changed')
    domains = {}
    with zipfile.ZipFile(path) as archive:
        for domain, (name, digest) in MEMBERS.items():
            raw = archive.read('sentiment labelled sentences/' + name)
            if record.sha(raw) != digest:
                raise ValueError('Unexpected public data member')
            # The two U+0085 characters in IMDb are part of sentences. UCI uses LF records.
            lines = raw.decode('utf-8-sig').removesuffix('\n').split('\n')
            values = [line.rsplit('\t', 1) for line in lines]
            if (len(values) != 1000 or any(len(row) != 2 or not row[0].strip() for row in values)
                    or Counter(row[1] for row in values) != {'0': 500, '1': 500}):
                raise ValueError('UCI row or class counts changed')
            domains[domain] = [(text, 'positive' if label == '1' else 'negative') for text, label in values]
    return domains


def register(folder, domains, held_domain, extra):
    folder.mkdir()
    held_x, held_y = zip(*domains[held_domain])
    held_groups = {normalized_text(x) for x in held_x}
    available = [row for domain, rows in domains.items() if domain != held_domain for row in rows]
    filtered = [row for row in available if normalized_text(row[0]) not in held_groups]
    train_x, train_y = zip(*filtered)
    if held_groups.intersection(map(normalized_text, train_x)):
        raise ValueError('A held-domain text group is in optimization')
    pack(folder/'train.zip', 'single_text', train_x, train_y, 'macro_f1', 'train')
    pack(folder/'features.zip', 'single_text', held_x, None, 'macro_f1', 'test')
    task = load_development_task(folder/'train.zip')
    rows = load_prediction_features(folder/'features.zip', task.kind, task.spec)
    training = content_digest(train_x, train_y)
    held_digest = content_digest(held_x, held_y)
    if (task.kind != 'single_text' or task.X_val is not None or task.y_val is not None
            or content_digest(task.X_train, task.y_train) != training or content_digest(rows, held_y) != held_digest
            or MetricSpec.for_task(task).name != 'macro_f1'):
        raise ValueError('Packed inputs changed their order, values, contract or split')
    splits = []
    for seed in (707,808,909):
        fit, query, _ = grouped_partition(task, seed, 'shared_text')
        splits.append(dict(seed=seed, train_indices=fit.tolist(), search_indices=query.tolist(),
                           row_ids_sha256=fingerprint(query), label_sha256=fingerprint(task.y_train[query])))
    inner = folder/'inner.json'
    record.write(inner, dict(version=1, kind=task.kind, grouping=effective_grouping(task.kind, 'shared_text'),
                             training_content_sha256=training, splits=splits))
    digest = ab_run.sha(inner)
    registered = folder/'registered'
    registered.mkdir()
    with patch.object(ab_run, 'grouped_partition', registered_partitioner(task, inner, digest)):
        _, trusted = ab_run.freeze_partitions(task, registered, 707, MetricSpec.for_task(task))
    contract = TaskSpec.from_task(task).record()
    caps = ab_run.registered_capabilities(contract, len(train_y), 'full')
    if list(caps)[-1] != extra:
        raise ValueError('New candidate was not appended to the common catalogue')
    before = {name: spec for name,spec in caps.items() if name != extra}
    return dict(contract=contract, rows=len(held_y), training_rows=len(train_y),
                removed_training_rows=len(available)-len(filtered), held_groups=len(held_groups),
                overlapping_groups_after_filter=0, held_content_sha256=held_digest, training_content_sha256=training,
                files={'train.zip': ab_run.sha(folder/'train.zip')}, features_sha256=ab_run.sha(folder/'features.zip'),
                inner_sha256=digest, splits=trusted, catalogs={
                    arm: dict(catalog_sha256=record.digest(values), candidate_ids=list(values))
                    for arm,values in (('original',before), ('extended',caps))})


def stage(case, packet, entry, arm):
    case.mkdir()
    for name in ('train.zip','inner.json'):
        os.link(packet/name, case/name)
    shutil.copytree(packet/'registered', case/'registered', copy_function=os.link)
    record.write(case/'entry.json', dict(entry, name=case.name, **entry['catalogs'][arm]))


def worker(args):
    if record.ROOT != args.root or record.digest(ab_run.sources()) != args.source:
        raise ValueError('Worker imported a different checkout')
    case = args.case
    entry = record.read(case/'entry.json')
    if args.phase == 'train':
        command = ('-m','tools.ab_registered_run','--split-manifest',case/'inner.json','--split-sha256',entry['inner_sha256'],
                   '--','run','--catalog-scope','full','--planner','local','--history-policy','strict')
        job = pilot.train_one(case,entry,args.source,launcher=command)
        actual = record.read(case/'run/run_manifest.json')
        if (actual['task_contract'] != entry['contract'] or list(actual['capabilities']) != entry['candidate_ids']
                or actual['fit_limit'] != 11 or actual['time_limit_seconds'] != 1200 or actual['trial_timeout_seconds'] != 300):
            raise ValueError('Worker ran another task, menu or budget')
        record.write(case/'training.json', job)
    else:
        if ab_run.sha(args.features) != entry['features_sha256']:
            raise ValueError('Frozen feature-only rows changed')
        record.write(case/'delivery.json',dict(csv_sha256=export_frozen(case,args.features,entry['rows'])))


def invoke(case, root, source, phase, features=None):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (root/'scripts/ab_acceptance_guard',root))))
    command = [sys.executable, Path(__file__).resolve(), 'worker', '--phase',phase,'--case',case,
               '--root',root,'--source',source]
    if features is not None:
        command += ['--features',features]
    with (case/(phase+'_driver.log')).open('x') as log:
        subprocess.run(list(map(str,command)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1320)


def study(args):
    from scripts.run_b_nonlinear_text import identity, pair_grade
    from superagent.ab_methods import TEXT_KERNEL
    from superagent.ab_sentence import verified_directory
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()),
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    os.environ.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
    roots = dict(original=args.control.resolve(),extended=record.ROOT)
    sources = identity(roots['original'])
    if verified_directory() is None:
        raise ValueError('Verified local encoder is required for both menus')
    record.write(out/'protocol.json',dict(batch='B-TEXT-KERNEL-05',source_url=DATASET_URL,source_sha256=ARCHIVE_SHA,
        members=MEMBERS,driver_sha256=ab_run.sha(Path(__file__)),plan_sha256=ab_run.sha(record.ROOT/PLAN),
        ordered_runs=ORDER,sources=sources,environment=pilot.environment(),fit_limit=11,max_total_fits=66,
        wall_seconds=1200,trial_seconds=300,model_api_limit=0))
    try:
        domains = read_original(args.archive)
        entries = {key:register(out/key,domains,key,TEXT_KERNEL) for key in MEMBERS}
        record.write(out/'preflight.json',entries)
        jobs = {}
        for domain,arm in ORDER:
            case = out/(domain+'_'+arm)
            stage(case,out/domain,entries[domain],arm)
            start = len(record.trace(out))
            invoke(case,roots[arm],sources[arm],'train')
            job = record.read(case/'training.json')
            observed = record.trace(out)[start:]
            expected = [row['candidate'] for row in job['workers'] for _ in range(row['attempted_classifier_fits'])]
            if [row['candidate'] for row in observed] != expected or len(observed) != job['fits']:
                raise ValueError('Independent fit counter differs from worker ledgers')
            history = record.read(case/'run/history.json')
            job['nonlinear_observations'] = [dict(trial=r['trial'],split=r['split_id'],score=r['search_score'])
                for r in history if r['candidate']==TEXT_KERNEL and r['status']=='completed']
            jobs[case.name] = job
            record.write(out/'checkpoint.json',jobs)
        record.write(out/'all_six_models_frozen.json',{k:v['model_sha256'] for k,v in jobs.items()})
        pairs = {}
        for domain,entry in entries.items():
            features = out/domain/'features.zip'
            truth = np.asarray([row[1] for row in domains[domain]])
            task = load_development_task(out/domain/'train.zip')
            if content_digest(load_prediction_features(features,task.kind,task.spec),truth) != entry['held_content_sha256']:
                raise ValueError('Frozen public domain changed')
            files = {}
            for arm in roots:
                case = out/(domain+'_'+arm)
                choice = record.read(case/'run/final_selection.json')
                if ab_run.sha(case/'run'/choice['model_file']) != jobs[case.name]['model_sha256']:
                    raise ValueError('Model changed before grading')
                invoke(case,roots[arm],sources[arm],'deliver',features)
                files[arm] = case/'export_0/rows.csv'
                if ab_run.sha(files[arm]) != record.read(case/'delivery.json')['csv_sha256']:
                    raise ValueError('Graded export differs from independently reloaded model')
            grades,delta = pair_grade(truth,task,files)
            pairs[domain] = dict(grades=grades,paired=delta)
        fits = len(record.trace(out))
        if fits != sum(j['fits'] for j in jobs.values()) or fits>66 or identity(roots['original'])!=sources:
            raise ValueError('Fit budget, sources or accounting changed')
        record.write(out/'results.json',dict(paths=jobs,pairs=pairs))
        record.write(out/'completion.json',dict(status='completed',new_fits=fits,new_cli_exports=12,
            new_model_api_requests=0,new_course_rmb=0,official_validation_test_or_H=False))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',reason=str(error),new_fits=len(record.trace(out))))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    parent = modes.add_parser('study')
    for name in ('archive','control','weights','output'):
        parent.add_argument('--'+name,type=Path,required=True)
    child = modes.add_parser('worker')
    for name in ('case','root'):
        child.add_argument('--'+name,type=Path,required=True)
    child.add_argument('--phase',choices=('train','deliver'),required=True)
    child.add_argument('--features',type=Path)
    child.add_argument('--source',required=True)
    parsed = parser.parse_args()
    if parsed.mode == 'study':
        study(parsed)
    else:
        worker(parsed)
