"""Reproduce both released baselines using the same native training and prediction commands."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import numpy as np

from superagent.ab_run import sha, sources
from superagent.splits import sample_groups


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'research/unified_release_1124'
PLAN = ROOT / 'docs/batches/unified_release_protocol_20261124_zh.md'
PANELS = {'table': (8117,2030), 'titles': (6608,940), 'titles_full': (8301,940)}
SCHEMA = dict(modality='pair_text', columns=[dict(name='option_a',type='text'),dict(name='option_b',type='text')],
              target='outcome',pair_semantics='preference',pair_label_one='left')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path=Path(path)
    if path.exists():
        require(read(path)==obj, 'Existing evidence differs: '+path.name)
    else:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def archive(packages, tag):
    files=list(packages.glob('Released_Task'+('1' if tag=='table' else '2')+'*.zip'))
    require(len(files)==1,'Ambiguous source package')
    return files[0]


def title_rows(path, split, *, labels):
    with zipfile.ZipFile(path) as z, z.open('Released_Task2-Headlines_CTR/'+split+'.csv') as raw:
        rows=csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig',newline=''))
        return [(r['headline_A'],r['headline_B'],int(r['label_pairwise'])) if labels else
                (r['headline_A'],r['headline_B']) for r in rows]


def make_zip(path, entries):
    with zipfile.ZipFile(path,'x') as z:
        for name,data in entries:
            item=zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0))
            item.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(item,data)


def text_packet(path, rows, training):
    split='train' if training else 'test'
    manifest=dict(schema_version=1,objective='classification',metric='accuracy',
                  splits={split:dict(format='csv',path='rows.csv')},**SCHEMA)
    buf=io.StringIO(newline='')
    writer=csv.writer(buf)
    writer.writerow(['option_a','option_b']+(['outcome'] if training else []))
    writer.writerows(rows)
    make_zip(path,[('manifest.json',json.dumps(manifest).encode()),('rows.csv',buf.getvalue().encode())])


def env(weights,physical):
    result=dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/ab_acceptance_guard',ROOT))),
                AB_B1_OFFLINE='1', AB_B1_PER_WORKER_FIT_LIMIT='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                SUPERAGENT_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OPENBLAS_NUM_THREADS='2',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for key,folder in (('MINILM','all-MiniLM-L6-v2'),('DEEP_MINILM','all-MiniLM-L12-v2'),
                       ('E5_SMALL','e5-small-v2'),('BGE_SMALL','bge-small-en-v1.5')):
        result['SUPERAGENT_'+key+'_WEIGHTS']=str(weights/folder)
    result['AB_B1_FIT_LOG']=str(physical)
    return result


def command(args, log, environment):
    with log.open('x') as output:
        result=subprocess.run([sys.executable,'-m','superagent',*map(str,args)],cwd=ROOT,env=environment,
                              stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,timeout=1320)
    require(result.returncode==0, 'Native command failed: '+log.name)


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False)
    rows=title_rows(archive(a.packages,'titles'),'train',labels=True)
    reserve=read(EVIDENCE/'title_reserve_ids.json')
    excluded=set(reserve)
    use=[i for i in range(len(rows)) if i not in excluded]
    require(len(use)==6608 and len(reserve)==len(excluded)==1693 and set(use)|excluded==set(range(8301)),'Incorrect released training membership')
    groups=sample_groups(SimpleNamespace(kind='pair_text',X_train=[r[:2] for r in rows]),'shared_text')
    require(not set(groups[use]) & set(groups[reserve]),'Title overlap across the development reserve')
    folders={tag:a.out/tag for tag in PANELS}
    for folder in folders.values():
        folder.mkdir()
    with zipfile.ZipFile(archive(a.packages,'table')) as z:
        names=('C_train.npy','y_train.npy','Readme.md')
        entries=[(n,z.read('in_vehicle_coupon_recommendation/'+n)) for n in names]
        make_zip(folders['table']/'train.zip',entries)
    text_packet(folders['titles']/'train.zip',[rows[i] for i in use],True)
    require(sha(folders['titles']/'train.zip')=='f4e30ae46244196b927b2ea278c2d900ba2c53bcffa7b93f165e47aed1190bd2','Historical training bytes differ')
    text_packet(folders['titles_full']/'train.zip',rows,True)
    text_packet(a.out/'development_features.zip',[rows[i][:2] for i in reserve],False)
    save(a.out/'development_truth.json',dict(labels=[rows[i][2] for i in reserve],groups=list(groups[reserve])))
    refs={tag:read(folder/'final_selection.json')['model_sha256'] for tag,folder in [('table',a.old_table),('titles',a.old_titles)]}
    for tag,folder in [('table',a.old_table),('titles',a.old_titles)]:
        require(sha(folder/'selected_model.joblib')==refs[tag], 'Historical model changed')
    save(a.out/'protocol.json',dict(source=sources(),plan=sha(PLAN),runner=sha(Path(__file__)),historical_models=refs,
         historical_expected_correct=dict(table=1532,titles=689),paper_test_used=False,api_calls_allowed=0,
         training={tag:dict(size=PANELS[tag][0],validation_size=PANELS[tag][1],packet=sha(folder/'train.zip')) for tag,folder in folders.items()},
         archives={tag:sha(archive(a.packages,tag)) for tag in ('table','titles')},
         reserve_ids=sha(EVIDENCE/'title_reserve_ids.json'),development_features=sha(a.out/'development_features.zip'),
         development_truth=sha(a.out/'development_truth.json'),common_seed=18643,common_fit_limit=10))
    print('Sealed common runtime and three training scopes',flush=True)


def train(a, tag):
    protocol=read(a.out/'protocol.json')
    folder=a.out/tag
    require(protocol['source']==sources() and sha(folder/'train.zip')==protocol['training'][tag]['packet'],'Frozen training inputs changed')
    command(['unified',folder/'train.zip','--run-dir',folder/'agent'], folder/'train.log',env(a.weights,folder/'physical.jsonl'))
    agent=folder/'agent'
    manifest=read(agent/'run_manifest.json')
    selection=read(agent/'final_selection.json')
    status=read(agent/'run_outcome.json')
    counts=read(agent/'classifier_fits.json')
    actual=[json.loads(r) for r in (folder/'physical.jsonl').read_text().splitlines()]
    require(manifest['source_files']==protocol['source'] and manifest['seed']==protocol['common_seed'] and
            manifest['fit_limit']==protocol['common_fit_limit'] and manifest['training_only'] and
            manifest['invoked_as']=='superagent unified' and selection['training_rows']==PANELS[tag][0] and
            status['new_api_requests']==0 and status['source_unchanged'] and status['full_refit_completed'] and
            status['requested_planner_completed'] and not status['formal_test_accessed'] and
            selection['model_sha256']==sha(agent/'selected_model.joblib') and len(actual)==counts['attempted_classifier_fits']
            ==counts['completed_classifier_fits']<=10,'Incomplete, inconsistent or over-budget cold run')
    save(folder/'model_frozen.json',dict(model=selection['model_sha256'],training_rows=selection['training_rows'],
         winner=selection['family'],members=selection['ensemble_components'],fits=len(actual),actions=read(agent/'actions.json'),
         invoked_as=manifest['invoked_as'],policy=manifest['unified_policy'],fit_records=actual,new_api_calls=0,
         source_digest=hashlib.sha256(json.dumps(sources(),sort_keys=True).encode()).hexdigest()))
    print(tag,selection['family'],selection['ensemble_components'],len(actual),flush=True)


def predictions(path, size):
    with path.open() as stream:
        records=list(csv.DictReader(stream))
    require([r['row_id'] for r in records]==list(map(str,range(size))),'Prediction row order changed')
    return np.array([int(r['prediction']) for r in records])


def export(a, full):
    tags=['titles_full'] if full else ['table','titles']
    if full:
        require((a.out/'scores.json').is_file(),'Baseline scores must be sealed before the full-data branch')
    stage=a.out/('full_validation' if full else 'validation')
    stage.mkdir(exist_ok=False)
    checked={tag:read(a.out/tag/'model_frozen.json') for tag in tags}
    sources_to_run={tag:a.out/tag/'agent' for tag in tags}
    if not full:
        sources_to_run.update(reference_table=a.old_table,reference_titles=a.old_titles)
    for key,folder in sources_to_run.items():
        expected=read(a.out/'protocol.json')['historical_models'][key.removeprefix('reference_')] if key.startswith('reference_') else checked[key]['model']
        require(sha(folder/'selected_model.joblib')==expected,'Pre-export model changed')
    for tag in tags:
        packet=stage/(tag+'.zip')
        if tag=='table':
            with zipfile.ZipFile(archive(a.packages,tag)) as z:
                make_zip(packet,[('C_test.npy',z.read('in_vehicle_coupon_recommendation/C_val.npy'))])
        else:
            text_packet(packet,title_rows(archive(a.packages,tag),'val',labels=False),False)
    seal={}
    for key,folder in sources_to_run.items():
        tag=key.removeprefix('reference_')
        exports=[]
        for i in range(2):
            dest=stage/f'{key}_{i}'/'predictions.csv'
            command(['predict',stage/(tag+'.zip'),'--run-dir',folder,'--output',dest],stage/f'{key}_{i}.log',env(a.weights,stage/'unexpected_fit.jsonl'))
            require(read(dest.parent/'prediction_manifest.json')['prediction_count']==PANELS[tag][1],'Wrong output size')
            exports.append(sha(dest))
        require(exports[0]==exports[1] and not (stage/'unexpected_fit.jsonl').exists(),'Cross-process prediction or cost changed')
        seal[key]=exports
    save(stage/'predictions_frozen.json',seal)
    scores={}
    for key in sources_to_run:
        tag=key.removeprefix('reference_')
        if tag=='table':
            with zipfile.ZipFile(archive(a.packages,tag)) as z:
                truth=np.load(io.BytesIO(z.read('in_vehicle_coupon_recommendation/y_val.npy')),allow_pickle=False)
        else:
            truth=np.array([r[2] for r in title_rows(archive(a.packages,tag),'val',labels=True)])
        current=predictions(stage/f'{key}_0/predictions.csv',len(truth))
        correct=int((current==truth).sum())
        scores[key]=dict(correct=correct,rows=len(truth),accuracy=correct/len(truth),prediction=seal[key][0])
        if key.startswith('reference_'):
            if tag=='table':
                archived=np.array(read(a.old_table/'validation_predictions.json'))
            else:
                archived=predictions(a.old_titles.parent/'designated/autonomous_export_0/rows.csv',len(truth))
            require(np.array_equal(current,archived) and correct==read(a.out/'protocol.json')['historical_expected_correct'][tag],
                    'Historical model did not reproduce its archived predictions')
        elif not full:
            ref=predictions(stage/f'reference_{tag}_0/predictions.csv',len(truth))
            scores[key].update(same_as_historical_rows=int((ref==current).sum()),
                               corrected=int(((ref!=truth)&(current==truth)).sum()),lost=int(((ref==truth)&(current!=truth)).sum()))
    save(a.out/('full_scores.json' if full else 'scores.json'),scores)
    print(json.dumps(scores),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','table','titles','titles_full','score','full_score'))
    for key in ('out','packages','weights','old-table','old-titles'):
        parser.add_argument('--'+key,required=True,type=lambda v:Path(v).resolve())
    a=parser.parse_args()
    if a.stage=='prepare':
        prepare(a)
    elif a.stage in PANELS:
        train(a,a.stage)
    else:
        export(a,a.stage=='full_score')
