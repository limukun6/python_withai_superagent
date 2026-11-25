"""Check the two autonomous releases and assemble the runnable engineering handoff from that source."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from scripts import verify_anonymous_planner as floor
from scripts.package_unified_agent import packed_source
from scripts.replay_anonymous_decisions import audit
from scripts.unified_released_study import ROOT, archive, env, read, require, save, sha, title_rows
from scripts.verify_unified_raw import feature_package
from superagent import ab_sentence
from superagent.ab_run import sources


TAGS = ('table','titles')


def anonymous_replay(raw, course, weights):
    """Rename the archives and the paired columns; replay the entire menu with the same measured observations."""
    import joblib
    from scripts.replay_anonymous_decisions import key
    from scripts.unified_released_study import make_zip, text_packet
    from superagent.ab_run import registered_capabilities
    from superagent.autonomous_run import installed_methods, partitions, planner_observation
    from superagent.contracts import TaskSpec
    from superagent.data import TrainingTask, load_training_task
    from superagent.experiment_policy import actions
    from superagent.metrics import MetricSpec
    from superagent.open_methods import MethodSpace
    from superagent.splits import take
    from superagent.task_profile import training_facts

    settings=env(weights,raw/'unexpected_fits.jsonl')
    os.environ.update({k:v for k,v in settings.items() if k.startswith('SUPERAGENT_') and k.endswith('_WEIGHTS')})
    result={}
    with tempfile.TemporaryDirectory() as tmp, installed_methods():
        for i,tag in enumerate(TAGS):
            folder=Path(tmp)/str(i)
            folder.mkdir()
            anonymous=folder/'opaque.zip'
            original=archive(course,tag)
            if tag=='titles':
                text_packet(anonymous,title_rows(original,'train',labels=True),True)
            else:
                with zipfile.ZipFile(original) as z:
                    entries=[('opaque/'+Path(n).name,z.read(n)) for n in floor.MEMBERS[tag] if n.endswith('.npy')]
                make_zip(anonymous,entries+[('opaque/README.md',b'Report accuracy.')])
            task=load_training_task(anonymous,read_declarations=True)
            spec,metric=TaskSpec.from_task(task).record(),MetricSpec.for_task(task)
            run=raw/tag/'run'/'agent'
            folds=partitions(task,read(run/'run_manifest.json')['seed'])
            for name,(fit,held,_) in folds.items():
                original_split=read(run/(name+'_split.json'))
                require(np.array_equal(fit,original_split['fit']) and np.array_equal(held,original_split['held']),
                        'Renamed input changed partition membership')
            fit=folds['primary'][0]
            profile=training_facts(TrainingTask(task.kind,take(task.X_train,fit),task.y_train[fit],task.spec))
            width=task.X_train.shape[1] if task.kind=='numeric_tabular' else None
            static=registered_capabilities(spec,len(task.y_train),numeric_kernel_width=width)
            generator=MethodSpace(spec,profile,folder)
            history,decisions=read(run/'history.json'),read(run/'decisions.json')
            observed,blocked={f:{} for f in folds},set()
            for step,d in enumerate(decisions):
                caps,_=generator.refresh(history[:step],static)
                menu=actions(caps,profile,history[:step],observed,metric,list(folds),d['remaining_fits'],blocked)
                require([key(a) for a in menu]==[key(a) for a in d['alternatives']] and key(menu[0])==key(d['choice']),
                        'Renaming changed an available action or its rank')
                if step<len(history):
                    row=history[step]
                    if row['status']!='completed':
                        blocked.add(row['candidate'])
                        continue
                    fold=row['split_id']
                    bundle=joblib.load(run/f'trial_{row["trial"]}.joblib')
                    payload=dict(labels=task.y_train[folds[fold][1]],sha256=sha(run/(fold+'.joblib')))
                    observed[fold][row['candidate']]=planner_observation(bundle,bundle['metrics'],payload,metric)
            result[tag]=dict(entire_ranked_menus_identical=len(decisions),same_group_membership=True,
                             archive_and_directory_renamed=True,paired_columns_renamed=tag=='titles',new_fits=0)
    save(raw/'anonymous-replay.json',result)


def validate(raw, course, weights):
    protocol = read(raw/'protocol.json')
    require(protocol['source']==floor.revision(),'Different source or evaluation adapter')
    settings=env(weights,raw/'unexpected_fits.jsonl')
    os.environ.update({k:v for k,v in settings.items() if k.startswith('SUPERAGENT_') and k.endswith('_WEIGHTS')})
    frozen={}
    evidence={}
    require(all((raw/tag/'sealed.json').is_file() for tag in TAGS),'Both autonomous models must be frozen')
    for tag in TAGS:
        folder=raw/tag
        sealed=read(folder/'sealed.json')
        run=folder/'run'/'agent'
        decisions=read(run/'decisions.json')
        history=read(run/'history.json')
        result=read(run/'run_outcome.json')
        require(result['status']=='completed' and result['new_api_requests']==0,'Unexpected incomplete or paid run')
        require(read(run/'run_manifest.json')['source_files']==sources(),'Workers did not use the floor source')
        for step,d in enumerate(decisions):
            require(d['observed_trials']==[r['trial'] for r in history[:step]],'Choice used future trials')
            require(d['choice'] in d['alternatives'] and d['source']=='local evidence policy','Choice outside local menu')
        require(decisions[-1]['choice']['action']=='stop' and decisions[-1]['choice']['members']==result['members'],
                'Final method differs from the recorded autonomous decision')
        require(sealed['model']==sha(run/'selected_model.joblib'),'Frozen model identity changed')
        evidence[tag]=dict(replay=audit(folder),decisions=len(decisions),fits=len(sealed['physical_fits']),
                           locally_chosen=result['members'],training_members=sealed['read_members'])
        features=folder/'features.zip'
        if not features.exists():
            feature_package(course,tag,features)
        hashes=[]
        for i in range(2):
            dest=folder/f'pred_{i}'/'predictions.csv'
            if not dest.exists():
                floor.execute(['predict',features,'--run-dir',folder/'run','--output',dest],
                              folder/f'prediction_{i}.log',env(weights,folder/'unexpected_fit.jsonl'))
            hashes.append(sha(dest))
        require(hashes[0]==hashes[1] and not (folder/'unexpected_fit.jsonl').exists(),'Frozen predictions changed')
        frozen[tag]=hashes[0]
    save(raw/'sealed-predictions.json',frozen)
    results={}
    for tag in TAGS:
        path=archive(course,tag)
        if tag=='table':
            with zipfile.ZipFile(path) as z:
                truth=np.load(io.BytesIO(z.read('in_vehicle_coupon_recommendation/y_val.npy')),allow_pickle=False)
        else:
            truth=np.asarray([r[2] for r in title_rows(path,'val',labels=True)])
        with (raw/tag/'pred_0'/'predictions.csv').open() as f:
            predictions=list(csv.DictReader(f))
        require([r['row_id'] for r in predictions]==list(map(str,range(len(truth)))),'Validation row order mismatch')
        hits=int((np.asarray([r['prediction'] for r in predictions],dtype=truth.dtype)==truth).sum())
        results[tag]=dict(correct=hits,rows=len(truth),accuracy=hits/len(truth),prediction=frozen[tag],
                          model=read(raw/tag/'sealed.json')['model'],members=evidence[tag]['locally_chosen'])
    save(raw/'results.json',results)
    save(raw/'autonomy.json',dict(source=protocol['source_commit'],evidence=evidence,
         validation_opened_after_all_predictions_frozen=True,formal_test_accessed=False,live_api_requests=0))


def handoff(raw, weights, dest):
    protocol = read(raw / 'protocol.json')
    require(protocol['source'] == floor.revision(), 'Package from the checkout that performed the floor runs')
    for tag in TAGS:
        run = raw / tag / 'run' / 'agent'
        require(read(run / 'run_manifest.json')['source_files'] == sources(),
                'The handoff source must match both trained models')
        require(sha(run / 'selected_model.joblib') == read(raw / tag / 'sealed.json')['model'],
                'The trained model changed after sealing')
    files,assets=packed_source(raw,weights)
    bge=ab_sentence.ALTERNATIVES['bge']
    identifier=bge['model']
    folder=identifier.split('/')[-1]
    assets[identifier]=dict(revision=bge['revision'],sha256={n:v[1] for n,v in bge['files'].items()})
    for name,(size,digest) in bge['files'].items():
        source=weights/folder/name
        require(source.stat().st_size==size and sha(source)==digest,'Incorrect offline encoder')
        files[f'weights/{folder}/{name}']=source
    files['requirements.txt']=ROOT/'requirements.txt'
    files['autonomy.json']=raw/'autonomy.json'
    files['protocol.json']=raw/'protocol.json'
    for tag in TAGS:
        path=raw/tag
        for name in ('decisions.json','history.json','run_manifest.json','run_outcome.json','classifier_fits.json',
                     'llm_usage.jsonl','events.jsonl','candidate_registry.json','training_profile.json'):
            files[f'models/{tag}/agent/{name}']=path/'run'/'agent'/name
        files[f'results/{tag}/validation_predictions.csv']=path/'pred_0'/'predictions.csv'
        files[f'results/{tag}/receipt.json']=path/'sealed.json'
    guide=(
        '# 自主规划提交基线\n\n源码提交 '+read(raw/'protocol.json')['source_commit']+'。\n\n'
        '在 Python 3.11 环境安装 requirements-ab-encoder-tested.txt 和 requirements.txt 的依赖。'
        '从解压目录为终端设置下面四个变量后，训练和预测可以离线运行。\n\n'
        '\x60\x60\x60sh\nexport SUPERAGENT_MINILM_WEIGHTS="$PWD/weights/all-MiniLM-L6-v2"\n'
        'export SUPERAGENT_DEEP_MINILM_WEIGHTS="$PWD/weights/all-MiniLM-L12-v2"\n'
        'export SUPERAGENT_E5_SMALL_WEIGHTS="$PWD/weights/e5-small-v2"\n'
        'export SUPERAGENT_BGE_SMALL_WEIGHTS="$PWD/weights/bge-small-en-v1.5"\n'
        'python -m unified_agent solve TASK.zip --prediction-input FEATURES.zip --run-dir runs/new --output delivery/predictions.csv\n'
        'python -m unified_agent predict FEATURES.zip --run-dir models/titles --output delivery/titles.csv\n\x60\x60\x60\n\n'
        'models/table 和 models/titles 保存本提交两次从头自主运行的交付模型。results 保存指定验证集预测。'
        '该工程包供复现和整合。最终课程报告、AI 使用声明及最终评测材料由团队汇编。\n')
    embedded={'README.md':guide.encode(),'weights.json':json.dumps(assets,indent=2).encode()}
    hashes={name:sha(src) for name,src in files.items()}
    hashes.update({name:hashlib.sha256(data).hexdigest() for name,data in embedded.items()})
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,src in files.items():
            z.write(src,name,compress_type=zipfile.ZIP_STORED if name.endswith('.safetensors') else zipfile.ZIP_DEFLATED)
        for name,data in embedded.items():
            z.writestr(name,data)
        z.writestr('SHA256SUMS',''.join(f'{hashes[k]}  {k}\n' for k in sorted(hashes)))
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        root=Path(tmp)
        with zipfile.ZipFile(dest) as z:
            z.extractall(root)
        require(all(sha(root/name)==digest for name,digest in hashes.items()),'Extracted content differs')
        import subprocess
        import sys
        config=env(root/'weights',root/'unexpected_fit.jsonl')
        config.pop('PYTHONPATH',None)
        isolated='import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module("unified_agent",run_name="__main__")'
        for tag in TAGS:
            output=root/'replay'/tag/'predictions.csv'
            subprocess.run([sys.executable,'-I','-B','-c',isolated,str(root),'predict',str(raw/tag/'features.zip'),
                           '--run-dir',str(root/'models'/tag),'--output',str(output)],cwd=root,env=config,check=True,
                           stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,timeout=480)
            require(sha(output)==read(raw/'results.json')[tag]['prediction'],'Portable predictions differ')
    save(raw/'portable.json',dict(commit=read(raw/'protocol.json')['source_commit'],archive=dest.name,
         sha256=sha(dest),bytes=dest.stat().st_size,isolated_replays=list(TAGS),bundled_encoders=len(assets)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['validate','aliases','package'])
    p.add_argument('raw',type=Path)
    p.add_argument('course',type=Path)
    p.add_argument('weights',type=Path)
    p.add_argument('--archive',type=Path)
    a=p.parse_args()
    if a.action=='validate':
        validate(a.raw.resolve(),a.course.resolve(),a.weights.resolve())
    elif a.action=='aliases':
        anonymous_replay(a.raw.resolve(),a.course.resolve(),a.weights.resolve())
    else:
        handoff(a.raw.resolve(),a.weights.resolve(),a.archive.resolve())
