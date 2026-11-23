"""One frozen shared-source autonomous run using every released training pair."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np

from superagent.ab_run import sha, sources
from superagent.ab_sentence import ALTERNATIVES, verified_directory, verified_files
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import normalized_text, sample_groups

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/batches/B_shared_full_task2_plan_20261123_zh.md'
BASE='a36296f550c85e49376caedd0149d21caf94dd63'
ARCHIVE='56b5439ea3f77c7b933d9c0bc49705f3759b3a1d302da49adee80031904834e4'
FEATURES='e4de6bcd251ff92e3f0ac519da3c4eedfca500ec29664b642735a9dca4f9dc5d'
PREVIOUS='43354ff27637f269cbbcea5dc63f3cfb3c2d262a0a0efdae6eaf4169d879be64'


def check(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    return json.loads(path.read_text())


def seal(path, value):
    if path.exists():
        check(read(path)==value, 'Prior receipt changed: '+path.name)
    else:
        with path.open('x',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2)


def release_rows(archive, part, fields):
    check(part in ('train','val'),'Only training and course validation are part of this run')
    member='Released_Task2-Headlines_CTR/'+part+'.csv'
    with zipfile.ZipFile(archive) as z,z.open(member) as f:
        reader=csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline=''))
        check(set(reader.fieldnames or [])=={'headline_A','headline_B','label_pairwise'},'Unexpected release schema')
        return [tuple(r[k] for k in fields) for r in reader]


def train_packet(path, examples):
    manifest=dict(schema_version=1,objective='classification',metric='accuracy',modality='pair_text',
                  columns=[dict(name='option_a',type='text'),dict(name='option_b',type='text')],
                  target='outcome',pair_semantics='preference',pair_label_one='left',
                  splits=dict(train=dict(format='csv',path='rows.csv')))
    table=io.StringIO(newline='')
    wr=csv.writer(table)
    wr.writerow(['option_a','option_b','outcome'])
    wr.writerows(examples)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        for name,data in [('manifest.json',json.dumps(manifest)),('rows.csv',table.getvalue())]:
            entry=zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(entry,data.encode())
    if path.exists():
        check(path.read_bytes()==buf.getvalue(),'Anonymous input differs')
    else:
        with path.open('xb') as f:
            f.write(buf.getvalue())


def command(args,log,env,limit):
    with log.open('x',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-m','superagent',*map(str,args)],cwd=ROOT,
                       env=env,stdout=f,stderr=subprocess.STDOUT,timeout=limit,check=True)


def exported(csv_file):
    with csv_file.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        check(r.fieldnames==['row_id','prediction'],'Unexpected prediction columns')
        rows=list(r)
    check([v['row_id'] for v in rows]==list(map(str,range(940))) and
          {v['prediction'] for v in rows}=={'0','1'},'Validation export row order or classes differ')
    return np.array([int(v['prediction']) for v in rows])


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    run_dir=a.out/'agent'
    current=sources()
    expected={k.removeprefix('superagent/'):v for k,v in
              read(ROOT/'research/a126_merged_checks/merge_verification.json')['source_files'].items()}
    check(current==expected and len(current)==91 and sha(a.archive)==ARCHIVE,'Shared source or released packet differs')
    weights=dict(SUPERAGENT_MINILM_WEIGHTS=str(a.weights/'all-MiniLM-L6-v2'),
                 SUPERAGENT_DEEP_MINILM_WEIGHTS=str(a.weights/'all-MiniLM-L12-v2'),
                 SUPERAGENT_E5_SMALL_WEIGHTS=str(a.weights/'e5-small-v2'))
    os.environ.update(weights)
    e5=ALTERNATIVES['e5']
    check(all((verified_directory(),verified_directory(deep=True),verified_files(e5['env'],e5['files']))),
          'A required local frozen encoder did not match the shared registry')
    examples=release_rows(a.archive,'train',('headline_A','headline_B','label_pairwise'))
    check(len(examples)==8301 and {r[2] for r in examples}=={'0','1'},'Training row count or labels differ')
    training=a.out/'anonymous.zip'
    train_packet(training,examples)
    task=load_development_task(training)
    check(task.X_val is None and task.y_val is None and task.kind=='pair_text' and
          [tuple(v) for v in task.X_train]==[r[:2] for r in examples] and
          np.array_equal(task.y_train,[int(r[2]) for r in examples]),'Anonymous training content changed')
    options=['--planner','local','--scheduler','ab-adaptive-complement','--portfolio','ab-v1',
             '--preference-catalog','deep-e5-familiarity','--cover-untried-encoder','--seed','18643',
             '--fit-limit','10','--trial-timeout','300','--time-limit','1200']
    seal(a.out/'protocol.json',dict(plan=sha(PLAN),runner=sha(Path(__file__)),shared_commit=BASE,production=current,
         training=sha(training),released_source=ARCHIVE,rows=len(examples),options=options,model_api_cap=0,
         course_validation_rows=940,previous_csv=PREVIOUS,feature_packet=FEATURES))
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/ab_acceptance_guard',ROOT))),
             AB_B1_OFFLINE='1',AB_B1_FIT_LOG=str(a.out/'physical.jsonl'),AB_B1_PER_WORKER_FIT_LIMIT='1',
             OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if not (run_dir/'run_outcome.json').exists():
        check(not run_dir.exists() and not (a.out/'physical.jsonl').exists(),'Incomplete run must be investigated')
        command(['run',training,*options,'--run-dir',run_dir],a.out/'training.log',env,1260)
    outcome,manifest,fit,selection=[read(run_dir/k) for k in
                                    ('run_outcome.json','run_manifest.json','classifier_fits.json','final_selection.json')]
    observed=[json.loads(line) for line in (a.out/'physical.jsonl').read_text().splitlines()]
    check(0<len(observed)<=10 and [r['candidate'] for r in observed]==[r['candidate'] for r in fit['workers']]
          and len(observed)==fit['attempted_classifier_fits']==fit['completed_classifier_fits']
          and fit['unknown_failed_attempts']==0,'Independent physical accounting mismatch')
    check(outcome['requested_planner_completed'] and outcome['new_api_requests']==0 and outcome['full_refit_completed']
          and not outcome['formal_test_accessed'] and outcome['source_unchanged'] and manifest['source_files']==current
          and manifest['task_archive_sha256']==sha(training) and manifest['cover_untried_encoder']
          and selection['training_rows']==8301 and selection['model_sha256']==sha(run_dir/'selected_model.joblib'),
          'Native autonomous path or frozen model did not match its contract')
    actions=read(run_dir/'actions.json')
    history=read(run_dir/'history.json')
    seal(a.out/'models_frozen.json',dict(protocol=sha(a.out/'protocol.json'),model=selection['model_sha256'],
         selection=selection['family'],components=selection['ensemble_components'],training_rows=8301,
         candidates=list(manifest['capabilities']),actions=actions,completed_fits=len(observed),
         physical=sha(a.out/'physical.jsonl'),raw_history=sha(run_dir/'history.json'),
         trials=[dict(trial=v['trial'],family=v['candidate'],split=v['split_id'],score=v['search_score'],
                      held_rows=v['search_metrics']['rows'],row_ids_sha256=v['row_ids_sha256']) for v in history]))
    source=a.previous/'designated/inputs/features.zip'
    prior=a.previous/'designated/autonomous_export_0/rows.csv'
    check(sha(source)==FEATURES and sha(prior)==PREVIOUS,'Prior validation artifact changed')
    val_x=release_rows(a.archive,'val',('headline_A','headline_B'))
    with zipfile.ZipFile(source) as z,z.open('rows.csv') as f:
        reader=csv.DictReader(io.TextIOWrapper(f,encoding='utf-8',newline=''))
        check(reader.fieldnames==['option_a','option_b'] and [(r['option_a'],r['option_b']) for r in reader]==val_x,
              'Feature-only archive does not match course validation')
    files=[]
    for n in (0,1):
        f=a.out/f'export_{n}/rows.csv'
        if not (f.parent/'prediction_manifest.json').exists():
            command(['predict',source,'--run-dir',run_dir,'--output',f],a.out/f'predict_{n}.log',env,600)
        meta=read(f.parent/'prediction_manifest.json')
        check(meta['test_labels_read'] is False and meta['prediction_count']==940,'Invalid independent export')
        files.append(f)
    check(sha(files[0])==sha(files[1]),'Independent exports differ')
    seal(a.out/'exports.json',dict(frozen=sha(a.out/'models_frozen.json'),features=FEATURES,old=PREVIOUS,
         new=[sha(f) for f in files],rows=len(val_x)))
    y=np.array([int(v[0]) for v in release_rows(a.archive,'val',('label_pairwise',))])
    check(fingerprint(y)=='b6a267d3011cc842051a8731de2746c2485041dcbcfa02943b80df7a782bbed5','Validation labels changed')
    previous,pred=exported(prior),exported(files[0])
    before,after=previous==y,pred==y
    check(int(before.sum())==689,'Published historical control changed')
    groups=sample_groups(SimpleNamespace(kind='pair_text',X_train=val_x),'shared_text')
    buckets={}
    for key,b,h in zip(groups,before,after,strict=True):
        buckets.setdefault(key,[]).append(int(b)+2*int(h))
    grouped=[sorted(buckets[k]) for k in sorted(buckets)]
    seal(a.out/'grouped_correctness.json',grouped)
    sizes=np.asarray([len(v) for v in grouped])
    gain=np.asarray([v.count(2)-v.count(1) for v in grouped])
    rng=np.random.default_rng(60423)
    intervals=[]
    for _ in range(100):
        picks=rng.integers(0,len(grouped),size=(50,len(grouped)))
        intervals.extend(100*gain[picks].sum(axis=1)/sizes[picks].sum(axis=1))
    seen_titles={normalized_text(t) for row in task.X_train for t in row}
    seen_pairs={tuple(sorted(map(normalized_text,row))) for row in task.X_train}
    result=dict(scope='course-designated validation after model freeze',rows=940,training_rows=8301,
         prior_training_rows=6608,previous=int(before.sum()),correct=int(after.sum()),accuracy=float(after.mean()),
         gained=int((~before&after).sum()),lost=int((before&~after).sum()),validation_groups=len(grouped),
         paired_group_bootstrap_95_pp=np.quantile(intervals,[.025,.975]).tolist(),
         rows_sharing_a_training_title=sum(any(normalized_text(t) in seen_titles for t in row) for row in val_x),
         rows_matching_a_training_pair=sum(tuple(sorted(map(normalized_text,row))) in seen_pairs for row in val_x),
         confusion=[[int(((y==i)&(pred==j)).sum()) for j in (0,1)] for i in (0,1)],
         selected=selection['family'],components=selection['ensemble_components'],actions=len(actions),
         physical_fits=len(observed),model_api_calls=0,input_llm_tokens=0,output_llm_tokens=0,new_rmb=0,
         course_test_opened=False)
    seal(a.out/'results.json',result)
    seal(a.out/'completion.json',{k:sha(a.out/(k+'.json')) for k in
         ('protocol','models_frozen','exports','grouped_correctness','results')})
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','weights','previous','out'):
        parser.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    run(parser.parse_args())
