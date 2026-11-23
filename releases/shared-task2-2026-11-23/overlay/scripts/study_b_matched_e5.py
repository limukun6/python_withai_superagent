"""Retest the original equal E5/L12 vote on the exact same training rows and held groups as the lexical partner."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from scripts.run_b_shared_full_task2 import check, read, seal
from scripts.study_b_grouped_blend import classify
from superagent.ab_complement import canonical_probabilities
from superagent.ab_methods import PREFERENCE_CATALOG_ENV, TRANSFER_METHODS
from superagent.ab_run import sha, sources
from superagent.data import load_development_task
from superagent.models import predict_with_model, search_candidates
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups, take

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'docs/batches/B_matched_e5_plan_20261124_zh.md'


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    parent, prior = read(a.parent/'protocol.json'), read(a.shadow/'protocol.json')
    sp = read(a.shadow/'frozen.json')
    manifest = read(a.parent/'agent/run_manifest.json')
    anchor, e5 = prior['anchor'], TRANSFER_METHODS['deep-e5'][1]
    check(sources() == manifest['source_files'] == parent['production'] and
          prior['source_run_protocol'] == sha(a.parent/'protocol.json') and
          sp['protocol'] == sha(a.shadow/'protocol.json') and
          sp['models'][0] == sha(a.shadow/'worker_0.joblib'), 'Frozen base or source changed')
    archive = a.parent/'anonymous.zip'
    check(sha(archive) == parent['training'], 'Parent training archive differs')
    task = load_development_task(archive)
    check(task.X_val is None and task.y_val is None, 'Expected an anonymous training-only archive')
    groups = sample_groups(task,'shared_text')
    opening = read(a.parent/'agent/primary_split.json')
    bucket = [int(hashlib.sha256(('b-borderline-20261123-v1/'+g).encode()).hexdigest(),16)%8 for g in groups]
    train = [i for i in opening['train_indices'] if bucket[i]>=2]
    panels = [[i for i in opening['train_indices'] if bucket[i]==k] for k in (0,1)]
    check(fingerprint(train) == prior['training_ids_sha256'] and
          [fingerprint(ids) for ids in panels] == [v['row_ids_sha256'] for v in prior['panels']]
          and set(groups[train]).isdisjoint(groups[panels[0]+panels[1]]) and
          set(groups[panels[0]]).isdisjoint(groups[panels[1]]), 'Development partitions changed')
    payload = a.shadow/'training.joblib'
    data = joblib.load(payload)
    check(data['pair_semantics'] == 'preference' and data['mode'] == 'refit' and
          data['X_train'] == take(task.X_train,train) and np.array_equal(data['y_train'],task.y_train[train])
          and data['X_search'] == take(task.X_train,train[:1]), 'Physical E5 training input differs from L12')
    def offline(event, args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            raise PermissionError('Matched E5 study is offline')
    sys.addaudithook(offline)
    os.environ.update(SUPERAGENT_DEEP_MINILM_WEIGHTS=str(a.weights/'all-MiniLM-L12-v2'),
                      SUPERAGENT_MINILM_WEIGHTS=str(a.weights/'all-MiniLM-L6-v2'),
                      SUPERAGENT_E5_SMALL_WEIGHTS=str(a.weights/'e5-small-v2'),
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.environ[PREFERENCE_CATALOG_ENV] = manifest['preference_catalog']
    catalog = {c.family:c.params for c in search_candidates('pair_text','all')}
    check(all(catalog.get(n)==manifest['capabilities'][n]['params'] for n in (anchor,e5)),
          'Candidate configurations unavailable')
    seal(a.out/'protocol.json',dict(plan=sha(PLAN),runner=sha(Path(__file__)), production=parent['shared_commit'],
         parent_protocol=sha(a.parent/'protocol.json'), shadow_completion=sha(a.shadow/'completion.json'),
         training_ids=prior['training_ids_sha256'], rows=len(train), payload=sha(payload), panels=prior['panels'],
         base_model=sp['models'][0], candidate=e5, parameters=catalog[e5], fit_cap=1, model_api_cap=0))
    admission = a.out/'admission.json'
    receipt = a.out/'worker_fit.json'
    seal(admission,dict(maximum_classifier_fits=1,ledger_path=str(receipt)))
    artifact=a.out/'e5.joblib'
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/ab_acceptance_guard',ROOT))),
             AB_B1_OFFLINE='1',AB_B1_FIT_LOG=str(a.out/'physical.jsonl'),AB_B1_PER_WORKER_FIT_LIMIT='1',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
    with (a.out/'worker.log').open('x') as log:
        subprocess.run([sys.executable,'-m','superagent.experiment_worker',str(payload),e5,str(artifact),
                        '--fit-admission',str(admission)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                       timeout=300,check=True)
    observed = [json.loads(line)['candidate'] for line in (a.out/'physical.jsonl').read_text().splitlines()]
    fits = read(receipt)
    check(observed==[e5] and fits['attempted_classifier_fits']==fits['completed_classifier_fits']==1 and
          fits['status']=='completed', 'Independent worker accounting mismatch')
    seal(a.out/'models_frozen.json',dict(protocol=sha(a.out/'protocol.json'), anchor=sp['models'][0],
         e5=sha(artifact),worker=sha(receipt),physical=sha(a.out/'physical.jsonl')))
    models=[joblib.load(f)['model'] for f in (a.shadow/'worker_0.joblib',artifact)]
    tables=[]
    for k,ids in enumerate(panels):
        x=take(task.X_train,ids)
        hard=[predict_with_model('pair_text',m,x) for m in models]
        avg=np.mean([canonical_probabilities(m,x)[1][:,1] for m in models],axis=0)
        vote=classify(avg,np.asarray([l<r for l,r in x]))
        check(fingerprint(hard[0])==sp['predictions'][k][0], 'Prior single model changed')
        tables.append([*hard,vote])
    seal(a.out/'exports.json',dict(frozen=sha(a.out/'models_frozen.json'),
         panels=[[fingerprint(col) for col in table] for table in tables],rows=[len(x) for x in panels]))
    records, evidence=[],[]
    for ids,table in zip(panels,tables,strict=True):
        masks=[col==task.y_train[ids] for col in table]
        counts=[int(col.sum()) for col in masks]
        group_cells=defaultdict(lambda:[0]*8)
        for g,b,c,d in zip(groups[ids],*masks,strict=True):
            group_cells[g][int(b)+2*int(c)+4*int(d)]+=1
        records.append(dict(rows=len(ids),groups=len(group_cells),single=counts[0],e5=counts[1],equal=counts[2],
                       gained_vs_anchor=int((~masks[0]&masks[2]).sum()),lost_vs_anchor=int((masks[0]&~masks[2]).sum()),
                       net_vs_stronger=counts[2]-max(counts[:2])))
        evidence.append([group_cells[g] for g in sorted(group_cells)])
    diff=[r['net_vs_stronger'] for r in records]
    result=dict(panels=records,net_vs_stronger=sum(diff),pass_gate=min(diff)>=0 and sum(diff)>=8,
                classifier_fits=1, model_api_calls=0, new_rmb=0, formal_test_opened=False, designated_validation_opened=False)
    seal(a.out/'grouped.json',evidence)
    seal(a.out/'results.json',result)
    seal(a.out/'completion.json',{k:sha(a.out/(k+'.json')) for k in ('protocol','models_frozen','exports','grouped','results')})
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','shadow','weights','out'):
        p.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    run(p.parse_args())
