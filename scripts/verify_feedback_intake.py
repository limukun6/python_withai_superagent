"""Two independent anonymous tasks share only training-derived hypotheses. A third can use campus Flash once."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import zipfile

import numpy as np
from sklearn.metrics import f1_score

from scripts.run_planner_self_review import packet, source_identity
from scripts.unified_released_study import ROOT, env, make_zip, read, save, sha

SEEDS={'learn':27011,'transfer':27029,'campus':27061}


def make_task(dest,seed):
    rng=np.random.default_rng(seed)
    n=1320
    x=rng.normal(size=(n,8))
    noisy=rng.random(n)<.26
    x[noisy]*=6
    y=np.argmax(x[:,:3],axis=1)
    y[noisy]=rng.integers(3,size=noisy.sum())
    truth=np.array([0,0,0,0,0,1,1,1,2,2])
    guess=np.array([0,0,0,1,1,0,1,2,2,2])
    second=np.array([0,0,0,0,1,1,2,2,2,2,2])
    proposed=np.array([0,0,0,2,1,2,1,1,2,2,0])
    evidence=dict(candidates=['accuracy','micro_f1','balanced_accuracy','macro_f1','weighted_f1'],
                  labels=[0,1,2],rounding='half_up',
                  examples=[dict(truth=t.tolist(),prediction=p.tolist(),score=f'{f1_score(t,p,average="macro"):.2f}')
                            for t,p in ((truth,guess),(second,proposed))])
    base=dest/'schema-source.zip'
    packet(base,'tabular','accuracy',x[:1000],y[:1000])
    with zipfile.ZipFile(base) as z:
        manifest=json.loads(z.read('manifest.json'));manifest.pop('metric')
        members=[(n,z.read(n)) for n in z.namelist() if n!='manifest.json']
    doc=('Predict the state from a set of measurements. Signals of different magnitudes can come from different conditions.\n'
         'Primary metric is the class recognition index defined by the worked examples.\n')
    if seed==SEEDS['transfer']:
        manifest['metric_evidence']=evidence
    else:
        doc+='\nScoring examples:\n```json\n'+json.dumps(evidence)+'\n```\n'
    make_zip(dest/'train.zip',[('manifest.json',json.dumps(manifest).encode()),('README.md',doc.encode()),*members])
    packet(dest/'input.zip','tabular','macro_f1',x[1000:])
    return y[1000:]


def prepare(root):
    root.mkdir(parents=True)
    (root/'evaluation').mkdir()
    cases={}
    for tag,seed in SEEDS.items():
        dest=root/tag;dest.mkdir()
        y=make_task(dest,seed)
        answer=root/'evaluation'/(tag+'.npy')
        np.save(answer,y,allow_pickle=False)
        cases[tag]=dict(train=sha(dest/'train.zip'),features=sha(dest/'input.zip'),answers=sha(answer),rows=len(y))
    save(root/'protocol.json',dict(agent=source_identity(ROOT),study=sha(Path(__file__)),
        base_plan=sha(ROOT/'research/reliable_delivery_1127/protocol.json'),fit_limit=5,cases=cases))


def run(root,tag,key=None,ledger=None):
    registered=read(root/'protocol.json')
    folder=root/tag
    assert registered['agent']==source_identity(ROOT) and registered['study']==sha(Path(__file__))
    spec=registered['cases'][tag]
    assert spec['train']==sha(folder/'train.zip') and spec['features']==sha(folder/'input.zip')
    paid=tag=='campus'
    settings=env((ROOT.parent/'pretrained_weights').resolve(),folder/'physical.jsonl')
    for k in ('SUPERAGENT_RESEARCH_REFLECTION_OUTBOX','SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_PLANNER_LEDGER',
              'SUPERAGENT_TASK_BUDGET_ID','SUPERAGENT_RESEARCH_PAIRED_REFLECTION','REFLECTION_STUDY_LIVE','AB_B1_PER_WORKER_FIT_LIMIT'):
        settings.pop(k,None)
    settings.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/reflection_study_observer',ROOT))),
        PROPOSAL_STUDY_OUTER_LABELS=str(root/'evaluation'),PROPOSAL_STUDY_OUTER_FEATURES=str(folder/'input.zip'),
        SUPERAGENT_LESSON_STORE=str(root/'internal-lessons.sqlite'),
        SUPERAGENT_LESSONS_READONLY='0' if tag=='learn' else '1')
    if paid:
        if not key or not ledger: raise ValueError('Campus case requires the existing billing outbox and local key file')
        scope='reliable-feedback-1127-'+root.name
        with sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True) as db:
            assert not db.execute('SELECT 1 FROM requests WHERE scope=? OR status!=?',(scope,'known')).fetchone()
        settings.update(REFLECTION_STUDY_LIVE='1',SUPERAGENT_TASK_BUDGET_ID=scope,
            SUPERAGENT_PLANNER_KEY_FILE=str(key),SUPERAGENT_RESEARCH_REFLECTION_OUTBOX=str(ledger),
            NO_PROXY='aigc-api.hkust-gz.edu.cn',no_proxy='aigc-api.hkust-gz.edu.cn')
    dest=folder/'output/p.csv'
    def call(arguments,log):
        with (folder/log).open('x') as f:
            result=subprocess.run([sys.executable,'-m','unified_agent',*map(str,arguments)],cwd=ROOT,env=settings,
                                  stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=230)
        if result.returncode: raise RuntimeError(str(folder/log))
    if tag!='learn':
        assert (root/'learn/sealed.json').is_file()
    call(['solve',folder/'train.zip','--prediction-input',folder/'input.zip','--run-dir',folder/'run',
          '--output',dest,'--fit-limit','5','--time-limit','180','--task-analysis',
          'feedback-flash' if paid else 'feedback-local'],'solve.log')
    agent=folder/'run/agent'
    opening=read(agent/'task_opening.json')
    outcome=read(agent/'run_outcome.json')
    metric=read(agent/'task_contract.json')['goal_resolution']['fields']['primary_metric']
    assert (metric['value'],metric['source'])==('macro_f1','verified_worked_examples')
    assert outcome['status']=='completed' and not outcome['formal_test_accessed'] and outcome['new_api_requests']<=int(paid)
    fits=len((folder/'physical.jsonl').read_text().splitlines())
    assert fits==outcome['attempted_classifier_fits']<=5
    first=read(folder/'run/first_delivery/prediction_manifest.json')
    assert first['prediction_count']==spec['rows'] and first['delivery_status']=='partial'
    reloaded=folder/'reloaded/p.csv'
    call(['predict',folder/'input.zip','--run-dir',folder/'run','--output',reloaded],'reload.log')
    assert sha(dest)==sha(reloaded) and len((folder/'physical.jsonl').read_text().splitlines())==fits
    usage=[json.loads(s) for s in (agent/'llm_usage.jsonl').read_text().splitlines()][1:]
    matched=False
    if paid and usage:
        with sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True) as db:
            rec=db.execute('SELECT id,status,course_units FROM requests WHERE scope=?',(scope,)).fetchall()
        matched=(len(usage)==len(rec)==1 and rec[0][:2]==(usage[0].get('budget_request_id'),'known')
                 and rec[0][2]==round((usage[0].get('course_cost_rmb') or 0)*1e9))
    info=dict(status=outcome['status'],rows=first['prediction_count'],selected=read(agent/'final_selection.json')['family'],
        physical_fits=fits,api=outcome['new_api_requests'],reloaded=True,first_sha256=first['prediction_sha256'],
        final_sha256=sha(dest),metric=metric,check=opening.get('check'),after_trial=opening.get('dispatch_after_trials'),
        prior_hints=opening['prior_compatible_checks'],status_at_decision=opening['decision_time_status'],
        hypothesis_source=opening.get('hypothesis_source_at_decision'),decision=opening['result'],
        internal_verdict=opening['check_status'],lesson=opening.get('training_lesson'),late=opening.get('late_hypothesis'),
        response_status=opening['interpretation_status'],changed_experiment=opening['changed_experiment'],
        metered_response_matched=matched,usage=[{k:v.get(k) for k in ('served_model','role','input_tokens','output_tokens','course_cost_rmb')} for v in usage])
    save(folder/'sealed.json',info)
    print(tag,json.dumps(info))


def evaluate(root):
    seals={tag:read(root/tag/'sealed.json') for tag in SEEDS if (root/tag/'sealed.json').is_file()}
    assert 'learn' in seals and 'transfer' in seals
    save(root/'predictions-frozen.json',{k:v['final_sha256'] for k,v in seals.items()})
    for tag,v in seals.items():
        truth=root/'evaluation'/(tag+'.npy');assert sha(truth)==read(root/'protocol.json')['cases'][tag]['answers']
        y=np.load(truth,allow_pickle=False);p=root/tag/'output/p.csv';assert sha(p)==v['final_sha256']
        with p.open(newline='') as f: rows=list(csv.DictReader(f))
        assert [r['row_id'] for r in rows]==list(map(str,range(len(y))))
        v['external_development_macro_f1']=f1_score(y,[int(r['prediction']) for r in rows],average='macro')
    save(root/'results.json',seals)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=('prepare','run','score'))
    parser.add_argument('output',type=lambda p:Path(p).resolve())
    parser.add_argument('--case',choices=SEEDS)
    for name in ('key','ledger'):parser.add_argument('--'+name,type=lambda p:Path(p).resolve())
    a=parser.parse_args()
    if a.step=='prepare': prepare(a.output)
    elif a.step=='run' and a.case:run(a.output,a.case,a.key,a.ledger)
    elif a.step=='score':evaluate(a.output)
    else:parser.error('Choose a case')
