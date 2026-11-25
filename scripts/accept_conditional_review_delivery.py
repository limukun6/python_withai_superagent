"""Exercise both real delivery outcomes on an archived disagreement; model-reviewer advice is a fixed fixture."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib

from scripts.unified_released_study import ROOT, env, read, require, save, sha
from superagent.ab_run import Executor
from superagent.autonomous_run import installed_methods, planner_observation, freeze_model
from superagent.conditional_review import FAIL, PASS, review_delivery
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.metrics import MetricSpec
from superagent.open_methods import capability


class ContractFixture:
    def ask(self,request,log,role,*,cost_ceiling):
        options=request['evidence']['options']
        require(len(options)==1 and role=='feedback_reflection','Unexpected fixture decision')
        return dict(decision='measure',option=0,cited_folds=[r['fold'] for r in options[0]['comparisons']],
                    if_pass=PASS,if_fail=FAIL,minimum_delta=None,reason='The first two comparisons disagree.',
                    opportunity_cost='One real measurement and a reserved full-data refit.')


def run(old,root):
    original=old/'r44/control/run/agent'
    task=load_training_task(old/'r44/train.zip',read_declarations=True)
    metric=MetricSpec.for_task(task)
    stop=read(original/'stop_review.json')
    before,other=stop['incumbent'][0],stop['candidate']
    history=read(original/'history.json')
    registry=read(original/'decisions.json')[-1]
    require(sha(original/registry['registry'])==registry['registry_sha256'],'Archive registry changed')
    entries=read(original/registry['registry'])['entries']
    caps={n:capability(entries[n]) for n in (before,other)}
    names=['primary','confirmation_1','confirmation_2']
    folds,payloads,observations={},{},{}
    for f in names:
        metadata=read(original/(f+'_split.json'))
        path=original/(f+'.joblib')
        require(sha(path)==metadata['payload_sha256'],'Archive fold changed')
        folds[f]=(metadata['fit'],metadata['held'],metadata['resource'])
        payloads[f]=dict(path=path,sha256=sha(path),labels=joblib.load(path)['y_search'])
        trial=next(row['trial'] for row in history if row['candidate']==before and row['split_id']==f)
        alternative=(old/'third_fold/r44/third_challenger.joblib' if f=='confirmation_2' else original/('challenge_'+f+'.joblib'))
        observations[f]={}
        for member,bundle_path in [(before,original/f'trial_{trial}.joblib'),(other,alternative)]:
            bundle=joblib.load(bundle_path)
            require(bundle['metrics']['resolved_configuration']['id']==member,'Wrong archived worker')
            observations[f][member]=planner_observation(bundle,bundle['metrics'],payloads[f],metric)
    outcomes={}
    for label,leader in [('recorded',before),('reversed',other)]:
        folder=root/label
        folder.mkdir()
        for n in ('refit_data.joblib','task_spec.json'):
            os.link(original/n,folder/n)
        os.link(original/registry['registry'],folder/'candidate_registry.json')
        log=RunLogger(folder)
        budget=2 if label=='recorded' else 3
        executor=Executor(folder,caps,metric,time.monotonic(),240.,90.,budget,log)
        spec=read(original/'task_contract.json')
        choice=dict(members=[leader],utility=observations['primary'][leader]['score'],
                    evidence={'scores':{f:observations[f][leader]['score'] for f in names[:2]}})
        if label=='recorded':
            selection=read(original/'incumbent_selection.json')
            os.link(original/selection['model_file'],folder/selection['model_file'])
        else:
            selection=freeze_model(task,spec,metric,choice,executor,folder,log,90.,challenger=True,challenger_prefix='starting_')
        save(folder/'final_selection.json',selection)
        bank={f:dict(observations[f]) for f in names[:2]}
        bank[names[2]]={leader:observations[names[2]][leader]}
        selected,_=review_delivery(task,spec,metric,selection,choice,executor,folder,log,90.,bank,payloads,folds,ContractFixture(),{})
        rec=read(folder/'conditional_review.json')
        expected=FAIL if label=='recorded' else PASS
        require(rec['executed']==expected and rec['status'] in ('measured','adopted') and
                sha(folder/selected['model_file'])==selected['model_sha256']==read(folder/'final_selection.json')['model_sha256'],
                'Conditional decision did not reach the selected artifact')
        pred=folder/'output/predictions.csv'
        subprocess.run([sys.executable,'-m','superagent','predict',str(old/'r44/input.zip'),'--run-dir',str(folder),
                        '--output',str(pred)],cwd=ROOT,check=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL)
        require(sha(pred)==sha(old/'r44/control/delivery/predictions.csv'),'Reloaded delivery differs')
        outcomes[label]=dict(executed=expected,third=rec['third'],fits=executor.accounting()['attempted_classifier_fits'],
                             final_source=selected['model_sha256'],predictions=sha(pred))
    fits=len((root/'physical.jsonl').read_text().splitlines())
    require(sum(r['fits'] for r in outcomes.values())==fits==4,'Unexpected physical count')
    save(root/'completion.json',dict(outcomes=outcomes,fits=fits,api_calls=0,reviewer='deterministic contract fixture',
                                     used_outer_labels=False,protocol=sha(ROOT/'research/conditional_review_1125/branch_acceptance_protocol.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('old',type=lambda s:Path(s).resolve())
    p.add_argument('root',type=lambda s:Path(s).resolve())
    p.add_argument('--weights',required=True,type=lambda s:Path(s).resolve())
    p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.worker:
        with installed_methods():
            run(a.old,a.root)
    else:
        a.root.mkdir(parents=True)
        subprocess.run([sys.executable,'-m','scripts.accept_conditional_review_delivery',str(a.old),str(a.root),
                        '--weights',str(a.weights),'--worker'],cwd=ROOT,env=env(a.weights,a.root/'physical.jsonl'),
                       stdin=subprocess.DEVNULL,check=True)
