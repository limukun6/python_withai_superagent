"""Measure the training claims of the actual campus suggestions including late replies."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np

from scripts.run_proposal_value_study import CASES, MODES
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from superagent.ab_run import Executor, registered_capabilities
from superagent.autonomous_run import installed_methods, planner_observation
from superagent.events import RunLogger
from superagent.experiment_policy import identity
from superagent.metrics import MetricSpec
from superagent.open_methods import capability


PLAN=ROOT/'research/proposal_value_1126/hypothesis_audit_protocol.json'


def audit(root,out):
    summary={}
    for case in CASES:
        work=root/case/MODES[2]
        r=read(work/'sealed.json')
        review=r['review']
        plan=review.get('plan',{})
        row=dict(reply=review['reply_status'],decision=plan.get('decision'),used_live=review.get('proposal_used',False),new_fits=0)
        summary[case]=row
        if plan.get('decision')!='experiment':
            continue
        agent=work/'run/agent'
        pkt=read(agent/'experiment_proposal_input.json')
        require(identity(pkt)==review['input_digest'] and sorted(plan['cited_trials'])==sorted(pkt['reference_trials']),'Wrong frozen evidence')
        name=plan['candidate']
        row.update(method=name,same_as_local_tail=name==review.get('local_alternative'),min_gain=plan['min_gain'],min_rescues=plan['min_rescues'])
        h=read(agent/'history.json')
        paths={t['candidate']:agent/f"trial_{t['trial']}.joblib" for t in h if t['status']=='completed' and t['split_id']=='primary'}
        if (agent/'proposal_primary.joblib').exists():
            paths[review['tested_candidate']]=agent/'proposal_primary.joblib'
        data=agent/'primary.joblib'
        payload=dict(path=data,labels=joblib.load(data)['y_search'],sha256=read(agent/'primary_split.json')['payload_sha256'])
        require(sha(data)==payload['sha256'],'Original primary split changed')
        spec=read(agent/'task_contract.json')
        metric=MetricSpec.from_payload(read(agent/'task_spec.json'))
        if name not in paths:
            folder=root/'hypothesis_audit'/case
            folder.mkdir(parents=True,exist_ok=True)
            registry=agent/'candidate_registry.json'
            if not (folder/'candidate_registry.json').exists():
                os.link(registry,folder/'candidate_registry.json')
            reg=read(registry)['entries']
            if name in reg:
                cap=capability(reg[name])
            else:
                inputs=joblib.load(data)['X_train']
                with installed_methods():
                    cap=registered_capabilities(spec,len(inputs),numeric_kernel_width=inputs.shape[1] if spec['kind']=='numeric_tabular' else None).get(name)
            if not cap or cap['classifier_fits']!=1 or r['fits']>=16:
                row['status']='unmeasured_under_remaining_limit'
                continue
            path=folder/'proposal.joblib'
            if not path.exists():
                executor=Executor(folder,{name:cap},metric,time.monotonic(),180,160,1,RunLogger(folder))
                with installed_methods():
                    executor.worker(data,name,path,160)
            report=read(folder/'classifier_fits.json')
            require(report['attempted_classifier_fits']==report['completed_classifier_fits']==1,'Supplemental fitting mismatch')
            paths[name]=path
            row['new_fits']=1
        def observe(path):
            bundle=joblib.load(path)
            return planner_observation(bundle,bundle['metrics'],payload,metric)
        left,right=[observe(agent/f'trial_{number}.joblib') for number in pkt['reference_trials']]
        v=observe(paths[name])
        y=payload['labels']
        shared=(left['predictions']!=y)&(right['predictions']!=y)
        require(int(sum(shared))==pkt['shared_errors'] and abs(left['score']-pkt['reference_score'])<1e-12,'Mismatched reference results')
        rescued=int(sum(shared & (v['predictions']==y)))
        delta=v['score']-left['score']
        row.update(status='measured',rescued=rescued,shared=int(sum(shared)),primary_gain=delta,
                   spoiled_vs_leader=int(sum((left['predictions']==y)&(v['predictions']!=y))),
                   passed=bool(rescued>=plan['min_rescues'] and delta+1e-12>=plan['min_gain']))
        if review.get('proposal_used') and 'hypothesis_passed' in review:
            require(row['passed']==review['hypothesis_passed'],'Deployed hypothesis trace differs from the replay')
    fits=sum(r['new_fits'] for r in summary.values())
    fitlog=root/'supplemental_fits.jsonl'
    require(fits==(len(fitlog.read_text().splitlines()) if fitlog.exists() else 0)<=4,'Unaccounted supplemental fit')
    save(out,dict(protocol=sha(PLAN),cases=summary,additional_fits=fits,additional_api_calls=0,outer_labels_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=lambda x:Path(x).resolve())
    p.add_argument('output',type=lambda x:Path(x).resolve())
    p.add_argument('--weights',type=lambda x:Path(x).resolve(),required=True)
    a=p.parse_args()
    if os.getenv('AB_B1_OFFLINE')=='1':
        audit(a.root,a.output)
    else:
        settings=env(a.weights,a.root/'supplemental_fits.jsonl')
        settings.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/reflection_study_observer',ROOT))),
                        PROPOSAL_STUDY_OUTER_LABELS=str(a.root/'evaluation'))
        subprocess.run([sys.executable,'-m','scripts.audit_proposal_hypotheses',*sys.argv[1:]],cwd=ROOT,env=settings,check=True)
