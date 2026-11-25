"""Freeze matched primary-weight counterfactuals, then independently check the study and its spend."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from scripts.run_conditional_review_study import get_predictions
from scripts.run_proposal_value_study import CASES, MODES
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from scripts.verify_feedback_reflection_study import independent_score
from superagent.data import load_prediction_features
from superagent.experiment_policy import identity
from superagent.models import SoftVoteModel, predict_with_model


PLAN=ROOT/'research/proposal_value_1126/diagnostic_protocol.json'


def freeze(root):
    frozen={}
    for case in CASES:
        frozen[case]={}
        folder=root/case
        for mode in MODES:
            work=folder/mode
            agent=work/'run/agent'
            r=read(work/'sealed.json')['review']
            h=read(agent/'history.json')
            paths={t['candidate']:agent/f"trial_{t['trial']}.joblib" for t in h if t['status']=='completed' and t['split_id']=='primary'}
            if (agent/'proposal_primary.joblib').is_file():
                paths[r['tested_candidate']]=agent/'proposal_primary.joblib'
            groups=dict(checkpoint=[r['backup']],core=r['core_members'],pre_guard=r['pre_guard_members'],adopted=r['final_members'])
            if r.get('tested_candidate') in paths:
                groups['tested']=[r['tested_candidate']]
            schema=read(agent/'task_spec.json')
            kind=schema['model_kind']
            query=load_prediction_features(str(folder/'input.zip'),kind,schema)
            primary_hash=read(agent/'primary_split.json')['payload_sha256']
            out=work/'matched_primary'
            out.mkdir(exist_ok=True)
            known={}
            records={}
            for name,members in groups.items():
                slug=identity(members)
                if slug not in known:
                    models=[]
                    for m in members:
                        bundle=joblib.load(paths[m])
                        require(bundle['metrics']['resolved_configuration']['id']==m and
                                bundle['prediction_record']['evaluation_payload_sha256']==primary_hash,'Invalid paired training model')
                        models.append(bundle['model'])
                    model=models[0] if len(models)==1 else SoftVoteModel(kind,models,np.array(schema['metric_spec']['labels']))
                    p=predict_with_model(kind,model,query)
                    target=out/(slug+'.csv')
                    if not target.exists():
                        with target.open('x',newline='') as f:
                            writer=csv.writer(f)
                            writer.writerow(['row_id','prediction'])
                            writer.writerows(enumerate(p.tolist()))
                    require(np.array_equal(get_predictions(target,p,sha(target)),p),'Stored counterfactual changed')
                    known[slug]=dict(path=str(target.relative_to(root)),sha=sha(target),members=members)
                records[name]=known[slug]
            frozen[case][mode]=records
    save(root/'matched_primary_frozen.json',dict(protocol=sha(PLAN),results=frozen))


def evaluate(a):
    root=a.root
    main=read(root/'completion.json')
    frozen=read(root/'matched_primary_frozen.json')
    require(frozen['protocol']==sha(PLAN),'Matched analysis plan changed')
    primary=read(root/'frozen_predictions.json')
    report={}
    scores_checked=0
    fitlog=0
    for case,item in main['cases'].items():
        y=np.load(root/'evaluation'/(case+'.npy'),allow_pickle=False)
        metric=item['metric']
        per_arm={}
        for mode,m in item['arms'].items():
            work=root/case/mode
            agent=work/'run/agent'
            for cut,info in primary[case][mode].items():
                if info:
                    score=independent_score(metric,y,get_predictions(root/info['path'],y,info['sha']))
                    require(abs(score-m['scores'][cut])<1e-12,'Independent outer score differs')
                    scores_checked+=1
            rr=m['review']
            trace=read(agent/'classifier_fits.json')
            require(sum(t['attempted_classifier_fits'] for t in trace['workers'])==m['fits'],'Worker count differs')
            fitlog+=m['fits']
            tail_fits=sum(t['attempted_classifier_fits'] for t in trace['workers'] if t['artifact'].startswith('proposal_'))
            s={}
            for name,info in frozen['results'][case][mode].items():
                p=get_predictions(root/info['path'],y,info['sha'])
                s[name]=independent_score(metric,y,p)
            published=read(agent/'proposal_checkpoints.json')
            require(all(b['elapsed']>=a0['elapsed'] and b['completed']>=a0['completed'] for a0,b in zip(published,published[1:])),
                    'Nonmonotonic model publication')
            require(rr['final_members']==frozen['results'][case][mode]['adopted']['members'],'Wrong deployed family')
            per_arm[mode]=dict(scores=s,tail_physical_fits=tail_fits,challenger_observed='tested' in s,
                  screened_out=bool('tested' in s and [rr['tested_candidate']]!=rr['pre_guard_members']),
                  screened_candidate_minus_core=(s['tested']-s['core'] if 'tested' in s else None),
                  retained_checkpoint_delta=s['adopted']-s['pre_guard'])
        reviews=[m['review'] for m in item['arms'].values()]
        for k in ('core_members','backup','local_alternative'):
            require(len({identity(r[k]) for r in reviews})==1,'Local alternatives differ before intervention')
        report[case]=per_arm
    require(fitlog==main['actual_fits'],'Batch fitting total differs')
    save(a.output,dict(protocol=sha(PLAN),independent_scores_checked=scores_checked,actual_fits_verified=fitlog,
                       matched_primary=report,diagnostic_only=True,new_fits=0,new_api_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('freeze','score'))
    p.add_argument('root',type=lambda x:Path(x).resolve())
    p.add_argument('--weights',type=lambda x:Path(x).resolve())
    p.add_argument('--output',type=lambda x:Path(x).resolve())
    a=p.parse_args()
    if a.stage=='score':
        evaluate(a)
    elif os.getenv('AB_B1_OFFLINE')=='1':
        freeze(a.root)
    else:
        fitlog=a.root/'unexpected_diagnostic_fits.jsonl'
        settings=env(a.weights,fitlog)
        settings.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/reflection_study_observer',ROOT))),
                        PROPOSAL_STUDY_OUTER_LABELS=str(a.root/'evaluation'))
        subprocess.run([sys.executable,'-m','scripts.audit_proposal_value',*sys.argv[1:]],cwd=ROOT,env=settings,check=True)
        require(not fitlog.exists() or not fitlog.read_text().strip(),'Unexpected fitting in the matched audit')
