"""Compare complete ordinary agent runs on the existing film-disjoint memorability task."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT,env,read,require,sha


PLAN=ROOT/'research/paired_encoder_search_1127/analogy_movie_transfer_protocol.json'
ARMS=('original','revised')


def solve(root,source,impl,arm):
    protocol=read(source/'protocol.json')
    task,features=source/'anonymous.zip',source/'features.zip'
    require(sha(task)==protocol['train_packet'] and sha(features)==protocol['features'] and
            not any(protocol[k] for k in ('shared_outer_components','shared_outer_texts','shared_outer_films')),'Invalid movie input split')
    out=root/'movie-adviser'/arm
    out.mkdir(parents=True,exist_ok=False)
    log=out/'physical.jsonl'
    weights=root/'discovery-library'
    settings=env(weights,log)
    for k in list(settings):
        if k.startswith('SUPERAGENT_'):
            del settings[k]
    settings['PYTHONPATH']=os.pathsep.join(map(str,(impl/'scripts/ab_acceptance_guard',impl)))
    require(source_identity(impl)==read(root/('analogy-portfolio-raw/portfolio_'+('control' if arm=='original' else 'revised'))/'protocol.json')['runtime_sha256'],'Wrong source for transfer arm')
    lock(out/'launch.json',dict(protocol=sha(PLAN),runtime=source_identity(impl),source_identity=sha(source/'protocol.json'),
         train=sha(task),features=sha(features),fit_limit=16,wall_seconds=1800,model_hint=None,policy_hint=None))
    output=out/'prediction.csv'
    replay=out/'reload.csv'
    run=out/'run'
    for label,args in [('solve',['solve',task,'--fit-limit',16,'--time-limit',1800,'--prediction-input',features,'--output',output]),
                       ('predict',['predict',features,'--output',replay])]:
        with (out/(label+'.log')).open('x') as f:
            subprocess.run([sys.executable,'-m','unified_agent',*map(str,args),'--run-dir',str(run),'--weights-dir',str(weights)],cwd=impl,
                env=settings,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=1980,check=True)
    agent=run/'agent'
    selection=read(agent/'final_selection.json')
    outcome=read(agent/'run_outcome.json')
    fits=len(log.read_text().splitlines())
    require(outcome['status']=='completed' and selection['refit'] and sha(agent/selection['model_file'])==selection['model_sha256']
            and selection['training_rows']==protocol['train_rows'] and fits==outcome['attempted_classifier_fits']==outcome['completed_classifier_fits']<=16
            and outcome['new_api_requests']==0 and sha(output)==sha(replay),'Incomplete movie delivery')
    review=read(agent/'analogy_review.json') if (agent/'analogy_review.json').exists() else {}
    result=dict(protocol=sha(out/'launch.json'),model=selection['model_sha256'],predictions=sha(output),independent_reload=True,
         members=outcome['members'],fits=fits,training_rows=selection['training_rows'],adviser=review.get('status'),locked_share=review.get('locked_weight'),
         initial_choices=len(read(agent/'decisions.json')[0]['alternatives']),api=0)
    lock(out/'sealed.json',result)
    print(arm,result,flush=True)


def score(root,source):
    folder=root/'movie-adviser'
    plan=read(source/'protocol.json')
    seals={}
    pred={}
    for arm in ARMS:
        current=folder/arm
        seals[arm]=read(current/'sealed.json')
        require(sha(current/'prediction.csv')==seals[arm]['predictions'] and seals[arm]['independent_reload'],'Movie seal changed')
        with (current/'prediction.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        require([int(r['row_id']) for r in rows]==list(range(plan['held_rows'])),'Prediction order or row count changed')
        pred[arm]=np.asarray([int(r['prediction']) for r in rows])
    lock(folder/'evaluation_lock.json',dict(protocol=sha(PLAN),external_score_file=plan['scoring'],seals={arm:sha(folder/arm/'sealed.json') for arm in ARMS}))
    scoring=source/'scoring_only.json'
    require(sha(scoring)==plan['scoring'],'Evaluation labels changed')
    eval_data=read(scoring)
    y=np.asarray(eval_data['labels']); groups=np.asarray(eval_data['components'])
    require(len(y)==plan['held_rows'] and len(np.unique(groups))==plan['held_components'],'Movie grouping changed')
    right={arm:p==y for arm,p in pred.items()}
    a,b=[right[k] for k in ARMS]
    fixed=int((b&~a).sum());lost=int((a&~b).sum())
    _,idx=np.unique(groups,return_inverse=True)
    delta=np.bincount(idx,weights=b.astype(float)-a)
    result=dict(protocol=sha(folder/'evaluation_lock.json'),fitting_films=plan['train_films'],held_films=plan['held_films'],rows=len(y),
         arms={arm:dict(correct=int(mask.sum()),evaluation_rows=len(y),accuracy=float(mask.mean()),**seals[arm]) for arm,mask in right.items()},
         revised_vs_original=dict(gained=fixed,lost=lost,net=fixed-lost,groups_ahead=int((delta>0).sum()),
                                  groups_behind=int((delta<0).sum()),groups_tied=int((delta==0).sum())),
         local_classifier_fits=sum(s['fits'] for s in seals.values()),model_api_calls=0,course_evaluation_opened=False,H2_opened=False)
    lock(folder/'results.json',result)
    print({a:(row['correct'],row['fits'],row['adviser']) for a,row in result['arms'].items()},result['revised_vs_original'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('solve','score'))
    for key in ('root','source','impl'):
        p.add_argument('--'+key,type=lambda x:Path(x).resolve(),required=key!='impl')
    p.add_argument('--arm',choices=ARMS)
    a=p.parse_args()
    if a.mode=='solve':
        solve(a.root,a.source,a.impl,a.arm)
    else:
        score(a.root,a.source)
