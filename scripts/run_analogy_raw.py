"""Run the equal-budget sealed raw control and challenger on independent source trees."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.compare_regrouped_portfolios import lock
from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, archive, env, read, require, sha
from scripts.verify_unified_raw import MEMBERS


PLAN=ROOT/'research/paired_encoder_search_1127/analogy_raw_protocol.json'


def run(root, impl, source, features, arm):
    limit=16
    if arm=='recruitment_raw':
        study=ROOT/'research/paired_encoder_search_1127/recruitment_raw_protocol.json'
        pair=read(root/'portfolio-recruitment/comparison/results.json')
        transfer=read(root/'portfolio-recruitment/table/results.json')
        replay=read(root/'recruitment-replay/results.json')['tasks']
        require(pair['paired_accuracy']['protected']['net']>=0 and transfer['received_score_after_sealing']
                and all(v['matched']==5 and v['neutral_retained'] and v['one_fit_short_retained'] for v in replay.values())
                and source_identity(impl)==read(root/'portfolio-recruitment/inputs.json')['challenger'],
                'Registered transfer and autonomy conditions have not passed')
        folder=root/'recruitment-raw'/arm
        limit=32
    elif arm=='corrected_delivery':
        study=ROOT/'research/paired_encoder_search_1127/corrected_delivery_protocol.json'
        folder=root/'corrected-delivery'/arm
        limit=22
    elif arm.startswith('portfolio_'):
        study=ROOT/'research/paired_encoder_search_1127/portfolio_adviser_raw_protocol.json'
        require(read(root/'analogy-portfolio-complement/results.json')['advance'],'Portfolio study did not earn a raw comparison')
        folder=root/'analogy-portfolio-raw'/arm
        limit=22
    elif arm=='close':
        study=ROOT/'research/paired_encoder_search_1127/analogy_near_tie_protocol.json'
        evidence=read(root/'analogy-close-confirm/results.json')
        require(evidence['advance'] and evidence['protocol']==sha(study),'The registered trial has not earned a full run')
        folder=root/'analogy-close-raw'/arm
    else:
        study=PLAN
        folder=root/'analogy-raw'/arm
    folder.mkdir(parents=True,exist_ok=False)
    raw=archive(source,'titles')
    original=read(ROOT/'research/b_task2_handoff_1127/protocol.json')['input']
    require(sha(raw)==original['raw'] and sha(features)==original['features'],'Changed raw packages')
    weights=root/'discovery-library'
    fitlog=folder/'physical.jsonl'
    opened=folder/'opened.jsonl'
    settings=env(weights,fitlog)
    for k in list(settings):
        if k.startswith('SUPERAGENT_') and (k.endswith('_WEIGHTS') or k in ('SUPERAGENT_WEIGHTS_DIR','SUPERAGENT_PLANNER_KEY_FILE',
            'SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_TASK_BUDGET_ID','SUPERAGENT_RESEARCH_REFLECTION_OUTBOX')):
            del settings[k]
    settings.update(PYTHONPATH=os.pathsep.join(map(str,(impl/'scripts/unified_acceptance',impl))),
                    SUPERAGENT_PAIRED_FEATURE_CACHE=str(root/'feature_bank'),UNIFIED_RAW_INPUT=str(raw),
                    UNIFIED_RAW_MEMBERS=json.dumps(MEMBERS['titles']),UNIFIED_RAW_LOG=str(opened))
    protocol=dict(runtime_sha256=source_identity(impl),plan_sha256=sha(study),arm=arm,fit_limit=limit,wall_seconds=1800,
                  model_hint=None,policy_hint=None,raw_sha256=sha(raw),features_sha256=sha(features))
    if arm in ('original','portfolio_control'):
        require(protocol['runtime_sha256']==read(root/'discovered-ordinary/discovered_default/protocol.json')['runtime_sha256'],'Unexpected control runtime')
    lock(folder/'protocol.json',protocol)
    run_dir=folder/'run'
    prediction=folder/'delivery/predictions.csv'
    replay=folder/'reloaded.csv'
    for name,args in [('solve',['solve',raw,'--fit-limit',limit,'--time-limit',1800,'--prediction-input',features,'--output',prediction]),
                      ('predict',['predict',features,'--output',replay])]:
        if name=='predict':
            settings.pop('SUPERAGENT_PAIRED_FEATURE_CACHE',None)
        with (folder/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-m','unified_agent',*map(str,args),'--run-dir',str(run_dir),'--weights-dir',str(weights)],
                           cwd=impl,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=settings,timeout=1990,check=True)
    agent=run_dir/'agent'
    outcome,selected=read(agent/'run_outcome.json'),read(agent/'final_selection.json')
    observed=sorted({json.loads(line) for line in opened.read_text().splitlines()})
    count=len(fitlog.read_text().splitlines())
    require(outcome['status']=='completed' and selected['refit'] and selected['training_rows']==8301 and observed==sorted(MEMBERS['titles'])
            and count==outcome['attempted_classifier_fits']==outcome['completed_classifier_fits']<=limit and outcome['new_api_requests']==0
            and sha(prediction)==sha(replay) and sha(agent/selected['model_file'])==selected['model_sha256'],'Incomplete raw delivery')
    lock(folder/'sealed.json',dict(protocol_sha256=sha(folder/'protocol.json'),prediction_sha256=sha(prediction),model_sha256=selected['model_sha256'],
        members=outcome['members'],rows=selected['training_rows'],fits=count,opened=observed,independent_reload=True,api=outcome['new_api_requests']))
    if arm in ('corrected_delivery','recruitment_raw'):
        prior=ROOT/'research/paired_encoder_search_1127/portfolio_adviser_seal.json'
        scored=ROOT/'research/paired_encoder_search_1127/portfolio_adviser_score.json'
        earlier=read(scored)['completed']['portfolio_revised']
        with prediction.open() as stream:
            rows=sum(1 for _ in stream)-1
        require(rows==earlier['rows'],'Incomplete prediction output')
        identical=(sha(prediction)==read(prior)['prediction_sha256'])
        lock(folder/'comparison.json',dict(input_seal=sha(folder/'sealed.json'),original_seal=sha(prior),previous_score=sha(scored),
             rows=rows,identical_predictions=identical,carried_correct=earlier['correct'] if identical else None,
             carried_accuracy=earlier['accuracy'] if identical else None,designated_labels_opened=False,
             official_test_opened=False,H2_opened=False))
        if arm=='recruitment_raw':
            before=root/'corrected-delivery/corrected_delivery/delivery/predictions.csv'
            require(sha(before)==read(prior)['prediction_sha256'],'Changed release comparison')
            with before.open(newline='') as a,prediction.open(newline='') as b:
                left,right=list(csv.DictReader(a)),list(csv.DictReader(b))
            require([r['row_id'] for r in left]==[r['row_id'] for r in right], 'Row comparison is misaligned')
            lock(folder/'unlabelled_changes.json',dict(comparison=sha(folder/'comparison.json'),
                 changed=sum(x['prediction']!=y['prediction'] for x,y in zip(left,right)),rows=len(right),
                 recruitment=sha(agent/'portfolio_review.json'),validation_labels_opened=False))
    print(arm,count,outcome['members'],read(folder/'sealed.json')['prediction_sha256'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','impl','course','features'):
        parser.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    parser.add_argument('--arm',choices=('original','revised','close','portfolio_control','portfolio_revised','corrected_delivery','recruitment_raw'),required=True)
    args=parser.parse_args()
    run(args.root,args.impl,args.course,args.features,args.arm)
