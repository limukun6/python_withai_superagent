"""Score both preregistered, independently reloaded raw deliveries with at most one new validation read."""
import argparse
from pathlib import Path

import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.score_followup_entry import prediction_file
from scripts.unified_released_study import archive, read, require, sha, title_rows


def score(root,course):
    source=root/'analogy-raw'
    output=root/'analogy-raw-scores'
    output.mkdir(exist_ok=True)
    cases=('original','revised')
    predictions={}
    seals={}
    for arm in cases:
        folder=source/arm
        seals[arm],predictions[arm]=prediction_file(folder)
        require(seals[arm]['protocol_sha256']==sha(folder/'protocol.json') and seals[arm]['fits']<=16
                and seals[arm]['api']==0 and seals[arm]['rows']==8301,'Incomplete treatment')
    baseline=root/'discovered-ordinary/discovered_default'
    base_seal,base_predictions=prediction_file(baseline)
    originals=[('discovered_default','discovered-ordinary/discovered_default','discovered-ordinary-scores'),
               ('discovered','discovered-raw/discovered','discovered-raw-scores'),('resource_default','resource-raw/resource_default','delivery_scores'),
               ('legacy_long','long-cold/original','delivery_scores'),('auto_portfolio','auto-raw/auto_portfolio','delivery_scores'),
               ('frontier','frontier-raw/frontier','delivery_scores'),('frontier_scale','frontier-raw/frontier_scale','delivery_scores')]
    prior={}
    for name,relative,scoring in originals:
        folder=root/relative
        scored=read(root/scoring/'results.json')
        fix=root/scoring/'evaluation_lock.json'
        require(sha(fix)==scored['lock_sha256'] and read(fix)['seals'][name]==sha(folder/'sealed.json'),'Broken previous score linkage')
        record=read(folder/'sealed.json')
        prior.setdefault(record['prediction_sha256'],(name,scored['completed'][name]['correct']))
    default_correct=prior[base_seal['prediction_sha256']][1]
    source_protocol=read(source/'original/protocol.json')
    revised=read(source/'revised/protocol.json')
    require(all(source_protocol[k]==revised[k] for k in ('plan_sha256','raw_sha256','features_sha256','fit_limit','wall_seconds','model_hint','policy_hint')),'Unequal conditions')
    lock(output/'evaluation_lock.json',dict(seals={a:sha(source/a/'sealed.json') for a in cases},
        training_plan=source_protocol['plan_sha256'],base_receipt=sha(baseline/'sealed.json'),reference_scores={s:sha(root/s/'results.json') for _,_,s in originals}))
    need_labels=any(v['prediction_sha256'] not in prior for v in seals.values())
    if need_labels:
        raw=archive(course,'titles')
        require(sha(raw)==source_protocol['raw_sha256'],'Changed designated raw archive')
        truth=np.asarray([r[2] for r in title_rows(raw,'val',labels=True)])
        require(len(truth)==len(base_predictions) and int(np.sum(base_predictions==truth))==default_correct,'Stored default result does not match')
    completed={}
    def comparison(pred,count,base,base_count):
        differences=int((pred!=base).sum())
        twice=differences+count-base_count
        require(twice%2==0 and 0<=twice<=2*differences,'Inconsistent paired counts')
        return dict(changed=differences,gained=twice//2,lost=differences-twice//2)
    for arm in cases:
        digest=seals[arm]['prediction_sha256']
        historical,saved=prior.get(digest,(None,None))
        pct=saved if saved is not None else int(np.sum(predictions[arm]==truth))
        completed[arm]=dict(rows=len(base_predictions),correct=pct,accuracy=pct/len(base_predictions),fits=seals[arm]['fits'],
             members=seals[arm]['members'],matches_existing_score=historical,
             vs_ordinary=comparison(predictions[arm],pct,base_predictions,default_correct))
    effect=comparison(predictions['revised'],completed['revised']['correct'],predictions['original'],completed['original']['correct'])
    result=dict(lock_sha256=sha(output/'evaluation_lock.json'),completed=completed,paired_vs_equal_budget=effect,
          designated_validation_new_read=need_labels,official_test_opened=False,H2_opened=False,new_model_requests=0)
    lock(output/'results.json',result)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','course'):
        parser.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    a=parser.parse_args()
    score(a.root,a.course)
