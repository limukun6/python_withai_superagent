"""Score a single, separately registered near-tie study after the full raw model is sealed."""
import argparse
from pathlib import Path

import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.score_followup_entry import prediction_file
from scripts.unified_released_study import archive, read, require, sha, title_rows


def run(root,course):
    box=root/'analogy-close-raw/close'
    out=root/'analogy-close-scores'
    out.mkdir(exist_ok=True)
    seal,prediction=prediction_file(box)
    raw_plan=read(box/'protocol.json')
    pilot=read(root/'analogy-close-confirm/results.json')
    research=root/'analogy-raw-scores'
    reference=read(research/'results.json')
    check=read(research/'evaluation_lock.json')
    require(sha(research/'evaluation_lock.json')==reference['lock_sha256'] and sha(box/'protocol.json')==seal['protocol_sha256']
            and raw_plan['plan_sha256']==pilot['protocol'] and pilot['advance'] and seal['api']==0 and seal['fits']<=16,'Incomplete raw comparison')
    paths={k:root/'analogy-raw'/k for k in ('original','revised')}
    control=read(paths['original']/'protocol.json')
    require(all(raw_plan[k]==control[k] for k in ('raw_sha256','features_sha256','fit_limit','wall_seconds','model_hint','policy_hint')),'Changed conditions')
    prior={}
    for k,d in paths.items():
        require(sha(d/'sealed.json')==check['seals'][k],'Broken comparator seal')
        old,votes=prediction_file(d)
        prior[k]=(old['prediction_sha256'],votes,reference['completed'][k]['correct'])
    decision=read(box/'run/agent/analogy_review.json')
    require(decision['status'] in ('accepted','retained_confirmation','retained_primary'),'An incomplete review cannot be scored')
    lock(out/'evaluation_lock.json',dict(seals={'close':sha(box/'sealed.json')},previous_score=sha(research/'results.json'),
         previous_lock=sha(research/'evaluation_lock.json'),training_decision=sha(box/'run/agent/analogy_review.json'),
         pilot=sha(root/'analogy-close-confirm/results.json')))
    matches=[k for k,(digest,_,_) in prior.items() if digest==seal['prediction_sha256']]
    if matches:
        count=prior[matches[0]][2]
    else:
        raw=archive(course,'titles')
        require(sha(raw)==raw_plan['raw_sha256'],'Raw course package changed')
        truth=np.asarray([v[2] for v in title_rows(raw,'val',labels=True)])
        for _,p,old_count in prior.values():
            require(len(p)==len(truth) and int(np.sum(p==truth))==old_count,'Cannot reproduce recorded reference result')
        count=int(np.sum(prediction==truth))
    paired={}
    for k,(_,p,old_count) in prior.items():
        differences=int((p!=prediction).sum())
        both=differences+count-old_count
        require(0<=both<=2*differences and both%2==0,'Invalid paired counts')
        paired[k]=dict(gained=both//2,lost=differences-both//2,net=count-old_count)
    final=dict(correct=count,rows=len(prediction),accuracy=count/len(prediction),fits=seal['fits'],members=seal['members'],
               weights=decision.get('weights'),matched_existing=matches,paired=paired)
    lock(out/'results.json',dict(lock_sha256=sha(out/'evaluation_lock.json'),completed={'close':final},
        new_designated_validation_read=not matches,new_model_requests=0,official_test_opened=False,H2_opened=False))
    print(final,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('root','course'):
        parser.add_argument('--'+k,type=lambda s:Path(s).resolve(),required=True)
    args=parser.parse_args()
    run(args.root,args.course)
