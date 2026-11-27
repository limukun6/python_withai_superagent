"""Replay core aliases and the adviser nomination from row-aligned, training-only evidence."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.audit_choice_aliases import audit
from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import read, require, sha
from superagent.analogy_review import SHARES, blend_counts, promoted
from superagent.autonomous_run import planner_observation
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec


def run(root):
    folder=root/'analogy-raw'
    activate_resource_directory(root/'discovery-library')
    core={arm:audit(folder/arm) for arm in ('original','revised')}
    agent=folder/'revised/run/agent'
    record=read(agent/'analogy_review.json')
    report=dict(status=record['status'],review_receipt=sha(agent/'analogy_review.json'))
    if 'locked_weight' in record:
        anchor=record['incumbent_members'][0]
        challenger=record['candidate']
        history=read(agent/'history.json')
        def get(name,fold):
            spec=read(agent/(fold+'_split.json'))
            payload=agent/(fold+'.joblib')
            require(sha(payload)==spec['payload_sha256'],'Changed audit split')
            data=joblib.load(payload)
            match=[e for e in history if e['candidate']==name and e['split_id']==fold and e['status']=='completed']
            path=agent/(f"trial_{match[0]['trial']}.joblib" if match else f'analogy_{fold}_{name}.joblib')
            artifact=joblib.load(path)
            actual=artifact['metrics']['resolved_configuration']
            require(actual['id']==name,'Changed model-to-measurement link')
            return planner_observation(artifact,artifact['metrics'],dict(sha256=sha(payload),labels=data['y_search']),MetricSpec.from_payload(data))
        base,near=[get(n,'primary') for n in (anchor,challenger)]
        observed=blend_counts(base,near,base['labels'],SHARES)
        inverted=dict(near,probabilities=np.asarray(near['probabilities'])[:,::-1])
        opposed=blend_counts(base,inverted,base['labels'],SHARES)
        identical=blend_counts(base,base,base['labels'],SHARES)
        require(observed==record['primary']['scores'] and SHARES[int(np.argmax(observed))]==record['locked_weight'],'Nomination did not replay')
        confirmation=[]
        for row in record['allocations']:
            a,b=[get(n,row['fold']) for n in (anchor,challenger)]
            before,after=blend_counts(a,b,a['labels'],[0.,record['locked_weight']])
            require([before,after]==[row['incumbent'],row['candidate']],'Confirmation did not replay')
            confirmation.append(dict(gain=after-before,rows=len(a['labels'])))
        passed=promoted(max(observed)-observed[0],confirmation)
        require(passed==(record['status']=='accepted'),'Final acceptance diverged from frozen rule')
        report.update(actual_share=record['locked_weight'],actual_accept=passed,primary_counts=observed,
                      actual_confirmations=confirmation,unchanged_feedback_share=SHARES[int(np.argmax(identical))],
                      opposite_feedback_share=SHARES[int(np.argmax(opposed))],opposite_feedback_counts=opposed,
                      source='two counterfactual evidence replays on primary training-held rows',new_fits=0)
    output=dict(core=core,adviser=report,model_api_requests=0,course_validation_read=False)
    lock(folder/'autonomy_audit.json',output)
    print({k:v['matched_opaque_replays'] for k,v in core.items()},report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=lambda x:Path(x).resolve(),required=True)
    run(p.parse_args().root)
