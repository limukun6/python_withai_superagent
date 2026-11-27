"""Verify that the fresh raw path makes the same choice as the frozen training-only pilot."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.audit_choice_aliases import audit
from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import read,require,sha
from superagent.analogy_review import nominate,promoted
from superagent.autonomous_run import planner_observation
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec


def verify(root):
    raw=root/'analogy-close-raw/close'
    agent=raw/'run/agent'
    activate_resource_directory(root/'discovery-library')
    core=audit(raw)
    review=read(agent/'analogy_review.json')
    prior=read(root/'analogy-close-confirm/results.json')
    require(review['transparent_nomination']==prior['decision'],'Fresh run did not reproduce the registered nomination')
    for a,b in zip(review['allocations'],prior['confirmations'],strict=True):
        require((a['row_key'],a['incumbent'],a['candidate'])==(b['row_key'],b['anchor'],b['chosen']),'Fresh confirmation differs')
    step=next(row for row in read(agent/'history.json') if row['candidate']==review['incumbent_members'][0] and row['split_id']=='primary')
    payload=agent/'primary.joblib'
    data=joblib.load(payload)
    def observe(path):
        bundle=joblib.load(path)
        return planner_observation(bundle,bundle['metrics'],dict(sha256=sha(payload),labels=data['y_search']),MetricSpec.from_payload(data))
    anchor=observe(agent/f"trial_{step['trial']}.joblib")
    neighbour=observe(agent/f"analogy_primary_{review['candidate']}.joblib")
    swapped=dict(neighbour,probabilities=np.asarray(neighbour['probabilities'])[:,::-1])
    real=nominate(anchor,neighbour,data['y_search'])
    negative=nominate(anchor,swapped,data['y_search'])
    identical=nominate(anchor,anchor,data['y_search'])
    accepted=promoted(real['primary_gain'],review['allocations'],primary_rows=len(data['y_search']))
    require(real==prior['decision'] and accepted==(review['status']=='accepted') and negative['weight']==identical['weight']==0.,
            'Fresh or perturbed decision did not replay')
    result=dict(neutral_or_reversed_adviser_share=[identical['weight'],negative['weight']],real_share=real['weight'],
        fresh_matches_prefix_pilot=True,core_alias_replays=core['matched_opaque_replays'],core=core,
        adviser_status=review['status'],fresh_classifier_fits=read(raw/'sealed.json')['fits'],new_replay_fits=0,api=0)
    lock(root/'analogy-close-raw/audit.json',result)
    print(result['core_alias_replays'],result['real_share'],result['neutral_or_reversed_adviser_share'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=lambda x:Path(x).resolve(),required=True)
    verify(p.parse_args().root)
