"""Check the actual raw group decision, including source choice under renamed and exchanged feedback."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.audit_choice_aliases import audit
from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import read,require,sha
from superagent.analogy_review import incumbent_evidence,incumbent_parts,nominate,representation_source
from superagent.autonomous_run import planner_observation
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec


def run(root):
    directory=root/'analogy-portfolio-raw'
    activate_resource_directory(root/'discovery-library')
    core={name:audit(directory/name) for name in ('portfolio_control','portfolio_revised')}
    agent=directory/'portfolio_revised/run/agent'
    review=read(agent/'analogy_review.json')
    summary=dict(status=review['status'],budget=read(agent/'run_outcome.json')['attempted_classifier_fits'],
                 source_required=review.get('source_member'),reported_choices=review.get('source_candidates',[]))
    if 'source_member' in review:
        final=read(agent/('analogy_incumbent.json' if review['status']=='accepted' else 'final_selection.json'))
        model=joblib.load(agent/final['model_file'])['model']
        members=review['incumbent_members']
        parts,weights=incumbent_parts(members,model)
        metric=MetricSpec.from_payload(dict(metric_spec=read(agent/'task_contract.json')['metric']))
        history=read(agent/'history.json')
        payload=agent/'primary.joblib'
        labels=joblib.load(payload)['y_search']
        def obs(path):
            b=joblib.load(path)
            return planner_observation(b,b['metrics'],dict(sha256=sha(payload),labels=labels),metric)
        records={}
        caps={}
        for n in members:
            item=next(e for e in history if e['split_id']=='primary' and e['candidate']==n and e['status']=='completed')
            records[n]=obs(agent/f"trial_{item['trial']}.joblib")
            caps[n]=dict(params=item['resolved_configuration']['params'])
        original,options=representation_source(members,parts,records,caps)
        require(summary['source_required']==original[0] and options==review['source_candidates'],'Could not replay source election')
        aliases={member:f'opaque{len(members)-i}' for i,member in enumerate(members)}
        inverse={v:k for k,v in aliases.items()}
        renamed,_=representation_source([aliases[x] for x in reversed(members)],list(reversed(parts)),
                   {aliases[k]:v for k,v in records.items()},{aliases[k]:v for k,v in caps.items()})
        require(inverse[renamed[0]]==original[0],'Display name or ordering changed the source')
        summary['renamed_source_same']=True
        if len(options)>1:
            n,m=[p['member'] for p in options[:2]]
            changed={k:dict(v) for k,v in records.items()}
            changed[n]['score'],changed[m]['score']=changed[m]['score'],changed[n]['score']
            alternate,_=representation_source(members,parts,changed,caps)
            summary.update(exchanged_top_source_scores=[n,m],counterfactual_source=alternate[0])
            if records[n]['score']!=records[m]['score']:
                require(alternate[0]==m,'Reversing distinct reported source quality did not reverse the choice')
        routed=review.get('candidate')
        if routed and 'locked_weight' in review:
            near=obs(agent/f'analogy_primary_{routed}.joblib')
            anchor=incumbent_evidence([records[n] for n in members],weights,metric)
            actual=nominate(anchor,near,labels)
            neutral=nominate(anchor,anchor,labels)
            opposite=nominate(anchor,dict(near,probabilities=np.asarray(near['probabilities'])[:,::-1]),labels)
            require(actual==review['transparent_nomination'],'Primary evidence did not yield the published share')
            summary.update(actual_share=actual['weight'],neutral_share=neutral['weight'],opposite_share=opposite['weight'],
                           deployed_weights=review.get('weights'))
    report=dict(review_sha=sha(agent/'analogy_review.json'),core=core,source_audit=summary,new_model_requests=0,
           new_fits=0,course_validation_read=False)
    lock(directory/'audit.json',report)
    print({a:v['matched_opaque_replays'] for a,v in core.items()},summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=lambda x:Path(x).resolve(),required=True)
    run(p.parse_args().root)
