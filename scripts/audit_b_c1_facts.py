"""Read fitted classifiers and compare the actual raw/checked input builders on registered states."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import joblib

from scripts import run_b1_ab_acceptance as rec
from scripts.compare_a11_b_decisions import compare_path
from scripts.measure_b1_headroom import archive_for
from scripts.replay_b1_observed_actions import POLICIES, SEEDS, TASKS, trusted_input
from superagent import ab_flash, ab_run
from superagent.ab_scheduler import next_action
from superagent.estimator_facts import observed_classifier, verify_configuration_claim


def actual_estimators(run):
    caps=rec.read(run/'run_manifest.json')['capabilities']
    checks=[]
    for measure in rec.read(run/'history.json'):
        trial=measure['trial']
        artifact=joblib.load(run/f'trial_{trial}.joblib')
        observed=observed_classifier(artifact['model'])
        if observed!=measure['estimator']:
            raise ValueError('Historical worker and fitted classifier disagree')
        name=measure['candidate']
        claim={'configuration_claim':{'classifier':caps[name]['estimator']}}
        checks.append(dict(candidate=name,trial=trial,split=measure['split_id'],
                           **verify_configuration_claim(claim,observed)))
    return checks


def proposals(arm, entry, source):
    caps,trusted=trusted_input(arm,entry,source)
    root=arm/'run'
    actions=rec.read(root/'actions.json')
    contexts=[]
    for i,action in enumerate(actions,1):
        state=rec.read(root/f'state_{i}.json')
        if next_action(state,capabilities=caps,trusted_partitions=trusted)!=action:
            raise ValueError('Native true local action replay differs')
        projected=deepcopy(state)
        projected['budget'].update(llm_request_limit=3,used_llm_requests=len(contexts))
        def inspect(ids):
            raw=ab_flash.context_for(projected,caps,ids,False)
            checked=ab_flash.context_for(projected,caps,ids,True)
            if (set(checked)!=set(raw)|{'checks','memory'} or {k:checked[k] for k in raw}!=raw or
                    [c['id'] for c in raw['available_candidates']]!=[c['id'] for c in state['available_candidates'] if c['id'] in ids] or
                    any(raw['candidate_facts'][n]['classifier']!=caps[n]['estimator'] for n in ids)):
                raise ValueError('Observable common evidence or eligible methods differ')
            history=projected['history']
            if [(v['trial'],v['split_id']) for v in checked['checks']]!=[(r['trial'],r['split_id']) for r in history]:
                raise ValueError('Checked evidence refers to different trials or partitions')
            def size(record):
                return len(json.dumps(record,ensure_ascii=False).encode('utf-8'))
            contexts.append(dict(step=i,eligible=len(ids),history=len(history),raw_bytes=size(raw),checked_bytes=size(checked),
                            eligible_sha256=rec.digest(list(ids)),identity_confirmed=sum(
                                r['metric_identity']['status']=='confirmed' for r in checked['checks']),
                            configuration_declared=sum(r['configuration_fact_check']['status']!='not_declared' for r in checked['checks'])))
            return action['candidate_id']
        replay=next_action(projected,capabilities=caps,trusted_partitions=trusted,on_explore=inspect)
        if any(replay[k]!=action[k] for k in ('action','candidate_id','split_id','estimated_classifier_fits')):
            raise ValueError('Holding the observed choice fixed changed an executable action')
    if len(contexts)!=min(3,sum(row['action']=='explore' for row in actions)-1):
        raise ValueError('Callback timing or request ceiling differs from registration')
    return dict(local_actions=len(actions),projected_calls=contexts,actual_classifier_checks=actual_estimators(root))


def run(args):
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    source=rec.digest(ab_run.sources())
    rec.write(out/'protocol.json',dict(batch='BC1-04',source=source,code=ab_run.sha(Path(__file__)),
              plan=ab_run.sha(rec.ROOT/'docs/batches/BC1_04_facts_plan.md'),legacy_paths=16,native_paths=3,
              fits_allowed=0,actual_model_request_limit=0,outer_scores_allowed=0))
    old,native={},{}
    try:
        for name in TASKS:
            folder=getattr(args,name)
            if (folder/'protocol.json').read_bytes()!=(archive_for(name)/'protocol.json').read_bytes():
                raise ValueError('Legacy training protocol differs: '+name)
            original=rec.read(folder/'protocol.json')
            old[name]={}
            for seed in SEEDS:
                for policy in POLICIES:
                    key=f'{seed}_{policy}'
                    replay=compare_path(folder,key,policy,original)
                    checks=actual_estimators(folder/key/'run')
                    old[name][key]=dict(replay=replay,facts=checks)
        for name,folder,arm in [('mrpc',args.mrpc,'mrpc'),('digits_original',args.digits,'existing'),('digits_extended',args.digits,'vision')]:
            protocol=rec.read(folder/'protocol.json')
            if protocol['source']!=source:
                raise ValueError('Native work was produced by a different production release')
            entry=protocol['mrpc'] if name=='mrpc' else protocol['tasks'][arm]
            native[name]=proposals(folder/arm,entry,source)
        rec.write(out/'results.json',dict(legacy=old,native=native))
        legacy=[r for by_task in old.values() for r in by_task.values()]
        a=[v for r in legacy for v in r['facts']]
        b=[v for r in native.values() for v in r['actual_classifier_checks']]
        rec.write(out/'completion.json',dict(status='measured',legacy_action_matches=sum(r['replay']['common_executable_prefix'] for r in legacy),
                  legacy_action_total=sum(r['replay']['earlier_path_actions'] for r in legacy),
                  legacy_classifier_status=dict(Counter(r['status'] for r in a)),native_classifier_status=dict(Counter(r['status'] for r in b)),
                  distinct_classifier_configs=len({r['candidate'] for r in [*a,*b]}),projected_context_pairs=sum(len(r['projected_calls']) for r in native.values()),
                  fits=0,http_requests=0,model_api_requests=0,outer_scores_read=0))
    except Exception as error:
        rec.write(out/'completion.json',dict(status='failed',reason=str(error),fits=0,model_api_requests=0,outer_scores_read=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*TASKS,'mrpc','digits','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    run(parser.parse_args())
