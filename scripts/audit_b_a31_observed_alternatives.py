"""Check real primary outcomes supporting the A31 coverage menus on other A29 paths."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

from scripts import run_b1_ab_acceptance as record
from scripts.check_b_a31_integration import A, signed
from superagent import ab_run


ARMS=('L0','L1','R0','E0','R1','E1')
EPS=1.e-12
BASE='research/a29_choice_matrix/'


def old_file(path):
    return json.loads(signed(A,BASE+path))


def observations(task,entry,source):
    by_candidate=defaultdict(list)
    lookup={}
    for arm in ARMS:
        prefix=task+'_'+arm+'/'
        manifest,history,actions,ledger=[old_file(prefix+n) for n in ('run_manifest.json','history.json','actions.json','classifier_fits.json')]
        events=[json.loads(line) for line in signed(A,BASE+prefix+'events.jsonl').splitlines()]
        admitted=[e for e in events if e['kind']=='scheduler_action_admitted']
        observed={e['state_sha256']:e for e in events if e['kind']=='scheduler_action_observed'}
        if (manifest['task_archive_sha256']!=entry['training_archive_sha256'] or manifest['task_contract']!=entry['task_spec']
                or manifest['source_files']!=source or manifest['seed']!=entry['splits']['primary']['seed']
                or manifest['capabilities']!=entry['capabilities'] or ledger['unknown_failed_attempts']
                or ledger['attempted_classifier_fits']!=ledger['completed_classifier_fits'] or ledger['attempted_classifier_fits']>11):
            raise ValueError('Cannot pool this path with the originally registered task')
        workers={w['artifact']:w for w in ledger['workers']}
        for e in admitted:
            act=e['action']
            if act['action']!='explore':
                continue
            other=observed[e['state_sha256']]
            if act!=other['action'] or e['split']!=entry['splits']['primary'] or act['split_id']!='primary':
                raise ValueError('Action or interior evaluation identity differs')
            trial=next(r['trial'] for r in history if r['candidate']==act['candidate_id'] and r['split_id']=='primary')
            row=history[trial-1]
            cap=entry['capabilities'][row['candidate']]
            worker=workers[f'trial_{trial}.joblib']
            witness=other['observation']
            if (actions[trial-1]!=act or e['candidate']!=cap or row['status']!='completed'
                    or row['params']!=cap['params'] or row['resolved_configuration']['params']!=cap['params']
                    or row['resolved_configuration']['id']!=cap['id'] or witness['search_metrics']!=row['search_metrics']
                    or witness['search_score']!=row['search_score'] or row['metric_spec']!=entry['task_spec']['metric']
                    or row['search_metrics']['score']!=row['search_score'] or row['metric_identity']['status']!='confirmed'
                    or worker['candidate']!=cap['id'] or worker['completed_classifier_fits']!=row['classifier_fits']):
                raise ValueError('A primary observation lacks matching worker, configuration or metric evidence')
            fact=dict(arm=arm,trial=trial,score=row['search_score'],candidate=row['candidate'])
            by_candidate[row['candidate']].append(fact)
            lookup[arm,trial]=fact
    return by_candidate,lookup


def audit(out):
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    previous=record.ROOT/'research/ba31_all_proposals/results.json'
    proposals=record.read(previous)['actions']
    original=old_file('protocol.json')
    record.write(out/'protocol.json',dict(batch='BA31-03',a_commit=A,original_protocol_sha256=record.sha(signed(A,BASE+'protocol.json')),
        original_projection_sha256=ab_run.sha(previous),plan_sha256=ab_run.sha(record.ROOT/'docs/batches/BA31_03_observed_alternative_support_plan.md'),
        driver_sha256=ab_run.sha(Path(__file__)),other_path_only=True,tolerance=EPS,fit_limit=0,model_api_limit=0))
    try:
        reports={}
        for entry in original['entries']:
            task=entry['task']
            pool,lookup=observations(task,entry,original['source_files'])
            records=[]
            for p in [r for r in proposals if r['task']==task]:
                pick=lookup[p['arm'],p['trial']]
                if pick['candidate']!=p['original_candidate']:
                    raise ValueError('Recorded proposed action differs from actual trained candidate')
                eligible=p['projection']['projected_eligible_ids']
                support={}
                for name in eligible:
                    available=[r for r in pool[name] if r['arm']!=p['arm']]
                    if available:
                        support[name]=dict(min=min(r['score'] for r in available),max=max(r['score'] for r in available),
                                           other_observations=available)
                better=[n for n,w in support.items() if w['min']>pick['score']+EPS]
                worse=[n for n,w in support.items() if w['max']<pick['score']-EPS]
                eq=[n for n,w in support.items() if abs(w['min']-pick['score'])<=EPS and abs(w['max']-pick['score'])<=EPS]
                records.append(dict(arm=p['arm'],trial=p['trial'],old_pick=pick,old_pick_allowed=p['original_choice_allowed'],
                    menu_size=len(eligible),distinct_other_path_matches=len(support),unmeasured=[n for n in eligible if n not in support],
                    higher=better,lower=worse,tied=eq,matched_other_arms=support))
            blocked=[r for r in records if not r['old_pick_allowed']]
            counts=Counter()
            for r in blocked:
                counts['blocked']+=1
                counts['at_least_one_measured']+=bool(r['distinct_other_path_matches'])
                counts['with_some_unmeasured']+=bool(r['unmeasured'])
                counts['with_strictly_higher']+=bool(r['higher'])
                counts['with_strictly_lower']+=bool(r['lower'])
                counts['all_matched_strictly_lower']+=bool(r['distinct_other_path_matches'] and len(r['lower'])==r['distinct_other_path_matches'])
            reports[task]=dict(primary_observations=len(lookup),distinct_measured_candidates=sum(bool(rows) for rows in pool.values()),
                matched_range_across_repeat_fit_max=max(max(r['score'] for r in rows)-min(r['score'] for r in rows) for rows in pool.values() if rows),
                blocked_summary=dict(counts),actions=records)
        if sum(r['blocked_summary']['blocked'] for r in reports.values())!=17 or sum(len(r['actions']) for r in reports.values())!=24:
            raise ValueError('Observed blocked action population changed')
        record.write(out/'results.json',reports)
        record.write(out/'completion.json',dict(status='completed',original_proposals=24,blocked=17,new_fits=0,
                    new_flash_requests=0,new_course_rmb=0,official_validation_test_or_H_read=False))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',reason=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    audit(parser.parse_args().output.resolve())
