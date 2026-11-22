"""Reconstruct frozen A29 selection and check disagreement on its actual interior partitions."""
import argparse
import json
from pathlib import Path

from scripts import run_b1_ab_acceptance as record
from scripts.audit_b_a31_observed_alternatives import ARMS, BASE, old_file
from scripts.check_b_a31_integration import A, signed
from superagent import ab_run
from superagent.ab_comparison import incumbent_review


PARTITIONS=('primary','confirmation_1','confirmation_2')
RULES=('stable','stable_and_paired')


def differences(measured,a,b):
    return {s:measured[b,s]['search_score']-measured[a,s]['search_score'] for s in PARTITIONS
            if (a,s) in measured and (b,s) in measured}


def account_row(action,admitted,observed,row,worker,entry,number):
    cap=entry['capabilities'][row['candidate']]
    split=row['split_id']
    witness=observed['observation']
    admitted_candidate=cap if split=='primary' else {k:cap[k] for k in ('id','params')}
    if (row['trial']!=number or row['status']!='completed' or split not in PARTITIONS
            or row['candidate']!=action['candidate_id'] or split!=action['split_id']
            or action['action']!=('explore' if split=='primary' else 'confirm')
            or entry['splits'][split]!=admitted['split'] or admitted['candidate']!=admitted_candidate
            or admitted['expected_classifier_fits']!=cap['classifier_fits'] or worker['maximum_classifier_fits']!=cap['classifier_fits']
            or row['params']!=cap['params'] or row['resolved_configuration']['id']!=cap['id']
            or row['resolved_configuration']['params']!=cap['params'] or row['metric_spec']!=entry['task_spec']['metric']
            or row['search_metrics']['score']!=row['search_score'] or row['metric_identity']['status']!='confirmed'
            or any(row.get(k)!=v for k,v in witness.items())
            or worker['artifact']!=f'trial_{number}.joblib' or worker['candidate']!=cap['id']
            or worker['status']!='completed' or worker['completed_classifier_fits']!=row['classifier_fits']
            or worker['attempted_classifier_fits']!=worker['completed_classifier_fits']):
        raise ValueError('Registered split, action, capability, worker count or measured score differs')


def inspect(entry,arm,expected_source):
    name=entry['task']+'_'+arm
    path=name+'/'
    manifest,history,actions,ledger,final=[old_file(path+n) for n in
        ('run_manifest.json','history.json','actions.json','classifier_fits.json','final_selection.json')]
    events=[json.loads(l) for l in signed(A,BASE+path+'events.jsonl').splitlines()]
    starts=[e for e in events if e['kind']=='scheduler_action_admitted']
    ends={e['state_sha256']:e for e in events if e['kind']=='scheduler_action_observed'}
    recorded=next(e for e in events if e['kind']=='selection_frozen')
    refit=ledger['workers'][-1]
    if (actions!=[s['action'] for s in starts] or len(history)+1!=len(actions) or len(ends)!=len(starts)
            or actions[-1]['action']!='stop' or ledger['unknown_failed_attempts'] or not final['refit']
            or len(ledger['workers'])!=len(history)+1 or not 1<ledger['attempted_classifier_fits']==ledger['completed_classifier_fits']<=11
            or refit['candidate']!=final['family'] or refit['artifact']!='refit_component_0.joblib' or refit['status']!='completed'
            or refit['attempted_classifier_fits']!=refit['completed_classifier_fits']
            or manifest['source_files']!=expected_source or manifest['task_archive_sha256']!=entry['training_archive_sha256']
            or manifest['capabilities']!=entry['capabilities'] or manifest['task_contract']!=entry['task_spec']):
        raise ValueError('Archived task, actions, selection or accounting differs')
    witnesses,measured,early,episodes={}, {},{},{}
    incumbents=[]
    spent=0
    for number,(action,start) in enumerate(zip(actions,starts,strict=True),1):
        end=ends[start['state_sha256']]
        before=start['budget_before']
        if (end['action']!=action or before['attempted_classifier_fits']!=spent or before['completed_classifier_fits']!=spent
                or before['remaining_fit_allowance']!=before['fit_limit']-spent or before['unknown_failed_attempts']!=0):
            raise ValueError('Original before-action budget or state identity differs')
        review=incumbent_review(measured,witnesses,PARTITIONS[1:])
        leader=review['incumbent']
        terminal=action['action']=='stop'
        if leader is not None:
            incumbents.append(leader)
            primary={n for n,s in measured if s=='primary'}
            stable=len(primary)>=3 and len(incumbents)>=3 and len(set(incumbents[-3:]))==1
            paired=False
            for opponent in sorted(primary-{leader}):
                margins=differences(measured,leader,opponent)
                paired=paired or len(margins)>=2
                pair=tuple(sorted((leader,opponent)))
                if (pair not in episodes and 'confirmation_1' in margins and 'confirmation_2' not in margins
                        and margins['primary']*margins['confirmation_1']<0):
                    todo=[n for n in (leader,opponent) if (n,'confirmation_2') not in measured]
                    needed=sum(entry['capabilities'][n]['classifier_fits'] for n in todo)
                    episodes[pair]=dict(at_observations=number-1,leader=leader,challenger=opponent,at_state_sha256=start['state_sha256'],
                        initial_margins=margins,second_confirmation_remaining=todo,remaining_search_fits=before['remaining_fit_allowance']-before['reserved_final_refit'],
                        both_still_affordable=needed+before['reserved_final_refit']<=before['remaining_fit_allowance'],
                        observed_next_action=[action['action'],action['candidate_id'],action['split_id']])
            for mode,eligible in zip(RULES,(stable,stable and paired),strict=True):
                if mode not in early and (eligible or terminal):
                    saved=ledger['completed_classifier_fits']-spent-entry['capabilities'][leader]['refit_classifier_fits']
                    if not 0<=saved<ledger['completed_classifier_fits']:
                        raise ValueError('Counterfactual allowance is inconsistent')
                    early[mode]=dict(at_observations=number-1,incumbent=leader,earlier=not terminal,same_winner=leader==final['family'],
                                    estimated_fits_saved=saved,state_sha256=start['state_sha256'])
        if terminal:
            original_review=recorded['confirmation']['b_scheduler_review']
            if dict(review,primary_metric=entry['task_spec']['metric']['name'])!=original_review or leader!=recorded['single'] or leader!=final['family']:
                raise ValueError('Reconstructed final promotion or selected model differs')
            break
        row=history[number-1]
        worker=ledger['workers'][number-1]
        account_row(action,start,end,row,worker,entry,number)
        identity=(row['candidate'],row['split_id'])
        if identity in measured:
            raise ValueError('Repeated candidate and split would overwrite an original trial')
        measured[identity]=row
        # incumbent_review uses membership of witnessed trial numbers. Here the witness
        # is A's published observed event and split identity, not a reconstructed vector.
        witnesses[row['trial']]=start['split']
        spent+=worker['completed_classifier_fits']
        if end['budget_after']['completed_classifier_fits']!=spent:
            raise ValueError('Original after-action accounting differs')
    if spent+refit['completed_classifier_fits']!=ledger['completed_classifier_fits']:
        raise ValueError('One original physical fit was not accounted for')
    for episode in episodes.values():
        leader,opponent=episode['leader'],episode['challenger']
        margins=differences(measured,leader,opponent)
        episode.update(final_margins=margins,both_completed_second='confirmation_2' in margins,
            final_is_challenger=final['family']==opponent,final_is_original_leader=final['family']==leader)
    return dict(task=entry['task'],arm=arm,original_selected=final['family'],actual_original_fits=ledger['completed_classifier_fits'],
        observed_states=len(actions),verified_trial_events=len(history),final_selection_reproduced=True,early_stops=early,
        conflicts=list(episodes.values()))


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    source=old_file('protocol.json')
    if source['source_files']['ab_comparison.py']!=ab_run.sha(record.ROOT/'superagent/ab_comparison.py'):
        raise ValueError('Registered A29 selection algorithm differs')
    record.write(out/'protocol.json',dict(batch='BA31-04',a_commit=A,a29_protocol_sha256=record.sha(signed(A,BASE+'protocol.json')),
        plan_sha256=ab_run.sha(record.ROOT/'docs/batches/BA31_04_confirmation_conflicts_plan.md'),
        driver_sha256=ab_run.sha(Path(__file__)),source_trial_scope='published training events and scalar scores',new_fit_limit=0,api_limit=0))
    try:
        results={}
        for entry in source['entries']:
            for arm in ARMS:
                results[entry['task']+'_'+arm]=inspect(entry,arm,source['source_files'])
        episodes=[c for r in results.values() for c in r['conflicts']]
        record.write(out/'results.json',results)
        record.write(out/'completion.json',dict(status='completed',paths=len(results),real_states=sum(r['observed_states'] for r in results.values()),
            original_search_and_confirm_fits=sum(r['verified_trial_events'] for r in results.values()),
            raw_path_pair_conflicts=len(episodes),fully_rechecked=sum(c['both_completed_second'] for c in episodes),
            new_fits=0,new_predictions=0,new_model_api_requests=0,new_course_rmb=0,designated_scores_or_test_read=False))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',error=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    run(parser.parse_args().output.resolve())
