"""Verify A31/B source ownership, replay observed proposals, or run the ordinary anonymous CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from scripts import run_b1_ab_acceptance as rec, run_b1_05_pair_pilot as pilot
from scripts.run_b_a19_released_delivery import export_frozen
from superagent import ab_run
from superagent.candidate_coverage import project_routes


A = 'd910dc4dac3a2cafda654d29d5de8c5a86b0f8e8'
B = 'fc8506bc3e52c43a6332c270ce89f44a87e5a2c2'
FILES = {A:('ab_run.py','cli.py','candidate_coverage.py'), B:('ab_catalog.py','ab_methods.py','ab_scheduler.py','ab_sentence.py')}
INPUTS = {'train.zip':'7981a2315440bf1dca79d13ed619b8f316d0add98da6a406608c82d63e368639',
          'test.zip':'ad8ff99b46db6162632a5c876b621a95653854c19e7d11ca3eb800f4b32a8e31'}


def signed(commit,relative):
    data = subprocess.check_output(['git','show',commit+':'+relative],cwd=rec.ROOT)
    if (rec.ROOT/relative).read_bytes()!=data:
        raise ValueError(f'Checked-out bytes differ from the signed record {relative}')
    return data


def replay(out):
    folder = 'research/a29_choice_matrix'
    fixed = json.loads(signed(A,'research/a31_coverage_gate/a31_coverage_replay/results.json'))
    prior = {(r['task'],r['arm'],r['trial']):r for r in fixed}
    result = []
    ids = set()
    by_arm = {}
    for task in ('coupon','preference'):
        for arm in ('R0','E0','R1','E1'):
            base = f'{folder}/{task}_{arm}/'
            manifest = json.loads(signed(A,base+'run_manifest.json'))
            actions = json.loads(signed(A,base+'actions.json'))
            events = [json.loads(line) for line in signed(A,base+'events.jsonl').splitlines()]
            caps = manifest['capabilities']
            requests,proposals = ([e for e in events if e['kind']==kind] for kind in ('llm_request','ab_flash_proposal'))
            if len(requests)!=3 or len(proposals)!=3:
                raise ValueError('Expected three complete observed Flash proposals')
            denied = 0
            for index,(request,proposal) in enumerate(zip(requests,proposals,strict=True)):
                state = json.loads(request['messages'][1]['content'])
                step = proposal['trial']
                pick = proposal['proposal']['candidate']
                menu = tuple(c['id'] for c in state['available_candidates'])
                budget_id = request['budget_request_id']
                if (not budget_id or budget_id in ids or state['budget']['used_llm_requests']!=index
                        or state['budget']['fit_limit']!=11 or state['budget']['llm_request_limit']!=3
                        or pick not in menu or actions[step-1]['candidate_id']!=pick or actions[step-1]['action']!='explore'
                        or any(c!=caps[c['id']] for c in state['available_candidates'])):
                    raise ValueError('Request identity, menu, budget or executed action changed')
                ids.add(budget_id)
                projected,audit = project_routes(state,caps,menu)
                if tuple(name for name in menu if name in projected)!=projected:
                    raise ValueError('Projection expanded or reordered the original menu')
                row = dict(task=task,arm=arm,trial=step,original_candidate=pick,original_choice_allowed=pick in projected,
                           projection=audit,original_request_messages_sha256=hashlib.sha256(json.dumps(request['messages'],sort_keys=True).encode()).hexdigest())
                if arm.endswith('0') and row!=prior[task,arm,step]:
                    raise ValueError('A previously published fixed-allocation replay differs')
                denied += int(pick not in projected)
                result.append(row)
            by_arm[task+'_'+arm] = dict(proposals=3,excluded=denied)
    if len(prior)!=12 or len(result)!=24 or rec.trace(out):
        raise ValueError('Replay accounting unexpectedly changed')
    rec.write(out/'results.json',dict(by_arm=by_arm,actions=result))
    return dict(status='completed',actual_archived_proposals=24,reproduced_existing_fixed=12,new_dynamic_replays=12,
                counterfactual_scores=0,new_fits=0,new_model_api_requests=0,new_course_rmb=0)


def anonymous(out,fixture):
    old = fixture/'run'
    facts = rec.read(fixture/'results.json')
    if any(ab_run.sha(fixture/file)!=sha for file,sha in INPUTS.items()):
        raise ValueError('Original anonymous task or its unsigned query rows changed')
    frozen = rec.read(old/'final_selection.json')
    if ab_run.sha(old/frozen['model_file'])!=facts['delivery']['model_sha256']:
        raise ValueError('Old delivery identity changed')
    case = out/'case'
    case.mkdir()
    dest = case/'run'
    pilot.child(['-m','superagent','run',fixture/'train.zip','--scheduler','ab-v1','--portfolio','ab-v1',
                 '--planner','local','--fit-limit','11','--seed','707','--time-limit','1200','--trial-timeout','300',
                 '--run-dir',dest],case/'train.log',1260)
    old_manifest,new_manifest = (rec.read(p/'run_manifest.json') for p in (old,dest))
    ledger = rec.read(dest/'classifier_fits.json')
    history = rec.read(dest/'history.json')
    actions = rec.read(dest/'actions.json')
    new_final = rec.read(dest/'run_outcome.json')
    if (new_manifest['candidate_coverage']!='off' or not new_final['source_unchanged'] or new_final['new_api_requests']
            or not new_final['full_refit_completed'] or new_final['formal_test_accessed']
            or rec.digest(new_manifest['source_files'])!=rec.digest(ab_run.sources())):
        raise ValueError('The default local policy or refit did not execute as expected')
    actual = rec.trace(out)
    workers = [w['candidate'] for w in ledger['workers'] for _ in range(w['attempted_classifier_fits'])]
    if ([r['candidate'] for r in actual]!=workers or len(actual)!=ledger['completed_classifier_fits'] or len(actual)>11
            or ledger['unknown_failed_attempts']):
        raise ValueError('Physical fit count differs from actual budget')
    digest = export_frozen(case,fixture/'test.zip',31)
    compare = dict(full_ordered_menu=list(old_manifest['capabilities'].items())==list(new_manifest['capabilities'].items()),
                   contract=old_manifest['task_contract']==new_manifest['task_contract'],actions=actions==rec.read(old/'actions.json'),
                   prediction_evidence=rec.read(old/'prediction_evidence.json')==rec.read(dest/'prediction_evidence.json'),
                   internal_rows=all(rec.read(old/(s+'_split.json'))==rec.read(dest/(s+'_split.json')) for s in ('primary','confirmation_1','confirmation_2')),
                   expected_prediction_rows=digest==facts['delivery']['prediction_csv_sha256']==ab_run.sha(fixture/'predict/rows.csv'),
                   selected=rec.read(dest/'final_selection.json')['family']==frozen['family'])
    rec.write(out/'results.json',dict(equality=compare,fits=len(actual),original_fits=rec.read(fixture/'completion.json')['actual_fits'],
              actions=actions,training_rounds=len(history),model_sha256=rec.read(dest/'final_selection.json')['model_sha256'],
              prediction_sha256=digest,source=rec.digest(ab_run.sources())))
    return dict(status='completed',all_comparison_fields_equal=all(compare.values()),new_fits=len(actual),new_cli_exports=2,
                query_rows=31,new_grades=0,new_model_api_requests=0,new_course_rmb=0)


def study(args):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    files = {f:rec.sha(signed(commit,'superagent/'+f)) for commit,items in FILES.items() for f in items}
    rec.write(out/'protocol.json',dict(batch='BA31-01' if args.mode=='replay' else 'BA31-02',a=A,b=B,source_files=files,
              runner_sha256=ab_run.sha(Path(__file__)),plan_sha256=ab_run.sha(rec.ROOT/'docs/batches/BA31_01_02_integration_plan.md'),
              maximum_fits=0 if args.mode=='replay' else 11,api_limit=0))
    try:
        result = replay(out) if args.mode=='replay' else anonymous(out,args.fixture.resolve())
        rec.write(out/'completion.json',result)
    except Exception as exc:
        rec.write(out/'completion.json',dict(status='failed',reason=str(exc),new_fits=len(rec.trace(out))))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('replay','anonymous'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',type=Path)
    study(parser.parse_args())
