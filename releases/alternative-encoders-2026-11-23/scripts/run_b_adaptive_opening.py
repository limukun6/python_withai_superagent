"""Evaluate a complete anonymous run using the joint candidate catalog at a previously unused split."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, command, read
from superagent import ab_run
from superagent.ab_methods import JOINT_PREFERENCE, TEXT_FAMILIARITY, TRANSFER_METHODS


def run(workspace, inputs, previous, fixed, out, policy):
    out.mkdir(parents=True, exist_ok=True)
    env = offline_environment(workspace)
    baseline = read(previous / 'protocol.json')
    packets = baseline['packets']
    check(ab_run.sha(inputs / 'anonymous.zip') == packets['anonymous.zip'], 'Training input differs')
    frozen_save(out / 'protocol.json', dict(seed=40531, fit_limit=10, model_api_cap=0, scheduler=policy,
        method_menu=JOINT_PREFERENCE, training=packets['anonymous.zip'], runner=ab_run.sha(Path(__file__)),
        plan=ab_run.sha(ROOT / 'docs/batches/B_adaptive_opening_1123_zh.md'), production=ab_run.sources(),
        fixed_reference=ab_run.sha(fixed / 'results.json'), previous_native=ab_run.sha(previous / 'results.json')))
    d, physical = out / 'autonomous', out / 'physical.jsonl'
    if not d.exists() and not physical.exists():
        args = ['run', str(inputs/'anonymous.zip'), '--portfolio','ab-v1','--planner','local','--scheduler',policy,
                '--preference-catalog',JOINT_PREFERENCE,'--seed','40531','--fit-limit','10',
                '--trial-timeout','300','--time-limit','1200','--run-dir',str(d)]
        code, _ = command(args, out/'run.log', dict(env,AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1'),1260)
        check(code == 0, 'Native run ended early. Inspect the bounded fit record.')
    m = read(d/'run_manifest.json')
    caps, prior_caps = m['capabilities'], read(previous/'autonomous/run_manifest.json')['capabilities']
    e5 = TRANSFER_METHODS['deep-e5'][1]
    check(TEXT_FAMILIARITY in caps and e5 in caps and {n:c for n,c in caps.items() if n!=e5} == prior_caps
          and m['seed']==40531 and m['complement_active'] and m.get('adaptive_opening_active',False)==(policy=='ab-adaptive-complement'),
          'Unexpected production menu or scheduler')
    for split in ('primary','confirmation_1','confirmation_2'):
        s=read(d/(split+'_split.json'))
        check(s['search_rows_sharing_text']==0 and s['row_ids_sha256']!=read(previous/'autonomous'/(split+'_split.json'))['row_ids_sha256'],
              'Reused split or overlapping title')
    decision = read(d/'confirmation.json')['b_scheduler_review']
    outcome, selection = read(d/'run_outcome.json'), read(d/'final_selection.json')
    members = decision['members']
    fits = read(d/'classifier_fits.json')
    history = read(d/'history.json')
    by_split = {s:[r['candidate'] for r in history if r['split_id']==s] for s in ('primary','confirmation_1')}
    actions = [a['action'] for a in read(d/'actions.json')]
    calls = [json.loads(line) for line in (d/'llm_usage.jsonl').read_text().splitlines()]
    physical_count = len(physical.read_text().splitlines())
    check(actions == ['explore']*len(by_split['primary'])+['confirm']*len(by_split['confirmation_1'])+['stop']
          and set(by_split['primary']) == set(decision['opening_correct'])
          and physical_count == fits['attempted_classifier_fits'] == fits['completed_classifier_fits'] == len(history)+len(members) <= 10,
          'Actions and supervised fit accounting differ')
    check(len(calls)==1 and calls[0]['event']=='ledger_initialized' and calls[0]['cost_rmb']==0
          and outcome['new_api_requests']==0 and not outcome['formal_test_accessed'] and outcome['source_unchanged']
          and outcome['requested_planner_completed'] and outcome['full_refit_completed'] and selection['training_rows']==6608
          and (selection['ensemble_components'] or [selection['family']])==members
          and selection['model_sha256']==ab_run.sha(d/'selected_model.joblib'), 'Incomplete selection or model')
    frozen_save(out/'model_frozen.json',dict(decision=decision, tried=by_split, actions=actions, model=selection['model_sha256'],
        physical_fits=physical_count, physical_sha256=ab_run.sha(physical), model_api_calls=0))
    new = verified_export(d, inputs, env, out)
    old = read(fixed/'exports_frozen.json')['new_vote']
    previous_deep = read(previous/'exports_frozen.json')['methods']['deep']
    frozen_save(out/'exports_frozen.json',dict(model=ab_run.sha(out/'model_frozen.json'), new=new, fixed_vote=old, deep=previous_deep))
    check(all(ab_run.sha(inputs/n)==packets[n] for n in ('features.zip','scoring_only.json','groups_only.json')), 'Reserve changed')
    y = np.asarray(read(inputs/'scoring_only.json')['labels'])
    g = np.asarray(read(inputs/'groups_only.json'))
    new_hits = classes(out/'autonomous_export_0/rows.csv',new,len(y))==y
    fixed_hits = classes(fixed/'e5_familiarity_export_0/rows.csv',old,len(y))==y
    check(len(y)==len(g)==1693 and int(fixed_hits.sum())==1237, 'Incorrect fixed reference')
    result = dict(seed=m['seed'],policy=policy, members=members, new_correct=int(new_hits.sum()), rows=len(y),
        fixed_research_reference_correct=int(fixed_hits.sum()), gained=int((new_hits & ~fixed_hits).sum()),
        lost=int((fixed_hits & ~new_hits).sum()), delta_against_fixed=paired(new_hits.astype(int)-fixed_hits,g),
        actual_new_fits=physical_count, new_api_calls=0, new_rmb=0, designated_validation_eligible=int(new_hits.sum())>=1235,
        course_designated_validation_used=False, formal_test_used=False, export=ab_run.sha(out/'exports_frozen.json'))
    frozen_save(out/'results.json',result)
    frozen_save(out/'completion.json',dict(status='complete',result=ab_run.sha(out/'results.json'),
        physical_fits=physical_count,model_api_calls=0))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('workspace','inputs','previous','fixed','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--scheduler',choices=('ab-complement','ab-adaptive-complement'),default='ab-adaptive-complement')
    a=p.parse_args()
    run(*(getattr(a,k).resolve() for k in ('workspace','inputs','previous','fixed','output')),a.scheduler)
