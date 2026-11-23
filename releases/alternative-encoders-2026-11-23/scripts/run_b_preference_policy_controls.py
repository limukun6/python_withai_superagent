"""Match native fixed and adaptive schedules with separately verified labels, row order and fit budgets."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, command, read
from superagent import ab_run
from superagent.ab_methods import JOINT_PREFERENCE

PLAN = Path(__file__).resolve().parents[1] / 'docs/batches/B_preference_policy_controls_1123_zh.md'
POLICIES = {'fixed': ('ab-fixed',11), 'dynamic': ('ab-adaptive-complement',11), 'first': ('ab-v1',2)}


def execute(arm, base, inputs, env):
    base.mkdir(exist_ok=True)
    folder, physical = base/'autonomous', base/'physical.jsonl'
    scheduler, cap = POLICIES[arm]
    if not any(f.exists() for f in (folder,physical,base/'run.log')):
        args = ['run',str(inputs/'anonymous.zip'),'--planner','local','--portfolio','ab-v1',
                '--scheduler',scheduler,'--preference-catalog',JOINT_PREFERENCE,'--seed','18643','--fit-limit',str(cap),
                '--trial-timeout','300','--time-limit','1200','--run-dir',str(folder)]
        if arm=='dynamic':
            args.append('--cover-untried-encoder')
        code,_ = command(args,base/'run.log',dict(env,AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1'),1260)
        check(code==0,'Native '+arm+' stopped. Inspect its durable log and fit receipts.')
    return folder, physical


def audit(arm, folder, physical, training, n):
    out, ledger, sel, m = [read(folder/f) for f in ('run_outcome.json','classifier_fits.json','final_selection.json','run_manifest.json')]
    first_state, actions = read(folder/'state_1.json'), read(folder/'actions.json')
    h = read(folder/'history.json')
    usage = [json.loads(line) for line in (folder/'llm_usage.jsonl').read_text().splitlines()]
    count = len(physical.read_text().splitlines())
    check(m['scheduler']==POLICIES[arm][0] and m['fit_limit']==POLICIES[arm][1] and m['task_archive_sha256']==training
          and m['seed']==18643 and m['planner_mode']=='local' and m['preference_catalog']==JOINT_PREFERENCE and
          not first_state['history'] and not first_state['prediction_evidence'], 'Planner or starting state differs')
    check(out['requested_planner_completed'] and out['source_unchanged'] and out['full_refit_completed'] and
          not out['formal_test_accessed'] and out['new_api_requests']==0 and len(usage)==1 and
          usage[0]['cost_rmb']==0 and usage[0]['event']=='ledger_initialized','Incomplete run or unexpected model API')
    check(count==ledger['completed_classifier_fits']==ledger['attempted_classifier_fits']<=m['fit_limit']
          and actions[-1]['action']=='stop' and sel['training_rows']==n
          and sel['model_sha256']==ab_run.sha(folder/'selected_model.joblib'),'Incomplete fits or model')
    if arm=='fixed':
        check(sum(a['action']=='explore' for a in actions)==4 and sum(a['action']=='confirm' for a in actions)==6
              and count==11,'Unexpected fixed allocation')
    if arm=='first':
        check(len(h)==1 and count==2 and h[0]['candidate']==sel['family'],'Unexpected first-only allocation')
    return dict(fits=count,selected=sel.get('ensemble_components') or [sel['family']],model=sel['model_sha256'],
                physical=ab_run.sha(physical),fit_limit=m['fit_limit'],actions=[{k:a.get(k) for k in
                    ('action','candidate_id','split_id','evidence_trials')} for a in actions],
                action_record=ab_run.sha(folder/'actions.json'),model_api_calls=0)


def partitions(folders):
    reference = folders['dynamic']
    rm = read(reference/'run_manifest.json')
    receipts = {}
    for name,folder in folders.items():
        m = read(folder/'run_manifest.json')
        for k in ('capabilities','source_files','task_contract','task_archive_sha256','seed','split_seed_offsets',
                  'dependencies','time_limit_seconds','trial_timeout_seconds','preference_catalog','llm_request_limit'):
            check(m[k]==rm[k],name+' differs in '+k)
        receipts[name] = {}
        for split in ('primary','confirmation_1','confirmation_2'):
            json_name,job_name=split+'_split.json',split+'.joblib'
            ra,sa=read(reference/json_name),read(folder/json_name)
            check({k:v for k,v in ra.items() if k!='evaluation_payload_sha256'}==
                  {k:v for k,v in sa.items() if k!='evaluation_payload_sha256'}, name+' split metadata differs')
            a,b=joblib.load(reference/job_name),joblib.load(folder/job_name)
            check(set(a)-{'preference_vote_probabilities'}==set(b)-{'preference_vote_probabilities'},'Payload fields differ')
            for k in ('X_train','y_train','X_search','y_search'):
                np.testing.assert_array_equal(a[k],b[k])
            for k in ('kind','metric_spec','pair_semantics'):
                check(a[k]==b[k],k)
            receipts[name][split] = {k:sa[k] for k in ('row_ids_sha256','label_sha256','evaluation_payload_sha256')}
    return receipts


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    env=offline_environment(a.workspace)
    info=read(a.inputs/'protocol.json')
    movie = a.task=='quotes'
    n=info['train_rows'] if movie else info['rows']
    packet=ab_run.sha(a.inputs/'anonymous.zip')
    check(packet==(info['train_packet'] if movie else info['training_sha']) and
          ab_run.sha(a.inputs/'features.zip')==(info['features'] if movie else info['feature_sha']),'Wrong input packet')
    if movie:
        check(n==1744 and a.reference is not None,'Quote reference required')
    else:
        check(n==6608 and a.reference is None,'Headline dynamic arm must have the same 11-fit cap as fixed')
    order=['fixed','first'] if movie else ['fixed','dynamic']
    frozen_save(a.out/'protocol.json',dict(task=a.task,plan=ab_run.sha(PLAN),code=ab_run.sha(Path(__file__)),
            sources=ab_run.sources(),training=packet,features=ab_run.sha(a.inputs/'features.zip'),seed=18643,
            registered_order=order,reused_dynamic=movie,reference_completion=ab_run.sha(a.reference/'completion.json') if movie else None,
            primary_fit_limit=11,new_fit_limit=sum(POLICIES[k][1] for k in order),model_api_cap=0))
    folders,records={},{}
    if movie:
        folders['dynamic']=a.reference/'autonomous'
        records['dynamic']=audit('dynamic',folders['dynamic'],a.reference/'physical.jsonl',packet,n)
    for arm in order:
        folder,physical=execute(arm,a.out/arm,a.inputs,env)
        folders[arm]=folder
        records[arm]=audit(arm,folder,physical,packet,n)
        frozen_save(a.out/arm/'decision.json',records[arm])
    evidence=partitions(folders)
    if movie:
        check(len({v['actions'][0]['candidate_id'] for v in records.values()})==1,'Initial candidates differ')
    frozen_save(a.out/'frozen_decisions.json',dict(arms=records,partitions=evidence))
    paths,seals={},{}
    for arm,folder in folders.items():
        home=a.reference if movie and arm=='dynamic' else a.out/arm
        seals[arm]=verified_export(folder,a.inputs,env,home)
        paths[arm]=home/'autonomous_export_0/rows.csv'
    frozen_save(a.out/'exports.json',seals)
    check(ab_run.sha(a.inputs/'scoring_only.json')==(info['scoring'] if movie else info['locked_scores']),'Scorer changed')
    truth=read(a.inputs/'scoring_only.json')
    y=np.asarray(truth['labels'])
    if movie:
        groups=np.asarray(truth['components'])
    else:
        check(ab_run.sha(a.inputs/'groups_only.json')==info['locked_groups'],'Headline groups changed')
        groups=np.asarray(read(a.inputs/'groups_only.json'))
    hits={k:classes(f,seals[k],len(y))==y for k,f in paths.items()}
    after=hits['dynamic']
    comparisons={}
    for k in ('fixed','first') if movie else ('fixed',):
        d=after.astype(int)-hits[k].astype(int)
        comparisons['dynamic_minus_'+k]=dict(gained=int((d>0).sum()),lost=int((d<0).sum()),delta=paired(d,groups))
    result=dict(task=a.task,rows=len(y),groups=len(set(groups)),arms={k:dict(correct=int(h.sum()),accuracy=float(h.mean()),
        fits=records[k]['fits'],selected=records[k]['selected']) for k,h in hits.items()},paired=comparisons,
        actual_new_fits=sum(records[k]['fits'] for k in order),reused_fits=records['dynamic']['fits'] if movie else 0,
        model_api_calls=0,new_rmb=0,designated_or_formal_course_used=False)
    frozen_save(a.out/'results.json',result)
    frozen_save(a.out/'completion.json',dict(results=ab_run.sha(a.out/'results.json'),protocol=ab_run.sha(a.out/'protocol.json'),
        decisions=ab_run.sha(a.out/'frozen_decisions.json'),exports=ab_run.sha(a.out/'exports.json'),new_fits=result['actual_new_fits'],model_api_calls=0))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('headlines','quotes'),required=True)
    for n in ('workspace','inputs','out','reference'):
        p.add_argument('--'+n,type=lambda v:Path(v).resolve(),required=n!='reference')
    run(p.parse_args())
