"""Replay the admitted native decisions and export a compact method-choice replay without labels."""
import argparse
import json
import os
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.run_b_varied_preferences import ARMS, SLOTS, native_env, read, require, save
from superagent.ab_catalog import untried_candidates
from superagent.ab_method_search import rank
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.ab_run import freeze_partitions, registered_capabilities, sha, sources
from superagent.ab_scheduler import PRIMARY, next_action, selection_evidence
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.scheduler_bridge import fingerprint, validate_action
from superagent.task_profile import training_facts


def selector_inputs(state, capabilities, eligible, verified):
    numbered={r['trial'] for r in verified}
    profile=state['training_profile']
    return dict(profile={'statistics':{k:v for k,v in profile['statistics'].items() if k in ('train_rows','word_count')},
                         'majority_class_fraction':profile['majority_class_fraction']},
                eligible=list(eligible),verified=[{k:r[k] for k in ('trial','candidate','search_score')} for r in verified],
                predictions={str(e['trial']):e['predictions'] for e in state['prediction_evidence'] if e['trial'] in numbered})


def portable_choice(caps,profile,predictions,step):
    state=dict(training_profile=profile,prediction_evidence=[
        dict(trial=r['trial'],predictions=predictions[str(r['trial'])]) for r in step['verified']])
    result=rank(state,caps,step['eligible'],step['verified'])
    reverse=rank(state,dict(reversed(caps.items())),list(reversed(step['eligible'])),step['verified'])
    require(result==reverse,'Method choice depends on candidate presentation order')
    return result


def action_key(a):
    return [a['action'],a['candidate_id'],a['split_id']]


def changed_primary(state,first,second):
    # Exchange entire paired outcomes on the same primary rows. These are hypothetical states only.
    changed=deepcopy(state)
    trials={row['trial']:row for row in changed['history']}
    evidence={row['trial']:row for row in changed['prediction_evidence']}
    a,b=trials[first],trials[second]
    e,f=evidence[first],evidence[second]
    require(a['split_id']==b['split_id']==PRIMARY and e['row_ids']==f['row_ids'] and
            e['evaluation_payload_sha256']==f['evaluation_payload_sha256'],'Incompatible hypothetical pair')
    for k in ('predictions','prediction_sha256'):
        e[k],f[k]=f[k],e[k]
    for k in ('search_score','search_accuracy','search_metrics'):
        a[k],b[k]=b[k],a[k]
    for row in (a,b):
        row['score_gap']=row['train_score']-row['search_score']
        row['generalization_gap']=row['train_accuracy']-row['search_accuracy']
    return changed


def audit_arm(a,name,*,arm_spec=None,input_slot=None):
    folder=a.out/name
    agent=folder/'agent'
    label,catalog,limit=arm_spec or ARMS[name]
    seed=read(a.out/'protocol.json')['seed']
    packet=a.out/'inputs'/(input_slot or SLOTS[label])/'train.zip'
    task=load_development_task(packet)
    spec=TaskSpec.from_task(task).record()
    profile=training_facts(task)
    manifest=read(agent/'run_manifest.json')
    with patch.dict(os.environ,{**native_env(a,folder/'unused'),PREFERENCE_CATALOG_ENV:catalog}):
        caps=registered_capabilities(spec,len(task.y_train))
    require(manifest['source_files']==sources() and manifest['task_contract']==spec and
            manifest['capabilities']==caps and sha(packet)==manifest['task_archive_sha256'] and
            manifest['planner_mode']=='local' and manifest['local_method_search']=='evidence' and
            manifest['scheduler']=='ab-v1' and manifest['llm_request_limit']==0,'Native planner contract changed')
    with TemporaryDirectory() as scratch:
        temp=Path(scratch)
        trusted,allowed=freeze_partitions(task,temp,seed,TaskSpec.from_task(task).metric)
        for split,p in allowed.items():
            info=read(agent/(split+'_split.json'))
            overlap='search_rows_sharing_text' if task.kind=='pair_text' else 'search_duplicate_rows'
            require(info==read(temp/(split+'_split.json')) and sha(agent/(split+'.joblib'))==p['evaluation_payload_sha256']
                    and info[overlap]==0,'Registered worker partition differs')
    history=read(agent/'history.json')
    actions=read(agent/'actions.json')
    outcomes=read(agent/'prediction_evidence.json')
    events=[json.loads(line) for line in (agent/'events.jsonl').read_text().splitlines()]
    admissions=[e for e in events if e['kind']=='scheduler_action_admitted']
    routes=[e for e in events if e['kind']=='planner_routing']
    require(len(admissions)==len(actions)==len(routes) and len(history)+1==len(actions),'Incomplete event sequence')
    summaries=[]
    portable=dict(capabilities=caps,profile=None,predictions={},steps=[])
    for i,action in enumerate(actions,1):
        state=read(agent/f'state_{i}.json')
        require(state['task']==spec and state['training_profile']==profile and state['history']==history[:i-1] and
                state['prediction_evidence']==outcomes[:i-1] and state['available_candidates']==untried_candidates(caps,history[:i-1])
                and state['budget']['attempted_classifier_fits']==i-1 and state['budget']['fit_limit']==limit and
                state['budget']['used_llm_requests']==0,'Training-only history or budget differs')
        gate=validate_action(state,action,caps,allowed)
        receipt=admissions[i-1]
        require(all(receipt[k]==v for k,v in gate.items()),'Admission event did not bind state to action')
        captured=[]
        def choose(eligible,verified):
            inputs=selector_inputs(state,caps,eligible,verified)
            result=rank(state,caps,eligible,verified)
            captured.append((inputs,result))
            return result
        actual=next_action(state,trusted_partitions=trusted,capabilities=caps,on_local_method=choose)
        require(actual==action,'Native action replay differs')
        reordered=dict(state,available_candidates=list(reversed(state['available_candidates'])))
        shuffled=next_action(reordered,trusted_partitions=trusted,capabilities=dict(reversed(caps.items())),
                             on_local_method=lambda ids,rows:rank(reordered,caps,ids,rows))
        require(shuffled==action,'Native action changed after reversing the menu')
        summary=dict(step=i,action=action_key(actual),state_digest=receipt['state_sha256'],evidence=action['evidence_trials'],
                     remaining=state['budget']['remaining_fit_allowance'],source=routes[i-1]['source'],
                     trial_pairs_swapped=0,swaps_changing_action=0)
        if action['action']=='explore':
            (inputs,result),=captured
            require(result==read(agent/f'method_choice_{i}.json') and routes[i-1]['source']=='local_model_evidence',
                    'Ranking differs from original exploration')
            require(portable['profile'] in (None,inputs['profile']),'Training profile changed during search')
            portable['profile']=inputs['profile']
            for key,values in inputs['predictions'].items():
                require(portable['predictions'].get(key,values)==values,'Primary predictions changed')
                portable['predictions'][key]=values
            step=dict(step=i,eligible=inputs['eligible'],verified=inputs['verified'],chosen=result['chosen'],ranking_sha=fingerprint(result))
            require(portable_choice(caps,inputs['profile'],inputs['predictions'],step)==result,'Reduced model-choice replay differs')
            portable['steps'].append(step)
            if len(inputs['verified'])>=2:
                masked=deepcopy(inputs['verified'])
                mean=sum(v['search_score'] for v in masked)/len(masked)
                for row in masked:
                    row['search_score']=mean
                masked_result=rank(state,caps,inputs['eligible'],masked)
                summary['mean_score_mask_winner']=masked_result['chosen']
                summary['mask_changes_candidate']=masked_result['chosen']!=result['chosen']
        else:
            require(not captured and routes[i-1]['source']=='shared_local_rule','Unexpected manual or external route')
        primaries=[r['trial'] for r in state['history'] if r['split_id']==PRIMARY]
        for first,second in combinations(primaries,2):
            changed=changed_primary(state,first,second)
            alt=next_action(changed,trusted_partitions=trusted,capabilities=caps,
                            on_local_method=lambda ids,rows:rank(changed,caps,ids,rows))
            require('requires reconciliation' not in alt['reason'],'Hypothetical outcome failed metric validation')
            summary['trial_pairs_swapped']+=1
            if action_key(alt)!=action_key(actual):
                summary['swaps_changing_action']+=1
                summary.setdefault('example_change',dict(trials=[first,second],alternative=action_key(alt)))
        summaries.append(summary)
    review=selection_evidence(state,trusted_partitions=trusted)
    frozen=read(folder/'frozen.json')
    usage=[json.loads(line) for line in (agent/'llm_usage.jsonl').read_text().splitlines()]
    require(review==frozen['selection'] and review['incumbent']==frozen['selected']==frozen['physical'][-1]['candidate'] and
            len(usage)==1 and usage[0]['event']=='ledger_initialized' and
            usage[0]['input_tokens']==usage[0]['output_tokens']==usage[0]['cost_rmb']==0 and
            len(frozen['physical'])==len(history)+1,
            'Selection, charged final fit or API usage differs')
    return dict(actions=len(actions),explorations=len(portable['steps']),confirmations=sum(x['action']=='confirm' for x in actions),
                stable_reordered_actions=len(actions),fit_count=len(frozen['physical']),selected=frozen['selected'],
                candidate_count=len(caps),partitions=allowed,steps=summaries,
                swap_tests=sum(x['trial_pairs_swapped'] for x in summaries),
                swap_tests_changing_action=sum(x['swaps_changing_action'] for x in summaries)),portable


def verify_portable(records):
    for name,arm in records.items():
        for step in arm['steps']:
            result=portable_choice(arm['capabilities'],arm['profile'],arm['predictions'],step)
            require(result['chosen']==step['chosen'] and fingerprint(result)==step['ranking_sha'],
                    'Different candidate or ranking in portable replay: '+name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('native','portable'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--arm',choices=ARMS)
    a=parser.parse_args()
    a.out=a.out.resolve()
    if a.mode=='native':
        require(a.weights is not None,'Native replay needs the same local encoder registry')
        a.weights=a.weights.resolve()
        names=[a.arm] if a.arm else list(ARMS)
        reports,inputs={},{}
        for name in names:
            reports[name],inputs[name]=audit_arm(a,name)
            print(name,'replayed',reports[name]['actions'],'native actions,',reports[name]['swap_tests_changing_action'],
                  'hypothetical changes',flush=True)
        verify_portable(inputs)
        if not a.arm:
            save(a.out/'decision_inputs.json',inputs)
            save(a.out/'decision_audit.json',dict(native=reports,portable_input_sha256=sha(a.out/'decision_inputs.json'),
                                                    hypothetical_cases_executed_as_training=0))
    else:
        verify_portable(read(a.out/'decision_inputs.json'))
        print('All published exploration rankings reproduced')
