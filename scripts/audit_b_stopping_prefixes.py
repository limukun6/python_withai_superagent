"""Ask whether a stable incumbent would have stopped a real historical search too early."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


RULES = ('stable_after_three_observations', 'also_paired_on_confirmation')
HERE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def same_files(directory, reference, files):
    for name in files:
        if sha(directory/name) != sha(reference/name):
            raise ValueError(f'Registered evidence changed: {directory.name}/{name}')


def inputs(source, cohort):
    if cohort == 'b1':
        for task in ('headlines','coupon','bean','sms'):
            titles = task == 'headlines'
            batch = 'b1_07_headlines_01' if titles else f'b1_08_{task}_01'
            root = source/'runs'/batch
            signed = source/'research'/('b1_07_allocation' if titles else f'b1_08_allocation/{task}')
            same_files(root, signed, ('protocol.json','all_models_frozen.json','completion.json'))
            if titles:
                receipt = read(signed/'recount/protocol.json')
                raw = read(root/'results.json')
                published = read(signed/'results.json')
                if (receipt['original_results_sha256']!=sha(root/'results.json') or
                        receipt['frozen_models_sha256']!=sha(root/'all_models_frozen.json') or
                        {k:v for k,v in raw.items() if k!='paired_counts'}!={k:v for k,v in published.items() if k!='paired_counts'}):
                    raise ValueError('Archived title evidence differs beyond its registered paired-count correction')
            else:
                same_files(root,signed,('results.json',))
            protocol = read(signed/'protocol.json')
            results = {r['name']:r for r in read(signed/'results.json')['paths']}
            frozen = read(signed/'all_models_frozen.json')
            for seed in ('45102','45103'):
                for policy in ('fixed','adaptive'):
                    key = seed+'_'+policy
                    yield (task,seed,policy,root/key,protocol['tasks'][key],protocol['source_sha256'],
                           results[key],frozen[key])
    else:
        root = source/'runs/b2_composition_01'
        signed = source/'research/b2_bounded_text_composition'
        same_files(root,signed,('protocol.json','preflight.json','results.json','all_six_models_frozen.json','completion.json'))
        expected = read(signed/'results.json')['paths']
        frozen = read(signed/'all_six_models_frozen.json')
        source_hash = read(signed/'protocol.json')['source']
        prereg = read(signed/'preflight.json')
        for task in ('headlines','sms'):
            pre = prereg[task]
            for policy in ('original','expanded','composed'):
                key = task+'_'+policy
                entry = dict(files={'train.zip':pre['train_sha256']},splits=pre['splits'],
                             catalog_sha256=pre['base_sha256' if policy=='original' else 'expanded_sha256'])
                yield task,'b2_development',policy,root/key,entry,source_hash,expected[key],frozen[key]


def paired_confirmation(state, incumbent):
    by_split = {}
    for proof in state['prediction_evidence']:
        by_split.setdefault(proof['split_id'],set()).add(proof.get('candidate_id',proof.get('candidate')))
    primary = by_split.get('primary',set())
    for split in state['registered_confirmation_splits']:
        shared = primary & by_split.get(split,set())
        if incumbent in shared and len(shared) >= 2:
            return True
    return False


def inspect_path(cohort, task, seed, policy, case, entry, source_hash, expected, frozen, current):
    from scripts.replay_b1_observed_actions import trusted_input
    from superagent.ab_scheduler import next_action, selection_evidence
    from superagent.scheduler_bridge import fingerprint

    if current != source_hash:
        raise ValueError('Use the original, hash-matched production source checkout for the cohort')
    caps, trusted = trusted_input(case,entry,source_hash)
    run = case/'run'
    actions = read(run/'actions.json')
    history = read(run/'history.json')
    evidence = read(run/'prediction_evidence.json')
    original = read(run/'final_selection.json')
    ledger = read(run/'classifier_fits.json')
    manifest = read(run/'run_manifest.json')
    opts = dict(capabilities=caps,trusted_partitions=trusted)
    if cohort == 'b1':
        opts['allocation'] = policy
        marks = read(run/'research_policy.json')['decisions']
    elif policy == 'composed':
        from superagent.ab_text_recipes import propose
        opts['on_local_recipe'] = propose
    if (actions != expected['actions'] or len(actions) != len(history)+1 or actions[-1]['action'] != 'stop'
            or not original['refit'] or original['family'] != expected['selected']
            or sha(run/original['model_file']) != frozen or original['model_sha256'] != frozen
            or ledger['attempted_classifier_fits'] != ledger['completed_classifier_fits'] or ledger['unknown_failed_attempts']
            or ledger['attempted_classifier_fits'] != expected['fits'] or not 1 < expected['fits'] <= 11):
        raise ValueError('Final model, original actions or log of physical fits differs')
    workers = ledger['workers']
    if (len(workers) != len(history)+1 or workers[-1]['artifact'] != 'refit_component_0.joblib'
            or not all(w['status']=='completed' and w['attempted_classifier_fits']==w['completed_classifier_fits'] for w in workers)):
        raise ValueError('Stopping audit requires a complete path with a single known final refit')
    fits, incumbent_ids, trajectory, chosen = 0,[],[],{}
    for step,action in enumerate(actions,1):
        state = read(run/f'state_{step}.json')
        if (state['history']!=history[:step-1] or state['prediction_evidence']!=[p for p in evidence if p['trial'] < step]
                or state['budget']['attempted_classifier_fits']!=fits or state['budget']['completed_classifier_fits']!=fits
                or next_action(state,**opts)!=action):
            raise ValueError(f'Genuine pre-action evidence or reproduced action differs at {case.name}/{step}')
        if cohort=='b1' and marks[step-1] != dict(trial=step,state_sha256=fingerprint(state),action_sha256=fingerprint(action)):
            raise ValueError('Historical signed state or action changed')
        winner = selection_evidence(state,trusted_partitions=trusted)['incumbent']
        if step>1:
            incumbent_ids.append(winner)
            primary = {p.get('candidate_id',p.get('candidate')) for p in state['prediction_evidence'] if p['split_id']=='primary'}
            stable = len(primary)>=3 and len(incumbent_ids)>=3 and len(set(incumbent_ids[-3:]))==1
            paired = paired_confirmation(state,winner)
            info = dict(observations=step-1,incumbent=winner,physical_search_fits=fits,primary_count=len(primary),
                        has_paired_confirmation=paired,state_sha256=fingerprint(state))
            trajectory.append(info)
            for name,eligible in zip(RULES,(stable,stable and paired),strict=True):
                if name not in chosen and (eligible or action['action']=='stop'):
                    remain = expected['fits']-(fits+caps[winner]['refit_classifier_fits'])
                    if not 0<=remain<expected['fits']:
                        raise ValueError('Invalid refit reserve or saved-fit estimate')
                    chosen[name]=dict(**info,rule_met=eligible,earlier=action['action']!='stop',potential_fits_saved=remain,
                                      same_winner=winner==original['family'])
        if step<=len(history):
            if workers[step-1]['candidate']!=history[step-1]['candidate'] or history[step-1]['classifier_fits']!=workers[step-1]['completed_classifier_fits']:
                raise ValueError('Trial order differs from physical workers')
            fits+=workers[step-1]['completed_classifier_fits']
    if incumbent_ids[-1] != original['family'] or fits+workers[-1]['completed_classifier_fits'] != expected['fits']:
        raise ValueError('Full observed winner or fit accounting failed to reproduce')
    # Hash the actual complete refit payload, including its inputs and objective.
    identity = dict(task=task,split=seed,train_sha256=sha(case/'train.zip'),source_sha256=source_hash,
                    full_refit_payload_sha256=sha(run/'refit_data.joblib'),seed=manifest['seed'],
                    dependencies=manifest['dependencies'],python=manifest['python'])
    return dict(task=task,split=seed,policy=policy,original=original['family'],original_fits=expected['fits'],
                full_search_observations=len(history),states_verified=len(actions),trajectory=trajectory,
                rules=chosen,identity=identity),caps


def archived_grade(case, cohort, expected):
    row = expected['delivery'] if cohort=='b1' else expected['grade']
    export_prefix='pred_' if cohort=='b1' else 'export_'
    inputs=[]
    for i in range(2):
        path=case/(export_prefix+str(i))
        manifest=read(path/'prediction_manifest.json')
        if sha(path/'rows.csv')!=row['csv_sha256'] or manifest['prediction_count']!=row['rows'] or manifest['test_labels_read']:
            raise ValueError('Archived development prediction export changed')
        inputs.append(sha(Path(manifest['input_source'])))
    if inputs[0]!=inputs[1]:
        raise ValueError('Archived evaluation inputs differ')
    metrics=row if cohort=='b1' else row['scores']
    return dict(rows=row['rows'],correct=row['correct'],metric=metrics['primary_metric'],score=metrics['score'],
                features_sha256=inputs[0],csv_sha256=row['csv_sha256'])


def study(args):
    source=args.source.resolve()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(source))
    from scripts import run_b1_ab_acceptance as rec
    from superagent.ab_run import sources
    if rec.ROOT.resolve()!=source:
        raise ValueError('Loaded the wrong historical code')
    rec.offline(out)
    current=rec.digest(sources())
    save(out/'protocol.json',dict(batch='B-STOPPING-PREFIX-01',cohort=args.cohort,source_sha256=current,
         driver_sha256=sha(Path(__file__)),plan_sha256=sha(HERE/'docs/batches/B_STOPPING_PREFIX_01_plan.md'),
         rules=list(RULES),new_fit_limit=0,model_api_limit=0))
    try:
        paths, originals, caps = {},{},{}
        for task,seed,policy,case,entry,source_hash,expected,frozen in inputs(source,args.cohort):
            name='/'.join((task,seed,policy))
            paths[name],caps[name]=inspect_path(args.cohort,task,seed,policy,case,entry,source_hash,expected,frozen,current)
            originals[name]=(case,expected)
        save(out/'all_prefix_decisions_frozen.json',paths)
        grades={name:archived_grade(case,args.cohort,x) for name,(case,x) in originals.items()}
        for name,p in paths.items():
            for effect in p['rules'].values():
                winner=effect['incumbent']
                matches=[n for n,q in paths.items() if q['identity']==p['identity'] and q['original']==winner
                         and caps[n][winner]==caps[name][winner] and grades[n]['features_sha256']==grades[name]['features_sha256']]
                if len({grades[n]['csv_sha256'] for n in matches})>1:
                    raise ValueError('Same eligible full-refit identity yielded conflicting actual exports')
                effect['matching_full_runs']=matches
                if matches:
                    a,b=grades[name],grades[matches[0]]
                    if (a['metric'],a['rows'])!=(b['metric'],b['rows']):
                        raise ValueError('Historical score definitions differ')
                    effect['existing_model_difference']=dict(rows=a['rows'],metric=a['metric'],correct=b['correct'],
                        full_correct=a['correct'],score=b['score'],full_score=a['score'],
                        difference_pp=100*(b['score']-a['score']),csv_sha256=b['csv_sha256'])
                else:
                    effect['existing_model_difference']=None
        summary={}
        for rule in RULES:
            values=[p['rules'][rule] for p in paths.values()]
            summary[rule]=dict(paths=len(values),early=sum(v['earlier'] for v in values),
                different_winner=sum(not v['same_winner'] for v in values),potential_fits_saved=sum(v['potential_fits_saved'] for v in values),
                matched_full_runs=sum(bool(v['matching_full_runs']) for v in values))
        save(out/'results.json',dict(summary=summary,paths=paths))
        save(out/'completion.json',dict(status='completed',cohort=args.cohort,paths=len(paths),
            states=sum(p['states_verified'] for p in paths.values()),new_fits=0,new_model_predictions=0,
            new_model_api_requests=0,new_course_rmb=0,opened_formal_validation_test_or_h=False))
    except Exception as error:
        save(out/'completion.json',dict(status='failed',error=str(error),new_fits=0,new_model_api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=('b1','b2'),required=True)
    for name in ('source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    study(parser.parse_args())
