"""Reproduce signed local decisions and inspect the first within-route policy divergence."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

from scripts import replay_b1_observed_actions as replay, run_b1_ab_acceptance as rec
from superagent import ab_run, ab_scheduler
from superagent.scheduler_bridge import fingerprint


BASE = 'fc8506bc3e52c43a6332c270ce89f44a87e5a2c2'
ARCHIVES = (
    ('ab_a19_release_delivery','ed7261fd6b4d52a7817a1f96ca52393cb0a3e181','ba19_released_delivery',
     'ba19_released_native_01',(('coupon','coupon_full'),)),
    ('ab_contextual_preference','a9cc24276610cd06c7fdacc0b66ec96743e2ebd2','b_contextual_preference',
     'context_headlines_02',(('headlines_45101','45101_original'),('headlines_45102','45102_original'))),
    ('ab_coverage_scheduler','aea5815e8a6a16ee6dcade10e2c465e40b71dbe8','bs01_coverage',
     'bs01_two_splits_01',(('mrpc_45101','45101_original'),('mrpc_45102','45102_original'))),
    ('ab_refit_time','00935216e85790dbb0232a00d11c3a902a5eb1d3','b_cold_probe_prospective',
     'prospective_clock_01',(('sms','sms'),('fashion','fashion'))),
    ('ab_numeric_kernel','d972111d83bb4c78d212be8b76262a80c5cc982c','bn01_dry_bean',
     'bn01_bean_01',(('bean','baseline'),)),
    ('ab_portable_categorical','442bee1d25bacaa4dad6e22a7a2c11de9c8b8a0c','bc02_car',
     'bc02_car_dev_01',(('car','baseline'),)),
)


def pinned(root, commit, collection, path):
    original = subprocess.check_output(['git','show',f'{commit}:research/{collection}/{path.name}'],cwd=root)
    if rec.sha(original) != ab_run.sha(path):
        raise ValueError(f'Archived input registration changed: {collection}/{path.name}')
    return json.loads(original)


def registration(name, child, pre, protocol, caps):
    if name.startswith('headlines'):
        return pre['entries'][child], protocol['production']['original']
    if name.startswith('mrpc'):
        return pre['entries'][child.split('_')[0]], protocol['sources']['original']
    if name in ('sms','fashion'):
        return pre[name], protocol['original_production']
    if name == 'coupon':
        entry = pre['coupon']
        menu = entry['catalogs']['full']
        result = dict(candidate_ids=menu['ids'],catalog_sha256=menu['sha256'],splits=entry['splits'],
                      files={'train.zip':entry['train_sha256']})
    elif name == 'bean':
        menu = pre['catalogs'][child]
        result = dict(candidate_ids=list(menu),catalog_sha256=rec.digest(menu),splits=pre['splits'],
                      files={'train.zip':protocol['train_and_features']['train.zip']})
    else:
        # This earlier study published the IDs, order, source and splits; the complete cap dictionary is in its manifest.
        result = dict(candidate_ids=pre['candidate_ids'][child],catalog_sha256=rec.digest(caps),splits=pre['inner_splits'],
                      files={'train.zip':pre['train_sha256']})
    return result, protocol['source']


def inspect(case, entry, source, old, old_sha):
    run = case/'run'
    manifest = rec.read(run/'run_manifest.json')
    final = rec.read(run/'run_outcome.json')
    if (manifest['source_files']['ab_scheduler.py'] != old_sha or manifest['planner_mode'] != 'local'
            or manifest['task_contract']['formal_test_available_to_agent'] or final['formal_test_accessed']
            or final['finalization']['provided_validation_available'] or not final['source_unchanged']
            or not final['full_refit_completed'] or final['new_api_requests']):
        raise ValueError('Expected an original local training-only run with a verified full refit')
    caps, trusted = replay.trusted_input(case,entry,source)
    if list(caps) != entry['candidate_ids']:
        raise ValueError('Ordered candidate list changed')
    args = dict(capabilities=caps,trusted_partitions=trusted)
    actual = rec.read(run/'actions.json')
    admitted = [e for raw in (run/'events.jsonl').read_text().splitlines()
                if (e:=json.loads(raw))['kind']=='scheduler_action_admitted']
    changes = []
    for step,(response,event) in enumerate(zip(actual,admitted,strict=True),1):
        state = rec.read(run/f'state_{step}.json')
        if response != event['action'] or event['state_sha256'] != fingerprint(state) or old.next_action(state,**args) != response:
            raise ValueError(f'Original action, evidence, or signed state changed: {case.name} step {step}')
        proposal = ab_scheduler.next_action(state,**args)
        if proposal != response:
            a,b = response['candidate_id'],proposal['candidate_id']
            if (step == 1 or response['action'] != 'explore' or proposal['action'] != 'explore'
                    or caps[a]['route'] != caps[b]['route'] or caps[a]['resource'] != caps[b]['resource']
                    or response['estimated_classifier_fits'] != proposal['estimated_classifier_fits']
                    or response['evidence_trials'] != proposal['evidence_trials'] or a == b):
                raise ValueError('The intervention changed a different part of the decision')
            changes.append(dict(step=step,state_sha256=event['state_sha256'],original=a,proposed=b,route=caps[b]['route'],
                                remaining_fits=state['budget']['remaining_fit_allowance'],evidence=proposal['evidence_trials']))
        if old.next_action(state,allocation='fixed',**args) != ab_scheduler.next_action(state,allocation='fixed',**args):
            raise ValueError('Fixed allocation changed')
        menus = []
        def select_last(ids):
            menus.append(list(ids))
            return ids[-1]
        before = old.next_action(state,on_explore=select_last,**args)
        split = len(menus)
        after = ab_scheduler.next_action(state,on_explore=select_last,**args)
        if before != after or menus[:split] != menus[split:]:
            raise ValueError('The external picker or its complete ordered menu changed')
    return dict(verified_actions=len(actual), source=source, train_sha256=entry['files']['train.zip'],
                catalog_sha256=rec.digest(caps), catalog_size=len(caps), splits=entry['splits'],
                first_change_on_actual_prefix=next(iter(changes),None), all_one_step_changes=changes)


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    old_file = args.control.resolve()/'superagent/ab_scheduler.py'
    expected = subprocess.check_output(['git','show',BASE+':superagent/ab_scheduler.py'],cwd=rec.ROOT)
    if rec.sha(expected) != ab_run.sha(old_file):
        raise ValueError('The original scheduler is not the registered A29 source')
    spec = importlib.util.spec_from_file_location('superagent._before_route_study',old_file)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    rec.write(out/'protocol.json',dict(batch='B-ROUTE-01',plan_sha256=ab_run.sha(rec.ROOT/'docs/batches/B_ROUTE_01_representative_plan.md'),
              runner_sha256=ab_run.sha(Path(__file__)),base=BASE,prior_scheduler=ab_run.sha(old_file),
              new_scheduler=ab_run.sha(rec.ROOT/'superagent/ab_scheduler.py'),archives=ARCHIVES,fit_limit=0,api_limit=0))
    reports = {}
    try:
        for folder, commit, collection, local, cases in ARCHIVES:
            root = args.workspace.resolve()/folder
            archive = root/'runs'/local
            proto,pre = (pinned(root,commit,collection,archive/f) for f in ('protocol.json','preflight.json'))
            for name,child in cases:
                case = archive/child
                caps = rec.read(case/'run/run_manifest.json')['capabilities']
                entry,source = registration(name,child,pre,proto,caps)
                reports[name] = inspect(case,entry,source,old,rec.sha(expected))
                rec.write(out/'checkpoint.json',reports)
        if len(reports)!=9 or rec.trace(out):
            raise ValueError('Unexpected number of runs or supervised fits')
        rec.write(out/'results.json',reports)
        rec.write(out/'completion.json',dict(status='completed',paths=len(reports),
                  verified_actions=sum(j['verified_actions'] for j in reports.values()),
                  divergent_paths=sum(j['first_change_on_actual_prefix'] is not None for j in reports.values()),
                  new_fits=0,new_model_requests=0,new_grades=0))
    except Exception as exc:
        rec.write(out/'completion.json',dict(status='failed',reason=str(exc),verified_paths=list(reports),new_fits=len(rec.trace(out))))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control','workspace','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
