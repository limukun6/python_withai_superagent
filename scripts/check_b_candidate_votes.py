"""Freeze three candidates from each fixed path's opening decision, then evaluate their archived votes."""
import argparse
from pathlib import Path

import numpy as np

from scripts import run_b1_ab_acceptance as fixture
from scripts.measure_b1_headroom import archive_for, checked_arm
from scripts.probe_task_headroom import observed_coverage, paired
from scripts.replay_b1_observed_actions import SEEDS, TASKS
from superagent.ab_comparison import MIN_GAIN
from superagent.ab_run import sources
from superagent.ab_scheduler import CONFIRMATIONS
from superagent.metrics import completed, score_of
from superagent.scheduler_bridge import fingerprint


def resolve_members(state):
    history = state['history']
    if (len(history) != 4 or len({r['candidate'] for r in history}) != 4 or
            any(r['split_id'] != 'primary' or not completed(r) for r in history)):
        raise ValueError('Expected four genuine primary trials before the first confirmation')
    ranked = sorted(history, key=lambda r: (-score_of(r), r['candidate']))
    return [r['candidate'] for r in ranked[:3]]


def prepare(roots, output):
    output.mkdir(parents=True, exist_ok=False)
    fixture.offline(output)
    code = Path(__file__)
    fixture.write(output / 'protocol.json', dict(batch='B1-11', driver_sha256=fixture.sha(code.read_bytes()),
        source_sha256=fixture.digest(sources()), plan_sha256=fixture.sha((fixture.ROOT/'docs/batches/B1_11_vote_plan.md').read_bytes()),
        source_protocols={n:fixture.sha((archive_for(n)/'protocol.json').read_bytes()) for n in TASKS},
        fits_allowed=0, new_model_predictions_allowed=0, api_allowed=0, expected_comparisons=16))
    menus = {}
    for name, root in roots.items():
        canonical = archive_for(name)/'protocol.json'
        if (root/'protocol.json').read_bytes() != canonical.read_bytes():
            raise ValueError('Original pre-registered tasks differ')
        for seed in SEEDS:
            arm = f'{seed}_fixed'
            run = root/arm/'run'
            state = fixture.read(run/'state_5.json')
            mark = fixture.read(run/'research_policy.json')['decisions'][4]
            if mark['trial'] != 5 or mark['state_sha256'] != fingerprint(state):
                raise ValueError('Pre-confirmation state is different from the frozen decision trace')
            menus[f'{name}/{arm}'] = dict(members=resolve_members(state), state_sha256=fingerprint(state))
    filename = output/'members.json'
    fixture.write(filename, menus)
    fixture.write(output/'completion.json',dict(status='prepared',sets=len(menus),members_sha256=fixture.sha(filename.read_bytes()),
                  confirmation_scores_read=0, new_fits=0, model_api_requests=0))


def majority(leader, second, third):
    a, b, c = (np.asarray(v) for v in (leader, second, third))
    if a.ndim != 1 or not len(a) or a.shape != b.shape or a.shape != c.shape:
        raise ValueError('All three predictions must cover the same ordered rows')
    return np.where(b == c, b, a)


def subset(metric, y, groups, pred, names, mask):
    if not np.any(mask):
        return dict(status='empty')
    y, groups = y[mask], groups[mask]
    three = {name:pred[name][mask] for name in names}
    lead = three[names[0]]
    voted = majority(*(three[n] for n in names))
    base = metric.evaluate(y,lead)['score']
    vote = metric.evaluate(y,voted)['score']
    return dict(status='observed', leader_primary=base, vote_primary=vote, **paired(metric,y,lead,voted,groups),
                label_aware_correctable=observed_coverage(metric,y,three,names[0])['correctable_among_observed'])


def score_arm(name, root, seed, members, frozen):
    tag = f'{seed}_fixed'
    canonical = archive_for(name)
    entry = fixture.read(canonical/'protocol.json')['tasks'][tag]
    model_hash = fixture.read(canonical/'all_models_frozen.json')[tag]
    arm = checked_arm(root/tag, entry, model_hash, frozen)
    baseline_groups = set(arm['trusted']['primary']['groups'])
    checks = {}
    for split in CONFIRMATIONS:
        bucket, trusted = arm['by_split'][split], arm['trusted'][split]
        if set(bucket) != set(members):
            checks[split] = dict(status='missing_or_different_members', measured=sorted(bucket))
            continue
        groups = np.asarray(trusted['groups'])
        y = np.asarray(trusted['labels'])
        pred = {n:np.asarray(bucket[n]['predictions']) for n in members}
        clean = np.array([g not in baseline_groups for g in groups])
        checks[split] = dict(status='observed', primary_eval_group_overlap_rows=int(np.sum(~clean)),
                             all_rows=subset(arm['metric'],y,groups,pred,members,np.ones(len(y), dtype=bool)),
                             outside_primary_eval_groups=subset(arm['metric'],y,groups,pred,members,clean))
    one, two = (arm['trusted'][k] for k in CONFIRMATIONS)
    return dict(metric=arm['metric'].name, members=members, partitions=checks,
                shared_confirmation_rows=len(set(one['row_ids']) & set(two['row_ids'])),
                shared_confirmation_groups=len(set(one['groups']) & set(two['groups'])))


def evaluate(roots, registered, expected_sha, output):
    output.mkdir(parents=True, exist_ok=False)
    fixture.offline(output)
    results = {}
    try:
        receipt = fixture.read(registered/'completion.json')
        file = registered/'members.json'
        original = fixture.read(registered/'protocol.json')
        frozen = fixture.digest(sources())
        if (receipt['status'] != 'prepared' or receipt['sets'] != 8 or receipt['members_sha256'] != expected_sha
                or fixture.sha(file.read_bytes()) != expected_sha or original['source_sha256'] != frozen
                or original['driver_sha256'] != fixture.sha(Path(__file__).read_bytes())
                or original['plan_sha256'] != fixture.sha((fixture.ROOT/'docs/batches/B1_11_vote_plan.md').read_bytes())):
            raise ValueError('Prepared candidate registration is incomplete or differs')
        fixture.write(output/'protocol.json',dict(original,members_sha256=expected_sha))
        menus = fixture.read(file)
        for name, root in roots.items():
            if fixture.sha((root/'protocol.json').read_bytes()) != original['source_protocols'][name]:
                raise ValueError('Different underlying development paths')
            seeds = {}
            for seed in SEEDS:
                tag = f'{seed}_fixed'
                row = menus[f'{name}/{tag}']
                state = fixture.read(root/tag/'run/state_5.json')
                if row['state_sha256'] != fingerprint(state) or row['members'] != resolve_members(state):
                    raise ValueError('Candidate list differs from original pre-confirmation state')
                seeds[str(seed)] = score_arm(name,root,seed,row['members'],frozen)
            gains = [p['all_rows']['right_minus_left_points'] for s in seeds.values() for p in s['partitions'].values()
                     if p['status'] == 'observed']
            results[name] = dict(seeds=seeds, observed_partitions=len(gains),
                                 meets_predeclared_followup=len(gains)==4 and min(gains)>0 and np.mean(gains)>=100*MIN_GAIN)
            results[name]['meets_predeclared_followup'] = bool(results[name]['meets_predeclared_followup'])
            fixture.write(output/'results.json',results)
        fixture.write(output/'completion.json',dict(status='passed', sets=8, expected_partitions=16,
            observed_partitions=sum(v['observed_partitions'] for v in results.values()),
            fits=0, new_model_predictions=0, outer_scores_read=0, model_api_requests=0))
    except Exception as error:
        fixture.write(output/'completion.json',dict(status='failed',reason=str(error),completed_tasks=list(results),
                      fits=0,new_model_predictions=0,outer_scores_read=0,model_api_requests=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','evaluate'])
    for name in (*TASKS,'output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registered',type=Path)
    parser.add_argument('--members-sha256')
    args = parser.parse_args()
    roots = {name:getattr(args,name) for name in TASKS}
    if args.stage == 'prepare':
        prepare(roots,args.output)
    else:
        if not args.registered or not args.members_sha256:
            parser.error('Evaluation requires the prepared directory and its sealed menu digest')
        evaluate(roots,args.registered,args.members_sha256,args.output)
