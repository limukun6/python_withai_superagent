"""Compare the new official A controls against the recorded decisions on B's earlier actual states."""
import argparse
from pathlib import Path

from scripts import run_b1_ab_acceptance as fixture
from scripts.measure_b1_headroom import archive_for
from scripts.replay_b1_observed_actions import POLICIES, SEEDS, TASKS, trusted_input
from superagent.ab_fixed_schedule import fixed_action
from superagent.ab_run import sources, validate_options
from superagent.ab_scheduler import next_action
from superagent.cli import build_parser
from superagent.scheduler_bridge import fingerprint, validate_action


EXECUTION = ('action','candidate_id','split_id','estimated_classifier_fits')


def official_options():
    result = {}
    for scheduler in ('ab-v1','ab-fixed'):
        for planner in ('local','raw-history','checked-history'):
            args = build_parser().parse_args(['run','task.zip','--scheduler',scheduler,'--portfolio','ab-v1',
                                              '--fit-limit','11','--planner',planner])
            try:
                limit = validate_options(args)
                result[scheduler+'/'+planner] = dict(accepted=True, actual_fit_limit=limit)
            except ValueError as error:
                result[scheduler+'/'+planner] = dict(accepted=False, reason=str(error))
    return result


def compare_path(root, key, policy, original):
    arm = root/key
    entry = original['tasks'][key]
    caps, trusted = trusted_input(arm,entry,original['source_sha256'])
    run = arm/'run'
    trace = fixture.read(run/'research_policy.json')['decisions']
    actual = fixture.read(run/'actions.json')
    counts = dict(full_field_matches=0,reason_changes=0,evidence_reference_changes=0,admitted_actions=0)
    differing = None
    for step,(record,old) in enumerate(zip(trace,actual,strict=True),1):
        state = fixture.read(run/f'state_{step}.json')
        if record != dict(trial=step,state_sha256=fingerprint(state),action_sha256=fingerprint(old)):
            raise ValueError('Original decision or prior state changed')
        decide = fixed_action if policy=='fixed' else next_action
        new = decide(state,capabilities=caps,trusted_partitions=trusted)
        validate_action(state,new,caps,entry['splits'])
        counts['admitted_actions'] += 1
        if any(old[k] != new[k] for k in EXECUTION):
            differing = dict(step=step, prior={k:old[k] for k in EXECUTION}, current={k:new[k] for k in EXECUTION})
            break
        counts['full_field_matches'] += int(old==new)
        counts['reason_changes'] += int(old['reason']!=new['reason'])
        counts['evidence_reference_changes'] += int(old['evidence_trials']!=new['evidence_trials'])
    return dict(**counts,earlier_path_actions=len(actual),common_executable_prefix=counts['admitted_actions']-int(differing is not None),
                first_execution_difference=differing)


def study(args):
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    fixture.offline(output)
    report = {}
    try:
        options = official_options()
        fixture.write(output/'protocol.json',dict(batch='BA11',original_source_by_task={n:fixture.read(archive_for(n)/'protocol.json')['source_sha256'] for n in TASKS},
            joint_production_sources=fixture.digest(sources()),plan_sha256=fixture.sha((fixture.ROOT/'docs/batches/BA11_scheduler_plan.md').read_bytes()),
            driver_sha256=fixture.sha(Path(__file__).read_bytes()),fits_allowed=0,api_allowed=0))
        for name in TASKS:
            root=getattr(args,name)
            if (root/'protocol.json').read_bytes() != (archive_for(name)/'protocol.json').read_bytes():
                raise ValueError('Original registration differs')
            original=fixture.read(root/'protocol.json')
            report[name]={}
            for seed in SEEDS:
                for policy in POLICIES:
                    key=f'{seed}_{policy}'
                    report[name][key]=compare_path(root,key,policy,original)
        fixture.write(output/'results.json',dict(tasks=report,official_options=options))
        rows = [x for paths in report.values() for x in paths.values()]
        fixture.write(output/'completion.json',dict(status='passed',paths=len(rows),
                executed_prefix_matches=sum(x['common_executable_prefix'] for x in rows),
                six_field_matches=sum(x['full_field_matches'] for x in rows),
                first_divergences=sum(x['first_execution_difference'] is not None for x in rows),
                fits=0,model_predictions=0,model_api_requests=0,outer_scores_read=0))
    except Exception as err:
        fixture.write(output/'completion.json',dict(status='failed',reason=str(err),started_tasks=list(report),
                      fits=0,model_api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for n in (*TASKS,'output'):
        parser.add_argument('--'+n,type=Path,required=True)
    study(parser.parse_args())
