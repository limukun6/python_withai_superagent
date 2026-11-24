"""Publish one anonymous historical decision state and B's pure action for interface adaptation."""
import argparse
from pathlib import Path

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_scheduler import next_action


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixture-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    state=read(args.fixture_run/'state_2.json');caps=read(args.fixture_run/'run_manifest.json')['capabilities']
    if 'training_profile' not in state or state['budget']['remaining_fit_allowance']<=state['budget']['reserved_final_refit']:
        raise ValueError('Expected an actual profiled decision-time snapshot with available fit allowance')
    action=next_action(state,capabilities=caps)
    if action['action']!='explore':raise ValueError('Anonymous example should demonstrate an actual next action')
    state['provenance']=dict(source_fixture_only=True,paid_ledger_eligible=False,historical_replay=True,
        executable_for_continuation=False,snapshot_sha256=sha(args.fixture_run/'state_2.json'),
        notice='Budget is the recorded budget at this historical decision, not current spend allowance')
    for name,value in (('decision_fixture.json',state),('action_fixture.json',action)):
        if (args.output/name).exists():raise FileExistsError('Do not overwrite an agreed example')
        save(args.output/name,value)
    print(dict(status='exported_fixture',B_action=action['action'],candidate=action['candidate_id'],actual_classifier_fits=0,real_api_requests=0))
