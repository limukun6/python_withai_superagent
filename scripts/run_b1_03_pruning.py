"""Replay the frozen B1-02 decision prefix and run one new anonymous path offline."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from superagent.ab_catalog import common_catalog
from superagent.ab_scheduler import next_action
from superagent.contracts import TaskSpec
from superagent.prediction_evidence import fingerprint
from scripts import run_b1_ab_acceptance as fixture


ARCHIVE = fixture.ROOT/'research/b1_ab_handoff/accepted'
ACTION_KEYS = ('action','candidate_id','split_id','estimated_classifier_fits')


def choice(action):
    return {key:action[key] for key in ACTION_KEYS}


def replay_prefix(path, baseline, partitions):
    """Read exactly one previously visible state at a time and stop at first divergence."""
    with path.open(encoding='utf-8') as stream:
        for number,old in enumerate(baseline['actions'],1):
            state = json.loads(next(stream))
            if fixture.digest(state) != old['state_sha256'] or len(state['history']) != number-1:
                raise ValueError('Input does not match the historical state committed for this step')
            new = next_action(state,trusted_partitions=partitions)
            if choice(new) != choice(old):
                return dict(matching_choices=number-1, divergence_at=number, state_sha256=old['state_sha256'],
                            prior=choice(old), revised=new, prior_fits_seen=state['budget']['completed_classifier_fits'])
    raise ValueError('No decision changed on the archived state prefix')


def summary(actions,final):
    counts = Counter(a['action'] for a in actions)
    return dict(explorations=counts['explore'],confirmations=counts['confirm'],refits=final['full_refits'],
                total_fits=final['task_classifier_fits'],selected=final['family'],
                selection_splits=final['selection_splits'],training_internal_score=final['selection_score'])


def run(directory, prior_states):
    directory.mkdir(parents=True,exist_ok=False)
    fixture.offline(directory)
    result = {}
    try:
        previous = fixture.read(ARCHIVE/'results.json')
        frozen = fixture.read(ARCHIVE/'protocol.json')
        task,query = fixture.packages(directory)
        partitions = fixture.freeze_partitions(task,directory)
        menu = common_catalog(TaskSpec.from_task(task).record(),train_rows=len(task.y_train))
        keys = ('evaluation_payload_sha256','row_ids_sha256')
        for split,part in partitions.items():
            observed = (part['evaluation_payload_sha256'],fingerprint(part['row_ids']))
            if observed != tuple(frozen['grouped_splits'][split][key] for key in keys):
                raise ValueError('Anonymous partition or its row ordering differs from B1-02')
        if fixture.digest(menu) != frozen['catalog_sha256']:
            raise ValueError('Candidate capabilities differ from B1-02')
        protocol = dict(batch='B1-03',parent='ad981ea04e9525d20c5ee43e7637d66cfa6e4903',
                        prior_results_sha256=fixture.sha((ARCHIVE/'results.json').read_bytes()),
                        source_sha256=fixture.digest(dict(fixture=fixture.sources(),
                            runner=fixture.sha(Path(__file__).read_bytes()),
                            plan=fixture.sha((fixture.ROOT/'docs/batches/B1_03_plan.md').read_bytes()))),
                        training_rows=len(task.y_train),data_seed=frozen['data_seed'],seeds=fixture.SEEDS,
                        grouped_splits=frozen['grouped_splits'],catalog_sha256=fixture.digest(menu),
                        original_states_supplied=prior_states is not None,fit_limit=11,paid_requests_allowed=0,
                        scope='anonymous numeric task only')
        fixture.write(directory/'protocol.json',protocol)
        if prior_states is not None:
            result['archived_state_replay'] = replay_prefix(prior_states,previous,partitions)
            if result['archived_state_replay']['divergence_at'] != 9:
                raise ValueError('Unexpected first divergence in archived-state replay')

        # The live path takes freshly generated data. Historical predictions and
        # the hypothetical replay decision above are not inputs to this controller.
        history,actions,began,review = fixture.observe(directory,task,partitions,menu)
        final = fixture.finish(directory,task,history,began,review)
        reload_receipt = fixture.predict_twice(directory,query)
        result.update(original=summary(previous['actions'],previous),revised=summary(actions,final),
                      actual_actions=actions,observations=history,promotions=review['promotions'],
                      physical_fit_calls=fixture.trace(directory),fresh_process_prediction=reload_receipt,
                      elapsed_seconds=round(time.monotonic()-began,3))
        if ([choice(a) for a in actions[:8]] != [choice(a) for a in previous['actions'][:8]]
                or history[:8] != previous['observations'][:8]):
            raise ValueError('Initial choices or observed results differed from the matched baseline')
        if previous['actions'][8]['action'] != 'confirm' or actions[8]['action'] != 'explore':
            raise ValueError('Unwinnable ninth action did not release the slot for exploration')
        if prior_states is not None and choice(actions[8]) != choice(result['archived_state_replay']['revised']):
            raise ValueError('Live action and pre-decision replay disagree')
        if not review['promotions'] or review['promotions'][0] != previous['incumbent_evidence']['promotions'][0]:
            raise ValueError('The observed comeback must remain eligible')
        if final['task_classifier_fits'] > 11 or final['full_refits'] != 1 or reload_receipt['rows_per_prediction'] != 31:
            raise ValueError('Fit cap, refit or prediction count mismatch')
        fixture.write(directory/'results.json',result)
        fixture.write(directory/'completion.json',dict(status='passed',first_divergence=9,
            first_eight_choices_and_scores_match=True,original_comeback_preserved=True,
            observed_fit_calls=final['task_classifier_fits'],api_requests=0,api_rmb=0,
            replay_classifier_fits=0,course_data_accessed=False,fresh_prediction_processes=2))
    except Exception as error:
        fixture.write(directory/'results.json',result)
        fixture.write(directory/'completion.json',dict(status='failed',error=str(error),
                                                       observed_fit_calls=len(fixture.trace(directory))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--prior-states',type=Path)
    args=parser.parse_args()
    run(args.directory.resolve(),args.prior_states)
