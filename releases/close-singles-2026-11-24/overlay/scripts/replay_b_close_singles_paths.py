"""Check all decisions and final members in the measured arms against the budget-checked policy."""
import argparse
import os
from pathlib import Path
import tempfile

from superagent import ab_adaptive_opening, ab_run
from superagent.ab_catalog import untried_candidates
from superagent.ab_close_singles import SingleConfirmationPolicy
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.scheduler_bridge import validate_action
from scripts.run_b_autonomous_close_singles import ROOT, env, read, require, seal


def verify(archive, folders):
    task = load_development_task(archive)
    require(task.y_val is None, 'Expected the exact anonymous training data')
    spec = TaskSpec.from_task(task).record()
    results = {}
    with tempfile.TemporaryDirectory() as destination:
        trusted, allowed = ab_run.freeze_partitions(task, Path(destination), 18643, MetricSpec.for_task(task), preference_vote=True)
        for arm, path, policy in folders:
            manifest = read(path/'run_manifest.json')
            os.environ[PREFERENCE_CATALOG_ENV] = manifest['preference_catalog']
            caps = ab_run.registered_capabilities(spec,len(task.y_train),manifest['catalog_scope'])
            require(spec == manifest['task_contract'] and caps == manifest['capabilities'] and
                    ab_run.sha(archive) == manifest['task_archive_sha256'], 'Different contract, menu or input')
            for name, split in allowed.items():
                require(split['evaluation_payload_sha256'] == ab_run.sha(path/(name+'.joblib')) and
                        read(Path(destination)/(name+'_split.json')) == read(path/(name+'_split.json')), 'Different split')
            actions = read(path/'actions.json')
            for step, actual in enumerate(actions, 1):
                state = read(path/f'state_{step}.json')
                require(untried_candidates(caps,state['history']) == state['available_candidates'], 'Candidate projection differs')
                expected = policy.action(state, caps, trusted, cover_untried_encoder=True)
                validate_action(state, expected, caps, allowed)
                require(actual == expected, 'Decision differs at '+arm+'/'+str(step))
            review = policy.review(state,trusted)
            require(review == read(path/'selection.json')['confirmation']['b_scheduler_review'], 'Final selection differs')
            results[arm] = dict(matching_actions=len(actions), action_file_sha256=ab_run.sha(path/'actions.json'),
                                selected_members=review['members'], final_review_matches=True, menu=len(caps))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cold','original','continued','weights','output'):
        parser.add_argument('--'+name, type=lambda s:Path(s).resolve(), required=True)
    a = parser.parse_args()
    os.environ.update(env(ROOT,a.weights,a.output.with_suffix('.unused')))
    from scripts import a38_predict_guard
    revised = SingleConfirmationPolicy(ab_adaptive_opening)
    results = verify(a.cold/'anonymous.zip', [('cold_control',a.cold/'control/agent',ab_adaptive_opening),
                                             ('cold_proposal',a.cold/'proposal/agent',revised)])
    results.update(verify(a.original/'anonymous.zip',[('resumed_proposal',a.continued/'agent',revised)]))
    seal(a.output,dict(paths=results, production=ab_run.sources(), runner=ab_run.sha(Path(__file__)),
                       physical_fits=0, model_api_calls=0))
    print({key:value['matching_actions'] for key,value in results.items()})
