"""Replay decisions from observed training histories and stop interpreting the new policy at its first divergence."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile

from superagent import ab_adaptive_opening, ab_run
from superagent.ab_catalog import untried_candidates
from superagent.ab_close_singles import SingleConfirmationPolicy
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.scheduler_bridge import validate_action


SOURCES = {
    'full_titles': '6aac89257cef32aa0c498cfb0c9de4df73db4b91393551c5441ffa771016c393',
    'movie_dialogue': 'f664b56b946d83446e75831ee899c31cb4016eeb2a798d0832d676c2ed47bf73',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def decision(action):
    return [action[k] for k in ('action', 'candidate_id', 'split_id')]


def replay(key, root):
    agent = root / ('agent' if key == 'full_titles' else 'autonomous')
    receipt_file = root / ('models_frozen.json' if key == 'full_titles' else 'model_frozen.json')
    receipt = read(receipt_file)
    manifest = read(agent / 'run_manifest.json')
    archive = root / 'anonymous.zip'
    require(manifest['task_archive_sha256'] == ab_run.sha(archive) == SOURCES[key], 'Different training packet')
    require(manifest['scheduler'] == 'ab-adaptive-complement' and manifest['planner_mode'] == 'local'
            and manifest['preference_catalog'] == 'deep-e5-familiarity', 'Different historical scheduler')
    os.environ[PREFERENCE_CATALOG_ENV] = manifest['preference_catalog']
    task = load_development_task(archive)
    require(task.X_val is None and task.y_val is None, 'Expected training-only input')
    contract = TaskSpec.from_task(task).record()
    caps = manifest['capabilities']
    current = ab_run.registered_capabilities(contract, len(task.y_train), manifest['catalog_scope'])
    projected = deepcopy(caps)
    revision_changes = {}
    if key == 'movie_dialogue':
        # Keep the configuration the historical planner actually saw when replaying its decisions.
        for name in ('minilm_l12_text_with_lexical_features', 'minilm_l12_with_corpus_familiarity'):
            old, now = caps[name]['params']['revision'], current[name]['params']['revision']
            require((old, now) == ('52f9d2dd711ffc3434e6cff17284b13a11629a90', 'a50ef00143b4d5391434df20ae11632588ac25be'),
                    'Movie history differs from the two recorded revision changes')
            projected[name]['params']['revision'] = now
            revision_changes[name] = {'observed': old, 'current': now}
    require(projected == current and contract == manifest['task_contract'], 'Unexplained candidate or task contract change')
    actions = read(agent / 'actions.json')
    receipt_actions = actions if key == 'full_titles' else [
        {k: v for k, v in a.items() if k != 'estimated_classifier_fits'} for a in actions]
    require(receipt['actions'] == receipt_actions, 'Action file differs from frozen receipt')
    splits, entries, diverged = {}, [], False
    new = SingleConfirmationPolicy(ab_adaptive_opening)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        trusted, allowed = ab_run.freeze_partitions(task, tmp, manifest['seed'], MetricSpec.for_task(task), preference_vote=True)
        for split in allowed:
            require(read(tmp / (split + '_split.json')) == read(agent / (split + '_split.json')), 'Partition changed')
            splits[split] = ab_run.sha(tmp / (split + '.joblib'))
        for i, expected in enumerate(actions, 1):
            state_path = agent / f'state_{i}.json'
            state = read(state_path)
            require(state['task'] == contract and state['available_candidates'] == untried_candidates(caps, state['history']),
                    'Historical candidates changed')
            options = {'cover_untried_encoder': manifest['cover_untried_encoder']}
            original = ab_adaptive_opening.action(state, caps, trusted, **options)
            validate_action(state, original, caps, allowed)
            require(original == expected, 'Default action did not reproduce its observed history')
            step = dict(state=i, state_sha256=ab_run.sha(state_path), original=decision(original))
            if not diverged:
                proposed = new.action(state, caps, trusted, **options)
                validate_action(state, proposed, caps, allowed)
                step['proposed'] = decision(proposed)
                if step['original'] != step['proposed']:
                    step['review'] = new.review(state, trusted).get('close_singles')
                    diverged = True
            entries.append(step)
        old_review = ab_adaptive_opening.review(state, trusted)
        require(actions[-1]['action'] == 'stop' and old_review == read(agent / 'selection.json')['confirmation']['b_scheduler_review'],
                'Default final selection review did not match')
        new_members = new.review(state, trusted)['members'] if not diverged else None
    return dict(archive_sha256=SOURCES[key], historical_receipt_sha256=ab_run.sha(receipt_file),
                training_rows=len(task.y_train), source_file_count=len(manifest['source_files']), candidate_count=len(caps),
                splits=splits, catalogue_revision_changes=revision_changes, decision_steps=entries,
                original_actions_reproduced=len(entries), diverged=diverged,
                observed_terminal_members=old_review['members'], new_terminal_members_if_reached=new_members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('full-titles', 'movie-dialogue', 'weights', 'output'):
        parser.add_argument('--' + option, required=True, type=lambda s: Path(s).resolve())
    args = parser.parse_args()
    def offline(event, unused):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            raise PermissionError('Offline history replay')
    sys.addaudithook(offline)
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(args.weights / 'all-MiniLM-L6-v2'),
                      SUPERAGENT_DEEP_MINILM_WEIGHTS=str(args.weights / 'all-MiniLM-L12-v2'),
                      SUPERAGENT_E5_SMALL_WEIGHTS=str(args.weights / 'e5-small-v2'),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    runs = {key: replay(key, getattr(args, key)) for key in SOURCES}
    result = dict(runs=runs, replay_source=ab_run.sources(), runner_sha256=ab_run.sha(Path(__file__)),
                  actual_classifier_fits=0, model_api_calls=0, new_scores_measured=False)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print({k: (v['original_actions_reproduced'], v['diverged']) for k, v in runs.items()})


if __name__ == '__main__':
    main()
