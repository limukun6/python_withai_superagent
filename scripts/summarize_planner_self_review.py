"""Verify frozen records and export aggregate research evidence without fitting classifiers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts.replay_planner_review_archive import challenge, episode, top
from scripts.run_planner_self_review import ARMS, source_identity
from scripts.unified_released_study import read, require, sha


def check_handoff(root, portable, source):
    archive = root / portable['archive']
    require(sha(archive) == portable['sha256'], 'Portable archive changed')
    frozen = root / 'pinned-source'
    with zipfile.ZipFile(archive) as bundle:
        checksums = {line[66:]: line[:64] for line in bundle.read('SHA256SUMS').decode().splitlines()}
        require(set(bundle.namelist()) == set(checksums) | {'SHA256SUMS'}, 'Unexpected bundled files')
        for name, digest in checksums.items():
            with bundle.open(name) as stream:
                require(hashlib.file_digest(stream, 'sha256').hexdigest() == digest, 'Archive hash mismatch: ' + name)
        modules = {key for key in checksums if key.endswith('.py')}
        original = {path.relative_to(frozen).as_posix()
                    for path in (frozen / 'superagent').rglob('*.py')} | {'unified_agent.py'}
        require(original == modules and all(sha(frozen / key) == checksums[key] for key in modules),
                'Portable runtime differs from the frozen source')
    require(source_identity(frozen) == source, 'Frozen runtime differs from the original arm')
    return len(modules)


def summarize(root):
    floor = root / 'floor'
    paired = root / 'paired'
    archive = root / 'cross-task-review'
    portable = read(floor / 'portable.json')
    audit = read(floor / 'autonomy.json')
    aliases = read(floor / 'anonymous-replay.json')
    grades = read(floor / 'results.json')
    protocol = read(paired / 'protocol.json')
    outcomes = read(paired / 'results.json')
    sealed = read(paired / 'sealed-predictions.json')
    require(audit['source'] == portable['commit'] and not audit['formal_test_accessed'],
            'Floor identity or split mismatch')
    modules = check_handoff(root, portable, protocol['sources']['original'])
    floor_summary = {}
    for key, row in audit['evidence'].items():
        replay = row['replay']
        floor_summary[key] = dict(grades[key], fits=row['fits'], decisions=row['decisions'],
                                  replayed=replay['replayed'], same_when_reordered=replay['unchanged_by_menu_order'],
                                  feedback_swaps=replay['exchanges'], changed_after_swap=replay['changed'],
                                  renamed=aliases[key], opened_in_training=row['training_members'])

    cases = {}
    wall_times = []
    for key, case in outcomes['cases'].items():
        sources, elapsed = {}, {}
        for arm in ARMS:
            run = paired / key / arm / 'run' / 'agent'
            evidence = case[arm]
            result = read(run / 'run_outcome.json')
            manifest = read(run / 'run_manifest.json')
            require(sha(paired / key / arm / 'delivery' / 'predictions.csv') == evidence['prediction'] == sealed[key][arm],
                    'Archived predictions differ from those evaluated')
            require(result['status'] == 'completed' and result['unknown_failed_attempts'] == 0
                    and evidence['fits'] == result['attempted_classifier_fits'] == result['completed_classifier_fits']
                    and evidence['api_calls'] == result['new_api_requests'] == 0, 'Unverified training outcome')
            sources[arm] = evidence['source']
            elapsed[arm] = result['wall_seconds']
            wall_times.append(result['wall_seconds'])
            require(manifest['seed'] == 18643 and manifest['wall_seconds'] == protocol['wall_seconds_per_arm'],
                    'One arm ran with inconsistent settings')
        require(sources == protocol['sources'], 'Paired source mismatch')
        proposals = []
        for version in case['review']['proposals']:
            errors = version['evaluation']
            require(all(x['trial'] > max(version['fit_trials']) for x in errors), 'Future outcome leaked')
            proposals.append({'version': version['version'], 'status': version['status'],
                              'evaluation_trials': [x['trial'] for x in errors],
                              'mean_absolute_error_gain': float(np.mean([x['parent_error'] - x['child_error']
                                                                         for x in errors])) if errors else None})
        cases[key] = {
            'rows': case['rows'], 'metric': case['metric'], 'scores': case['scores'],
            'score_difference': case['delta'], 'fit_counts': {arm: case[arm]['fits'] for arm in ARMS},
            'first_option_changes': case['review']['changed_first_options'], 'proposals': proposals,
            'winners': {arm: case[arm]['winner'] for arm in ARMS},
            'prediction_files_identical': len(set(sealed[key].values())) == 1,
            'training_seconds': elapsed, 'prediction_sha256': sealed[key],
        }
    stages = read(archive / 'results.json')['stages']
    for stage in stages:
        require(not set(stage['fit_sources']).intersection(stage['challenge_sources']), 'Challenge reused for fitting')
        scores = stage['scores'].values()
        gains = [row['gain'] for row in scores]
        changed = sum(row['changed_options'] for row in scores)
        passed = all(x is not None for x in gains) and np.mean(gains) >= .003 and min(gains) >= -.02 and changed > 0
        require(bool(passed) == (stage['status'] == 'accepted'), 'Archived adoption disagrees with frozen rule')
    first_accepted = next((s for s in stages if s['status'] == 'accepted'), None)
    transfer = {}
    if first_accepted is not None:
        # Compare to generation zero on tasks strictly later than the first accepted proposal.
        weights = np.asarray(first_accepted['child_weights'])
        for stage in stages[first_accepted['version']:]:
            for key in stage['challenge_sources']:
                task = episode(paired / key / 'original' / 'run' / 'agent')
                initial = np.zeros_like(weights)
                transfer[key] = challenge(initial, weights, task)
                moves = Counter()
                for prior, menu in task['menus']:
                    old, proposed = top(menu, prior, initial), top(menu, prior, weights)
                    if old != proposed:
                        moves[f'{old[0]}_to_{proposed[0]}'] += 1
                require(sum(moves.values()) == transfer[key]['changed_options'], 'Ranking change count disagrees')
                transfer[key]['changed_action_kinds'] = dict(moves)
    return {
        'submission_commit': portable['commit'], 'floor': floor_summary, 'package': portable,
        'verified_runtime_module_count': modules, 'paired_protocol_sha256': sha(paired / 'protocol.json'),
        'paired_harness_sha256': protocol['study_sha256'], 'paired_runtime_digests': protocol['sources'],
        'paired_cases': cases, 'mean_score_difference': outcomes['mean_delta'],
        'qualifies_for_replication': outcomes['qualifies_for_replication'],
        'actual_settings': {'outer_split_seed': 602911, 'sms_sampling_seed': 602911, 'training_seed': 18643,
                            'time_limit_seconds': protocol['wall_seconds_per_arm'],
                            'maximum_wall_seconds_observed': max(wall_times),
                            'initial_plan_sms_sampling_seed': 70931, 'initial_plan_time_limit_seconds': 900,
                            'observer': protocol['observer']},
        'archive_protocol_sha256': sha(archive / 'protocol.json'),
        'archive_implementation_sha256': read(archive / 'protocol.json')['implementation_sha256'],
        'cross_task_stages': stages,
        'later_transfer_diagnostic': {
            'first_accepted_version': first_accepted['version'] if first_accepted else None,
            'method': 'Post-hoc comparison to generation zero on chronologically later training traces only.',
            'cases': transfer, 'mean_forecast_gain': float(np.mean([x['gain'] for x in transfer.values()])) if transfer else None,
        },
        'classifier_fits': {'released_floor': sum(r['fits'] for r in floor_summary.values()),
                            'paired_research': outcomes['actual_fits'], 'archived_replay': 0},
        'model_api_requests': 0, 'course_cost_rmb': 0, 'formal_test_accessed': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = summarize(args.run_root.resolve())
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
