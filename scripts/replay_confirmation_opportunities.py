"""Find the first change the evidence scheduler would have made in logged ordinary decisions."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from scripts.unified_released_study import ROOT, env, read, require, save, sha
from superagent.ab_run import registered_capabilities
from superagent.autonomous_run import installed_methods
from superagent.open_methods import capability
from superagent.task_opening import TaskOpening


PLAN = ROOT / 'research/confirmation_replay_1127/protocol.json'


def replay(folder):
    decisions = read(folder / 'decisions.json')
    history = {r['trial']: r for r in read(folder / 'history.json')}
    cap_sources = read(folder / 'run_manifest.json')['source_files']
    for name in ('open_methods.py', 'ab_catalog.py', 'ab_run.py'):
        require(cap_sources[name] == sha(ROOT / 'superagent' / name), 'Historical fit-price code differs')
    spec = read(folder / 'task_contract.json')
    profile = read(folder / 'training_profile.json')['statistics']
    folds = {}
    for path in sorted(folder.glob('*_split.json')):
        split = read(path)
        folds[path.name.removesuffix('_split.json')] = (split['fit'], split['held'])
    fit, held = folds['primary']
    with installed_methods():
        static = registered_capabilities(spec, len(set(fit) | set(held)),
            numeric_kernel_width=profile['feature_count'] if spec['kind'] == 'numeric_tabular' else None)
    opportunity = read(folder / 'task_opening.json').get('opportunity', {})
    answer = {'decisions_sha256': sha(folder / 'decisions.json'), 'recorded_steps': len(decisions),
              'visited_steps': 0, 'first_alternative': None}
    registries = {}
    for d in decisions:
        answer['visited_steps'] += 1
        require(d['choice'] == d['alternatives'][0], 'Baseline overrode its ordinary menu')
        trials = d['observed_trials']
        observed = {name: {} for name in folds}
        for i in trials:
            row = history[i]
            if row['status'] == 'completed':
                observed[row['split_id']][row['candidate']] = True
        registry = folder / d['registry']
        require(sha(registry) == d['registry_sha256'], 'Archived candidate registry changed')
        if registry not in registries:
            registries[registry] = {**static, **{name: capability(entry)
                                    for name, entry in read(registry)['entries'].items()}}
        # Reserve only after the logged first observation. A failed worker, if present, is
        # charged its actual fit attempts instead of equating a trial with one fit.
        charged = sum(read(folder / f'trial_{i}.fits.json')['attempted_classifier_fits'] for i in trials)
        record = {'opportunity': opportunity if opportunity and charged >= opportunity['after_fits'] else {}}
        state = SimpleNamespace(mode='opening-feedback-confirm', folds=folds, record=record, save=lambda: None)
        replacement = TaskOpening.protect_confirmation(state, d['alternatives'], registries[registry],
                                                       observed, d['remaining_fits'])
        if replacement:
            answer['first_alternative'] = dict(step=d['step'], observed_trials=trials,
                proposed_action=replacement['action'], proposed_members=replacement['members'],
                **record['protected_confirmations'][0])
            break
    return answer


if __name__ == '__main__':
    import os
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('outputs', 'weights', 'result'):
        p.add_argument('--' + key, required=True, type=lambda s: Path(s).resolve())
    args = p.parse_args()
    for name, value in env(args.weights, args.result / 'unused').items():
        if name.startswith('SUPERAGENT_') and name.endswith('_WEIGHTS'):
            os.environ[name] = value
    paths = {case: replay(args.outputs / study / 'development' / case / 'feedback-local/run/agent')
             for study, cases in read(PLAN)['cohort'].items() for case in cases}
    result = dict(protocol_sha256=sha(PLAN), paths=paths, total_visited_steps=sum(r['visited_steps'] for r in paths.values()),
                  changed_paths=sum(r['first_alternative'] is not None for r in paths.values()),
                  new_fits=0, new_model_requests=0, external_labels_read=False)
    args.result.mkdir(exist_ok=True, parents=True)
    save(args.result / 'results.json', result)
