"""Run matched training-budget and active-evidence comparisons without opening evaluation labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from scripts.run_planner_self_review import source_identity, packet
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from superagent.experiment_policy import identity


STUDY = ROOT / 'research/rsi_budget_1125/protocol.json'
PREVIOUS = ROOT / 'research/rsi_planner_1125/completion.json'
WALL = 600


def prepare(root, prior):
    plan, previous = read(STUDY), read(PREVIOUS)
    old = prior / 'paired'
    old_plan = read(old / 'protocol.json')
    require(sha(old / 'protocol.json') == previous['paired_protocol_sha256'], 'Prior packets or settings changed')
    root.mkdir()
    (root / 'evaluation').mkdir()
    cases = {}
    for case in plan['legacy_cases']:
        folder = root / case
        folder.mkdir()
        spec = old_plan['cases'][case]
        for source, name, digest in [
            (old / case / 'p.zip', folder / 'p.zip', spec['train_sha256']),
            (old / case / 'q.zip', folder / 'q.zip', spec['input_sha256']),
            (old / 'evaluation' / (case + '.npy'), root / 'evaluation' / (case + '.npy'), spec['truth_sha256']),
        ]:
            require(sha(source) == digest, 'Archived packet changed')
            shutil.copyfile(source, name)
        reference = read(old / case / 'review' / 'sealed.json')
        require(reference['source'] == previous['paired_runtime_digests']['review'] and
                reference['prediction'] == previous['paired_cases'][case]['prediction_sha256']['review'],
                'Prior passive control identity mismatch')
        cases[case] = spec
    transfer = plan['independent_check']['id']
    folder = root / transfer
    folder.mkdir()
    x, y = make_classification(n_samples=1000, n_features=16, n_informative=8, n_redundant=3,
                               n_classes=3, n_clusters_per_class=2, weights=[.57, .29, .14],
                               class_sep=1.1, flip_y=.09, random_state=941227)
    require(len(np.unique(x, axis=0)) == len(y), 'Unexpected duplicate transfer rows')
    fit, held = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=730357)
    packet(folder / 'p.zip', 'tabular', 'balanced_accuracy', x[fit], y[fit])
    packet(folder / 'q.zip', 'tabular', 'balanced_accuracy', x[held])
    truth = root / 'evaluation' / (transfer + '.npy')
    np.save(truth, y[held], allow_pickle=False)
    cases[transfer] = dict(modality='tabular', metric='balanced_accuracy', fit_rows=len(fit), held_rows=len(held),
                           train_sha256=sha(folder / 'p.zip'), input_sha256=sha(folder / 'q.zip'), truth_sha256=sha(truth))
    save(root / 'protocol.json', {
        'preregistration': sha(STUDY), 'runtime': source_identity(ROOT), 'harness': sha(Path(__file__)),
        'parent_baselines': sha(PREVIOUS), 'cases': cases, 'cells': plan['cells'], 'wall_seconds': WALL,
    })


def action_key(action):
    return [action['action'], action['members'], action['split']]


def audit(path, cell):
    run = path / 'run' / 'agent'
    outcome, final = read(run / 'run_outcome.json'), read(run / 'final_selection.json')
    manifest, history = read(run / 'run_manifest.json'), read(run / 'history.json')
    actions = read(run / 'decisions.json')
    fits = read(run / 'classifier_fits.json')
    policy = read(run / 'policy_review.json')
    limit = read(STUDY)['cells'][cell]
    mode = 'active' if cell.startswith('active') else True
    require(manifest['fit_limit'] == limit and manifest['self_review'] == mode and
            manifest['seed'] == 18643 and manifest['wall_seconds'] == WALL, 'Unexpected configuration')
    require(outcome['status'] == 'completed' and outcome['formal_test_accessed'] is False and
            outcome['new_api_requests'] == 0 and outcome['unresolved_api_requests'] == 0 and final['refit'],
            'Incomplete training, external split access or unintended model API call')
    pairs = {f"trial_{r['trial']}.joblib": r for r in history}
    physical = len((path / 'fits.jsonl').read_text().splitlines())
    counts = dict(explore=0, confirm=0, refit=0)
    for worker in fits['workers']:
        artifact = worker['artifact']
        if artifact.startswith('refit_component_'):
            category = 'refit'
        else:
            trial = pairs[artifact]
            category = 'explore' if trial['split_id'] == 'primary' else 'confirm'
        counts[category] += worker['attempted_classifier_fits']
    require(0 < sum(counts.values()) == physical == fits['attempted_classifier_fits'] ==
            outcome['completed_classifier_fits'] <= limit and not outcome['unknown_failed_attempts'],
            'Inconsistent physical fitting records')
    changed, adopted_changes = [], 0
    sequence = []
    budget_at_step = limit
    for i, row in enumerate(actions):
        require(row['observed_trials'] == [v['trial'] for v in history[:i]], 'Feedback from the future')
        chosen = row['choice']
        key = action_key(chosen)
        require(row['remaining_fits'] == budget_at_step and row['source'] == 'local evidence policy'
                and key == action_key(row['alternatives'][0]), 'Budget or autonomous action mismatch')
        sequence.append(key)
        old = row.get('first_before_information', chosen)
        if row.get('information_changed_first'):
            info = chosen['evidence']['information_value']
            require(chosen['action'] == 'explore' and info['credit'] > 0 and
                    info['required_fits_including_followup_and_refit'] <= budget_at_step,
                    'A changed action lacks usable information or a feasible follow-up')
            changed.append(dict(step=row['step'], before=action_key(old), after=key, remaining=budget_at_step,
                                credit=info['credit'], followup=info['followup_action']))
        if row['policy_generation'] and action_key(row['initial_policy_first']) != action_key(old):
            adopted_changes += 1
        if i < len(history):
            match = next(r for r in fits['workers'] if r['artifact'] == f"trial_{history[i]['trial']}.joblib")
            budget_at_step -= match['attempted_classifier_fits']
    require(sequence[-1][0] == 'stop' and sequence[-1][1] == outcome['members'], 'Untraced final model')
    proposals = []
    for item in policy['proposals']:
        trials = [obs['trial'] for obs in item['evaluation']]
        require(len(set(trials)) == len(trials) and all(t > max(item['fit_trials']) for t in trials),
                'A rule was evaluated on its own proposal data')
        proposals.append(dict(version=item['version'], status=item['status'], tests=len(trials),
                              gain=item.get('mean_absolute_error_gain')))
    pred = path / 'delivery' / 'predictions.csv'
    digest = sha(pred)
    require(sha(path / 'replay' / 'predictions.csv') == digest and
            sha(run / final['model_file']) == final['model_sha256'], 'Fresh-process prediction or model mismatch')
    split = read(run / 'primary_split.json')
    return dict(metric=final['primary_metric'], model=final['model_sha256'], prediction=digest, fits=physical,
                counts=counts, actions=sequence, winner=outcome['members'], seconds=outcome['wall_seconds'],
                primary_split=identity({k:split[k] for k in ('fit','held','row_sha256','label_sha256')}),
                proposals=proposals, information_changed=changed, adopted_rule_changes=adopted_changes,
                api_calls=outcome['new_api_requests'])


def run_one(root, case, cell, weights, config):
    folder = root / case
    dest = folder / cell
    require(sha(folder / 'p.zip') == config['cases'][case]['train_sha256'] and
            sha(folder / 'q.zip') == config['cases'][case]['input_sha256'], 'Input packet changed')
    if (dest / 'sealed.json').exists():
        require(audit(dest, cell) == read(dest / 'sealed.json'), 'Previously finished arm changed')
        return
    dest.mkdir()
    environ = env(weights, dest / 'fits.jsonl')
    environ.pop('AB_B1_PER_WORKER_FIT_LIMIT', None)
    for key in ('SUPERAGENT_PLANNER_KEY_FILE', 'SUPERAGENT_PLANNER_LEDGER', 'SUPERAGENT_TASK_BUDGET_ID'):
        environ.pop(key, None)
    mode = '--active-self-review' if cell.startswith('active') else '--self-review'
    jobs = [
        ('solve', ['solve', folder / 'p.zip', '--prediction-input', folder / 'q.zip', '--run-dir', dest / 'run',
                   '--output', dest / 'delivery' / 'predictions.csv', '--fit-limit', config['cells'][cell],
                   '--time-limit', WALL, mode]),
        ('predict', ['predict', folder / 'q.zip', '--run-dir', dest / 'run',
                     '--output', dest / 'replay' / 'predictions.csv']),
    ]
    for label, args in jobs:
        with (dest / (label + '.log')).open('x') as log:
            subprocess.run([sys.executable, '-m', 'unified_agent', *map(str, args)], cwd=ROOT, env=environ,
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=WALL+90, check=True)
    record = audit(dest, cell)
    save(dest / 'sealed.json', record)
    print(case, cell, record['counts'], len(record['information_changed']), 'information-driven moves', flush=True)


def run(root, prior, weights):
    config = read(root / 'protocol.json')
    plan = read(STUDY)
    require(config['runtime'] == source_identity(ROOT) and config['preregistration'] == sha(STUDY) and
            config['harness'] == sha(Path(__file__)) and config['parent_baselines'] == sha(PREVIOUS), 'Frozen code or plan changed')
    sentinel = plan['legacy_cases'][0]
    run_one(root, sentinel, 'passive9', weights, config)
    original = audit(prior / 'paired' / sentinel / 'review', 'passive9')
    repeat = read(root / sentinel / 'passive9' / 'sealed.json')
    require(all(original[k] == repeat[k] for k in ('prediction','actions','counts','primary_split','winner','proposals')),
            'The passive sentinel did not reproduce the previous policy')
    jobs = [(case, cell) for case in plan['legacy_cases'] for cell in ('active9','passive16','active16')]
    jobs.extend((plan['independent_check']['id'], cell) for cell in plan['independent_check']['arms'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda args: run_one(root, *args, weights, config), jobs))
    require(config['runtime'] == source_identity(ROOT), 'Runtime changed during training')
    records = {case + '/' + cell: read(root / case / cell / 'sealed.json')['prediction']
               for case, cell in [(sentinel, 'passive9'), *jobs]}
    save(root / 'sealed-predictions.json', records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('root', type=Path)
    parser.add_argument('prior', type=Path)
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.root.resolve(), args.prior.resolve())
    else:
        run(args.root.resolve(), args.prior.resolve(), args.weights.resolve())
