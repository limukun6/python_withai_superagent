"""Reproduce real decisions and exchange complete observed evidence on common action prefixes."""
import argparse
from pathlib import Path

from scripts import run_b1_ab_acceptance as fixture
from superagent.ab_run import sources
from superagent.ab_scheduler import PRIMARY, CONFIRMATIONS, next_action
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint as vector_hash
from superagent.scheduler_bridge import fingerprint as record_hash
from superagent.splits import sample_groups


SEEDS = (45102, 45103)
POLICIES = ('fixed', 'adaptive')
TASKS = ('headlines', 'coupon', 'bean', 'sms')


def key(action):
    return [action[k] for k in ('action', 'candidate_id', 'split_id')]


def trusted_input(path, entry, frozen):
    if fixture.sha((path / 'train.zip').read_bytes()) != entry['files']['train.zip']:
        raise ValueError('Training archive differs from preregistration')
    run = path / 'run'
    manifest = fixture.read(run / 'run_manifest.json')
    caps = manifest['capabilities']
    if (fixture.digest(manifest['source_files']) != frozen or fixture.digest(caps) != entry['catalog_sha256']
            or manifest['task_archive_sha256'] != entry['files']['train.zip']):
        raise ValueError('Run manifest differs from current production code or original catalogue and inputs')
    task = load_development_task(path / 'train.zip')
    groups = sample_groups(task, 'shared_text')
    trusted = {}
    for split in (PRIMARY, *CONFIRMATIONS):
        record = fixture.read(run / (split + '_split.json'))
        registered = entry['splits'][split]
        payload_hash = fixture.sha((run / (split + '.joblib')).read_bytes())
        ids = record['search_indices']
        labels = task.y_train[ids].tolist()
        checks = dict(evaluation_payload_sha256=payload_hash, row_ids_sha256=vector_hash(ids), label_sha256=vector_hash(labels))
        if any(record[k] != v or registered[k] != v for k, v in checks.items()):
            raise ValueError('Archived internal rows, labels or payload differ from the registered split')
        trusted[split] = dict(row_ids=ids, labels=labels, groups=groups[ids].tolist(), evaluation_payload_sha256=payload_hash)
    return caps, trusted


def replay(path, entry, frozen, policy):
    run = path / 'run'
    caps, trusted = trusted_input(path, entry, frozen)
    actions = fixture.read(run / 'actions.json')
    trace = fixture.read(run / 'research_policy.json')
    if trace['policy'] != policy or len(trace['decisions']) != len(actions):
        raise ValueError('Archived policy trace does not cover all actions')
    for index, (actual, mark) in enumerate(zip(actions, trace['decisions'], strict=True), 1):
        state = fixture.read(run / f'state_{index}.json')
        if mark != dict(trial=index, state_sha256=record_hash(state), action_sha256=record_hash(actual)):
            raise ValueError('Archived state or action differs from execution fingerprints')
        reproduced = next_action(state, capabilities=caps, trusted_partitions=trusted, allocation=policy)
        if reproduced != actual:
            raise ValueError(f'Action replay differs at {path.name}, step {index}')
    return dict(run=run, actions=actions, caps=caps, trusted=trusted)


def divergence(left, right):
    for i, (a, b) in enumerate(zip(left['actions'], right['actions']), 1):
        if key(a) != key(b):
            return dict(step=i, left=key(a), right=key(b))
    if len(left['actions']) != len(right['actions']):
        raise ValueError('One complete trace ended before another without a different stop decision')
    return dict(step=None)


def substitute(left, right):
    change = divergence(left, right)
    if change['step'] is None:
        return dict(**change, status='no_observed_difference')
    step = change['step']
    a, b = (fixture.read(q['run'] / f'state_{step}.json') for q in (left, right))
    keys = ('task', 'available_candidates', 'registered_confirmation_splits', 'ab_schema_version')
    controls = {k: a[k] == b[k] for k in keys}
    controls['capabilities'] = left['caps'] == right['caps']
    controls['budget_except_elapsed'] = ({k: v for k,v in a['budget'].items() if k != 'elapsed_seconds'} ==
                                        {k: v for k,v in b['budget'].items() if k != 'elapsed_seconds'})
    if not all(controls.values()):
        return dict(**change, status='incomparable_controls', controls=controls)
    reports = []
    for base, donor, original, replacement in ((a,b,left,right),(b,a,right,left)):
        # Change only the two feedback fields and their matching locally verified subject rows.
        mixed = dict(base, history=donor['history'], prediction_evidence=donor['prediction_evidence'])
        counter = next_action(mixed, capabilities=original['caps'], trusted_partitions=replacement['trusted'])
        expected = replacement['actions'][step - 1]
        if counter != expected:
            raise ValueError('Swapped evidence failed to reproduce the donor decision with all six fields')
        reports.append(dict(state_sha256=record_hash(base), donor_evidence_sha256=record_hash(donor['prediction_evidence']),
            retained_elapsed_seconds=base['budget']['elapsed_seconds'], remaining_fits=base['budget']['remaining_fit_allowance'],
            original=original['actions'][step-1], replayed_with_other_observations=counter))
    return dict(**change, status='bidirectional_change', controls=controls, comparisons=reports)


def study(args):
    dest = args.output
    dest.mkdir(parents=True, exist_ok=False)
    fixture.offline(dest)
    frozen = fixture.digest(sources())
    roots = {k: getattr(args,k) for k in TASKS}
    fixture.write(dest / 'protocol.json', dict(batch='B1-09', source_sha256=frozen,
        driver_sha256=fixture.sha(Path(__file__).read_bytes()), plan_sha256=fixture.sha((fixture.ROOT / 'docs/batches/B1_09_replay_plan.md').read_bytes()),
        input_protocols={k: fixture.sha((r / 'protocol.json').read_bytes()) for k,r in roots.items()}, fits_allowed=0, api_allowed=0))
    results = {}
    try:
        for name, root in roots.items():
            if fixture.read(root / 'completion.json')['status'] != 'passed':
                raise ValueError('Incomplete recorded path')
            protocol = fixture.read(root / 'protocol.json')
            arms = {}
            for seed in SEEDS:
                for policy in POLICIES:
                    label = f'{seed}_{policy}'
                    arms[seed,policy] = replay(root / label, protocol['tasks'][label], frozen, policy)
            results[name] = dict(reproduced_actions={f'{s}_{p}':len(v['actions']) for (s,p),v in arms.items()},
                policy_first_difference={str(s):divergence(arms[s,'fixed'], arms[s,'adaptive']) for s in SEEDS},
                adaptive_first_cross_split_difference=substitute(arms[SEEDS[0],'adaptive'], arms[SEEDS[1],'adaptive']))
            fixture.write(dest / 'results.json', results)
        fixture.write(dest / 'completion.json', dict(status='passed', tasks=len(results), paths=4*len(results),
            reproduced_actions=sum(sum(t['reproduced_actions'].values()) for t in results.values()),
            fits=0, new_model_predictions=0, outer_scores_read=0, api_requests=0))
    except Exception as error:
        fixture.write(dest / 'completion.json', dict(status='failed', error=str(error), completed_tasks=list(results),
            fits=0, new_model_predictions=0, outer_scores_read=0, api_requests=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*TASKS, 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    study(parser.parse_args())
