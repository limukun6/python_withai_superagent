"""Audit the candidate scores and paired errors already observed on the four B1 development tasks."""
import argparse
import math
from pathlib import Path

from scripts import run_b1_ab_acceptance as fixture
from scripts.probe_task_headroom import observed_coverage, paired, rankings
from scripts.replay_b1_observed_actions import POLICIES, SEEDS, TASKS, trusted_input
from superagent.ab_run import sources
from superagent.ab_scheduler import _observations, CONFIRMATIONS, PRIMARY
from superagent.ab_comparison import MIN_GAIN, MIN_PAIRED_WINS
from superagent.metrics import MetricSpec
from superagent.scheduler_bridge import fingerprint as state_digest


SPLITS = (PRIMARY, *CONFIRMATIONS)


def archive_for(task):
    root = fixture.ROOT / 'research'
    return root / 'b1_07_allocation' if task == 'headlines' else root / 'b1_08_allocation' / task


def checked_arm(path, entry, model_hash, source_hash):
    caps, trusted = trusted_input(path, entry, source_hash)
    run = path / 'run'
    task_spec = fixture.read(run / 'task_spec.json')
    metric = MetricSpec.from_payload(task_spec)
    if metric.name not in ('accuracy', 'macro_f1') or metric.name != task_spec['metric']:
        raise ValueError('Expected a registered B1 primary metric')
    history = fixture.read(run / 'history.json')
    evidence = fixture.read(run / 'prediction_evidence.json')
    actions = fixture.read(run / 'actions.json')
    state = fixture.read(run / f'state_{len(actions)}.json')
    trace = fixture.read(run / 'research_policy.json')['decisions'][-1]
    if (actions[-1]['action'] != 'stop' or len(actions) != 1 + len(history)
            or trace['state_sha256'] != state_digest(state) or trace['action_sha256'] != state_digest(actions[-1])
            or state['history'] != history or state['prediction_evidence'] != evidence):
        raise ValueError('Observations differ from the state consumed by the stop decision')
    selected = fixture.read(run / 'final_selection.json')
    if (selected['model_sha256'] != model_hash or fixture.sha((run / 'selected_model.joblib').read_bytes()) != model_hash
            or selected['single_candidate'] != selected['family'] or selected['primary_metric'] != metric.name):
        raise ValueError('Selection differs from the previously frozen model')
    _, measurements, predictions = _observations(history, evidence, metric, CONFIRMATIONS, trusted)
    if len(measurements) != len(history) or len(predictions) != len(history):
        raise ValueError('Expected all B1 trials to have completed row evidence')
    by_split = {k: {} for k in SPLITS}
    for (name, split), item in measurements.items():
        trial = item['trial']
        fit = fixture.read(run / f'trial_{trial}.fits.json')
        if (name not in caps or not (caps[name]['params'] == item['params'] == item['resolved_configuration']['params'])
                or item['resolved_configuration']['id'] != name or fit['candidate'] != name or fit['status'] != 'completed'
                or fit['completed_classifier_fits'] != item['classifier_fits'] or fit['attempted_classifier_fits'] != fit['completed_classifier_fits']):
            raise ValueError('Worker configuration or counted fits differ')
        pred = predictions[trial]
        values = metric.evaluate(trusted[split]['labels'], pred)
        if not math.isclose(values['accuracy'], item['search_accuracy'], rel_tol=0, abs_tol=1e-10):
            raise ValueError('Accuracy disagrees with the saved predictions')
        by_split[split][name] = dict(predictions=pred, search_score=values['score'], search_accuracy=values['accuracy'],
                                     configuration_sha256=fixture.digest(item['resolved_configuration']))
    return dict(metric=metric, declared=list(caps), trusted=trusted, selected=selected['family'],
                by_split=by_split, model_sha256=model_hash, trials=len(measurements))


def merge_checked(fixed, adaptive, split):
    if (fixed['trusted'][split] != adaptive['trusted'][split] or fixed['metric'] != adaptive['metric']
            or fixed['declared'] != adaptive['declared']):
        raise ValueError('Policy arms disagree on rows, grouping, metric or catalogue')
    rows = {}
    for arm in (fixed, adaptive):
        for name, item in arm['by_split'][split].items():
            if name in rows and rows[name] != item:
                raise ValueError('Repeated candidate/configuration on the same partition differs')
            rows[name] = item
    return rows


def describe(fixed, adaptive, split):
    rows = merge_checked(fixed, adaptive, split)
    source, metric = fixed['trusted'][split], fixed['metric']
    y, groups = source['labels'], source['groups']
    ranks = rankings(rows)
    fs, ds = fixed['selected'], adaptive['selected']
    pair_result = dict(status='missing_candidate', missing=sorted({fs, ds}.difference(rows)))
    if fs in rows and ds in rows:
        pair_result = dict(status='observed', **paired(metric, y, rows[fs]['predictions'], rows[ds]['predictions'], groups))
    result = dict(rows=len(y), groups=len(set(groups)), catalogue_size=len(fixed['declared']),
                  tried={key: sorted(arm['by_split'][split]) for key, arm in zip(POLICIES, (fixed,adaptive), strict=True)},
                  observed=len(rows), unmeasured=sorted(set(fixed['declared']).difference(rows)), ranks=ranks,
                  scores={n: {k: v[k] for k in ('search_score','search_accuracy','configuration_sha256')} for n,v in rows.items()},
                  selected_pair=pair_result)
    if split == PRIMARY:
        result['label_aware_coverage'] = observed_coverage(metric, y, {n:v['predictions'] for n,v in rows.items()}, ds)
    return result


def supported_alternatives(splits, selected):
    scores = {k: part['scores'] for k,part in splits.items()}
    answers = {}
    for name in scores[PRIMARY]:
        if name == selected:
            continue
        gaps = {s: table[name]['search_score'] - table[selected]['search_score']
                for s, table in scores.items() if name in table and selected in table}
        if sum(x > 0 for x in gaps.values()) >= MIN_PAIRED_WINS and sum(gaps.values()) / len(gaps) >= MIN_GAIN:
            answers[name] = gaps
    return answers


def measure(task, root, frozen):
    canonical = archive_for(task)
    for filename in ('protocol.json','all_models_frozen.json','completion.json'):
        if (root / filename).read_bytes() != (canonical / filename).read_bytes():
            raise ValueError(f'{task}: {filename} differs from the original archived batch')
    if fixture.read(canonical / 'completion.json')['status'] != 'passed':
        raise ValueError('Earlier study was incomplete')
    entries = fixture.read(canonical / 'protocol.json')['tasks']
    hashes = fixture.read(canonical / 'all_models_frozen.json')
    seeds = {}
    for seed in SEEDS:
        arms = [checked_arm(root / f'{seed}_{mode}', entries[f'{seed}_{mode}'], hashes[f'{seed}_{mode}'], frozen)
                for mode in POLICIES]
        parts = {s: describe(*arms, s) for s in SPLITS}
        seeds[str(seed)] = dict(metric=arms[0]['metric'].record(),
            paths={mode: {k: arm[k] for k in ('selected', 'model_sha256', 'trials')} for mode,arm in zip(POLICIES,arms,strict=True)},
            splits=parts, supported_alternatives=supported_alternatives(parts, arms[1]['selected']))
    return dict(seeds=seeds, repeated_supported_alternatives=sorted(set.intersection(*(set(s['supported_alternatives']) for s in seeds.values()))))


def study(args):
    out = args.output
    out.mkdir(exist_ok=False, parents=True)
    fixture.offline(out)
    frozen = fixture.digest(sources())
    fixture.write(out / 'protocol.json', dict(batch='B1-10', production_sha256=frozen,
                  driver_sha256=fixture.sha(Path(__file__).read_bytes()),
                  headroom_sha256=fixture.sha((fixture.ROOT/'scripts/probe_task_headroom.py').read_bytes()),
                  plan_sha256=fixture.sha((fixture.ROOT/'docs/batches/B1_10_headroom_plan.md').read_bytes()),
                  tasks={n:fixture.sha((archive_for(n)/'protocol.json').read_bytes()) for n in TASKS},
                  seeds=list(SEEDS), policies=list(POLICIES), splits=list(SPLITS), fits_allowed=0, api_allowed=0))
    results = {}
    try:
        for task in TASKS:
            results[task] = measure(task, getattr(args,task), frozen)
            fixture.write(out / 'results.json', results)
        observations = sum(p['trials'] for v in results.values() for s in v['seeds'].values() for p in s['paths'].values())
        distinct = sum(s['observed'] for v in results.values() for t in v['seeds'].values() for s in t['splits'].values())
        fixture.write(out / 'completion.json', dict(status='passed',tasks=len(results), paths=16,
                      verified_trial_observations=observations, distinct_candidate_partition_observations=distinct,
                      fits=0, new_model_predictions=0, outer_scores_read=0, model_api_requests=0))
    except Exception as error:
        fixture.write(out / 'completion.json', dict(status='failed', reason=str(error), completed_tasks=list(results),
                      fits=0, new_model_predictions=0, outer_scores_read=0, model_api_requests=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*TASKS, 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    study(parser.parse_args())
