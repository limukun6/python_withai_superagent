"""Test descendant planning rules through the ordinary solver before opening development labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts import planner_lineage_data, replay_planner_review_archive
from scripts.planner_lineage_data import prepare_packets
from scripts.replay_planner_review_archive import episode, propose
from scripts.run_planner_self_review import source_identity
from scripts.run_review_budget_study import action_key
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from superagent.experiment_policy import identity
from superagent.exploration_rule import forecast, snapshot
from superagent.metrics import MetricSpec


PLAN = ROOT/'research/rsi_lineage_1125/protocol.json'
LIMIT = 16
WALL = 600


def provenance():
    modules = [Path(__file__), Path(planner_lineage_data.__file__), Path(replay_planner_review_archive.__file__),
               ROOT/'scripts/run_planner_self_review.py', ROOT/'scripts/run_review_budget_study.py',
               ROOT/'scripts/unified_released_study.py', ROOT/'scripts/ab_acceptance_guard/sitecustomize.py']
    return dict(plan=sha(PLAN), runtime=source_identity(ROOT),
                drivers={m.name: sha(m) for m in modules})


def original_traces(prior):
    paths = {k: prior/'floor'/k/'run/agent' for k in ('table', 'titles')}
    paths.update({k: prior/'paired'/k/'original/run/agent' for k in ('n17', 'n29', 'n43', 'n61')})
    return paths


def trace_digest(path):
    return identity([sha(path/'decisions.json'), sha(path/'history.json')])


def prepare(root, prior, downloads, cornell):
    root.mkdir(parents=True, exist_ok=False)
    (root/'rules').mkdir()
    packets = prepare_packets(root, read(PLAN), downloads, cornell, prior)
    traces = {k: trace_digest(v) for k, v in original_traces(prior).items()}
    save(root/'protocol.json', dict(source=provenance(), cases=packets, traces=traces,
        existing_proposal=sha(prior/'cross-task-review/proposal_1.json'),
        existing_verdict=sha(prior/'cross-task-review/evaluation_1.json'),
        source_data={name:sha(downloads/name) for name in ('sms.zip','fashion-images.gz','fashion-labels.gz')}
                    | {'cornell': sha(cornell)}, limit=LIMIT, wall=WALL, maximum_physical_fits=384))


def check_files(root, key, spec):
    require(spec['train'] == sha(root/key/'p.zip') and spec['features'] == sha(root/key/'q.zip'),
            'An anonymous input packet changed')


def freeze(root, weights):
    rule = snapshot(weights)
    digest = identity(rule)
    save(root/'rules'/(digest+'.json'), rule)
    return digest


def run_one(root, key, rule, weights, config):
    spec = config['cases'][key]
    check_files(root, key, spec)
    work = root/key/rule
    path = root/'rules'/(rule+'.json')
    require(identity(read(path)) == rule, 'Frozen coefficients no longer match identifier')
    run = work/'run'
    agent = run/'agent'
    delivery = work/'delivery/predictions.csv'
    replay = work/'replay/predictions.csv'
    if not (work/'sealed.json').is_file():
        work.mkdir()
        settings = env(weights, work/'fits.jsonl')
        settings.pop('AB_B1_PER_WORKER_FIT_LIMIT', None)
        for k in ('SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_TASK_BUDGET_ID'):
            settings.pop(k, None)
        for command, args in [
            ('solve', [root/key/'p.zip', '--prediction-input', root/key/'q.zip', '--run-dir', run, '--output',
                       delivery, '--fit-limit', LIMIT, '--time-limit', WALL, '--protected-self-review',
                       '--exploration-rule', path]),
            ('predict', [root/key/'q.zip', '--run-dir', run, '--output', replay]),
        ]:
            with (work/(command+'.log')).open('x') as log:
                subprocess.run([sys.executable, '-m', 'unified_agent', command, *map(str, args)], cwd=ROOT,
                               env=settings, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               timeout=WALL+100, check=True)
    hist, decisions = read(agent/'history.json'), read(agent/'decisions.json')
    final, out = read(agent/'final_selection.json'), read(agent/'run_outcome.json')
    manifest, stop = read(agent/'run_manifest.json'), read(agent/'stop_review.json')
    count = len((work/'fits.jsonl').read_text().splitlines())
    delivery_manifest = read(delivery.parent/'prediction_manifest.json')
    require(identity(read(run/'exploration_rule.json')) == identity(read(agent/'exploration_rule.json')) ==
            read(run/'training_protocol.json')['exploration_rule_digest'] == manifest['exploration_rule_digest'] == rule,
            'The solver consumed a different rule')
    require(out['status'] == 'completed' and out['training_only'] and not out['formal_test_accessed']
            and out['unresolved_api_requests'] == out['unknown_failed_attempts'] == out['new_api_requests'] == 0
            and out['complete_run_course_cost_rmb'] == 0 and manifest['fit_limit'] == LIMIT
            and manifest['self_review'] == 'protected' and manifest['controller'] == 'local evidence policy'
            and 0 < count == out['attempted_classifier_fits'] == out['completed_classifier_fits'] <= LIMIT
            and final['refit'] and final['primary_metric'] == spec['metric'] and final['training_rows'] == spec['train_rows']
            and final['model_sha256'] == sha(agent/final['model_file']) == stop['delivered_model']
            and stop['fits_at_stop'] + stop['additional_fits'] == count, 'Unverified fit accounting or model')
    require(sha(delivery) == sha(replay) == delivery_manifest['prediction_sha256']
            and not delivery_manifest['test_labels_read'] and delivery_manifest['prediction_count'] == spec['held_rows']
            and delivery_manifest['delivery_status'] == 'completed', 'Prediction replay mismatch')
    require(len(decisions) == len(hist)+1 and decisions[-1]['choice']['action'] == 'stop' and
            all(d['observed_trials'] == [h['trial'] for h in hist[:i]] and d['source'] == 'local evidence policy'
                and d['inherited_rule']['digest'] == rule for i, d in enumerate(decisions)), 'Invalid chronology')
    gains = [row['delta'] for row in stop['comparisons']]
    passed = len(gains) == 2 and min(gains) > 0 and np.mean(gains) >= .003
    require((stop['status'] == 'accepted') == passed and stop['status'] in
            ('accepted', 'retained', 'insufficient_room_or_evidence'), 'Invalid protected stop')
    actions = [action_key(d['choice']) for d in decisions]
    if stop['comparisons']:
        actions.extend([['challenge', [stop['candidate']], r['fold']] for r in stop['comparisons']])
    actions.append(['deliver', out['members'], None])
    record = dict(rule=rule, case=key, actions=actions, fits=count, api=out['new_api_requests'],
                  prediction=sha(delivery), model=final['model_sha256'], winner=out['members'],
                  primary_split=sha(agent/'primary_split.json'), training_trace=trace_digest(agent),
                  inherited_top_changes=sum(d['inherited_rule']['changed_first'] for d in decisions),
                  protected_verdict=stop['status'])
    save(work/'sealed.json', record)
    print(key, rule[:8], count, 'fits, replay verified', flush=True)
    return record


def run_panel(root, names, roles, weights, config, stage):
    require(config['source'] == provenance(), 'Implementation changed after preregistration')
    cells = [(key, rule) for key in names for rule in dict.fromkeys(roles.values())]
    records = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {cell: pool.submit(run_one, root, *cell, weights, config) for cell in cells}
        for cell, result in futures.items():
            records[cell] = result.result()
    require(all(len({records[k, r]['primary_split'] for r in roles.values()}) == 1 for k in names),
            'Arms saw different primary folds')
    save(root/(stage+'_sealed.json'), {k: {r: records[k, r]['prediction'] for r in roles.values()} for k in names})
    grades = {}
    for key in names:
        spec = config['cases'][key]
        truth = root/'evaluation'/(key+'.npy')
        require(spec['truth'] == sha(truth), 'Development benchmark labels changed')
        y = np.load(truth, allow_pickle=False)
        metric = MetricSpec(spec['metric'], tuple(np.unique(y).tolist()))
        predictions, grades[key] = {}, {}
        for role, rule in roles.items():
            result = records[key, rule]
            path = root/key/rule/'delivery/predictions.csv'
            require(sha(path) == result['prediction'], 'Prediction changed after sealing')
            with path.open(encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            require([r['row_id'] for r in rows] == list(map(str, range(len(y)))), 'Row order mismatch')
            pred = np.array([r['prediction'] for r in rows], dtype=y.dtype)
            predictions[role] = pred
            grades[key][role] = dict(rule=rule, score=metric.evaluate(y, pred)['score'], fits=result['fits'],
                correct=int(np.sum(pred == y)), rows=len(y), metric=spec['metric'], prediction=result['prediction'],
                winner=result['winner'], inherited_top_changes=result['inherited_top_changes'])
        ref = next(iter(roles))
        for role, rule in roles.items():
            grades[key][role].update(changed_predictions=int(np.sum(predictions[ref] != predictions[role])),
                changed_path=records[key, roles[ref]]['actions'] != records[key, rule]['actions'])
    return grades


def verdict(grades):
    deltas = [r['child']['score'] - r['parent']['score'] for r in grades.values()]
    parent_fits = sum(r['parent']['fits'] for r in grades.values())
    child_fits = sum(r['child']['fits'] for r in grades.values())
    changed = any(r['child']['changed_path'] for r in grades.values())
    quality = np.mean(deltas) >= .005 and min(deltas) >= -.01
    efficiency = min(deltas) >= -1e-12 and child_fits <= .85 * parent_fits
    return dict(promote=bool(changed and (quality or efficiency)), quality_gate=bool(quality),
                cost_gate=bool(efficiency), changed_path=changed, mean_delta=float(np.mean(deltas)),
                worst_delta=min(deltas), parent_fits=parent_fits, child_fits=child_fits)


def study(root, prior, weights):
    config, plan = read(root/'protocol.json'), read(PLAN)
    paths = original_traces(prior)
    digests = {k: trace_digest(p) for k, p in paths.items()}
    require(config['source'] == provenance() and digests == config['traces'], 'Frozen source or archive changed')
    memory = {k: episode(p) for k, p in paths.items()}
    seed_path = prior/'cross-task-review/proposal_1.json'
    old_verdict = prior/'cross-task-review/evaluation_1.json'
    parent, step = np.zeros(3), .6
    seed = propose(parent, {'table': memory['table']}, step)
    require(sha(seed_path) == config['existing_proposal'] and sha(old_verdict) == config['existing_verdict']
            and read(old_verdict)['status'] == 'accepted' and np.allclose(seed, read(seed_path)['child_weights'],
                                                                       rtol=0, atol=1e-14), 'Old proposal not reproduced')
    seed = np.array(read(seed_path)['child_weights'])
    ancestor = current = freeze(root, parent)
    ancestors = [current]
    waves = []
    for i, cohort in enumerate(plan['waves'], 1):
        names = [row['id'] for row in cohort]
        stage = 'wave_'+str(i)
        require(not set(names).intersection(memory), 'Proposal saw its challenge cohort')
        candidate = seed if i == 1 else propose(parent, memory, step)
        child = freeze(root, candidate)
        roles = dict(parent=current, child=child)
        sources = {'table': digests['table']} if i == 1 else dict(digests)
        save(root/(stage+'_proposal.json'), dict(roles=roles, weights=candidate.tolist(), step=step,
                                                fit_sources=sources, challenge=names))
        grades = run_panel(root, names, roles, weights, config, stage)
        decision = verdict(grades)
        forecast_changes = {}
        for k in names:
            path = root/k/current/'run/agent'
            trace = episode(path)
            rows = trace['observations']
            def mae(w):
                return float(np.mean([abs(forecast(w, n, evidence)-score) for n, evidence, score, _ in rows]))
            forecast_changes[k] = dict(observations=len(rows), parent_mae=mae(parent) if rows else None,
                                       child_mae=mae(candidate) if rows else None)
            memory[k], digests[k] = trace, trace_digest(path)
        if decision['promote']:
            current, parent = child, candidate
            ancestors.append(child)
            step = min(1., step*1.25)
        else:
            step = max(.15, step*.5)
        wave = dict(roles=roles, cases=grades, verdict=decision, next_parent=current, parent_trace_mae=forecast_changes)
        save(root/(stage+'_result.json'), wave)
        waves.append(wave)
        print(stage, decision, flush=True)
    roles = dict(root=ancestor, selected=current)
    if len(ancestors) > 1:
        roles['immediate_parent'] = ancestors[-2]
    roles['one_shot'] = freeze(root, propose(np.zeros(3), memory, .6))
    if len(ancestors) == 1:
        roles['rejected_diagnostic'] = child
    names = [r['id'] for r in plan['final_panel']]
    require(not set(names).intersection(memory), 'Final cases leaked into memory')
    save(root/'final_freeze.json', dict(roles=roles, accepted_line=ancestors, fit_sources=digests, challenge=names))
    final = run_panel(root, names, roles, weights, config, 'final')
    receipts = [read(f) for f in sorted(root.glob('*/*/sealed.json'))]
    fits = sum(r['fits'] for r in receipts)
    physical = sum(len(f.read_text().splitlines()) for f in root.glob('*/*/fits.jsonl'))
    require(0 < fits == physical <= config['maximum_physical_fits'] and all(r['api'] == 0 for r in receipts),
            'Unexpected physical research spend')
    save(root/'completion.json', dict(source=config['source'], protocol=sha(root/'protocol.json'), waves=waves,
        final=final, final_roles=roles, accepted_line=ancestors, promotions=len(ancestors)-1, runs=len(receipts),
        fits=fits, api_calls=0, course_test_accessed=False))
    print('Complete:', len(receipts), 'cold runs,', fits, 'actual fits,', len(ancestors)-1, 'promotions', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','run'])
    parser.add_argument('root', type=Path)
    parser.add_argument('prior', type=Path)
    parser.add_argument('--downloads', type=Path)
    parser.add_argument('--cornell', type=Path)
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.root.resolve(), args.prior.resolve(), args.downloads.resolve(), args.cornell.resolve())
    else:
        study(args.root.resolve(), args.prior.resolve(), args.weights.resolve())
