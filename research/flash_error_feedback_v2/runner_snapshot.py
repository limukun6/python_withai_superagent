"""Four Flash decisions followed by controlled third-action training."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
from unittest.mock import patch

from run_composition_study import (REPO, INTEGRATION, SEEDS, ThirdActionPolicy,
                                   validate, check_prefix)
import requests
from superagent.adaptive_local import save, sha
from superagent.benchmark import run_one
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.cli import CAMPUS_CHAT_ENDPOINT, CAMPUS_FLASH_MODEL
from superagent.models import search_candidates
from superagent.recipe_models import materialize_recipe
from superagent.recipe_registry import RecipeRegistry
from superagent.evidence_schedule import duel
from tools.expanded_feedback_pilot import RUN_OPTIONS, SEARCH_SEED
from tools.search_budget_study import outer_partition, read, predict

RECIPE_SPECS = [dict(kind='pair_text', analyzer=analyzer, representation=representation,
                     classifier=classifier, C=c, dimensions=32)
                for analyzer, representation, classifier, c in [
                    ('word', 'lexical', 'linear_svm', 1.),
                    ('char_wb', 'lexical', 'linear_svm', .3),
                    ('word', 'hybrid', 'logistic', .3),
                    ('word', 'lexical', 'logistic', .1)]]
RECIPES = [materialize_recipe(spec) for spec in RECIPE_SPECS]


class MenuRegistry(RecipeRegistry):
    def propose(self, history):
        if len(history) != 2:
            return []
        entries = []
        for spec, candidate in zip(RECIPE_SPECS, RECIPES):
            entry = dict(spec=spec, params=candidate.params,
                         origin=dict(rule='preregistered_shared_action_menu', after_trial=2))
            self.document['entries'][candidate.family] = entry
            entries.append(dict(id=candidate.family, **entry))
        self.save()
        return entries


class StudySession(requests.Session):
    def request(self, method, url, **kwargs):
        kwargs['timeout'] = 60
        return super().request(method, url, **kwargs)


def context_for(baseline, detailed):
    history = read(baseline / 'trials.json')[:2]
    excluded = {r['candidate'] for r in history}
    excluded.add(read(baseline / 'trials.json')[3]['candidate'])
    catalog = [dict(id=c.family, params=c.params)
               for c in [*search_candidates('pair_text', 'expanded'), *RECIPES]
               if c.family not in excluded]
    observations = [{k: row[k] for k in ('trial', 'candidate', 'params', 'status', 'search_score')}
                    for row in history]
    if detailed:
        for row, observation in zip(history, observations):
            observation['error_slices'] = row['error_slices']
    return dict(task_kind='pair_text', primary_metric='accuracy',
        available_candidates=catalog, observed_trials=observations,
        decision_trial=3, remaining_search_actions=2,
        instruction='Choose exactly one available candidate to maximize inner validation accuracy. '
            'Return JSON with candidate, reason (at most 60 words), evidence_trials (prior trial IDs), '
            'target_slice (slice ID or null), hypothesis (one sentence), and falsifier (one sentence). '
            'Use only supplied observations. Untried candidates have unknown performance. '
            'Error groups overlap and are descriptive. Never claim a cause has been established. '
            'Feature dimensions are bounded. The fourth search action and final selection are controlled externally.')


def validate_answer(answer, context):
    ids = {c['id'] for c in context['available_candidates']}
    if answer.get('candidate') not in ids:
        raise ValueError('Flash selected an unavailable candidate')
    for field in ('reason', 'hypothesis', 'falsifier'):
        if not isinstance(answer.get(field), str) or not answer[field].strip():
            raise ValueError(f'Missing decision field: {field}')
    refs = answer.get('evidence_trials')
    if not isinstance(refs, list) or any(type(r) is not int or r not in (1, 2) for r in refs):
        raise ValueError('Invalid evidence trial references')
    target = answer.get('target_slice')
    observed = {s['slice_id'] for r in context['observed_trials']
                for s in (r.get('error_slices') or {}).get('slices', [])}
    if target is not None and target not in observed:
        raise ValueError('Decision cites an unobserved error slice')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=REPO / 'runs/composition_study_v1')
    parser.add_argument('--api-key-file', type=Path, default=REPO / 'api_key.txt')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(root)
    logger.initialize_usage_ledger()
    started = time.monotonic()
    client = ExperimentLLM(CAMPUS_CHAT_ENDPOINT, str(args.api_key_file), CAMPUS_FLASH_MODEL,
        batch_budget=.5, max_requests=4, streaming=True,
        allowed_served_models=['DeepSeek-V4.1-Flash', 'DeepSeek-V4.1-Flash-1017', 'DSF'])
    baseline_protocol = read(args.baseline / 'protocol.json')
    for name, digest in baseline_protocol['source_sha256'].items():
        if sha(INTEGRATION / name) != digest:
            raise ValueError('Integrated source changed since baseline')
    task_path = REPO / 'data/released_task2.zip'
    if sha(task_path) != baseline_protocol['archive_sha256']:
        raise ValueError('Task archive changed')
    # Alternate prompt order by seed. Each API call has an independent history.
    order = [(SEEDS[0], 'scalar'), (SEEDS[0], 'detailed'),
             (SEEDS[1], 'detailed'), (SEEDS[1], 'scalar')]
    save(root / 'protocol.json', dict(model=CAMPUS_FLASH_MODEL, streaming=True, timeout=60,
         maximum_requests=4, estimated_course_cost_ceiling_rmb=.5, order=order,
         recipe_specs=RECIPE_SPECS, seeds=SEEDS, baseline_protocol_sha256=sha(args.baseline/'protocol.json'),
         runner_sha256=sha(Path(__file__)), baseline_runner_sha256=sha(Path(__file__).with_name('run_composition_study.py')),
         source_sha256=baseline_protocol['source_sha256'],
         difference_between_prompts='error_slices field only',
         training_budget_per_arm=11, selection_scope='training-derived partitions only',
         equivalent_paths_reused=True, retries=0))
    decisions = {}
    try:
        for seed, arm in order:
            baseline = args.baseline / str(seed) / 'baseline'
            context = context_for(baseline, arm == 'detailed')
            folder = root / str(seed)
            folder.mkdir(exist_ok=True)
            save(folder / f'{arm}_prompt.json', context)
            print(f'{seed} {arm} API request started', flush=True)
            with patch('requests.Session', StudySession):
                answer = client.ask(context, logger, f'{seed}_{arm}')
            save(folder / f'{arm}_answer.json', answer)
            validate_answer(answer, context)
            decisions[seed, arm] = answer
            print(f'{seed} {arm} selected {answer["candidate"]}', flush=True)
        # All decisions freeze before any new fit or external evaluation.
        results = []
        actual_fits = 0
        for seed in SEEDS:
            folder = root / str(seed)
            baseline = args.baseline / str(seed) / 'baseline'
            baseline_arm = validate(baseline)
            task, features, labels, partition = outer_partition(task_path, seed)
            if partition != read(args.baseline / str(seed) / 'partition.json'):
                raise ValueError('Outer partition changed')
            save(folder / 'partition.json', partition)
            arms = dict(baseline=baseline_arm | dict(new_fits=0))
            prior_fixed = validate(args.baseline / str(seed) / 'fixed')
            known = {tuple(baseline_arm['path']): arms['baseline'],
                     tuple(prior_fixed['path']): prior_fixed | dict(new_fits=0)}
            for arm in ('scalar', 'detailed'):
                chosen = decisions[seed, arm]['candidate']
                path = list(baseline_arm['path'])
                path[2] = chosen
                if tuple(path) in known:
                    record = deepcopy(known[tuple(path)]) | dict(reused=True, new_fits=0)
                    print(f'{seed} {arm} reuses matching complete action path', flush=True)
                else:
                    destination = folder / arm
                    print(f'{seed} {arm} training started', flush=True)
                    with patch('superagent.recipe_registry.RecipeRegistry', MenuRegistry):
                        run_one(task, 'local_interleaved_routes', SEARCH_SEED, destination, None,
                            **(RUN_OPTIONS | dict(error_feedback=True)), refinement='compose',
                            decision_policy=ThirdActionPolicy(baseline_arm['path'], chosen))
                    record = validate(destination) | dict(new_fits=11)
                    check_prefix(baseline, destination)
                    if record['path'] != path:
                        raise ValueError('Intervention path mismatch')
                    known[tuple(path)] = record
                    actual_fits += 11
                arms[arm] = record
            save(folder / 'frozen_arms.json', arms)
            predictions = {arm: predict(record, features, labels) for arm, record in arms.items()}
            correct = {arm: values == labels for arm, values in predictions.items()}
            for arm, record in arms.items():
                trials = read(Path(record['model_file']).parent / 'trials.json')
                third = trials[2]
                record.update(accuracy=float(correct[arm].mean()), correct=int(correct[arm].sum()),
                    third_search_score=third['search_score'],
                    third_in_shortlist=third['candidate'] in record['finalist_ids'],
                    third_won=third['candidate'] == record['candidate'])
                if arm != 'baseline':
                    target = decisions[seed, arm].get('target_slice')
                    record['target_slice'] = target
                    record['third_error_slices'] = third.get('error_slices')
            comparisons = {f'{b}_minus_{a}': duel(correct[a], correct[b])
                           for a, b in [('baseline', 'scalar'), ('baseline', 'detailed'), ('scalar', 'detailed')]}
            result = dict(seed=seed, rows=len(labels), arms=arms, comparisons=comparisons,
                action_changed=decisions[seed, 'scalar']['candidate'] != decisions[seed, 'detailed']['candidate'])
            results.append(result)
            save(folder / 'outer_predictions.json', {a: v.tolist() for a, v in predictions.items()})
            save(root / 'results.json', results)
            print(str(dict(seed=seed, comparisons=comparisons)), flush=True)
        save(root / 'completion.json', dict(status='complete', requests=client.request_count,
            course_cost_rmb=client.spent, actual_account_debit=None, new_fits=actual_fits,
            elapsed_seconds=time.monotonic() - started))
    except Exception as error:
        details = f'{type(error).__name__}: {error}'.replace(client.key, '[redacted]')
        save(root / 'completion.json', dict(status='stopped', error=details,
            requests_attempted=client.request_count, known_course_cost_rmb=client.spent,
            retries=0, elapsed_seconds=time.monotonic() - started))
        print(details, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
