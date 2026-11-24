"""Training-only, third-action intervention using the integrated executor."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
INTEGRATION = REPO / 'runs' / 'integration'
if not (INTEGRATION / 'superagent').is_dir():
    INTEGRATION = REPO
sys.path.insert(0, str(INTEGRATION))

import joblib
import numpy as np
from superagent.adaptive_local import save, sha
from superagent.benchmark import run_one
from superagent.contracts import ExperimentProposal
from superagent.error_slices import ErrorSlices
from superagent.recipe_models import materialize_recipe
from superagent.recipe_registry import RecipeRegistry
from tools.expanded_feedback_pilot import RUN_OPTIONS, SEARCH_SEED
from tools.search_budget_study import outer_partition, arm_record, predict, read
from superagent.evidence_schedule import duel

SEEDS = (43111, 43117)
FIXED_SPEC = dict(kind='pair_text', analyzer='word', representation='lexical',
                  classifier='linear_svm', C=1.0, dimensions=32)


class FixedRegistry(RecipeRegistry):
    def propose(self, history):
        if len(history) != 2:
            return []
        candidate = materialize_recipe(FIXED_SPEC)
        entry = dict(spec=deepcopy(FIXED_SPEC), params=candidate.params,
                     origin=dict(rule='preregistered_fixed_recipe', after_trial=2,
                                 hypothesis='Fixed lexical SVM improves internal accuracy'))
        self.document['entries'][candidate.family] = entry
        self.save()
        return [dict(id=candidate.family, **entry)]


class ThirdActionPolicy:
    def __init__(self, baseline_path, third):
        self.path = baseline_path
        self.third = third

    def propose(self, state):
        record = state.record()
        index = record['next_trial'] - 1
        chosen = self.third if index == 2 else self.path[index]
        config = next(row for row in record['available_candidates'] if row['id'] == chosen)
        return ExperimentProposal.from_decision(index + 1, config,
            'Preregistered intervention at trial three; shared baseline actions elsewhere',
            None, record['history'])


def validate(folder):
    summary = read(folder / 'summary.json')
    if not (summary['valid_policy_run'] and summary['refit_succeeded']
            and summary['confirmation']['valid_confirmation']
            and summary['failed_trials'] == 0 and summary['llm_calls'] == 0
            and summary['trials_attempted'] == 4 and not summary['test_accessed']
            and summary['classifier_fit_accounting']['completed_classifier_fits'] == 11):
        raise RuntimeError(f'Incomplete run: {folder}')
    return arm_record(folder)


def check_prefix(baseline, challenger):
    for trial in (1, 2):
        left = joblib.load(baseline / f'trial_{trial}.joblib')
        right = joblib.load(challenger / f'trial_{trial}.joblib')
        if not np.array_equal(left['search_predictions'], right['search_predictions']):
            raise ValueError('Shared prefix predictions changed')
    for arm in (baseline, challenger):
        if any(r['error_slices']['status'] != 'completed' for r in read(arm / 'trials.json')):
            raise ValueError('Missing inner error evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', type=Path, default=REPO / 'data/released_task2.zip')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    options = RUN_OPTIONS | dict(error_feedback=True)
    save(root / 'protocol.json', dict(seeds=SEEDS, fixed_spec=FIXED_SPEC,
         options=options, search_seed=SEARCH_SEED, archive_sha256=sha(args.task),
         runner_sha256=sha(Path(__file__)),
         source_sha256={str(p.relative_to(INTEGRATION)): sha(p)
                        for p in sorted((INTEGRATION / 'superagent').glob('*.py'))},
         logical_fits_per_arm=11, arms=['baseline', 'fixed', 'evidence'],
         intervention_trial=3, api_calls=0,
         equivalence_reuse='Identical complete action paths reuse the fitted arm',
         scope='Two development splits; outer labels are used only after all models freeze'))
    results = []
    started = time.monotonic()
    for seed in SEEDS:
        case = root / str(seed)
        case.mkdir()
        task, features, labels, partition = outer_partition(args.task, seed)
        save(case / 'partition.json', partition)
        baseline = case / 'baseline'
        print(f'{seed} baseline started', flush=True)
        run_one(task, 'local_interleaved_routes', SEARCH_SEED, baseline, None, **options)
        arms = dict(baseline=validate(baseline))
        history = read(baseline / 'trials.json')
        # Reconstruct evidence independently from persisted predictions as a boundary check.
        payload = joblib.load(baseline / 'inner_data.joblib')
        slicer = ErrorSlices(task.kind, payload['X_train'])
        for row in history[:2]:
            artifact = joblib.load(baseline / f"trial_{row['trial']}.joblib")
            rebuilt = slicer.describe(payload['X_search'], payload['y_search'], artifact['search_predictions'])
            if rebuilt != row['error_slices']:
                raise ValueError('Saved error slices disagree with predictions')
        preview = RecipeRegistry(task.kind, 'expanded', case / 'evidence_preview.json')
        proposals = preview.propose(history[:2])
        if not proposals:
            raise ValueError('No evidence-driven proposal')
        evidence_id = proposals[0]['id']
        fixed_id = materialize_recipe(FIXED_SPEC).family
        save(case / 'third_action_plan.json', dict(fixed_id=fixed_id, evidence_id=evidence_id,
             proposals=proposals, action_changed=evidence_id != fixed_id,
             hypothesis='Evidence-based composition improves search accuracy over its reference',
             falsifier='Third-action search accuracy fails to exceed reference accuracy',
             reference_trial=proposals[0]['origin']['reference_trial']))
        print(f'{seed} proposal analyzer={proposals[0]["spec"]["analyzer"]}; '
              f'gap={proposals[0]["origin"]["observed_gap"]:.4f}; '
              f'fixed/evidence differ={evidence_id != fixed_id}', flush=True)
        for name, third in [('fixed', fixed_id), ('evidence', evidence_id)]:
            if name == 'evidence' and third == fixed_id:
                arms[name] = deepcopy(arms['fixed']) | dict(reused=True, new_fits=0,
                    equivalent_to='fixed', proposal_file=str(case / 'third_action_plan.json'))
                print(f'{seed} evidence reuses identical fixed action path', flush=True)
                continue
            folder = case / name
            policy = ThirdActionPolicy(arms['baseline']['path'], third)
            print(f'{seed} {name} started', flush=True)
            registry_type = FixedRegistry if name == 'fixed' else RecipeRegistry
            with patch('superagent.recipe_registry.RecipeRegistry', registry_type):
                run_one(task, 'local_interleaved_routes', SEARCH_SEED, folder, None,
                        **options, refinement='compose', decision_policy=policy)
            arms[name] = validate(folder) | dict(new_fits=11)
            check_prefix(baseline, folder)
            if arms[name]['path'][3] != arms['baseline']['path'][3]:
                raise ValueError('Fourth action changed')
        save(case / 'frozen_arms.json', arms)
        predictions = {name: predict(record, features, labels) for name, record in arms.items()}
        correct = {name: pred == labels for name, pred in predictions.items()}
        for name, arm in arms.items():
            arm.update(accuracy=float(correct[name].mean()), correct=int(correct[name].sum()))
            folder = Path(arm['model_file']).parent
            third_row = read(folder / 'trials.json')[2]
            arm.update(third_search_score=third_row['search_score'],
                       third_in_shortlist=third_row['candidate'] in arm['finalist_ids'],
                       third_won=third_row['candidate'] == arm['candidate'])
        comparisons = {f'{right}_minus_{left}': duel(correct[left], correct[right])
                       for left, right in [('baseline', 'fixed'), ('baseline', 'evidence'), ('fixed', 'evidence')]}
        result = dict(seed=seed, rows=len(labels), arms=arms, comparisons=comparisons,
                      evidence_action_changed=evidence_id != fixed_id)
        results.append(result)
        save(case / 'outer_predictions.json', {k: v.tolist() for k, v in predictions.items()})
        save(root / 'results.json', results)
        print(json.dumps(dict(seed=seed, comparisons=comparisons)), flush=True)
    save(root / 'completion.json', dict(status='complete', elapsed_seconds=time.monotonic() - started,
         cases=len(results), new_fits=sum(11 + sum(a.get('new_fits', 0) for n, a in r['arms'].items()
                                                  if n != 'baseline') for r in results), api_calls=0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
