"""Controlled edit-feature intervention and structured hypothesis evaluation."""
from pathlib import Path
import time

from run_composition_study import REPO, INTEGRATION, SEEDS, validate, check_prefix
from superagent.adaptive_local import save, sha
from superagent.benchmark import run_one
from superagent.contracts import ExperimentProposal
from superagent.hypothesis_agent import check_expectation
from superagent.evidence_schedule import duel
from tools.expanded_feedback_pilot import RUN_OPTIONS, SEARCH_SEED
from tools.search_budget_study import outer_partition, read, predict


class EditPolicy:
    def __init__(self, baseline_path):
        self.path = baseline_path

    def propose(self, state):
        record = state.record()
        trial = record['next_trial']
        chosen = 'edit_pair_C0.3' if trial == 3 else self.path[trial - 1]
        config = next(c for c in record['available_candidates'] if c['id'] == chosen)
        proposal = None
        if trial == 3:
            proposal = dict(candidate=chosen, task_kind='pair_text',
                configuration_claim={'classifier': 'logistic'}, evidence_trials=[2],
                hypothesis='Aligned edit features improve high-overlap accuracy and overall search accuracy',
                expected_change=dict(metric='search_accuracy', reference_trial=2,
                                     direction='increase', min_delta=.001),
                slice_expectation=dict(slice_id='pair_token_overlap_high', reference_trial=2, min_delta=.001))
        return ExperimentProposal.from_decision(trial, config,
            'Controlled third-action intervention with shared fourth action', proposal, record['history'])


def main():
    root = REPO / 'runs/targeted_edit_study_v1'
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    options = RUN_OPTIONS | dict(portfolio='challenger', error_feedback=True)
    save(root / 'protocol.json', dict(seeds=SEEDS, options=options,
         runner_sha256=sha(Path(__file__)),
         source_sha256={p.name: sha(p) for p in sorted((INTEGRATION / 'superagent').glob('*.py'))},
         third_action='edit_pair_C0.3', matched_feature_control='word_char_stats_C0.3',
         target_slice='pair_token_overlap_high', min_delta=.001, api_calls=0,
         interpretation='Development study of a preregistered action, with matched same-C feature control'))
    results = []
    for seed in SEEDS:
        case = root / str(seed)
        case.mkdir()
        previous = REPO / 'runs/composition_study_v1' / str(seed)
        baseline = previous / 'baseline'
        control = validate(baseline)
        task, features, labels, partition = outer_partition(REPO / 'data/released_task2.zip', seed)
        if partition != read(previous / 'partition.json'):
            raise ValueError('Baseline partition changed')
        save(case / 'partition.json', partition)
        folder = case / 'edit'
        print(f'{seed} targeted edit training started', flush=True)
        run_one(task, 'local_interleaved_routes', SEARCH_SEED, folder, None,
                **options, decision_policy=EditPolicy(control['path']))
        edit = validate(folder)
        check_prefix(baseline, folder)
        trials = read(folder / 'trials.json')
        if edit['path'] != control['path'][:2] + ['edit_pair_C0.3', control['path'][3]]:
            raise ValueError('Intervention path mismatch')
        # Compare edit and no-edit at identical C, representation and search partition.
        matched_proposal = dict(slice_expectation=dict(slice_id='pair_token_overlap_high', reference_trial=4, min_delta=.001),
            expected_change=dict(metric='search_accuracy', reference_trial=4, direction='increase', min_delta=.001))
        matched_check = check_expectation(matched_proposal, trials[2], [trials[3]])
        arms = dict(baseline=control, edit=edit)
        save(case / 'frozen_arms.json', arms)
        predictions = {name: predict(arm, features, labels) for name, arm in arms.items()}
        correct = {name: values == labels for name, values in predictions.items()}
        for name, arm in arms.items():
            arm.update(accuracy=float(correct[name].mean()), correct=int(correct[name].sum()))
        row = dict(seed=seed, arms=arms, third_search_score=trials[2]['search_score'],
                   prospective_check=trials[2]['hypothesis_check'],
                   matched_feature_check=matched_check,
                   matched_check_scope='Preplanned analysis after both measurements; never shown to third-action policy',
                   third_in_shortlist='edit_pair_C0.3' in edit['finalist_ids'],
                   third_won=edit['candidate'] == 'edit_pair_C0.3',
                   comparison=duel(correct['baseline'], correct['edit']))
        results.append(row)
        save(root / 'results.json', results)
        save(case / 'outer_predictions.json', {k: v.tolist() for k, v in predictions.items()})
        print(str(dict(seed=seed, target=row['prospective_check'], comparison=row['comparison'])), flush=True)
    save(root / 'completion.json', dict(status='complete', new_fits=22, api_calls=0,
         elapsed_seconds=time.monotonic() - started))


if __name__ == '__main__':
    main()
