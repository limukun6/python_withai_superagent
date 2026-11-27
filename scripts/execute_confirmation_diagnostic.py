"""Cold-run the only scheduler change found in the fit-free historical replay."""
import argparse
import shutil
from pathlib import Path

from scripts import compare_joint_feedback as study
from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, read, require, save, sha


PLAN = ROOT / 'research/confirmation_replay_1127/execution_plan.json'


def execute(a):
    replay = read(a.replay)
    keys = [case for case, row in replay['paths'].items() if row['first_alternative']]
    require(len(keys) == 1 and source_identity(ROOT) == read(PLAN)['runtime'], 'Unexpected cohort or runtime')
    case = keys[0]
    prior = a.previous / 'development' / case
    ledger = replay['paths'][case]
    old = prior / 'feedback-local/run/agent'
    require(sha(old / 'decisions.json') == ledger['decisions_sha256'], 'Comparator differs from replay')
    archived = read(a.previous / 'development/registered.json')['cases'][case]
    new = a.root / 'development' / case
    new.mkdir(parents=True, exist_ok=False)
    for name in ('task.zip', 'query.zip'):
        shutil.copyfile(prior / name, new / name)
    frozen = dict(runtime=source_identity(ROOT), runner=sha(Path(__file__)), guard=sha(study.GUARD),
                  plan=sha(PLAN), entry_runner=sha(Path(study.__file__)), replay_sha256=sha(a.replay))
    save(a.root / 'development/registered.json', dict(frozen=frozen, cases={case: archived}))
    a.case, a.arm = case, 'feedback-confirm'
    study.run(a, expected_fingerprint=frozen, additional_sealed_directory=a.previous)
    treatment = new / a.arm / 'run/agent'
    seal = read(new / a.arm / 'sealed.json')
    old_seal = read(prior / 'feedback-local/sealed.json')
    expected = ledger['first_alternative']
    at = expected['step'] - 1
    actual = read(treatment / 'decisions.json')
    controls = read(old / 'decisions.json')
    fields = ('action', 'members', 'split')
    def before(d): return tuple(d['choice'][k] for k in fields)
    def measured(path):
        return [(r['candidate'],r['split_id'],r['status'],r['search_score']) for r in read(path / 'history.json')]
    prefix = all(before(x) == before(y) for x, y in zip(actual[:at], controls[:at]))
    choice = actual[at]['choice'] if at < len(actual) else {}
    selected = read(treatment / 'final_selection.json')
    result = dict(case=case, runtime_sha256=source_identity(ROOT), registered_first_change=expected,
        actions_match_before_change=len(actual) >= at and prefix,
        scores_match_before_change=measured(treatment)[:at] == measured(old)[:at],
        actual_change={k: choice.get(k) for k in fields}, actual_source=actual[at]['source'] if at < len(actual) else None,
        protection_events=read(treatment / 'task_opening.json').get('protected_confirmations',[]),
        new_fold_measurements=[r for r in measured(treatment)[at:] if r[0] in expected['proposed_members']],
        comparator_final_members=old_seal['core'], selected_members=seal['core'],
        comparator_final_folds=read(old / 'final_selection.json')['fold_scores'], selected_folds=selected['fold_scores'],
        fits=seal['fits'], comparator_preexisting_fits=old_seal['fits'], new_model_requests=seal['requests'],
        refit=selected['refit'], independent_reload=seal['fresh_reload'], prediction_sha256=seal['output_sha'],
        same_predictions_as_control=seal['output_sha'] == old_seal['output_sha'], external_labels_opened=False)
    save(a.root / 'actual_confirmation.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('replay', 'previous', 'root', 'weights'):
        p.add_argument('--' + name, required=True, type=lambda s: Path(s).resolve())
    execute(p.parse_args())
