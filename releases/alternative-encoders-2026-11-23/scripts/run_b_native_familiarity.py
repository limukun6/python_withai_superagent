"""Run a new catalog through the unchanged native local complement scheduler."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, command, read
from superagent import ab_run
from superagent.ab_methods import TEXT_FAMILIARITY, TRANSFER_METHODS


def run(workspace, inputs, old, encoders, matched, out):
    out.mkdir(parents=True, exist_ok=True)
    env = offline_environment(workspace)
    previous = read(old / 'protocol.json')
    matched_result = read(matched / 'results.json')
    check(matched_result['qualified_for_full_refit'] and previous['seed'] == 18643
          and read(old / 'results.json')['actual_new_fits'] == 8, 'Wrong previous trials')
    check(ab_run.sha(inputs / 'anonymous.zip') == previous['packets']['anonymous.zip'], 'Wrong training data')
    frozen_save(out / 'protocol.json', dict(plan=ab_run.sha(ROOT / 'docs/batches/B_native_familiarity_1123_zh.md'),
        runner=ab_run.sha(Path(__file__)), production=ab_run.sources(), old=ab_run.sha(old / 'results.json'),
        matched=ab_run.sha(matched / 'results.json'), packets=previous['packets'], seed=18643,
        fit_cap=8, model_api_cap=0, designated_validation_eligibility=1235))
    folder = out / 'autonomous'
    log = out / 'physical.jsonl'
    if not folder.exists() and not log.exists():
        opts = ['run', str(inputs / 'anonymous.zip'), '--portfolio', 'ab-v1', '--scheduler', 'ab-complement',
                '--planner', 'local', '--preference-catalog', 'deep-familiarity', '--seed', '18643',
                '--fit-limit', '8', '--trial-timeout', '300', '--time-limit', '1200', '--run-dir', str(folder)]
        status, _ = command(opts, out / 'run.log',
                            dict(env, AB_B1_FIT_LOG=str(log), AB_B1_PER_WORKER_FIT_LIMIT='1'), 1260)
        check(status == 0, 'Native entry failed; inspect recorded expenditures before proceeding')
    old_run = old / 'autonomous'
    manifest = read(folder / 'run_manifest.json')
    old_menu = read(old_run / 'run_manifest.json')['capabilities']
    menu = manifest['capabilities']
    e5 = TRANSFER_METHODS['deep-e5'][1]
    check(set(menu) - set(old_menu) == {TEXT_FAMILIARITY} and set(old_menu) - set(menu) == {e5}
          and [n for n in menu if n != TEXT_FAMILIARITY] == [n for n in old_menu if n != e5]
          and all(menu[n] == old_menu[n] for n in set(menu) & set(old_menu))
          and manifest['seed'] == 18643 and manifest['complement_active'], 'Unexpected experimental changes')
    for split in ('primary', 'confirmation_1', 'confirmation_2'):
        check(ab_run.sha(folder / (split + '.joblib')) == ab_run.sha(old_run / (split + '.joblib'))
              and read(folder / (split + '_split.json')) == read(old_run / (split + '_split.json')),
              'Input split differs')
    def indexed(directory):
        return {(r['split_id'], r['candidate_id']): r for r in read(directory / 'prediction_evidence.json')}
    before, now = indexed(old_run), indexed(folder)
    common = sorted(set(before) & set(now))
    check(len(common) >= 1 and all(before[k]['prediction_sha256'] == now[k]['prediction_sha256'] for k in common),
          'Shared worker outputs differ on identical input')
    selection = read(folder / 'final_selection.json')
    outcome = read(folder / 'run_outcome.json')
    decision = read(folder / 'confirmation.json')['b_scheduler_review']
    fits = read(folder / 'classifier_fits.json')
    count = len(log.read_text().splitlines())
    usage = [json.loads(line) for line in (folder / 'llm_usage.jsonl').read_text().splitlines()]
    check(count == fits['attempted_classifier_fits'] == fits['completed_classifier_fits'] <= 8
          and count == 4 + (2 if decision['confirmation_rows'] else 0) + len(decision['members']),
          'Wrong number of actual fits')
    check(len(usage) == 1 and usage[0]['event'] == 'ledger_initialized' and
          all(usage[0][k] == 0 for k in ('input_tokens', 'output_tokens', 'cost_rmb'))
          and outcome['new_api_requests'] == 0, 'Unexpected model API usage')
    check(outcome['requested_planner_completed'] and outcome['full_refit_completed'] and outcome['source_unchanged']
          and not outcome['formal_test_accessed'] and selection['training_rows'] == 6608
          and ab_run.sha(folder / 'selected_model.joblib') == selection['model_sha256']
          and (selection['ensemble_components'] or [selection['family']]) == decision['members'], 'Incomplete final delivery')
    frozen_save(out / 'selection_frozen.json', dict(model=selection['model_sha256'], selection=selection['family'],
        decision=decision, checked_shared_predictions=[list(k) for k in common], physical_log=ab_run.sha(log),
        actions=ab_run.sha(folder / 'actions.json'), original_fits_reused=8, new_fits=count))
    proposed = verified_export(folder, inputs, env, out)
    exports = dict(new=proposed, old=read(old / 'exports_frozen.json')['new'],
                   deep=read(encoders / 'exports_frozen.json')['exports']['deep-e5'],
                   feature_only=read(matched / 'exports_frozen.json')['familiarity'])
    frozen_save(out / 'exports_frozen.json', dict(methods=exports, selection=ab_run.sha(out / 'selection_frozen.json')))
    check(all(ab_run.sha(inputs / n) == previous['packets'][n] for n in ('features.zip','scoring_only.json','groups_only.json')),
          'Development reserve changed')
    y = np.asarray(read(inputs / 'scoring_only.json')['labels'])
    groups = np.asarray(read(inputs / 'groups_only.json'))
    directories = {'new': out / 'autonomous_export_0', 'old': old / 'autonomous_export_0',
                   'deep': encoders / 'deep-e5_export_0', 'feature_only': matched / 'full_export_0'}
    hits = {k: classes(d / 'rows.csv', exports[k], len(y)) == y for k, d in directories.items()}
    correct = {k: int(h.sum()) for k, h in hits.items()}
    check(len(y) == len(groups) == 1693 and [correct[k] for k in ('old','deep','feature_only')] == [1208,1224,1232],
          'Original matched controls changed')
    new = hits['new']
    result = dict(members=decision['members'], autonomous_choice=selection['family'], offered=TEXT_FAMILIARITY in menu,
        explored=TEXT_FAMILIARITY in decision['opening_correct'], selected_new=TEXT_FAMILIARITY in decision['members'],
        rows=len(y), correct=correct, comparisons={k: dict(gained=int((new & ~h).sum()), lost=int((~new & h).sum()),
        delta=paired(new.astype(int) - h, groups)) for k, h in hits.items() if k != 'new'},
        eligible_for_designated_validation=correct['new'] >= 1235, new_fits=count, old_fits_reused=8,
        model_api_requests=0, new_rmb=0, official_validation_used=False, formal_test_used=False,
        export_seal=ab_run.sha(out / 'exports_frozen.json'))
    frozen_save(out / 'results.json', result)
    frozen_save(out / 'completion.json', dict(status='complete', result=ab_run.sha(out / 'results.json'),
        actual_new_fits=count, model_api_requests=0))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'inputs', 'old', 'encoders', 'matched', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    run(*(getattr(a, n).resolve() for n in ('workspace', 'inputs', 'old', 'encoders', 'matched', 'output')))
