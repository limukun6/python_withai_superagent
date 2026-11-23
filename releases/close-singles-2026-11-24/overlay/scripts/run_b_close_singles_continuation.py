"""Run the native decision loop on a known stopping case, reusing only its five attested primary fits."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import joblib

from superagent import ab_run
from superagent.cli import build_parser
from scripts.run_b_autonomous_close_singles import OPTIONS, ROOT, SOURCE, cli, env, packet, read, require, seal
from superagent.data import load_development_task


PLAN = ROOT / 'docs/batches/B_resume_close_singles_plan_20261124_zh.md'


def prepare(args):
    prior = args.original / 'agent'
    full = read(args.original / 'models_frozen.json')
    original = read(prior / 'run_manifest.json')
    source = ab_run.sources()
    history = read(prior / 'history.json')
    archive = args.original / 'anonymous.zip'
    require(ab_run.sha(archive) == original['task_archive_sha256'] == SOURCE and
            ab_run.sha(prior / 'history.json') == full['raw_history'] and
            ab_run.sha(args.original / 'physical.jsonl') == full['physical'] and
            ab_run.sha(prior / 'selected_model.joblib') == full['model'], 'Original evidence chain differs')
    replayed = read(args.replay)
    trace = replayed['runs']['full_titles']['decision_steps'][-1]
    require(trace['state'] == 6 and trace['original'][0] == 'stop' and trace['proposed'][0] == 'confirm'
            and trace['state_sha256'] == ab_run.sha(prior / 'state_6.json'), 'Known divergence did not reproduce')
    measured = replayed['replay_source']
    require(set(measured) == set(source) and [k for k in sorted(source) if source[k] != measured[k]] == ['ab_close_singles.py'],
            'Expected only the additional pre-confirmation budget guard')
    verified_modules = ('experiment_worker.py', 'benchmark.py', 'ab_sentence.py', 'models.py', 'ab_complement.py',
                        'metrics.py', 'prediction_evidence.py', 'score_evidence.py', 'word_familiarity.py', 'splits.py')
    require(all(original['source_files'][n] == source[n] for n in verified_modules), 'A reused training implementation changed')
    require(len(history) == 5 and all(v['split_id'] == 'primary' for v in history) and
            original['fit_limit'] == 10 and full['actions'] == read(prior / 'actions.json'), 'Wrong historical opening')
    old_state = read(prior / 'state_6.json')
    require(old_state['history'] == history and len(old_state['prediction_evidence']) == 5, 'Frozen trace differs')
    cached = []
    for row, prediction in zip(history, old_state['prediction_evidence'], strict=True):
        i = row['trial']
        artifact = prior / f'trial_{i}.joblib'
        meter = artifact.with_suffix('.fits.json')
        fits = read(meter)
        require(fits['status'] == 'completed' and fits['candidate'] == row['candidate'] and
                fits['attempted_classifier_fits'] == fits['completed_classifier_fits'] == 1, 'Unknown historical fit')
        raw = joblib.load(artifact)
        record = raw['prediction_record']
        require(raw['metrics']['resolved_configuration']['params'] == original['capabilities'][row['candidate']]['params']
                and raw['metrics']['search_score'] == row['search_score'] and
                record['predictions'].tolist() == prediction['predictions'] and record['score_record'] == prediction['score_record']
                and record['evaluation_payload_sha256'] == row['evaluation_payload_sha256']
                and record['prediction_sha256'] == prediction['prediction_sha256'], 'Prior model or row evidence differs')
        cached.append(dict(trial=i, candidate=row['candidate'], artifact=ab_run.sha(artifact), fits=ab_run.sha(meter),
                           payload=record['evaluation_payload_sha256']))
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(plan=ab_run.sha(PLAN), runner=ab_run.sha(Path(__file__)), source_files=source,
                    original_manifest=ab_run.sha(prior / 'run_manifest.json'), original_frozen=ab_run.sha(args.original / 'models_frozen.json'),
                    replay=ab_run.sha(args.replay), opening=cached, archive=SOURCE, options=OPTIONS + ['--confirm-close-singles'],
                    newly_admitted_confirmations=2, newly_admitted_final_refits=1, model_api_cap=0)
    seal(args.output / 'protocol.json', protocol)
    return cached, original, history


def run(args):
    cached, original, old_history = prepare(args)
    prior = args.original / 'agent'
    out = args.output
    archive = args.original / 'anonymous.zip'
    physical_log = out / 'physical.jsonl'
    os.environ.update(env(ROOT, args.weights, physical_log))
    from scripts.ab_acceptance_guard.sitecustomize import offline
    sys.addaudithook(offline)
    directory = out / 'agent'
    native_worker = ab_run.train_candidate
    reused, new = [], []

    def resume_worker(payload, name, destination, timeout, *, fit_admission_path):
        target = Path(destination)
        kind = Path(payload).name
        if kind == 'primary.joblib':
            require(len(reused) < len(cached) and not new, 'Unexpected later opening or interleaved cache reuse')
            ref = cached[len(reused)]
            require(ref['candidate'] == name and ab_run.sha(payload) == ref['payload'] and
                    target.name == f"trial_{ref['trial']}.joblib", 'Requested primary action differs')
            require(read(directory / 'run_manifest.json')['capabilities'] == original['capabilities'], 'Candidate menu differs')
            model_file = prior / target.name
            meter = model_file.with_suffix('.fits.json')
            require(ab_run.sha(model_file) == ref['artifact'] and ab_run.sha(meter) == ref['fits'], 'Cached worker changed')
            shutil.copyfile(model_file, target)
            shutil.copyfile(meter, target.with_suffix('.fits.json'))
            artifact = joblib.load(target)
            reused.append(dict(candidate=name, trial=ref['trial'], source_sha256=ref['artifact']))
            return artifact['model'], dict(status='completed', **artifact['metrics'],
                                           worker_wall_seconds=old_history[len(reused) - 1]['worker_wall_seconds'])
        require(len(reused) == 5 and (kind == 'confirmation_1.joblib' and len(new) < 2 or
                kind == 'refit_data.joblib' and len(new) == 2), 'Unexpected new fit or exceeding cap')
        if kind == 'confirmation_1.joblib':
            require(ab_run.sha(payload) == ab_run.sha(prior / kind), 'Registered confirmation payload changed')
        new.append(dict(candidate=name, stage=kind, payload_sha256=ab_run.sha(payload)))
        return native_worker(payload, name, target, timeout, fit_admission_path=fit_admission_path)

    parsed = build_parser().parse_args(['run', str(archive), *OPTIONS, '--confirm-close-singles', '--run-dir', str(directory)])
    with patch.object(ab_run, 'train_candidate', resume_worker), (out / 'native.log').open('x') as log, redirect_stdout(log):
        parsed.handler(parsed)
    outcome = read(directory / 'run_outcome.json')
    require(outcome['status'] == 'completed' and outcome['requested_planner_completed'] and outcome['full_refit_completed']
            and outcome['source_unchanged'] and not outcome['formal_test_accessed'] and outcome['new_api_requests'] == 0,
            'Native continuation did not complete')
    final = read(directory / 'final_selection.json')
    check = outcome['confirmation']['b_scheduler_review']['close_singles']
    actions = read(directory / 'actions.json')
    physical = [json.loads(line)['candidate'] for line in physical_log.read_text().splitlines()]
    charged = read(directory / 'classifier_fits.json')
    require(actions[:5] == read(prior / 'actions.json')[:5] and [a['action'] for a in actions[5:]] == ['confirm','confirm','stop']
            and [a['candidate_id'] for a in actions[5:7]] == [check['anchor'], check['challenger']]
            and check['status'] in ('retain','switch') and len(new) == len(physical) == 3 and physical == [v['candidate'] for v in new]
            and charged['attempted_classifier_fits'] == charged['completed_classifier_fits'] == 8 and charged['unknown_failed_attempts'] == 0
            and final['training_rows'] == 8301 and ab_run.sha(directory / 'selected_model.joblib') == final['model_sha256'],
            'Actions, physical fit meters, or final refit differ')
    seal(out / 'model_frozen.json', dict(protocol=ab_run.sha(out / 'protocol.json'), model=final['model_sha256'],
         family=final['family'], reused_primary=reused, new_workers=new, actions=actions, review=check))
    # Only exercise prediction delivery. These rows belong to the training archive and are never scored.
    task = load_development_task(archive)
    packet(out / 'delivery_features.zip', task.X_train[:32])
    hashes = []
    for n in (0, 1):
        csv = out / f'export_{n}' / 'rows.csv'
        cli(ROOT, ['predict', out / 'delivery_features.zip', '--run-dir', directory, '--output', csv],
            dict(os.environ), out / f'predict_{n}.log', 300)
        report = read(csv.parent / 'prediction_manifest.json')
        require(report['prediction_count'] == 32 and not report['test_labels_read'], 'Delivery check differs')
        hashes.append(ab_run.sha(csv))
    require(hashes[0] == hashes[1], 'Prediction processes disagree')
    seal(out / 'results.json', dict(frozen=ab_run.sha(out / 'model_frozen.json'), old_winner=read(args.original / 'models_frozen.json')['selection'],
         delivered=final['family'], review=check, physical_fits_reused=5, physical_fits_new=3, logical_path_fits=8, training_rows=8301,
         prediction_count_per_export=32, predictions_sha256=hashes, model_api_calls=0, new_rmb=0,
         new_generalization_score_measured=False, designated_validation_opened=False, formal_test_opened=False))
    seal(out / 'completion.json', {k: ab_run.sha(out / (k + '.json')) for k in ('protocol','model_frozen','results')})
    print(json.dumps(read(out / 'results.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('original','replay','weights','output'):
        parser.add_argument('--' + option, required=True, type=lambda s: Path(s).resolve())
    run(parser.parse_args())
