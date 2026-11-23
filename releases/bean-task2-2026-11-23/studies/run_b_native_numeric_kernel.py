"""Compare a fresh numeric portfolio run with the registered dynamic control, then score their exports."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from scripts.b_frozen_head_frontier import paired
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_c1_matrix import read_export
from scripts.run_b_local_component_pair import check, command, read, save
from superagent import ab_run, numeric_kernel
from superagent.ab_fixed_schedule import fixed_action
from superagent.ab_scheduler import next_action
from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_prediction_features
from superagent.prediction_evidence import fingerprint
from superagent.splits import grouped_partition, sample_groups

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'docs/batches/B_native_numeric_kernel_plan_1123_zh.md'
SHA = ab_run.sha
TRAIN = 'f181e6244d88b3785535484983e8a9fc56308ce6410217106c3f877b124bdcb6'
ROWS = '29cf548f039665f608c9c77906931b2f9dddaa3779c5149ca13bab0eabd1dc22'
FEATURES = '7ce0e8007b92dcd5ebdc58c8cb49d75a1f86d95784ece90194ade0ab3c35abd4'
HOLDOUT = 'c1ddc6ba8b78ff419b53d97063e83a824d8735ef1198018dc8b3b0145199f0d3'
CONTROL = '2d756200e6b7d31076e6d7daa775d8a23ed726e4fe8e328d2ee12885ea5b5043'
ORIGINAL_RUNNER = '3cfb8c93f807b07ceafc6353e7745ed5451fa4909d8c40f3c55bdf980df42345'
CONTROL_HASHES = {
    'ab-v1': ('9d18b8619281da8683cbe5d48010a4c37a27a3c294663e0a1a44bdb26494fdaf',
              '80197526fd17fe0bcef2fa2cd5e90cb28757d731e67d1697543de1adf06b7c98',
              '690948bceb48867d24469f4a6947c0d862223247f75cfcf474658108974efe15'),
    'ab-fixed': ('c461cd6e47be7766484e58841b94ad14d7019074f9e4b92de8f9b29d923b5d45',
                 '8942cdfad3e1fcd232bc138ca2fd00151ef515e06430ab10bf2789775d2607ae',
                 '86ce47dfd66adc79e5a65f36fe9cc915f94bcf588f10a4bceb962dc3af95710e')}


def preflight(data, old, scheduler):
    check(SHA(data / 'train.zip') == TRAIN and SHA(data / 'registered_rows.json') == ROWS,
          'Training input identity differs')
    check(SHA(old / 'selected_model.joblib') == CONTROL and
          tuple(SHA(old / f) for f in ('run_manifest.json','actions.json','history.json')) ==
          CONTROL_HASHES[scheduler], 'Historical control changed')
    task = load_development_task(data / 'train.zip')
    manifest = read(old / 'run_manifest.json')
    check(task.X_val is None and task.y_val is None and task.X_train.shape == (8710,16)
          and manifest['scheduler'] == scheduler and manifest['planner_mode'] == 'local', 'Unexpected training scope')
    spec = TaskSpec.from_task(task).record()
    caps = ab_run.registered_capabilities(spec, len(task.y_train))
    expanded = ab_run.registered_capabilities(spec, len(task.y_train), numeric_kernel_width=task.X_train.shape[1])
    check(caps == manifest['capabilities'] and {k:v for k,v in expanded.items() if k != numeric_kernel.NAME} == caps
          and numeric_kernel.NAME in expanded, 'New menu differs by more than one addition')
    groups = sample_groups(task)
    trusted = {}
    for row in read(data / 'registered_rows.json')['splits']:
        name = row['id']
        train, held, _ = grouped_partition(task, row['seed'], 'shared_text')
        np.testing.assert_array_equal(row['train_indices'], train)
        np.testing.assert_array_equal(row['search_indices'], held)
        old_split = read(old / (name + '_split.json'))
        check(fingerprint(held) == old_split['row_ids_sha256'] and fingerprint(task.y_train[held]) ==
              old_split['label_sha256'] and SHA(old / (name + '.joblib')) == old_split['evaluation_payload_sha256'],
              'Original split differs')
        trusted[name] = dict(row_ids=held.tolist(), labels=task.y_train[held].tolist(), groups=groups[held].tolist(),
                             evaluation_payload_sha256=old_split['evaluation_payload_sha256'])
    actions = read(old / 'actions.json')
    policy = fixed_action if scheduler == 'ab-fixed' else next_action
    for i, action in enumerate(actions, 1):
        state = read(old / f'state_{i}.json')
        check(policy(state, capabilities=caps, trusted_partitions=trusted) == action,
              f'Default scheduling rules do not reproduce historical action {i}')
    return dict(actions_replayed=len(actions), training=TRAIN, registered_rows=ROWS, reference=CONTROL,
                old_menu=list(caps), new_menu=list(expanded), splits={
                    name: fingerprint(part['row_ids']) for name, part in trusted.items()})


def execute(a):
    a.out.mkdir(parents=True,exist_ok=True)
    run = a.out / 'agent'
    pr = preflight(a.data, a.control, a.scheduler)
    protocol = dict(plan=SHA(PLAN), runner=SHA(Path(__file__)), source=ab_run.sources(),
                    reference=pr, fit_cap=11, model_api_cap=0, features=FEATURES, holdout=HOLDOUT)
    if a.resume_original_exports:
        check(a.scheduler == 'ab-v1' and (a.out/'exports.json').is_file() and (run/'run_outcome.json').is_file()
              and SHA(a.out/'exports.json') == '12ca46c03460433edc9188a0adc8750fec4f702bec952b05a695b46f8ec095e5',
              'Expected the original sealed exports')
        protocol['runner'] = ORIGINAL_RUNNER
    else:
        protocol['schedule_plan'] = SHA(ROOT/'docs/batches/B_numeric_schedule_comparison_1123_zh.md')
        protocol['scheduler'] = a.scheduler
    frozen_save(a.out / 'protocol.json', protocol)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (ROOT/'scripts/ab_acceptance_guard', ROOT))),
               AB_B1_OFFLINE='1', AB_B1_PER_WORKER_FIT_LIMIT='1', AB_B1_FIT_LOG=str(a.out/'physical.jsonl'),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
    options = ['run',str(a.data/'train.zip'),'--planner','local','--portfolio','ab-v1','--scheduler',a.scheduler,
               '--seed','707','--fit-limit','11','--trial-timeout','300','--time-limit','1200','--numeric-kernel',
               '--run-dir',str(run)]
    if not (run/'run_outcome.json').exists():
        check(not run.exists() and not (a.out/'physical.jsonl').exists(), 'A prior path has incomplete accounting')
        status, _ = command(options, a.out/'training.log', env, 1260)
        check(status == 0, 'Native run failed; inspect the training log')
    outcome, fits, selection, manifest = [read(run/f) for f in ('run_outcome.json','classifier_fits.json',
                                                                'final_selection.json','run_manifest.json')]
    physical = [json.loads(line) for line in (a.out/'physical.jsonl').read_text().splitlines()]
    count = len(physical)
    check(0 < count <= 11 and count == fits['attempted_classifier_fits'] == fits['completed_classifier_fits']
          and fits['unknown_failed_attempts'] == 0 and [r['candidate'] for r in physical] ==
          [r['candidate'] for r in fits['workers']], 'Supervised counts differ')
    check(outcome['requested_planner_completed'] and outcome['full_refit_completed'] and outcome['new_api_requests'] == 0
          and not outcome['formal_test_accessed'] and outcome['source_unchanged'] and selection['training_rows'] == 8710
          and manifest['task_archive_sha256'] == TRAIN and manifest['source_files'] == ab_run.sources()
          and manifest['numeric_kernel_width'] == 16 and list(manifest['capabilities']) == pr['new_menu'],
          'Native path was not completed on the frozen inputs')
    for split, order in pr['splits'].items():
        current = read(run / (split + '_split.json'))
        previous = read(a.control / (split + '_split.json'))
        check(current['row_ids_sha256'] == order and all(current[k] == previous[k] for k in
              ('train_indices','search_indices','label_sha256','evaluation_payload_sha256')), 'Unmatched trial split')
    model = run/'selected_model.joblib'
    check(SHA(model) == selection['model_sha256'], 'Exported model differs')
    frozen_save(a.out/'models_frozen.json',dict(protocol=SHA(a.out/'protocol.json'), control=CONTROL, new=SHA(model),
                decision=SHA(run/'actions.json'), history=SHA(run/'history.json'), physical=SHA(a.out/'physical.jsonl'),
                selection=selection['family'], fits=count))
    features = a.evaluation / 'features.zip'
    check(SHA(features) == FEATURES, 'Unknown evaluation features')
    predictions, exports = {}, {}
    for key, folder in [('control',a.control), ('new_first',run), ('new_second',run)]:
        csv = a.out/key/'rows.csv'
        if not (csv.parent/'prediction_manifest.json').exists():
            status, _ = command(['predict',str(features),'--run-dir',str(folder),'--output',str(csv)],
                                a.out/(key+'.log'),env,300)
            check(status == 0, 'New-process export failed for '+key)
        predictions[key] = np.asarray(read_export(csv, 2178))
        exports[key] = SHA(csv)
    check(exports['new_first'] == exports['new_second'], 'Repeated export differs')
    frozen_save(a.out/'exports.json',dict(models=SHA(a.out/'models_frozen.json'), exports=exports, rows=2178))
    if a.freeze_only:
        print(json.dumps(dict(scheduler=a.scheduler, selected=selection['family'], fits=count, exports=exports)))
        return
    frozen_save(a.out/'scoring_protocol.json', dict(runner=SHA(Path(__file__)), collection=SHA(a.out/'protocol.json'),
                plan=SHA(ROOT/'docs/batches/B_numeric_schedule_comparison_1123_zh.md'), exports=SHA(a.out/'exports.json')))
    held = a.evaluation/'holdout.joblib'
    check(SHA(held) == HOLDOUT, 'Unrecognized evaluation file')
    truth = joblib.load(held)
    x = load_prediction_features(features, 'numeric_tabular', read(run/'task_spec.json'))
    np.testing.assert_array_equal(x, truth['X'])
    check(fingerprint(truth['row_ids']) == 'd6a1c77682555166435d87ec099e7d9842948f79d4670efc1b26127ecb644147'
          and len(set(truth['row_ids'])) == len(x), 'Original A94 source row order differs')
    y = np.asarray(truth['y'])
    fitted = joblib.load(model)['model']
    prob = fitted.predict_proba(x)
    check(np.isfinite(prob).all() and np.all(prob >= 0) and np.allclose(prob.sum(axis=1),1), 'Invalid probabilities')
    np.testing.assert_array_equal(fitted.classes_, np.unique(y))
    np.testing.assert_array_equal(fitted.classes_[prob.argmax(axis=1)], predictions['new_first'])
    old, new = predictions['control'] == y, predictions['new_first'] == y
    check(int(old.sum()) == 1989, 'Historical L1 development result did not reproduce')
    delta = new.astype(int) - old.astype(int)
    groups = sample_groups(SimpleNamespace(kind='numeric_tabular', X_train=x))
    counts = {}
    for group, before, after in zip(groups, old, new, strict=True):
        counts.setdefault(group, []).append(int(before) + 2 * int(after))
    frozen_save(a.out/'grouped_correctness.json', [sorted(counts[k]) for k in sorted(counts)])
    result = dict(rows=len(y), baseline=int(old.sum()), new=int(new.sum()), gained=int((delta>0).sum()),
                  lost=int((delta<0).sum()), paired=paired(delta,groups), selected=selection['family'],
                  keep_optional=bool(delta.sum() >= 8), physical_fits=count, model_api_calls=0, new_rmb=0,
                  historical_control_fits=read(a.control/'classifier_fits.json')['completed_classifier_fits'],
                  historical_actions_replayed=pr['actions_replayed'], class_order=fitted.classes_.tolist(),
                  trials=[dict(candidate=h['candidate'],split=h['split_id'],score=h['search_accuracy'])
                          for h in read(run/'history.json')], course_test_opened=False)
    frozen_save(a.out/'results.json', result)
    frozen_save(a.out/'completion.json', {k: SHA(a.out/(k+'.json')) for k in
                ('protocol','models_frozen','exports','scoring_protocol','results','grouped_correctness')})
    print(json.dumps({k:v for k,v in result.items() if k not in ('trials','class_order')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scheduler',choices=list(CONTROL_HASHES),default='ab-v1')
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--resume-original-exports',action='store_true')
    for key in ('data','control','evaluation','out'):
        parser.add_argument('--'+key, type=lambda v: Path(v).resolve(), required=True)
    execute(parser.parse_args())
