"""Compare E5 plus each of two existing twelve-layer heads on the same held groups."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, read, save
from superagent import ab_run
from superagent.ab_complement import canonical_probabilities, PreferenceProbabilityVote
from superagent.ab_methods import TRANSFER_METHODS, PREFERENCE_CATALOG_ENV
from superagent.benchmark import train_candidate
from superagent.data import load_development_task
from superagent.external_assets import model_resources, verify_resources
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups

E5 = TRANSFER_METHODS['deep-e5'][1]


def e5_fit(payload, folder):
    folder.mkdir(exist_ok=True)
    admission = folder / 'admission.json'
    ledger = folder / 'fits.json'
    physical = folder / 'physical.jsonl'
    file = folder / 'worker.joblib'
    if not admission.exists():
        save(admission, dict(maximum_classifier_fits=1, ledger_path=str(ledger)))
        with patch.dict(os.environ, {PREFERENCE_CATALOG_ENV: 'deep-e5', 'AB_B1_FIT_LOG': str(physical),
                                     'AB_B1_PER_WORKER_FIT_LIMIT': '1'}):
            model, facts = train_candidate(payload, E5, file, 360, fit_admission_path=admission)
        check(model is not None, 'E5 worker failed: ' + str(facts.get('error', 'unavailable')))
    lines = [json.loads(s) for s in physical.read_text().splitlines()]
    receipt = read(ledger)
    check(len(lines) == receipt['attempted_classifier_fits'] == receipt['completed_classifier_fits'] == 1
          and lines[0]['candidate'] == receipt['candidate'] == E5 and receipt['status'] == 'completed',
          'Unexpected fit count')
    frozen_save(folder / 'receipt.json', dict(candidate=E5, actual_fits=1, physical=ab_run.sha(physical),
        fit_record=ab_run.sha(ledger), worker=ab_run.sha(file), payload=ab_run.sha(payload)))
    return joblib.load(file)


def summarize(hard, truth, groups):
    hits = {n: v == truth for n, v in hard.items()}
    new, original = hits['new_vote'], hits['original_vote']
    return dict(rows=len(truth), correct={n: int(h.sum()) for n, h in hits.items()},
        gained=int((new & ~original).sum()), lost=int((original & ~new).sum()),
        against_original_vote=paired(new.astype(int) - original, groups))


def full_comparison(matched, prior_vote, inputs, out, env):
    folder = out / 'e5_familiarity'
    folder.mkdir(exist_ok=True)
    families = (prior_vote / 'e5_full', matched / 'full')
    configs = [read(f / 'final_selection.json') for f in families]
    for f, c in zip(families, configs):
        check(c['training_rows'] == 6608 and c['refit'] and ab_run.sha(f / 'selected_model.joblib') == c['model_sha256'],
              'Incompatible full-fit artifacts')
    check(configs[0]['task_spec_sha256'] == configs[1]['task_spec_sha256'], 'Task contracts differ')
    components = [joblib.load(f / 'selected_model.joblib')['model'] for f in families]
    vote = PreferenceProbabilityVote('pair_text', components, np.array([0, 1]))
    resources, _ = model_resources(vote)
    verify_resources(resources)
    check({r['name'] for r in resources} == {'e5', 'minilm_l12'}, 'Wrong fixed full models')
    file = folder / 'selected_model.joblib'
    name = 'equal_e5_familiarity_vote'
    if not file.exists():
        shutil.copyfile(families[1] / 'task_spec.json', folder / 'task_spec.json')
        joblib.dump(dict(kind='pair_text', candidate=name, model=vote), file)
    frozen_save(folder / 'final_selection.json', dict(family=name, selected_experiment=name, model_file=file.name,
        model_format='candidate_bundle_v1', model_sha256=ab_run.sha(file), training_rows=6608,
        task_spec_sha256=configs[0]['task_spec_sha256'], refit=True, test_accessed=False, resources=resources,
        choice_source='pre-registered equal vote of two previous full refits', members=[c['model_sha256'] for c in configs]))
    exports = read(prior_vote / 'exports_frozen.json')
    new = verified_export(folder, inputs, env, out)
    frozen_save(out / 'exports_frozen.json', dict(new_vote=new, original_vote=exports['soft_vote'],
        e5=exports['e5_full'], familiarity=read(matched / 'exports_frozen.json')['familiarity']))
    seals = read(out / 'exports_frozen.json')
    y = np.asarray(read(inputs / 'scoring_only.json')['labels'])
    groups = np.asarray(read(inputs / 'groups_only.json'))
    dirs = dict(new_vote=out / 'e5_familiarity_export_0', original_vote=prior_vote / 'soft_vote_export_0',
                e5=prior_vote / 'e5_full_export_0', familiarity=matched / 'full_export_0')
    hard = {n: classes(f / 'rows.csv', seals[n], len(y)) for n, f in dirs.items()}
    check(len(y) == len(groups) == 1693 and int((hard['original_vote'] == y).sum()) == 1234, 'Wrong reserve')
    return summarize(hard, y, groups)


def run(inputs, matched, encoder_batch, prior_vote, out, env):
    out.mkdir(parents=True, exist_ok=True)
    protocols = read(matched / 'protocol.json'), read(encoder_batch / 'protocol.json')
    expected = protocols[1]['identities']['packets']
    packet = inputs / 'anonymous.zip'
    check(ab_run.sha(packet) == expected['anonymous.zip'] == protocols[0]['training'] and protocols[0]['seed'] == 29111,
          'Wrong paired input')
    frozen_save(out / 'protocol.json', dict(plan=ab_run.sha(ROOT / 'docs/batches/B_e5_familiarity_1123_zh.md'),
        code=ab_run.sha(Path(__file__)), production=ab_run.sources(), prior=ab_run.sha(matched / 'results.json'),
        full_e5=read(prior_vote / 'models_frozen.json')['e5_full'], actual_new_fit_cap=2, model_api_cap=0))
    task = load_development_task(packet)
    groups = sample_groups(task, 'shared_text')
    opening = set()
    records, pooled = {}, {}
    for split in ('primary', 'confirmation_1'):
        payload = matched / 'partitions' / (split + '.joblib')
        identity = ab_run.sha(payload)
        split_info = read(matched / 'partitions' / (split + '_split.json'))
        held = np.asarray(split_info['search_indices'])
        g = groups[held]
        data = joblib.load(payload)
        y = data['y_search']
        check(split_info['evaluation_payload_sha256'] == identity and split_info['row_ids_sha256'] == fingerprint(held)
              and np.array_equal(task.y_train[held], y) and not set(groups[split_info['train_indices']]).intersection(g),
              'Wrong data partition')
        artifacts = {'e5': e5_fit(payload, out / split)}
        for arm in ('original', 'familiarity'):
            folder = matched / (split + '_' + arm)
            file = folder / 'worker.joblib'
            check(ab_run.sha(file) == read(folder / 'receipt.json')['model'], 'The original paired checkpoint changed')
            artifacts[arm] = joblib.load(file)
        prob, hard = {}, {}
        for n, a in artifacts.items():
            rec = a['prediction_record']
            check(rec['label_sha256'] == fingerprint(y) and rec['evaluation_payload_sha256'] == identity
                  and rec['prediction_sha256'] == fingerprint(a['search_predictions']), 'Prediction evidence differs')
            hard[n] = a['search_predictions']
            prob[n] = canonical_probabilities(a['model'], data['X_search'])[1][:, 1]
        params = artifacts['e5']['metrics']['resolved_configuration']['params']
        check(params == read(encoder_batch / 'deep-e5/run_manifest.json')['capabilities'][E5]['params'],
              'E5 differs from historical method')
        tie = np.asarray([left < right for left, right in data['X_search']])
        for n, member in (('original_vote', 'original'), ('new_vote', 'familiarity')):
            avg = (prob['e5'] + prob[member]) / 2
            hard[n] = ((avg > .5) | ((avg == .5) & tie)).astype(int)
        unique = np.asarray([group not in opening for group in g])
        check(unique.any(), 'Missing group-disjoint confirmation')
        for name, mask in [('all', np.ones(len(y), dtype=bool)), ('new_groups', unique)]:
            records[split + '_' + name] = summarize({n: v[mask] for n, v in hard.items()}, y[mask], g[mask])
        if split == 'primary':
            opening.update(g)
        for n, v in dict(**hard, labels=y, groups=g).items():
            pooled.setdefault(n, []).extend(v[unique])
    y, g = np.asarray(pooled.pop('labels')), np.asarray(pooled.pop('groups'))
    both = summarize({n: np.asarray(v) for n, v in pooled.items()}, y, g)
    correct = both['correct']
    passed = all(r['gained'] > r['lost'] for r in records.values()) and both['against_original_vote']['net_pp'] >= .2 and (
        correct['new_vote'] > max(correct['familiarity'], correct['e5']))
    frozen_save(out / 'inner_results.json', dict(folds=records, group_disjoint_pool=both, passed=passed))
    reserve = None
    if passed:
        check(all(ab_run.sha(inputs / n) == expected[n] for n in ('features.zip','scoring_only.json','groups_only.json')),
              'Development reserve identity changed')
        reserve = full_comparison(matched, prior_vote, inputs, out, env)
    physical = sum(len(f.read_text().splitlines()) for f in out.glob('*/physical.jsonl'))
    check(physical == 2, 'Mismatch in stage fit total')
    result = dict(folds=records, group_disjoint_pool=both, passed=passed, development_reserve=reserve,
        physical_new_fits=physical, paid_model_calls=0, paid_model_rmb=0, official_validation=False, formal_test=False,
        decision_source='registered model comparison')
    frozen_save(out / 'results.json', result)
    frozen_save(out / 'completion.json', dict(status='complete', results=ab_run.sha(out / 'results.json'),
        physical_fits=physical, model_api_requests=0))
    print('group-disjoint', both, 'reserve', reserve, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('workspace', 'inputs', 'matched', 'encoder_batch', 'prior_vote', 'output'):
        p.add_argument('--' + k.replace('_', '-'), dest=k, type=Path, required=True)
    a = p.parse_args()
    environment = offline_environment(a.workspace.resolve())
    with patch.dict(os.environ, environment):
        run(*(getattr(a, k).resolve() for k in ('inputs','matched','encoder_batch','prior_vote','output')), environment)
