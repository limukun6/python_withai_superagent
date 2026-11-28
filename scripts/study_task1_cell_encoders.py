"""Resume an audited solver after its original full delivery, or grade four sealed training panels."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import joblib
import numpy as np

from scripts.run_planner_self_review import source_identity
from scripts.study_task1_outer import compare, prediction_csv
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from superagent.ab_run import Executor
from superagent.autonomous_run import planner_observation
from superagent.cell_encoder_review import after_delivery
from superagent.cell_encoders import LIBRARY
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.metrics import MetricSpec
from superagent.table_foundation import SHA256, WEIGHT_ENV

CASES = dict(novel0='matched_catalogue_1128/unseen_profile_stress_1128/0',
             novel3='matched_catalogue_1128/unseen_profile_stress_1128/3', ordinary0='seed_policy/0', ordinary2='seed_policy/2')
PLAN = ROOT/'research/task1_feedback_extension_1127/cell_encoder_tournament.json'
FOLDS = ('primary', 'confirmation_1', 'confirmation_2')


def control(panel):
    receipt, launch = read(panel/'replicated/sealed.json'), read(panel/'replicated/launch.json')
    require(receipt['status']=='completed' and receipt['independent_reload'] and not receipt['model_api_requests']
            and sha(panel/'replicated/solve.csv')==receipt['prediction']==sha(panel/'replicated/reload.csv')
            and receipt['launch']==sha(panel/'replicated/launch.json') and launch['inputs']==sha(panel/'inputs.json')
            and launch['cap']==64 and launch['method_hint'] is None and launch['feedback_mode']=='seed-replication'
            and launch['optional_resources']['tabicl_v2']==SHA256, 'Original ordinary run not verified')
    return receipt, launch


def resume(args):
    prior = args.root/CASES[args.case]
    dest = args.root/'cell_portfolio_1128'/args.case
    original, launch = control(prior)
    require(sha(dest/'inputs.json') == sha(prior/'inputs.json') and sha(args.weight)==SHA256, 'Unmatched resource or roles')
    folder = dest/'portfolio'
    folder.mkdir(exist_ok=False)
    native = folder/'run/agent'
    native.mkdir(parents=True)
    archive, query = dest/'anonymous/training.zip', dest/'anonymous/queries.zip'
    data = read(dest/'inputs.json')
    require(sha(archive)==data['entry_inputs'] and sha(query)==data['entry_queries'], 'Changed raw packets')
    settings = env(folder/'unused', folder/'physical.jsonl')
    for key in list(settings):
        if key.startswith('SUPERAGENT_'):
            del settings[key]
    settings.update({LIBRARY:str(args.library), WEIGHT_ENV:str(args.weight), 'SUPERAGENT_OFFLINE':'1',
                     'PYTHONPATH':os.pathsep.join(map(str, [ROOT/'scripts/unified_acceptance', ROOT, args.runtime])),
                     'UNIFIED_RAW_INPUT':str(archive), 'UNIFIED_RAW_LOG':str(folder/'opened.jsonl'),
                     'UNIFIED_RAW_MEMBERS':json.dumps(['C_train.npy','y_train.npy','Readme.md'])})
    acquisition = dict(source=source_identity(ROOT), original=sha(prior/'replicated/sealed.json'), protocol=sha(PLAN),
                       cap=64, original_fits=original['fits'], original_seconds=original['seconds'],
                       inputs=sha(dest/'inputs.json'), method_hint=None)
    save(folder/'launch.json', acquisition)
    subprocess.run([sys.executable,'-m','scripts.study_task1_cell_encoders','boundary','--case',args.case,
                    '--root',str(args.root)], env=settings, cwd=ROOT, check=True)
    for name in ('solve','reload'):
        with (folder/(name+'.log')).open('x') as stream:
            subprocess.run([sys.executable,'-m','unified_agent','predict',str(query),'--run-dir',str(native.parent),
                            '--output',str(folder/(name+'.csv'))], env=settings, cwd=ROOT, check=True, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=180)
        prediction = read(folder/'prediction_manifest.json')
        selection = read(native/'final_selection.json')
        require(prediction['selected_experiment']==selection['selected_experiment'] and not prediction['test_labels_read']
                and prediction['prediction_sha256']==sha(folder/(name+'.csv'))
                and selection['model_sha256']==sha(native/selection['model_file']), 'Different model exported')
    final = read(native/'run_outcome.json')
    counts = read(native/'classifier_fits.json')
    additional = len((folder/'physical.jsonl').read_text().splitlines()) if (folder/'physical.jsonl').exists() else 0
    require(original['fits']+additional==counts['attempted_classifier_fits']<=64 and not counts['unknown_failed_attempts']
            and sha(folder/'solve.csv')==sha(folder/'reload.csv'), 'Physical count or model reload mismatch')
    require(len(prediction_csv(folder/'solve.csv',original['rows']))==original['rows'], 'Incomplete result')
    save(folder/'sealed.json', dict(launch=sha(folder/'launch.json'), members=final['members'],
         prediction=sha(folder/'solve.csv'), rows=original['rows'], fits=original['fits']+additional, new_fits=additional,
         seconds=final['wall_seconds'], status='completed', independent_reload=True, model_api_requests=0,
         opened=sorted({json.loads(line) for line in (folder/'opened.jsonl').read_text().splitlines()}),
         model=read(native/'final_selection.json')['model_sha256'], score=None))
    print(args.case, 'sealed', additional, 'new fits', flush=True)


def boundary(args):
    prior = args.root/CASES[args.case]
    frozen = prior/'replicated/run/agent'
    orig, launch = control(prior)
    dest = args.root/'cell_portfolio_1128'/args.case
    native = dest/'portfolio/run/agent'
    start = time.monotonic()-orig['seconds']
    task = load_training_task(dest/'anonymous/training.zip', read_declarations=True)
    spec, choice, selected = read(frozen/'task_contract.json'), read(frozen/'selection.json')['choice'], read(frozen/'final_selection.json')
    require(selected['model_sha256']==orig['model']==sha(frozen/selected['model_file'])
            and sha(frozen/'task_spec.json')==selected['task_spec_sha256'], 'Frozen selection changed')
    metric = MetricSpec.from_payload(dict(metric_spec=spec['metric']))
    for name in ['candidate_registry.json','task_spec.json','task_contract.json','selection.json','final_selection.json','run_manifest.json',
                 'history.json','seed_replication.json','classifier_fits.json',*(f+'_split.json' for f in FOLDS)]:
        shutil.copyfile(frozen/name,native/name)
    files = {selected['model_file'],'refit_data.joblib',*(f+'.joblib' for f in FOLDS),*(p.name for p in frozen.glob('seed_pilot*.joblib'))}
    for name in files:
        os.link(frozen/name,native/name)
    observed, payloads, caps = {}, {}, {}
    members = choice['members']
    history = read(frozen/'history.json')
    accounts = read(frozen/'classifier_fits.json')
    physical = (prior/'replicated/physical.jsonl').read_text().splitlines()
    require(orig['fits']==accounts['attempted_classifier_fits']==len(physical) and not accounts['unknown_failed_attempts'],
            'Original fit ledger changed')
    for fold in FOLDS:
        path = native/(fold+'.joblib')
        split = read(native/(fold+'_split.json'))
        bundle = joblib.load(path)
        digest = sha(path)
        require(digest==split['payload_sha256'], 'Changed confirmation rows')
        np.testing.assert_array_equal(task.y_train[split['held']],bundle['y_search'])
        np.testing.assert_array_equal(task.X_train[split['held']],bundle['X_search'])
        payloads[fold] = dict(path=path, labels=bundle['y_search'], sha256=digest)
        observed[fold] = {}
        for row in history:
            name = row['candidate']
            if row['split_id']!=fold or name not in members or row['status']!='completed':
                continue
            trial = joblib.load(frozen/f"trial_{row['trial']}.joblib")
            facts = trial['metrics']
            observed[fold][name] = planner_observation(trial,facts,payloads[fold],metric)
            caps[name] = dict(params=facts['resolved_configuration']['params'], estimator=facts['estimator']['family'],
                              origin='shared_baseline', resource='cpu', classifier_fits=facts['classifier_fits'],
                              refit_classifier_fits=facts['classifier_fits'])
    require(all(set(rows)==set(members) for rows in observed.values()), 'Missing full prior evidence')
    log = RunLogger(native)
    executor = Executor(native,caps,metric,start,launch['seconds'],300,launch['cap'],log,restrictions=spec.get('method_restrictions',()))
    executor.records = list(accounts['workers'])
    elected, choice = after_delivery(task,spec,metric,selected,choice,observed,payloads,executor,native,log,300)
    log.write_json('run_outcome.json',dict(read(frozen/'run_outcome.json'),members=choice['members'],winner=elected['family'],
                                          **executor.accounting(),wall_seconds=time.monotonic()-start))
    verdict = read(native/'cell_encoder_review.json')
    print(args.case, verdict['action'], verdict.get('reason',''), flush=True)


def grade(root):
    study = root/'cell_portfolio_1128'
    locks = {}
    for key, where in CASES.items():
        panel, folder = root/where, study/key/'portfolio'
        base, _ = control(panel)
        receipt, report = read(folder/'sealed.json'), read(folder/'run/agent/cell_encoder_review.json')
        require(sha(folder/'solve.csv')==receipt['prediction']==sha(folder/'reload.csv') and receipt['independent_reload']
                and receipt['model']==report['exported_model'] and receipt['fits']<=64 and not receipt['model_api_requests']
                and sha(folder/'launch.json')==receipt['launch'] and sha(study/key/'inputs.json')==sha(panel/'inputs.json')
                and set(receipt['opened']) <= {'C_train.npy','y_train.npy','Readme.md'}
                and report['action'] in ('retain','adopt') and report.get('nominee') and len(report['screened'])>=2
                and not report.get('reason','').startswith('Optional extension could not complete')
                and not any('error' in record for record in report['screened'].values()), 'Incomplete treatment run '+key)
        locks[key] = dict(control=base['prediction'],receipt=sha(folder/'sealed.json'),prediction=receipt['prediction'],
                          inputs=sha(panel/'inputs.json'), decision=sha(folder/'run/agent/cell_encoder_review.json'))
    save(study/'outer_lock.json',dict(protocol=sha(PLAN),panels=locks))
    result = {}
    by_regime = {}
    for key, where in CASES.items():
        panel, folder = root/where, study/key/'portfolio'
        info = read(panel/'inputs.json')
        answers = panel/'outer_answers.joblib'
        require(sha(answers) == (info.get('outer_answers') or info['external_answers']), 'Changed grading labels')
        packet = joblib.load(answers)
        y = np.asarray(packet['y']).astype(str)
        old, new = [prediction_csv(p,len(y)) for p in (panel/'replicated/solve.csv',folder/'solve.csv')]
        groups = np.asarray(next(packet[k] for k in ('profiles','groups','profile') if k in packet))
        require(len(groups)==len(y), 'Missing outer groups')
        old, new = old==y, new==y
        receipt, report = read(folder/'sealed.json'), read(folder/'run/agent/cell_encoder_review.json')
        run_totals = read(folder/'run/agent/classifier_fits.json')
        reference_receipt, _ = control(panel)
        result[key]=dict(rows=len(y),control_correct=int(old.sum()),treatment_correct=int(new.sum()),**compare(new,old,groups),
            action=report['action'],reason=report.get('reason'),candidate_names=report['available'],
            primary=report['screened'],original_primary=report['primary_incumbent'],proposal=report['nominee'],
            confirmations=report['confirmations'],fits=receipt['fits'],new_fits=receipt.get('new_fits',receipt['fits']),
            failed_fits=run_totals['attempted_classifier_fits']-run_totals['completed_classifier_fits'],
            model_api_requests=receipt['model_api_requests'],same_prediction_hash=receipt['prediction']==reference_receipt['prediction'])
        regime = 'novel' if key.startswith('novel') else 'ordinary'
        by_regime.setdefault(regime,[]).append((old,new,groups))
    regimes = {}
    for regime,parts in by_regime.items():
        old,new,groups = map(np.concatenate,zip(*parts))
        regimes[regime]=dict(rows=len(old),control_correct=int(old.sum()),treatment_correct=int(new.sum()),
                             **compare(new,old,groups))
    eligible = all(reg['gain']>=.003 for reg in regimes.values()) and all(row['net']>=0 for row in result.values())
    save(study/'results.json',dict(lock=sha(study/'outer_lock.json'),protocol=sha(PLAN),panels=result,regimes=regimes,
        advances=eligible,course_validation_opened=False,official_test_opened=False))
    print(json.dumps(regimes,indent=2), 'advance',eligible,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['resume','boundary','grade'])
    p.add_argument('--root',type=lambda v:Path(v).resolve(),required=True)
    p.add_argument('--case',choices=tuple(CASES))
    for name in ('weight','library','runtime'):
        p.add_argument('--'+name,type=lambda v:Path(v).resolve())
    args=p.parse_args()
    if args.action=='resume':
        resume(args)
    elif args.action=='boundary':
        boundary(args)
    else:
        grade(args.root)
