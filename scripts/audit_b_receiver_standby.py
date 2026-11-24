"""Use an explicitly packaged, previously trained backup when the frozen winner cannot run."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np

from scripts import run_b1_ab_acceptance as record
from scripts.replay_b1_observed_actions import trusted_input
from scripts.run_b_c1_matrix import read_export
from scripts.verify_b_a31_bundle import cases, call, file_hash
from superagent.ab_scheduler import selection_evidence
from superagent.data import load_development_task, load_prediction_features, load_training_task
from superagent.metrics import MetricSpec, score_of
from tools.ab_registered_rows import content_digest


PORTABLE = {'logistic','extra_trees','rbf_svm'}


def requirements(cap):
    resource = cap['resource']
    estimator = cap['estimator']
    if resource not in ('cpu','local_frozen_encoder_cpu') or estimator not in PORTABLE|{'catboost'}:
        raise ValueError('Unknown deployment requirements')
    return [name for name,needed in (('minilm',resource=='local_frozen_encoder_cpu'),('catboost',estimator=='catboost')) if needed]


def register(name, case, entry, source, primary, dest):
    run = case/'run'
    caps,trusted = trusted_input(case,entry,source)
    history = record.read(run/'history.json')
    state = record.read(run/f'state_{len(history)+1}.json')
    selected = record.read(case/'relocated/final_selection.json')
    main = selected['family']
    ledger = record.read(run/'classifier_fits.json')
    if (state['history']!=history or selection_evidence(state,trusted_partitions=trusted)['incumbent']!=main
            or file_hash(case/'relocated'/selected['model_file'])!=primary or ledger['unknown_failed_attempts']
            or not 1<ledger['attempted_classifier_fits']==ledger['completed_classifier_fits']<=11):
        raise ValueError('Original final model, primary evidence or path accounting differs')
    needs = requirements(caps[main])
    standby = None
    if needs:
        verified = {r['trial'] for r in state['prediction_evidence']}
        choices = sorted((r for r in history if r['trial'] in verified and r['status']=='completed'
                          and r['split_id']=='primary' and not requirements(caps[r['candidate']])),
                         key=lambda r:(-score_of(r),r['trial']))
        if choices:
            row = choices[0]
            source_file = run/f"trial_{row['trial']}.joblib"
            checkpoint = joblib.load(source_file)
            if checkpoint['metrics']['resolved_configuration']['id']!=row['candidate']:
                raise ValueError('Registered trial has a different model configuration')
            dest.mkdir()
            shutil.copy2(case/'relocated/task_spec.json',dest/'task_spec.json')
            exported = dest/'standby.joblib'
            joblib.dump(dict(kind=entry['contract']['kind'],candidate=row['candidate'],model=checkpoint['model']),exported)
            standby = dict(family=row['candidate'],selected_experiment='trial_'+str(row['trial']),trial=row['trial'],
                source_primary_score=score_of(row),source_artifact_sha256=file_hash(source_file),refit=False,
                model_file=exported.name,model_sha256=file_hash(exported),model_format='candidate_bundle_v1',
                task_spec_sha256=file_hash(dest/'task_spec.json'),delivery_role='explicit_environment_standby',
                required_optional_components=[],extra_model_bytes=exported.stat().st_size)
            record.write(dest/'final_selection.json',standby)
    return dict(case=name,primary=main,primary_sha256=primary,required_optional_components=needs,
        original_fit_attempts=ledger['attempted_classifier_fits'],standby=standby)


def registered_entry(repo, collection, name):
    folder = repo/'research'/collection
    proto = record.read(folder/'protocol.json')
    pre = record.read(folder/'preflight.json')
    if name=='coupons':
        return pre['entries']['45102'],proto['sources']['representative']
    if name=='sms_kernel':
        return pre['sms'],proto['sources']['extended']
    entry = pre['sms']
    return dict(files={'train.zip':entry['train_sha256']},catalog_sha256=entry['expanded_sha256'],splits=entry['splits'],
                contract=entry['contract']),proto['source']


def grade(name, case, features, entry, staged, work, dest, model_choice, receipt):
    task = load_development_task(case/'train.zip')
    actual = load_prediction_features(features,task.kind,task.spec)
    if name=='coupons':
        expected = record.read(record.ROOT/'research/ba19_released_delivery/preflight.json')['coupon']['source_staged_sha256']
        if file_hash(staged)!=expected:
            raise ValueError('Wrong staged training-only coupon source')
        packet = features.parent
        row_ids = gzip.decompress((packet/'outer_ids.json.gz').read_bytes())
        if record.sha(row_ids)!=entry['outer_ids_sha256']:
            raise ValueError('Coupon development row order changed')
        truth = load_training_task(staged).y_train[json.loads(row_ids)['held']]
        if content_digest(actual,truth)!=entry['held_content_sha256']:
            raise ValueError('Coupon development contents changed')
    else:
        prior = work/'ab_refit_time/runs/prospective_clock_01'
        data = prior/'sms/holdout.joblib'
        if (file_hash(prior/'preflight.json')!='630a11762c8154429d5b82d67ae97f1b7fd53736c66c29bfd600e5129f754378'
                or file_hash(data)!=record.read(prior/'preflight.json')['sms']['files']['holdout.joblib']):
            raise ValueError('SMS development rows changed')
        original = joblib.load(data)
        truth = original['y']
        if content_digest(actual,truth)!=content_digest(original['X'],truth):
            raise ValueError('SMS feature or row order changed')
    reference = case/'export_0/rows.csv'
    if file_hash(reference)!=receipt['csv_sha256']:
        raise ValueError('Original output changed')
    old,pred = [np.asarray(read_export(path,len(truth)),dtype=truth.dtype) for path in (reference,dest)]
    metric = MetricSpec.for_task(task)
    old_score,new_score = [metric.evaluate(truth,p) for p in (old,pred)]
    return dict(role=model_choice,rows=len(truth),primary_metric=metric.name,original_score=old_score['score'],
        deployed_score=new_score['score'],original_correct=int((old==truth).sum()),deployed_correct=int((pred==truth).sum()),
        gained=int(((old!=truth)&(pred==truth)).sum()),lost=int(((old==truth)&(pred!=truth)).sum()),
        delta_pp=100*(new_score['score']-old_score['score']),identical=file_hash(reference)==file_hash(dest))


def study(args):
    out, work = args.output.resolve(),args.workspace.resolve()
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    receipt = record.read(args.verification)
    prior = record.ROOT/'research/b_receiver_runtime'
    if file_hash(args.verification)!=record.read(prior/'protocol.json')['verification_sha256']:
        raise ValueError('Wrong receiving bundle evidence')
    observations = {(r['case'],r['environment']):r for r in record.read(prior/'results.json')}
    needed_cases = {name for name,_ in observations}
    clone = out/'repo'
    call(['git','clone','--quiet','--no-checkout','--no-local','--no-tags',args.receiver.resolve(),clone],out)
    empty = out/'empty_weights'
    empty.mkdir()
    full = work/'pretrained_weights/all-MiniLM-L6-v2'
    environments = [('pinned_with_weights',sys.executable,full),('pinned_missing_weights',sys.executable,empty),
                    ('light_with_weights',str(args.light_python.absolute()),full)]
    record.write(out/'protocol.json',dict(batch='B-RECEIVER-ENVIRONMENT-02',driver_sha256=file_hash(Path(__file__)),
        plan_sha256=file_hash(record.ROOT/'docs/batches/B_RECEIVER_ENVIRONMENT_02_plan.md'),
        prior_environment_results_sha256=file_hash(prior/'results.json'),bundle_sha256=receipt['bundle_sha256'],
        portable_estimators=sorted(PORTABLE),fit_limit=0,api_limit=0))
    try:
        selections,setup,processes = {},{},{}
        for branch,name,case,features,rows,collection,_,_ in cases(work):
            if name not in needed_cases:
                continue
            call(['git','switch','-q','--detach',receipt['branch_tips'][branch]],clone)
            entry,source = registered_entry(clone,collection,name)
            expected = receipt['predictions'][name]
            if file_hash(features)!=expected['features_sha256']:
                raise ValueError('Feature-only file changed')
            selections[name] = register(name,case,entry,source,expected['model_sha256'],out/('standby_'+name))
            setup[name] = (branch,case,features,rows,entry)
        record.write(out/'all_selections_frozen.json',selections)
        for name,(branch,case,features,rows,entry) in setup.items():
            choice = selections[name]
            call(['git','switch','-q','--detach',receipt['branch_tips'][branch]],clone)
            for env_name,interpreter,weights in environments:
                route = [clone/'scripts/ab_acceptance_guard',clone]
                if env_name=='light_with_weights':
                    route.append(work/'ab_portable_inputs/runs/py314_http_import_dependencies')
                env = dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,route)),SUPERAGENT_OFFLINE='1',
                    SUPERAGENT_MINILM_WEIGHTS=str(weights),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                env.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
                probe = "import json,importlib.util; from superagent.ab_sentence import verified_directory; print(json.dumps(dict(minilm=verified_directory() is not None,catboost=importlib.util.find_spec('catboost') is not None)))"
                resources = json.loads(call([interpreter,'-c',probe],clone,env))
                missing = [part for part in choice['required_optional_components'] if not resources[part]]
                role = 'standby' if missing else 'primary'
                previous = observations[name,env_name]
                if bool(missing)!=bool(previous['returncode']):
                    raise ValueError('Probed availability differs from prior real native prediction attempts')
                if missing and choice['standby'] is None:
                    processes[name+'_'+env_name] = dict(case=name,environment=env_name,status='no_executable_candidate',missing=missing)
                    continue
                model = out/('standby_'+name) if missing else case/'relocated'
                source_selection = record.read(model/'final_selection.json')
                identity = choice['standby']['model_sha256'] if missing else choice['primary_sha256']
                if file_hash(model/source_selection['model_file'])!=identity:
                    raise ValueError('Model changed after selection freeze')
                dest = out/(name+'_'+env_name)
                dest.mkdir()
                command = [interpreter,'-m','superagent','predict',str(features),'--run-dir',str(model),'--output',str(dest/'rows.csv')]
                result = subprocess.run(command,cwd=clone,env=env,capture_output=True,text=True,timeout=240)
                (dest/'process.log').write_text(result.stdout+result.stderr)
                processes[name+'_'+env_name] = dict(case=name,environment=env_name,role=role,missing=missing,
                    status='delivered' if result.returncode==0 else 'failed',model_sha256=identity,returncode=result.returncode,
                    csv_sha256=file_hash(dest/'rows.csv') if (dest/'rows.csv').is_file() else None)
                if result.returncode==0:
                    read_export(dest/'rows.csv',rows)
        record.write(out/'all_prediction_attempts_frozen.json',processes)
        scores = {}
        for key,r in processes.items():
            if r['status']=='delivered':
                name = r['case']
                _,case,features,_,entry = setup[name]
                scores[key] = grade(name,case,features,entry,args.coupon_source,work,out/key/'rows.csv',r['role'],receipt['predictions'][name])
                if r['role']=='primary' and not scores[key]['identical']:
                    raise ValueError('Healthy primary delivery changed')
        if len(processes)!=9 or record.trace(out):
            raise ValueError('Unexpected matrix size or supervised fit')
        record.write(out/'results.json',dict(processes=processes,scores=scores))
        record.write(out/'completion.json',dict(status='completed',attempts=len(processes),delivered=len(scores),
            explicit_standby_deliveries=sum(s['role']=='standby' for s in scores.values()),new_fits=0,new_model_api_requests=0,
            new_course_rmb=0,official_validation_test_or_H=False))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',reason=str(error)))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','verification','receiver','light-python','coupon-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    study(parser.parse_args())
